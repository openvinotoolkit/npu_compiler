//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <numeric>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLPERMUTETONNDMA
#define GEN_PASS_DEF_UNROLLPERMUTETONNDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    const auto elemType = originType.getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        return originType.changeShapeElemType(shape, newQType);
    }

    return originType.changeShape(shape);
}

vpux::NDTypeInterface getPerClusterInputType(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType,
                                             mlir::AffineMap memPerm, ShapeRef outShape, ShapeRef offset) {
    auto inputShape = inputType.getShape();

    // Back infer the input shape from output shape and mem_Perm attribution
    // For example: Input: 1x8x1x32xfp16, #NHWC -> 1x32x1x8xfp16, #NHWC, memPerm: [0, 1, 3, 2]
    // If want get right input shape from per cluster output shape. There are three step:
    //   1) Get the output real physical shape: 1x32x1x8xfp16, #NHWC -> 1x1x8x32
    //   2) Using memPerm to back infer the input real physical shape: 1x1x8x32 -> 1x1x32x8
    //   3) Got the input logic shape: 1x1x32x8 -> 1x8x1x32xfp16, #NHWC
    const auto inOrder = inputType.getDimsOrder();
    const auto outOrder = outputType.getDimsOrder();
    auto backInferInputShape = [&](ShapeRef subOutShape) -> Shape {
        // After Expand fuse into Permute and got one PermuteDMA Op
        // The channel size of input and output are not same
        // For example: input (NCHW) 1x3x32x32, output(NHWC) 1x16x32x32
        // The channel size need align with the input
        auto inLogicShape = to_small_vector(subOutShape);
        if (inputType.getShape().totalSize() != outputType.getShape().totalSize()) {
            VPUX_THROW_UNLESS(subOutShape[Dims4D::Act::C] != inputShape[Dims4D::Act::C],
                              "Got unexpected input {0} output {1} type of PermuteDMA", inputType, outputType);
            inLogicShape[Dims4D::Act::C.ind()] = inputShape[Dims4D::Act::C];
        }

        Shape outPhysicalShape(inLogicShape.size());
        for (const auto idx : irange(inLogicShape.size())) {
            outPhysicalShape[Dim(idx)] = inLogicShape[outOrder.dimAt(idx).ind()];
        }

        Shape inPhysicalShape(inLogicShape.size());
        for (const auto idx : irange(inLogicShape.size())) {
            inPhysicalShape[DimsOrder::fromAffineMap(memPerm).dimAt(idx)] = outPhysicalShape[Dim(idx)];
        }

        for (const auto idx : irange(inLogicShape.size())) {
            inLogicShape[inOrder.dimAt(idx).ind()] = inPhysicalShape[Dim(idx)];
        }

        return Shape(inLogicShape);
    };

    return changeShape(inputType, backInferInputShape(outShape), offset);
}

//
// PermuteRewriter
//

class PermuteRewriter final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    PermuteRewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("PermuteRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unrollSegmentedOrOverlappedOutput(VPUIP::PermuteDMAOp permuteOp,
                                                          VPUIP::DistributedBufferType distributedType,
                                                          mlir::AffineMap memPerm,
                                                          mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicatedOutput(VPUIP::PermuteDMAOp permuteOp,
                                               VPUIP::DistributedBufferType distributedType, mlir::AffineMap memPerm,
                                               mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicatedInputAndOutput(VPUIP::PermuteDMAOp permuteOp, mlir::AffineMap memPerm,
                                                       mlir::PatternRewriter& rewriter) const;
    mlir::LogicalResult unrollDuplicatedInput(VPUIP::PermuteDMAOp permuteOp, mlir::AffineMap memPerm,
                                              mlir::PatternRewriter& rewriter) const;
    mlir::LogicalResult rewritePermuteDMA(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter) const;
    int64_t _dmaPortCount;
    Logger _log;
};

mlir::LogicalResult PermuteRewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
    // Skip PermuteDMA ops which have been unrolled by checking mem_perm attribute
    if (permuteOp.getMemPermAttr() == nullptr) {
        return mlir::failure();
    }

    const auto input = permuteOp.getInput();
    const auto output = permuteOp.getOutputBuff();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    auto inDistributedType = inputType.dyn_cast<VPUIP::DistributedBufferType>();
    auto outDistributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();

    // Unroll by distributed type of input/output
    if (inDistributedType != nullptr || outDistributedType != nullptr) {
        _log.trace("process permute with DistributedType at {0}", permuteOp);

        VPUX_THROW_UNLESS(permuteOp.getMemPerm().has_value(),
                          "Can not get memPerm attribute from PermuteDMA layer at {0}", permuteOp.getLoc());
        const auto memPerm = permuteOp.getMemPerm().value();

        if (inDistributedType != nullptr && outDistributedType != nullptr) {
            return unrollDuplicatedInputAndOutput(permuteOp, memPerm, rewriter);
        } else if (inDistributedType != nullptr) {
            return unrollDuplicatedInput(permuteOp, memPerm, rewriter);
        }

        VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                                  outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Unexpected memory space. Got: input {0}, output {1}", inputType.getMemoryKind(),
                          outputType.getMemoryKind());

        VPUX_THROW_WHEN(outDistributedType == nullptr, "Expect distributed type for permute op output, actual: {0}",
                        outputType);

        VPUX_THROW_UNLESS(VPUIP::doesPermuteDMATileDimSupportWrapInCluster(inputType, outputType, memPerm,
                                                                           outDistributedType, _log),
                          "Unsupported PermuteDMA under cluster tiling at '{0}'", permuteOp->getLoc());

        const auto distributionAttr = outDistributedType.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return unrollSegmentedOrOverlappedOutput(permuteOp, outDistributedType, memPerm, rewriter);
        } else if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
            return unrollDuplicatedOutput(permuteOp, outDistributedType, memPerm, rewriter);
        } else {
            VPUX_THROW("Unsupported distributed mode");
        }
    }

    _log.trace("Permute rewriter operation '{0}' at '{1}'", permuteOp->getName(), permuteOp->getLoc());

    // Rewrite the Permute operation itself
    return rewritePermuteDMA(permuteOp, rewriter);
}

/// @brief Rewrites PermuteDMAOp using its mem_perm attribute to update dma_descriptor attr value
mlir::LogicalResult PermuteRewriter::rewritePermuteDMA(VPUIP::PermuteDMAOp permuteOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto vpurtTask = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto srcDeclBuff = permuteOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(srcDeclBuff != nullptr, "Can't get buffer for operand: {0}", permuteOp.getInput());

    auto dstDeclBuff = permuteOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto inType = permuteOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = permuteOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto srcOffset = srcDeclBuff.getByteOffset();
    auto dstOffset = dstDeclBuff.getByteOffset();

    // For unrolled DMA which is inside of cluster tiling, the dma descriptor is already calculated
    auto dmaDescriptorAttr = permuteOp.getDmaDescriptorAttr();
    const auto memPerm = permuteOp.getMemPerm().value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inType, memPerm);
    auto numPlaneDim = VPUIP::getPermuteDMANumPlaneDim(inType, memPerm);

    auto portIsAlreadyAssigned = true;
    if (dmaDescriptorAttr == nullptr) {
        auto ctx = permuteOp->getContext();
        auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inType, outType, memPerm, _log).value();
        auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, _log).value();
        auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);
        portIsAlreadyAssigned = false;
    }

    auto subInput =
            VPUIP::getPermuteDMASubInputShapes(VPU::getArch(permuteOp), inType, outType, memPerm, _dmaPortCount, _log);
    VPUX_THROW_UNLESS(subInput.has_value(), "Cannot get unrolled subInputShapes for PermuteDMA op {0}", permuteOp);
    auto subInputShapes = subInput.value();
    auto subOutputShapes = VPUIP::getPermuteDMASubOutputShapes(subInputShapes, inType, outType, memPerm);

    _log.trace("Unrolling PermuteDMAOp '{0}' at '{1}'", permuteOp->getName(), permuteOp->getLoc());

    int64_t dmaPort = 0;
    SmallVector<VPUIP::PermuteDMAOp> firstPermuteDMAsOnPorts;
    SmallVector<VPUIP::PermuteDMAOp> lastPermuteDMAsOnPorts;
    SmallVector<VPUIP::PermuteDMAOp> newPermuteDMAs;
    for (auto idx = 0; idx < checked_cast<int64_t>(subInputShapes.size()); idx++) {
        auto newDMADescriptorAttr = VPUIP::updateNumPlanes(dmaDescriptorAttr, subInputShapes[idx][numPlaneDim]);

        const auto dimOrder = (subInputShapes[0].size() == 2) ? DimsOrder::NC : DimsOrder::CHW;
        auto newSrcStrides =
                (subInputShapes[idx].size() == 2)
                        ? SmallVector<vpux::Bit>{Bit(subInputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())}
                        : SmallVector<vpux::Bit>{Bit(subInputShapes[idx][Dim(1)] * subInputShapes[idx][Dim(2)] *
                                                     Bit(elemTypeSize).count()),
                                                 Bit(subInputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};

        auto newSrcMemRef = vpux::getMemRefType(subInputShapes[idx], srcType.getElementType(), dimOrder,
                                                srcType.getMemSpace(), Strides(newSrcStrides));

        auto newSrcBuff =
                srcType.getMemSpace().getIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                  newSrcMemRef, srcDeclBuff.getSection(),
                                                                  srcType.getMemSpace().getIndex().value(), srcOffset)
                        : srcDeclBuff.getSectionIndex().has_value()
                                  ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                            rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                            srcDeclBuff.getSection(),
                                            parseIntArrayAttr<int64_t>(srcDeclBuff.getSectionIndex().value()),
                                            srcOffset)
                                  : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                            newSrcMemRef, srcDeclBuff.getSection(),
                                                                            srcOffset);

        auto newDstStrides =
                (subOutputShapes[idx].size() == 2)
                        ? SmallVector<vpux::Bit>{Bit(subOutputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())}
                        : SmallVector<vpux::Bit>{Bit(subOutputShapes[idx][Dim(1)] * subOutputShapes[idx][Dim(2)] *
                                                     Bit(elemTypeSize).count()),
                                                 Bit(subOutputShapes[idx][Dim(2)] * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};
        mlir::Type newDstType;
        if (auto dstDistributedType = dstType.dyn_cast<VPUIP::DistributedBufferType>()) {
            auto ctx = permuteOp->getContext();
            auto distributionAttr = dstDistributedType.getDistribution();
            VPUX_THROW_WHEN(
                    distributionAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "Issues with unrolling PermuteNNDMA; Buffer has distributed type != DUPLICATED after unroll");
            if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
                distributionAttr = VPU::getNonOverlappedDistributedAttr(
                        subOutputShapes[idx], distributionAttr.getMode(), nullptr, distributionAttr.getNumClusters(),
                        nullptr, distributionAttr.getUniformDistributedSegments(), dstDeclBuff.getContext());
            }

            const auto layout = mlir::AffineMapAttr::get(dimOrder.toAffineMap(ctx));
            newDstType = VPUIP::DistributedBufferType::get(ctx, subOutputShapes[idx].raw(), dstType.getElementType(),
                                                           layout, dstType.getMemSpace(), distributionAttr);
        } else {
            newDstType = vpux::getMemRefType(subOutputShapes[idx], dstType.getElementType(), dimOrder,
                                             dstType.getMemSpace(), Strides(newDstStrides));
        }

        VPUX_THROW_UNLESS(dstType.getMemSpace().getIndex().has_value() || dstDeclBuff.getSectionIndex().has_value(),
                          "No section index find at '{}'", dstDeclBuff.getLoc());
        auto newDstBuff =
                dstType.getMemSpace().getIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                                                  dstDeclBuff.getSection(),
                                                                  dstType.getMemSpace().getIndex().value(), dstOffset)
                        : dstDeclBuff.getSectionIndex().has_value()
                                  ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                            rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                            dstDeclBuff.getSection(),
                                            parseIntArrayAttr<int64_t>(dstDeclBuff.getSectionIndex().value()),
                                            dstOffset)
                                  : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(),
                                                                            newDstType, dstDeclBuff.getSection(),
                                                                            dstOffset);

        _log.trace("Create unrolled PermuteDMA operation with input/output shape: {0}/{1}, SrcMemory at {2}, "
                   "DstMemory at {3}",
                   subInputShapes[idx], subOutputShapes[idx], newSrcBuff.getSection(), newDstBuff.getSection());

        const auto newLoc = appendLoc(vpurtTask->getLoc(), "_unrolled_permuteDMA");
        auto newDmaPort = portIsAlreadyAssigned ? permuteOp.getPort().value() : dmaPort;
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, newSrcBuff, newDstBuff,
                vpux::getIntAttr(rewriter, newDmaPort), permuteOp.getIsOutOfOrderAttr(), permuteOp.getIsCriticalAttr(),
                /*mem_perm*/ nullptr, newDMADescriptorAttr, permuteOp.getDmaHwpIdAttr(),
                permuteOp.getProfilingMetadataAttr());

        newPermuteDMAs.push_back(newPermuteDMAOp);

        // find the first and last DMAs on different ports
        if (firstPermuteDMAsOnPorts.size() < static_cast<size_t>(_dmaPortCount)) {
            firstPermuteDMAsOnPorts.push_back(newPermuteDMAOp);
            lastPermuteDMAsOnPorts.push_back(newPermuteDMAOp);
        } else {
            lastPermuteDMAsOnPorts[newDmaPort] = newPermuteDMAOp;
        }

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        auto numPlaneValue = newDMADescriptorAttr.getNumPlanes().getInt();
        auto srcPlaneStrideValue = newDMADescriptorAttr.getSrcPlaneStride().getInt();
        auto dstPlaneStrideValue = newDMADescriptorAttr.getDstPlaneStride().getInt();
        srcOffset += numPlaneValue * srcPlaneStrideValue;
        dstOffset += numPlaneValue * dstPlaneStrideValue;
    }

    for (auto& dmaOp : newPermuteDMAs) {
        auto vpurtTask = dmaOp->getParentOfType<VPURT::TaskOp>();

        // remove wait barrier dependency for these new permute DMA except first ones on each port
        if (std::find(firstPermuteDMAsOnPorts.begin(), firstPermuteDMAsOnPorts.end(), dmaOp) ==
            firstPermuteDMAsOnPorts.end()) {
            vpurtTask.getWaitBarriersMutable().clear();
        }

        // remove update barrier dependency for these new permute DMA except last ones on each port
        if (std::find(lastPermuteDMAsOnPorts.begin(), lastPermuteDMAsOnPorts.end(), dmaOp) ==
            lastPermuteDMAsOnPorts.end()) {
            vpurtTask.getUpdateBarriersMutable().clear();
        }
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult PermuteRewriter::unrollSegmentedOrOverlappedOutput(VPUIP::PermuteDMAOp permuteOp,
                                                                       VPUIP::DistributedBufferType distributedType,
                                                                       mlir::AffineMap memPerm,
                                                                       mlir::PatternRewriter& rewriter) const {
    auto loc = permuteOp->getLoc();
    auto ctx = permuteOp->getContext();

    const auto input = permuteOp.getInput();
    const auto output = permuteOp.getOutputBuff();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto originalOutputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = distributedType.getCompactType();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distributed mode");
    const auto perClusterOutShapes = distributedType.getPerClusterMemoryShapes();
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto vpurtTask = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteOp);

    const auto tileInputType = [&](vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
            newTypes[clusterId] = getPerClusterInputType(inputType, outputType, memPerm, perClusterOutShapes[clusterId],
                                                         perClusterShapeOffsets[clusterId]);
        }
        return newTypes;
    };

    const auto tileOutputType = [&](vpux::NDTypeInterface outputType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(outputType, perClusterOutShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }
        return newTypes;
    };

    auto inTypes = tileInputType(inputType, outputType);
    const auto originStride = inputType.getStrides();

    for (size_t clusterId = 0; clusterId < perClusterOutShapes.size(); ++clusterId) {
        inTypes[clusterId] = inTypes[clusterId].changeStrides(originStride);
    }

    const auto outTypes = tileOutputType(outputType);

    rewriter.setInsertionPointAfter(vpurtTask);
    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                mlir::Operation* insertionPoint, Byte offset) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(permuteOp->getLoc(), cst, perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterOutShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, permuteOp->getLoc(), newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(ctx, ArrayRef({clusterId})),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.getByteOffset()};
        ddrOffset += offset;

        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, ddrOffset);
    };

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptors and ddr offsets
    SmallVector<VPUIP::DMADescriptorAttr> subDmaDescriptors;
    SmallVector<Byte> ddrOffsets;
    SmallVector<Shape> subMergedOutputShapes;

    const auto mergedOutputDimList = VPUIP::getPermuteDMAOutputMergedDimList(outputType, mergedOutputShape);
    auto tileDimForMergedOutput =
            VPUIP::getTileDimForPermuteDMA(inputType, outputType, memPerm, distributedType, _log).value();

    const auto numTileSize = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tileDimIter = std::find_if(numTileSize.begin(), numTileSize.end(), [](const int64_t dim) {
        return dim > 1;
    });
    VPUX_THROW_UNLESS(tileDimIter != numTileSize.end(), "Can not find tile dim.");
    auto tileDim = Dim(std::distance(numTileSize.begin(), tileDimIter));

    auto getSrcOffset = [&](vpux::ShapeRef offset) -> vpux::Byte {
        auto outputShape = originalOutputType.getShape();

        const auto splitDimList = mergedOutputDimList[tileDimForMergedOutput.ind()];
        VPUX_THROW_UNLESS(std::any_of(splitDimList.begin(), splitDimList.end(),
                                      [&](vpux::Dim dim) {
                                          return dim == tileDim;
                                      }),
                          "tileDim is not exist in splitDimList.");

        const auto totalOffsetSize = mergedOutputShape[tileDimForMergedOutput];
        return Byte(totalOffsetSize / outputShape[tileDim] * offset[tileDim] * elemTypeSize.count());
    };

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto mergedSubOutputShape =
                VPUIP::getPermuteDMAOutputShape(inTypes[clusterId], outTypes[clusterId], memPerm, _log).value();
        ddrOffsets.push_back(getSrcOffset(perClusterShapeOffsets[clusterId]));
        subMergedOutputShapes.push_back(mergedSubOutputShape);
    }
    subDmaDescriptors = dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, subMergedOutputShapes,
                                                        tileDimForMergedOutput, elemTypeSize);

    int64_t dmaPort = 0;
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();
    SmallVector<VPUIP::PermuteDMAOp> newPermuteDMAs;
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto inputBuffer = getOperand(clusterId, input, newInputType, inputInsertionPoint, ddrOffsets[clusterId]);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, outputInsertionPoint, Byte(0));
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, dmaPort), permuteOp.getIsOutOfOrderAttr(), permuteOp.getIsCriticalAttr(),
                permuteOp.getMemPermAttr(), subDmaDescriptors[clusterId], permuteOp.getDmaHwpIdAttr(),
                permuteOp.getProfilingMetadataAttr());

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);

        newPermuteDMAs.push_back(newPermuteDMAOp);
    }
    rewriter.eraseOp(vpurtTask);

    // unrolling per distributed type is done, now rewrite PermuteOp itself
    for (const auto& permuteDMA : newPermuteDMAs) {
        if (rewritePermuteDMA(permuteDMA, rewriter).failed()) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

mlir::LogicalResult PermuteRewriter::unrollDuplicatedOutput(VPUIP::PermuteDMAOp permuteOp,
                                                            VPUIP::DistributedBufferType distributedType,
                                                            mlir::AffineMap memPerm,
                                                            mlir::PatternRewriter& rewriter) const {
    auto loc = permuteOp->getLoc();
    auto ctx = permuteOp->getContext();

    const auto input = permuteOp.getInput();
    const auto output = permuteOp.getOutputBuff();

    const auto inputType = permuteOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = permuteOp.getOutputBuff().getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", distributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteOp);

    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::DUPLICATED, "Unsupported distributed mode");

    rewriter.setInsertionPointAfter(vpurtTask);

    const auto perClusterOutShape = distributedType.getPerClusterMemoryShapes().front();
    const auto perClusterShapeOffset = distributedType.getPerClusterMemoryShapeOffsets().front();

    const auto getOperand = [&](mlir::Value operand, vpux::NDTypeInterface newType) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(permuteOp->getLoc(), cst, perClusterShapeOffset.raw(),
                                                     perClusterOutShape.raw());
        }
        auto insertionPoint = operand.getDefiningOp();

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, permuteOp->getLoc(), newType,
                                                           VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, clusters),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }
        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, Byte(0));
    };

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DMADescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);
    const auto newInputType =
            getPerClusterInputType(inputType, outputType, memPerm, perClusterOutShape, perClusterShapeOffset);

    const auto changeShapeElemTypeForDistributedBuff = [](VPUIP::DistributedBufferType buff, ShapeRef shape,
                                                          mlir::Type elemType) {
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(buff.getDistribution())) {
            auto distribution = buff.getDistribution();
            VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                            "DistributedBuffer has mode different from DUPLICATED after unrolling");
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    shape, distribution.getMode(), nullptr, distribution.getNumClusters(), nullptr,
                    distribution.getUniformDistributedSegments(), buff.getContext());
            return buff.changeShapeElemTypeForExplicitDistribution(shape, elemType, newDistribution);
        }

        return buff.changeShapeElemType(shape, elemType);
    };

    auto newOutType = changeShapeElemTypeForDistributedBuff(distributedType, perClusterOutShape,
                                                            distributedType.getElementType());

    const auto inputBuffer = getOperand(input, newInputType);
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);
    const auto outBuffer = getOperand(output, newOutType);
    _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, inputBuffer, outBuffer,
            vpux::getIntAttr(rewriter, 0), permuteOp.getIsOutOfOrderAttr(), permuteOp.getIsCriticalAttr(),
            permuteOp.getMemPermAttr(), subDmaDescriptor, permuteOp.getDmaHwpIdAttr(),
            permuteOp.getProfilingMetadataAttr());

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    rewriter.eraseOp(vpurtTask);

    return rewritePermuteDMA(newPermuteDMAOp, rewriter);
}

mlir::LogicalResult PermuteRewriter::unrollDuplicatedInputAndOutput(VPUIP::PermuteDMAOp permuteOp,
                                                                    mlir::AffineMap memPerm,
                                                                    mlir::PatternRewriter& rewriter) const {
    auto loc = permuteOp->getLoc();
    auto ctx = permuteOp->getContext();

    const auto input = permuteOp.getInput();
    const auto output = permuteOp.getOutputBuff();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    const auto inDistributedType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    const auto outDistributedType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();

    const auto inMode = inDistributedType.getDistribution().getMode().getValue();
    const auto outMode = outDistributedType.getDistribution().getMode().getValue();
    VPUX_THROW_UNLESS(VPU::bitEnumContainsAny(inMode, VPU::DistributionMode::DUPLICATED) &&
                              VPU::bitEnumContainsAny(outMode, VPU::DistributionMode::DUPLICATED),
                      "Unsupported mode");

    const auto distributionAttr = outDistributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", outDistributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteOp);

    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::DUPLICATED, "Unsupported distributed mode");

    rewriter.setInsertionPointAfter(vpurtTask);

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DMADescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);

    // create new input buffer
    auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, 0)});
    const auto inType = inDistributedType.getCompactType().cast<vpux::NDTypeInterface>();
    const auto newInType = inType.changeMemSpace(symbolAttr);
    auto inputBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, ArrayRef({0})),
            inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

    // create new output buffer
    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer offset");
    auto outBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
            getIntArrayAttr(ctx, ArrayRef(clusters)), outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, inputBuffer, outBuffer,
            vpux::getIntAttr(rewriter, 0), permuteOp.getIsOutOfOrderAttr(), permuteOp.getIsCriticalAttr(),
            permuteOp.getMemPermAttr(), subDmaDescriptor, permuteOp.getDmaHwpIdAttr(),
            permuteOp.getProfilingMetadataAttr());

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    rewriter.eraseOp(vpurtTask);

    return rewritePermuteDMA(newPermuteDMAOp, rewriter);
}

mlir::LogicalResult PermuteRewriter::unrollDuplicatedInput(VPUIP::PermuteDMAOp permuteOp, mlir::AffineMap memPerm,
                                                           mlir::PatternRewriter& rewriter) const {
    auto loc = permuteOp->getLoc();
    auto ctx = permuteOp->getContext();

    const auto input = permuteOp.getInput();
    const auto output = permuteOp.getOutputBuff();

    const auto inDistributedType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    const auto inMode = inDistributedType.getDistribution().getMode().getValue();
    VPUX_THROW_UNLESS(VPU::bitEnumContainsAny(inMode, VPU::DistributionMode::DUPLICATED), "Unsupported mode");

    const auto inputType = inDistributedType.getCompactType().dyn_cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    auto vpurtTask = permuteOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", permuteOp);

    rewriter.setInsertionPointAfter(vpurtTask);

    auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inputType, outputType, memPerm, _log).value();
    auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inputType, outputType, memPerm, _log).value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inputType, memPerm);
    auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    // calculate the dma descriptor
    VPUIP::DMADescriptorAttr subDmaDescriptor =
            dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);

    // create new input buffer
    auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer offset");
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, 0)});
    const auto newInType = inputType.changeMemSpace(symbolAttr);
    auto inputBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, ArrayRef({0})),
            inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

    // create new output buffer
    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();

    auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), loc, inputBuffer, outDeclBuff,
            vpux::getIntAttr(rewriter, 0), permuteOp.getIsOutOfOrderAttr(), permuteOp.getIsCriticalAttr(),
            permuteOp.getMemPermAttr(), subDmaDescriptor, permuteOp.getDmaHwpIdAttr(),
            permuteOp.getProfilingMetadataAttr());

    _log.trace("Insert new permute dma : '{0}'", newPermuteDMAOp);
    rewriter.eraseOp(vpurtTask);

    return rewritePermuteDMA(newPermuteDMAOp, rewriter);
}

//
// UnrollPermuteToNNDMAPass
//

class UnrollPermuteToNNDMAPass final : public VPUIP::impl::UnrollPermuteToNNDMABase<UnrollPermuteToNNDMAPass> {
public:
    explicit UnrollPermuteToNNDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPermuteToNNDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollPermuteToNNDMAPass)) {
        return;
    }
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteRewriter>(&ctx, dmaPortCount, _log.nest());
    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPermuteToNNDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollPermuteToNNDMAPass(Logger log) {
    return std::make_unique<UnrollPermuteToNNDMAPass>(log);
}
