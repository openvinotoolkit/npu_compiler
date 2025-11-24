//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_space_to_depth_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {

// Minimum rank is 3, as defined by the opset
static constexpr auto MIN_RANK = 3;

SingleClusterSpaceToDepthDMARewriter::SingleClusterSpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                           Logger log)
        : mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(std::move(log)) {
    setDebugName("SingleClusterSpaceToDepthDMARewriter");
}

bool isMultiClusterSpaceToDepthDMAOp(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp) {
    const auto outDistributedType =
            mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(spaceToDepthDMAOp.getOutputBuff().getType());

    return (outDistributedType != nullptr);
}

mlir::LogicalResult SingleClusterSpaceToDepthDMARewriter::matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (spaceToDepthDMAOp.getInternalDataFlow().has_value()) {
        _log.trace("This SpaceToDepthDMAOp has already been unrolled.");
        return mlir::failure();
    }

    if (!isMultiClusterSpaceToDepthDMAOp(spaceToDepthDMAOp)) {
        _log.trace("Got SpaceToDepthDMAOp '{0}' at '{1}'", spaceToDepthDMAOp->getName(), spaceToDepthDMAOp->getLoc());
        return unroll(spaceToDepthDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterSpaceToDepthDMARewriter::unroll(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto vpurtTask = spaceToDepthDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inDeclBuff = spaceToDepthDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outDeclBuff = spaceToDepthDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto inType = mlir::cast<vpux::NDTypeInterface>(spaceToDepthDMAOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(spaceToDepthDMAOp.getOutput().getType());

    VPUX_THROW_WHEN(inType.getRank() < MIN_RANK, "Unrolling is supported only for rank {0} or higher shapes", MIN_RANK);
    VPUX_THROW_WHEN(outType.getRank() != inType.getRank(),
                    "Unrolling is not supported for input and output of different ranks");

    const auto blockSize = spaceToDepthDMAOp.getBlockSize();
    const auto blocksFirst = spaceToDepthDMAOp.getMode() == IE::SpaceToDepthMode::BLOCKS_FIRST;

    const auto buildTaskOp = [&](auto internalInputMemRef, auto inputBuffer, auto internalOutputMemRef,
                                 auto outputBuffer, auto internalInToOutMapping, int64_t dmaPort) {
        auto mappingOrder = mlir::AffineMapAttr::get(internalInToOutMapping);
        // After internal input and output representations have been obtained, the optimal loop order to obtain the
        // minimal number of DMA transactions can be computed/fetched from a cache.
        auto loopOrder = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(VPUIP::getLinearMemOrder(internalInputMemRef), ctx));
        auto internalDataFlowAttr = VPUIP::InternalDataFlowAttr::get(ctx, internalInputMemRef, internalOutputMemRef,
                                                                     mappingOrder, loopOrder);

        VPURT::wrapIntoTaskOp<VPUIP::SpaceToDepthDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), inputBuffer,
                outputBuffer, vpux::getIntAttr(rewriter, dmaPort), spaceToDepthDMAOp.getBlockSizeAttr(),
                spaceToDepthDMAOp.getModeAttr(), nullptr, spaceToDepthDMAOp.getIsOutOfOrderAttr(),
                spaceToDepthDMAOp.getIsCriticalAttr(), spaceToDepthDMAOp.getDmaHwpIdAttr(),
                spaceToDepthDMAOp.getProfilingMetadataAttr(), internalDataFlowAttr);
    };

    // Before determining the internal representation of the data movement of a single transaction, we would need to
    // ensure the transaction can be executed by the DMA. Here we would need to query the DMA engine limits from the
    // DMAEngineLimits class. However, no actual unrolling for engine capabilities is done for now here as the
    // maximum transfer and stride level for NPU4+ is sufficient for all but the largest transfers (i.e. > 4 GB).
    // For now, we unroll solely to cover a target number of ports.

    VPUIP::splitSpaceToDepth(rewriter, buildTaskOp, vpurtTask, inType, inDeclBuff, outType, outDeclBuff, blockSize,
                             blocksFirst, _dmaPortCount);

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

MultiClusterSpaceToDepthDMARewriter::MultiClusterSpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                         Logger log)
        : mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(std::move(log)) {
    setDebugName("MultiClusterSpaceToDepthDMARewriter");
}

mlir::LogicalResult MultiClusterSpaceToDepthDMARewriter::matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    if (isMultiClusterSpaceToDepthDMAOp(spaceToDepthDMAOp)) {
        _log.trace("Got SpaceToDepthDMAOp '{0}' at '{1}'", spaceToDepthDMAOp->getName(), spaceToDepthDMAOp->getLoc());
        return unroll(spaceToDepthDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult MultiClusterSpaceToDepthDMARewriter::unroll(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto ctx = spaceToDepthDMAOp->getContext();
    auto loc = spaceToDepthDMAOp->getLoc();

    const auto input = spaceToDepthDMAOp.getInput();
    const auto output = spaceToDepthDMAOp.getOutputBuff();

    const auto outDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());
    VPUX_THROW_WHEN(outDistributedType == nullptr,
                    "Expect distributed type for SpaceToDepthDMA op output, but got: {0}", output.getType());

    const auto distributionAttr = outDistributedType.getDistribution();
    VPUX_THROW_WHEN(distributionAttr == nullptr, "Failed to extract distribution attribute from distributed type.");

    const auto modeAttr = distributionAttr.getMode();
    VPUX_THROW_WHEN(modeAttr == nullptr, "Failed to extract mode from distribution attribute.");
    const auto mode = modeAttr.getValue();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(outDistributedType.getCompactType());

    VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                              outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                      "Unexpected memory space: input {0}, output {1}", inputType.getMemoryKind(),
                      outputType.getMemoryKind());

    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distribution mode: {0}", modeAttr);

    const auto blockSize = spaceToDepthDMAOp.getBlockSize();

    const auto perClusterOutShapes = outDistributedType.getPerClusterMemoryShapes();
    const auto perClusterOutShapeOffsets = outDistributedType.getPerClusterMemoryShapeOffsets();
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    const auto backInferInputShape = [&](ShapeRef outShape, int64_t blockSize) {
        auto inShape = Shape(outShape.raw());
        inShape[Dims4D::Act::H] *= blockSize;
        inShape[Dims4D::Act::W] *= blockSize;
        inShape[Dims4D::Act::C] /= (blockSize * blockSize);
        return inShape;
    };

    const auto origStrides = inputType.getStrides();
    const auto numClusters = perClusterOutShapes.size();

    SmallVector<NDTypeInterface> inTypes(numClusters);
    SmallVector<NDTypeInterface> outTypes(numClusters);
    for (auto clusterId : irange(numClusters)) {
        const auto offsets = backInferInputShape(perClusterOutShapeOffsets[clusterId], blockSize);
        const auto shape = backInferInputShape(perClusterOutShapes[clusterId], blockSize);
        inTypes[clusterId] = inputType.extractDenseTile(offsets, shape).changeStrides(origStrides);
        outTypes[clusterId] =
                outputType.extractDenseTile(perClusterOutShapeOffsets[clusterId], perClusterOutShapes[clusterId]);
    }

    auto vpurtTask = spaceToDepthDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", spaceToDepthDMAOp);

    rewriter.setInsertionPointAfter(vpurtTask);

    const auto getInputOperand = [&](mlir::Value operand, vpux::NDTypeInterface newType,
                                     mlir::Operation* insertionPoint, Byte offset) -> mlir::Value {
        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        Byte cmxOffset{declBuff.getByteOffset()};
        cmxOffset += offset;

        auto declBuffType = mlir::cast<vpux::NDTypeInterface>(declBuff.getType());
        VPUX_THROW_UNLESS(declBuffType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Currently only support input in CMX");
        auto sectionIndex = declBuffType.getMemSpace().getIndex();
        VPUX_THROW_UNLESS(sectionIndex.has_value() && sectionIndex.value() == 0,
                          "Currently only support input in CMX0");

        auto section = declBuff.getSection();
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)), 0);
        newType = newType.changeMemSpace(symbolAttr);

        auto newDeclBuff = declBuff;

        auto memSpaceIndex = declBuffType.getMemSpace().getIndex();
        if (memSpaceIndex.has_value()) {
            newDeclBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                                  memSpaceIndex.value(), cmxOffset.count());
        } else {
            newDeclBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                                  cmxOffset.count());
        }

        return newDeclBuff;
    };

    const auto getOutputOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                      mlir::Operation* insertionPoint) -> mlir::Value {
        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
        auto newCMXType = newType.changeMemSpace(symbolAttr);

        return VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, insertionPoint, spaceToDepthDMAOp->getLoc(), newCMXType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(ctx, ArrayRef({clusterId})), declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
    };

    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tilingDim = Dim(VPU::getDistributedTilingAxis(tilingScheme));

    for (auto clusterId : irange(numClusters)) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto cmxOffset =
                perClusterOutShapeOffsets[clusterId][tilingDim] * static_cast<Byte>(newOutType.getStrides()[tilingDim]);
        const auto inputBuffer = getInputOperand(input, newInputType, inputInsertionPoint, cmxOffset);

        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOutputOperand(clusterId, output, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        const auto dmaPort = clusterId % _dmaPortCount;
        auto newSpaceToDepthDMAOp = VPURT::wrapIntoTaskOp<VPUIP::SpaceToDepthDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, dmaPort), spaceToDepthDMAOp.getBlockSizeAttr(),
                spaceToDepthDMAOp.getModeAttr(),
                /*dma_descriptor*/ nullptr, spaceToDepthDMAOp.getIsOutOfOrderAttr(),
                spaceToDepthDMAOp.getIsCriticalAttr(), spaceToDepthDMAOp.getDmaHwpIdAttr(),
                spaceToDepthDMAOp.getProfilingMetadataAttr(),
                /*InternalDataFlowAttr*/ nullptr);

        _log.trace("Insert new SpaceToDepthDMA: '{0}'", newSpaceToDepthDMAOp);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

}  // namespace vpux::VPUIP
