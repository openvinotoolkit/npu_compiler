//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numeric>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLPERAXISTILEDMA
#define GEN_PASS_DEF_UNROLLPERAXISTILEDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// PerAxisTileDMARewriter
//

class PerAxisTileDMARewriter final : public mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp> {
public:
    PerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("PerAxisTileDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unrollSegmentedOrOverlapped(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                    VPUIP::DistributedBufferType distributedType,
                                                    mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicated(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                         VPUIP::DistributedBufferType distributedType,
                                         mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollPerAxisTile(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                          mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset, int64_t padAxis) {
    const auto elemType = originType.getElementType();
    auto newShape = to_small_vector(shape);
    newShape[padAxis] = originType.getShape()[Dim(padAxis)];
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, ShapeRef(newShape), offset);
        return originType.changeShapeElemType(ShapeRef(newShape), newQType);
    }

    return originType.changeShape(ShapeRef(newShape));
}

mlir::LogicalResult PerAxisTileDMARewriter::matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Process PerAxisTileDMAOp: {0}", perAxisTileDMAOp);

    if (perAxisTileDMAOp.getTilesAttr() == nullptr && perAxisTileDMAOp.getAxisAttr() == nullptr) {
        return mlir::failure();
    }

    const auto output = perAxisTileDMAOp.getOutput();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();

    // if PerAxisTileDMAOp has output of DistributedType -- unroll according to DistributedType first,
    // then unroll per axis/tile
    if (distributedType != nullptr) {
        _log.trace("PerAxisTile Op with distributed type at {0}", perAxisTileDMAOp);

        VPUX_THROW_WHEN(mlir::isa<VPUIP::DistributedBufferType>(perAxisTileDMAOp.getInput().getType()),
                        "Input buffer of PerAxisTileDMAOp cannot be Distributed");

        const auto distributionAttr = distributedType.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return unrollSegmentedOrOverlapped(perAxisTileDMAOp, distributedType, rewriter);
        } else if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
            return unrollDuplicated(perAxisTileDMAOp, distributedType, rewriter);
        } else {
            VPUX_THROW("Unsupported distributed mode");
        }
    }
    // otherwise -- unroll only per axis/tile
    return unrollPerAxisTile(perAxisTileDMAOp, rewriter);
}

mlir::LogicalResult PerAxisTileDMARewriter::unrollPerAxisTile(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = perAxisTileDMAOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = perAxisTileDMAOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcDeclBuff = perAxisTileDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = perAxisTileDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();

    auto createSubPerAxisTileDMAOp = [&](ShapeRef subInShape, ShapeRef subOutShape, int64_t srcOffset,
                                         int64_t dstOffset, VPUIP::DMADescriptorAttr dmaDescriptor, int64_t port) {
        const auto dimOrder = DimsOrder::CHW;
        auto getStrides = [](ShapeRef shape, Byte elemTypeSize) -> Strides {
            const auto strides = SmallVector<vpux::Bit>{Bit(shape[Dim(1)] * shape[Dim(2)] * Bit(elemTypeSize).count()),
                                                        Bit(shape.back() * Bit(elemTypeSize).count()),
                                                        Bit(Bit(elemTypeSize).count())};
            return Strides(strides);
        };

        auto newSrcMemRef = vpux::getMemRefType(subInShape, srcType.getElementType(), dimOrder, srcType.getMemSpace(),
                                                getStrides(subInShape, elemTypeSize));

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

        mlir::Type newDstType;
        if (auto dstDistributedType = dstType.dyn_cast<VPUIP::DistributedBufferType>()) {
            auto distributionAttr = dstDistributedType.getDistribution();
            VPUX_THROW_WHEN(
                    distributionAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "Issues with unrolling PerAxiTileDMA; Buffer has distributed type != DUPLICATED after unroll");
            if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
                distributionAttr = VPU::getNonOverlappedDistributedAttr(
                        subOutShape, distributionAttr.getMode(), nullptr, distributionAttr.getNumClusters(), nullptr,
                        distributionAttr.getUniformDistributedSegments(), dstDeclBuff.getContext());
            }

            const auto layout = mlir::AffineMapAttr::get(dimOrder.toAffineMap(ctx));
            newDstType = VPUIP::DistributedBufferType::get(ctx, subOutShape.raw(), dstType.getElementType(), layout,
                                                           dstType.getMemSpace(), distributionAttr);
        } else {
            newDstType = vpux::getMemRefType(subOutShape, dstType.getElementType(), dimOrder, dstType.getMemSpace(),
                                             getStrides(subOutShape, elemTypeSize));
        }

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

        _log.trace("Creating Sub-PerAxisTileDMAOp with inShape: {0} outShape: {1}, SrcMemory at {2}, DstMemory at {3}",
                   subInShape, subOutShape, newSrcBuff.getSection(), newDstBuff.getSection());

        VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), newSrcBuff,
                newDstBuff, vpux::getIntAttr(rewriter, port), nullptr, nullptr, dmaDescriptor,
                perAxisTileDMAOp.getIsOutOfOrderAttr(), perAxisTileDMAOp.getIsCriticalAttr(),
                perAxisTileDMAOp.getDmaHwpIdAttr(), perAxisTileDMAOp.getProfilingMetadataAttr());
    };

    _log.trace("Unroll PerAxisTileDMAOp {0}", perAxisTileDMAOp->getLoc());

    auto axis = perAxisTileDMAOp.getAxis();
    auto tiles = perAxisTileDMAOp.getTiles();
    VPUX_THROW_UNLESS(axis.has_value() && tiles.has_value(), "Cannot get PerAxisTile attribution");
    auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inType, outType, axis.value(), tiles.value());
    auto dmaDescriptorAttr = perAxisTileDMAOp.getDmaDescriptorAttr();
    auto portIsAlreadyAssigned = true;
    if (dmaDescriptorAttr == nullptr) {
        auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, tiles.value(),
                                                            elemTypeSize.count());
        portIsAlreadyAssigned = false;
    }

    const auto arch = VPU::getArch(perAxisTileDMAOp);
    auto subInputShapes = VPUIP::getPerAxisTileDMASubShapes(arch, mergedShapes.first);
    auto subOutputShapes = VPUIP::getPerAxisTileDMASubShapes(arch, mergedShapes.second);
    VPUX_THROW_UNLESS(subInputShapes.size() == subOutputShapes.size(),
                      "Unexpected PerAxisTileDMA subInput '{0}' and subOutput '{1}' number", subInputShapes.size(),
                      subOutputShapes.size());

    auto srcOffset = srcDeclBuff.getByteOffset();
    auto dstOffset = dstDeclBuff.getByteOffset();
    for (size_t idx = 0; idx < subInputShapes.size(); idx++) {
        auto dmaPort = idx % _dmaPortCount;

        auto newDmaPort = portIsAlreadyAssigned ? perAxisTileDMAOp.getPort().value() : dmaPort;
        auto newDMADescriptorAttr = VPUIP::updateNumPlanes(dmaDescriptorAttr, subInputShapes[idx][Dim(0)]);
        createSubPerAxisTileDMAOp(subInputShapes[idx], subOutputShapes[idx], srcOffset, dstOffset, newDMADescriptorAttr,
                                  newDmaPort);

        srcOffset += subInputShapes[idx].totalSize() * elemTypeSize.count();
        dstOffset += subOutputShapes[idx].totalSize() * elemTypeSize.count();
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult PerAxisTileDMARewriter::unrollSegmentedOrOverlapped(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                        VPUIP::DistributedBufferType distributedType,
                                                                        mlir::PatternRewriter& rewriter) const {
    auto loc = perAxisTileDMAOp->getLoc();
    auto ctx = perAxisTileDMAOp->getContext();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can't get VPURT.TaskOp for {0}", perAxisTileDMAOp);
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto input = perAxisTileDMAOp.getInput();
    const auto output = perAxisTileDMAOp.getOutputBuff();
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = distributedType.getCompactType();
    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto originInShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size(),
                      "Input shape size '{0}' and tiles array size '{1}' don't match", originInShape.size(),
                      numTiles.size());

    const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' don't match", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' don't match", perClusterShapeOffsets.size(),
                      numClusters);

    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    const auto tilingAxis = std::distance(numTiles.begin(), llvm::find_if(numTiles, isValidTile));

    const auto getOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                mlir::Operation* insertionPoint) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            return rewriter.create<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                     perClusterShapes[clusterId].raw());
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't find DeclareBuffer");

        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            auto newCMXType = newType.changeMemSpace(symbolAttr);

            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(ctx, ArrayRef({clusterId})),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
        }

        Byte ddrOffset{declBuff.getByteOffset()};
        ddrOffset += perClusterShapeOffsets[clusterId][Dim(tilingAxis)] *
                     static_cast<Byte>(newType.getStrides()[Dim(tilingAxis)]);

        auto section = declBuff.getSection();
        auto sectionIndex = declBuff.getSectionIndex();

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)));
        newType = newType.changeMemSpace(symbolAttr);

        if (sectionIndex.has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                           sectionIndex.value(), ddrOffset.count(), nullptr);
        }

        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                       ddrOffset.count());
    };

    auto padAxis = perAxisTileDMAOp.getAxis();
    auto padTiles = perAxisTileDMAOp.getTiles();
    VPUX_THROW_UNLESS(padAxis.has_value() && padTiles.has_value(), "Cannot get PerAxisTile attribute");
    VPUX_THROW_UNLESS(padAxis.value() != tilingAxis,
                      "TilePerAxis expand axis '{0}' should not be the same as tiling axis '{1}'", padAxis.value(),
                      tilingAxis);

    auto elemTypeSize = Byte(inputType.getElemTypeSize());
    auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);

    const auto tileType = [&](vpux::NDTypeInterface type) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(type, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId], padAxis.value());
        }

        return newTypes;
    };

    const auto inTypes = tileType(inputType);
    const auto outTypes = tileType(outputType);
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();
    SmallVector<VPUIP::PerAxisTileDMAOp> newOps;
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        auto mergedShapes =
                VPUIP::getPerAxisTileDMAMergedShape(newInputType, newOutType, padAxis.value(), padTiles.value());
        auto dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second,
                                                                 padTiles.value(), elemTypeSize.count());

        const auto inputBuffer = getOperand(clusterId, input, newInputType, inputInsertionPoint);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOperand(clusterId, output, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newDMAPort = clusterId % _dmaPortCount;
        auto newPerAxisTileDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, newDMAPort), perAxisTileDMAOp.getAxisAttr(), perAxisTileDMAOp.getTilesAttr(),
                dmaDescriptorAttr, perAxisTileDMAOp.getIsOutOfOrderAttr(), perAxisTileDMAOp.getIsCriticalAttr(),
                perAxisTileDMAOp.getDmaHwpIdAttr(), perAxisTileDMAOp.getProfilingMetadataAttr());

        _log.trace("Insert new PerAxisTile dma : '{0}'", newPerAxisTileDMAOp);

        newOps.push_back(newPerAxisTileDMAOp);
    }

    rewriter.eraseOp(vpurtTask);

    // unrolling per cluster tiling is done, now unroll per axis/tile
    for (const auto& op : newOps) {
        if (unrollPerAxisTile(op, rewriter).failed()) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

mlir::LogicalResult PerAxisTileDMARewriter::unrollDuplicated(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                             VPUIP::DistributedBufferType distributedType,
                                                             mlir::PatternRewriter& rewriter) const {
    auto loc = perAxisTileDMAOp->getLoc();
    auto ctx = perAxisTileDMAOp->getContext();

    const auto input = perAxisTileDMAOp.getInput();
    const auto output = perAxisTileDMAOp.getOutputBuff();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    VPUX_THROW_WHEN(numClusters == 0, "Invalid number of clusters for {0}", distributedType);

    SmallVector<int64_t> clusters(numClusters);
    std::iota(clusters.begin(), clusters.end(), 0);

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can't get VPURT.TaskOp for {0}", perAxisTileDMAOp);
    rewriter.setInsertionPointAfter(vpurtTask);

    auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

    auto cmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
            rewriter, outDeclBuff, loc, outDeclBuff.getType(), VPURT::BufferSection::CMX_NN,
            getIntArrayAttr(ctx, clusters), outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());

    _log.trace("Insert new CMX buffer declaration: '{0}'", cmxBuffer);

    auto axis = perAxisTileDMAOp.getAxis();
    auto tiles = perAxisTileDMAOp.getTiles();
    VPUX_THROW_UNLESS(axis.has_value() && tiles.has_value(), "Cannot get PerAxisTile attributes");
    auto elemTypeSize = Byte(inputType.getElemTypeSize());

    auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inputType, outputType, axis.value(), tiles.value());
    auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, tiles.value(),
                                                         elemTypeSize.count());

    const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
    const auto newPerAxisTileDMA = VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, input, cmxBuffer,
            vpux::getIntAttr(rewriter, 0), perAxisTileDMAOp.getAxisAttr(), perAxisTileDMAOp.getTilesAttr(),
            dmaDescriptor, perAxisTileDMAOp.getIsOutOfOrderAttr(), perAxisTileDMAOp.getIsCriticalAttr(),
            perAxisTileDMAOp.getDmaHwpIdAttr(), perAxisTileDMAOp.getProfilingMetadataAttr());

    _log.trace("Insert new PerAxisTileDMA op: '{0}'", newPerAxisTileDMA);

    rewriter.eraseOp(vpurtTask);

    // unrolling per cluster tiling is done, now unroll per axis/tile
    return unrollPerAxisTile(newPerAxisTileDMA, rewriter);
}

//
// UnrollPerAxisTileDMAPass
//

class UnrollPerAxisTileDMAPass final : public VPUIP::impl::UnrollPerAxisTileDMABase<UnrollPerAxisTileDMAPass> {
public:
    explicit UnrollPerAxisTileDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPerAxisTileDMAPass::safeRunOnFunc() {
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollPerAxisTileDMAPass)) {
        return;
    }
    auto& ctx = getContext();

    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PerAxisTileDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPerAxisTileDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollPerAxisTileDMAPass(Logger log) {
    return std::make_unique<UnrollPerAxisTileDMAPass>(log);
}
