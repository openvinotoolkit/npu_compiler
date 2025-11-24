//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_per_axis_tile_dma.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numeric>

namespace vpux::VPUIP {

using namespace vpux;

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset, int64_t padAxis) {
    const auto elemType = originType.getElementType();
    auto newShape = to_small_vector(shape);
    newShape[padAxis] = originType.getShape()[Dim(padAxis)];
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto newQType = tileScalesAndZP(qType, ShapeRef(newShape), offset);
        return originType.changeShapeElemType(ShapeRef(newShape), newQType);
    }

    return originType.changeShape(ShapeRef(newShape));
}

bool isMultiClusterPerAxisTileDMA(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp) {
    const auto output = perAxisTileDMAOp.getOutput();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);

    if (distributedType != nullptr) {
        const auto distributionAttr = distributedType.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
            VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
            auto outDeclBuff = perAxisTileDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

            return !outDeclBuff.getSectionIndex().has_value();
        }

        return true;
    }

    return false;
}

SingleClusterPerAxisTileDMARewriter::SingleClusterPerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                         Logger log)
        : mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
    setDebugName("SingleClusterPerAxisTileDMARewriter");
}

mlir::LogicalResult SingleClusterPerAxisTileDMARewriter::matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Process PerAxisTileDMAOp: {0}", perAxisTileDMAOp);

    if (perAxisTileDMAOp.getTilesAttr() == nullptr && perAxisTileDMAOp.getAxisAttr() == nullptr) {
        return mlir::failure();
    }

    if (!isMultiClusterPerAxisTileDMA(perAxisTileDMAOp)) {
        return unroll(perAxisTileDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterPerAxisTileDMARewriter::unroll(VPUIP::PerAxisTileDMAOp, mlir::PatternRewriter&) const {
    VPUX_UNUSED(_dmaPortCount);
    return mlir::failure();
}

MultiClusterPerAxisTileDMARewriter::MultiClusterPerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                       bool useDMADescriptorAttr, Logger log)
        : mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp>(ctx),
          _dmaPortCount(dmaPortCount),
          _log(log),
          _useDMADescriptorAttr(useDMADescriptorAttr) {
    setDebugName("MultiClusterPerAxisTileDMARewriter");
}

mlir::LogicalResult MultiClusterPerAxisTileDMARewriter::matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Process PerAxisTileDMAOp: {0}", perAxisTileDMAOp);

    if (perAxisTileDMAOp.getTilesAttr() == nullptr && perAxisTileDMAOp.getAxisAttr() == nullptr) {
        return mlir::failure();
    }

    if (isMultiClusterPerAxisTileDMA(perAxisTileDMAOp)) {
        const auto output = perAxisTileDMAOp.getOutput();
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
        const auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);

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

    return mlir::failure();
}

mlir::LogicalResult MultiClusterPerAxisTileDMARewriter::unrollSegmentedOrOverlapped(
        VPUIP::PerAxisTileDMAOp perAxisTileDMAOp, VPUIP::DistributedBufferType distributedType,
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
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
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

        DMADescriptorAttr dmaDescriptorAttr;
        if (_useDMADescriptorAttr) {
            auto mergedShapes =
                    VPUIP::getPerAxisTileDMAMergedShape(newInputType, newOutType, padAxis.value(), padTiles.value());
            dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second,
                                                                padTiles.value(), elemTypeSize.count());
        }

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

    return mlir::success();
}

mlir::LogicalResult MultiClusterPerAxisTileDMARewriter::unrollDuplicated(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                         VPUIP::DistributedBufferType distributedType,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto loc = perAxisTileDMAOp->getLoc();
    auto ctx = perAxisTileDMAOp->getContext();

    const auto input = perAxisTileDMAOp.getInput();
    const auto output = perAxisTileDMAOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());

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

    DMADescriptorAttr dmaDescriptorAttr;
    if (_useDMADescriptorAttr) {
        auto elemTypeSize = Byte(inputType.getElemTypeSize());
        auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inputType, outputType, axis.value(), tiles.value());
        auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, tiles.value(),
                                                            elemTypeSize.count());
    }

    const auto newLoc = appendLoc(loc, "_broadcast_copy_to_CMX[{0},{1}]", clusters.front(), clusters.back());
    const auto newPerAxisTileDMA = VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, input, cmxBuffer,
            vpux::getIntAttr(rewriter, 0), perAxisTileDMAOp.getAxisAttr(), perAxisTileDMAOp.getTilesAttr(),
            dmaDescriptorAttr, perAxisTileDMAOp.getIsOutOfOrderAttr(), perAxisTileDMAOp.getIsCriticalAttr(),
            perAxisTileDMAOp.getDmaHwpIdAttr(), perAxisTileDMAOp.getProfilingMetadataAttr());

    _log.trace("Insert new PerAxisTileDMA op: '{0}'", newPerAxisTileDMA);

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

}  // namespace vpux::VPUIP
