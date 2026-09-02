//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LogSoftmaxTopKOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LogSoftmaxTopKOpAdaptor logSoftmaxTopK(operands, attrs, prop);
    if (mlir::failed(logSoftmaxTopK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(logSoftmaxTopK.getInput().getType());
    const auto dstElemType = logSoftmaxTopK.getDstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    auto si64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);

    if (logSoftmaxTopK.getAxisIndAttr() == nullptr) {
        return mlir::failure();
    }

    const auto axis = logSoftmaxTopK.getAxisIndAttr().getValue().getSExtValue();
    if (axis < 0 || axis >= static_cast<int64_t>(inType.getRank())) {
        return mlir::failure();
    }

    const auto inputShape = inType.getShape().raw();
    SmallVector<int64_t> topKShape(inputShape.begin(), inputShape.end());
    topKShape[axis] = 1;

    if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(inType)) {
        const auto& distribution = distributedType.getDistribution();

        // Create new distribution with updated shapes for width=1
        const auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                ShapeRef(topKShape), distribution.getMode(), distribution.getNumTiles(), distribution.getNumClusters(),
                distribution.getAlignment(), distribution.getUniformDistributedSegments(), ctx);

        inferredReturnTypes.push_back(VPU::DistributedTensorType::get(
                ctx, topKShape, si64Type, distributedType.getOrder(), distributedType.getMemSpace(), newDistribution));
    } else {
        inferredReturnTypes.push_back(inType.changeShape(ShapeRef(topKShape)).changeElemType(si64Type));
    }

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::LogSoftmaxTopKOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                 vpux::Logger /*log*/) {
    return TilingInfo(outputTile);
}

vpux::OutputTiling vpux::VPU::LogSoftmaxTopKOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                                vpux::Logger /*log*/) {
    const auto axis = getAxisIndAttr().getValue().getSExtValue();
    return logSoftmaxTopKOutputTiling(firstOutputTile, axis);
}

vpux::TileInfo vpux::VPU::LogSoftmaxTopKOp::getMainOutputTile(mlir::OpResult /*secondaryOutput*/,
                                                              const vpux::TileInfo& secondaryOutputTile,
                                                              vpux::Logger /*log*/) {
    auto mainShape = secondaryOutputTile.shape;
    auto mainOffsets = secondaryOutputTile.offsets;
    auto mainAxis = secondaryOutputTile.axis;

    // No tiling is done on axis dim (see getTilingStrategy impl),
    // therefore it is safe to infer main output from topk output
    const auto axis = getAxisIndAttr().getValue().getSExtValue();
    const auto axisDim = Dim(axis);
    mainShape[axisDim] = getBoundedShape(getOutput())[axisDim];
    mainOffsets[axisDim] = 0;
    mainAxis[axisDim] = 1;

    return vpux::TileInfo(mainShape, mainOffsets, mainAxis);
}

void vpux::VPU::LogSoftmaxTopKOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::LogSoftmaxTopKOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for LogSoftmaxTopK currently, for op {0} at '{1}'",
                    baseOp->getName(), getLoc());
    auto axis = this->getAxisIndAttr().getValue().getSExtValue();
    int64_t tileDim = 0;
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        if (tileDim == axis) {
            ++tileDim;
        } else {
            if (nTilesOnDim[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
                ++tileDim;
            } else {
                ++nTilesOnDim[Dim(tileDim)];
            }
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    auto origTiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
    return origTiles;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::LogSoftmaxTopKOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto inShape = inputType.getShape();
    auto numClusters = VPU::getOptimalNumClusters(getOperation(), outputType.getShape(), strategy);

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    // Split input/output by H dim when axisInd is not point to H
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && getAxisInd() != Dims4D::Act::H.ind() &&
        inShape[Dims4D::Act::H] >= numClusters) {
        return true;
    }

    // Split input/output by C dim when axisInd is not point to C
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && getAxisInd() != Dims4D::Act::C.ind() &&
        inShape[Dims4D::Act::C] >= numClusters) {
        return true;
    }

    // Split input/output by W dim when axisInd is not point to W
    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth && getAxisInd() != Dims4D::Act::W.ind() &&
        inShape[Dims4D::Act::W] >= numClusters) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::LogSoftmaxTopKOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::LogSoftmaxTopKOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    const auto expectedBuffers = 3;  // 1 input + 2 outputs
    VPUX_THROW_UNLESS(buffers.size() == expectedBuffers,
                      "LogSoftmaxTopKOp requires 1 input and {0} output(s), but the number of buffer is {1}",
                      expectedBuffers, buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::LogSoftmaxTopKOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::LogSoftmaxTopKOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::LogSoftmaxTopKOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                        ::mlir::Value input, ::mlir::IntegerAttr axisInd, ::mlir::IntegerAttr padSize,
                                        ::mlir::TypeAttr dstElemType) {
    build(odsBuilder, odsState, input, axisInd, padSize, dstElemType, nullptr);
}
