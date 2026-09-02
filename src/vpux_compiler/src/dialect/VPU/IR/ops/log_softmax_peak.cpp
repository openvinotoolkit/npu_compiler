//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LogSoftmaxPeakOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LogSoftmaxPeakOpAdaptor logSoftmaxPeak(operands, attrs, prop);
    if (mlir::failed(logSoftmaxPeak.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(logSoftmaxPeak.getInput().getType());
    const auto dstElemType = logSoftmaxPeak.getDstElemType();

    auto inputShape = inType.getShape().raw();
    SmallVector<int64_t> topKShape(inputShape.begin(), inputShape.end());
    topKShape.back() = 1;

    auto si64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);

    mlir::Type ouputType;
    mlir::Type topKOutType;
    if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(inType)) {
        auto distribution = distributedType.getDistribution();

        // Create new distribution with updated shapes for width=1
        auto newDistributionOut = VPU::getNonOverlappedDistributedAttr(
                ShapeRef(topKShape), distribution.getMode(), distribution.getNumTiles(), distribution.getNumClusters(),
                distribution.getAlignment(), distribution.getUniformDistributedSegments(), ctx);

        ouputType = VPU::DistributedTensorType::get(ctx, topKShape, dstElemType, distributedType.getOrder(),
                                                    distributedType.getMemSpace(), newDistributionOut);

        auto newDistributionTopK = VPU::getNonOverlappedDistributedAttr(
                ShapeRef(topKShape), distribution.getMode(), distribution.getNumTiles(), distribution.getNumClusters(),
                distribution.getAlignment(), distribution.getUniformDistributedSegments(), ctx);

        topKOutType = VPU::DistributedTensorType::get(ctx, topKShape, si64Type, distributedType.getOrder(),
                                                      distributedType.getMemSpace(), newDistributionTopK);
    } else {
        ouputType = inType.changeShape(ShapeRef(topKShape)).changeElemType(dstElemType);
        topKOutType = inType.changeShape(ShapeRef(topKShape)).changeElemType(si64Type);
    }

    inferredReturnTypes.push_back(ouputType);
    inferredReturnTypes.push_back(topKOutType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::LogSoftmaxPeakOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                 vpux::Logger /*log*/) {
    // The input shape differs from output shape -> input has full width on axis dimension
    // Output shape has axis dimension = 1, but input needs full size on that dimension
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto numDims = static_cast<int64_t>(inputShape.size());

    auto axis = getAxisInd();
    if (axis < 0) {
        axis += numDims;
    }

    // Create input tile info based on output tile, but with full size on axis dimension
    SmallVector<int64_t> inputTileShapeVec(outputTile.shape.begin(), outputTile.shape.end());
    SmallVector<int64_t> inputTileOffsetsVec(outputTile.offsets.begin(), outputTile.offsets.end());
    SmallVector<int64_t> inputTileAxisVec(outputTile.axis.begin(), outputTile.axis.end());

    // The axis dimension of input needs the full size (not tiled)
    inputTileShapeVec[axis] = inputShape[Dim(axis)];
    inputTileOffsetsVec[axis] = 0;

    Shape inputTileShape(inputTileShapeVec);
    Shape inputTileOffsets(inputTileOffsetsVec);
    Shape inputTileAxis(inputTileAxisVec);

    TileInfo inputTile(inputTileShape, inputTileOffsets, inputTileAxis);

    return TilingInfo(inputTile);
}

vpux::OutputTiling vpux::VPU::LogSoftmaxPeakOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                                vpux::Logger /*log*/) {
    return OutputTiling({firstOutputTile, firstOutputTile});
}

vpux::TileInfo vpux::VPU::LogSoftmaxPeakOp::getMainOutputTile(mlir::OpResult /*secondaryOutput*/,
                                                              const vpux::TileInfo& secondaryOutputTile,
                                                              vpux::Logger /*log*/) {
    return secondaryOutputTile;
}

void vpux::VPU::LogSoftmaxPeakOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::LogSoftmaxPeakOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for LogSoftmaxPeak currently, for op {0} at '{1}'",
                    baseOp->getName(), getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    const auto numDims = static_cast<int64_t>(outputShape.size());

    // Normalize axis to positive value
    auto axis = this->getAxisIndAttr().getValue().getSExtValue();
    if (axis < 0) {
        axis += numDims;
    }

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t tileDim = 0;

    // Skip axis dimension from the start if tileDim starts at axis
    if (tileDim == axis) {
        ++tileDim;
    }

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        log.trace("outputShape size: {0}, tileDim: {1}, axis: {2}", numDims, tileDim, axis);
        // Check if we've exhausted all dimensions
        if (tileDim >= numDims) {
            log.warning("LogSoftmaxPeakOp: Unable to find valid tiling strategy");
            return mlir::failure();
        }

        if (nTilesOnDim[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
            ++tileDim;
            // Skip the axis dimension (can't tile along softmax axis)
            if (tileDim == axis) {
                ++tileDim;
            }
        } else {
            ++nTilesOnDim[Dim(tileDim)];
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    auto origTiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
    return origTiles;
}
//
// ClusteredOpInterface
//

bool vpux::VPU::LogSoftmaxPeakOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
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

vpux::VPU::DistributionInfo vpux::VPU::LogSoftmaxPeakOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::LogSoftmaxPeakOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    const auto expectedBuffers = 3;  // 1 input + 2 outputs
    VPUX_THROW_UNLESS(buffers.size() == expectedBuffers,
                      "LogSoftmaxPeakOp requires 1 input and 2 output, but the number of buffer is {1}",
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

bool vpux::VPU::LogSoftmaxPeakOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::LogSoftmaxPeakOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::LogSoftmaxPeakOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                        ::mlir::Value input, ::mlir::IntegerAttr axisInd, ::mlir::IntegerAttr padSize,
                                        ::mlir::TypeAttr dstElemType) {
    build(odsBuilder, odsState, input, axisInd, padSize, dstElemType, nullptr);
}
