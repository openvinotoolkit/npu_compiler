
//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherNDOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherNDOpAdaptor gatherND(operands, attrs, prop);
    if (mlir::failed(gatherND.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(gatherND.getInput().getType());
    auto originalShapeOptional = gatherND.getOriginalShape();
    Shape inputShape = originalShapeOptional.has_value()
                               ? Shape(parseIntArrayAttr<int64_t>(originalShapeOptional.value()))
                               : Shape(inType.getShape());

    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(gatherND.getIndices().getType());
    const auto indicesShape = indicesType.getShape().raw();

    const auto batchDims = gatherND.getBatchDims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());

    SmallVector<int64_t> outShape(indicesShape.begin(), indicesShape.end() - 1);
    if (batchDims + lastIndices != inputRank) {
        outShape.append(inputShape.begin() + batchDims + lastIndices, inputShape.end());
    }

    const auto tensorAttr = createOutTensorAttrFromType(inType, outShape.size());
    if (mlir::failed(tensorAttr)) {
        return mlir::failure();
    }
    auto outType = mlir::cast<NDTypeInterface>(
            mlir::RankedTensorType::get(outShape, inType.getElementType(), tensorAttr.value()));
    if (auto boundedIndices = mlir::dyn_cast<Core::BoundedTensorType>(indicesType)) {
        auto indicesBounds = boundedIndices.getBounds();

        Bounds outBounds(indicesBounds.begin(), indicesBounds.end() - 1);
        if (batchDims + lastIndices != inputRank) {
            outBounds.append(inputShape.begin() + batchDims + lastIndices, inputShape.end());
        }

        outType = Core::BoundedTensorType::get(outType, outBounds);
    }

    inferredReturnTypes.emplace_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::GatherNDOp::verify() {
    const auto op = getOperation();
    const auto inType = mlir::cast<mlir::ShapedType>(getInput().getType());
    auto originalShapeOptional = getOriginalShape();
    vpux::Shape inputShape = originalShapeOptional.has_value()
                                     ? vpux::Shape(parseIntArrayAttr<int64_t>(originalShapeOptional.value()))
                                     : vpux::Shape(inType.getShape());
    const auto indicesShape = mlir::cast<vpux::NDTypeInterface>(getIndices().getType()).getShape().raw();
    const auto batchDims = getBatchDims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());
    const auto indicesRank = static_cast<int64_t>(indicesShape.size());

    if (batchDims >= inputRank) {
        return errorAt(op, "batch_dims {0} exceeds input rank {1}", batchDims, inputRank);
    }

    if (batchDims >= indicesRank) {
        return errorAt(op, "batch_dims {0} exceeds indices rank {1}", batchDims, indicesRank);
    }

    if (batchDims + lastIndices > inputRank) {
        return errorAt(op, "Slice index is out of bound");
    }

    for (size_t i = 0; i < static_cast<size_t>(batchDims); i++) {
        if (inputShape[Dim(i)] != indicesShape[i]) {
            return errorAt(op, "Batch dimensions of data and indices must be the same");
        }
    }

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherNDOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origIndicesShape = getShape(getIndices());
    const auto batchDims = getBatchDims();

    auto originalShapeOptional = getOriginalShape();
    Shape originalShapeAttrVal = originalShapeOptional.has_value()
                                         ? Shape(parseIntArrayAttr<int64_t>(getOriginalShape().value()))
                                         : Shape(origInputShape);

    return vpux::backInferGatherNDTile(outputTile, origInputShape, origIndicesShape, batchDims, originalShapeAttrVal,
                                       log);
}

void vpux::VPU::GatherNDOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    if (!getOriginalShape().has_value()) {
        return;
    }

    const auto batchDims = getBatchDims();
    const auto inTileShape = inputTiling.tiles[0].shape;

    // Input data with coord part cannot be tiled
    // Only other dimension needs update
    auto newShape = parseIntArrayAttr<int64_t>(getOriginalShape().value());
    for (auto idx = 0; idx < batchDims; idx++) {
        newShape[idx] = inTileShape[Dim(idx)];
    }
    newShape.back() = inTileShape.back();

    setOriginalShapeAttr(getIntArrayAttr(getContext(), newShape));
}

mlir::FailureOr<OutputTiling> vpux::VPU::GatherNDOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GatherND currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    const auto outputRank = outputShape.size();
    Shape nTilesOnDimforGatherND(outputRank, 1);
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    const auto tileDimRange = [isSupportedTileSize, &nTilesOnDimforGatherND, tilingMode, outputShape](
                                      const int64_t rangeBegin, const int64_t rangeEnd) -> void {
        auto tileDim = rangeBegin;
        while (!isSupportedTileSize(nTilesOnDimforGatherND, tilingMode)) {
            if (tileDim >= rangeEnd) {
                break;
            } else if (nTilesOnDimforGatherND[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
                ++tileDim;
            } else {
                ++nTilesOnDimforGatherND[Dim(tileDim)];
            }
        }
    };

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inputSize = inputType.getTotalAllocSize();

    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(getIndices().getType());
    const auto indicesSize = indicesType.getTotalAllocSize();
    const auto indicesRank = indicesType.getShape().size();

    const auto batchDims = getBatchDims();

    // The number of tiles on different dimensions impacts the total size of the data differently,
    // to reduce the number of kernel invocations, we first prioritize tiling on dimensions that reduce the most data:
    // 1) outputShape[:batchDims]               - output, input and indices sizes are reduced
    // 2) outputShape[batchDims:indicesRank-1]  - output and indices sizes are reduced
    // 3) outputShape[indicesRank-1:]           - output and input sizes are reduced

    tileDimRange(0, batchDims);
    if (inputSize > indicesSize) {
        tileDimRange(indicesRank - 1, outputRank);
        tileDimRange(batchDims, indicesRank);
    } else {
        tileDimRange(batchDims, indicesRank);
        tileDimRange(indicesRank - 1, outputRank);
    }

    VPUX_THROW_UNLESS(isSupportedTileSize(nTilesOnDimforGatherND, tilingMode), "Operation `GatherND` cannot be tiled");

    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGatherND);
    return fillDividedTiles(baseOp, nTilesOnDimforGatherND, outputShape);
}

//
// build
//

void vpux::VPU::GatherNDOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                  ::mlir::Value indices, ::mlir::IntegerAttr batch_dims) {
    build(builder, state, input, indices, batch_dims, /*original_shape=*/{}, /*multiClusterStrategy=*/nullptr);
}

void vpux::VPU::GatherNDOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                  ::mlir::Value indices, ::mlir::IntegerAttr batch_dims,
                                  ::mlir::ArrayAttr original_shape) {
    build(builder, state, input, indices, batch_dims, original_shape, /*multiClusterStrategy=*/nullptr);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GatherNDOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::GatherNDOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::GatherNDOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "GatherNDOp requires 2 input and 1 output, but the number of buffer is {0}",
                      buffers.size());

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

bool vpux::VPU::GatherNDOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GatherNDOp::supportCycleCostCalculation() {
    return false;
}
