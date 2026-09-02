//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ScatterElementsUpdateOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ScatterElementsUpdateOpAdaptor scatter(operands, attrs, prop);
    if (mlir::failed(scatter.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatter.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ScatterElementsUpdateOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                        vpux::Logger) {
    const auto origIndicesShape = getShape(getIndices());
    const auto origUpdateShape = getShape(getUpdates());
    const auto axis = getAxis();

    TileInfo indicesTile(outputTile);
    TileInfo updateTile(outputTile);
    indicesTile.shape[Dim(axis)] = origIndicesShape[Dim(axis)];
    updateTile.shape[Dim(axis)] = origUpdateShape[Dim(axis)];

    return InputTiling{{outputTile, std::move(indicesTile), std::move(updateTile)}};
}

void vpux::VPU::ScatterElementsUpdateOp::adjustAttrs(const TilingInfo& /*inputTiling*/,
                                                     const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::ScatterElementsUpdateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    const auto axis = checked_cast<size_t>(getAxis());
    const auto rank = outputShape.size();

    Shape nTilesOnDim(outputShape.size(), 1);

    SmallVector<Dim> tileDimOrder;
    for (size_t i = 0; i < rank; ++i) {
        if (i != axis) {
            tileDimOrder.push_back(Dim(i));
        }
    }
    if (tileDimOrder.empty()) {
        return mlir::failure();
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    while (tileDimIter < tileDimOrder.end() && !isSupportedTileSize(nTilesOnDim, tilingMode)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            dimToTile = *(++tileDimIter);
        } else {
            ++nTilesOnDim[Dim(dimToTile)];
        }
    }

    if (isSupportedTileSize(nTilesOnDim, tilingMode)) {
        return fillDividedTiles(op, nTilesOnDim, outputShape);
    }

    return mlir::failure();
}

bool vpux::VPU::ScatterElementsUpdateOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inShape = getShape(getInput());
    const auto indicesShape = getShape(getIndices());
    const auto updatesShape = getShape(getUpdates());
    const auto axis = getAxis();

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    // A cluster split is bit-exact only when the split dim is not the scatter axis and is fully
    // addressed by indices and updates. axis != dim covers the non-scatter requirement (axis < dim was
    // sufficient but rejected the scatter-innermost layout); indicesShape == inShape and
    // updatesShape == inShape on the dim guarantee full addressing so no passthrough slice is
    // re-scattered. Both operands are checked explicitly because the op definitions do not enforce
    // updatesShape == indicesShape.
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && axis != Dims4D::Act::H.ind() &&
        inShape[Dims4D::Act::H] > 1 && indicesShape[Dims4D::Act::H] == inShape[Dims4D::Act::H] &&
        updatesShape[Dims4D::Act::H] == inShape[Dims4D::Act::H]) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && axis != Dims4D::Act::C.ind() &&
        inShape[Dims4D::Act::C] > 1 && indicesShape[Dims4D::Act::C] == inShape[Dims4D::Act::C] &&
        updatesShape[Dims4D::Act::C] == inShape[Dims4D::Act::C]) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth && axis != Dims4D::Act::W.ind() &&
        inShape[Dims4D::Act::W] > 1 && indicesShape[Dims4D::Act::W] == inShape[Dims4D::Act::W] &&
        updatesShape[Dims4D::Act::W] == inShape[Dims4D::Act::W]) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::ScatterElementsUpdateOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::ScatterElementsUpdateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
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

bool vpux::VPU::ScatterElementsUpdateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::ScatterElementsUpdateOp::supportCycleCostCalculation() {
    return false;
}

void vpux::VPU::ScatterElementsUpdateOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state,
                                               ::mlir::Value input, ::mlir::Value indices, ::mlir::Value updates,
                                               ::mlir::IntegerAttr axis,
                                               vpux::IE::ScatterElementsUpdateReductionTypeAttr reduction,
                                               ::mlir::BoolAttr use_init_val) {
    build(builder, state, input.getType(), input, indices, updates, axis, reduction, use_init_val, {});
}
