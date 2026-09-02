//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {
auto calculateOutputShape(const llvm::ArrayRef<int64_t>& inputShape, const llvm::ArrayRef<int64_t>& indicesShape,
                          int64_t batchDims, int64_t axisVal, int64_t indicesRank) {
    SmallVector<int64_t> shape;
    int64_t outRank = inputShape.size() + indicesRank - 1 - batchDims;
    VPUX_THROW_UNLESS(outRank >= 0, "Calculated output rank expected to be non-negative, but got {0}", outRank);
    int64_t i = 0;

    for (; i < batchDims; i++) {
        VPUX_THROW_WHEN(inputShape[i] != indicesShape[i],
                        "The first dimensions in Input and Indices shapes are expected to be equal");
        shape.push_back(inputShape[i]);
    }
    for (; i < axisVal; i++) {
        shape.push_back(inputShape[i]);
    }
    for (; i < axisVal + indicesRank - batchDims; i++) {
        shape.push_back(indicesShape[batchDims - axisVal + i]);
    }
    for (; i < outRank; i++) {
        shape.push_back(inputShape[batchDims + 1 - indicesRank + i]);
    }
    // To avoid shape size 0 error, set the shape 1.
    if (shape.empty()) {
        shape.push_back(1);
    }
    return shape;
};

}  // namespace

mlir::LogicalResult vpux::VPU::GatherOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherOpAdaptor gather(operands, attrs, prop);
    if (mlir::failed(gather.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<mlir::ShapedType>(gather.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto indicesType = mlir::cast<mlir::ShapedType>(gather.getIndices().getType());
    const auto indicesShape = indicesType.getShape();

    const auto axis = gather.getAxisValue();
    auto batch = gather.getBatchDims();
    auto rank = gather.getIndicesRank().value_or(indicesShape.size());

    auto outShape = calculateOutputShape(inputShape, indicesShape, batch, axis, rank);

    Bounds bounds;
    if (vpux::details::isDynamicDimValues(outShape)) {
        auto boundedInputTensor = mlir::dyn_cast<Core::BoundedTensorType>(inputType);
        auto boundedIndicesTensor = mlir::dyn_cast<Core::BoundedTensorType>(indicesType);
        auto actualDataShape = boundedInputTensor ? boundedInputTensor.getBounds().raw() : inputShape;
        auto actualIndicesShape = boundedIndicesTensor ? boundedIndicesTensor.getBounds().raw() : indicesShape;
        bounds = Bounds(calculateOutputShape(actualDataShape, actualIndicesShape, batch, axis, rank));
    }

    const auto outDesc =
            vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), /*memSpace=*/nullptr, bounds);

    inferredReturnTypes.push_back(mlir::RankedTensorType::get(outShape, inputType.getElementType(), outDesc));

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origIndicesShape = getShape(getIndices());
    bool hasAxisTensor = false;

    const int64_t axisValue = getAxisValue();
    int64_t batchDims = 0;
    if (getBatchDimsAttr() != nullptr) {
        batchDims = mlir::dyn_cast_or_null<mlir::IntegerAttr>(getBatchDimsAttr()).getValue().getSExtValue();
    }

    const auto indicesRank = getIndicesRank().value_or(origIndicesShape.size());

    return vpux::backInferGatherTile(outputTile, origInputShape, origIndicesShape, axisValue, batchDims, hasAxisTensor,
                                     indicesRank, log);
}

void vpux::VPU::GatherOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::GatherOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for Gather currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());
    const int64_t axisValue = getAxisValue();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();

    int64_t batchDims = 0;
    if (getBatchDimsAttr() != nullptr) {
        batchDims = mlir::dyn_cast_or_null<mlir::IntegerAttr>(getBatchDimsAttr()).getValue().getSExtValue();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inputSize = inputType.getCompactAllocSize();
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(getIndices().getType());

    const auto indicesSize = indicesType.getCompactAllocSize();
    const auto indicesRank = getIndicesRank().value_or(indicesType.getRank());
    const auto outputRank = static_cast<int64_t>(outputShape.size());

    SmallVector<int64_t> batchDimsRange, dataBeforeAxisRange, indicesRange, dataAfterAxisRange;
    for (int64_t i = 0; i < outputRank; ++i) {
        if (i < batchDims) {
            batchDimsRange.push_back(i);
        } else if (batchDims <= i && i < axisValue) {
            dataBeforeAxisRange.push_back(i);
        } else if (axisValue <= i && i < axisValue + indicesRank - batchDims) {
            indicesRange.push_back(i);
        } else {
            dataAfterAxisRange.push_back(i);
        }
    }
    SmallVector<int64_t> tileDimOrder;
    tileDimOrder.insert(tileDimOrder.end(), batchDimsRange.begin(), batchDimsRange.end());
    if (inputSize > indicesSize) {
        // TileDimOrder: {batchDimsRange, dataBeforeAxisRange, dataAfterAxisRange, indicesRange}.
        tileDimOrder.insert(tileDimOrder.end(), dataBeforeAxisRange.begin(), dataBeforeAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataAfterAxisRange.begin(), dataAfterAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), indicesRange.begin(), indicesRange.end());
    } else {
        // TileDimOrder: {batchDimsRange, indicesRange, dataBeforeAxisRange, dataAfterAxisRange}.
        tileDimOrder.insert(tileDimOrder.end(), indicesRange.begin(), indicesRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataBeforeAxisRange.begin(), dataBeforeAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataAfterAxisRange.begin(), dataAfterAxisRange.end());
    }

    auto nTilesOnDimforGather = getSupportedNTilesOnDimforGather(tileDimOrder, baseOp, tilingMode, log);

    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGather);
    return fillDividedTiles(baseOp, nTilesOnDimforGather, outputShape);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GatherOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto outputShape = getShape(getOutput());

    if (outputShape.isDynamic()) {
        return false;
    }

    const auto is4DInputs = std::all_of(getOperands().begin(), getOperands().end(), [](auto input) {
        return getShape(input).size() == 4;
    });
    const auto is4DOutput = (getShape(getOutput()).size() == 4);
    if (!is4DInputs || !is4DOutput) {
        return false;
    }

    auto ddrAccessOp = mlir::dyn_cast<VPU::DDRAccessOpInterface>(getOperation());
    if (ddrAccessOp != nullptr && ddrAccessOp.isDDRAccessNecessaryOrBeneficial(Logger::global())) {
        return false;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::GatherOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

void vpux::VPU::GatherOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                mlir::Value indices, mlir::IntegerAttr axis_value, mlir::IntegerAttr batch_dims,
                                mlir::IntegerAttr indices_rank) {
    build(builder, state, input, indices, axis_value, batch_dims, indices_rank, {});
}

//
// SWOpInterface
//

bool vpux::VPU::GatherOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "GatherOp requires 2 inputs and 1 outputs, but the number of buffer is {0}",
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

bool vpux::VPU::GatherOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GatherOp::supportCycleCostCalculation() {
    return false;
}
