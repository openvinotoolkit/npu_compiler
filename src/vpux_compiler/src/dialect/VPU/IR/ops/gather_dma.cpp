//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherDMAOp::inferReturnTypes(mlir::MLIRContext*, std::optional<mlir::Location>,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPU::GatherDMAOpAdaptor gatherDMAOp(operands, attrs, prop);
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(gatherDMAOp.getIndices().getType());
    const auto indicesShape = indicesType.getShape();

    const auto axis = gatherDMAOp.getAxisValue();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(gatherDMAOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    auto outputShape = inputShape.toValues();
    outputShape[Dim(axis)] = indicesShape[Dim(axis)];

    auto outType = mlir::RankedTensorType::get(to_small_vector(outputShape), inputType.getElementType(),
                                               createTensorAttrFromType(inputType));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherDMAOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origIndicesShape = getShape(getIndices());
    bool hasAxisTensor = false;

    const int64_t axisValue = getAxisValue();
    return vpux::backInferGatherDMATile(outputTile, origInputShape, origIndicesShape, axisValue, hasAxisTensor, log);
}

void vpux::VPU::GatherDMAOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::GatherDMAOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for Gather currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    const auto axisValue = getAxisValue();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inputSize = inputType.getCompactAllocSize();
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(getIndices().getType());
    const auto indicesSize = indicesType.getCompactAllocSize();
    const auto outputRank = static_cast<int64_t>(outputShape.size());

    auto isSub8BitElementType = [](vpux::NDTypeInterface outputType) -> bool {
        const auto outputElemType = outputType.getElementType();
        auto getStorageBitWidth = [](mlir::Type elemType) -> std::optional<uint32_t> {
            return mlir::TypeSwitch<mlir::Type, std::optional<uint32_t>>(elemType)
                    .Case<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType,
                          vpux::type::QuantileType>([](auto quantType) {
                        if (auto quantileStorage =
                                    mlir::dyn_cast<vpux::type::QuantileType>(quantType.getStorageType())) {
                            return mlir::cast<mlir::IntegerType>(quantileStorage.getStorageType()).getWidth();
                        }
                        return mlir::cast<mlir::IntegerType>(quantType.getStorageType()).getWidth();
                    })
                    .Default([](mlir::Type type) -> std::optional<uint32_t> {
                        if (type.isIntOrFloat()) {
                            return type.getIntOrFloatBitWidth();
                        }
                        return std::nullopt;
                    });
        };

        if (auto bitWidth = getStorageBitWidth(outputElemType)) {
            return *bitWidth < 8;
        }

        return false;
    };

    SmallVector<int64_t> dataBeforeAxisRange, indicesRange, dataAfterAxisRange;
    for (int64_t i = 0; i < outputRank; ++i) {
        if (i < axisValue) {
            dataBeforeAxisRange.push_back(i);
        } else if (axisValue == i) {
            indicesRange.push_back(i);
        } else {
            dataAfterAxisRange.push_back(i);
        }
    }

    SmallVector<int64_t> tileDimOrder;
    // Prefer tiling along indices or higher dimensions when input element type is less than 8 bits, to
    // avoid unsupported DMA like transferring 12 bits. This is not a 100% workable solution, but in most cases,
    // if the data is 4-bit, the size of the innermost dimension is even.
    // Therefore, tiling along indices or higher dimensions is preferred.
    if (inputSize > indicesSize && !isSub8BitElementType(outputType)) {
        // TileDimOrder: {dataBeforeAxisRange, dataAfterAxisRange, indicesRange}.
        tileDimOrder.insert(tileDimOrder.end(), dataBeforeAxisRange.begin(), dataBeforeAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataAfterAxisRange.begin(), dataAfterAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), indicesRange.begin(), indicesRange.end());
    } else {
        // TileDimOrder: {indicesRange, dataBeforeAxisRange, dataAfterAxisRange}.
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

bool vpux::VPU::GatherDMAOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto indicesShape = getShape(getIndices());
    const auto outputShape = getShape(getOutput());
    if (indicesShape.size() != 4) {
        return false;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return indicesShape[Dims4D::Act::N] >= checked_cast<int64_t>(numTiles);
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return indicesShape[Dims4D::Act::H] == 1 && outputShape[Dims4D::Act::H] >= checked_cast<int64_t>(numTiles);
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return indicesShape[Dims4D::Act::W] == 1 && outputShape[Dims4D::Act::W] >= checked_cast<int64_t>(numTiles);
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::GatherDMAOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    VPUX_THROW_UNLESS(distributionMode != VPU::DistributionMode::OVERLAPPED,
                      "Overlapped distribution mode is not supported for GatherDMAOp");

    return getNonOverlappedDistributedNative(shape, distributionMode, numTiles, numClusters, alignment,
                                             uniformDistributedSegments);
}

vpux::NDTypeInterface vpux::VPU::GatherDMAOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                             bool hasExplicitDistributedAttr,
                                                                             SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<GatherDMAOp>(getOperation());

    if (operand.get() == origOp.getInput()) {
        return mlir::dyn_cast<NDTypeInterface>(origOp.getInput().getType());
    }

    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    if (operand.get() == origOp.getIndices()) {
        if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
            auto* ctx = clusteredOp->getContext();
            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(origOp.getOutput()), strategy);
            auto activationTensorNumTiles =
                    getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters, strategy));

            return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::SEGMENTED,
                                               activationTensorNumTiles, {}, strategy, hasExplicitDistributedAttr,
                                               siblingsAnalysis);
        }

        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }

    VPUX_THROW("Failed to compute distributed type for op operand {0}", clusteredOp);
    return nullptr;
}

bool vpux::VPU::GatherDMAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "GatherDMAOp has 2 inputs and 1 output, and we only need to fit indices and output in CMX, but"
                      "the number of buffer is { 0 } ",
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

bool vpux::VPU::GatherDMAOp::fitIntoCMX(vpux::NDTypeInterface indices, vpux::NDTypeInterface output, Byte reservedMem) {
    SmallVector<Byte> buffers = {indices.getTotalAllocSize(), output.getTotalAllocSize()};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::GatherDMAOp::fitIntoCMX(vpux::NDTypeInterface indices, vpux::NDTypeInterface output) {
    return fitIntoCMX(indices, output, Byte(0));
}
