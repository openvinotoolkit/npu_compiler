//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMGatesOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LSTMGatesOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialCellState().getType());

    inferredReturnTypes.push_back(inType);  // outputHiddenState
    inferredReturnTypes.push_back(inType);  // outputCellState

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::LSTMGatesOp::verify() {
    const auto gatesInputShape = getShape(getGatesInput()).raw();
    const auto initialCellStateShape = getShape(getInitialCellState()).raw();
    VPUX_THROW_UNLESS(initialCellStateShape.size() == 2 || initialCellStateShape.size() == 4,
                      "LSTMGatesOp requires the input shape size is 2 or 4, but here is {0}",
                      initialCellStateShape.size());

    const auto batchSize = gatesInputShape.size() == 2 ? initialCellStateShape[0] : initialCellStateShape[2];
    const auto hiddenSize = gatesInputShape.size() == 2 ? initialCellStateShape[1] : initialCellStateShape[3];

    if (gatesInputShape != ArrayRef<int64_t>({batchSize, 4 * hiddenSize}) &&
        gatesInputShape != ArrayRef<int64_t>({1, 1, batchSize, 4 * hiddenSize})) {
        return errorAt(*this,
                       "Incompatible input shapes. Expected gatesInput shape: [batch_size, 4*hidden_size], "
                       "initialCellState shape: [batch_size, hidden_size]. Got gatesInput shape: {0}, initialCellState "
                       "shape: {1}",
                       gatesInputShape, initialCellStateShape);
    }

    return mlir::success();
}

void vpux::VPU::LSTMGatesOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   ::mlir::Value gatesInput, ::mlir::Value initialCellState) {
    build(odsBuilder, odsState, gatesInput, initialCellState, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::LSTMGatesOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : getInputs()) {
        const auto curShape = getShape(origInput);

        auto curTile = outputTile;
        curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];

        inputTiles.push_back(curTile);
    }

    return TilingInfo{inputTiles};
}

OutputTiling vpux::VPU::LSTMGatesOp::getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return OutputTiling{outputTile, outputTile};
}

void vpux::VPU::LSTMGatesOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::LSTMGatesOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, log);
}

SmallVector<int64_t> vpux::VPU::LSTMGatesOp::getMaxNumTiles() {
    auto op = getOperation();
    SmallVector<int64_t> maxNumTiles;
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getResult(0).getType());
    const auto outputRank = outputType.getShape().size();
    SmallVector<int64_t> axes{checked_cast<int64_t>(outputRank - 1)};
    maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);

    return vpux::getMaxNumTiles(op, false, false, maxNumTiles);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::LSTMGatesOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::LSTMGatesOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::LSTMGatesOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 4, "LSTMGatesOp requires 2 input and 2 output, but the number of buffer is {0}",
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

bool vpux::VPU::LSTMGatesOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::LSTMGatesOp::supportCycleCostCalculation() {
    return false;
}
