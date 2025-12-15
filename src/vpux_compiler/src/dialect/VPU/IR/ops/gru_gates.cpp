//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUGatesOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUGatesOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInitialHiddenState().getType());

    inferredReturnTypes.push_back(inType);  // outputHiddenState

    return mlir::success();
}

void vpux::VPU::GRUGatesOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  ::mlir::Value inputData, ::mlir::Value initialHiddenState, ::mlir::Value hiddenData,
                                  ::mlir::Value biases) {
    build(odsBuilder, odsState, inputData, initialHiddenState, hiddenData, biases, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::GRUGatesOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    SmallVector<TileInfo> inputTiles;

    auto curTile = outputTile;
    auto curShape = getShape(getInputData());
    curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
    inputTiles.push_back(curTile);

    curTile = outputTile;
    curShape = getShape(getInitialHiddenState());
    curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
    inputTiles.push_back(curTile);

    curTile = outputTile;
    curShape = getShape(getHiddenData());
    curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
    inputTiles.push_back(curTile);

    curShape = getShape(getBiases());
    TileInfo bTile(curShape);
    inputTiles.push_back(bTile);

    return TilingInfo{inputTiles};
}

void vpux::VPU::GRUGatesOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::GRUGatesOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    SmallVector<int64_t> maxNumTiles;
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getResult().getType());
    const auto outputRank = outputType.getShape().size();
    SmallVector<int64_t> axes{checked_cast<int64_t>(outputRank - 1)};
    maxNumTiles = getMaxNumTilesWithAxesExclusion(this->getOperation(), axes);

    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log, maxNumTiles);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GRUGatesOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::GRUGatesOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::GRUGatesOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS((buffers.size() >= 4) && (buffers.size() <= 5),
                      "GRUGatesOp requires 4 or 3 inputs and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::GRUGatesOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GRUGatesOp::supportCycleCostCalculation() {
    return false;
}
