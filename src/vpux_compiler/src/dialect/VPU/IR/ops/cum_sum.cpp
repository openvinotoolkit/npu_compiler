//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::CumSumOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::CumSumOpAdaptor cumSum(operands, attrs, prop);
    if (mlir::failed(cumSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cumSum.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::CumSumOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return TilingInfo(outputTile);
}

void vpux::VPU::CumSumOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::CumSumOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return getSWLayerTilingStrategy(getOperation(), tilingMode, log);
}

SmallVector<int64_t> vpux::VPU::CumSumOp::getMaxNumTiles() {
    const auto op = getOperation();
    int64_t axisValue = 0;

    if (getAxisValueAttr() != nullptr) {
        axisValue = mlir::cast<mlir::IntegerAttr>(getAxisValueAttr()).getValue().getSExtValue();
    }

    SmallVector<int64_t> axes{axisValue};
    SmallVector<int64_t> maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);

    return vpux::getMaxNumTiles(op, false, false, maxNumTiles);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::CumSumOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    int64_t axisValue = 0;

    if (getAxisValueAttr() != nullptr) {
        axisValue = mlir::cast<mlir::IntegerAttr>(getAxisValueAttr()).getValue().getSExtValue();
    }

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return true;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return Dim(axisValue) != Dims4D::Act::H;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return Dim(axisValue) != Dims4D::Act::C;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return Dim(axisValue) != Dims4D::Act::W;
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return Dim(axisValue) != Dims4D::Act::N;
    default:
        return false;
    }
}

vpux::VPU::DistributionInfo vpux::VPU::CumSumOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::CumSumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "CumSumOp requires 1 input and 1 output, but the number of buffers is {0} ",
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

bool vpux::VPU::CumSumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::CumSumOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::CumSumOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                ::mlir::IntegerAttr axis_value, ::mlir::UnitAttr exclusive, ::mlir::UnitAttr reverse) {
    build(builder, state, input, axis_value, exclusive, reverse, {});
}
