//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LRNOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LRNOpAdaptor lrn(operands, attrs, prop);
    if (mlir::failed(lrn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lrn.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::LRNOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inShape = inputType.getShape();

    auto axesVec = parseIntArrayAttr<int64_t>(getAxesAttr());
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    auto isCompatibleStrategy{[&](auto strategyToCheck, auto dimensionToCheck) {
        return strategy == strategyToCheck && inShape[dimensionToCheck] > 1 &&
               std::find(axesVec.begin(), axesVec.end(), dimensionToCheck.ind()) == axesVec.end();
    }};

    if (isCompatibleStrategy(VPU::MultiClusterStrategy::SplitOverHeight, Dims4D::Act::H)) {
        return true;
    }

    if (isCompatibleStrategy(VPU::MultiClusterStrategy::SplitOverKernel, Dims4D::Act::C)) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::LRNOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::LRNOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "LRNOp requires 1 input and 1 output, but the number of buffers is {0}",
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

bool vpux::VPU::LRNOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::LRNOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::LRNOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return InputTiling{{outputTile}};
}

void vpux::VPU::LRNOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::LRNOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}

SmallVector<int64_t> vpux::VPU::LRNOp::getMaxNumTiles() {
    const auto axes = parseIntArrayAttr<int64_t>(getAxesAttr());
    auto maxNumTilesPerDim = getMaxNumTilesWithAxesExclusion(getOperation(), axes);
    return vpux::getMaxNumTiles(getOperation(), false, false, maxNumTilesPerDim);
}
