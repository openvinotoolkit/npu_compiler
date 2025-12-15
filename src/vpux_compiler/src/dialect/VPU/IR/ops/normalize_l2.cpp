//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NormalizeL2Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NormalizeL2OpAdaptor normalizeL2(operands, attrs, prop);
    if (mlir::failed(normalizeL2.verify(loc))) {
        return mlir::failure();
    }
    const auto axes = parseIntArrayAttr<int64_t>(normalizeL2.getAxesValueAttr());
    if (axes.empty()) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(normalizeL2.getData().getType());
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NormalizeL2Op::verify() {
    const auto inRank = mlir::cast<vpux::NDTypeInterface>(getData().getType()).getRank();
    auto axesVec = parseIntArrayAttr<int64_t>(getAxesValueAttr());

    for (auto& axis : axesVec) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(*this, "Axes values should be unique");
    }

    return mlir::success();
}

bool vpux::VPU::NormalizeL2Op::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getData().getType());
    const auto inShape = inputType.getShape();

    auto axesVec = parseIntArrayAttr<int64_t>(getAxesValueAttr());
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

vpux::VPU::DistributionInfo vpux::VPU::NormalizeL2Op::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

bool vpux::VPU::NormalizeL2Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "NormalizeL2Op requires 1 inputs and 1 outputs, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::NormalizeL2Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::NormalizeL2Op::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::NormalizeL2Op::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return InputTiling{{outputTile}};
}

void vpux::VPU::NormalizeL2Op::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::NormalizeL2Op::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}
