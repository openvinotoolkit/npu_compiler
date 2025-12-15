//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReverseOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReverseOpAdaptor reverse(operands, attrs, prop);
    if (mlir::failed(reverse.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reverse.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ReverseOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return TilingInfo(outputTile);
}

void vpux::VPU::ReverseOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::ReverseOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();

    SmallVector<int64_t> axes = parseIntArrayAttr<int64_t>(getAxisValueAttr());
    SmallVector<int64_t> maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);

    return getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}

// Return a list with all dims that can be tiled, meaning the dims that are not in 'axes' list.
DimArr vpux::VPU::ReverseOp::getTileableDims() {
    const auto rank = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getRank();
    VPUX_THROW_UNLESS(rank == 4, "Function valid only for 4D shape, got {0}D", rank);

    DimArr dims;
    const auto axes = parseIntArrayAttr<int64_t>(getAxisValueAttr());

    for (int64_t dimIdx = 0; dimIdx < rank; dimIdx++) {
        if (std::find(axes.begin(), axes.end(), dimIdx) == axes.end()) {
            dims.push_back(Dim(dimIdx));
        }
    }

    return dims;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::ReverseOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto tileableDims = this->getTileableDims();

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::N) != tileableDims.end();
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::C) != tileableDims.end();
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::H) != tileableDims.end();
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return std::find(tileableDims.begin(), tileableDims.end(), Dims4D::Act::W) != tileableDims.end();
    }

    return false;
}

bool VPU::ReverseOp::isOperationSplitOverBatchCompatible(vpux::ShapeRef outputShape) {
    return VPU::isOperationSplitOverBatchCompatible(getOperation(), outputShape);
}

vpux::VPU::DistributionInfo vpux::VPU::ReverseOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::ReverseOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "ReverseOp requires 1 input and 1 output, but the number of buffers is {0} ",
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

bool vpux::VPU::ReverseOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::ReverseOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::ReverseOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                 ::mlir::ArrayAttr axis_value, vpux::IE::ReverseModeAttr mode) {
    build(builder, state, input, axis_value, mode, {});
}
