//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RoPEOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RoPEOpAdaptor rope(operands, attrs, prop);
    if (mlir::failed(rope.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(rope.getInput().getType());
    inferredReturnTypes.push_back(inType);
    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::RoPEOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

void vpux::VPU::RoPEOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
                              ::mlir::Value input_cos, ::mlir::Value input_sin, ::mlir::UnitAttr is_interleaved) {
    build(odsBuilder, odsState, input, input_cos, input_sin, is_interleaved, {});
}

vpux::VPU::DistributionInfo vpux::VPU::RoPEOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::RoPEOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 4, "RoPEOp requires 3 inputs and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::RoPEOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::RoPEOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::RoPEOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    TileInfo cosTile(getShape(getInputCos()));
    TileInfo sinTile(getShape(getInputSin()));
    auto inTile = outputTile;
    // The Cosine and Sine operations offer flexibility in channel configuration:
    // - Channels: You can choose to match the input's number of channels or set it to 1
    // - Height: Unlike channels, the height for Cosine and Sine operations can differ from the input height
    if (cosTile.shape[Dim(Dims4D::Act::C)] > 1) {
        if (cosTile.shape[Dim(Dims4D::Act::H)] != inTile.shape[Dim(Dims4D::Act::H)]) {
            transferTilingInfo(sinTile, inTile, {Dim(Dims4D::Act::C)});
            transferTilingInfo(cosTile, inTile, {Dim(Dims4D::Act::C)});
        } else {
            cosTile = inTile;
            sinTile = inTile;
        }
    } else {
        transferTilingInfo(sinTile, inTile, {Dim(Dims4D::Act::H), Dim(Dims4D::Act::N)});
        transferTilingInfo(cosTile, inTile, {Dim(Dims4D::Act::H), Dim(Dims4D::Act::N)});
    }

    return TilingInfo{{std::move(inTile), std::move(cosTile), std::move(sinTile)}};
}

void vpux::VPU::RoPEOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::RoPEOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outputRank = outputType.getShape().size();
    SmallVector<int64_t> maxNumTiles;
    maxNumTiles = getMaxNumTilesWithAxesExclusion(op, /*axis:*/ {checked_cast<int64_t>(outputRank - 1)});
    return vpux::getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}
