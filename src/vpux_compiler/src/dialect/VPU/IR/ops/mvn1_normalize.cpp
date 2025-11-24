//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN1NormalizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN1NormalizeOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::MVN1NormalizeOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                vpux::Logger /*log*/) {
    TileInfo meanVarTile(getShape(getMeanVar()));
    return TilingInfo{{outputTile, std::move(meanVarTile)}};
}

void vpux::VPU::MVN1NormalizeOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::MVN1NormalizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MVN1NormalizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto order = DimsOrder::fromValue(getOutput());
    // Note: disable SOK for NHWC as may lead to suboptimal Shave configuration (i.e. C not multiple of 32)
    //       meaning slow SCALAR Shave operation
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           ((strategy == VPU::MultiClusterStrategy::SplitOverKernel) && (order == DimsOrder::NCHW)) ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::MVN1NormalizeOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::MVN1NormalizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "MVN1NormalizeOp requires 2 inputs and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::MVN1NormalizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MVN1NormalizeOp::supportCycleCostCalculation() {
    return false;
}
//
// build
//

void vpux::VPU::MVN1NormalizeOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                       ::mlir::Value meanVar, ::mlir::BoolAttr across_channels,
                                       ::mlir::BoolAttr normalize_variance) {
    build(builder, state, input.getType(), input, meanVar, across_channels, normalize_variance, {}, {});
}

void vpux::VPU::MVN1NormalizeOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                       ::mlir::Value meanVar, ::mlir::BoolAttr across_channels,
                                       ::mlir::BoolAttr normalize_variance, ::mlir::BoolAttr high_precision_normalize) {
    build(builder, state, input.getType(), input, meanVar, across_channels, normalize_variance,
          high_precision_normalize, {});
}
