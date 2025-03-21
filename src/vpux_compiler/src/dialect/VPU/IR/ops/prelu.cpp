//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::VPU::PReluOp::verify() {
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto slopeType = getNegativeSlope().getType().cast<vpux::NDTypeInterface>();
    const auto slopeShape = slopeType.getShape().raw();

    if (slopeShape.size() != 4 || inShape.size() != 4) {
        return errorAt(*this, "Tiling restrictions require slope to have a 4D shape, got size of {0}",
                       slopeShape.size());
    }

    if (inShape[Dims4D::Act::C.ind()] != slopeShape[Dims4D::Act::C.ind()] ||
        slopeShape[Dims4D::Act::C.ind()] != slopeType.getShape().totalSize()) {
        return errorAt(*this,
                       "4D slope shape should have the last dim equal to the channel input dim, as broadcast with "
                       "numpy values is not supported: {0} != {1}",
                       inShape[Dims4D::Act::C.ind()], slopeShape[Dims4D::Act::C.ind()]);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::PReluOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PReluOpAdaptor prelu(operands, attrs, prop);
    if (mlir::failed(prelu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = prelu.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::PReluOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    TileInfo inputTile(getShape(getInput()));
    TileInfo slopeTile(getShape(getNegativeSlope()));
    inputTile = outputTile;
    if (outputTile.shape[Dims4D::Act::C] != slopeTile.shape[Dims4D::Act::C]) {
        // Tile slope by channel, align the offsets and axis to outputTile
        slopeTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        slopeTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
        slopeTile.axis[Dims4D::Act::C] = outputTile.axis[Dims4D::Act::C];
    }

    return TilingInfo{{std::move(inputTile), std::move(slopeTile)}};
}

void vpux::VPU::PReluOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing here
}

mlir::FailureOr<OutputTiling> vpux::VPU::PReluOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// build
//

void vpux::VPU::PReluOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input1,
                               ::mlir::Value input2) {
    build(odsBuilder, odsState, input1, input2, nullptr);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::PReluOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::PReluOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::PReluOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "PReluOp requires 2 input and 1 output, but the number of buffer is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::PReluOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::PReluOp::supportCycleCostCalculation() {
    return false;
}
