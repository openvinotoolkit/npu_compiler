//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::VPU::PReluOp::verify() {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inShape = inType.getShape().raw();
    const auto slopeType = mlir::cast<vpux::NDTypeInterface>(getNegativeSlope().getType());
    const auto slopeShape = slopeType.getShape().raw();

    if (slopeShape.size() != 4 || inShape.size() != 4) {
        return errorAt(
                *this,
                "Tiling restrictions require both input and slope to be 4D. Got input rank {0} and slope rank {1}",
                inShape.size(), slopeShape.size());
    }

    // Accept any slope shape that numpy-broadcasts to the input: each slope dim must be
    // either 1 or exactly equal to the corresponding input dim.
    for (size_t i = 0; i < slopeShape.size(); ++i) {
        if (slopeShape[i] != 1 && slopeShape[i] != inShape[i]) {
            return errorAt(*this,
                           "Unsupported slope shape for PRelu: slope must numpy-broadcast to input "
                           "(each dim must be 1 or match the input). Got input {0} and slope {1}",
                           inShape, slopeShape);
        }
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

    // Tile slope along every dim where it is non-broadcast (size > 1).
    // Dims where slope size is 1 stay at size 1 / offset 0 (broadcast dims).
    const auto slopeShape = getShape(getNegativeSlope());
    for (auto dim : {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}) {
        if (slopeShape[dim] > 1 && outputTile.shape[dim] != slopeTile.shape[dim]) {
            slopeTile.shape[dim] = outputTile.shape[dim];
            slopeTile.offsets[dim] = outputTile.offsets[dim];
            slopeTile.axis[dim] = outputTile.axis[dim];
        }
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
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
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

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::PReluOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::PReluOp::supportCycleCostCalculation() {
    return false;
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::PReluOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                          mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}
