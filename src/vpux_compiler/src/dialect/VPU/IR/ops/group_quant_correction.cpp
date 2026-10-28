//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <utility>

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::GroupQuantCorrectionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupQuantCorrectionOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto actOutType = mlir::cast<vpux::NDTypeInterface>(op.getActOut().getType());
    inferredReturnTypes.push_back(actOutType);

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::GroupQuantCorrectionOp::verify() {
    const auto actOutShape = mlir::cast<vpux::NDTypeInterface>(getActOut().getType()).getShape();
    if (actOutShape[Dims4D::Act::H] != 1 || actOutShape[Dims4D::Act::W] != 1) {
        return errorAt(*this, "act_out must have shape [G, C, 1, 1].");
    }

    const auto actInShape = mlir::cast<vpux::NDTypeInterface>(getActIn().getType()).getShape();
    if (actInShape[Dims4D::Act::N] != actOutShape[Dims4D::Act::N] || actInShape[Dims4D::Act::C] != 1 ||
        actInShape[Dims4D::Act::H] != 1) {
        return errorAt(*this, "act_in must have shape [G, 1, 1, GS] with G matching act_out.");
    }

    const auto scalesShape = mlir::cast<vpux::NDTypeInterface>(getScales().getType()).getShape();
    if (scalesShape != actOutShape) {
        return errorAt(*this, "scales must have shape [G, C, 1, 1] matching act_out.");
    }

    const auto zpShape = mlir::cast<vpux::NDTypeInterface>(getZp().getType()).getShape();
    if (zpShape != actOutShape) {
        return errorAt(*this, "zp must have shape [G, C, 1, 1] matching act_out.");
    }

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GroupQuantCorrectionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverBatch ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering;
}

bool vpux::VPU::GroupQuantCorrectionOp::isOperationSplitOverBatchCompatible(ShapeRef outputShape) {
    if (outputShape == ShapeRef()) {
        outputShape = getBoundedShape(getResult());
    }
    return outputShape[Dims4D::Act::N] > 1;
}

bool vpux::VPU::GroupQuantCorrectionOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef, ShapeRef) {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(getResult());
    }
    auto numOfCluster = config::getNumOfTiles(*this);
    auto outputChannels = outputShape[Dims4D::Act::C];
    return outputChannels >= NCEInvariant::VPU_CHANNEL_ALIGNMENT * numOfCluster &&
           outputChannels % NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0;
}

vpux::VPU::DistributionInfo vpux::VPU::GroupQuantCorrectionOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::GroupQuantCorrectionOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    // 4 inputs (conv_output, input, scales, zero_points) + 1 output
    VPUX_THROW_UNLESS(buffers.size() == 5, "GroupQuantCorrectionOp requires 4 inputs and 1 output, got {0} buffers",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto& buf) {
        return buf.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::GroupQuantCorrectionOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GroupQuantCorrectionOp::supportCycleCostCalculation() {
    return false;
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::GroupQuantCorrectionOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getActOut(), getLoc()));
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

mlir::FailureOr<OutputTiling> vpux::VPU::GroupQuantCorrectionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

vpux::InputTiling vpux::VPU::GroupQuantCorrectionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    // act_in: [G, 1, 1, GS] — G (N) tiles with output; GS (W) is group size, never tiled
    TileInfo actInTile(getShape(getActIn()));
    transferTilingInfo(actInTile, outputTile, {Dim(Dims4D::Act::N)});

    // scales and zp: [G, OC, 1, 1] — same layout as output
    TileInfo scalesTile(getShape(getScales()));
    transferTilingInfo(scalesTile, outputTile, {Dim(Dims4D::Act::N), Dim(Dims4D::Act::C)});

    TileInfo zpTile(getShape(getZp()));
    transferTilingInfo(zpTile, outputTile, {Dim(Dims4D::Act::N), Dim(Dims4D::Act::C)});

    return TilingInfo{{outputTile, std::move(actInTile), std::move(scalesTile), std::move(zpTile)}};
}

void vpux::VPU::GroupQuantCorrectionOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo&
                                                    /*outputTile*/) {
    // Do nothing
}
