//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN1SumOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN1SumOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto iShape = iType.getShape().raw();
    const auto outN = iShape[Dims4D::Act::N.ind()];
    auto outC = op.getAcrossChannels() ? 1 : iShape[Dims4D::Act::C.ind()];
    auto outW = op.getNormalizeVariance() ? 2 : 1;  // {sum, sqSum} or {sum}
    SmallVector<int64_t> oShape{outN, outC, op.getOutputHeight(), outW};

    // output-precision = f32, irrespective of input-precision
    // output-layout = NHWC
    const auto outOrder = DimsOrder::NHWC;
    const auto tensorAttr = vpux::getTensorAttr(ctx, outOrder.toAffineMap(ctx), iType.getMemSpace(), getBounds(iType));
    auto oType = mlir::RankedTensorType::get(oShape, mlir::Float32Type::get(ctx), tensorAttr);

    inferredReturnTypes.push_back(oType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::MVN1SumOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    // only SOK works for MVN1_SUM, also across_channels must be false
    auto inputShape = getShape(getInput());
    TileInfo inputTile({inputShape[Dims4D::Act::N], outputTile.shape[Dims4D::Act::C], inputShape[Dims4D::Act::H],
                        inputShape[Dims4D::Act::W]});
    return TilingInfo{{std::move(inputTile)}};
}

void vpux::VPU::MVN1SumOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::MVN1SumOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MVN1SumOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getShape();
    const auto outputShape = mlir::cast<vpux::NDTypeInterface>(getSum().getType()).getShape();
    // only SOK works for MVN1_SUM, also across_channels must be false
    return (strategy == VPU::MultiClusterStrategy::Clustering) ||
           (strategy == VPU::MultiClusterStrategy::SplitOverHeight &&
            inputShape[Dims4D::Act::H] >= checked_cast<int64_t>(numTiles) &&
            outputShape[Dims4D::Act::H] >= checked_cast<int64_t>(numTiles)) ||
           (strategy == VPU::MultiClusterStrategy::SplitOverKernel &&
            inputShape[Dims4D::Act::C] >= checked_cast<int64_t>(numTiles) &&
            outputShape[Dims4D::Act::C] >= checked_cast<int64_t>(numTiles));
}

vpux::VPU::DistributionInfo vpux::VPU::MVN1SumOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::MVN1SumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "MVN1SumOp requires 1 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::MVN1SumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

// static clone of 'fitIntoCMX' to eval i/o buffer sizes in absence of an op-instance
bool vpux::VPU::MVN1SumOp::buffsFitIntoCMX(mlir::ModuleOp module, vpux::NDTypeInterface in, vpux::NDTypeInterface out) {
    SmallVector<Byte> buffersSize;
    buffersSize.push_back(in.getTotalAllocSize());
    buffersSize.push_back(out.getTotalAllocSize());
    const auto arch = config::getArch(module);
    auto totalAvailableCMXSize = getTotalCMXSize(module).count();
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffersSize).count() <= totalAvailableCMXSize;
}

bool vpux::VPU::MVN1SumOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MVN1SumOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                 bool across_channels, bool normalize_variance, int64_t output_height) {
    build(builder, state, input, across_channels, normalize_variance, output_height, {});
}
