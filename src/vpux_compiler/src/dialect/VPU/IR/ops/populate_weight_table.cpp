//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PopulateWeightTableOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PopulateWeightTableOpAdaptor populateWT(operands, attrs, prop);
    if (mlir::failed(populateWT.verify(loc))) {
        return mlir::failure();
    }

    auto inType = mlir::cast<NDTypeInterface>(populateWT.getScale().getType());
    auto shape = inType.getShape();
    llvm::SmallVector<int64_t> wtShape = {shape[Dims4D::Filter::OC], 1, 1, 4};

    const auto newOutputType = TypeComponents()
                                       .setDimsOrder(inType.getDimsOrder())
                                       .setShape(ShapeRef(wtShape))
                                       .setElementType(getSInt32Type(ctx));

    vpux::DimsOrder outOrder = newOutputType.dimsOrder.value();
    const auto tensorAttr =
            vpux::getTensorAttr(ctx, outOrder.toAffineMap(ctx), inType.getMemSpace(), getBounds(inType));
    auto outputType = mlir::RankedTensorType::get(wtShape, getSInt32Type(ctx), tensorAttr);

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::PopulateWeightTableOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                vpux::Logger /*log*/) {
    return TilingInfo(outputTile);
}

void vpux::VPU::PopulateWeightTableOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::PopulateWeightTableOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::PopulateWeightTableOp::checkStrategyCompatibility(VPU::MultiClusterStrategy, size_t) {
    // strategy inferred from weight table
    return true;
}

vpux::VPU::DistributionInfo vpux::VPU::PopulateWeightTableOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::PopulateWeightTableOp::fitIntoCMX(ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 1, "PopulateWeightTableOp requires 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::PopulateWeightTableOp::fitIntoCMX(ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::PopulateWeightTableOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::PopulateWeightTableOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state,
                                             ::mlir::Value scale, int64_t base, int64_t step) {
    build(builder, state, scale, base, step, {});
}
