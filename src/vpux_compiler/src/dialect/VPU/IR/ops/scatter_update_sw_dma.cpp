//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module, int64_t numIndices) {
    int KER_WSZ_BYTES = numIndices * sizeof(int64_t) + 8 * 1024;
    return mlir::RankedTensorType::get({1, 1, 1, KER_WSZ_BYTES}, getUInt8Type(module.getContext()));
}

}  // namespace

//
// InferReturnTypes
//

mlir::LogicalResult vpux::VPU::ScatterUpdateSwDmaOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ScatterUpdateSwDmaOpAdaptor adaptor(operands, attrs, prop);
    if (mlir::failed(adaptor.verify(loc))) {
        return mlir::failure();
    }

    // Output has the same type as the data input.
    inferredReturnTypes.push_back(adaptor.getInput().getType());
    return mlir::success();
}

//
// Build
//

void vpux::VPU::ScatterUpdateSwDmaOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                            ::mlir::Value input, ::mlir::Value indices, ::mlir::Value updates,
                                            ::mlir::IntegerAttr axisValue) {
    auto loc = odsState.location;
    auto module = getModuleOp(odsBuilder);
    const auto numIndices = mlir::cast<vpux::NDTypeInterface>(indices.getType()).getNumElements();
    auto auxBufferType = getAuxiliaryBufferType(module, numIndices);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, loc, auxBufferType);
    build(odsBuilder, odsState, input.getType(), input, indices, updates, auxBuffer, axisValue, nullptr);
}

//
// SWOpInterface
//

bool vpux::VPU::ScatterUpdateSwDmaOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/,
                                                 Byte /*reservedMem*/) {
    return false;
}

bool vpux::VPU::ScatterUpdateSwDmaOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/) {
    return false;
}

bool vpux::VPU::ScatterUpdateSwDmaOp::supportCycleCostCalculation() {
    return false;
}

//
// AuxiliaryBufferOpInterface
//

SmallVector<mlir::OpOperand*> vpux::VPU::ScatterUpdateSwDmaOp::getAuxiliaryBuffers() {
    return {&getAuxBufferMutable()};
}

//
// ClusteredOpInterface
//

bool vpux::VPU::ScatterUpdateSwDmaOp::checkStrategyCompatibility(VPU::MultiClusterStrategy /*strategy*/,
                                                                 size_t /*numClusters*/) {
    return false;  // force single tile
}

vpux::VPU::DistributionInfo vpux::VPU::ScatterUpdateSwDmaOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /*memoryNumTiles*/) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface vpux::VPU::ScatterUpdateSwDmaOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    if (operand.get() == getAuxBuffer()) {
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }
    return mlir::cast<NDTypeInterface>(operand.get().getType());
}

vpux::NDTypeInterface vpux::VPU::ScatterUpdateSwDmaOp::getDistributedTypeForOpResult(
        mlir::Value result, [[maybe_unused]] VPU::MultiClusterStrategy strategy,
        [[maybe_unused]] SiblingOpsAnalysis& siblingsAnalysis, [[maybe_unused]] bool hasExplicitDistributedAttr) {
    return mlir::cast<NDTypeInterface>(result.getType());
}
