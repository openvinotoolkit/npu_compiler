//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

//
// SWOpInterface
//

bool Shave::RepeatedExternalKernelOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/,
                                                 Byte /*reservedMem*/) {
    return false;
}

bool Shave::RepeatedExternalKernelOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/) {
    return false;
}

bool Shave::RepeatedExternalKernelOp::supportCycleCostCalculation() {
    return false;
}

//
// ClusteredOpInterface
//

bool Shave::RepeatedExternalKernelOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributionInfo Shave::RepeatedExternalKernelOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface Shave::RepeatedExternalKernelOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, VPU::SiblingOpsAnalysis& siblingsAnalysis) {
    // Locate this operand among the scratch_buffers values.
    // Only CMX scratch buffers need DUPLICATED distribution across tiles;
    // DDR and AUTO scratch buffers stay non-distributed (each tile shares the same DDR allocation).
    // AUTO buffers are considered DDR when repetition is ON
    auto scratchBuffers = getCmxScratchBuffers();

    if (llvm::find(scratchBuffers, operand.get()) != scratchBuffers.end()) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }

    // Non-scratch inputs are not distributed; each tile reads from the same DDR tensors.
    return mlir::dyn_cast<NDTypeInterface>(operand.get().getType());
}

vpux::NDTypeInterface Shave::RepeatedExternalKernelOp::getDistributedTypeForOpResult(
        mlir::Value result, [[maybe_unused]] VPU::MultiClusterStrategy strategy,
        [[maybe_unused]] VPU::SiblingOpsAnalysis& siblingsAnalysis, [[maybe_unused]] bool hasExplicitDistributedAttr) {
    return mlir::dyn_cast<NDTypeInterface>(result.getType());
}

//
// AuxiliaryBufferOpInterface
//

vpux::SmallVector<mlir::OpOperand*> Shave::RepeatedExternalKernelOp::getAuxiliaryBuffers() {
    SmallVector<mlir::OpOperand*> scratchBuffs{};

    for (auto& cmxScratchBuffer : getCmxScratchBuffersMutable()) {
        scratchBuffs.emplace_back(&cmxScratchBuffer);
    }

    for (auto& ddrScratchBuffer : getDdrScratchBuffersMutable()) {
        scratchBuffs.emplace_back(&ddrScratchBuffer);
    }

    return scratchBuffs;
}
