//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

#include <optional>

#include <vpu/layer_split_info.h>

namespace VPUNN {
class VPUCostModel;
}  // namespace VPUNN

namespace vpux::VPU {

/**
 * Intermediate representation for a single DPU workload before materialization into IR.
 * Holds all the arguments needed to create a VPU.DPU.Workload op.
 */
struct WorkloadDescriptor {
    Shape outOffsets;
    Shape outSizes;
    PadInfo padding;
    VPU::MPEMode mpeMode;
    std::optional<int64_t> clusterId;

    // Input workload (populated by backInferInputWorkloads)
    Shape inOffsets;
    Shape inSizes;
};

/**
 * Compute workload descriptors using the heuristic path (DpuTiler + VPUNN L1 cost).
 * Returns the descriptors and sets dpuCost to the best split cost.
 */
SmallVector<WorkloadDescriptor, 4> computeWorkloadDescriptors(VPU::NCEOpInterface nceOp, config::ArchKind arch,
                                                              int64_t numDPU, VPUNN::VPUCostModel& costModel,
                                                              int64_t& dpuCost, Logger log);

/**
 * Compute workload descriptors for the SCF flow using pre-built WorkloadCostParams.
 * Unlike the above overload, this does NOT read shapes/pads/strategy from the op's types.
 * The caller provides a fully populated WorkloadCostParams (via getWorkloadCostParamForSCF)
 * and an explicit outputShape for MPE mode computation.
 */
SmallVector<WorkloadDescriptor, 4> computeWorkloadDescriptorsForSCF(VPU::NCEOpInterface nceOp,
                                                                    VPUIP::WorkloadCostParams& costParams,
                                                                    ShapeRef outputShape,
                                                                    VPUNN::VPUCostModel& costModel, int64_t& dpuCost,
                                                                    Logger log);

/**
 * Create VPU.DPU.Workload ops from a list of workload descriptors.
 */
void materializeWorkloads(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder, ArrayRef<WorkloadDescriptor> workloads);

/**
 * Apply corrections to workload descriptors for a single NCE operation:
 * - Depthwise channel correction (channels must be from supported set)
 * - Small kernel optimization filtering
 * - ODU autopad channel adjustment
 *
 */
void correctWorkloadDescriptors(VPU::NCEOpInterface nceOp, SmallVector<WorkloadDescriptor, 4>& workloads,
                                mlir::MLIRContext* ctx, Logger log,
                                SmallVector<int64_t>* outSupportedChannels = nullptr);

/**
 * Back-infer input workload offsets/sizes from output workload descriptors.
 * Uses kernel, stride, padding from the op to compute the input region each workload reads.
 * For distributed tensors, applies per-cluster offset adjustments.
 */
void backInferInputWorkloads(VPU::NCEOpInterface nceOp, SmallVector<WorkloadDescriptor, 4>& workloads, Logger log);

// TODO: #E215880 — reimplement shiftWorkloadsForHalo once MC-in-SCF is fully fleshed out

/**
 * Generic method to split workload with heuristic MPE mode
 * Split pattern is decided by VPUNN L1 API cost
 */
mlir::LogicalResult genericNCEWorkloadSplit(VPU::NCEOpInterface nceOp, mlir::PatternRewriter& rewriter,
                                            config::ArchKind arch, int64_t numDPU,
                                            std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log);

/**
 * Check if the operation is supported by pre-split VPUNN API
 * Special cases still use generic VPUNN API because of inaccurate cost
 */
bool isSupportedPreSplitNCEOp(VPU::NCEOpInterface nceOp);

/**
 * Materialize workloads for an NCE op with a dynamic channel dimension.
 *
 * Instead of emitting a fixed number of VPU.DPU.Workload ops, this emits an scf.for loop
 * inside the workload region. The loop iterates over dynamically-computed workload segments
 * whose channel sizes are selected from the caller-provided `supportedChannels` list.
 *
 * @param nceOp              The NCE operation to add workloads to.
 * @param builder            OpBuilder used to emit auxiliary IR; the function sets the insertion
 *                           point internally relative to `nceOp` as needed.
 * @param baseDescriptors    Workload descriptors computed for the max (bounded) shape.
 *                           Used to determine MPE mode and non-channel dimensions.
 * @param dynChannelCount    SSA value (index type) holding the runtime output channel count.
 * @param supportedChannels  Valid channel sizes, typically provided in descending order
 *                           (for example {64, 32, 16}), but not restricted to a fixed set.
 */
void materializeWorkloadsDynamic(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder,
                                 ArrayRef<WorkloadDescriptor> baseDescriptors, mlir::Value dynChannelCount,
                                 ArrayRef<int64_t> supportedChannels, Logger log);

/**
 * Collect all VPU::DPUWorkloadOp ops from the NCE op's workloads region, including
 * those nested inside scf.if / scf.for blocks (conditional or dynamic workloads).
 *
 * Unlike `getWorkloads().getOps<VPU::DPUWorkloadOp>()` which only returns top-level
 * ops, this helper uses `walk()` to recurse into nested regions.
 */
SmallVector<VPU::DPUWorkloadOp> collectAllWorkloads(VPU::NCEOpInterface nceOp);

/**
 * Split nceOp onto workloads inplace according to the splitInfo from VPUNN
 * Used when pre-split is supported
 */
void splitWorkloadsWithInfo(VPU::NCEOpInterface nceOp, mlir::OpBuilder& builder, const VPUNN::LayerSplitInfo& splitInfo,
                            Logger log);

VPU::DistributedTensorType getDistributedTensor(const mlir::Value value);

/**
 * Map distributedType mode to VPUNN:ISIStrategy and outputWriteTiles
 *     DistributionMode         ISIStrategy      OutputWriteTiles
 * SEGMENTED|MULTICASTED        CLUSTERING          numTiles
 * SEGMENTED|DUPLICATED         SPLIT_OVER_K        numTiles
 *      else                    CLUSTERING              1
 */
template <typename TensorType, typename = std::enable_if_t<std::is_same_v<VPU::DistributedTensorType, TensorType> ||
                                                           std::is_same_v<VPUIP::DistributedBufferType, TensorType>>>
VPUNN::ISIStrategy getISIStrategyForType(TensorType type, unsigned int& outputWriteTiles) {
    const auto distributionAttr = type.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    auto numClusters = distributionAttr.getNumClusters().getInt();

    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        outputWriteTiles = numClusters;
        return VPUNN::ISIStrategy::CLUSTERING;
    }
    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        outputWriteTiles = numClusters;
        return VPUNN::ISIStrategy::SPLIT_OVER_K;
    }
    if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED)) {
        return VPUNN::ISIStrategy::SPLIT_OVER_K_nxtHW;
    }
    return VPUNN::ISIStrategy::CLUSTERING;
}
}  // namespace vpux::VPU
