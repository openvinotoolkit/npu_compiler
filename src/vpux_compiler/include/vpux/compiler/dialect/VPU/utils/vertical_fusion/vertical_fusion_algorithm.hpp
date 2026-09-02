//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLFunctionalExtras.h>

#include <deque>

namespace vpux::VPU::VF::v2 {

std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>> getSchedulingScenarios(VFCase::VFConfigType& config,
                                                                                        Logger log);

// find optimal VF configuration based on operations merged in VF
// the algorithm searches for optimal tiling axis, tiling number and scheduling
VPU::VF::v2::VFCase getVFCaseWithTiling(
        VFCase::VFConfigType& config, Dim dim, const VFSplit& split,
        const std::function<int64_t(Dim, const VFSplit&)>& minNumCalc,
        const std::function<int64_t(Dim, const VFSplit&)>& maxNumCalc, Logger log,
        const std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>>& vfSchedulingChecks);

/// Filter callback type used by findBestVFCaseFromSplits.
/// Called for each candidate split dimension before computing VFCase.
/// Returns true if the split should be skipped (filtered out).
using SplitFilterFn = std::function<bool(Dim dim, const VFSplit& split)>;

/// Find the best VFCase by trying all candidate splits, computing VFCase for each,
/// and selecting the one with lowest cost.
/// @param config       merged VFConfig
/// @param splits       candidate splits to evaluate
/// @param minNumCalc   callback returning minimum number of tiles for a given dim/split
/// @param maxNumCalc   callback returning maximum number of tiles for a given dim/split
/// @param costFunction cost model for evaluating VFCase cost
/// @param log          logger
/// @param splitFilter  optional filter to skip certain splits (returns true to skip)
/// @param ctx          MLIRContext for parallel execution
/// @param bestSplitOut optional output for the winning split
std::optional<VFCase> findBestVFCaseFromSplits(VFConfig& config, ArrayRef<VFSplit> splits,
                                               const std::function<int64_t(Dim, const VFSplit&)>& minNumCalc,
                                               const std::function<int64_t(Dim, const VFSplit&)>& maxNumCalc,
                                               const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log,
                                               mlir::MLIRContext* ctx, const SplitFilterFn& splitFilter = nullptr);

/// Core cost comparison logic for VF merge decisions.
/// Compares the merged VFCase cost against the sum of individual VF block costs.
/// Returns true if merging is profitable (merged cost <= prev + current cost).
bool isVFMergeProfitable(VFCase& mergedCase, StrategyCost prevCost, StrategyCost currentCost,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

// Detect VF split for SCF tiling based on output type and allowed dimensions.
// Used as the split getter callback in the HostCompile pipeline.
VF::v2::VFSplit getVFSplitForSCF(VF::v2::VFConfig& config, DimArrRef allowedDims);

VPU::VF::v2::VFCase computeVFSCFCase(vpux::NDTypeInterface outputType, const VPU::VF::v2::VFSplit& vfSplit,
                                     VPU::VF::v2::VFConfig& config, mlir::Operation* rootOp, Logger log);

/// Check if a view-like operation is eligible to be vertically fused.
/// Returns true when the operation is a pure TilingViewLikeOp that
/// supports VF and is not sourced exclusively from block arguments or constants.
bool isViewLikeOpVFCompatible(mlir::Operation* op);

/// Compute a baseline cost for a standalone operation.
/// Accounts for per-operation compute cost and spilling overhead due to spatial tiling.
/// Used by the SCF vertical fusion pass to evaluate merge profitability.
StrategyCost extractBaselineCost(mlir::Operation* operation, ShapeRef tilingDims,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                    ArrayRef<TileInfo> inputTiles, const bool enablePrefetching);

// Find a compatible MC strategy for a producer op in VF by iterating potential strategies
// (HKSwitch, SOH, SOK). The isCompatible callback receives the candidate strategy and the
// distributed output type of the producer under that strategy. Returns the first strategy
// for which all internal checks pass and isCompatible returns true.
std::optional<VPU::MultiClusterStrategy> findCompatibleMCStrategyForVF(
        VPU::ClusteredOpInterface producerOp,
        llvm::function_ref<bool(VPU::MultiClusterStrategy, VPU::DistributedTensorType)> isCompatible);

}  // namespace vpux::VPU::VF::v2
