//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/merge_vf_region_base_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux::VPU::VF::v2 {

//
// Data structure to record the op in VF and their input view ops, which have incompatible multicluster strategy
// with previous VF
//
struct OpWithViewInputs {
    int64_t operandIdx;
    VPU::ClusteredOpInterface clusteredOp;
    SmallVector<mlir::Operation*> viewOps;
};

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public MergeVFRegionBaseRewriter<VFCase> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, bool enablePrefetchTiling,
                          const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : MergeVFRegionBaseRewriter<VFCase>(ctx, enableVerticalFusionPipelining, enablePrefetchTiling, costFunction,
                                                log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    std::optional<VFCase> findVFCase(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                     VPU::VerticalFusionOp mergedOp) const override;
    bool checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                             VFCase& mergedCase) const override;
    bool canMergeVFOpsWithoutCostCheck(VFCase& mergedCase) const override;
    bool canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const override;
    VPU::StrategyCost extractVFCost(VFConfig& vfConfig) const override;

    bool isMCStrategyAligned(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    bool adjustMCStrategyInMergedVF(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp,
                                    VPU::VerticalFusionOp mergedVF) const;
    mlir::FailureOr<VPU::MultiClusterStrategy> alignMCStrategy(
            const OpWithViewInputs& opToAdjust, VPU::ClusteredOpInterface userOp,
            const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) const;

    std::shared_ptr<IVFScheduling<VFConfig>> detectScenario(VFConfig& vfConfig) const override;
    std::optional<VFCase> findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                       VPU::VerticalFusionOp currentOp) const;
};

}  // namespace vpux::VPU::VF::v2
