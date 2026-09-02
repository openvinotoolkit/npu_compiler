//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/merge_vf_region_base_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux::VPU::VF::v2 {

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public MergeVFRegionBaseRewriter<VFCase> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, bool enablePrefetchTiling,
                          const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log, VFCacheAnalysis& cache);

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    VFConfig createVFConfig(VPU::VerticalFusionOp vfOp) const override;
    std::optional<VFCase> findVFCase(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                     VPU::VerticalFusionOp mergedOp) const override;
    bool checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                             VFCase& mergedCase) const override;
    bool canMergeVFOpsWithoutCostCheck(VFCase& mergedCase) const override;
    bool canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const override;
    VPU::StrategyCost extractVFCost(VFConfig& vfConfig) const override;
    bool checkOtherVFInput(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const override;

    bool isMCStrategyAligned(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    std::shared_ptr<IVFScheduling<VFConfig>> detectScenario(VFConfig& vfConfig) const override;
    std::optional<VFCase> findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                       VPU::VerticalFusionOp currentOp, const MergeTilingPolicy& tilingPolicy) const;
    StrategyCost getOriginalVFCost(VPU::VerticalFusionOp vfOp) const;

    VFCacheAnalysis& _cache;
    SmallVector<std::unique_ptr<MergeTilingPolicy>> _tilingPolicies;
};

}  // namespace vpux::VPU::VF::v2
