//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy.hpp"

namespace vpux::VPU::VF::v2 {

// Specialized VF merge tiling policy for a convolution VF followed by an eltwise VF, preserving the previous VF split
// shape and avoiding fallback split search when the pattern-specific merge is not applicable.
class ConvPostEltwiseMergeTilingPolicy final : public MergeTilingPolicy {
public:
    ConvPostEltwiseMergeTilingPolicy(bool enableVerticalFusionPipelining, Logger log, VFCacheAnalysis& cache);

    StringRef getPatternName() const override;
    bool match(VFConfig& prevConfig, VFConfig& currentConfig) const override;
    SmallVector<VFSplit> getSplits(DimArrRef dimsToCheck, DimArrRef allowedDims, VFConfig& vfConfig,
                                   const VFSplit& prevVFSplit, const VFSplit& curVFSplit) const override;
    bool adjustMCStrategyInMergedVF(VPU::VerticalFusionOp currentVF, VPU::VerticalFusionOp prevVF,
                                    VPU::VerticalFusionOp mergedOp) const override;
    mlir::FailureOr<VPU::MultiClusterStrategy> alignMCStrategy(
            const OpWithViewInputs& parentOpInfo, VPU::ClusteredOpInterface userOp,
            const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) const override;
    int64_t getMinimumNumber(Dim dim, const VFSplit& split, ArrayRef<int64_t> currentTiling,
                             ArrayRef<int64_t> prevTiling, VFConfig& currentConfig, VFConfig& prevConfig,
                             size_t linkNumber) const override;
    int64_t getMaximalNumber(Dim dim, const VFSplit& split, VFConfig& vfConfig) const override;
    bool hasFallbackSplits() const override;

private:
    bool matchPattern(VFConfig& prevConfig, VFConfig& currentConfig) const;
};

}  // namespace vpux::VPU::VF::v2
