//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy.hpp"

namespace vpux::VPU::VF::v2 {

// Default VF merge tiling policy that accepts any previous-current VF pair and derives candidate splits from the
// requested dimensions without pattern-specific restrictions.
class GeneralMergeTilingPolicy final : public MergeTilingPolicy {
public:
    GeneralMergeTilingPolicy(bool enableVerticalFusionPipelining, Logger log, VFCacheAnalysis& cache);

    StringRef getPatternName() const override;
    bool match(VFConfig&, VFConfig&) const override;
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
};

}  // namespace vpux::VPU::VF::v2
