//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
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

enum class MergeTilingPolicyType { General, ConvPostEltwise };

// Strategy interface used by the VF merge rewriter to choose a merge pattern, generate candidate tiling splits,
// compute tile-count bounds, and optionally align multi-cluster strategies after two VF regions are merged.
class MergeTilingPolicy {
public:
    MergeTilingPolicy(MergeTilingPolicyType type, VFCacheAnalysis& cache, bool enableVerticalFusionPipelining,
                      Logger log)
            : _type(type), _cache(cache), _enableVerticalFusionPipelining(enableVerticalFusionPipelining), _log(log) {
    }

    virtual ~MergeTilingPolicy() = default;

    // Returns the policy kind used to select specialized VF merge behavior.
    MergeTilingPolicyType getPolicyType() const {
        return _type;
    }

    // Returns the printable policy name used in tracing and diagnostics.
    virtual StringRef getPatternName() const = 0;

    // Checks whether this policy can handle the previous-current VF pair.
    virtual bool match(VFConfig& prevConfig, VFConfig& currentConfig) const = 0;

    // Builds candidate tiling splits for the merged VF according to the current policy.
    virtual SmallVector<VFSplit> getSplits(DimArrRef dimsToCheck, DimArrRef allowedDims, VFConfig& vfConfig,
                                           const VFSplit& prevVFSplit, const VFSplit& curVFSplit) const = 0;

    // Returns the minimal tile count for a dimension in a candidate split.
    virtual int64_t getMinimumNumber(Dim dim, const VFSplit& split, ArrayRef<int64_t> currentTiling,
                                     ArrayRef<int64_t> prevTiling, VFConfig& currentConfig, VFConfig& prevConfig,
                                     size_t linkNumber) const = 0;

    // Returns the maximal tile count for a dimension in a candidate split.
    virtual int64_t getMaximalNumber(Dim dim, const VFSplit& split, VFConfig& vfConfig) const = 0;

    // Indicates whether the rewriter may try lower-priority split dimensions when preferred splits fail.
    virtual bool hasFallbackSplits() const = 0;

    // Adjusts multi-cluster strategies inside the merged VF when required by the policy.
    virtual bool adjustMCStrategyInMergedVF(VPU::VerticalFusionOp currentVF, VPU::VerticalFusionOp prevVF,
                                            VPU::VerticalFusionOp mergedVF) const = 0;

    // Finds a parent operation strategy compatible with the user input distribution after view-op inference.
    virtual mlir::FailureOr<VPU::MultiClusterStrategy> alignMCStrategy(
            const OpWithViewInputs& parentOpInfo, VPU::ClusteredOpInterface userOp,
            const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) const = 0;

protected:
    MergeTilingPolicyType _type;
    VFCacheAnalysis& _cache;
    bool _enableVerticalFusionPipelining;
    Logger _log;
};

// Creates a concrete merge tiling policy for the requested policy kind.
std::unique_ptr<MergeTilingPolicy> createMergeTilingPolicy(MergeTilingPolicyType policyType,
                                                           bool enableVerticalFusionPipelining, Logger log,
                                                           VFCacheAnalysis& cache);

}  // namespace vpux::VPU::VF::v2
