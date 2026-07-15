//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"

#include <functional>
#include <map>
#include <optional>

namespace vpux::VPU::VF::v2 {

class VFCase;
class VFConfig;

using VFSplit = std::map<Dim, std::optional<int64_t>>;

/// Configuration for VF merge decisions.
/// Encapsulates the split detection and merge policy callbacks so that
/// different compilation pipelines (DefaultHW, HostCompile) can plug in
/// their own strategies without duplicating control flow.
struct VFMergeConfiguration {
    /// Callback to detect the tiling split for the given set of VF operations.
    /// Takes the VFConfig and allowed dims; returns the VFCase describing
    /// how the output should be partitioned.
    using OptimalSplitGetterFn = std::function<VFCase(VFConfig& config, DimArrRef allowedDims)>;

    /// Callback to decide whether to proceed with merging for a given VFCase.
    /// Returns true if the VFCase is valid and merging should proceed.
    using MergeDecisionFn = std::function<bool(VFCase& vfCase)>;

    /// Callback to determine if view like operation can be processed by vertical fusion.
    using ViewLikePolicyFn = std::function<bool(mlir::Operation*)>;

    OptimalSplitGetterFn splitGetter;
    MergeDecisionFn mergeDecision;
    ViewLikePolicyFn viewLikePolicy;
};

}  // namespace vpux::VPU::VF::v2

namespace vpux::VPU {
// Alias for backward compatibility with code using the unqualified name
using MergeConfiguration = VF::v2::VFMergeConfiguration;
}  // namespace vpux::VPU
