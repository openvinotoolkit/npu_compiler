//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/scheduling/vf_allocate_linear.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/vf_scenario_base.hpp"

#include <functional>
#include <optional>

namespace vpux::VPU {

// VF strategy recipe
struct VfStrategyRecipe {
    VPU::VFScenario id;
    VfSchedStrategyDescriptor::FetchScope fetchScp;
    VfSchedStrategyDescriptor::OutputResidency outputRes;
};

// Outcome of runStrategySearch — carries the
// VfAllocateResult for selected strategy
struct SearchOutcome {
    VfAllocateResult result;
    std::optional<VPU::VFScenario> selectedVFStrategy;
};

// Local region scheduler for Vertical Fusion (LoopType::VF) compute regions.
// Phase 1 stub (E#202070): only getScheduleStrategy is meaningful.
class UndefinedVF : public VFScenarioBase {
public:
    // Strategy-search allocator callback (see `runStrategySearch`):
    // evaluates a single recipe and returns an allocation result.
    using AllocatorFn = std::function<VfAllocateResult(const VfStrategyRecipe&)>;

    UndefinedVF();
    llvm::StringRef getName() const override;
    // Generates a schedule for the given VF loop region and memory size using a simple allocation strategy:
    //   1. Analyze buffer usage across iterations to identify shared buffers and dependencies. Classify buffers as
    //      persistent or temporary
    //   2. Reserve (not allocate) space for shared buffers
    //   3. Allocate per-iteration buffers depending on strategy
    LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                           vpux::AddressType memorySize) const override;

    // Iterate over predefined list of strategies and return the first feasible one:
    // FULL_PREFETCHING, LASTOP_PREFETCHING, WEIGHTS_PREFETCHING, MINIMAL.
    // Note: the current linear allocator supports only MINIMAL (other recipes are treated as infeasible).
    SearchOutcome runStrategySearch(const AllocatorFn& allocator) const;

private:
    Logger _log;
};

}  // namespace vpux::VPU
