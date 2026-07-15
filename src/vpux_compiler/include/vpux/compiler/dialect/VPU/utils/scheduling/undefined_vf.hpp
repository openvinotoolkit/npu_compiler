//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/vf_scenario_base.hpp"

namespace vpux::VPU {

// Local region scheduler for Vertical Fusion (LoopType::VF) compute regions.
// Phase 1 stub (E#202070): only getScheduleStrategy is meaningful.
class UndefinedVF : public VFScenarioBase {
public:
    UndefinedVF();
    llvm::StringRef getName() const override;
    LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                           vpux::AddressType memorySize) const override;

private:
    Logger _log;
};

}  // namespace vpux::VPU
