//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/utils/logger/logger.hpp"

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

}  // namespace vpux::VPU::VF::v2
