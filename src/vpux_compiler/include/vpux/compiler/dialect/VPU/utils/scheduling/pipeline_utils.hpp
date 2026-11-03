//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"

namespace vpux::VPU {

bool supportsVFPipelining(const ComputeRegion& region);

// Rewrites a VF_PIPELINING schedule into a wavefront: hoists dependency-safe
// lead-executor stages of iteration k+1 (with their input DMAs) into iteration k,
// inserted before the opposite executor's own input DMAs. When there are fewer
// opposite-kind anchors than independent lead-kind segments to hoist (e.g.
// DPU1->DPU2->DPU3->SHV->DPU has 3 DPU segments but only 1 SHV anchor), the
// segments beyond the first (anchorCount - 1) are all merged into the last anchor's
// group - the full overflow chain is always hoisted together, since partially
// hoisting it can split a buffer's allocation from its consumer across the
// iteration boundary. Has no effect when anchors are already sufficient.
bool rewriteVFPipelinedScheduleIntoPairwiseSteps(PredefinedSchedule& schedule);

}  // namespace vpux::VPU
