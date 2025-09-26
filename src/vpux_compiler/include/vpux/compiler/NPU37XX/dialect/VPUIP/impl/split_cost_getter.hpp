//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

namespace VPUNN {
class VPUCostModel;
}  // namespace VPUNN

namespace vpux::VPUIP::arch37xx {

int64_t computeSplitCost(const WorkloadSplit& split, const WorkloadCostParams& params, VPUNN::VPUCostModel& costModel,
                         LogCb logCb = emptyLogCb);

}  // namespace vpux::VPUIP::arch37xx
