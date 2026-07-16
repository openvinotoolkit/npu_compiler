//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <cstdint>

namespace vpux {
namespace VPUIPDPU {

struct DpuPvpCounts {
    uint64_t fpOps;
    uint64_t totalOps;
};

// Walks VPUIPDPU::DPUInvariantOp nodes in the ELF main and accumulates
// the total op count and the floating-point op count for DPU PVP throttling.
DpuPvpCounts computeDpuPvpCounts(ELF::MainOp elfMain, const Logger& log);

}  // namespace VPUIPDPU
}  // namespace vpux
