//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/ops_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <cstdint>

namespace vpux {
namespace VPU {

constexpr double INVALID_AF = -1;

struct FrequencyTable {
    uint32_t base;
    uint32_t step;
};

uint32_t getBWBase();
uint32_t getBWStep();
uint32_t getNumEntries();
double getProfClk();
const SmallVector<float>& getBWScales();
SmallVector<SmallVector<uint64_t>> getBWTicks(mlir::ModuleOp module);
double getActivityFactor(config::ExecutorKind execKind, config::ComputeResourceOpInterface res);

}  // namespace VPU
}  // namespace vpux
