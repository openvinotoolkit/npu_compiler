//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vpux/utils/core/range.hpp>
#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/utils/core/error.hpp"

#include "mlir/IR/BuiltinOps.h"

#include <algorithm>

namespace vpux::VPUASM {

template <typename PerfMetricsType>
void populatePerformanceMetrics(PerfMetricsType& perf, mlir::ModuleOp module) {
    const auto& freqTable = config::getNPUConstraints(module->getContext()).frequencyTable;
    perf.freq_base = freqTable.base;
    perf.freq_step = freqTable.step;
    perf.bw_base = VPU::getBWBase();
    perf.bw_step = VPU::getBWStep();

    auto tileResources = config::getTileExecutor(module);
    const auto execKind = config::getKindValue<config::ExecutorKind>(tileResources);
    if (config::ExecutorKind::NCE == execKind) {
        perf.activity_factor = static_cast<float>(VPU::getActivityFactor(execKind, tileResources));
    }
    VPUX_THROW_WHEN(perf.activity_factor == VPU::INVALID_AF, "Invalid activity factor {0}!", perf.activity_factor);

    const auto numEntries = VPU::getNumEntries();
    auto& byBWScales = VPU::getBWScales();
    auto byBWTicks = VPU::getBWTicks(module);
    for (auto row : irange(numEntries)) {
        std::copy_n(&byBWScales[0], numEntries, &perf.scalability[row][0]);
        std::copy_n(&byBWTicks[row][0], numEntries, &perf.ticks[row][0]);
    }
}

}  // namespace vpux::VPUASM
