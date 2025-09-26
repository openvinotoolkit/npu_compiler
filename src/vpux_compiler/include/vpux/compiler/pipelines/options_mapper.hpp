//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/init.hpp"

#include <memory>

namespace vpux {

config::ArchKind getArchKind(const intel_npu::Config& config);
config::CompilationMode getCompilationMode(const intel_npu::Config& config);
std::optional<int> getRevisionID(const intel_npu::Config& config);
std::optional<int> getNumberOfDPUGroups(const intel_npu::Config& config);
std::optional<int> getNumberOfDMAEngines(const intel_npu::Config& config);
std::optional<bool> getWlmRollback(const intel_npu::Config& config);
Byte getAvailableCmx(const intel_npu::Config& config);
std::optional<bool> getWlmEnabled(const intel_npu::Config& config);
std::optional<bool> getQDQOptimization(const intel_npu::Config& config);
std::optional<bool> getQDQOptimizationAggressive(const intel_npu::Config& config);
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config);
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config);
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config);
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config);
std::optional<bool> getCompilerDynamicQuantization(const intel_npu::Config& config);
std::optional<bool> getPerfCount(const intel_npu::Config& config);

#ifdef BACKGROUND_FOLDING_ENABLED
struct ConstantFoldingConfig {
    bool foldingInBackgroundEnabled;
    int64_t maxConcurrentTasks;
    bool collectStatistics;
    int64_t memoryUsageLimit;
    double cacheCleanThreshold;
};
std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config);
#endif

std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config);

}  // namespace vpux
