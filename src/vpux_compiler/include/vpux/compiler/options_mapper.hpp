//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/init.hpp"

namespace vpux {

VPU::InitCompilerOptions getInitCompilerOptions(const intel_npu::Config& config);

VPU::ArchKind getArchKind(const intel_npu::Config& config);
VPU::CompilationMode getCompilationMode(const intel_npu::Config& config);
std::optional<int> getRevisionID(const intel_npu::Config& config);
std::optional<int> getNumberOfDPUGroups(const intel_npu::Config& config);
std::optional<int> getNumberOfDMAEngines(const intel_npu::Config& config);
uint32_t getPlatformDPUClusterNum(const std::string& platform);
std::optional<bool> getWlmRollback(const intel_npu::Config& config);
Byte getAvailableCmx(const intel_npu::Config& config);
std::optional<bool> getEnableFP16CompressConv(const intel_npu::Config& config);
std::optional<bool> getReduceOperations(const intel_npu::Config& config);
std::optional<bool> getWlmEnabled(const intel_npu::Config& config);
std::optional<bool> getEnableAutoPaddingIDU(const intel_npu::Config& config);
std::optional<bool> getEnableAutoPaddingODU(const intel_npu::Config& config);
std::optional<bool> getEnableSEPtrsOperations(const intel_npu::Config& config);
std::optional<bool> getEnableExperimentalSEPtrsOperations(const intel_npu::Config& config);
std::optional<bool> getEnableAdaptiveStripping(const intel_npu::Config& config);
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config);
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config);
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config);
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config);
bool getEnableExtraShapeBoundOps(const intel_npu::Config& config);

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
