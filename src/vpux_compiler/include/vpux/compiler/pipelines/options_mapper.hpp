//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/utils/compiler_profiling/compiler_profiling.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/ov/config.hpp"

namespace vpux {

config::Platform getPlatform(const vpux::OV::Config& config);
config::ArchKind getArchKind(const vpux::OV::Config& config);
config::CompilationMode getCompilationMode(const vpux::OV::Config& config);
std::optional<int> getRevisionID(const vpux::OV::Config& config);
std::optional<int> getNumberOfDPUGroups(const vpux::OV::Config& config);
std::optional<int> getNumberOfDMAEngines(const vpux::OV::Config& config);
std::optional<bool> getQDQOptimization(const vpux::OV::Config& config);
std::optional<bool> getQDQOptimizationAggressive(const vpux::OV::Config& config);
std::optional<bool> getEnableVerifiers(const vpux::OV::Config& config);
std::optional<bool> getEnableMemoryUsageCollector(const vpux::OV::Config& config);
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const vpux::OV::Config& config);
std::optional<DummyOpMode> getDummyOpReplacement(const vpux::OV::Config& config);
std::optional<bool> getCompilerDynamicQuantization(const vpux::OV::Config& config);
std::optional<bool> getPerfCount(const vpux::OV::Config& config);
std::optional<bool> getEnableDecomposeSDPA(const vpux::OV::Config& config);
std::set<std::string> getIoWithDynamicStrides(const vpux::OV::Config& config);
std::optional<bool> getEnablePipelinedCmdListRecording(const vpux::OV::Config& config);

std::optional<std::string> getPerformanceHintOverride(const vpux::OV::Config& config);
std::optional<std::string> getCustomKernelConfigPath(const vpux::OV::Config& config);

vpux::compiler_profiling::CompilerProfiler getCompilerProfilerTool(const vpux::OV::Config& config);

}  // namespace vpux
