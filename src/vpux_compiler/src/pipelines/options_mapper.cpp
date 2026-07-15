//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/options.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/compiler.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/utils/ov/private_properties.hpp"

#include "vpux/compiler/NPU37XX/pipeline_options.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/NPU50XX/pipeline_options.hpp"

#include <openvino/runtime/properties.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

namespace {

uint32_t getPlatformDPUClusterNum(const intel_npu::Config& config) {
    return VPU::getPlatformCapabilities(getPlatform(config)).maxTiles;
}

std::optional<int> getMaxTilesValue(const intel_npu::Config& config) {
    if (config.has<intel_npu::MAX_TILES>()) {
        auto logger = vpux::Logger::global();
        int maxTiles = checked_cast<int>(config.get<intel_npu::MAX_TILES>());
        // E#117389: remove overrides and change to exceptions once driver & plugin will be fixed
        const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(config));
        if (maxTiles < 1 || maxTiles > maxArchTiles) {
            logger.warning("Invalid number of NPU_MAX_TILES for requested arch, got {0}. Override to {1}", maxTiles,
                           maxArchTiles);
            maxTiles = maxArchTiles;
        }
        return maxTiles;
    }
    return std::nullopt;
}

int getMaxDPUClusterNum(const intel_npu::Config& config) {
    const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(config));
    const auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value()) {
        return maybeMaxTiles.value();
    }
    return maxArchTiles;
}

template <typename Options>
std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    auto& perfHint = options->performanceHintOverride;
    VPUX_THROW_WHEN(perfHint.hasValue() && !config.has<intel_npu::MAX_TILES>(),
                    "performance-hint-override is not supported in offline compilation");
    return perfHint;
}

std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getPerformanceHintOverride<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getPerformanceHintOverride<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getPerformanceHintOverride<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

int getNumberOfDPUGroupsUnchecked(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    const bool is50XXPlatform =
            platform == ov::intel_npu::Platform::NPU5010 || platform == ov::intel_npu::Platform::NPU5020;
    const auto& performanceHintOverride = getPerformanceHintOverride(config);
    // NPUPerformanceMode consists of same enums as ov::hint::PerformanceMode + EFFICIENCY
    // In future, ov::hint::PerformanceMode can be extended with the new value, so
    // we do not need to have our own enum class
    enum class NPUPerformanceMode {
        LATENCY = 1,                //!<  Optimize for latency
        THROUGHPUT = 2,             //!<  Optimize for throughput
        CUMULATIVE_THROUGHPUT = 3,  //!<  Optimize for cumulative throughput
        EFFICIENCY = 4,             //!<  Optimize for power efficiency
    };

    const auto performanceMode = [&] {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::LATENCY:
            VPUX_THROW_WHEN(!performanceHintOverride.has_value(), "performance-hint-override does not hold a value.");
            if (performanceHintOverride.value() == "efficiency") {
                return NPUPerformanceMode::EFFICIENCY;
            } else if (performanceHintOverride.value() == "latency") {
                return NPUPerformanceMode::LATENCY;
            }
            VPUX_THROW("Unknown value `{0}` for performance-hint-override. Possible values: `latency`, `efficiency`",
                       performanceHintOverride.value());
        case ov::hint::PerformanceMode::THROUGHPUT:
        default:
            break;
        }
        return static_cast<NPUPerformanceMode>(config.get<intel_npu::PERFORMANCE_HINT>());
    }();

    if (platform == ov::intel_npu::Platform::NPU3720) {
        switch (performanceMode) {
        case NPUPerformanceMode::THROUGHPUT:
        case NPUPerformanceMode::LATENCY:
        case NPUPerformanceMode::EFFICIENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        switch (performanceMode) {
        case NPUPerformanceMode::LATENCY:
            return getMaxDPUClusterNum(config);
        case NPUPerformanceMode::EFFICIENCY:
            return 4;
        case NPUPerformanceMode::THROUGHPUT:
        default:
            return 2;
        }
    } else if (is50XXPlatform) {
        switch (performanceMode) {
        case NPUPerformanceMode::THROUGHPUT:
            return 1;
        case NPUPerformanceMode::EFFICIENCY:
        case NPUPerformanceMode::LATENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    } else {
        switch (performanceMode) {
        case NPUPerformanceMode::THROUGHPUT:
            return 1;
        case NPUPerformanceMode::EFFICIENCY:
        case NPUPerformanceMode::LATENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    }
}

}  // namespace

namespace vpux {

//
// getPlatform
//

config::Platform getPlatform(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    if (platform == ov::intel_npu::Platform::NPU3720) {
        return config::Platform::NPU3720;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return config::Platform::NPU4000;
    } else if (platform == ov::intel_npu::Platform::NPU5010) {
        return config::Platform::NPU5010;
    } else if (platform == ov::intel_npu::Platform::NPU5020) {
        return config::Platform::NPU5020;
    } else {
        VPUX_THROW("Unsupported NPU platform");
    }
}

//
// getArchKind
//

config::ArchKind getArchKind(const intel_npu::Config& config) {
    config::Platform platform = getPlatform(config);
    return config::getArch(platform);
}

//
// getCompilationMode
//

config::CompilationMode getCompilationMode(const intel_npu::Config& config) {
    if (!config.has<intel_npu::COMPILATION_MODE>()) {
        return config::CompilationMode::DefaultHW;
    }

    const auto parsed = config::symbolizeCompilationMode(config.get<intel_npu::COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported compilation mode '{0}'",
                      config.get<intel_npu::COMPILATION_MODE>());
    return parsed.value();
}

//
// getRevisionID
//

std::optional<int> getRevisionID(const intel_npu::Config& config) {
    if (config.has<intel_npu::STEPPING>()) {
        return checked_cast<int>(config.get<intel_npu::STEPPING>());
    }
    return std::nullopt;
}

//
// getNumberOfDPUGroups
//

std::optional<int> getNumberOfDPUGroups(const intel_npu::Config& config) {
    if (config.has<intel_npu::TILES>()) {
        int requestedNpuTiles = checked_cast<int>(config.get<intel_npu::TILES>());
        int maxTiles = getMaxDPUClusterNum(config);
        if (requestedNpuTiles > maxTiles) {
            vpux::Logger::global().warning(
                    "Requested number of NPU tiles is larger than maximum available tiles: ({0}) "
                    "> ({1}). Override to ({1})",
                    requestedNpuTiles, maxTiles);
            requestedNpuTiles = maxTiles;
        }
        return requestedNpuTiles;
    }

    int numOfDpuGroups = getNumberOfDPUGroupsUnchecked(config);
    auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value() && (numOfDpuGroups > maybeMaxTiles.value())) {
        vpux::Logger::global().warning(
                "PERFORMANCE_HINT parameter used more NPU_TILES ({0}) than MAX_TILES ({1}). Override to ({1})",
                numOfDpuGroups, maybeMaxTiles.value());
        numOfDpuGroups = maybeMaxTiles.value();
    }

    return numOfDpuGroups;
}

//
// getNumberOfDMAEngines
//

std::optional<int> getNumberOfDMAEngines(const intel_npu::Config& config) {
    if (config.has<intel_npu::DMA_ENGINES>()) {
        return checked_cast<int>(config.get<intel_npu::DMA_ENGINES>());
    }

    auto archKind = vpux::getArchKind(config);
    auto numOfDpuGroups = getNumberOfDPUGroups(config);
    int maxDmaPorts = VPU::getMaxDMAPorts(getPlatform(config));
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (archKind == config::ArchKind::NPU37XX || archKind == config::ArchKind::NPU40XX) {
        // Without FWLM we do not support the case in which the number of DMA engines is greater than number of DPU
        // groups
        return std::min(maxDmaPorts, numOfDpuGroups.value_or(maxDmaPorts));
    } else {
        return maxDmaPorts;
    }
}  // namespace vpux

//
// getAvailableCmx
//

Byte getAvailableCmx(const intel_npu::Config& config) {
    return VPU::getPlatformCapabilities(getPlatform(config)).cmxWorkspaceSize;
}

template <typename Options>
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableVerifiers;
}

std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getEnableVerifiers<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getEnableVerifiers<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getEnableVerifiers<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getWlmEnabled(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->workloadManagementEnable;
}

std::optional<bool> getWlmEnabled(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return std::nullopt;
    } else if (arch == config::ArchKind::NPU40XX) {
        return getWlmEnabled<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getWlmEnabled<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

// Adaptive Stripping

std::optional<bool> getQDQOptimization(const intel_npu::Config& config) {
    if (config.has<intel_npu::QDQ_OPTIMIZATION>()) {
        return config.get<intel_npu::QDQ_OPTIMIZATION>();
    }

    return std::nullopt;
}

std::optional<bool> getQDQOptimizationAggressive(const intel_npu::Config& config) {
    if (config.has<intel_npu::QDQ_OPTIMIZATION_AGGRESSIVE>()) {
        return config.get<intel_npu::QDQ_OPTIMIZATION_AGGRESSIVE>();
    }

    return std::nullopt;
}

template <typename Options>
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableMemoryUsageCollector;
}

std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getEnableMemoryUsageCollector<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getEnableMemoryUsageCollector<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getEnableMemoryUsageCollector<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableFunctionStatisticsInstrumentation;
}

std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableDummyOpReplacement ? DummyOpMode::ENABLED : DummyOpMode::DISABLED;
}

std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getDummyOpReplacement<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getDummyOpReplacement<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getDummyOpReplacement<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getCompilerDynamicQuantization(const intel_npu::Config& config) {
    if (config.has<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>()) {
        return config.get<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>();
    }

    return std::nullopt;
}

#ifdef BACKGROUND_FOLDING_ENABLED

template <typename Options>
std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return ConstantFoldingConfig{options->constantFoldingInBackground, options->constantFoldingInBackgroundNumThreads,
                                 options->constantFoldingInBackgroundCollectStatistics,
                                 options->constantFoldingInBackgroundMemoryUsageLimit,
                                 options->constantFoldingInBackgroundCacheCleanThreshold};
}

std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getConstantFoldingInBackground<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getConstantFoldingInBackground<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getConstantFoldingInBackground<DefaultHWOptions50XX>(config);
    } else {
        return std::nullopt;
    }
}

#endif

// Profiling

std::optional<bool> getPerfCount(const intel_npu::Config& config) {
    if (config.has<intel_npu::PERF_COUNT>()) {
        return config.get<intel_npu::PERF_COUNT>();
    }

    return std::nullopt;
}

namespace {

template <typename Options>
std::optional<bool> getEnableDecomposeSDPA(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableDecomposeSDPA;
}

}  // namespace

std::optional<bool> getEnableDecomposeSDPA(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getEnableDecomposeSDPA<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        // Disable ngraph SDPA decomposition for NPU40XX - custom decomposition is applied in IE pipeline
        return std::nullopt;
    } else if (arch == config::ArchKind::NPU50XX) {
        // Disable ngraph SDPA decomposition for NPU50XX - custom decomposition is applied in IE pipeline
        return std::nullopt;
    } else {
        return std::nullopt;
    }
}

std::set<std::string> getIoWithDynamicStrides(const intel_npu::Config& config) {
    if (getArchKind(config) == config::ArchKind::NPU37XX && config.has<intel_npu::ENABLE_STRIDES_FOR>()) {
        VPUX_THROW("ENABLE_STRIDES_FOR not supported on NPU3720");
    }
    auto enableStridesFor = config.get<intel_npu::ENABLE_STRIDES_FOR>();
    std::istringstream stream(enableStridesFor);
    std::string name;
    std::set<std::string> dynamicStridesIos;
    while (std::getline(stream, name, ',')) {
        dynamicStridesIos.insert(name);
    }
    return dynamicStridesIos;
}

std::optional<bool> getEnablePipelinedCmdListRecording(const intel_npu::Config& config) {
    const auto options = parseCompilationModeParams<HostExec::HostExecOptions>(
            config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enablePipelinedCmdListRecording;
}

}  // namespace vpux
