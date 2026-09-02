//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/options.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/compiler.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

#include "vpux/compiler/NPU37XX/pipeline_options.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/NPU50XX/pipeline_options.hpp"

#include <openvino/runtime/properties.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

namespace {
// NPUPerformanceMode consists of same enums as ov::hint::PerformanceMode + EFFICIENCY.
// In future, ov::hint::PerformanceMode can be extended with the new value, so
// we do not need to have our own enum class.
enum class NPUPerformanceMode {
    LATENCY = 1,                //!<  Optimize for latency
    THROUGHPUT = 2,             //!<  Optimize for throughput
    CUMULATIVE_THROUGHPUT = 3,  //!<  Optimize for cumulative throughput
    EFFICIENCY = 4,             //!<  Optimize for power efficiency
};

// Number of DPU groups to use per performance hint, for a single platform.
// A field set to `maxTiles` means the maximum allowed for the platform.
struct DpuGroupsPolicy {
    int latency;
    int throughput;
    int efficiency;
    int fallback;
};

template <typename OptionsType>
std::unique_ptr<OptionsType> parseAllOptions(const vpux::OV::Config& config) {
    return parseCompilationModeParams<OptionsType>(config.get<vpux::OV::COMPILATION_MODE_PARAMS>(),
                                                   getArchKind(config));
}

std::unique_ptr<DefaultHWOptionsBase> getCompilerSetupOptions(const vpux::OV::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return parseAllOptions<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return parseAllOptions<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return parseAllOptions<DefaultHWOptions50XX>(config);
    } else {
        return nullptr;
    }
}

uint32_t getPlatformDPUClusterNum(const vpux::OV::Config& config) {
    return VPU::getPlatformCapabilities(getPlatform(config)).maxTiles;
}

std::optional<int> getMaxTilesValue(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::MAX_TILES>()) {
        auto logger = vpux::Logger::global();
        int maxTiles = checked_cast<int>(config.get<vpux::OV::MAX_TILES>());
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

int getMaxDPUClusterNum(const vpux::OV::Config& config) {
    const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(config));
    const auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value()) {
        return maybeMaxTiles.value();
    }
    return maxArchTiles;
}

std::optional<std::string> getPerformanceHintOverride(const vpux::OV::Config& config) {
    const auto options = getCompilerSetupOptions(config);
    if (options == nullptr) {
        return std::nullopt;
    }
    auto& perfHint = options->performanceHintOverride;
    VPUX_THROW_WHEN(perfHint.hasValue() && !config.has<vpux::OV::MAX_TILES>(),
                    "performance-hint-override is not supported in offline compilation");
    return perfHint;
}

// Per-platform DPU group policy. Each field is the number of DPU groups for the corresponding
// performance mode.
DpuGroupsPolicy getDpuGroupsPolicyByPlatform(const std::string& platform, const vpux::OV::Config& config) {
    const int maxTiles = getMaxDPUClusterNum(config);

    if (platform == vpux::OV::Platform::NPU3720) {
        return {/*latency=*/maxTiles, /*throughput=*/maxTiles, /*efficiency=*/maxTiles,
                /*fallback=*/maxTiles};
    }
    if (platform == vpux::OV::Platform::NPU4000) {
        return {/*latency=*/maxTiles, /*throughput=*/2, /*efficiency=*/4, /*fallback=*/2};
    }
    if (platform == vpux::OV::Platform::NPU5010 || platform == vpux::OV::Platform::NPU5020) {
        return {/*latency=*/maxTiles, /*throughput=*/1, /*efficiency=*/maxTiles,
                /*fallback=*/maxTiles};
    }
    vpux::Logger::global().warning("No explicit DPU groups policy for platform '{0}'; using the generic default. ",
                                   platform);
    return {/*latency=*/maxTiles, /*throughput=*/1, /*efficiency=*/maxTiles, /*fallback=*/maxTiles};
}

NPUPerformanceMode getRequestedPerformanceMode(const vpux::OV::Config& config) {
    const auto performanceHintOverride = getPerformanceHintOverride(config);
    switch (config.get<vpux::OV::PERFORMANCE_HINT>()) {
    case vpux::OV::PerformanceMode::LATENCY:
        VPUX_THROW_WHEN(!performanceHintOverride.has_value(), "performance-hint-override does not hold a value.");
        if (performanceHintOverride.value() == "efficiency") {
            return NPUPerformanceMode::EFFICIENCY;
        }
        if (performanceHintOverride.value() == "latency") {
            return NPUPerformanceMode::LATENCY;
        }
        VPUX_THROW("Unknown value `{0}` for performance-hint-override. Possible values: `latency`, `efficiency`",
                   performanceHintOverride.value());
    case vpux::OV::PerformanceMode::THROUGHPUT:
    default:
        break;
    }
    return static_cast<NPUPerformanceMode>(config.get<vpux::OV::PERFORMANCE_HINT>());
}

int getNumberOfDPUGroupsUnchecked(const vpux::OV::Config& config) {
    const std::string platform = vpux::OV::Platform::standardize(config.get<vpux::OV::PLATFORM>());
    const auto policy = getDpuGroupsPolicyByPlatform(platform, config);

    switch (getRequestedPerformanceMode(config)) {
    case NPUPerformanceMode::LATENCY:
        return policy.latency;
    case NPUPerformanceMode::THROUGHPUT:
        return policy.throughput;
    case NPUPerformanceMode::EFFICIENCY:
        return policy.efficiency;
    default:
        return policy.fallback;
    }
}

}  // namespace

namespace vpux {

//
// getPlatform
//

config::Platform getPlatform(const vpux::OV::Config& config) {
    const std::string platform = vpux::OV::Platform::standardize(config.get<vpux::OV::PLATFORM>());
    if (platform == vpux::OV::Platform::NPU3720) {
        return config::Platform::NPU3720;
    } else if (platform == vpux::OV::Platform::NPU4000) {
        return config::Platform::NPU4000;
    } else if (platform == vpux::OV::Platform::NPU5010) {
        return config::Platform::NPU5010;
    } else if (platform == vpux::OV::Platform::NPU5020) {
        return config::Platform::NPU5020;
    } else {
        VPUX_THROW("Unsupported NPU platform");
    }
}

//
// getArchKind
//

config::ArchKind getArchKind(const vpux::OV::Config& config) {
    config::Platform platform = getPlatform(config);
    return config::getArch(platform);
}

//
// getCompilationMode
//

config::CompilationMode getCompilationMode(const vpux::OV::Config& config) {
    if (!config.has<vpux::OV::COMPILATION_MODE>()) {
        return config::CompilationMode::DefaultHW;
    }

    const auto parsed = config::symbolizeCompilationMode(config.get<vpux::OV::COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported compilation mode '{0}'",
                      config.get<vpux::OV::COMPILATION_MODE>());
    return parsed.value();
}

//
// getRevisionID
//

std::optional<int> getRevisionID(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::STEPPING>()) {
        auto revision = checked_cast<int>(config.get<vpux::OV::STEPPING>());
        if (revision == static_cast<uint16_t>(-1)) {
            return static_cast<int>(config::RevisionID::REVISION_NONE);
        }
        return revision;
    }
    return std::nullopt;
}

//
// getNumberOfDPUGroups
//

std::optional<int> getNumberOfDPUGroups(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::TILES>()) {
        int requestedNpuTiles = checked_cast<int>(config.get<vpux::OV::TILES>());
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

std::optional<int> getNumberOfDMAEngines(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::DMA_ENGINES>()) {
        return checked_cast<int>(config.get<vpux::OV::DMA_ENGINES>());
    }

    auto archKind = vpux::getArchKind(config);
    auto numOfDpuGroups = getNumberOfDPUGroups(config);
    int maxDmaPorts = VPU::getMaxDMAPorts(getPlatform(config));
    const std::string platform = vpux::OV::Platform::standardize(config.get<vpux::OV::PLATFORM>());

    if (archKind == config::ArchKind::NPU37XX || archKind == config::ArchKind::NPU40XX) {
        // Without FWLM we do not support the case in which the number of DMA engines is greater than number of DPU
        // groups
        return std::min(maxDmaPorts, numOfDpuGroups.value_or(maxDmaPorts));
    } else {
        return maxDmaPorts;
    }
}  // namespace vpux

std::optional<bool> getEnableVerifiers(const vpux::OV::Config& config) {
    const auto options = getCompilerSetupOptions(config);
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableVerifiers;
}

// Adaptive Stripping

std::optional<bool> getQDQOptimization(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::QDQ_OPTIMIZATION>()) {
        return config.get<vpux::OV::QDQ_OPTIMIZATION>();
    }

    return std::nullopt;
}

std::optional<bool> getQDQOptimizationAggressive(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::QDQ_OPTIMIZATION_AGGRESSIVE>()) {
        return config.get<vpux::OV::QDQ_OPTIMIZATION_AGGRESSIVE>();
    }

    return std::nullopt;
}

std::optional<bool> getEnableMemoryUsageCollector(const vpux::OV::Config& config) {
    const auto options = getCompilerSetupOptions(config);
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableMemoryUsageCollector;
}

std::optional<bool> getEnableFunctionStatisticsInstrumentation(const vpux::OV::Config& config) {
    const auto options = getCompilerSetupOptions(config);
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableFunctionStatisticsInstrumentation;
}

std::optional<DummyOpMode> getDummyOpReplacement(const vpux::OV::Config& config) {
    const auto options = getCompilerSetupOptions(config);
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableDummyOpReplacement ? DummyOpMode::ENABLED : DummyOpMode::DISABLED;
}

std::optional<bool> getCompilerDynamicQuantization(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::COMPILER_DYNAMIC_QUANTIZATION>()) {
        return config.get<vpux::OV::COMPILER_DYNAMIC_QUANTIZATION>();
    }

    return std::nullopt;
}

// Profiling

std::optional<bool> getPerfCount(const vpux::OV::Config& config) {
    if (config.has<vpux::OV::PERF_COUNT>()) {
        return config.get<vpux::OV::PERF_COUNT>();
    }

    return std::nullopt;
}

std::optional<bool> getEnableDecomposeSDPA(const vpux::OV::Config& config) {
    const auto options = getCompilerSetupOptions(config);
    if (options == nullptr) {
        return std::nullopt;
    }

    // FIXME: E#224094 temporary keep the behavior introduced in PR23378
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU40XX) {
        // Disable ngraph SDPA decomposition for NPU0XX - custom decomposition is applied in IE pipeline
        return std::nullopt;
    } else if (arch == config::ArchKind::NPU50XX) {
        // Disable ngraph SDPA decomposition for NPU50XX - custom decomposition is applied in IE pipeline
        return std::nullopt;
    }

    return options->enableDecomposeSDPA;
}

std::set<std::string> getIoWithDynamicStrides(const vpux::OV::Config& config) {
    if (getArchKind(config) == config::ArchKind::NPU37XX && config.has<vpux::OV::ENABLE_STRIDES_FOR>()) {
        VPUX_THROW("ENABLE_STRIDES_FOR not supported on NPU3720");
    }
    auto enableStridesFor = config.get<vpux::OV::ENABLE_STRIDES_FOR>();
    std::istringstream stream(enableStridesFor);
    std::string name;
    std::set<std::string> dynamicStridesIos;
    while (std::getline(stream, name, ',')) {
        dynamicStridesIos.insert(name);
    }
    return dynamicStridesIos;
}

std::optional<bool> getEnablePipelinedCmdListRecording(const vpux::OV::Config& config) {
    const auto options = parseCompilationModeParams<HostExec::HostExecOptions>(
            config.get<vpux::OV::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enablePipelinedCmdListRecording;
}

std::optional<std::string> getCustomKernelConfigPath(const vpux::OV::Config& config) {
    return config.has<vpux::OV::CUSTOM_KERNEL_CONFIG_PATH>()
                   ? std::make_optional(config.get<vpux::OV::CUSTOM_KERNEL_CONFIG_PATH>())
                   : std::nullopt;
}

template <typename Options>
vpux::compiler_profiling::CompilerProfiler getCompilerProfilerTool(const vpux::OV::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<vpux::OV::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return vpux::compiler_profiling::CompilerProfiler{};
    }

    return vpux::compiler_profiling::CompilerProfiler::createFromString(options->compilerProfilerTool.getValue());
}

vpux::compiler_profiling::CompilerProfiler getCompilerProfilerTool(const vpux::OV::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        return getCompilerProfilerTool<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        return getCompilerProfilerTool<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        return getCompilerProfilerTool<DefaultHWOptions50XX>(config);
    } else {
        return vpux::compiler_profiling::CompilerProfiler{};
    }
}

}  // namespace vpux
