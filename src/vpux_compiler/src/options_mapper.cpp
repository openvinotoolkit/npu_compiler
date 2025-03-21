//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"

#include "vpux/compiler/compiler.hpp"
#include "vpux/compiler/options_mapper.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/utils/IE/private_properties.hpp"

#include "vpux/compiler/NPU37XX/pipelines.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"

#include <openvino/runtime/properties.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

namespace {

std::optional<int> getMaxTilesValue(const intel_npu::Config& config) {
    if (config.has<intel_npu::MAX_TILES>()) {
        auto logger = vpux::Logger::global();
        int maxTiles = checked_cast<int>(config.get<intel_npu::MAX_TILES>());
        std::string platformName = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
        // E#117389: remove overrides and change to exceptions once driver & plugin will be fixed
        const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(platformName));
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
    std::string platformName = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(platformName));
    const auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value()) {
        return maybeMaxTiles.value();
    }
    return maxArchTiles;
}

int getNumberOfDPUGroupsUnchecked(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
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

std::optional<bool> getCompilerDynamicQuantization(const intel_npu::Config& config) {
    if (config.has<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>()) {
        return config.get<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>();
    } else {
        return std::nullopt;
    }
}

}  // namespace

VPU::InitCompilerOptions vpux::getInitCompilerOptions(const intel_npu::Config& config) {
    const auto archKind = getArchKind(config);
    const auto compilationMode = getCompilationMode(config);
    const auto revisionID = getRevisionID(config);
    const auto numOfDPUGroups = getNumberOfDPUGroups(config);
    const auto numOfDMAPorts = getNumberOfDMAEngines(config);
    const auto wlmRollback = getWlmRollback(config);
    const auto availableCmx = getAvailableCmx(config);
    const auto enableFP16CompressedConv = getEnableFP16CompressConv(config);
    const auto enableAutoPaddingIDU = getEnableAutoPaddingIDU(config);
    const auto enableAutoPaddingODU = getEnableAutoPaddingODU(config);
    const auto enableSEPtrsOperations = getEnableSEPtrsOperations(config);
    const auto enableExperimentalSEPtrsOperations = getEnableExperimentalSEPtrsOperations(config);
    const auto enableIsReduceSupported = getReduceOperations(config);
    const auto compilerDynamicQuantization = getCompilerDynamicQuantization(config);
    const auto enableAdaptiveStripping = getEnableAdaptiveStripping(config);

    return {archKind,
            compilationMode,
            revisionID,
            numOfDPUGroups,
            numOfDMAPorts,
            wlmRollback,
            availableCmx,
            enableFP16CompressedConv,
            enableAutoPaddingIDU,
            enableAutoPaddingODU,
            enableSEPtrsOperations,
            enableExperimentalSEPtrsOperations,
            enableIsReduceSupported,
            compilerDynamicQuantization,
            enableAdaptiveStripping};
}

//
// getArchKind
//

VPU::ArchKind vpux::getArchKind(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::AUTO_DETECT) {
        return VPU::ArchKind::UNKNOWN;
    } else if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPU::ArchKind::NPU37XX;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPU::ArchKind::NPU40XX;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

//
// getCompilationMode
//

VPU::CompilationMode vpux::getCompilationMode(const intel_npu::Config& config) {
    if (!config.has<intel_npu::COMPILATION_MODE>()) {
        return VPU::CompilationMode::DefaultHW;
    }

    const auto parsed = VPU::symbolizeCompilationMode(config.get<intel_npu::COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported compilation mode '{0}'",
                      config.get<intel_npu::COMPILATION_MODE>());
    return parsed.value();
}

//
// getRevisionID
//

std::optional<int> vpux::getRevisionID(const intel_npu::Config& config) {
    if (config.has<intel_npu::STEPPING>()) {
        return checked_cast<int>(config.get<intel_npu::STEPPING>());
    }
    return std::nullopt;
}

//
// getNumberOfDPUGroups
//

std::optional<int> vpux::getNumberOfDPUGroups(const intel_npu::Config& config) {
    if (config.has<intel_npu::TILES>() && config.has<intel_npu::DPU_GROUPS>()) {
        VPUX_THROW("Config conflict! NPU_TILES and NPU_DPU_GROUPS both set. Please only set NPU_TILES");
    } else if (config.has<intel_npu::TILES>()) {
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
    } else if (config.has<intel_npu::DPU_GROUPS>()) {
        int requestedDpuGroups = checked_cast<int>(config.get<intel_npu::DPU_GROUPS>());
        int maxTiles = getMaxDPUClusterNum(config);

        if (requestedDpuGroups > maxTiles) {
            vpux::Logger::global().warning(
                    "Requested number of DPU Groups is larger than maximum available tiles: ({0}) "
                    "> ({1}). Override to ({1})",
                    requestedDpuGroups, maxTiles);
            requestedDpuGroups = maxTiles;
        }
        return requestedDpuGroups;
    }

    int numOfDpuGroups = getNumberOfDPUGroupsUnchecked(config);
    auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value() && (numOfDpuGroups > maybeMaxTiles.value())) {
        vpux::Logger::global().warning(
                "PERFORMANCE_HINT parameter used more DPU_GROUPS ({0}) than MAX_TILES ({1}). Override to ({1})",
                numOfDpuGroups, maybeMaxTiles.value());
        numOfDpuGroups = maybeMaxTiles.value();
    }

    return numOfDpuGroups;
}

//
// getNumberOfDMAEngines
//

std::optional<int> vpux::getNumberOfDMAEngines(const intel_npu::Config& config) {
    if (config.has<intel_npu::DMA_ENGINES>()) {
        return checked_cast<int>(config.get<intel_npu::DMA_ENGINES>());
    }

    auto archKind = vpux::getArchKind(config);
    auto numOfDpuGroups = getNumberOfDPUGroups(config);
    int maxDmaPorts = VPU::getMaxDMAPorts(archKind);
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    const auto maxArchTiles = getPlatformDPUClusterNum(platform);

    auto getNumOfDmaPortsWithDpuCountLimit = [&]() {
        /*For architectures that have only 1 cluster, we want to bypass
        (numOfDPUGroups >= numOfDMAPorts) requirement

        Revert this back to "return maxDmaPorts" and implement E#135226 */
        if (maxArchTiles == 1) {
            return 1;
        } else {
            return std::min(maxDmaPorts, numOfDpuGroups.value_or(maxDmaPorts));
        }
    };

    if (platform == ov::intel_npu::Platform::NPU3720) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    } else {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    }
}

//
// getAvailableCmx
//

Byte vpux::getAvailableCmx(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPUX37XX_CMX_WORKSPACE_SIZE;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPUX40XX_CMX_WORKSPACE_SIZE;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

namespace vpux {

template <typename Options>
std::optional<bool> getWlmRollback(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->wlmRollback;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getWlmRollback(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getWlmRollback<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getWlmRollback<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getWlmRollback(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getWlmRollback<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getWlmRollback<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

uint32_t getPlatformDPUClusterNum(const std::string& platform) {
    if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPUX37XX_MAX_DPU_GROUPS;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPUX40XX_MAX_DPU_GROUPS;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

namespace {

template <typename Options>
std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->performanceHintOverride;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getPerformanceHintOverride<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getPerformanceHintOverride<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getPerformanceHintOverride<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

}  // namespace

std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getPerformanceHintOverride<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getPerformanceHintOverride<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableFP16CompressConv(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableFP16CompressedConvolution;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableFP16CompressConv(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableFP16CompressConv<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableFP16CompressConv<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableFP16CompressConv(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableFP16CompressConv<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableFP16CompressConv<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getReduceOperations(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableIsReduceSupported;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getReduceOperations(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getReduceOperations<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getReduceOperations<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getReduceOperations(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getReduceOperations<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getReduceOperations<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getWlmEnabled(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->workloadManagementEnable;
}

std::optional<bool> getWlmEnabled(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return std::nullopt;
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getWlmEnabled<BackendCompilationOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableAutoPaddingIDU(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableAutoPaddingIDU;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableAutoPaddingIDU(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableAutoPaddingIDU<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableAutoPaddingIDU<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableAutoPaddingIDU(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableAutoPaddingIDU<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableAutoPaddingIDU<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableAutoPaddingODU(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableAutoPaddingODU;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableAutoPaddingODU(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableAutoPaddingODU<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableAutoPaddingODU<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableAutoPaddingODU(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableAutoPaddingODU<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableAutoPaddingODU<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

// SEP

template <typename Options>
std::optional<bool> getEnableSEPtrsOperations(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableSEPtrsOperations;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableSEPtrsOperations(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableSEPtrsOperations<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableSEPtrsOperations<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableSEPtrsOperations(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableSEPtrsOperations<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableSEPtrsOperations<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableExperimentalSEPtrsOperations(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableExperimentalSEPtrsOperations;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableExperimentalSEPtrsOperations(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableExperimentalSEPtrsOperations<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableExperimentalSEPtrsOperations<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableExperimentalSEPtrsOperations(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableExperimentalSEPtrsOperations<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableExperimentalSEPtrsOperations<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

// Adaptive Stripping

template <typename Options>
std::optional<bool> getEnableAdaptiveStripping(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->enableAdaptiveStripping;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableAdaptiveStripping(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableAdaptiveStripping<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableAdaptiveStripping<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableAdaptiveStripping(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableAdaptiveStripping<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableAdaptiveStripping<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableVerifiers;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableVerifiers<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableVerifiers<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getEnableVerifiers<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableVerifiers<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableVerifiers<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableMemoryUsageCollector;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableMemoryUsageCollector<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableMemoryUsageCollector<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getEnableMemoryUsageCollector<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableMemoryUsageCollector<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableMemoryUsageCollector<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableFunctionStatisticsInstrumentation;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableFunctionStatisticsInstrumentation<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableFunctionStatisticsInstrumentation<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableFunctionStatisticsInstrumentation<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableDummyOpReplacement ? DummyOpMode::ENABLED : DummyOpMode::DISABLED;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getDummyOpReplacement<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getDummyOpReplacement<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getDummyOpReplacement<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getDummyOpReplacement<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getDummyOpReplacement<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

#ifdef BACKGROUND_FOLDING_ENABLED

template <typename Options>
std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }
    return ConstantFoldingConfig{options->constantFoldingInBackground, options->constantFoldingInBackgroundNumThreads,
                                 options->constantFoldingInBackgroundCollectStatistics,
                                 options->constantFoldingInBackgroundMemoryUsageLimit,
                                 options->constantFoldingInBackgroundCacheCleanThreshold};
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getConstantFoldingInBackground<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getConstantFoldingInBackground<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getConstantFoldingInBackground<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getConstantFoldingInBackground<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getConstantFoldingInBackground<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

#endif

template <typename Options>
bool getEnableExtraShapeBoundOps(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return false;
    }
    return options->enableExtraShapeBoundOps;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
bool getEnableExtraShapeBoundOps(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getEnableExtraShapeBoundOps<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getEnableExtraShapeBoundOps<DefaultHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        return getEnableExtraShapeBoundOps<DefaultHWOptions>(config);
    } else {
        return false;
    }
}

bool getEnableExtraShapeBoundOps(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableExtraShapeBoundOps<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableExtraShapeBoundOps<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return false;
    }
}

}  // namespace vpux
