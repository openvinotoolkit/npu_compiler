//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"

#include <ie_plugin_config.hpp>

namespace vpux {

//
// register
//

void registerCommonOptions(OptionsDesc& desc);

//
// PERFORMANCE_HINT
//

enum class PerformanceHint {
    Latency,
    Throughput,
};

StringLiteral stringifyEnum(PerformanceHint val);

struct PERFORMANCE_HINT final : OptionBase<PERFORMANCE_HINT, PerformanceHint> {
    static StringRef key() {
        return CONFIG_KEY(PERFORMANCE_HINT);
    }

    static PerformanceHint parse(StringRef val);
};

//
// PERF_COUNT
//

struct PERF_COUNT final : OptionBase<PERF_COUNT, bool> {
    static StringRef key() {
        return CONFIG_KEY(PERF_COUNT);
    }

    static bool defaultValue() {
        return true;
    }
};

//
// LOG_LEVEL
//

struct LOG_LEVEL final : OptionBase<LOG_LEVEL, LogLevel> {
    static StringRef key() {
        return CONFIG_KEY(LOG_LEVEL);
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_LOG_LEVEL";
    }
#endif

    static LogLevel defaultValue() {
        return LogLevel::None;
    }
};

//
// PLATFORM
//

StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::VPUXPlatform val);

struct PLATFORM final : OptionBase<PLATFORM, InferenceEngine::VPUXConfigParams::VPUXPlatform> {
    static StringRef key() {
        return VPUX_CONFIG_KEY(PLATFORM);
    }

    static InferenceEngine::VPUXConfigParams::VPUXPlatform defaultValue() {
        return InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO;
    }

    static InferenceEngine::VPUXConfigParams::VPUXPlatform parse(StringRef val);

    static bool isPublic() {
        return false;
    }
};

//
// DEVICE_ID
//

struct DEVICE_ID final : OptionBase<DEVICE_ID, std::string> {
    static StringRef key() {
        return CONFIG_KEY(DEVICE_ID);
    }

    static std::string defaultValue() {
        return {};
    }
};

}  // namespace vpux
