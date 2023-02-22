//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/optional.hpp"

#include "vpux/properties.hpp"
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include <ie_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>

namespace InferenceEngine {

namespace VPUXConfigParams {

llvm::StringLiteral stringifyEnum(InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg val);

}  // namespace VPUXConfigParams

}  // namespace InferenceEngine

namespace vpux {

//
// register
//

void registerRunTimeOptions(OptionsDesc& desc);

//
// EXCLUSIVE_ASYNC_REQUESTS
//

struct EXCLUSIVE_ASYNC_REQUESTS final : OptionBase<EXCLUSIVE_ASYNC_REQUESTS, bool> {
    static StringRef key() {
        return CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS);
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// INFERENCE_SHAVES
//

struct INFERENCE_SHAVES final : OptionBase<INFERENCE_SHAVES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::inference_shaves.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {"VPUX_VPUAL_INFERENCE_SHAVES"};
    }

    static int64_t defaultValue() {
        return 0;
    }

    static void validateValue(int64_t v) {
        VPUX_THROW_UNLESS(0 <= v && v <= 16,
                          "Attempt to set invalid number of shaves for NnCore: '{0}', valid numbers are from 0 to 16",
                          v);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// CSRAM_SIZE
//

struct CSRAM_SIZE final : OptionBase<CSRAM_SIZE, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::csram_size.name();
    }

    static int64_t defaultValue() {
        return 0;
    }

    static void validateValue(int64_t v) {
        constexpr Byte MAX_CSRAM_SIZE = 1_GB;

        VPUX_THROW_UNLESS(v >= -1 && v <= MAX_CSRAM_SIZE.count(),
                          "Attempt to set invalid CSRAM size in bytes: '{0}', valid values are -1, 0 and up to 1 Gb",
                          v);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// GRAPH_COLOR_FORMAT
//

struct GRAPH_COLOR_FORMAT final : OptionBase<GRAPH_COLOR_FORMAT, InferenceEngine::ColorFormat> {
    static StringRef key() {
        return ov::intel_vpux::graph_color_format.name();
    }

    static InferenceEngine::ColorFormat defaultValue() {
        return InferenceEngine::ColorFormat::BGR;
    }

    static InferenceEngine::ColorFormat parse(StringRef val);

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PREPROCESSING_SHAVES
//

struct PREPROCESSING_SHAVES final : OptionBase<PREPROCESSING_SHAVES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::preprocessing_shaves.name();
    }

    static int64_t defaultValue() {
        return 4;
    }

    static void validateValue(int64_t v) {
        VPUX_THROW_UNLESS(0 <= v && v <= 16,
                          "Attempt to set invalid number of shaves for SIPP: '{0}', valid numbers are from 0 to 16", v);
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PREPROCESSING_LPI
//

struct PREPROCESSING_LPI final : OptionBase<PREPROCESSING_LPI, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::preprocessing_lpi.name();
    }

    static int64_t defaultValue() {
        return 8;
    }

    static void validateValue(int64_t v) {
        VPUX_THROW_UNLESS(0 < v && v <= 16 && isPowerOfTwo(v),
                          "Attempt to set invalid lpi value for SIPP: '{0}',  valid values are 1, 2, 4, 8, 16", v);
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PREPROCESSING_PIPES
//

struct PREPROCESSING_PIPES final : OptionBase<PREPROCESSING_PIPES, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::preprocessing_pipes.name();
    }

    static int64_t defaultValue() {
        return 1;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// USE_SIPP
//

struct USE_SIPP final : OptionBase<USE_SIPP, bool> {
    static StringRef key() {
        return ov::intel_vpux::use_sipp.name();
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {};
    }

    static bool defaultValue() {
        return true;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

int64_t getNumOptimalInferRequests(const Config& config);
//
// INFERENCE_TIMEOUT_MS
//

struct INFERENCE_TIMEOUT_MS final : OptionBase<INFERENCE_TIMEOUT_MS, int64_t> {
    static StringRef key() {
        return ov::intel_vpux::inference_timeout.name();
    }

    static int64_t defaultValue() {
        // 5 seconds -> milliseconds
        return 5 * 1000;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// PRINT_PROFILING
//

struct PRINT_PROFILING final : OptionBase<PRINT_PROFILING, InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg> {
    static StringRef key() {
        return ov::intel_vpux::print_profiling.name();
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg defaultValue() {
        return cvtProfilingOutputType(ov::intel_vpux::ProfilingOutputTypeArg::NONE);
    }

    static InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg parse(StringRef val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static StringRef envVar() {
        return "IE_VPUX_PRINT_PROFILING";
    }
#endif
};

struct PROFILING_OUTPUT_FILE final : OptionBase<PROFILING_OUTPUT_FILE, std::string> {
    static StringRef key() {
        return ov::intel_vpux::profiling_output_file.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// MODEL_PRIORITY
//

struct MODEL_PRIORITY final : OptionBase<MODEL_PRIORITY, ov::hint::Priority> {
    static StringRef key() {
        return ov::hint::model_priority.name();
    }

    static ov::hint::Priority defaultValue() {
        return ov::hint::Priority::MEDIUM;
    }

    static ov::hint::Priority parse(StringRef val);

    static std::string toString(const ov::hint::Priority& val);

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

}  // namespace vpux
