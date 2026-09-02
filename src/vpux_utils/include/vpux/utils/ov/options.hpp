//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "config.hpp"
#include "openvino/core/type/element_type.hpp"

namespace vpux {

namespace OV {

namespace Platform {

constexpr std::string_view AUTO_DETECT = "AUTO_DETECT";  // Auto detection
constexpr std::string_view NPU3720 = "3720";             // NPU3720
constexpr std::string_view NPU4000 = "4000";             // NPU4000
constexpr std::string_view NPU5010 = "5010";             // NPU5010
constexpr std::string_view NPU5020 = "5020";             // NPU5020

/**
 * @brief Converts the given platform value to the standard one.
 * @details The same platform value can be defined in multiple ways (e.g. "3720" vs "VPU3720" vs "NPU3720"). The current
 * function converts the prefixed variants to the non-prefixed ones in order to enable the comparison between platform
 * values.
 *
 * The values already found in the standard form are returned as they are.
 *
 * @param platform The value to be converted.
 * @return The same platform value given as parameter but converted to the standard form.
 */
inline std::string standardize(const std::string_view platform) {
    constexpr std::string_view VPUPrefix = "VPU";
    constexpr std::string_view NPUPrefix = "NPU";

    if (!platform.compare(0, VPUPrefix.length(), VPUPrefix) || !platform.compare(0, NPUPrefix.length(), NPUPrefix)) {
        return std::string(platform).substr(NPUPrefix.length());
    }

    return std::string(platform);
}

}  // namespace Platform

/**
 * @brief Enum to define possible performance mode hints
 */
enum class PerformanceMode {
    LATENCY = 1,                //!<  Optimize for latency
    THROUGHPUT = 2,             //!<  Optimize for throughput
    CUMULATIVE_THROUGHPUT = 3,  //!<  Optimize for cumulative throughput
};

inline std::ostream& operator<<(std::ostream& os, const PerformanceMode& performance_mode) {
    switch (performance_mode) {
    case PerformanceMode::LATENCY:
        return os << "LATENCY";
    case PerformanceMode::THROUGHPUT:
        return os << "THROUGHPUT";
    case PerformanceMode::CUMULATIVE_THROUGHPUT:
        return os << "CUMULATIVE_THROUGHPUT";
    default:
        VPUX_THROW("Unsupported performance mode hint");
    }
}

inline std::istream& operator>>(std::istream& is, PerformanceMode& performance_mode) {
    std::string str;
    is >> str;
    if (str == "LATENCY") {
        performance_mode = PerformanceMode::LATENCY;
    } else if (str == "THROUGHPUT") {
        performance_mode = PerformanceMode::THROUGHPUT;
    } else if (str == "CUMULATIVE_THROUGHPUT") {
        performance_mode = PerformanceMode::CUMULATIVE_THROUGHPUT;
    } else {
        VPUX_THROW("Unsupported performance mode: {0}", str);
    }
    return is;
}

/**
 * @brief Enum to define possible cache mode
 */
enum class CacheMode {
    OPTIMIZE_SIZE = 0,   //!< smaller cache size
    OPTIMIZE_SPEED = 1,  //!< faster loading time
};

inline std::ostream& operator<<(std::ostream& os, const CacheMode& mode) {
    switch (mode) {
    case CacheMode::OPTIMIZE_SIZE:
        return os << "optimize_size";
    case CacheMode::OPTIMIZE_SPEED:
        return os << "optimize_speed";
    default:
        VPUX_THROW("Unsupported cache mode");
    }
}

inline std::istream& operator>>(std::istream& is, CacheMode& mode) {
    std::string str;
    is >> str;
    if (str == "OPTIMIZE_SIZE" || str == "optimize_size") {
        mode = CacheMode::OPTIMIZE_SIZE;
    } else if (str == "OPTIMIZE_SPEED" || str == "optimize_speed") {
        mode = CacheMode::OPTIMIZE_SPEED;
    } else {
        VPUX_THROW("Unsupported cache mode: {0}", str);
    }
    return is;
}

/**
 * @brief [Only for NPU Plugin]
 * Type: String. Default is "AUTO".
 * This option is added for enabling batching on plugin.
 * Possible values: "AUTO", "COMPILER", "PLUGIN".
 */
enum class BatchMode {
    AUTO = 0,
    COMPILER = 1,
    PLUGIN = 2,
};

/**
 * @brief Prints a string representation of vpux::BatchMode to a stream
 * @param out An output stream to send to
 * @param fmt A value for batching on plugin to print to a stream
 * @return A reference to the `out` stream
 */
inline std::ostream& operator<<(std::ostream& out, const BatchMode& fmt) {
    switch (fmt) {
    case BatchMode::AUTO: {
        out << "AUTO";
    } break;
    case BatchMode::COMPILER: {
        out << "COMPILER";
    } break;
    case BatchMode::PLUGIN: {
        out << "PLUGIN";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for NPU Plugin]
 * Type: string
 * Type of NPU compiler to be used for compilation of a network
 */
enum class CompilerType { PLUGIN, DRIVER, PREFER_PLUGIN };

/**
 * @brief Prints a string representation of vpux::CompilerType to a stream
 * @param out An output stream to send to
 * @param fmt A compiler type value to print to a stream
 * @return A reference to the `out` stream
 */
inline std::ostream& operator<<(std::ostream& out, const CompilerType& fmt) {
    switch (fmt) {
    case CompilerType::PLUGIN: {
        out << "PLUGIN";
    } break;
    case CompilerType::DRIVER: {
        out << "DRIVER";
    } break;
    case CompilerType::PREFER_PLUGIN: {
        out << "PREFER_PLUGIN";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief Enum to define possible execution mode hints
 */
enum class ExecutionMode {
    PERFORMANCE = 1,  //!<  Optimize for max performance, may apply properties which slightly affect accuracy
    ACCURACY = 2,     //!<  Optimize for max accuracy
};

inline std::ostream& operator<<(std::ostream& os, const ExecutionMode& mode) {
    switch (mode) {
    case ExecutionMode::PERFORMANCE:
        return os << "PERFORMANCE";
    case ExecutionMode::ACCURACY:
        return os << "ACCURACY";
    default:
        VPUX_THROW("Unsupported execution mode hint");
    }
}

inline std::istream& operator>>(std::istream& is, ExecutionMode& mode) {
    std::string str;
    is >> str;
    if (str == "PERFORMANCE") {
        mode = ExecutionMode::PERFORMANCE;
    } else if (str == "ACCURACY") {
        mode = ExecutionMode::ACCURACY;
    } else {
        VPUX_THROW("Unsupported execution mode: {0}", str);
    }
    return is;
}

/**
 * @brief [Only for NPU Plugin]
 * Default is "ITERATIVE".
 * Switches between different implementations of the "weights separation" feature.
 */
enum class WSVersion {
    ONE_SHOT = 0,
    ITERATIVE = 1,
};

inline std::ostream& operator<<(std::ostream& out, const WSVersion& wsVersion) {
    switch (wsVersion) {
    case WSVersion::ONE_SHOT: {
        out << "ONE_SHOT";
    } break;
    case WSVersion::ITERATIVE: {
        out << "ITERATIVE";
    } break;
    default: {
        VPUX_THROW("Unsupported value for the weights separation version: {0}", static_cast<uint32_t>(wsVersion));
    }
    }
    return out;
}

inline std::istream& operator>>(std::istream& is, WSVersion& wsVersion) {
    std::string str;
    is >> str;
    if (str == "ONE_SHOT") {
        wsVersion = WSVersion::ONE_SHOT;
    } else if (str == "ITERATIVE") {
        wsVersion = WSVersion::ITERATIVE;
    } else {
        VPUX_THROW("Unsupported value for the weights separation version: {0}", str);
    }
    return is;
}

/**
 * @brief [Only for NPU Plugin]
 * Default is "AUTO".
 * Switches between different implementations of the VCL serializer.
 */
enum class ModelSerializerVersion {
    AUTO = 0,
    ALL_WEIGHTS_COPY = 1,
    NO_WEIGHTS_COPY = 2,
};

inline std::ostream& operator<<(std::ostream& out, const ModelSerializerVersion& modelSerializerVersion) {
    switch (modelSerializerVersion) {
    case ModelSerializerVersion::AUTO: {
        out << "AUTO";
    } break;
    case ModelSerializerVersion::ALL_WEIGHTS_COPY: {
        out << "ALL_WEIGHTS_COPY";
    } break;
    case ModelSerializerVersion::NO_WEIGHTS_COPY: {
        out << "NO_WEIGHTS_COPY";
    } break;
    default: {
        VPUX_THROW("Unsupported value for the model serializer version: {0}",
                   static_cast<uint32_t>(modelSerializerVersion));
    }
    }
    return out;
}

inline std::istream& operator>>(std::istream& is, ModelSerializerVersion& modelSerializerVersion) {
    std::string str;
    is >> str;
    if (str == "AUTO") {
        modelSerializerVersion = ModelSerializerVersion::AUTO;
    } else if (str == "ALL_WEIGHTS_COPY") {
        modelSerializerVersion = ModelSerializerVersion::ALL_WEIGHTS_COPY;
    } else if (str == "NO_WEIGHTS_COPY") {
        modelSerializerVersion = ModelSerializerVersion::NO_WEIGHTS_COPY;
    } else {
        VPUX_THROW("Unsupported value for the model serializer version: {0}", str);
    }
    return is;
}

/**
 * @brief Enum to describe the compatibility of a compiled model blob with the current device environment.
 *
 */
enum class CompatibilityCheck {
    NOT_APPLICABLE = 0,  //!< The device does not support this check, or no requirements string was provided.
    SUPPORTED = 1,       //!< Requirements are met; import is expected to succeed with optimal performance.
    UNSUPPORTED = 2,     //!< Requirements are not met; import will fail.
};

inline std::ostream& operator<<(std::ostream& os, const CompatibilityCheck& compatibility) {
    switch (compatibility) {
    case CompatibilityCheck::NOT_APPLICABLE:
        return os << "NOT_APPLICABLE";
    case CompatibilityCheck::SUPPORTED:
        return os << "SUPPORTED";
    case CompatibilityCheck::UNSUPPORTED:
        return os << "UNSUPPORTED";
    default:
        VPUX_THROW("Unsupported CompatibilityCheck value");
    }
}

inline std::istream& operator>>(std::istream& is, CompatibilityCheck& compatibility) {
    std::string str;
    is >> str;
    if (str == "NOT_APPLICABLE") {
        compatibility = CompatibilityCheck::NOT_APPLICABLE;
    } else if (str == "SUPPORTED") {
        compatibility = CompatibilityCheck::SUPPORTED;
    } else if (str == "UNSUPPORTED") {
        compatibility = CompatibilityCheck::UNSUPPORTED;
    } else {
        VPUX_THROW("Unsupported CompatibilityCheck value: {0}", str);
    }
    return is;
}

/**
 * @brief Defines the options corresponding to the legacy set of values.
 */
enum class LegacyPriority {
    LOW = 0,     //!<  Low priority
    MEDIUM = 1,  //!<  Medium priority
    HIGH = 2     //!<  High priority
};

inline std::ostream& operator<<(std::ostream& os, const LegacyPriority& priority) {
    switch (priority) {
    case LegacyPriority::LOW:
        return os << "MODEL_PRIORITY_LOW";
    case LegacyPriority::MEDIUM:
        return os << "MODEL_PRIORITY_MED";
    case LegacyPriority::HIGH:
        return os << "MODEL_PRIORITY_HIGH";
    default:
        VPUX_THROW("Unsupported model priority value");
    }
}

inline std::istream& operator>>(std::istream& is, LegacyPriority& priority) {
    std::string str;
    is >> str;
    if (str == "MODEL_PRIORITY_LOW") {
        priority = LegacyPriority::LOW;
    } else if (str == "MODEL_PRIORITY_MED") {
        priority = LegacyPriority::MEDIUM;
    } else if (str == "MODEL_PRIORITY_HIGH") {
        priority = LegacyPriority::HIGH;
    } else {
        VPUX_THROW("Unsupported model priority: {0}", str);
    }
    return is;
}

struct PERFORMANCE_HINT final : OptionBase<PERFORMANCE_HINT, PerformanceMode> {
    static std::string_view key() {
        return "PERFORMANCE_HINT";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::PerformanceMode";
    }

    static PerformanceMode defaultValue() {
        return PerformanceMode::LATENCY;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static PerformanceMode parse(std::string_view val) {
        if (val.empty()) {
            return PerformanceMode::LATENCY;
        } else if (val == "LATENCY") {
            return PerformanceMode::LATENCY;
        } else if (val == "THROUGHPUT") {
            return PerformanceMode::THROUGHPUT;
        } else if (val == "CUMULATIVE_THROUGHPUT") {
            return PerformanceMode::CUMULATIVE_THROUGHPUT;
        }

        VPUX_THROW("Value '{0}' is not a valid PERFORMANCE_HINT option", val);
    }

    static std::string toString(const PerformanceMode& val) {
        std::stringstream strStream;
        switch (val) {
        case PerformanceMode::LATENCY:
            strStream << "LATENCY";
            break;
        case PerformanceMode::THROUGHPUT:
            strStream << "THROUGHPUT";
            break;
        case PerformanceMode::CUMULATIVE_THROUGHPUT:
            strStream << "CUMULATIVE_THROUGHPUT";
            break;
        default:
            VPUX_THROW("Invalid vpux::PerformanceMode setting: {0}", static_cast<int>(val));
            break;
        }
        return strStream.str();
    }
};

struct PERFORMANCE_HINT_NUM_REQUESTS final : OptionBase<PERFORMANCE_HINT_NUM_REQUESTS, uint32_t> {
    static std::string_view key() {
        return "PERFORMANCE_HINT_NUM_REQUESTS";
    }

    /**
     * @brief Returns configuration value if it is valid, otherwise throws
     * @details This is the same function as "InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue",
     * slightly modified as to not rely on the legacy API anymore.
     * @param configuration value as string
     * @return configuration value as number
     */
    static uint32_t parse(std::string_view val) {
        int val_i = -1;
        try {
            val_i = std::stoi(val.data());
            if (val_i >= 0) {
                return val_i;
            } else {
                throw std::logic_error("wrong val");
            }
        } catch (const std::exception&) {
            VPUX_THROW("Wrong value of {0} for property key {1}. Expected only positive integer numbers", val.data(),
                       PERFORMANCE_HINT_NUM_REQUESTS::key());
        }
    }

    static uint32_t defaultValue() {
        // Default value depends on PERFORMANCE_HINT, see getOptimalNumberOfInferRequestsInParallel
        // 1 corresponds to LATENCY and default mode (hints not specified)
        return 1u;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct INFERENCE_PRECISION_HINT final : OptionBase<INFERENCE_PRECISION_HINT, ov::element::Type> {
    static std::string_view key() {
        return "INFERENCE_PRECISION_HINT";
    }

    static constexpr std::string_view getTypeName() {
        return "ov::element::Type";
    }

    static ov::element::Type defaultValue() {
        return ov::element::f16;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 4);
    }

    static bool isPublic() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static ov::element::Type parse(std::string_view val) {
        if (val.empty() || (val == "f16")) {
            return ov::element::f16;
        } else if (val == "i8") {
            return ov::element::i8;
        } else {
            VPUX_THROW("Wrong value {0} for property key {1}. Supported values: f16, i8", val.data(),
                       INFERENCE_PRECISION_HINT::key());
        }
    };
};

struct PERF_COUNT final : OptionBase<PERF_COUNT, bool> {
    static std::string_view key() {
        return "PERF_COUNT";
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct LOG_LEVEL final : OptionBase<LOG_LEVEL, LogLevel> {
    static std::string_view key() {
        return "LOG_LEVEL";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::LogLevel";
    }

    static std::string_view envVar() {
        return "OV_NPU_LOG_LEVEL";
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static LogLevel defaultValue() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
        return LogLevel::Warning;
#else
        return LogLevel::Error;
#endif
    }

    static bool isPublic() {
        return true;
    }
};

struct PLATFORM final : OptionBase<PLATFORM, std::string> {
    static std::string_view key() {
        return "NPU_PLATFORM";
    }

    static std::string defaultValue() {
        return std::string(Platform::AUTO_DETECT);
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_PLATFORM";
    }
#endif

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct DEVICE_ID final : OptionBase<DEVICE_ID, std::string> {
    static std::string_view key() {
        return "DEVICE_ID";
    }

    static std::string defaultValue() {
        return {};
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct CACHE_MODE final : OptionBase<CACHE_MODE, CacheMode> {
    static std::string_view key() {
        return "CACHE_MODE";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::CacheMode";
    }

    static CacheMode defaultValue() {
        return CacheMode::OPTIMIZE_SPEED;
    }

    static bool isPublic() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static CacheMode parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        CacheMode cacheMode;
        stringStream >> cacheMode;
        return cacheMode;
    }

    static std::string toString(const CacheMode& val) {
        std::stringstream strStream;
        strStream << val;
        return strStream.str();
    }
};

// BATCH_MODE is required to maintain backward/forward compatibility
struct BATCH_MODE final : OptionBase<BATCH_MODE, BatchMode> {
    static std::string_view key() {
        return "NPU_BATCH_MODE";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::BatchMode";
    }

    static BatchMode defaultValue() {
        return BatchMode::AUTO;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 5);
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static BatchMode parse(std::string_view val) {
        if (val == "AUTO") {
            return BatchMode::AUTO;
        } else if (val == "COMPILER") {
            return BatchMode::COMPILER;
        } else if (val == "PLUGIN") {
            return BatchMode::PLUGIN;
        }

        VPUX_THROW("Value '{0}' is not a valid BATCH_MODE option", val);
    }

    static std::string toString(const BatchMode& val) {
        std::stringstream strStream;

        strStream << val;

        return strStream.str();
    }
};

struct MODEL_PRIORITY final : OptionBase<MODEL_PRIORITY, LegacyPriority> {
    static std::string_view key() {
        return "MODEL_PRIORITY";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::OV::LegacyPriority";
    }

    static LegacyPriority defaultValue() {
        return LegacyPriority::MEDIUM;
    }

    static LegacyPriority parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        LegacyPriority priority;

        stringStream >> priority;

        return priority;
    }

    static std::string toString(const LegacyPriority& val) {
        std::ostringstream stringStream;

        stringStream << val;

        return stringStream.str();
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct TURBO final : OptionBase<TURBO, bool> {
    static std::string_view key() {
        return "NPU_TURBO";
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 21);
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct COMPILER_TYPE final : OptionBase<COMPILER_TYPE, CompilerType> {
    static std::string_view key() {
        return "NPU_COMPILER_TYPE";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::CompilerType";
    }

    static CompilerType defaultValue() {
        return CompilerType::PREFER_PLUGIN;
    }

    static std::string_view envVar() {
#ifdef VPUX_DEVELOPER_BUILD
        return "IE_NPU_COMPILER_TYPE";
#else
        return "";
#endif
    }

    static CompilerType parse(std::string_view val) {
        if (val == "PLUGIN") {
            return CompilerType::PLUGIN;
        } else if (val == "DRIVER") {
            return CompilerType::DRIVER;
        } else if (val == "PREFER_PLUGIN") {
            return CompilerType::PREFER_PLUGIN;
        }

        VPUX_THROW("Value '{0}' is not a valid COMPILER_TYPE option", val);
    }

    static std::string toString(const CompilerType& val) {
        std::stringstream strStream;
        if (val == CompilerType::PLUGIN) {
            strStream << "PLUGIN";
        } else if (val == CompilerType::DRIVER) {
            strStream << "DRIVER";
        } else if (val == CompilerType::PREFER_PLUGIN) {
            strStream << "PREFER_PLUGIN";
        } else {
            VPUX_THROW("No valid string for current COMPILER_TYPE option: {0}", static_cast<int>(val));
        }

        return strStream.str();
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct COMPILATION_MODE final : OptionBase<COMPILATION_MODE, std::string> {
    static std::string_view key() {
        return "NPU_COMPILATION_MODE";
    }

    static std::string defaultValue() {
        return "";
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_COMPILATION_MODE";
    }
#endif

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct EXECUTION_MODE_HINT final : OptionBase<EXECUTION_MODE_HINT, ExecutionMode> {
    static std::string_view key() {
        return "EXECUTION_MODE_HINT";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::ExecutionMode";
    }

    static ExecutionMode defaultValue() {
        return ExecutionMode::PERFORMANCE;
    }

    static ExecutionMode parse(std::string_view val) {
        std::string strVal(val);
        std::istringstream is(strVal);
        ExecutionMode mode;
        is >> mode;
        return mode;
    }

    static std::string toString(const ExecutionMode& val) {
        std::ostringstream os;
        os << val;
        return os.str();
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 6);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct DYNAMIC_SHAPE_TO_STATIC final : OptionBase<DYNAMIC_SHAPE_TO_STATIC, bool> {
    static std::string_view key() {
        return "NPU_DYNAMIC_SHAPE_TO_STATIC";
    }

    static bool defaultValue() {
        return false;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DYNAMIC_SHAPE_TO_STATIC";
    }
#endif

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct COMPILATION_MODE_PARAMS final : OptionBase<COMPILATION_MODE_PARAMS, std::string> {
    static std::string_view key() {
        return "NPU_COMPILATION_MODE_PARAMS";
    }

    static std::string defaultValue() {
        return {};
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct TILES final : OptionBase<TILES, int64_t> {
    static std::string_view key() {
        return "NPU_TILES";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 4);
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_TILES";
    }
#endif
};

struct STEPPING final : OptionBase<STEPPING, int64_t> {
    static std::string_view key() {
        return "NPU_STEPPING";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 3);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct MAX_TILES final : OptionBase<MAX_TILES, int64_t> {
    static std::string_view key() {
        return "NPU_MAX_TILES";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 3);
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct DMA_ENGINES final : OptionBase<DMA_ENGINES, int64_t> {
    static std::string_view key() {
        return "NPU_DMA_ENGINES";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

#ifdef VPUX_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DMA_ENGINES";
    }
#endif
};

struct BACKEND_COMPILATION_PARAMS final : OptionBase<BACKEND_COMPILATION_PARAMS, std::string> {
    static std::string_view key() {
        return "NPU_BACKEND_COMPILATION_PARAMS";
    }

    static std::string defaultValue() {
        return {};
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct COMPILATION_NUM_THREADS final : OptionBase<COMPILATION_NUM_THREADS, int32_t> {
    static std::string_view key() {
        return "COMPILATION_NUM_THREADS";
    }

    static int32_t defaultValue() {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    static void validateValue(const int32_t& num) {
        if (num <= 0) {
            VPUX_THROW("{0} must be positive int32 value", COMPILATION_NUM_THREADS::key());
        }
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(0, 0);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct COMPILER_DYNAMIC_QUANTIZATION final : OptionBase<COMPILER_DYNAMIC_QUANTIZATION, bool> {
    static std::string_view key() {
        return "NPU_COMPILER_DYNAMIC_QUANTIZATION";
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 1);
    }

    static bool isPublic() {
        return true;
    }
};

struct QDQ_OPTIMIZATION final : OptionBase<QDQ_OPTIMIZATION, bool> {
    static std::string_view key() {
        return "NPU_QDQ_OPTIMIZATION";
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 20);
    }
};

struct QDQ_OPTIMIZATION_AGGRESSIVE final : OptionBase<QDQ_OPTIMIZATION_AGGRESSIVE, bool> {
    static std::string_view key() {
        return "NPU_QDQ_OPTIMIZATION_AGGRESSIVE";
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct BATCH_COMPILER_MODE_SETTINGS final : OptionBase<BATCH_COMPILER_MODE_SETTINGS, std::string> {
    static std::string_view key() {
        return "NPU_BATCH_COMPILER_MODE_SETTINGS";
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 4);
    }

    static bool isPublic() {
        return false;
    }
};

struct ENABLE_WEIGHTLESS final : OptionBase<ENABLE_WEIGHTLESS, bool> {
    static std::string_view key() {
        return "ENABLE_WEIGHTLESS";
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct SEPARATE_WEIGHTS_VERSION final : OptionBase<SEPARATE_WEIGHTS_VERSION, WSVersion> {
    static std::string_view key() {
        return "NPU_SEPARATE_WEIGHTS_VERSION";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::WSVersion";
    }

    static WSVersion defaultValue() {
        // Note: if the compiler-in-plugin is used (intel_npu::compiler_type = intel_npu::CompilerType::PLUGIN), then
        // the default is actually WSVersion::ONE_SHOT
        return WSVersion::ITERATIVE;
    }

    static WSVersion parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        WSVersion wsVersion;
        stringStream >> wsVersion;
        return wsVersion;
    }

    static std::string toString(const WSVersion& val) {
        std::stringstream strStream;
        strStream << val;
        return strStream.str();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct WS_COMPILE_CALL_NUMBER final : OptionBase<WS_COMPILE_CALL_NUMBER, uint32_t> {
    static std::string_view key() {
        return "WS_COMPILE_CALL_NUMBER";
    }

    static uint32_t defaultValue() {
        return 0;
    }

    static uint32_t parse(std::string_view val) {
        int val_i = -1;
        try {
            val_i = std::stoi(val.data());
            if (val_i >= 0) {
                return val_i;
            } else {
                throw std::logic_error("wrong val");
            }
        } catch (const std::exception&) {
            VPUX_THROW("Wrong value of '{0}' for property key '{1}'. Expected only positive integer numbers",
                       val.data(), WS_COMPILE_CALL_NUMBER::key());
        }
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct MODEL_SERIALIZER_VERSION final : OptionBase<MODEL_SERIALIZER_VERSION, ModelSerializerVersion> {
    static std::string_view key() {
        return "NPU_MODEL_SERIALIZER_VERSION";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::ModelSerializerVersion";
    }

    static ModelSerializerVersion defaultValue() {
        return ModelSerializerVersion::AUTO;
    }

    static ModelSerializerVersion parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        ModelSerializerVersion version;
        stringStream >> version;
        return version;
    }

    static std::string toString(const ModelSerializerVersion& val) {
        std::stringstream strStream;
        strStream << val;
        return strStream.str();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct ENABLE_STRIDES_FOR final : OptionBase<ENABLE_STRIDES_FOR, std::string> {
    static std::string_view key() {
        return "NPU_ENABLE_STRIDES_FOR";
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

struct COMPATIBILITY_CHECK final : OptionBase<COMPATIBILITY_CHECK, CompatibilityCheck> {
    static std::string_view key() {
        return "COMPATIBILITY_CHECK";
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::CompatibilityCheck";
    }

    static CompatibilityCheck defaultValue() {
        return CompatibilityCheck::NOT_APPLICABLE;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static CompatibilityCheck parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        CompatibilityCheck check_result;
        stringStream >> check_result;

        return check_result;
    }

    static std::string toString(const CompatibilityCheck& val) {
        std::ostringstream stringStream;
        stringStream << val;

        return stringStream.str();
    }

    static bool isPublic() {
        return true;
    }
};

struct EXTENSION_LIB_PATH final : OptionBase<EXTENSION_LIB_PATH, std::string> {
    static std::string_view key() {
        return "OV_EXTENSION_LIB_PATH";
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct CUSTOM_KERNEL_CONFIG_PATH final : OptionBase<CUSTOM_KERNEL_CONFIG_PATH, std::string> {
    static std::string_view key() {
        return "OV_CUSTOM_KERNEL_CONFIG_PATH";
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

}  // namespace OV

}  // namespace vpux
