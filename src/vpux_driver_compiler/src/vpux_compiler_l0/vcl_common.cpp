//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_common.hpp"

#include <istream>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "intel_npu/config/compiler.hpp"
#include "npu_driver_compiler.h"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "vcl_compiler.hpp"
#include "vpux/utils/IE/private_properties.hpp"

namespace {

/**
 * @name Key of build flags
 * @{
 */
constexpr std::string_view KEY_INPUTS_PRECISIONS = "--inputs_precisions";
constexpr std::string_view KEY_INPUTS_LAYOUTS = "--inputs_layouts";
constexpr std::string_view KEY_INPUTS_MODEL_LAYOUTS = "--inputs_model_layouts";
constexpr std::string_view KEY_OUTPUTS_PRECISIONS = "--outputs_precisions";
constexpr std::string_view KEY_OUTPUTS_LAYOUTS = "--outputs_layouts";
constexpr std::string_view KEY_OUTPUTS_MODEL_LAYOUTS = "--outputs_model_layouts";
/// The seperator of input output info and compilation configs
constexpr std::string_view KEY_CONFIGS = "--config";

// <option key>="<option value>"
constexpr std::string_view KEY_VALUE_SEPARATOR = "=";
constexpr std::string_view VALUE_DELIMITER = "\"";  // marks beginning and end of value
/** @} */

const std::unordered_set<std::string> SUPPORTED_LAYOUTS = {"NCDHW", "NDHWC", "NCHW", "NHWC",      "CHW",
                                                           "HWC",   "NC",    "C",    "**SCALAR**"};

}  // namespace

using namespace vpux;

namespace VPUXDriverCompiler {

/**
 * @brief Parse single option and create map to save the key and value
 *
 * @tparam T The type of value
 * @param option The content may like $KEY_INPUTS_PRECISIONS="InputName:InputPrecision ...", content depend on the type
 * of key
 * @param vclLogger The logger of current compiler
 * @param results The arrays to store parsed results
 * @param function The function to convert string value to the T type
 * @return vcl_result_t
 */
template <typename T>
vcl_result_t parseSingleOption(const std::string& option, VCLLogger* vclLogger,
                               std::unordered_map<std::string, T>& results, T (*function)(std::string, bool&)) {
    /// The content of option may like --inputs_precisions="A:fp16", the final key is A, value is
    /// ov::element::Type_t::f16
    std::size_t firstDelimPos = option.find_first_of('"');
    std::size_t lastDelimPos = option.find_last_of('"');
    /// The stream may like A:FP32 B:FP32 C:U8
    std::istringstream stream(option.substr(firstDelimPos + 1, lastDelimPos - (firstDelimPos + 1)));
    std::string elem;
    /// Not all values are legal and can be converted to special type by function
    bool matched;
    /// Parse and save value for each element
    while (stream >> elem) {
        /// ':' is the seperator of element name and element value
        std::size_t lastDelimPos = elem.find_last_of(':');
        if (lastDelimPos == std::string::npos) {
            vclLogger->outputError(formatv("Failed to find delim in option! Value: {0}", elem));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        std::string key = elem.substr(0, lastDelimPos);
        std::string val = elem.substr(lastDelimPos + 1);
        vclLogger->debug("ioInfo options - key: {0} value: {1}", key, val);
        results[key] = function(val, matched);
        if (!matched) {
            /// Return error if the setting is not in list.
            /// Support "ANY" layout and "UNSPECIFIED" precision can increase robustness.
            vclLogger->outputError(formatv("Failed to find {0} for {1}", val, key));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }
    return VCL_RESULT_SUCCESS;
}

template <typename T>
inline void myTransform(T& value) {
}

/**
 * @brief Convert string content to upper case
 */
template <>
inline void myTransform<std::string>(std::string& value) {
    std::transform(value.begin(), value.end(), value.begin(), toupper);
}

/**
 * @brief Helper function to convert value to special type in container
 *
 * @tparam KEY The type of key in container
 * @tparam VAL The type of value in container
 * @param key The key that need to be converted to value in container
 * @param matched If we can find the key in container, it is matched
 * @param con The container of the KEY and VALUE we supported
 * @param defaultValue If we do not find the key in container, we will use defaultValue for the key
 * @return VAL The matched value of the key in container
 */
template <typename KEY, typename VAL>
VAL getElementFromCon(KEY& key, bool& matched, const std::unordered_map<KEY, VAL>& con, VAL defaultValue) {
    myTransform<KEY>(key);
    const auto elem = con.find(key);
    if (elem == con.end()) {
        // For unknown value, use default value.
        matched = false;
        return defaultValue;
    } else {
        matched = true;
        return elem->second;
    }
}

BuildInfo::BuildInfo(VPUXCompilerL0* pvc): pvc(pvc), parsedConfig(pvc->getOptions()), logger(pvc->getLogger()) {
}

ov::element::Type_t BuildInfo::stringToOVPrecision(std::string value, bool& matched) {
    /// Ticket: E-88902
    /// @todo Update the map when zero backend begin to support more types
    static const std::unordered_map<std::string, ov::element::Type_t> supported_precisions = {
            {"UNSPECIFIED", ov::element::Type_t::undefined},
            {"DYNAMIC", ov::element::Type_t::dynamic},
            {"BOOL", ov::element::Type_t::boolean},
            {"BF16", ov::element::Type_t::bf16},
            {"FP16", ov::element::Type_t::f16},
            {"FP32", ov::element::Type_t::f32},
            {"FP64", ov::element::Type_t::f64},
            {"I4", ov::element::Type_t::i4},
            {"I8", ov::element::Type_t::i8},
            {"I16", ov::element::Type_t::i16},
            {"I32", ov::element::Type_t::i32},
            {"I64", ov::element::Type_t::i64},
            {"BIN", ov::element::Type_t::u1},
            {"U4", ov::element::Type_t::u4},
            {"U8", ov::element::Type_t::u8},
            {"U16", ov::element::Type_t::u16},
            {"U32", ov::element::Type_t::u32},
            {"U64", ov::element::Type_t::u64},
            {"NF4", ov::element::Type_t::nf4},
    };

    return getElementFromCon<std::string, ov::element::Type_t>(value, matched, supported_precisions,
                                                               ov::element::Type_t::undefined);
}

std::string BuildInfo::checkSupportedLayout(std::string value, bool& matched) {
    /// Update map when the supported layout changed
    if (value.find('?') != std::string::npos || value.find('.') != std::string::npos) {
        /// For partial layout, use it directly
        matched = true;
    } else {
        matched = SUPPORTED_LAYOUTS.count(value);
    }
    return value;
}

vcl_result_t BuildInfo::parseIOOption(const std::vector<std::string>& ioInfoOptions) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    /// Parse the precision && layout of input && output
    for (auto& option : ioInfoOptions) {
        if (option.find(KEY_INPUTS_PRECISIONS) != std::string::npos) {
            ret = parseSingleOption(option, logger, inputPrecisions, stringToOVPrecision);
        } else if (option.find(KEY_INPUTS_LAYOUTS) != std::string::npos) {
            ret = parseSingleOption(option, logger, inputLayouts, checkSupportedLayout);
        } else if (option.find(KEY_OUTPUTS_PRECISIONS) != std::string::npos) {
            ret = parseSingleOption(option, logger, outputPrecisions, stringToOVPrecision);
        } else if (option.find(KEY_OUTPUTS_LAYOUTS) != std::string::npos) {
            ret = parseSingleOption(option, logger, outputLayouts, checkSupportedLayout);
        } else {
            logger->outputError(formatv("Invalid key in option! Option: {0}", option));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (ret != VCL_RESULT_SUCCESS)
            return ret;
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t BuildInfo::prepareConfig(const std::string& descOptions) {
    const std::size_t configSeparator = descOptions.find(KEY_CONFIGS);

    /// If user sepecify preferred log level, update our logger
    std::string logMark = "LOG_LEVEL=\"";
    size_t start = descOptions.find(logMark);
    if (start != std::string::npos) {
        start += logMark.length();
        size_t end = descOptions.find('"', start);
        if (end != std::string::npos) {
            ov::log::Level level;
            std::stringstream{descOptions.substr(start, end - start)} >> level;
            logger->setLevel(getLogLevel(level));
        }
    }

    /// Parse the compilation options
    std::vector<std::string> options;
    if (configSeparator != std::string::npos) {
        /// Skip "--config" during parsing, The content may like:
        /// NPU_PLATFORM="3720"  NPU_COMPILATION_MODE_PARAMS="swap-transpose-with-fq=1 force-z-major-concat=1"
        std::string content = descOptions.substr(configSeparator + strlen(KEY_CONFIGS.data()));
        // From 5.0.0, compiler only support NPU_ prefix, replace VPUX_ or VPU_ with NPU_
        std::regex reg("VPUX_");
        content = std::regex_replace(content, reg, "NPU_");
        reg = "VPU_";
        content = std::regex_replace(content, reg, "NPU_");

        // As a consequence of complying to the conventions established in the 2.0 OV API, the set of values
        // corresponding to the "model priority" key has been modified. The change was introduced in the 5.2 version of
        // the driver->compiler adapter.
        const auto& getTargetRegex = [](const ov::intel_npu::LegacyPriority& priorityValue) -> std::regex {
            std::ostringstream result;
            result << ov::intel_npu::legacy_model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                   << priorityValue << VALUE_DELIMITER;
            return std::regex(result.str());
        };
        const auto& getStringReplacement = [](const ov::hint::Priority& priorityValue) -> std::string {
            std::ostringstream result;
            result << ov::hint::model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << priorityValue
                   << VALUE_DELIMITER;
            return result.str();
        };

        // E.g. (valid as of writing this): MODEL_PRIORITY="MODEL_PRIORITY_MED" -> MODEL_PRIORITY="MEDIUM"
        content = std::regex_replace(content, getTargetRegex(ov::intel_npu::LegacyPriority::LOW),
                                     getStringReplacement(ov::hint::Priority::LOW));
        content = std::regex_replace(content, getTargetRegex(ov::intel_npu::LegacyPriority::MEDIUM),
                                     getStringReplacement(ov::hint::Priority::MEDIUM));
        content = std::regex_replace(content, getTargetRegex(ov::intel_npu::LegacyPriority::HIGH),
                                     getStringReplacement(ov::hint::Priority::HIGH));

        std::stringstream input(content);

        /// A singleOption is consist of one or more words, like value of VPUX_COMPILATION_MODE_PARAMS
        std::string word;
        std::string singleOption = "";
        while (input >> word) {
            if (singleOption.compare("") == 0) {
                /// Save the first word
                singleOption = word;
            } else {
                /// Save the word that belongs to this option
                singleOption = singleOption + " " + word;
            }

            /// If this is not the last word of this option, contine for this option
            if (word[word.size() - 1] != '"') {
                continue;
            }

            /// Save current option
            options.push_back(singleOption);
            /// Clean for the next option
            singleOption = "";
        }
        /// Save the last option
        if (singleOption.compare("") != 0) {
            options.push_back(std::move(singleOption));
        }
    }

    /// Show all parsed options
    for (auto& op : options) {
        logger->debug("option : {0}", op);
    }

    /// Save the parsed configs from user
    std::map<std::string, std::string> config;
    vcl_compiler_desc_t compilerDesc = pvc->getCompilerDesc();
    vcl_device_desc_t deviceDesc = pvc->getDeviceDesc();
    if (static_cast<size_t>(deviceDesc.size) != sizeof(vcl_device_desc_t)) {
        logger->warning("Host VCL version:{0}.{1}, device VCL version:{2}.{3}", compilerDesc.version.major,
                        compilerDesc.version.minor, VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR);
    }

    /// Set default value of NPU_STEPPING property, -1u is invalid value
    if (deviceDesc.revision != static_cast<uint16_t>(-1)) {
        config[ov::intel_npu::stepping.name()] = std::to_string(deviceDesc.revision);
    }

    /// Set default value of NPU_MAX_TILES property
    config[ov::intel_npu::max_tiles.name()] = std::to_string(deviceDesc.tileCount);

    /// Pase compilation options and save to config
    /// User options will overwrite default values in config
    try {
        for (auto& option : options) {
            if (option.find_first_not_of(' ') == std::string::npos) {
                continue;
            }
            size_t length = option.size();
            /// Skip the terminator of string
            if (option[length - 1] == '\0') {
                length--;
            }

            std::size_t lastDelimPos = option.find_first_of('=');
            /// Use 2 to skip =" , the format shall follow key="value"
            if (lastDelimPos == std::string::npos || lastDelimPos + 2 > length) {
                throw std::logic_error(option + " is in bad format!");
            }
            std::string key = option.substr(0, lastDelimPos);
            /// For key="value", the val shall be value
            /// Skip =" in the front and " at the end
            config[key] = option.substr(lastDelimPos + 2, length - 1 - (lastDelimPos + 2));
            logger->debug("config options - key: {0} value: {1}", key, config[key]);
        }
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError("Internal exception in config parser!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// Foce to use MLIR compiler.
    config[ov::intel_npu::compiler_type.name()] = "MLIR";

    // If platform exists, check if it has a valid value
    if (config.find(ov::intel_npu::platform.name()) != config.end()) {
        const auto standardizedPlatform = ov::intel_npu::Platform::standardize(config[ov::intel_npu::platform.name()]);
        if (ov::intel_npu::Platform::NPU3720 != standardizedPlatform &&
            ov::intel_npu::Platform::NPU4000 != standardizedPlatform &&
            "AUTO_DETECT" != config[ov::intel_npu::platform.name()]) {
            logger->outputError("Unknown value for platform: " + config[ov::intel_npu::platform.name()]);
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }

    // Use platform information provided by driver if platform config is either not found or set on AUTO_DETECT
    if (config.find(ov::intel_npu::platform.name()) == config.end() ||
        "AUTO_DETECT" == config[ov::intel_npu::platform.name()]) {
        /// Set NPU_PLATFORM and DEVICE_ID
        /// Value from VpuFamilyId.h
        switch (deviceDesc.deviceID) {
        case 0x7D1D:
        case 0xAD1D:
            config[ov::intel_npu::platform.name()] = "3720";
            config[ov::device::id.name()] = "3720";
            break;
        case 0x643E:
            config[ov::intel_npu::platform.name()] = "4000";
            config[ov::device::id.name()] = "4000";
            break;
        default:
            logger->outputError(formatv("Unrecognized device ID! 0x{0:X}", deviceDesc.deviceID));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        };
    }

    /// When we use LOG_INFO, show vcl level profiling log
    std::map<std::string, std::string>::iterator iter = config.find(ov::log::level.name());
    if (iter != config.end()) {
        if (iter->second == "LOG_INFO")
            enableProfiling = true;
    }

    /// Update default compilation config options with the new values we parsed from user descriptions
    try {
        parsedConfig.update(config, intel_npu::OptionMode::CompileTime);
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError(formatv("Internal exception! Can not update config! DescOptions: {0}", descOptions));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    logger->debug("Current compiler desc: deviceID:{0:X}, revision:{1}, tileCount:{2}", deviceDesc.deviceID,
                  deviceDesc.revision, deviceDesc.tileCount);
    logger->debug("Final compilation configs: {0}", parsedConfig.toString());
    return VCL_RESULT_SUCCESS;
}

vcl_result_t BuildInfo::prepareBuildFlags(const std::string& descOptions) {
    /// Find the location of special separator in descOptions, the separator helps us to find input options, output
    /// options, config options
    std::size_t inputPrecisionSeparator = descOptions.find(KEY_INPUTS_PRECISIONS);
    std::size_t inputLayoutSeparator = descOptions.find(KEY_INPUTS_LAYOUTS);
    std::size_t inputModelLayoutSeparator = descOptions.find(KEY_INPUTS_MODEL_LAYOUTS);
    std::size_t outputPrecisionSeparator = descOptions.find(KEY_OUTPUTS_PRECISIONS);
    std::size_t outputLayoutSeparator = descOptions.find(KEY_OUTPUTS_LAYOUTS);
    std::size_t outputModelLayoutSeparator = descOptions.find(KEY_OUTPUTS_MODEL_LAYOUTS);
    std::size_t configSeparator = descOptions.find(KEY_CONFIGS);

    /// Parse the options for input && output
    std::vector<std::string> ioInfoOptions;
    if (inputPrecisionSeparator != std::string::npos && inputLayoutSeparator != std::string::npos &&
        outputPrecisionSeparator != std::string::npos && outputLayoutSeparator != std::string::npos) {
        /// Separate ioInfo to different section
        ioInfoOptions.push_back(descOptions.substr(inputPrecisionSeparator, inputLayoutSeparator));
        if (inputModelLayoutSeparator != std::string::npos) {
            ioInfoOptions.push_back(
                    descOptions.substr(inputLayoutSeparator, inputModelLayoutSeparator - inputLayoutSeparator));
            ioInfoOptions.push_back(descOptions.substr(inputModelLayoutSeparator,
                                                       outputPrecisionSeparator - inputModelLayoutSeparator));
        } else {
            ioInfoOptions.push_back(
                    descOptions.substr(inputLayoutSeparator, outputPrecisionSeparator - inputLayoutSeparator));
        }
        ioInfoOptions.push_back(
                descOptions.substr(outputPrecisionSeparator, outputLayoutSeparator - outputPrecisionSeparator));
        if (configSeparator != std::string::npos) {
            if (outputModelLayoutSeparator != std::string::npos) {
                ioInfoOptions.push_back(
                        descOptions.substr(outputLayoutSeparator, outputModelLayoutSeparator - outputLayoutSeparator));
                ioInfoOptions.push_back(
                        descOptions.substr(outputModelLayoutSeparator, configSeparator - outputModelLayoutSeparator));
            } else {
                ioInfoOptions.push_back(
                        descOptions.substr(outputLayoutSeparator, configSeparator - outputLayoutSeparator));
            }
        } else {
            if (outputModelLayoutSeparator != std::string::npos) {
                ioInfoOptions.push_back(
                        descOptions.substr(outputLayoutSeparator, outputModelLayoutSeparator - outputLayoutSeparator));
                ioInfoOptions.push_back(descOptions.substr(outputModelLayoutSeparator));
            } else {
                ioInfoOptions.push_back(descOptions.substr(outputLayoutSeparator));
            }
        }
    } else {
        /// Return error if the mandatory ioInfo options are not passed
        /// Skip ioInfo missing if is used for debug.
        bool skipIOInfo = false;
#if defined(VPUX_DEVELOPER_BUILD)
        if (const auto env = std::getenv("IE_VPUX_VCL_SKIP_IOINFO")) {
            skipIOInfo = std::stoi(env);
        }
#endif
        if (skipIOInfo == false) {
            logger->outputError(formatv("Mandatory ioInfo options are missing! DescOptions: {0}", descOptions));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }

    /// Parse and update config
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    try {
        ret = prepareConfig(descOptions);
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError(
                formatv("Internal exception! Can't parse and update config! DescOptions: {0}", descOptions));
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (ret != VCL_RESULT_SUCCESS) {
        logger->outputError(formatv("Failed to parse and update config! DescOptions: {0}", descOptions));
        return ret;
    }

    /// Show compiler ID which helps to find the commit of compiler
    logger->info("Current driver compiler ID: {0}", pvc->getCompilerProp().id);
    logger->info("Current build flags: {0}", descOptions);

    /// Parse precision and layout info of input && output from user
    try {
        ret = parseIOOption(ioInfoOptions);
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError(formatv("Internal exception! Can't parse ioInfo! DescOptions: {0}", descOptions));
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (ret != VCL_RESULT_SUCCESS) {
        logger->outputError(formatv("Failed to parse ioInfoOptions! DescOptions: {0}", descOptions));
        return ret;
    }
    return ret;
}

vcl_result_t BuildInfo::prepareModel(const uint8_t* modelIR, uint64_t modelIRSize) {
    if (modelIR == nullptr) {
        logger->outputError("Invalid model ir pointer!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// The API version of current compiler, adapter fill its version in modelIR, shall be same value
    vcl_version_info_t currentAPIVersion = pvc->getCompilerProp().version;
    uint32_t offset = 0;
    vcl_version_info_t APIVersion;
    memcpy(&APIVersion, modelIR, sizeof(APIVersion));
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        logger->outputError(formatv("Unsupported IR API version! Val: {0}.{1}", APIVersion.major, APIVersion.minor));
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(vcl_version_info_t);

    /// The number of elements in buffer shall not exceed limitation
    uint32_t numOfElements = 0;
    memcpy(&numOfElements, modelIR + offset, sizeof(numOfElements));
    if (numOfElements >= maxNumberOfElements) {
        logger->outputError("Bad elements number in IR!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(numOfElements);

    /// The size of model data
    uint64_t bufferSize = 0;
    memcpy(&bufferSize, modelIR + offset, sizeof(bufferSize));
    if (bufferSize == 0 || bufferSize >= maxSizeOfXML) {
        logger->outputError("Bad buffer size in IR!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(bufferSize);

    /// The offset to model xml
    uint64_t bufferOffset = offset;
    offset += bufferSize;

    /// The size of model weight
    uint64_t weightsSize = 0;
    memcpy(&weightsSize, modelIR + offset, sizeof(weightsSize));
    if (weightsSize >= maxSizeOfWeights) {
        logger->outputError("Bad weights size in IR!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(weightsSize);

    /// The offset to model weight
    uint64_t weightsOffset = offset;
    if (offset + weightsSize > modelIRSize) {
        logger->outputError("The IR content and size mismatch!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }

    /// The pointer to model xml
    const uint8_t* buffer = modelIR + bufferOffset;
    /// The pointer to model weight
    const uint8_t* weights = modelIR + weightsOffset;
    /// Deserialize the model
    size_t modelHash;
    try {
        std::string modelData(buffer, buffer + bufferSize);
        modelHash = std::hash<std::string>()(modelData);

        ov::Tensor weightsTensor;
        if (weightsSize > 0)
            weightsTensor = ov::Tensor(ov::element::u8, {weightsSize}, const_cast<uint8_t*>(weights));
        ov::Core core;
        // core needs to add internal operators via extensions
        // might need to refactor if extension list will have many elements
        core.add_extension(
                std::vector<ov::Extension::Ptr>({std::make_shared<ov::OpExtension<ov::op::internal::RMS>>(),
                                                 std::make_shared<ov::OpExtension<ov::op::internal::RoPE>>()}));

        StopWatch stopWatch;
        if (enableProfiling) {
            stopWatch.start();
        }

        model = core.read_model(modelData, weightsTensor);

        if (enableProfiling) {
            stopWatch.stop();
            logger->info("The time to convert data to model: {0} ms", stopWatch.delta_ms());
        }
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        logger->outputError("Internal exception! Could not deserialize the model!");
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    const std::string hashOption = "model-hash=" + std::to_string(modelHash);
    std::string compilationOptions = hashOption;
    if (parsedConfig.has<intel_npu::COMPILATION_MODE_PARAMS>()) {
        const auto existingOptions = parsedConfig.get<intel_npu::COMPILATION_MODE_PARAMS>();
        compilationOptions = existingOptions + " " + hashOption;
    }
    std::map<std::string, std::string> config = {{ov::intel_npu::compilation_mode_params.name(), compilationOptions}};

    /// Update default compilation config options with the new values we parsed from user descriptions
    try {
        parsedConfig.update(config, intel_npu::OptionMode::CompileTime);
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError(formatv("Internal exception! Can not update config!"));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return VCL_RESULT_SUCCESS;
}

}  // namespace VPUXDriverCompiler
