//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/model_preprocessor.hpp"

#include "vpux/compiler/frontend/xml_deserializer.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/private_properties.hpp"

#include <openvino/core/op_extension.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/rt_info/weightless_caching_attributes.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/group_query_attention.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include <intel_npu/config/options.hpp>
#include <intel_npu/ops/flash_attention_tile.hpp>

#include <regex>
#include <utility>

namespace vpux {

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

const std::string KEY_IR_VERSION = "version";
const std::string KEY_USE_INDICES_FOR_IO_METADATA = "use_indices_for_io_metadata";

const std::unordered_set<std::string> SUPPORTED_LAYOUTS = {"NCDHW", "NDHWC", "NCHW", "NHWC",      "CHW",
                                                           "HWC",   "NC",    "C",    "**SCALAR**"};

constexpr int64_t OLDEST_IR_VERSION_SUPPORTED = 10;

/**
 * @brief Parse single option and create map to save the key and value
 *
 * @tparam T The type of value
 * @param option The content may like $KEY_INPUTS_PRECISIONS="InputName:InputPrecision ...", content depend on the type
 * of key
 * @param function The function to convert string value to the T type
 * @return Map of option key and value
 */
template <typename T>
std::unordered_map<std::string, T> parseSingleOption(const std::string& option,
                                                     T (*function)(const std::string&, bool&)) {
    /// The content of option may like --inputs_precisions="A:fp16", the final key is A, value is
    /// ov::element::Type_t::f16
    std::size_t firstDelimPos = option.find_first_of('"');
    std::size_t lastDelimPos = option.find_last_of('"');
    /// The stream may like A:FP32 B:FP32 C:U8
    std::istringstream stream(option.substr(firstDelimPos + 1, lastDelimPos - (firstDelimPos + 1)));
    std::string elem;
    /// Not all values are legal and can be converted to special type by function
    bool matched = false;
    std::unordered_map<std::string, T> results;
    /// Parse and save value for each element
    while (stream >> elem) {
        /// ':' is the seperator of element name and element value
        std::size_t lastDelimPos = elem.find_last_of(':');
        if (lastDelimPos == std::string::npos) {
            VPUX_THROW("Failed to find delim in option! Value: {0}", elem);
        }
        std::string key = elem.substr(0, lastDelimPos);
        std::string val = elem.substr(lastDelimPos + 1);
        Logger::global().debug("ioInfo options - key: {0} value: {1}", key, val);
        results[key] = function(val, matched);
        if (!matched) {
            /// Return error if the setting is not in list.
            /// Support "ANY" layout and "UNSPECIFIED" precision can increase robustness.
            VPUX_THROW("Failed to find {0} for {1}", val, key);
        }
    }
    return results;
}

template <typename T>
inline void myTransform(T& /*value*/) {
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
VAL getElementFromCon(KEY key, bool& matched, const std::unordered_map<KEY, VAL>& con, VAL defaultValue) {
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

/**
 * @brief Return the valid tile value based on user config and deviceDesc
 *
 * @param config The passed user config
 * @param deviceDesc The default compiler info
 * @return std::string
 */
std::string getValidTileValue(std::map<std::string, std::string>& config, vcl_device_desc_t deviceDesc) {
    auto isValidTileConfig = [](const std::string& v) {
        return v[0] != '-' && v != "0";
    };
    // If maxTile in the user config is valid, use the smaller value between user config and deviceDesc
    if (config.find(ov::intel_npu::max_tiles.name()) != config.end() &&
        isValidTileConfig(config[ov::intel_npu::max_tiles.name()])) {
        std::string validTileValue =
                static_cast<uint32_t>(std::stoi(config[ov::intel_npu::max_tiles.name()])) < deviceDesc.tileCount
                        ? config[ov::intel_npu::max_tiles.name()]
                        : std::to_string(deviceDesc.tileCount);
        return validTileValue;
    }

    // If maxTile does not exist in the user config, use the value from deviceDesc
    return std::to_string(deviceDesc.tileCount);
}

/**
 * @brief Restore the original WeightlessCache attributes from the model.
 *
 * @note The WeightlessCache attribute is used to mark original constants. The IR frontend always populates the OV model
 * with the WeightlessCache attribute. But since the model can be changed before it is passed to the driver, the new
 * attributes will not match the original model when deserialised.
 *
 * @param model The OV model passed for compilation
 * @param vclLogger The logger of current compiler
 */
void restoreWeightsOffsets(const std::shared_ptr<ov::Model>& model, Logger& log) {
    // E#103359: replace log level with trace
    log.info("Remove all \"fake\" WeightlessCache attributes that appeared after second deserialization of the "
             "model");
    log.info("And restore original WeightlessCache attributes");

    ov::RTMap& modelRuntimeInfoMap = model->get_rt_info();

    size_t constantId = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ov::RTMap& nodeRuntimeInfoMap = node->get_rt_info();
            const auto& nodeWeightlessCacheAttrIt =
                    nodeRuntimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());
            if (nodeWeightlessCacheAttrIt != nodeRuntimeInfoMap.end()) {
                // Delete the "fake" value
                nodeRuntimeInfoMap.erase(nodeWeightlessCacheAttrIt);
            }

            const std::string constantIdString = std::to_string(constantId++);
            const auto& modelBinOffsetIt = modelRuntimeInfoMap.find("ws_bin_offset_" + constantIdString);
            const auto& modelOriginalSizeIt = modelRuntimeInfoMap.find("ws_original_size_" + constantIdString);
            const auto& modelOriginalDtypeIt = modelRuntimeInfoMap.find("ws_original_dtype_" + constantIdString);
            if (modelBinOffsetIt != modelRuntimeInfoMap.end() && modelOriginalSizeIt != modelRuntimeInfoMap.end() &&
                modelOriginalDtypeIt != modelRuntimeInfoMap.end()) {
                // Restore the bin offset value
                nodeRuntimeInfoMap[ov::WeightlessCacheAttribute::get_type_info_static()] = ov::WeightlessCacheAttribute(
                        modelOriginalSizeIt->second.as<size_t>(), modelBinOffsetIt->second.as<size_t>(),
                        modelOriginalDtypeIt->second.as<ov::element::Type>());
            }
        }
    }
}

ov::element::Type_t stringToOVPrecision(const std::string& value, bool& matched) {
    /// Ticket: E-88902
    /// @todo Update the map when zero backend begin to support more types
    static const std::unordered_map<std::string, ov::element::Type_t> supportedPrecisions = {
            {"DYNAMIC", ov::element::Type_t::dynamic}, {"BOOL", ov::element::Type_t::boolean},
            {"FP8_E4M3", ov::element::Type_t::f8e4m3}, {"FP8_E5M2", ov::element::Type_t::f8e5m2},
            {"FP8_E8M0", ov::element::Type_t::f8e8m0}, {"BF16", ov::element::Type_t::bf16},
            {"FP16", ov::element::Type_t::f16},        {"FP32", ov::element::Type_t::f32},
            {"FP64", ov::element::Type_t::f64},        {"I4", ov::element::Type_t::i4},
            {"I8", ov::element::Type_t::i8},           {"I16", ov::element::Type_t::i16},
            {"I32", ov::element::Type_t::i32},         {"I64", ov::element::Type_t::i64},
            {"BIN", ov::element::Type_t::u1},          {"U2", ov::element::Type_t::u2},
            {"U4", ov::element::Type_t::u4},           {"U8", ov::element::Type_t::u8},
            {"U16", ov::element::Type_t::u16},         {"U32", ov::element::Type_t::u32},
            {"U64", ov::element::Type_t::u64},         {"NF4", ov::element::Type_t::nf4},
    };

    return getElementFromCon<std::string, ov::element::Type_t>(value, matched, supportedPrecisions,
                                                               ov::element::Type_t::dynamic);
}

std::string checkSupportedLayout(const std::string& value, bool& matched) {
    /// Update map when the supported layout changed
    if (value.find('?') != std::string::npos || value.find('.') != std::string::npos) {
        /// For partial layout, use it directly
        matched = true;
    } else {
        matched = SUPPORTED_LAYOUTS.count(value);
    }
    return value;
}

std::pair<Precisions, Layouts> parseIOOption(const std::vector<std::string>& ioInfoOptions) {
    Precisions precisions;
    Layouts layouts;
    /// Parse the precision && layout of input && output
    for (auto& option : ioInfoOptions) {
        if (option.find(KEY_INPUTS_PRECISIONS) != std::string::npos) {
            precisions.inputPrecisions = parseSingleOption(option, stringToOVPrecision);
        } else if (option.find(KEY_INPUTS_LAYOUTS) != std::string::npos) {
            layouts.inputLayouts = parseSingleOption(option, checkSupportedLayout);
        } else if (option.find(KEY_OUTPUTS_PRECISIONS) != std::string::npos) {
            precisions.outputPrecisions = parseSingleOption(option, stringToOVPrecision);
        } else if (option.find(KEY_OUTPUTS_LAYOUTS) != std::string::npos) {
            layouts.outputLayouts = parseSingleOption(option, checkSupportedLayout);
        } else {
            VPUX_THROW("Invalid key in option! Option: {0}", option);
        }
    }
    return {precisions, layouts};
}

std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "SCALAR";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

}  // namespace

void prepareConfig(const std::string& descOptions, const vcl_compiler_desc_t& compilerDesc,
                   const vcl_device_desc_t& deviceDesc, intel_npu::Config& config, bool isDeviceDescEmpty) {
    Logger log = Logger::global();
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
            log.setLevel(getLogLevel(level));
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
        // Replace NPU_DPU_GROUPS with NPU_TILES since NPU_DPU_GROUPS is deprecated in OV
        content = std::regex_replace(content, std::regex("NPU_DPU_GROUPS"), "NPU_TILES");

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
        log.debug("option : {0}", op);
    }

    /// Save the parsed configs from user
    std::map<std::string, std::string> configOptions;
    if (!isDeviceDescEmpty) {
        if (static_cast<size_t>(deviceDesc.size) != sizeof(vcl_device_desc_t)) {
            log.warning("Host VCL version:{0}.{1}, device VCL version:{2}.{3}", compilerDesc.version.major,
                        compilerDesc.version.minor, VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR);
        }

        /// Set default value of NPU_STEPPING property, -1u is invalid value
        if (deviceDesc.revision != static_cast<uint16_t>(-1)) {
            configOptions[ov::intel_npu::stepping.name()] = std::to_string(deviceDesc.revision);
        }
    }

    /// Parse compilation options and save to config
    /// User options will overwrite default values in config
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
        configOptions[key] = option.substr(lastDelimPos + 2, length - 1 - (lastDelimPos + 2));
        log.debug("config options - key: {0} value: {1}", key, configOptions[key]);
    }

    // If platform exists, check if it has a valid value
    auto iter = configOptions.find(ov::intel_npu::platform.name());
    if (iter != configOptions.end()) {
        const auto standardizedPlatform =
                ov::intel_npu::Platform::standardize(configOptions[ov::intel_npu::platform.name()]);

        if (standardizedPlatform == ov::intel_npu::Platform::AUTO_DETECT) {
            // Platform is already auto-detected in VCL based on deviceID, keeping NPU_PLATFORM=AUTO_DETECT for
            // backwards comaptibility with old OV versions.
            VPUX_THROW_WHEN(!config.has(ov::intel_npu::platform.name()),
                            "Auto-detect platform is not supported when device description is not set. Please specify "
                            "the platform in config.");
            // Do not override platform detected in VCL with AUTO_DETECT
            configOptions.erase(iter);
        } else {
            const bool isSupportedPlatform = standardizedPlatform == ov::intel_npu::Platform::NPU3720 ||
                                             standardizedPlatform == ov::intel_npu::Platform::NPU4000 ||
                                             standardizedPlatform == ov::intel_npu::Platform::NPU5010 ||
                                             standardizedPlatform == ov::intel_npu::Platform::NPU5020;

            VPUX_THROW_WHEN(!isSupportedPlatform, "Unknown value for platform: {0}",
                            configOptions[ov::intel_npu::platform.name()]);
        }
    }

    /// Update maxTiles config with compiler desc
    ///  - If deviceDesc is valid, compare its tileCount with user config and use the smaller value
    ///  - If deviceDesc is empty, its tileCount is invalid, then just handle it according to the config
    ///  - If deviceDesc's tileCount is invalid, then skip updating maxTiles
    if (!isDeviceDescEmpty && (deviceDesc.tileCount != static_cast<uint32_t>(-1))) {
        configOptions[ov::intel_npu::max_tiles.name()] = getValidTileValue(configOptions, deviceDesc);
        log.debug("NPU_MAX_TILES is updated to {0}", configOptions[ov::intel_npu::max_tiles.name()]);
    } else {
        log.debug("DeviceDesc is empty or tileCount is invalid, skip updating NPU_MAX_TILES");
    }

    /// Remove runtime option MODEL_PRIORITY passed by plugin versions <= OV25.1
    iter = configOptions.find(ov::hint::model_priority.name());
    if (iter != configOptions.end()) {
        configOptions.erase(iter);
    }

    /// Remove runtime option CACHE_DIR passed by plugin versions <= OV25.1
    iter = configOptions.find(ov::cache_dir.name());
    if (iter != configOptions.end()) {
        configOptions.erase(iter);
    }

    /// Update default compilation config options with the new values we parsed from user descriptions
    config.update(configOptions);

    if (!isDeviceDescEmpty) {
        log.debug("Current compiler desc: deviceID:{0:X}, revision:{1}, tileCount:{2}", deviceDesc.deviceID,
                  deviceDesc.revision, deviceDesc.tileCount);
    } else {
        log.debug("Current compiler desc is empty! No default setting for platform, deviceid, stepping and tile "
                  "config. If you need to configure them, please set them in compilation config.");
    }
    log.debug("Final compilation configs: {0}", config.toString());
}

std::pair<Precisions, Layouts> prepareBuildFlags(const std::string& descOptions,
                                                 const vcl_compiler_desc_t& compilerDesc,
                                                 const vcl_compiler_properties_t& compilerProp,
                                                 const vcl_device_desc_t& deviceDesc, intel_npu::Config& config,
                                                 bool isDeviceDescEmpty) {
    Logger log = Logger::global();
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
        log.warning("The ioInfo options are missing! DescOptions: {0}", descOptions);
    }

    /// Parse and update config
    prepareConfig(descOptions, compilerDesc, deviceDesc, config, isDeviceDescEmpty);

    /// Show compiler ID which helps to find the commit of compiler
    log.info("Current compiler ID: {0}", compilerProp.id);
    log.info("Current build flags: {0}", descOptions);

    return parseIOOption(ioInfoOptions);
}

std::shared_ptr<ov::Model> prepareModel(const uint8_t* modelIR, uint64_t modelIRSize, intel_npu::Config& config,
                                        const vcl_compiler_properties_t& compilerProp) {
    Logger log = Logger::global();

    const vcl_version_info_t currentAPIVersion = compilerProp.version;
    const std::vector<ov::Extension::Ptr> extensionsVector{
            std::make_shared<ov::OpExtension<ov::op::internal::RMS>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::RoPE>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::GroupQueryAttention>>(),
            std::make_shared<ov::OpExtension<ov::intel_npu::op::FlashAttentionTile>>()};

    if (!config.has<intel_npu::MODEL_SERIALIZER_VERSION>()) {
        // Plugin versions older than the "model_serializer_version" config option won't specify a serialization
        // algorithm. The default is "AUTO" which results in an error in the code below. In this case, the older
        // deserializer ("all weights copy") should be used.
        config.update({{ov::intel_npu::model_serializer_version.name(), "ALL_WEIGHTS_COPY"}});
    }

    std::shared_ptr<ov::Model> model;
    std::ostringstream errorMessage;

    switch (config.get<intel_npu::MODEL_SERIALIZER_VERSION>()) {
    case ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY:
        model = deserializeIrModelBase(const_cast<uint8_t*>(modelIR), modelIRSize, currentAPIVersion, extensionsVector);
        break;
    case ov::intel_npu::ModelSerializerVersion::NO_WEIGHTS_COPY:
        model = deserializeIrModelOptimized(const_cast<uint8_t*>(modelIR), modelIRSize, currentAPIVersion,
                                            extensionsVector);
        break;
    case ov::intel_npu::ModelSerializerVersion::AUTO:
        errorMessage << "The driver-compiler adapter received an unsupported value for the "
                        "\"ov::intel_npu::model_serializer_version\" config option. Received: AUTO. AUTO can only "
                        "be used by the NPU plugin to allow it to choose a fitting version of the model "
                        "marshalling algorithm. This adapter requires the version to be specified explicitly. "
                        "Supported values: "
                     << ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY << ", "
                     << ov::intel_npu::ModelSerializerVersion::NO_WEIGHTS_COPY << ".";
        throw std::invalid_argument(errorMessage.str());
    default:
        errorMessage << "The driver-compiler adapter received an unsupported value for the "
                        "\"ov::intel_npu::model_serializer_version\" config option. Received: "
                     << config.get<intel_npu::MODEL_SERIALIZER_VERSION>()
                     << ". Supported: " << ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY << ", "
                     << ov::intel_npu::ModelSerializerVersion::NO_WEIGHTS_COPY << ".";
        throw std::invalid_argument(errorMessage.str());
    }

    restoreWeightsOffsets(model, log);

    return model;
}

void preprocessModel(const std::shared_ptr<ModelData>& modelData) {
    VPUX_THROW_WHEN(!modelData, "preprocessModel: modelData pointer is null");
    VPUX_THROW_WHEN(!modelData->model, "preprocessModel: model is null, prepareModel must be called first");
    Logger log = Logger::global();
    auto& runtimeInfoMap = modelData->model->get_rt_info();

    int64_t irVersion = OLDEST_IR_VERSION_SUPPORTED;
    if (const auto irVersionMatch = runtimeInfoMap.find(KEY_IR_VERSION); irVersionMatch != runtimeInfoMap.end()) {
        irVersion = irVersionMatch->second.as<int64_t>();
    }

    bool useIndices = false;
    if (const auto useIndicesMatch = runtimeInfoMap.find(KEY_USE_INDICES_FOR_IO_METADATA);
        useIndicesMatch != runtimeInfoMap.end()) {
        useIndices = useIndicesMatch->second.as<bool>();
    }

    log.debug("useIndices is {0}, using {1} for parameter/result node identification", useIndices,
              useIndices ? "indices" : "names");

    // Compiler needs to maintain compatibility with older OpenVINO (plugin) versions.
    // Compiler needs to support applications that are still using OV 1.0 API.
    //     | OV API Version | IR version | Needs preprocessing? |
    //     | 1.0            | v10        | Yes                  |
    //     | 2.0            | v10        | Yes                  |
    //     | 1.0            | v11        | Invalid usecase      |
    //     | 2.0            | v11        | NO                   |
    // OpenVINO releases >= 23.2 are passing a runtime attribute "is_new_api" to inform
    // the compiler if API2.0 is being used. However, given the compatibility matrix
    // above, compiler will ignore "is_new_api" to maintain compatibility with even
    // older applications that use OV versions < 23.2 and IRv11
    if (irVersion >= 11) {
        log.info("IR version >= 11. Preprocessing will be skipped.");
        return;
    }
    log.info("IR version < 11. Preprocessing will be performed.");

    bool hasRequiredIOInfo = modelData->precisions.isValid() && modelData->layouts.isValid();
    if (!hasRequiredIOInfo) {
        log.warning("The ioInfo options are missing for IR version < 11! Preprocessing will be skipped.");
        return;
    }
    log.info("The ioInfo options are provided. Preprocessing will be performed.");

    auto preprocessor = ov::preprocess::PrePostProcessor(modelData->model);
    const ov::ParameterVector& parameters = modelData->model->get_parameters();
    const ov::ResultVector& results = modelData->model->get_results();

    log.trace("Configuring {0} parameter nodes for preprocessing...", parameters.size());
    for (size_t parameterIndex = 0; parameterIndex < parameters.size(); ++parameterIndex) {
        const std::shared_ptr<ov::op::v0::Parameter>& parameter = parameters[parameterIndex];
        const std::string inputID = useIndices ? std::to_string(parameterIndex) : parameter->get_friendly_name();

        const ov::Layout tensorLayout(modelData->layouts.inputLayouts.at(inputID));
        const size_t rank = parameter->get_shape().size();
        const ov::Layout modelLayout(rankToLegacyLayoutString(rank));

        ov::preprocess::InputInfo& inputInfo = preprocessor.input(parameterIndex);
        inputInfo.tensor().set_layout(tensorLayout);
        inputInfo.model().set_layout(modelLayout);
        inputInfo.tensor().set_element_type(modelData->precisions.inputPrecisions.at(inputID));
    }
    log.trace("Completed the configuration of all parameter nodes!");

    log.trace("Configuring {0} result nodes for preprocessing...", results.size());
    for (size_t resultIndex = 0; resultIndex < results.size(); ++resultIndex) {
        const std::shared_ptr<ov::op::v0::Result>& result = results[resultIndex];

        std::string outputID;

        if (useIndices) {
            outputID = std::to_string(resultIndex);
        } else {
            // Otherwise, the legacy name of the result node (refers to the name of its parent node) will be used
            outputID = result->get_input_node_ptr(0)->get_friendly_name();
            if (result->get_input_node_ptr(0)->get_output_size() != 1) {
                // If the parent node does not have exactly 1 output port
                if (!modelData->layouts.outputLayouts.count(outputID)) {
                    // If the legacy name is not found, append the index of the parent node's output port linked to this
                    // result node. Otherwise, do not append anything.
                    outputID += "." + std::to_string(result->input_value(0).get_index());
                }
            }
        }

        ov::Layout tensorLayout;

        try {
            tensorLayout = ov::Layout(modelData->layouts.outputLayouts.at(outputID));
        } catch (const std::out_of_range& e) {
            // Throw an error and print the list of available indices/names
            std::string availableIDs;
            for (const auto& allIDs : modelData->layouts.outputLayouts) {
                availableIDs += "\n" + allIDs.first;
            }

            throw std::runtime_error(std::string(e.what()) + "\nFailed to resolve obtained " +
                                     std::string(useIndices ? "index" : "name") + " '" + outputID + "' at the " +
                                     std::to_string(resultIndex) +
                                     "th result node (node name: " + result->get_friendly_name() + "). Available " +
                                     std::string(useIndices ? "indices" : "names") + ": " + availableIDs);
        }

        const size_t rank = result->get_shape().size();
        const ov::Layout modelLayout(rankToLegacyLayoutString(rank));

        ov::preprocess::OutputInfo& outputInfo = preprocessor.output(resultIndex);
        outputInfo.tensor().set_layout(tensorLayout);
        outputInfo.model().set_layout(modelLayout);
        outputInfo.tensor().set_element_type(modelData->precisions.outputPrecisions.at(outputID));
    }
    log.trace("Completed the configuration of all result nodes!");

    modelData->model = preprocessor.build();
}

}  // namespace vpux
