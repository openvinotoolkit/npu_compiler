//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/utils/config_parser.hpp"
#include <algorithm>
#include <cctype>
#include <memory>
#include <pugixml.hpp>
#include <sstream>
#include <string>
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux::Shave;

namespace {

// Mandatory string attribute; throws if the attribute is missing.
std::string get_str_attr(const pugi::xml_node& node, const char* name) {
    auto attr = node.attribute(name);
    VPUX_THROW_UNLESS(!attr.empty(), "node <{0}> is missing mandatory attribute: {1}", node.name(), name);
    return attr.value();
}

// Mandatory integer attribute; throws if the attribute is missing or not an integer.
int get_int_attr(const pugi::xml_node& node, const char* name) {
    auto attr = node.attribute(name);
    VPUX_THROW_UNLESS(!attr.empty(), "node <{0}> is missing mandatory attribute: {1}", node.name(), name);
    std::string value = attr.value();
    std::size_t idx = 0;
    int intValue = 0;
    try {
        intValue = std::stoi(value, &idx, 10);
    } catch (const std::exception&) {
        VPUX_THROW("node <{0}> has attribute \"{1}\" = \"{2}\" which is not an integer", node.name(), name, value);
    }
    VPUX_THROW_UNLESS(idx == value.length(), "node <{0}> has attribute \"{1}\" = \"{2}\" which is not an integer",
                      node.name(), name, value);
    return intValue;
}

// Optional boolean attribute; returns defVal if the attribute is missing.
bool get_bool_attr(const pugi::xml_node& node, const char* name, bool defVal) {
    auto attr = node.attribute(name);
    if (attr.empty()) {
        return defVal;
    }
    std::string value = attr.value();
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return std::tolower(ch);
    });
    if (value == "true" || value == "1") {
        return true;
    }
    if (value == "false" || value == "0") {
        return false;
    }
    VPUX_THROW("node <{0}> has unsupported boolean attribute \"{1}\" = \"{2}\"", node.name(), name, attr.value());
}

std::optional<TensorRank> parseTensorRank(::llvm::StringRef str) {
    return ::llvm::StringSwitch<::std::optional<TensorRank>>(str)
            .Case("ND", TensorRank::_ND)
            .Case("1D", TensorRank::_1D)
            .Case("2D", TensorRank::_2D)
            .Case("3D", TensorRank::_3D)
            .Case("4D", TensorRank::_4D)
            .Case("5D", TensorRank::_5D)
            .Default(::std::nullopt);
}

std::vector<TensorRank> mapStringToSupportedRanksVector(const std::string& ranksStr) {
    std::vector<TensorRank> supportedRanks{};

    std::stringstream ss(ranksStr);
    std::string singleRank{};
    char del = ',';

    while (std::getline(ss, singleRank, del)) {
        auto rankEnumVal = parseTensorRank(llvm::StringRef(singleRank).trim());
        VPUX_THROW_UNLESS(rankEnumVal.has_value(), "Unrecognized Custom Layer Tensor Rank");
        supportedRanks.emplace_back(rankEnumVal.value());
    }
    return supportedRanks;
}

std::vector<TensorDataType> mapStringToSupportedDataTypesVector(const std::string& dataTypesStr) {
    std::vector<TensorDataType> supportedDataTypes{};

    std::stringstream ss(dataTypesStr);
    std::string singleDataType{};
    char del = ',';

    while (std::getline(ss, singleDataType, del)) {
        auto dataTypeEnumVal = symbolizeTensorDataType(llvm::StringRef(singleDataType).trim());
        VPUX_THROW_UNLESS(dataTypeEnumVal.has_value(), "Unrecognized Custom Layer Tensor Data Type");
        supportedDataTypes.emplace_back(dataTypeEnumVal.value());
    }
    return supportedDataTypes;
}

CustomLayerConfig::SpecificConfigType mapStringToConfigType(const std::string& configStr) {
    using ConfigType = CustomLayerConfig::SpecificConfigType;
    if (configStr == "Kernel") {
        return ConfigType::KernelConfig;
    } else if (configStr == "Buffers") {
        return ConfigType::IOConfig;
    } else if (configStr == "Heap") {
        return ConfigType::HeapConfig;
    } else if (configStr == "Tiling") {
        return ConfigType::TilingConfig;
    }
    return ConfigType::UNDEFINED;
}

}  // namespace

//
// CustomLayerConfigParserImpl
//

namespace vpux::Shave {

class CustomLayerConfigParserImpl {
public:
    CustomLayerConfigParserImpl(const std::string& configFilePath, Logger log)
            : _log(std::move(log)), _configFilePath(configFilePath) {
    }

    void loadConfigs(CustomLayerConfigMap& configs);

private:
    // loads a collection of Custom Layer Configs
    void loadConfigCollection(CustomLayerConfigMap& configs);

    // loads a single Custom Layer Config
    // returns layer unique id
    std::string loadSingleConfig(pugi::xml_node& configNode, CustomLayerConfigPtr layerConfig);

    // methods to process individual config parts
    void processKernelConfig(pugi::xml_node& configNode, CustomLayerConfig::KernelConfig& config);
    void processIOConfig(pugi::xml_node& configNode, CustomLayerConfig::IOConfig& config);
    void processHeapConfig(pugi::xml_node& configNode, CustomLayerConfig::HeapConfig& config);
    void processTilingConfig(pugi::xml_node& configNode, CustomLayerConfig::TilingConfig& config);

private:
    Logger _log;
    std::string _configFilePath{};

    pugi::xml_document _xmlDoc;
};

void CustomLayerConfigParserImpl::loadConfigs(CustomLayerConfigMap& configs) {
    pugi::xml_parse_result res = _xmlDoc.load_file(_configFilePath.c_str());
    VPUX_THROW_UNLESS(res.status == pugi::status_ok,
                      "Failed to parse Custom Layer Config XML at path {0}, error at offset {1}", _configFilePath,
                      res.offset);

    loadConfigCollection(configs);
}

void CustomLayerConfigParserImpl::loadConfigCollection(CustomLayerConfigMap& configs) {
    auto rootNode = _xmlDoc.document_element();

    VPUX_THROW_UNLESS(std::string(rootNode.name()) == "CustomLayerCollection",
                      "CustomLayerConfigParser: Root XML Node needs to be of type CustomLayerCollection, actual: {0}",
                      rootNode.name());

    // Check for target="NPU" attribute
    auto target = rootNode.attribute("target");
    VPUX_THROW_UNLESS(target && std::string(target.value()) == "NPU",
                      "CustomLayerConfigParser: CustomLayerCollection target must be \"NPU\"");

    // Iterate through the individual custom layers in the collection
    for (auto individualConfigNode : rootNode.children()) {
        CustomLayerConfigPtr layerConfig = std::make_shared<Shave::CustomLayerConfig>();
        auto layerName = loadSingleConfig(individualConfigNode, layerConfig);
        VPUX_THROW_UNLESS(configs.find(layerName) == configs.end(),
                          "CustomLayerConfigParser: Duplicate CustomLayer name {0} in config collection", layerName);
        configs[layerName] = std::move(layerConfig);
    }
}

std::string CustomLayerConfigParserImpl::loadSingleConfig(pugi::xml_node& configNode,
                                                          CustomLayerConfigPtr layerConfig) {
    VPUX_THROW_UNLESS(
            std::string(configNode.name()) == "CustomLayer",
            "CustomLayerConfigParser: Individual config XML node expected to be named \"CustomLayer\", actual: {0}",
            configNode.name());

    // Check for name attribute
    auto uniqueNameAttr = configNode.attribute("name");
    VPUX_THROW_UNLESS(uniqueNameAttr && std::string(uniqueNameAttr.value()) != "",
                      "CustomLayer Node needs to have the name attribute populated");

    layerConfig->setUniqueId(uniqueNameAttr.value());

    auto isValidConfig = false;
    for (auto specificConfig : configNode.children()) {
        auto nodeType = std::string(specificConfig.name());
        switch (mapStringToConfigType(nodeType)) {
            using ConfigType = CustomLayerConfig::SpecificConfigType;
        case ConfigType::KernelConfig:
            processKernelConfig(specificConfig, layerConfig->kernelConfig);
            isValidConfig = true;
            break;
        case ConfigType::IOConfig:
            processIOConfig(specificConfig, layerConfig->ioConfig);
            break;
        case ConfigType::HeapConfig:
            processHeapConfig(specificConfig, layerConfig->heapConfig);
            break;
        case ConfigType::TilingConfig:
            processTilingConfig(specificConfig, layerConfig->tilingConfig);
            break;
        default:
            VPUX_THROW("Unrecognized config type: {0}", nodeType);
        }
    }
    VPUX_THROW_UNLESS(isValidConfig, "CustomLayerConfigParser: Kernel config is missing for {0}",
                      layerConfig->getUniqueId());
    return layerConfig->getUniqueId();
}

void CustomLayerConfigParserImpl::processKernelConfig(pugi::xml_node& configNode,
                                                      CustomLayerConfig::KernelConfig& config) {
    // Process Kernel Type
    auto kernelTypeAttr = get_str_attr(configNode, "type");
    auto kernelTypeEnumVal = symbolizeKernelType(kernelTypeAttr);
    VPUX_THROW_UNLESS(kernelTypeEnumVal, "Error: Custom Layer Kernel Type is unrecognized");
    config.kernelType = kernelTypeEnumVal.value();

    // Process Kernel Source
    auto sourceNode = configNode.child("Source");
    VPUX_THROW_UNLESS(sourceNode, "Error: Kernel Source must be specified");
    config.sourcePath = get_str_attr(sourceNode, "path");

    // Process (optional) Kernel Compiler Options
    auto compilerOptionsNode = configNode.child("CompilerOptions");
    if (compilerOptionsNode) {
        config.usesDMA = get_bool_attr(compilerOptionsNode, "uses-dma", false);
        config.usesDPU = get_bool_attr(compilerOptionsNode, "uses-dpu", false);
    }
}

void CustomLayerConfigParserImpl::processIOConfig(pugi::xml_node& configNode, CustomLayerConfig::IOConfig& config) {
    size_t ioConfigCount = 0;
    for (auto childNode : configNode.children()) {
        // Process Tensor IO Config
        if (std::string(childNode.name()) == "Tensor") {
            CustomLayerConfig::IOTensorConfig tensorConfig{};

            // Argument index
            tensorConfig.argIndex = get_int_attr(childNode, "arg-index");

            // IO Type (input/output)
            auto ioTypeStr = get_str_attr(childNode, "type");
            auto ioTypeEnumVal = symbolizeIOType(ioTypeStr);
            VPUX_THROW_UNLESS(ioTypeEnumVal.has_value(), "Unrecognized Custom Layer IO Type");
            tensorConfig.ioType = ioTypeEnumVal.value();

            // Supported ranks (optional)
            if (childNode.attribute("supported-ranks")) {
                auto supportedRanksStr = get_str_attr(childNode, "supported-ranks");
                tensorConfig.supportedRanks = mapStringToSupportedRanksVector(supportedRanksStr);
            }

            // Supported Data Types (optional)
            if (childNode.attribute("supported-data-types")) {
                auto supportedDataTypesStr = get_str_attr(childNode, "supported-data-types");
                tensorConfig.supportedDataTypes = mapStringToSupportedDataTypesVector(supportedDataTypesStr);
            }

            // Enforced rank (optional)
            if (childNode.attribute("enforce-rank")) {
                auto enforcedRankStr = get_str_attr(childNode, "enforce-rank");
                auto enforcedRankEnumVal = parseTensorRank(enforcedRankStr);
                VPUX_THROW_UNLESS(enforcedRankEnumVal.has_value(), "Unrecognized Custom Layer Enforced Tensor Rank");
                tensorConfig.enforcedRank = enforcedRankEnumVal.value();
                if (!tensorConfig.supportedRanks.empty()) {
                    _log.info("Custom Layer Tensor arg-index {0}: enforce-rank is set alongside "
                              "supported-ranks. The supported-ranks check may be redundant.",
                              tensorConfig.argIndex);
                }
            }

            // Enforced data type (optional)
            if (childNode.attribute("enforce-data-type")) {
                auto enforcedDataTypeStr = get_str_attr(childNode, "enforce-data-type");
                auto enforcedDataTypeEnumVal = symbolizeTensorDataType(enforcedDataTypeStr);
                VPUX_THROW_UNLESS(enforcedDataTypeEnumVal.has_value(),
                                  "Unrecognized Custom Layer Enforced Tensor Data Type");
                tensorConfig.enforcedDataType = enforcedDataTypeEnumVal.value();
                if (!tensorConfig.supportedDataTypes.empty()) {
                    _log.info("Custom Layer Tensor arg-index {0}: enforce-data-type is set alongside "
                              "supported-data-types. The supported-data-types check may be redundant.",
                              tensorConfig.argIndex);
                }
            }

            // Enforced layout (optional)
            if (childNode.attribute("enforce-layout")) {
                auto enforcedLayoutStr = get_str_attr(childNode, "enforce-layout");
                auto enforcedLayoutEnumVal = symbolizeTensorLayout(enforcedLayoutStr);
                VPUX_THROW_UNLESS(enforcedLayoutEnumVal.has_value(),
                                  "Unrecognized Custom Layer Enforced Tensor Layout");
                tensorConfig.enforcedLayout = enforcedLayoutEnumVal.value();
            }
            config.ioTensorConfigs.push_back(std::move(tensorConfig));
        }
        // Process Scratch IO Config
        else if (std::string(childNode.name()) == "Scratch") {
            CustomLayerConfig::ScratchConfig scratchConfig{};

            // Argument index
            scratchConfig.argIndex = get_int_attr(childNode, "arg-index");

            // Scratch Location
            auto locationStr = get_str_attr(childNode, "location");
            auto locationEnumVal = symbolizeScratchLocation(locationStr);
            VPUX_THROW_UNLESS(locationEnumVal.has_value(), "Unrecognized Custom Layer Scratch Buffer Location");
            scratchConfig.location = locationEnumVal.value();

            // Scratch buffer size
            scratchConfig.size = get_int_attr(childNode, "size");

            config.scratchConfigs.push_back(scratchConfig);
        } else {
            VPUX_THROW("Unrecognized IO Config: {0}", childNode.name());
        }
        ioConfigCount++;
    }
    VPUX_THROW_UNLESS(ioConfigCount > 0, "No IO Configs were provided for the Custom Layer");
}

void CustomLayerConfigParserImpl::processHeapConfig(pugi::xml_node& configNode, CustomLayerConfig::HeapConfig& config) {
    config.size = get_int_attr(configNode, "max_size");
}

void CustomLayerConfigParserImpl::processTilingConfig(pugi::xml_node& configNode,
                                                      CustomLayerConfig::TilingConfig& config) {
    auto tilingModeStr = get_str_attr(configNode, "mode");
    auto tilingModeEnumVal = symbolizeTilingMode(tilingModeStr);
    VPUX_THROW_UNLESS(tilingModeEnumVal.has_value(), "Unrecognized Custom Layer Tiling Mode");
    config.tilingMode = tilingModeEnumVal.value();
}

}  // namespace vpux::Shave

//
// CustomLayerConfigParser
//

CustomLayerConfigParser::CustomLayerConfigParser(const std::string& configFilePath, Logger log)
        : _impl(std::make_unique<CustomLayerConfigParserImpl>(configFilePath, std::move(log))) {
}

CustomLayerConfigParser::~CustomLayerConfigParser() = default;
CustomLayerConfigParser::CustomLayerConfigParser(CustomLayerConfigParser&&) noexcept = default;
CustomLayerConfigParser& CustomLayerConfigParser::operator=(CustomLayerConfigParser&&) noexcept = default;

void CustomLayerConfigParser::loadConfigs(CustomLayerConfigMap& configs) {
    _impl->loadConfigs(configs);
}
