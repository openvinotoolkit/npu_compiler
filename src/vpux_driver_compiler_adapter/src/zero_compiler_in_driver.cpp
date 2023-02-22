//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "zero_compiler_in_driver.h"
#include "ie_layouts.h"
#include "vpux/al/config/common.hpp"
#include "vpux/utils/IE/itt.hpp"

namespace vpux {
namespace driverCompilerAdapter {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
// TODO #-30200 : Not all Precision from IE listed in ze_graph_ext
// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
IE::Precision toIEPrecision(const ze_graph_argument_precision_t ze_precision) {
    switch (ze_precision) {
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return IE::Precision::FP32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return IE::Precision::FP16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return IE::Precision::U16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return IE::Precision::U8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return IE::Precision::I32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return IE::Precision::I16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return IE::Precision::I8;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return IE::Precision::BIN;

    default:
        return IE::Precision::UNSPECIFIED;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
IE::Layout toIELayout(const ze_graph_argument_layout_t ze_layout) {
    switch (ze_layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return IE::Layout::NCHW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return IE::Layout::NHWC;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return IE::Layout::NCDHW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return IE::Layout::NDHWC;

    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return IE::Layout::OIHW;

    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return IE::Layout::C;

    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return IE::Layout::CHW;

    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return IE::Layout::HW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return IE::Layout::NC;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return IE::Layout::CN;

    case ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED:
        return IE::Layout::BLOCKED;
    default:
        return IE::Layout::ANY;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
size_t getDimCount(const IE::Layout layout) {
    switch (layout) {
    case IE::Layout::C:
        return 1;
    case IE::Layout::CN:
        return 2;
    case IE::Layout::HW:
        return 2;
    case IE::Layout::NC:
        return 2;
    case IE::Layout::CHW:
        return 3;
    case IE::Layout::NCHW:
        return 4;
    case IE::Layout::NHWC:
        return 4;
    case IE::Layout::NCDHW:
        return 5;
    case IE::Layout::NDHWC:
        return 5;
    default:
        // TODO #-30200 Extend to support all cases
        return 0;
    }

    return 0;
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ze_graph_argument_layout_t toZeLayout(const IE::Layout layout) {
    switch (layout) {
    case IE::Layout::NCHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NCHW;
    case IE::Layout::NHWC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NHWC;
    case IE::Layout::NCDHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW;
    case IE::Layout::NDHWC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC;

    case IE::Layout::OIHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_OIHW;

    case IE::Layout::C:
        return ZE_GRAPH_ARGUMENT_LAYOUT_C;

    case IE::Layout::CHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_CHW;

    case IE::Layout::HW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_HW;
    case IE::Layout::NC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NC;
    case IE::Layout::CN:
        return ZE_GRAPH_ARGUMENT_LAYOUT_CN;

    case IE::Layout::BLOCKED:
        return ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED;
    default:
        return ZE_GRAPH_ARGUMENT_LAYOUT_ANY;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ze_graph_argument_precision_t toZePrecision(const IE::Precision precision) {
    switch (precision) {
    case IE::Precision::I8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case IE::Precision::U8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case IE::Precision::I16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case IE::Precision::U16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case IE::Precision::I32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case IE::Precision::FP16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case IE::Precision::FP32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case IE::Precision::BIN:
        return ZE_GRAPH_ARGUMENT_PRECISION_BIN;
    default:
        return ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
    }
}

//------------------------------------------------------------------------------
LevelZeroCompilerInDriver::LevelZeroCompilerInDriver(): _logger("LevelZeroCompilerInDriver", LogLevel::Warning) {
    auto result = zeInit(ZE_INIT_FLAG_VPU_ONLY);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to initialize zeAPI. Error code: " << std::hex << result;
    }

    uint32_t drivers = 0;
    result = zeDriverGet(&drivers, nullptr);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get information about zeDriver. Error code: " << std::hex << result;
    }

    std::vector<ze_driver_handle_t> all_drivers(drivers);
    result = zeDriverGet(&drivers, all_drivers.data());
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get zeDriver. Error code: " << std::hex << result;
    }

    // Get our target driver
    for (uint32_t i = 0; i < drivers; ++i) {
        // arbitrary test at this point
        if (i == drivers - 1) {
            _driver_handle = all_drivers[i];
        }
    }

    // Load our graph extension
    result = zeDriverGetExtensionFunctionAddress(_driver_handle, "ZE_extension_graph",
                                                 reinterpret_cast<void**>(&_graph_ddi_table_ext));
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to initialize zeDriver. Error code: " << std::hex << result;
    }

    uint32_t device_count = 1;
    // Get our target device
    result = zeDeviceGet(_driver_handle, &device_count, &_device_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get device. Error code: " << std::hex << result;
    }

    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(_driver_handle, &context_desc, &_context);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to initialize context for device. Error code: " << std::hex << result;
    }
}

LevelZeroCompilerInDriver::~LevelZeroCompilerInDriver() {
    if (_context) {
        auto result = zeContextDestroy(_context);
        if (ZE_RESULT_SUCCESS != result) {
            _logger.warning("zeContextDestroy failed {0:X+}", uint64_t(result));
        }
    }
    _logger.debug("LevelZeroCompilerInDriver obj destroyed");
}

using SerializedIR = std::vector<uint8_t>;
/**
 * @brief Place xml + weights in sequential memory
 * @details Format of the memory:
 */
SerializedIR LevelZeroCompilerInDriver::serializeIR(const std::vector<char>& xml, const std::vector<char>& weights) {
    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    ze_device_graph_properties_t device_graph_properties{};
    auto result = _graph_ddi_table_ext->pfnDeviceGetGraphProperties(_device_handle, &device_graph_properties);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get graph properties from compiler";
    }

    const auto compilerVersion = device_graph_properties.compilerVersion;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(xml.size());
    const uint64_t weightsSize = static_cast<uint64_t>(weights.size());

    IE_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        IE_THROW() << "Xml file is too big to process.";
    }
    if (weightsSize >= maxSizeOfWeights) {
        IE_THROW() << "Bin file is too big to process.";
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xml.size() + sizeof(weightsSize) + weights.size();

    std::vector<uint8_t> serializedIR;
    serializedIR.resize(sizeOfSerializedIR);

    uint64_t offset = 0;
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, xml.data(), xmlSize);
    offset += xmlSize;
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, weights.data(), weightsSize);
    offset += weightsSize;

    IE_ASSERT(offset == sizeOfSerializedIR);

    return serializedIR;
}

/**
 * @brief Serialize input / output information to string format
 * Format:
 * --inputs_precisions="<input1Name>:<input1Precision> [<input2Name>:<input2Precision>]"
 * --inputs_layouts="<input1Name>:<input1Layout> [<input2Name>:<input2Layout>]"
 * --outputs_precisions="<output1Name>:<output1Precision>"
 * --outputs_layouts="<output1Name>:<output1Layout>"
 */
std::string LevelZeroCompilerInDriver::serializeIOInfo(const InferenceEngine::InputsDataMap& inputsInfo,
                                                       const InferenceEngine::OutputsDataMap& outputsInfo) {
    const std::string inputsPrecisionsKey = "--inputs_precisions";
    const std::string inputsLayoutsKey = "--inputs_layouts";
    const std::string outputsPrecisionsKey = "--outputs_precisions";
    const std::string outputsLayoutsKey = "--outputs_layouts";

    // <option key>="<option value>"
    const std::string keyValueSeparator = "=";
    const std::string valueDelimiter = "\"";  // marks beginning and end of value

    // Format inside "<option value>"
    // <name1>:<value (precision / layout)> [<name2>:<value>]
    const std::string nameValueSeparator = ":";
    const std::string valuesSeparator = " ";

    auto serializeOptionValue = [&](auto& portsInfo, auto& precisionSS, auto& layoutSS) {
        for (auto&& port : portsInfo) {
            if (port.first != portsInfo.cbegin()->first) {
                precisionSS << valuesSeparator;
                layoutSS << valuesSeparator;
            }
            precisionSS << port.first << nameValueSeparator << port.second->getPrecision();
            layoutSS << port.first << nameValueSeparator << port.second->getLayout();
        }

        precisionSS << valueDelimiter;
        layoutSS << valueDelimiter;
    };

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;

    inputsPrecisionSS << inputsPrecisionsKey << keyValueSeparator << valueDelimiter;
    inputsLayoutSS << inputsLayoutsKey << keyValueSeparator << valueDelimiter;

    serializeOptionValue(inputsInfo, inputsPrecisionSS, inputsLayoutSS);

    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    outputsPrecisionSS << outputsPrecisionsKey << keyValueSeparator << valueDelimiter;
    outputsLayoutSS << outputsLayoutsKey << keyValueSeparator << valueDelimiter;

    serializeOptionValue(outputsInfo, outputsPrecisionSS, outputsLayoutSS);

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + valuesSeparator + inputsLayoutSS.str() + valuesSeparator +
           outputsPrecisionSS.str() + valuesSeparator + outputsLayoutSS.str();
}

std::string toString(const ze_graph_argument_precision_t& precision) {
    switch (precision) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return "UINT8";
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return "FP32";
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return "FP16";
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return "UINT16";
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return "INT32";
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return "INT16";
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return "INT8";
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return "BIN";
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return "BF16";
    default:
        return "UNKNOWN";
    }
}

std::string toString(const ze_graph_argument_layout_t& layout) {
    switch (layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return "NCHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return "NHWC";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return "NCDHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return "NDHWC";
    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return "OIHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return "C";
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return "CHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return "HW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return "NC";
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return "CN";
    case ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED:
        return "BLOCKED";
    default:
        return "ANY";
    }
}

static std::string serializeConfig(const vpux::Config& config) {
    return "--config " + config.toString();
}

INetworkDescription::Ptr LevelZeroCompilerInDriver::compileIR(
        const std::string& graphName, const std::vector<char>& xml, const std::vector<char>& weights,
        const IE::InputsDataMap& inputsInfo, const IE::OutputsDataMap& outputsInfo, const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::compileIR");
    auto serializedIR = serializeIR(xml, weights);

    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;

    if (inputsInfo.empty() || outputsInfo.empty()) {
        THROW_IE_EXCEPTION << "Information about inputs or outputs is not provided.";
    }

    std::string build_flags;

    build_flags += serializeIOInfo(inputsInfo, outputsInfo);
    build_flags += " ";
    build_flags += serializeConfig(config);

    _logger.debug("Build flags : {0}", build_flags);

    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                         nullptr,
                         format,
                         serializedIR.size(),
                         serializedIR.data(),
                         build_flags.c_str()};

    // TODO #-30202 Store graph_handle inside NetworkDesc instead of blob. But this will require changes in zeroAPI

    // Graph handle should be used only in scope of compile / parse functions.
    ze_graph_handle_t graph_handle;
    auto result = _graph_ddi_table_ext->pfnCreate(_context, _device_handle, &desc, &graph_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to compile network. Error code: " << std::hex << result;
    }

    // Get blob size first
    size_t blobSize = -1;
    result = _graph_ddi_table_ext->pfnGetNativeBinary(graph_handle, &blobSize, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get blob size. Error code: " << std::hex << result;
    }

    std::vector<char> blob(blobSize);
    // Get blob data
    result = _graph_ddi_table_ext->pfnGetNativeBinary(graph_handle, &blobSize, reinterpret_cast<uint8_t*>(blob.data()));
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get compiled network. Error code: " << std::hex << result;
    }

    const auto networkMeta = getNetworkMeta(graph_handle);
    result = _graph_ddi_table_ext->pfnDestroy(graph_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to destroy graph handle. Error code: " << std::hex << result;
    }

    _logger.debug("LevelZeroCompilerInDriver::compileIR end");
    return std::make_shared<driverCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

std::shared_ptr<INetworkDescription> LevelZeroCompilerInDriver::parseBlob(const std::string& graphName,
                                                                          const std::vector<char>& blob,
                                                                          const vpux::Config& config) {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::VPUXPlugin, "LevelZeroCompilerInDriver::parseBlob", "desc");
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::getNetworkMeta");
    ze_graph_handle_t graph_handle;

    if (!blob.empty()) {
        _logger.debug("Import network case");
        ze_graph_format_t format = ZE_GRAPH_FORMAT_NATIVE;
        ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,       nullptr, format, blob.size(),
                             reinterpret_cast<const uint8_t*>(blob.data()), nullptr};

        auto result = _graph_ddi_table_ext->pfnCreate(_context, _device_handle, &desc, &graph_handle);
        OV_ITT_TASK_NEXT(PARSE_BLOB, "_graph_ddi_table_ext");
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "Failed to import blob. Error code: " << std::hex << result;
        }
    } else {
        THROW_IE_EXCEPTION << "Empty blob";
    }

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    const auto networkMeta = getNetworkMeta(graph_handle);
    OV_ITT_TASK_NEXT(PARSE_BLOB, "NetworkDescription");

    auto result = _graph_ddi_table_ext->pfnDestroy(graph_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to destroy graph handle. Error code: " << std::hex << result;
    }

    return std::make_shared<driverCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

size_t LevelZeroCompilerInDriver::getSupportedOpset() {
    _logger.debug("LevelZeroCompilerInDriver::getSupportedOpset");
    ze_device_graph_properties_t graph_properties;
    auto result = _graph_ddi_table_ext->pfnDeviceGetGraphProperties(_device_handle, &graph_properties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get opset version from compiler";
    }
    const auto maxOpsetVersion = graph_properties.maxOVOpsetVersionSupported;
    _logger.info("Max supported version of opset in CiD: {0}", maxOpsetVersion);
    return maxOpsetVersion;
}

NetworkMeta LevelZeroCompilerInDriver::getNetworkMeta(ze_graph_handle_t graph_handle) {
    ze_graph_properties_t graph_properties{};
    auto result = _graph_ddi_table_ext->pfnGetProperties(graph_handle, &graph_properties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get information about graph.";
    }

    DataMap net_inputs;
    DataMap dev_inputs;

    DataMap net_outputs;
    DataMap dev_outputs;

    for (uint32_t index = 0; index < graph_properties.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        result = _graph_ddi_table_ext->pfnGetArgumentProperties(graph_handle, index, &arg);
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "Failed to get information about inputs/outputs.";
        }

        IE::Precision net_precision = toIEPrecision(arg.networkPrecision);
        IE::Layout net_layout = toIELayout(arg.networkLayout);
        IE::SizeVector net_dims(arg.dims, arg.dims + getDimCount(net_layout));
        IE::TensorDesc net_dataDesc(net_precision, net_dims, net_layout);

        IE::Precision dev_precision = toIEPrecision(arg.devicePrecision);
        IE::Layout dev_layout = toIELayout(arg.deviceLayout);
        IE::SizeVector dev_dims(arg.dims, arg.dims + getDimCount(dev_layout));
        IE::TensorDesc dev_dataDesc(dev_precision, dev_dims, dev_layout);

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            _logger.debug("Found input \"{0}\"", arg.name);

            net_inputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, net_dataDesc));
            dev_inputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, dev_dataDesc));
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            _logger.debug("Found output \"{0}\"", arg.name);

            net_outputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, net_dataDesc));
            dev_outputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, dev_dataDesc));
        }
    }

    // TODO: support this information in CiD [track: E#33479]
    int numStreams = 1;

    return NetworkMeta{net_inputs, net_outputs, dev_inputs, dev_outputs, numStreams};
}

}  // namespace driverCompilerAdapter
}  // namespace vpux
