//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/xml_deserializer.hpp"

#include "vpux/compiler/icompiler.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <openvino/op/group_query_attention.hpp>
#include <openvino/opsets/opset.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/shared_buffer.hpp>
#include <openvino/runtime/string_aligned_buffer.hpp>
#include <openvino/util/xml_parse_utils.hpp>
#include <openvino/xml_util/weights_provider.hpp>
#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

constexpr uint32_t MAX_NUMBER_OF_ELEMENTS = 10;
constexpr uint64_t MAX_SIZE_OF_XML = std::numeric_limits<uint64_t>::max() / 3;
constexpr uint64_t MAX_SIZE_OF_WEIGHTS = MAX_SIZE_OF_XML * 2;

// Verifies [regionOffset, regionOffset + regionSize) stays within the buffer without overflow.
bool isInRange(uint64_t regionOffset, uint64_t regionSize, size_t serializedModelSize) {
    return regionOffset <= serializedModelSize && regionSize <= serializedModelSize - regionOffset;
}

// Parses extensionLibPaths into a vector of separated paths
std::vector<std::string> parseExtensionLibPaths(const std::string& extensionLibPaths) {
    std::vector<std::string> paths;
    vpux::Logger log("VCLextensions", vpux::Logger::global().level());

    if (extensionLibPaths.empty()) {
        return paths;
    }

    const char pathSeparator = ';';

    std::istringstream stream(extensionLibPaths);
    std::string path;
    while (std::getline(stream, path, pathSeparator)) {
        // remove leading and trailing whitespaces
        const auto first = path.find_first_not_of(" \t\n\r\f\v");
        if (first == std::string::npos) {
            continue;
        }
        path.erase(0, first);
        const auto last = path.find_last_not_of(" \t\n\r\f\v");
        path.erase(last + 1);
        if (!path.empty()) {
            log.info("Adding extension library path: {0}", path);
            paths.push_back(path);
        }
    }
    return paths;
}

}  // namespace

namespace vpux {

void loadBaseOpExtensionsForMap(const std::string& path,
                                std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensionsMap,
                                std::vector<ov::Extension::Ptr>& sharedObjectExtensions);

std::shared_ptr<ov::Model> deserializeIrModelBase(uint8_t* serializedModel, const size_t serializedModelSize,
                                                  const vcl_version_info_t& currentAPIVersion,
                                                  const std::vector<ov::Extension::Ptr>& extensionsVector,
                                                  const std::string& extensionLibPaths) {
    Logger log("deserializeIrModelBase", Logger::global().level());

    if (serializedModel == nullptr) {
        throw InvalidIrError("The serialized model buffer is null!");
    }

    uint64_t offset = 0;

    if (!isInRange(offset, sizeof(vcl_version_info_t), serializedModelSize)) {
        throw InvalidIrError(
                formatv("The IR is too small to contain the API version! Buffer size: {0}", serializedModelSize).str());
    }
    const auto& APIVersion = *reinterpret_cast<const vcl_version_info_t*>(serializedModel + offset);
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        throw InvalidIrError(std::string("Unsupported IR API version! Val: ") + std::to_string(APIVersion.major) + "." +
                             std::to_string(APIVersion.minor));
    }
    offset += sizeof(vcl_version_info_t);

    if (!isInRange(offset, sizeof(uint32_t), serializedModelSize)) {
        throw InvalidIrError(
                formatv("The IR is too small to contain the elements number! Buffer size: {0}, offset: {1}",
                        serializedModelSize, offset)
                        .str());
    }
    const auto& numOfElements = *reinterpret_cast<const uint32_t*>(serializedModel + offset);
    if (numOfElements >= MAX_NUMBER_OF_ELEMENTS) {
        throw InvalidIrError("Bad elements number in IR!");
    }
    offset += sizeof(numOfElements);

    if (!isInRange(offset, sizeof(uint64_t), serializedModelSize)) {
        throw InvalidIrError(formatv("The IR is too small to contain the buffer size! Buffer size: {0}, offset: {1}",
                                     serializedModelSize, offset)
                                     .str());
    }
    const auto& bufferSize = *reinterpret_cast<const uint64_t*>(serializedModel + offset);
    if (bufferSize == 0 || bufferSize >= MAX_SIZE_OF_XML) {
        throw InvalidIrError("Bad buffer size in IR!");
    }
    offset += sizeof(bufferSize);

    const uint64_t bufferOffset = offset;
    if (!isInRange(bufferOffset, bufferSize, serializedModelSize)) {
        throw InvalidIrError(formatv("The XML buffer does not fit in the IR! Buffer size: {0}, bufferOffset: {1}, "
                                     "declared bufferSize: {2}",
                                     serializedModelSize, bufferOffset, bufferSize)
                                     .str());
    }
    offset += bufferSize;

    if (!isInRange(offset, sizeof(uint64_t), serializedModelSize)) {
        throw InvalidIrError(formatv("The IR is too small to contain the weights size! Buffer size: {0}, offset: {1}",
                                     serializedModelSize, offset)
                                     .str());
    }
    auto& weightsSize = *reinterpret_cast<uint64_t*>(serializedModel + offset);
    if (weightsSize >= MAX_SIZE_OF_WEIGHTS) {
        throw InvalidIrError("Bad weights size in IR!");
    }
    offset += sizeof(weightsSize);

    const uint64_t weightsOffset = offset;
    if (!isInRange(weightsOffset, weightsSize, serializedModelSize)) {
        throw InvalidIrError(formatv("The weights buffer does not fit in the IR! Buffer size: {0}, weightsOffset: {1}, "
                                     "declared weightsSize: {2}",
                                     serializedModelSize, weightsOffset, weightsSize)
                                     .str());
    }

    log.debug("Parsed IR header - bufferSize: {0}, weightsSize: {1}, serializedModelSize: {2}", bufferSize, weightsSize,
              serializedModelSize);

    const uint8_t* buffer = serializedModel + bufferOffset;
    const uint8_t* weights = serializedModel + weightsOffset;
    std::string modelData(buffer, buffer + bufferSize);

    ov::Tensor weightsTensor;
    if (weightsSize > 0) {
        weightsTensor = ov::Tensor(ov::element::u8, {weightsSize}, weights);
    }
    ov::Core core;
    core.add_extension(extensionsVector);
    // Add extensions from paths
    for (const auto& path : parseExtensionLibPaths(extensionLibPaths)) {
        core.add_extension(path);
    }

    return core.read_model(modelData, weightsTensor);
}

std::shared_ptr<ov::Model> deserializeIrModelOptimized(uint8_t* serializedModel, const size_t serializedModelSize,
                                                       const vcl_version_info_t& currentAPIVersion,
                                                       const std::vector<ov::Extension::Ptr>& extensionsVector,
                                                       const std::string& extensionLibPaths) {
    Logger log("deserializeIrModelOptimized", Logger::global().level());

    if (serializedModel == nullptr) {
        throw InvalidIrError("The serialized model buffer is null!");
    }

    const auto& dataHeader = *reinterpret_cast<const ov::pass::StreamSerialize::DataHeader*>(serializedModel);
    if (serializedModelSize < sizeof(dataHeader)) {
        throw InvalidIrError(formatv("The serialized model size is too small to contain the data header! Buffer size: "
                                     "{0}, required: {1}",
                                     serializedModelSize, sizeof(dataHeader))
                                     .str());
    }

    if (!isInRange(dataHeader.custom_data_offset, sizeof(vcl_version_info_t), serializedModelSize)) {
        throw InvalidIrError(
                formatv("The API version does not fit in the IR! Buffer size: {0}, custom_data_offset: {1}",
                        serializedModelSize, dataHeader.custom_data_offset)
                        .str());
    }
    if (!isInRange(dataHeader.model_offset, dataHeader.model_size, serializedModelSize)) {
        throw InvalidIrError(formatv("The model buffer does not fit in the IR! Buffer size: {0}, model_offset: {1}, "
                                     "model_size: {2}",
                                     serializedModelSize, dataHeader.model_offset, dataHeader.model_size)
                                     .str());
    }
    if (!isInRange(dataHeader.consts_offset, dataHeader.consts_size, serializedModelSize)) {
        throw InvalidIrError(formatv("The consts buffer does not fit in the IR! Buffer size: {0}, consts_offset: {1}, "
                                     "consts_size: {2}",
                                     serializedModelSize, dataHeader.consts_offset, dataHeader.consts_size)
                                     .str());
    }

    log.debug("Parsed IR data header - model_offset: {0}, model_size: {1}, consts_offset: {2}, consts_size: {3}, "
              "serializedModelSize: {4}",
              dataHeader.model_offset, dataHeader.model_size, dataHeader.consts_offset, dataHeader.consts_size,
              serializedModelSize);

    const auto& serializedAPIVersion =
            *reinterpret_cast<const vcl_version_info_t*>(serializedModel + dataHeader.custom_data_offset);
    if (serializedAPIVersion.major != currentAPIVersion.major ||
        serializedAPIVersion.minor != currentAPIVersion.minor) {
        throw InvalidIrError(std::string("The API version found in the serialized model is not supported. Found: ") +
                             std::to_string(serializedAPIVersion.major) + "." +
                             std::to_string(serializedAPIVersion.minor) + ". Expected: " +
                             std::to_string(currentAPIVersion.major) + "." + std::to_string(currentAPIVersion.minor));
    }

    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_buffer(serializedModel + dataHeader.model_offset, dataHeader.model_size,
                                                    pugi::parse_default, pugi::encoding_utf8);
    OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
    pugi::xml_node root = xmlDoc.document_element();

    std::shared_ptr<ov::AlignedBuffer> weightsBuffer = std::make_shared<ov::SharedBuffer<void*>>(
            reinterpret_cast<char*>(serializedModel + dataHeader.consts_offset), dataHeader.consts_size, nullptr);
    auto weightsProvider = std::make_shared<ov::util::BufferWeightsProvider>(weightsBuffer);

    std::unordered_map<std::string, ov::OpSet> opsets;
    for (const auto& it : ov::get_available_opsets()) {
        opsets[it.first] = it.second();
    }

    // Build extensionsMap from both the vector and extensionLibPaths
    std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> extensionsMap;
    // Add extensions from the vector
    for (const auto& ext : extensionsVector) {
        if (auto baseExt = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext)) {
            extensionsMap.insert({baseExt->get_type_info(), baseExt});
        }
    }

    // Keep extension wrappers alive until model destruction, so underlying shared libraries remain loaded.
    std::vector<ov::Extension::Ptr> sharedObjectExtensions;
    // TODO: E#220612 use core.add_extension() to add extensions by path, instead of using a local mechanism
    for (const auto& path : parseExtensionLibPaths(extensionLibPaths)) {
        // Add extensions from paths using OV add_extensions() equivalent flow
        loadBaseOpExtensionsForMap(path, extensionsMap, sharedObjectExtensions);
    }

    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
    size_t version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

    XmlDeserializer visitor(root, weightsProvider, opsets, extensionsMap, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);

    // Pin shared object extensions to model rt_info to keep them alive as long as the model is alive
    if (!sharedObjectExtensions.empty()) {
        model->get_rt_info()["__ov_extension_path_shared_object_extensions"] = sharedObjectExtensions;
    }
    model->get_rt_info()["version"] = int64_t(version);

    return model;
}

XmlDeserializer::XmlDeserializer(const pugi::xml_node& node,
                                 const std::shared_ptr<ov::util::WeightsProvider>& weightsProvider,
                                 const std::unordered_map<std::string, ov::OpSet>& opsets,
                                 const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                                 std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                                 size_t version)
        : ov::util::XmlDeserializer(node, weightsProvider, opsets, extensions, variables, version) {
}

std::optional<intel_npu::WeightsPointerAttribute> XmlDeserializer::parse_weights_pointer_attribute(
        const pugi::xml_node& node) const {
    if (pugi::xml_node rtInfo = node.child("rt_info")) {
        for (const pugi::xml_node& child : rtInfo.children()) {
            if (strcmp(child.attribute("name").value(),
                       intel_npu::WeightsPointerAttribute::get_type_info_static().name) == 0) {
                const auto ptr = reinterpret_cast<const void*>(ov::util::pugixml::get_uint64_attr(
                        child, intel_npu::WeightsPointerAttribute::POINTER_KEY.data()));
                const auto byteSize = ov::util::pugixml::get_uint64_attr(
                        child, intel_npu::WeightsPointerAttribute::BYTE_SIZE_KEY.data());
                return intel_npu::WeightsPointerAttribute{ptr, byteSize};
            }
        }
    }
    return std::nullopt;
}

void XmlDeserializer::set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter) {
    const auto node = get_node();
    auto wpAttribute = parse_weights_pointer_attribute(node);
    if (!wpAttribute.has_value()) {
        // The weights metadata is missing. Extract the values using the weights provider.
        ov::util::XmlDeserializer::set_constant_num_buffer(adapter);
        return;
    }

    const auto& dn = node.child("data");
    const auto elementType = ov::element::Type(ov::util::pugixml::get_str_attr(dn, "element_type"));

    char* ptr = reinterpret_cast<char*>(wpAttribute->memory_pointer);
    size_t byteSize = wpAttribute->byte_size;

    std::shared_ptr<ov::AlignedBuffer> buffer;
    if (elementType != ov::element::string) {
        buffer = std::make_shared<ov::SharedBuffer<void*>>(ptr, byteSize, nullptr);
    } else {
        buffer = std::make_shared<ov::SharedStringAlignedBuffer>(ptr, byteSize);
    }
    adapter.set(buffer);
}

std::unique_ptr<ov::util::XmlDeserializer> XmlDeserializer::make_visitor(
        const pugi::xml_node& node, const std::shared_ptr<ov::util::WeightsProvider>& weightsProvider,
        const std::unordered_map<std::string, ov::OpSet>& opsets,
        const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables, size_t version) const {
    return std::make_unique<XmlDeserializer>(node, weightsProvider, opsets, extensions, variables, version);
}

}  // namespace vpux
