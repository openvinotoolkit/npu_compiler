//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_deserializer.hpp"
#include "xml_deserializer.hpp"

#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/shared_buffer.hpp>
#include <openvino/runtime/string_aligned_buffer.hpp>
#include <openvino/util/common_util.hpp>
#include <openvino/util/xml_parse_utils.hpp>

#include "intel_npu/weights_pointer_attribute.hpp"

namespace {

/**
 * @name Limitation of modelIRData
 * @see vcl_executable_desc_t for the structure
 * @{
 */
constexpr uint32_t maxNumberOfElements = 10;
/// Use offset to get the location of xml and weight from memory, shall not exceed uint64_t now
constexpr uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
constexpr uint64_t maxSizeOfWeights = maxSizeOfXML * 2;
/** @} */

}  // namespace

namespace VPUXDriverCompiler {

std::shared_ptr<ov::Model> deserialize_ir_model_base(uint8_t* serializedModel, const size_t serializedModelSize,
                                                     const vcl_version_info_t currentAPIVersion,
                                                     const std::vector<ov::Extension::Ptr>& extensionsVector) {
    /// The API version of current compiler, adapter fill its version in serializedModel, shall be same value
    uint32_t offset = 0;
    vcl_version_info_t APIVersion;
    memcpy(&APIVersion, serializedModel, sizeof(APIVersion));
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        throw invalid_ir_error(std::string("Unsupported IR API version! Val: ") + std::to_string(APIVersion.major) +
                               "." + std::to_string(APIVersion.minor));
    }
    offset += sizeof(vcl_version_info_t);

    /// The number of elements in buffer shall not exceed limitation
    uint32_t numOfElements = 0;
    memcpy(&numOfElements, serializedModel + offset, sizeof(numOfElements));
    if (numOfElements >= maxNumberOfElements) {
        throw invalid_ir_error("Bad elements number in IR!");
    }
    offset += sizeof(numOfElements);

    /// The size of model data
    uint64_t bufferSize = 0;
    memcpy(&bufferSize, serializedModel + offset, sizeof(bufferSize));
    if (bufferSize == 0 || bufferSize >= maxSizeOfXML) {
        throw invalid_ir_error("Bad buffer size in IR!");
    }
    offset += sizeof(bufferSize);

    /// The offset to model xml
    uint64_t bufferOffset = offset;

    offset += bufferSize;
    uint64_t weightsSize = 0;
    memcpy(&weightsSize, serializedModel + offset, sizeof(weightsSize));
    if (weightsSize >= maxSizeOfWeights) {
        throw invalid_ir_error("Bad weights size in IR!");
    }
    offset += sizeof(weightsSize);

    /// The offset to model weight
    uint64_t weightsOffset = offset;
    if (offset + weightsSize > serializedModelSize) {
        throw invalid_ir_error("The IR content and size mismatch!");
    }

    /// The pointer to model xml
    const uint8_t* buffer = serializedModel + bufferOffset;
    /// The pointer to model weight
    const uint8_t* weights = serializedModel + weightsOffset;
    /// Deserialize the model
    std::string modelData(buffer, buffer + bufferSize);

    ov::Tensor weightsTensor;
    if (weightsSize > 0) {
        weightsTensor = ov::Tensor(ov::element::u8, {weightsSize}, const_cast<uint8_t*>(weights));
    }
    ov::Core core;
    core.add_extension(extensionsVector);

    return core.read_model(modelData, weightsTensor);
}

std::shared_ptr<ov::Model> deserialize_ir_model_optimized(uint8_t* serializedModel,
                                                          const vcl_version_info_t currentAPIVersion,
                                                          const std::vector<ov::Extension::Ptr>& extensionsVector) {
    ov::pass::StreamSerialize::DataHeader dataHeader;
    memcpy(&dataHeader, serializedModel, sizeof(dataHeader));

    // Extract and check the API version from the "Custom data" segment of the buffer
    vcl_version_info_t serializedAPIVersion;
    memcpy(&serializedAPIVersion, serializedModel + dataHeader.custom_data_offset, sizeof(serializedAPIVersion));
    if (serializedAPIVersion.major != currentAPIVersion.major ||
        serializedAPIVersion.minor != currentAPIVersion.minor) {
        throw invalid_ir_error(std::string("The API version found in the serialized model is not supported. Found: ") +
                               std::to_string(serializedAPIVersion.major) + "." +
                               std::to_string(serializedAPIVersion.minor) + ". Expected: " +
                               std::to_string(currentAPIVersion.major) + "." + std::to_string(currentAPIVersion.minor));
    }

    // Prepare the arguments required for launching the deserializer
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_buffer(serializedModel + dataHeader.model_offset, dataHeader.model_size,
                                                    pugi::parse_default, pugi::encoding_utf8);
    OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
    pugi::xml_node root = xmlDoc.document_element();

    std::shared_ptr<ov::AlignedBuffer> weightsBuffer = std::make_shared<ov::SharedBuffer<void*>>(
            reinterpret_cast<char*>(serializedModel + dataHeader.consts_offset), dataHeader.consts_size, nullptr);

    std::unordered_map<std::string, ov::OpSet> opsets;
    for (const auto& it : ov::get_available_opsets()) {
        opsets[it.first] = it.second();
    }
    auto createExtensionsMap = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
        std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> extensionsMap;

        for (const auto& ext : extensionsVector) {
            if (auto baseExt = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext)) {
                extensionsMap.insert({baseExt->get_type_info(), baseExt});
            }
        }
        return extensionsMap;
    }();
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
    size_t version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

    XmlDeserializer visitor(root, weightsBuffer, opsets, createExtensionsMap, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);
    model->get_rt_info()["version"] = int64_t(version);

    return model;
}

}  // namespace VPUXDriverCompiler
