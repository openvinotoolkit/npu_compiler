//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/xml_deserializer.hpp"

#include "vpux/compiler/icompiler.hpp"

#include <openvino/op/group_query_attention.hpp>
#include <openvino/opsets/opset.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/shared_buffer.hpp>
#include <openvino/runtime/string_aligned_buffer.hpp>
#include <openvino/util/common_util.hpp>
#include <openvino/util/xml_parse_utils.hpp>
#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

namespace {

constexpr uint32_t MAX_NUMBER_OF_ELEMENTS = 10;
constexpr uint64_t MAX_SIZE_OF_XML = std::numeric_limits<uint64_t>::max() / 3;
constexpr uint64_t MAX_SIZE_OF_WEIGHTS = MAX_SIZE_OF_XML * 2;

}  // namespace

namespace vpux {

std::shared_ptr<ov::Model> deserializeIrModelBase(uint8_t* serializedModel, const size_t serializedModelSize,
                                                  const vcl_version_info_t& currentAPIVersion,
                                                  const std::vector<ov::Extension::Ptr>& extensionsVector) {
    uint64_t offset = 0;
    vcl_version_info_t APIVersion;
    memcpy(&APIVersion, serializedModel, sizeof(APIVersion));
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        throw InvalidIrError(std::string("Unsupported IR API version! Val: ") + std::to_string(APIVersion.major) + "." +
                             std::to_string(APIVersion.minor));
    }
    offset += sizeof(vcl_version_info_t);

    uint32_t numOfElements = 0;
    memcpy(&numOfElements, serializedModel + offset, sizeof(numOfElements));
    if (numOfElements >= MAX_NUMBER_OF_ELEMENTS) {
        throw InvalidIrError("Bad elements number in IR!");
    }
    offset += sizeof(numOfElements);

    uint64_t bufferSize = 0;
    memcpy(&bufferSize, serializedModel + offset, sizeof(bufferSize));
    if (bufferSize == 0 || bufferSize >= MAX_SIZE_OF_XML) {
        throw InvalidIrError("Bad buffer size in IR!");
    }
    offset += sizeof(bufferSize);

    const uint64_t bufferOffset = offset;
    offset += bufferSize;

    uint64_t weightsSize = 0;
    memcpy(&weightsSize, serializedModel + offset, sizeof(weightsSize));
    if (weightsSize >= MAX_SIZE_OF_WEIGHTS) {
        throw InvalidIrError("Bad weights size in IR!");
    }
    offset += sizeof(weightsSize);

    const uint64_t weightsOffset = offset;
    if (offset + weightsSize > serializedModelSize) {
        throw InvalidIrError("The IR content and size mismatch!");
    }

    const uint8_t* buffer = serializedModel + bufferOffset;
    const uint8_t* weights = serializedModel + weightsOffset;
    std::string modelData(buffer, buffer + bufferSize);

    ov::Tensor weightsTensor;
    if (weightsSize > 0) {
        weightsTensor = ov::Tensor(ov::element::u8, {weightsSize}, const_cast<uint8_t*>(weights));
    }
    ov::Core core;
    core.add_extension(extensionsVector);

    return core.read_model(modelData, weightsTensor);
}

std::shared_ptr<ov::Model> deserializeIrModelOptimized(uint8_t* serializedModel, const size_t serializedModelSize,
                                                       const vcl_version_info_t& currentAPIVersion,
                                                       const std::vector<ov::Extension::Ptr>& extensionsVector) {
    ov::pass::StreamSerialize::DataHeader dataHeader;
    if (serializedModelSize < sizeof(dataHeader)) {
        throw InvalidIrError("The serialized model size is too small to contain the data header!");
    }
    memcpy(&dataHeader, serializedModel, sizeof(dataHeader));

    auto inRange = [&](uint64_t offset, uint64_t regionSize) {
        return offset <= serializedModelSize && regionSize <= serializedModelSize - offset;
    };

    if (!inRange(dataHeader.custom_data_offset, sizeof(vcl_version_info_t)) ||
        !inRange(dataHeader.model_offset, dataHeader.model_size) ||
        !inRange(dataHeader.consts_offset, dataHeader.consts_size)) {
        throw InvalidIrError("The IR content and size mismatch!");
    }

    vcl_version_info_t serializedAPIVersion;
    memcpy(&serializedAPIVersion, serializedModel + dataHeader.custom_data_offset, sizeof(serializedAPIVersion));
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

XmlDeserializer::XmlDeserializer(const pugi::xml_node& node, const std::shared_ptr<ov::AlignedBuffer>& weights,
                                 const std::unordered_map<std::string, ov::OpSet>& opsets,
                                 const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                                 std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                                 size_t version)
        : ov::util::XmlDeserializer(node, weights, opsets, extensions, variables, version) {
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
        // The weights metadata is missing. Extract the values from buffer.
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
        const pugi::xml_node& node, const std::shared_ptr<ov::AlignedBuffer>& originalWeights,
        const std::unordered_map<std::string, ov::OpSet>& opsets,
        const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables, size_t version) const {
    return std::make_unique<XmlDeserializer>(node, originalWeights, opsets, extensions, variables, version);
}

}  // namespace vpux
