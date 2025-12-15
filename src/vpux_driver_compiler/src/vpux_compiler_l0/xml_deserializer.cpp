//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "xml_deserializer.hpp"

#include <openvino/op/group_query_attention.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/shared_buffer.hpp>
#include <openvino/runtime/string_aligned_buffer.hpp>
#include <openvino/util/common_util.hpp>
#include <openvino/util/xml_parse_utils.hpp>
#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

namespace VPUXDriverCompiler {

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

}  // namespace VPUXDriverCompiler
