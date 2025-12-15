//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include <openvino/xml_util/xml_deserialize_util.hpp>

#include <intel_npu/weights_pointer_attribute.hpp>

namespace VPUXDriverCompiler {

/**
 * @brief Deserializer meant to mirror the "intel_npu::XmlSerializer" found in the NPU plugin.
 * @details The main optimization brought by this deserializer is the ability to reconstruct weights based on a pointer
 * and a size in bytes. This feature is optional and can be controlled in a hybrid manner. I.e. a subset of weights can
 * be reconstructed based on pointers, the remaining can be extracted from the buffer passed to this class.
 *
 * The "ov::Constant" nodes containing the "WeightsPointerAttribute" in their runtime information field will be
 * reconstructed based on pointers. Otherwise, extraction from buffer will be performed.
 */
class XmlDeserializer : public ov::util::XmlDeserializer {
public:
    XmlDeserializer(const pugi::xml_node& node, const std::shared_ptr<ov::AlignedBuffer>& weights,
                    const std::unordered_map<std::string, ov::OpSet>& opsets,
                    const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                    size_t version);

    /**
     * @brief Looks for the "WeightsPointerAttribute" within the runtime information field and reconstructs the
     * attribute.
     * @details This attribute contains a pointer towards some weights and their size in bytes. If found, the
     * deserializer will use this information to reconstruct the "ov::Constant" node. Otherwise, the algorithm expects
     * the weights to be found in the buffer provided to its constructor.
     *
     * @return The reconstructed attribute if found, containing a pointer towards some weights and their size in bytes.
     */
    std::optional<intel_npu::WeightsPointerAttribute> parse_weights_pointer_attribute(const pugi::xml_node& node) const;

    /**
     * @brief Reconstructs the buffer based on the "WeightsPointerAttribute" if found. Otherwise, the weights are
     * expected to be found in the buffer provided to the constructor.
     */
    void set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter) override;

    std::unique_ptr<ov::util::XmlDeserializer> make_visitor(
            const pugi::xml_node& node, const std::shared_ptr<ov::AlignedBuffer>& originalWeights,
            const std::unordered_map<std::string, ov::OpSet>& opsets,
            const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
            std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
            size_t version) const override;
};

}  // namespace VPUXDriverCompiler
