//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// The OpenVINO Intel NPU header is the source of truth for this
// attribute. This VPUX copy is consumed during model deserialization.
// Keep the RTTI name/version and serialized keys synchronized with OV.
//

#pragma once

#include <string_view>

#include "openvino/core/runtime_attribute.hpp"

namespace intel_npu {

/**
 * @brief Attribute containing the memory address of a weights buffer and the size of the buffer in bytes.
 * @details Used as part of the model marshalling process to avoid the need of copying weights.
 */
class WeightsPointerAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightsPointerAttribute", "0", ov::RuntimeAttribute);

    WeightsPointerAttribute() = delete;

    WeightsPointerAttribute(const void* pointer, const size_t size)
            : memory_pointer(reinterpret_cast<size_t>(pointer)), byte_size(size) {
    }

    /**
     * @note The names of the attributes have been kept short in order to save some memory (there may be a lot of
     * "ov::Constant" nodes in a model). While deserializing, the name of the attribute ("WeightsPointerAttribute") is
     * also used as part of identification in order to avoid collision.
     */
    static constexpr const std::string_view POINTER_KEY = "mp";
    static constexpr const std::string_view BYTE_SIZE_KEY = "ms";

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute(POINTER_KEY.data(), memory_pointer);
        visitor.on_attribute(BYTE_SIZE_KEY.data(), byte_size);
        return true;
    }

    bool is_deterministic() const override {
        return false;
    }

    size_t memory_pointer;
    size_t byte_size;
};

}  // namespace intel_npu
