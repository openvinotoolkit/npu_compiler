//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_driver_compiler.h"

#include <openvino/core/extension.hpp>
#include <openvino/core/model.hpp>

namespace VPUXDriverCompiler {

class invalid_ir_error : public std::runtime_error {
public:
    explicit invalid_ir_error(const std::string& message): runtime_error(message) {
    }
};

/**
 * @brief Deserializes the model using the legacy algorithm. The value of the weights is also found in the buffer.
 * @details This deserialization function corresponds to the base serialization algorithm from the NPU plugin that
 * performs weights copies in a separate buffer.
 *
 * @param serializedModel A buffer containing in this order:
 *   * The compiler version.
 *   * The number of buffers following this field (2).
 *   * The size of the XML graph.
 *   * The graph.
 *   * The size of the weights.
 *   * The weights.
 * @param serializedModelSize The size of the buffer to be deserialized.
 * @param currentAPIVersion
 * @return The deserialized model.
 * @throws invalid_ir_error if an unexpected buffer format is received.
 */
std::shared_ptr<ov::Model> deserialize_ir_model_base(uint8_t* serializedModel, const size_t serializedModelSize,
                                                     const vcl_version_info_t currentAPIVersion,
                                                     const std::vector<ov::Extension::Ptr>& extensionsVector);

/**
 * @brief Deserializes the model using the optimized algorithm. The weights are reconstructed based on pointers towards
 * their buffers. The weights that are missing pointers are expected to have their values copied into the buffer
 * provided to this function.
 * @details This deserialization function corresponds to the optimized serialization algorithm from the NPU plugin that
 * stores pointers towards weights buffers instead of copying them. The capability to extract weights from the dedicated
 * buffer is meant to be used as a fallback mechanism.
 *
 * @param serializedModel Buffer obtained after running the "intel_npu::StreamSerialize" pass on an ov::Model object.
 * Format:
 *   * Data header (offsets and sizes)
 *   * Custom data (only the API version is found here for now)
 *   * Weights (if applicable)
 *   * The graph expressed as an XML object
 * @param currentAPIVersion
 * @return The deserialized model.
 * @throws invalid_ir_error if an unexpected buffer format is received.
 */
std::shared_ptr<ov::Model> deserialize_ir_model_optimized(uint8_t* serializedModel,
                                                          const vcl_version_info_t currentAPIVersion,
                                                          const std::vector<ov::Extension::Ptr>& extensionsVector);

}  // namespace VPUXDriverCompiler
