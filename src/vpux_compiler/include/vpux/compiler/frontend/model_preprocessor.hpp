//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/icompiler.hpp"

#include <memory>
#include <string>
#include <utility>

namespace vpux {

/**
 * @brief Adds precision conversion and transposition layers to the model in order to comply with the given
 * precision and layout values.
 * @details In the legacy scenarios when either the older API or the IR version 10 is being used, the "ov::Model"
 * object may not hold the correct I/O metadata values (either a wrong precision or a transposed shape may be used).
 * The objective of the current function is to correct this misalignment by introducing additional precision
 * conversion or transposition layers.
 *
 * Note that the correct precision/layout values are given by the driver. Depending on the plugin version, the
 * origin of these values may be either the metadata stored by the user application in a legacy
 * "InferenceEngine::CNNNetwork" object, or the values found within the "ov::Model" one, which could have been
 * altered as a result of the serialization process.
 * @param modelData The model data containing the model and reference precision/layout values.
 *        The model inside modelData is updated in-place.
 */
void preprocessModel(const std::shared_ptr<ModelData>& modelData);

/**
 * @brief Prepares the compiler and device configuration based on the provided description options.
 * @param descOptions The description options as a string.
 * @param compilerDesc Compiler description.
 * @param deviceDesc Device description.
 * @param config The runtime compilation config from user.
 * @param isDeviceDescEmpty Indicates whether the device description is empty.
 */
void prepareConfig(const std::string& descOptions, const vcl_compiler_desc_t& compilerDesc,
                   const vcl_device_desc_t& deviceDesc, intel_npu::Config& config, bool isDeviceDescEmpty);

/**
 * @brief Parse the build flags from vcl_executable_desc_t
 *
 * @param descOptions The info includes input and output info of model, runtime configuration of compiler
 * @param compilerDesc Compiler description.
 * @param compilerProp Compiler capabilities.
 * @param deviceDesc Device description.
 * @param config The runtime compilation config from user.
 * @param isDeviceDescEmpty Indicates whether the device description is empty.
 * @return A pair of Precisions and Layouts parsed from descOptions
 */
std::pair<Precisions, Layouts> prepareBuildFlags(const std::string& descOptions,
                                                 const vcl_compiler_desc_t& compilerDesc,
                                                 const vcl_compiler_properties_t& compilerProp,
                                                 const vcl_device_desc_t& deviceDesc, intel_npu::Config& config,
                                                 bool isDeviceDescEmpty);

/**
 * @brief Parse the model and weight from modelIR.
 * @param modelIR The memory which contains the model xml and bin data
 * @param modelIRSize The size of the memory which is pointed by modelIR
 * @param config The runtime compilation config from user.
 * @param compilerProp Compiler capabilities, used to determine deserialization format.
 * @return Deserialized model.
 */
std::shared_ptr<ov::Model> prepareModel(const uint8_t* modelIR, uint64_t modelIRSize, intel_npu::Config& config,
                                        const vcl_compiler_properties_t& compilerProp);

}  // namespace vpux
