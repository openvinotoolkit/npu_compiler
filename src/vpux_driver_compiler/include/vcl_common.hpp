//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_common.hpp
 * @brief The helper functions to check time and parse build info
 */

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "npu_driver_compiler.h"
#include "vcl_logger.hpp"

#include "intel_npu/icompiler.hpp"
#include "vpux/utils/IE/config.hpp"

namespace VPUXDriverCompiler {

/**
 * @brief Calc the time cost
 *
 */
struct StopWatch {
    using fp_milliseconds = std::chrono::duration<double, std::chrono::milliseconds::period>;

    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    void stop() {
        stopTime = std::chrono::steady_clock::now();
    }

    double delta_ms() const {
        return std::chrono::duration_cast<fp_milliseconds>(stopTime - startTime).count();
    }

    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point stopTime;
};

class VPUXCompilerL0;

/**
 * @brief Parse and store all the info which is used to create blob
 *
 */
struct BuildInfo {
    explicit BuildInfo(VPUXCompilerL0* pvc);

    /**
     * @brief Convert value to ov defined precision, not all precisions are supported
     *
     * @param value The string represents precison. For example: FP32
     * @param matched If the value is not supported in the inside table, assign false
     */
    static ov::element::Type_t stringToOVPrecision(std::string value, bool& matched);

    /**
     * @brief Checks if the given string corresponds to a supported layout value.
     *
     * @param value The string represents layout. For example: NCHW
     * @param matched If the value is not supported in the inside table, assign false
     */
    static std::string checkSupportedLayout(std::string value, bool& matched);

    /**
     * @brief Parse the ioInfo string to real values and store them
     *
     * @param ioInfoOptions the input and output config which is created by prepareBuildFlags
     *
     * @see prepareBuildFlags()
     */
    vcl_result_t parseIOOption(const std::vector<std::string>& ioInfoOptions);

    /**
     * @brief Parse the config string and update parsedConfig
     *
     * @param descOptions The info must contain runtime configuration of compiler, but it can also have other info
     * which will be ignored
     *
     * @return vcl_result_t
     * @see prepareBuildFlags()
     */
    vcl_result_t prepareConfig(const std::string& descOptions);

    /**
     * @brief Parse the build flags from vcl_executable_desc_t and store the results
     *
     * @param descOptions The info includes input and output info of model, runtime configuration of compiler
     * @return vcl_result_t
     */
    vcl_result_t prepareBuildFlags(const std::string& descOptions);

    /**
     * @brief Parse the model and weight from modelIR, store the results
     *
     * @param modelIR The memory which contains the model xml and bin data
     * @param modelIRSize The size of the memory which is pointed by modelIR
     * @return vcl_result_t
     */
    vcl_result_t prepareModel(const uint8_t* modelIR, uint64_t modelIRSize);

    /**
     * @brief The model deserialized by prepareModel()
     */
    std::shared_ptr<ov::Model> model;

    /**
     * @name Input and output setting from user
     * @{
     */
    std::unordered_map<std::string, ov::element::Type_t> inputPrecisions;
    std::unordered_map<std::string, std::string> inputLayouts;
    std::unordered_map<std::string, ov::element::Type_t> outputPrecisions;
    std::unordered_map<std::string, std::string> outputLayouts;
    /** @} */

    /// The runtime compilation config from user
    intel_npu::Config parsedConfig;

    /// Calc time cost on VCL level
    bool enableProfiling = false;
    VPUXCompilerL0* pvc = nullptr;
    VCLLogger* logger = nullptr;
};  // BuildInfo

template <class Callable>
class Scoped {
public:
    explicit Scoped(Callable&& callable): _callable(std::forward<Callable>(callable)) {
    }
    ~Scoped() {
        _callable();
    }

private:
    Callable _callable;
};

}  // namespace VPUXDriverCompiler
