//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_profiling.hpp
 * @brief Define VPUXProfilingL0 which parses profiling data
 */

#pragma once

#include "npu_driver_compiler.h"

#include "vcl_compiler_loader.hpp"
#include "vcl_logger.hpp"

#include "vpux/compiler/icompiler.hpp"

#include <memory>
#include <vector>

namespace VPUXDriverCompiler {

/**
 * @brief Parse the profiling output with blob.
 *
 * Check @ref how_to_use_profiling.md about how to collect the data
 */
class VPUXProfilingL0 final {
public:
    /**
     * @brief Construct a new VPUXProfilingL0 object
     *
     * @param profInput Include the blob and correspond profiling output
     * @param vclLogger The logger instance
     */
    VPUXProfilingL0(p_vcl_profiling_input_t profInput, VCLLogger* vclLogger)
            : _blobData(profInput->blobData),
              _blobSize(profInput->blobSize),
              _profData(profInput->profData),
              _profSize(profInput->profSize),
              _compilerLoader(std::make_unique<CompilerLoader>()),
              _logger(vclLogger) {
    }

    vcl_result_t getTaskInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getLayerInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getRawInfo(p_vcl_profiling_output_t profOutput);
    vcl_profiling_properties_t getProperties() const;
    VCLLogger* getLogger() const {
        return _logger;
    }

private:
    const uint8_t* _blobData;  ///< Pointer to the buffer with the blob
    uint64_t _blobSize;        ///< Size of the blob in bytes
    const uint8_t* _profData;  ///< Pointer to the raw profiling output
    uint64_t _profSize;        ///< Size of the raw profiling output

    std::unique_ptr<CompilerLoader> _compilerLoader;            ///< The handle of the compiler instance
    std::vector<vpux::ze::ze_profiling_task_info> _taskInfo;    ///< Per-task (DPU, DMA, SW) profiling info
    std::vector<vpux::ze::ze_profiling_layer_info> _layerInfo;  ///< Per-layer profiling info
    VCLLogger* _logger;                                         ///< Internal logger
};

}  // namespace VPUXDriverCompiler
