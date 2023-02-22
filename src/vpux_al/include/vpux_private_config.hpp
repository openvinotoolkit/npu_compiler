//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

/**
 * @brief VPUX platform configuration
 *
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 */

#pragma once

#include <string>
#include <vpux/vpux_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @enum VPUXPlatform
 * @brief VPUX device
 */
enum class VPUXPlatform : int {
    AUTO = 0,        // auto detection
    VPU3400_A0 = 1,  // VPUX30XX A0
    VPU3400 = 2,     // VPUX30XX B0
    VPU3700 = 3,     // VPUX30XX B0
    VPU3800 = 4,     // VPUX311X Prime
    VPU3900 = 5,     // VPUX311X Full
    VPU3720 = 6,     // VPU3720
    EMULATOR = 7,    // emulator
};

/**
 * @brief [Only for VPUX Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
DECLARE_VPUX_CONFIG_KEY(PLATFORM);

/**
 * @brief [Only for VPUX Plugin]
 * Type: "RGB", "BGR", default is "BGR"
 * This option allows to specify output format of image after SIPP preprocessing.
 * Does not affect preprocessing running on CPU. If a wrong value specified an exception will be thrown
 */
DECLARE_VPUX_CONFIG_KEY(GRAPH_COLOR_FORMAT);
DECLARE_VPUX_CONFIG_VALUE(BGR);
DECLARE_VPUX_CONFIG_VALUE(RGB);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 4.
 * Number of shaves to be used by SIPP during preprocessing
 */
DECLARE_VPUX_CONFIG_KEY(PREPROCESSING_SHAVES);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 8.
 * Lines per iteration value to be used by SIPP during preprocessing
 */
DECLARE_VPUX_CONFIG_KEY(PREPROCESSING_LPI);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 1.
 * Number of preprocessing pipelines to be used by particular network,
 * these pipelines will work in parallel and make preprocessing
 * for all infer requests of this network
 */
DECLARE_VPUX_CONFIG_KEY(PREPROCESSING_PIPES);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
DECLARE_VPUX_CONFIG_KEY(USE_SIPP);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 5 minutes = 60 * 1000 * 5.
 * Time interval during which to wait for backend pull to complete
 */
DECLARE_VPUX_CONFIG_KEY(INFERENCE_TIMEOUT);

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is MLIR.
 * Type of VPU compiler to be used for compilation of a network
 */
enum class CompilerType { MLIR, DRIVER };
DECLARE_VPUX_CONFIG_KEY(COMPILER_TYPE);
DECLARE_VPUX_CONFIG_VALUE(MLIR);
DECLARE_VPUX_CONFIG_VALUE(DRIVER);

DECLARE_VPUX_CONFIG_KEY(COMPILATION_MODE);

/**
 * @brief [Only for VPUX compiler]
 * Type: std::string, default is empty.
 * Config for HW-mode's pipeline
 * Available values: low-precision=true/low-precision=false
 */
DECLARE_VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DPU groups
 */
DECLARE_VPUX_CONFIG_KEY(DPU_GROUPS);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DMA engines
 */
DECLARE_VPUX_CONFIG_KEY(DMA_ENGINES);

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is NONE
 * NONE - do not print profiling info
 * TEXT, JSON - print detailed profiling info during inference in requested format
 */
enum class ProfilingOutputTypeArg { NONE, TEXT, JSON };
DECLARE_VPUX_CONFIG_KEY(PRINT_PROFILING);
DECLARE_VPUX_CONFIG_VALUE(NONE);
DECLARE_VPUX_CONFIG_VALUE(TEXT);
DECLARE_VPUX_CONFIG_VALUE(JSON);

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is empty.
 * File that contains profiling output.
 * std::cout is used if this string is empty
 */
DECLARE_VPUX_CONFIG_KEY(PROFILING_OUTPUT_FILE);

/**
 * @brief [Only for VPUX Plugin]
 * Type: String. Default is "NO".
 * This option is added for enabling ELF backend.
 * Possible values: "YES", "NO".
 */
DECLARE_VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
