//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

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
    VPU3400_A0 = 1,  // Keem bay A0
    VPU3400 = 2,     // Keem bay B0
    VPU3700 = 3,     // Keem bay B0
    VPU3800 = 4,     // Thunder bay harbor Prime
    VPU3900 = 5,     // Thunder bay harbor A0
    VPU3720 = 6,     // Meteor lake
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
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
DECLARE_VPUX_CONFIG_KEY(USE_M2I);

/**
 * @deprecated Use VPUX_USE_M2I instead
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_M2I);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I)
 * SHAVE only version module for image pre-processing
 */
DECLARE_VPUX_CONFIG_KEY(USE_SHAVE_ONLY_M2I);

/**
 * @deprecated Use VPUX_USE_SHAVE_ONLY_M2I instead
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I)
 * SHAVE only version module for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_SHAVE_ONLY_M2I);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
DECLARE_VPUX_CONFIG_KEY(USE_SIPP);

/**
 * @deprecated Use VPUX_USE_SIPP instead
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_SIPP);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 1.
 * Number of executor streams
 */
DECLARE_VPUX_CONFIG_KEY(EXECUTOR_STREAMS);

/**
 * @deprecated Use VPUX_EXECUTOR_STREAMS instead
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 1.
 * Number of executor streams
 */
DECLARE_VPU_KMB_CONFIG_KEY(EXECUTOR_STREAMS);

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
enum class CompilerType { MCM, MLIR };
DECLARE_VPUX_CONFIG_KEY(COMPILER_TYPE);
DECLARE_VPUX_CONFIG_VALUE(MCM);
DECLARE_VPUX_CONFIG_VALUE(MLIR);

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
 * Type: "YES", "NO", default is "NO"
 * Print detailed profiling info during inference
 */
DECLARE_VPUX_CONFIG_KEY(PRINT_PROFILING);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
