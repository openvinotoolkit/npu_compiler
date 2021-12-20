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

/**
 * @brief A header that defines advanced related properties for VPU MCM compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 */

#pragma once

#include <vpux/vpux_compiler_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for MCM compiler]
 * name of xml (IR) file name to serialize prepared for sending to mcmCompiler CNNNetwork
 * Type: Arbitrary string. Empty means "no serialization", default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE);

DECLARE_VPU_COMPILER_CONFIG_KEY(REFERENCE_MODE);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_U8_INPUT_FOR_FP16_MODELS);

DECLARE_VPU_COMPILER_CONFIG_KEY(SCALESHIFT_FUSING);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_PERMUTE_ND);

DECLARE_VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER);

DECLARE_VPU_COMPILER_CONFIG_KEY(OPTIMIZE_INPUT_PRECISION);

/**
 * @brief [Currently supported only for MCM Compiler + Level0 backend]
 * Type: "YES", "NO", default is "NO"
 * This option allows to perform FP32/FP16 to U8 input quantization on VPUX Plugin side via CPU
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(FORCE_PLUGIN_INPUT_QUANTIZATION);

/**
 * @brief [Currently supported only for MCM Compiler]
 * Type: "YES", "NO", default is "NO"
 * This option allows to perform FP16 to FP32 output conversion on VPUX Plugin side via CPU
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(OUTPUT_FP16_TO_FP32_HOST_CONVERSION);

/**
 * @brief [Only for MCM compiler]
 * Type: Arbitrary string. Empty means ("."), default: "";
 * path where the mcmCompilator resulting files (blobs, json, dots, etc.) should be placed
 * in folders named "<TARGET_DESCRIPTOR>/<COMPILATION_DESCRIPTOR>"
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH);

/**
 * @brief [Only for MCM compiler]
 * Type: Arbitrary string. Empty means ("<network name>"), default: "";
 * name of mcmCompilator resulting files (blobs, json, dots, etc.)
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS);

/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB);
/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(PARSING_ONLY);

/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_JSON);

/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "NO".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_DOT);

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:strategy
 * Multiple entries separated by comma. eg, conv1:SplitOverK,conv2:SplitOverH
 * Overrides the GO split strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "split_strategy":
    [   { "name_filter": "conv1",
          "strategy": "SplitOverK" }
    ]
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_SPLIT_STRATEGIES);

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:streamsW:streamsH:streamsC:streamsK:streamsN
 * Multiple entries separated by comma. eg, conv1:1:1:1:1,conv2:1:2:1:2:1
 * Overrides the GO streaming strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "streaming_strategy":
   [ {
     "name_filter": "conv1",
     "splits":
      [ { "W": 1 }, { "H": 1 }, { "C": 1 }, { "K": 1 }, { "N": 1 } ]
    } ]
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_STREAM_STRATEGIES);

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:input_sparsity:output_sparsity:weights_sparsity
 * Multiple entries separated by comma. eg, conv1:true:true:true,conv2:false:true:false
 * Overrides the GO sparsity strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "sparsity_strategy":
   [ {
     "name_filter": "conv1",
     "inputActivationSparsity": true,
     "outputActivationSparsity": true,
     "weightsSparsity": true
    } ]
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_SPARSITY_STRATEGIES);

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:location
 * Multiple entries separated by comma. eg, conv1:DDR,conv2:NNCMX
 * Overrides the GO location strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "tensor_placement_override":
   [ {
     "name_filter": "conv1",
     "mem_location": "DDR"
    } ]
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_LOCATION_STRATEGIES);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
