//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

/**
 * @brief A header that defines advanced related properties for VPU compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_compiler_config.hpp
 */

#pragma once

#include <vpux/vpux_compiler_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for vpu compiler]
 * name of xml (IR) file name to serialize prepared for sending to mcmCompiler CNNNetwork
 * Type: Arbitrary string. Empty means "no serialization", default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE);

DECLARE_VPU_COMPILER_CONFIG_KEY(REFERENCE_MODE);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_U8_INPUT_FOR_FP16_MODELS);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_CONVERT_INPUT_PRECISION_TO_U8);

DECLARE_VPU_COMPILER_CONFIG_KEY(SCALESHIFT_FUSING);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_PERMUTE_ND);

DECLARE_VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
