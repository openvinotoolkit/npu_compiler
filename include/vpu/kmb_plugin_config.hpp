//
// Copyright 2019 Intel Corporation.
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
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <vpu/vpu_plugin_config.hpp>

//
// KMB plugin options
//

#define VPU_KMB_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_KMB_##name)
#define VPU_KMB_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_KMB_##name

#define DECLARE_VPU_KMB_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_KMB_##name)
#define DECLARE_VPU_KMB_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_KMB_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

//
// KMB plugin options
//

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("ma2490"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("debug_ma2490"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB);
/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "NO".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("."), default: "";
 * path where the mcmCompilator resulting files (blob, json, dot and png) should be placed
 * in folders named "<MCM_TARGET_DESCRIPTOR>/<MCM_COMPILATION_DESCRIPTOR>"
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("<network name>"), default: "";
 * name of mcmCompilator resulting files (blob, json, dot and png)
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "NO".
 */
DECLARE_VPU_KMB_CONFIG_KEY(KMB_EXECUTOR);

/**
 * @brief [Only for kmbPlugin]
 * Type: integer, default is 1.
 */
DECLARE_VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
