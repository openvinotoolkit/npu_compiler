//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <vpux/vpux_compiler_config.hpp>

//
// VPUX plugin options
//

#define VPUX_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPUX_##name)
#define VPUX_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::name

#define DECLARE_VPUX_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPUX_##name)
#define DECLARE_VPUX_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(name)

namespace InferenceEngine {
namespace VPUXConfigParams {

//
// VPUX plugin options
//

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 * Configuration API v1.0
 */
DECLARE_VPUX_CONFIG_KEY(INFERENCE_SHAVES);

/**
 * Type: Arbitrary string. Default is "-1".
 * This option allows to specify CSRAM size in bytes
 * When the size is -1, low-level SW is responsible for determining the required amount of CSRAM
 * When the size is 0, CSRAM isn't used
 * Configuration API v1.0
 */
DECLARE_VPUX_CONFIG_KEY(CSRAM_SIZE);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
