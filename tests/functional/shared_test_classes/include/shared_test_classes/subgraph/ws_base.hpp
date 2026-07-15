//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu_ov2_layer_test.hpp"

#include <string>

namespace ov {
namespace test {

/**
 * @brief This class serves as a base for all weight separation tests.
 *
 * The main purpose of this class is to handle the creation and cleanup of temporary files
 * used for model serialization. Model serialization is required due to the nature of the
 * Weight Separation (WS) feature, as it relies on the WeightlessCacheAttribute.
 *
 * This attribute is populated during deserialization, which does not normally occur in
 * typical functional tests.
 */

class WsBaseTest : public VpuOv2LayerTest {
public:
    WsBaseTest(std::string modelName) {
        setupSpecialEnvironment(modelName);
    }

protected:
    /**
     * @brief Remove temporary files created for model serialization.
     */
    void TearDown() override;

private:
    /**
     * @brief Set up the environment for weight separation tests:
     *          - generating unique file paths for model serialization
     *          - set configuration options, i.e. ENABLE_WEIGHTLESS, WEIGHTS_PATH and NPU_COMPILER_TYPE
     */
    void setupSpecialEnvironment(std::string modelName);

protected:
    std::string _xmlPath{};
    std::string _binPath{};
};

}  // namespace test
}  // namespace ov
