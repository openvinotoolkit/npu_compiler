//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/power.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPowerLayerTest : public PowerLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
        }
    }
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::KmbSkipTestException("comparison fails");
    }
};

TEST_P(KmbPowerLayerTest, PowerCheck) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {{{1, 8}},   {{2, 16}},  {{3, 32}},  {{4, 64}},
                                                          {{5, 128}}, {{6, 256}}, {{7, 512}}, {{8, 1024}}};

std::vector<std::vector<float>> Power = {
        {0.0f}, {0.5f}, {1.0f}, {1.1f}, {1.5f}, {2.0f},
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

// Test works only when power = 1, in all other cases there are similar errors:
// C++ exception with description "Operation PowerIE Power_xxxxx has unsupported power N (where N unequals to 1)
// vpux-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:640
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#41811]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_power, KmbPowerLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::ValuesIn(Power)),
                         KmbPowerLayerTest::getTestCaseName);

}  // namespace
