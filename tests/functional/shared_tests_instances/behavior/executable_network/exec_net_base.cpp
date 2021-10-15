// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8
};

const std::vector<std::map<std::string, std::string>> configs = {};

// double free detected
// [Track number: S#27337]
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                        ::testing::Combine(
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                         ExecutableNetworkBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                 ::testing::ValuesIn(configs)),
                         ExecNetSetPrecision::getTestCaseName);
}  // namespace
