// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid, {}},
    {Tanh,    {}},
    {Relu,    {}},
    {Exp,     {}},
    {Log,     {}},
    {Sign,    {}},
    {Abs,      {}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50}, {}},
    {{1, 128}, {}},
};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY));

INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
