// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {
namespace {
std::set<ngraph::helpers::ActivationTypes> supportedTypesMCM {
    ngraph::helpers::Relu,
    ngraph::helpers::Sigmoid,
    ngraph::helpers::HSwish,
    ngraph::helpers::Swish,
    ngraph::helpers::Tanh,
    ngraph::helpers::SoftPlus,
    ngraph::helpers::Elu,
    ngraph::helpers::Mish,
    ngraph::helpers::Floor,
    ngraph::helpers::RoundHalfToEven,
    ngraph::helpers::RoundHalfAwayFromZero,
    ngraph::helpers::Erf,
    ngraph::helpers::Gelu,
    ngraph::helpers::Log,
    ngraph::helpers::Ceiling,
    ngraph::helpers::Exp,
    ngraph::helpers::PReLu,
};

std::set<ngraph::helpers::ActivationTypes> supportedTypesMLIR {
    ngraph::helpers::Relu,
    ngraph::helpers::Sigmoid,
    ngraph::helpers::Clamp,
    ngraph::helpers::Elu,
    ngraph::helpers::HSwish,
    ngraph::helpers::Floor,
    ngraph::helpers::Mish,
    ngraph::helpers::Erf,
    ngraph::helpers::Tanh,
    ngraph::helpers::PReLu,
    ngraph::helpers::LeakyRelu,
    ngraph::helpers::Swish,
    ngraph::helpers::Negative,
    ngraph::helpers::Exp,
    ngraph::helpers::RoundHalfToEven,
    ngraph::helpers::RoundHalfAwayFromZero,
    ngraph::helpers::Sqrt,
    ngraph::helpers::Log,
    ngraph::helpers::Ceiling,
};

} // namespace

class KmbActivationLayerTest : public ActivationLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationParam;
        std::tie(activationParam,
                 std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore, std::ignore, std::ignore) = GetParam();

        const auto activationType = activationParam.first;

        if (isCompilerMCM()) {
            if (supportedTypesMCM.find(activationType) == supportedTypesMCM.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Unsupported activation types in MCM compiler");
            }
        } else {
            if (supportedTypesMLIR.find(activationType) == supportedTypesMLIR.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Experimental compiler doesn't supports activation type " +
                                                            LayerTestsDefinitions::activationNames[activationType] +
                                                            " yet");
            }
        }

        // [Track number: #E20853]
        if (getBackendName(*getCore()) == "LEVEL0") {
                throw LayerTestsUtils::KmbSkipTestException("Level0: sporadic failures on device");
            }
    }
};

class KmbActivationLayerTest_MTL : public KmbActivationLayerTest {
    void SkipBeforeLoad() override {
#if defined(__arm__) || defined(__aarch64__)
        throw LayerTestsUtils::KmbSkipTestException("Does not compile on ARM.");
#endif
    }
    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Runtime issue.");
    }
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::KmbSkipTestException("Runtime issue.");
    }
};

TEST_P(KmbActivationLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbActivationLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbActivationLayerTest_MTL, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setPlatformMTL();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid,  {{1.0f}}},
    {Tanh,     {{1.0f}}},
    {Relu,     {{1.0f}}},
    {Elu,      {{1.0f}}},
    {Clamp,    {{-1.0f, 1.0f}}},
    {HSwish,   {{1.0f}}},
    {Mish,     {{1.0f}}},
    {SoftPlus, {{1.0f}}},
    {Mish,     {{1.0f}}},
    {Floor,    {{1.0f}}},
    {Sqrt,     {{1.0f}}},
    {Erf,      {{1.0f}}},
    {Gelu,     {{1.0f}}},
    {Exp,      {{1.0f}}},
    {Log,      {{1.0f}}},
    {Swish,    {{1.0f}}},
    {Negative, {{1.0f}}},
    {RoundHalfToEven,       {}},
    {RoundHalfAwayFromZero, {}},
#if 0 // Unsupported layers
    {Sign,     {{1.0f}}},
    {Abs,      {{1.0f}}},
#endif
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
    {PReLu,    {{0.01f}}},
    {LeakyRelu,{{0.01f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesND = {
    {Sigmoid,  {{1.0f}}},
    {Tanh,     {{1.0f}}},
    {Relu,     {{1.0f}}},
    {Elu,      {{1.0f}}},
    {Clamp,    {{-1.0f, 1.0f}}},
    {HSwish,   {{1.0f}}},
    {Exp,      {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesFP16Only = {
    {Ceiling,  {{1.0f}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50, 1, 1}, {{}}},
    {{1, 128, 1, 1}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
    {{1, 50, 1, 1}, {{1}, {50}}},
    {{1, 128, 1, 1}, {{1}, {128}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicNDCase = {
    {{1, 50}, {{}}},
    {{1, 128, 1}, {{}}},
};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto basicPReluCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto basicNDCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesND)),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::ValuesIn(CommonTestUtils::combineParams(basicNDCase)),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

// For operations that only support FP16 input values in 'vpuip_2'
const auto basicFP16OnlyCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesFP16Only)),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::FP16),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test, KmbActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test_PRelu, KmbActivationLayerTest, basicPReluCases, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test_ND, KmbActivationLayerTest, basicNDCases, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test_FP16Only, KmbActivationLayerTest, basicFP16OnlyCases, ActivationLayerTest::getTestCaseName);

// ------ MTL ------

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicShapesMTL = {
        {{1, 1, 1, 1000}, {{}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesMTL = {
        {Sigmoid,  {{1.0f}}},
};

const auto basicCasesMTL = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesMTL)),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basicShapesMTL)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test, KmbActivationLayerTest_MTL, basicCasesMTL, ActivationLayerTest::getTestCaseName);

}  // namespace
