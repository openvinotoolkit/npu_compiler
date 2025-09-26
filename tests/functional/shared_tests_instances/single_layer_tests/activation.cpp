//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/activation.hpp"
#include <pretty_test_arguments.hpp>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using ov::test::ActivationParamLayerTest;

namespace ov {
namespace test {

class ActivationLayerTestCommon : public ActivationLayerTest, virtual public VpuOv2LayerTest {};

class ActivationLayerTest_FP32 : public ActivationLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

class ActivationLayerTest_SW_FP16 : public ActivationLayerTestCommon {};
class ActivationLayerTest_HW_FP16 : public ActivationLayerTestCommon {};
class DynamicActivationLayerTest_SW_FP16 : public ActivationLayerTestCommon {};
class DynamicActivationLayerTest_HW_FP16 : public ActivationLayerTestCommon {};

class ActivationLayerTest_SW_FP32 : public ActivationLayerTest_FP32 {};
class ActivationLayerTest_HW_FP32 : public ActivationLayerTest_FP32 {};

class ShaveCodeGenActivationLayerTest_FP16_Profiling : public ActivationLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};
class ShaveCodeGenActivationLayerTest_FP16 : public ActivationLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};
class ShaveCodeGenActivationLayerTest_Integer : public ActivationLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

// 3720
// SW
TEST_P(ActivationLayerTest_SW_FP16, NPU3720) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ActivationLayerTest_SW_FP32, NPU3720) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicActivationLayerTest_SW_FP16, NPU3720) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

// HW
TEST_P(ActivationLayerTest_HW_FP16, NPU3720) {
    abs_threshold = 0.0056;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ActivationLayerTest_HW_FP32, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicActivationLayerTest_HW_FP16, NPU3720) {
    abs_threshold = 0.0056;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

// 4000
// SW
TEST_P(ActivationLayerTest_SW_FP16, NPU4000) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ActivationLayerTest_SW_FP32, NPU4000) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicActivationLayerTest_SW_FP16, NPU4000) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

// HW
TEST_P(ActivationLayerTest_HW_FP32, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicActivationLayerTest_HW_FP16, NPU4000) {
    abs_threshold = 0.0056;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// Enable test in CI on Linux for the moment
TEST_P(ShaveCodeGenActivationLayerTest_FP16, NPU4000) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenActivationLayerTest_FP16_Profiling, NPU4000) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    enableProfiling();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenActivationLayerTest_Integer, NPU4000) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::f16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid, {{1.0f}}},
        {Sign, {{1.0f}}},
        {Tanh, {{1.0f}}},
        {Sin, {{1.0f}}},
        {Cos, {{1.0f}}},
        {Relu, {{1.0f}}},
        {Elu, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}},
        {HSwish, {{1.0f}}},
        {Mish, {{1.0f}}},
        {SoftPlus, {{1.0f}}},
        {Floor, {{1.0f}}},
        {Sqrt, {{1.0f}}},
        {Sinh, {{1.0f}}},
        {Cosh, {{1.0f}}},
        {Asinh, {{1.0f}}},
        {Acosh, {{1.0f}}},
        {Atanh, {{1.0f}}},
        {Erf, {{1.0f}}},
        {Gelu, {{1.0f}}},
        {Exp, {{1.0f}}},
        {Log, {{1.0f}}},
        {Selu, {{1.6732f, 1.0507f}}},
        {Swish, {{1.0f}}},
        {Negative, {{1.0f}}},
        {Abs, {{1.0f}}},
        {Atan, {{1.0f}}},
        {Asin, {{1.0f}}},
        {Acos, {{1.0f}}},
        {HSigmoid, {{1.0f}}},
        {HardSigmoid, {{0.2f, 0.5f}}},
        {RoundHalfToEven, {}},
        {RoundHalfAwayFromZero, {}},
        {Ceiling, {{1.0f}}},
        {Tan, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationDynamicTypes = {
        {Gelu, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> preluTypes = {
        {PReLu, {{0.01f}}},
        {LeakyRelu, {{0.01f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes2D = {
        {HSigmoid, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesTiling = {
        {Sigmoid, {{1.0f}}},      {Elu, {{1.0f}}},       {Sqrt, {{1.0f}}},       {Exp, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}}, {Tanh, {{1.0f}}},      {LeakyRelu, {{0.01f}}}, {Log, {{1.0f}}},
        {Relu, {{1.0f}}},         {Negative, {{0.01f}}}, {Ceiling, {{1.0f}}}};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesHWFP32 = {
        {Log, {{1.0f}}},
        {Abs, {{1.0f}}},
        {Sqrt, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesSWFP32 = {
        {Log, {{1.0f}}},
        {Relu, {{1.0f}}},
        {Exp, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}},
};

const std::vector<ov::element::Type> integerNetPrecisions = {ov::element::i32};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenActivationTypes = {
        {Cos, {{1.0f}}},          {Exp, {{1.0f}}},   {Log, {{1.0f}}},       {Sin, {{1.0f}}},
        {Erf, {{1.0f}}},          {Sqrt, {{1.0f}}},  {RoundHalfToEven, {}}, {RoundHalfAwayFromZero, {}},
        {Clamp, {{-1.0f, 1.0f}}}, {Tanh, {{1.0f}}},  {Tan, {{1.0f}}},       {Sinh, {{1.0f}}},
        {Cosh, {{1.0f}}},         {Atanh, {{1.0f}}}, {Atan, {{1.0f}}},      {Abs, {{1.0f}}},
        {Negative, {{0.01f}}},    {Sign, {{1.0f}}},  {HSwish, {{1.0f}}},    {HSigmoid, {{1.0f}}}};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenIntActivationTypes = {
        {Clamp, {{-1.0f, 1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenActivationProfilingTypes = {
        {Cos, {{1.0f}}}};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basic = {{{{1, 50, 1, 1}}, {}}, {{{1, 128, 1, 1}}, {}}};

std::vector<ov::test::InputShape> dynamicBasic = {generateTestShape(256_Dyn), generateTestShape(1, 64_Dyn),
                                                  generateTestShape(1, 8_Dyn, 3072), generateTestShape(1, 50_Dyn, 1, 1),
                                                  generateTestShape(1, 128_Dyn, 1, 1)};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basicCos = {{{{1, 1, 50, 120}}, {}}, {{{1, 20, 50, 150}}, {}}};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> preluBasic = {
        {{{1, 50, 1, 1}}, {{50}}},
        {{{1, 128, 1, 1}}, {{128}}},
        {{{1, 32, 96, 96}}, {{32}}},
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> preluTiling = {
        {{{1, 9, 80, 1280}}, {{9}}},
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basic2DShape = {
        {{{120, 50}}, {}}, {{{90, 128}}, {}}, {{{21, 30}}, {}}};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basicTiling = {{{{1, 8, 80, 1280}}, {}},
                                                                        {{{1, 320, 1, 1280}}, {}}};

auto staticShapesParamTransform = [](const std::vector<std::pair<std::vector<ov::Shape>, ov::Shape>>& originalShapes) {
    std::vector<std::pair<std::vector<ov::test::InputShape>, ov::Shape>> newShapes;
    for (const auto& shapeElement : originalShapes) {
        newShapes.emplace_back(ov::test::static_shapes_to_test_representation(shapeElement.first), shapeElement.second);
    }
    return newShapes;
};

auto dynamicShapesParamTransform = [](const std::vector<ov::test::InputShape>& originalShapes) {
    std::vector<std::pair<std::vector<ov::test::InputShape>, ov::Shape>> newShapes;
    for (const auto& shape : originalShapes) {
        newShapes.emplace_back(std::vector<ov::test::InputShape>{shape}, ov::Shape{});
    }
    return newShapes;
};

const auto basicCases =
        ::testing::Combine(::testing::ValuesIn(::combineParams(activationTypes)),  // Activation type and constant
                           ::testing::ValuesIn(netPrecisions),                     // Model type
                           ::testing::ValuesIn(staticShapesParamTransform(
                                   ov::test::utils::combineParams(basic))),  // Input shapes and input const shape
                           ::testing::Values(test_utils::TARGET_DEVICE));    // Target device name

const auto basicCosCases = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(shaveCodeGenActivationProfilingTypes)),  // Activation type and constant
        ::testing::ValuesIn(netPrecisions),                                          // Model type
        ::testing::ValuesIn(staticShapesParamTransform(
                ov::test::utils::combineParams(basicCos))),  // Input shapes and input const shape
        ::testing::Values(test_utils::TARGET_DEVICE));       // Target device name

const auto basicCasesSWFP32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSWFP32)), ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basic))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicCasesHWFP32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesHWFP32)), ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basicTiling))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicPReluCases =
        ::testing::Combine(::testing::ValuesIn(::combineParams(preluTypes)), ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(preluBasic))),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicTilingPReluCases =
        ::testing::Combine(::testing::ValuesIn(::combineParams(preluTypes)), ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(preluTiling))),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicCases2D = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(activationTypes2D)), ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basic2DShape))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicTilingCases = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(activationTypesTiling)), ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basicTiling))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicShaveCodeGenCases = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(shaveCodeGenActivationTypes)), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basic))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicShaveCodeGenIntCases = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(shaveCodeGenIntActivationTypes)), ::testing::ValuesIn(integerNetPrecisions),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basic))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicClampI32 = ::testing::Combine(
        ::testing::ValuesIn(
                ::combineParams(std::map<ActivationTypes, std::vector<std::vector<float>>>{{Clamp, {{-1.0f, 1.0f}}}})),
        ::testing::ValuesIn(integerNetPrecisions),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(
                std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>({{{{1, 50, 1, 1}}, {}}})))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicDynamicCasesSWFP16 = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(activationDynamicTypes)), ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(dynamicShapesParamTransform(dynamicBasic)), ::testing::Values(test_utils::TARGET_DEVICE));

const auto basicDynamicCasesHWFP16 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationDynamicTypes)),
        ::testing::Values(ov::element::f16), ::testing::ValuesIn(dynamicShapesParamTransform(dynamicBasic)),
        ::testing::Values(test_utils::TARGET_DEVICE));

// ------ NPU SW FP16 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation, ActivationLayerTest_SW_FP16, basicCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_PRelu, ActivationLayerTest_SW_FP16, basicPReluCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_tiling_Activation_PRelu, ActivationLayerTest_SW_FP16, basicTilingPReluCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_2D, ActivationLayerTest_SW_FP16, basicCases2D,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Activation, ActivationLayerTest_SW_FP16, basicTilingCases,
                         ActivationLayerTest::getTestCaseName);

// ------ ShaveCodeGen ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_Cos0_ShaveCodeGen_Profiling,
                         ShaveCodeGenActivationLayerTest_FP16_Profiling, basicCosCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_Int_ShaveCodeGen, ShaveCodeGenActivationLayerTest_Integer,
                         basicShaveCodeGenIntCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_ShaveCodeGen, ShaveCodeGenActivationLayerTest_FP16,
                         basicShaveCodeGenCases, ActivationLayerTest::getTestCaseName);

// ------ NPU3720 HW FP16 ------

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Activation, ActivationLayerTest_HW_FP16, basicTilingCases,
                         ActivationLayerTest::getTestCaseName);

// ------ NPU SW FP32 ------

INSTANTIATE_TEST_SUITE_P(smoke_Activation, ActivationLayerTest_SW_FP32, basicCasesSWFP32,
                         ActivationLayerTest::getTestCaseName);

// ------ NPU HW FP32 ------
INSTANTIATE_TEST_SUITE_P(smoke_tiling_Activation, ActivationLayerTest_HW_FP32, basicCasesHWFP32,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Clamp_I32, ActivationLayerTest_HW_FP32, basicClampI32,
                         ActivationLayerTest::getTestCaseName);

// ------ [DYNAMIC] NPU FP16 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Dynamic, DynamicActivationLayerTest_SW_FP16,
                         basicDynamicCasesSWFP16, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Dynamic, DynamicActivationLayerTest_HW_FP16, basicDynamicCasesHWFP16,
                         ActivationLayerTest::getTestCaseName);

}  // namespace
