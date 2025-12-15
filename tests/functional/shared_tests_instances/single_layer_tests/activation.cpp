//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <common_test_utils/ov_tensor_utils.hpp>

#include <pretty_test_arguments.hpp>
#include "single_op_tests/activation.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using ov::test::ActivationParamLayerTest;
using ShapeMap = std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>;

namespace ov {
namespace test {

class ActivationLayerTestCommon : public ActivationLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

class ActivationLayerTest_SW_FP : public ActivationLayerTestCommon {};
class ActivationLayerTest_HW_FP : public ActivationLayerTestCommon {};

class DynamicActivationLayerTest_SW_FP16 : public ActivationLayerTestCommon {};
class DynamicActivationLayerTest_HW_FP16 : public ActivationLayerTestCommon {};

class ShaveCodeGenActivationLayerTest : public ActivationLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

class ShaveCodeGenActivationLayerTest_Profiling : public ShaveCodeGenActivationLayerTest {};

#define DEFINE_ACT_TESTS(PLATFORM, DISABLE_SW, DISABLE_HW, EXTRA_SW, EXTRA_HW) \
    TEST_P(ActivationLayerTest_SW_FP, DISABLE_SW##PLATFORM) {                  \
        abs_threshold = 0.0056;                                                \
        EXTRA_SW;                                                              \
        setReferenceSoftwareMode();                                            \
        run(Platform::PLATFORM);                                               \
    }                                                                          \
    TEST_P(ActivationLayerTest_HW_FP, DISABLE_HW##PLATFORM) {                  \
        abs_threshold = 0.0056;                                                \
        EXTRA_HW;                                                              \
        setDefaultHardwareMode();                                              \
        run(Platform::PLATFORM);                                               \
    }

#define DEFINE_DYNAMIC_ACT_TESTS(PLATFORM, DISABLE_DYN_SW, DISABLE_DYN_HW, EXTRA_DYN_SW, EXTRA_DYN_HW) \
    TEST_P(DynamicActivationLayerTest_SW_FP16, DISABLE_DYN_SW##PLATFORM) {                             \
        abs_threshold = 0.0056;                                                                        \
        EXTRA_DYN_SW;                                                                                  \
        setReferenceSoftwareMode();                                                                    \
        run(Platform::PLATFORM);                                                                       \
    }                                                                                                  \
    TEST_P(DynamicActivationLayerTest_HW_FP16, DISABLE_DYN_HW##PLATFORM) {                             \
        abs_threshold = 0.0056;                                                                        \
        EXTRA_DYN_HW;                                                                                  \
        setDefaultHardwareMode();                                                                      \
        run(Platform::PLATFORM);                                                                       \
    }

#define DEFINE_SHAVE_CODE_GEN_TESTS(PLATFORM, DISABLE_SW, DISABLE_PROFILING, EXTRA_SW, EXTRA_PROF) \
    TEST_P(ShaveCodeGenActivationLayerTest, DISABLE_SW##PLATFORM) {                                \
        const auto type = std::get<1>(GetParam());                                                 \
        if (type == ov::element::f16) {                                                            \
            abs_threshold = 0.0056;                                                                \
        }                                                                                          \
        EXTRA_SW;                                                                                  \
        setReferenceSoftwareMode();                                                                \
        setMLIRCompilerType();                                                                     \
        run(Platform::PLATFORM);                                                                   \
    }                                                                                              \
    TEST_P(ShaveCodeGenActivationLayerTest_Profiling, DISABLE_PROFILING##PLATFORM) {               \
        abs_threshold = 0.0056;                                                                    \
        EXTRA_PROF;                                                                                \
        setReferenceSoftwareMode();                                                                \
        setMLIRCompilerType();                                                                     \
        enableProfiling();                                                                         \
        run(Platform::PLATFORM);                                                                   \
    }

DEFINE_ACT_TESTS(NPU3720, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);
DEFINE_ACT_TESTS(NPU4000, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);
DEFINE_ACT_TESTS(NPU5010, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);

DEFINE_DYNAMIC_ACT_TESTS(NPU3720, /*DISABLED_DYN_SW_*/, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);
DEFINE_DYNAMIC_ACT_TESTS(NPU4000, /*DISABLED_DYN_SW_*/, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);
DEFINE_DYNAMIC_ACT_TESTS(NPU5010, /*DISABLED_DYN_SW_*/, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);

DEFINE_SHAVE_CODE_GEN_TESTS(NPU4000, /*DISABLED_SW_*/, /*DISABLED_PROF_*/, /*CONFIG_SW*/, /*CONFIG_PROF*/);
DEFINE_SHAVE_CODE_GEN_TESTS(NPU5010, /*DISABLED_SW_*/, /*DISABLED_PROF_*/, /*CONFIG_SW*/, /*CONFIG_PROF*/);

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

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

const auto genActLessParams = [](auto activationTypes, auto basic, auto type) {
    return ::testing::Combine(
            ::testing::ValuesIn(::combineParams(activationTypes)),                                   // Activation type
            ::testing::Values(type),                                                                 // Model type
            ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(basic))),  // Input shapes
            ::testing::Values(test_utils::TARGET_DEVICE));
};

const auto genActLessParamsDyn = [](auto activationTypes, auto dynamicShapes, auto type) {
    return ::testing::Combine(::testing::ValuesIn(::combineParams(activationTypes)),            // Activation type
                              ::testing::Values(type),                                          // Model type
                              ::testing::ValuesIn(dynamicShapesParamTransform(dynamicShapes)),  // Input shapes
                              ::testing::Values(test_utils::TARGET_DEVICE));
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesSWFP16 = {
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
        {Gelu, {{1.0f}}}, {Atan, {{1.0f}}}, {Cos, {{1.0f}}}, {Sin, {{1.0f}}}, {Sqrt, {{1.0f}}}, {Log, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> preluParamTypes = {
        {PReLu, {{}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> preluConstTypes = {
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

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenActivationTypes = {
        {Cos, {{1.0f}}},
        {Exp, {{1.0f}}},
        {Log, {{1.0f}}},
        {Sin, {{1.0f}}},
        {Erf, {{1.0f}}},
        {Sqrt, {{1.0f}}},
        {RoundHalfToEven, {}},
        {RoundHalfAwayFromZero, {}},
        {Clamp, {{-1.0f, 1.0f}}},
        {Tanh, {{1.0f}}},
        {Tan, {{1.0f}}},
        {Sinh, {{1.0f}}},
        {Cosh, {{1.0f}}},
        {Atanh, {{1.0f}}},
        {Atan, {{1.0f}}},
        {Abs, {{1.0f}}},
        {Negative, {{0.01f}}},
        {Sign, {{1.0f}}},
        {HSwish, {{1.0f}}},
        {HSigmoid, {{1.0f}}},
        {Elu, {{1.0f}}},
        {Gelu, {{1.0f}}},
        {Selu, {{1.6732f, 1.0507f}}},
        {PReLu, {{0.01f}}},
        {SoftPlus, {{1.0f}}},
        {Mish, {{1.0f}}}};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenIntActivationTypes = {
        {Clamp, {{-1.0f, 1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenActivationProfilingTypes = {
        {Cos, {{1.0f}}}};

ShapeMap basic = {{{{1, 50, 1, 1}}, {}}, {{{1, 128, 1, 1}}, {}}};

std::vector<ov::test::InputShape> dynamicBasic = {generateTestShape(256_Dyn), generateTestShape(1, 64_Dyn),
                                                  generateTestShape(1, 8_Dyn, 3072), generateTestShape(1, 50_Dyn, 1, 1),
                                                  generateTestShape(1, 128_Dyn, 1, 1)};

ShapeMap profilingBasic = {{{{1, 1, 50, 120}}, {}}, {{{1, 20, 50, 150}}, {}}};

ShapeMap preluBasicShapes = {
        {{{1, 50, 1, 1}}, {{50}}},
        {{{1, 128, 1, 1}}, {{128}}},
        {{{1, 32, 96, 96}}, {{32}}},
};

ShapeMap preluTiling = {
        {{{1, 9, 80, 1280}}, {{9}}},
};

ShapeMap preluParamShapes = {{{{1, 32, 42, 43}}, {{1, 32, 42, 43}}}};

ShapeMap basic2DShape = {{{{120, 50}}, {}}, {{{90, 128}}, {}}, {{{21, 30}}, {}}};

ShapeMap basicTiling = {{{{1, 8, 80, 1280}}, {}}, {{{1, 320, 1, 1280}}, {}}};

const auto basicCasesSWFP16 = genActLessParams(activationTypesSWFP16, basic, ov::element::f16);
const auto basicCasesSWFP32 = genActLessParams(activationTypesSWFP32, basic, ov::element::f32);
const auto basicCasesHWFP32 = genActLessParams(activationTypesHWFP32, basicTiling, ov::element::f32);
const auto profilingCases = genActLessParams(shaveCodeGenActivationProfilingTypes, profilingBasic, ov::element::f16);

const auto basicPReluBasicCases = genActLessParams(preluConstTypes, preluBasicShapes, ov::element::f16);
const auto basicPReluParamCases = genActLessParams(preluParamTypes, preluParamShapes, ov::element::f16);
const auto basicTilingPReluCases = genActLessParams(preluConstTypes, preluTiling, ov::element::f16);
const auto basicCases2D = genActLessParams(activationTypes2D, basic2DShape, ov::element::f16);
const auto basicTilingCases = genActLessParams(activationTypesTiling, basicTiling, ov::element::f16);
const auto basicShaveCodeGenFpCases = genActLessParams(shaveCodeGenActivationTypes, basic, ov::element::f16);
const auto basicShaveCodeGenIntCases = genActLessParams(shaveCodeGenIntActivationTypes, basic, ov::element::i32);
const auto basicDynamicCasesSWFP16 = genActLessParamsDyn(activationDynamicTypes, dynamicBasic, ov::element::f16);
const auto basicDynamicCasesHWFP16 = genActLessParamsDyn(activationDynamicTypes, dynamicBasic, ov::element::f16);

const auto basicClampI32 = genActLessParams(
        std::map<ActivationTypes, std::vector<std::vector<float>>>{{Clamp, {{-1.0f, 1.0f}}}},
        std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>({{{{1, 50, 1, 1}}, {}}}), ov::element::i32);

// --------------------- 3720+ ---------------------
// ------ NPU SW  ------
INSTANTIATE_TEST_SUITE_P(precommit_Act_fp16, ActivationLayerTest_SW_FP, basicCasesSWFP16,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_fp32, ActivationLayerTest_SW_FP, basicCasesSWFP32,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu, ActivationLayerTest_SW_FP, basicPReluBasicCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu_tiling, ActivationLayerTest_SW_FP, basicTilingPReluCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu_Slope_param, ActivationLayerTest_SW_FP, basicPReluParamCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_2D, ActivationLayerTest_SW_FP, basicCases2D,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Act_tiling_Sw, ActivationLayerTest_SW_FP, basicTilingCases,
                         ActivationLayerTest::getTestCaseName);

// ------ NPU HW ------
INSTANTIATE_TEST_SUITE_P(smoke_Act_fp32, ActivationLayerTest_HW_FP, basicCasesHWFP32,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Clamp_I32, ActivationLayerTest_HW_FP, basicClampI32,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Act_tiling_Hw, ActivationLayerTest_HW_FP, basicTilingCases,
                         ActivationLayerTest::getTestCaseName);

// --------------------- 4000+ ---------------------
// ------ ShaveCodeGen ------
INSTANTIATE_TEST_SUITE_P(precommit_Cos0_Prof, ShaveCodeGenActivationLayerTest_Profiling, profilingCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Int, ShaveCodeGenActivationLayerTest, basicShaveCodeGenIntCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Basic, ShaveCodeGenActivationLayerTest, basicShaveCodeGenFpCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu, ShaveCodeGenActivationLayerTest, basicPReluBasicCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu_Slope_param, ShaveCodeGenActivationLayerTest, basicPReluParamCases,
                         ActivationLayerTest::getTestCaseName);

// --------------------- 3720+ ---------------------
// ------ [DYNAMIC] NPU FP16 ------
INSTANTIATE_TEST_SUITE_P(precommit_Act_Dynamic, DynamicActivationLayerTest_SW_FP16, basicDynamicCasesSWFP16,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Act_Dynamic, DynamicActivationLayerTest_HW_FP16, basicDynamicCasesHWFP16,
                         ActivationLayerTest::getTestCaseName);

}  // namespace
