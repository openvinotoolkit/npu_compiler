//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <common_test_utils/ov_tensor_utils.hpp>

#include <pretty_test_arguments.hpp>
#include "single_op_tests/activation.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using ov::test::ActivationParamLayerTest;
using ShapeMap = std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ActivationLayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ActivationParamLayerTest);

class ActivationLayerTestCommon : public ActivationLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "disabled-passes=convert-precision-to-fp";
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

class ActivationLayerTest_SCFTiling : public ActivationLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "scf-tiling=true";
        // E#190336 for MC support
        configuration["NPU_TILES"] = "1";
    }

    // Generate inputs in a numerically safe range:
    //  * away from fp16 subnormals (|x| >= 2^-10 ~ 9.77e-4) so kernels with FTZ
    //    (Sqrt, Mish, SoftPlus, ...) do not flush small magnitudes to zero
    //    and produce mismatches against the CPU reference;
    //  * bounded in [-2.0, 2.0] so Mish / SoftPlus / Swish stay outside the
    //    near-zero noise floor where SHV-kernel rounding noise dominates.
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::test::utils::InputGenerateData inGenData;
            inGenData.start_from = 1.0;
            inGenData.range = 4;
            inGenData.resolution = 256;  // step ~= range / resolution = 1/64 ~ 0.0156
            inGenData.seed = 1;
            auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                  targetInputStaticShapes[i], inGenData);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

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
        setSkipCompilationCallback([](std::stringstream& skip) {                                       \
            if (Platform::PLATFORM == Platform::NPU5010) {                                             \
                const auto actType = std::get<0>(GetParam());                                          \
                if (actType.first == ActivationTypes::Log) {                                           \
                    skip << "Log issue E#207309";                                                      \
                }                                                                                      \
            }                                                                                          \
        });                                                                                            \
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
        setPluginCompilerType();                                                                   \
        run(Platform::PLATFORM);                                                                   \
    }                                                                                              \
    TEST_P(ShaveCodeGenActivationLayerTest_Profiling, DISABLE_PROFILING##PLATFORM) {               \
        abs_threshold = 0.0056;                                                                    \
        EXTRA_PROF;                                                                                \
        setReferenceSoftwareMode();                                                                \
        setPluginCompilerType();                                                                   \
        enableProfiling();                                                                         \
        run(Platform::PLATFORM);                                                                   \
    }

DEFINE_ACT_TESTS(NPU3720, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);
DEFINE_ACT_TESTS(NPU4000, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);
DEFINE_ACT_TESTS(NPU5010, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);
DEFINE_ACT_TESTS(NPU5020, /*DISABLED_SW_*/, /*DISABLED_HW_*/, /*CONFIG_SW*/, /*CONFIG_HW*/);

DEFINE_DYNAMIC_ACT_TESTS(NPU3720, /*DISABLED_DYN_SW_*/, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);
DEFINE_DYNAMIC_ACT_TESTS(NPU4000, /*DISABLED_DYN_SW_*/, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);
// Tracking number [E#207309]
DEFINE_DYNAMIC_ACT_TESTS(NPU5010, DISABLED_DYN_SW_, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);
DEFINE_DYNAMIC_ACT_TESTS(NPU5020, /*DISABLED_DYN_SW_*/, /*DISABLED_DYN_HW_*/, /*CONFIG_DYN_SW*/, /*CONFIG_DYN_HW*/);

DEFINE_SHAVE_CODE_GEN_TESTS(NPU4000, /*DISABLED_SW_*/, /*DISABLED_PROF_*/, /*CONFIG_SW*/, /*CONFIG_PROF*/);
DEFINE_SHAVE_CODE_GEN_TESTS(NPU5010, /*DISABLED_SW_*/, /*DISABLED_PROF_*/, /*CONFIG_SW*/, /*CONFIG_PROF*/);
DEFINE_SHAVE_CODE_GEN_TESTS(NPU5020, /*DISABLED_SW_*/, /*DISABLED_PROF_*/, /*CONFIG_SW*/, /*CONFIG_PROF*/);

TEST_P(ActivationLayerTest_SCFTiling, NPU4000_SW) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ActivationLayerTest_SCFTiling, NPU5010_SW) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ActivationLayerTest_SCFTiling, NPU5020_SW) {
    abs_threshold = 0.0056;
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

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
        {SoftSign, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationDynamicTypes = {
        {Gelu, {{1.0f}}}, {Atan, {{1.0f}}}, {Cos, {{1.0f}}},     {Sin, {{1.0f}}},
        {Sqrt, {{1.0f}}}, {Log, {{1.0f}}},  {Ceiling, {{1.0f}}},
};
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationDynamicTypesLarge = {
        {Atan, {{1.0f}}},
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
        {Mish, {{1.0f}}},
        {Floor, {{1.0f}}},
        {Ceiling, {{1.0f}}},
        {Asinh, {{1.0f}}},
        {Acosh, {{1.0f}}},
        {Asin, {{1.0f}}},
        {Acos, {{1.0f}}}};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenIntActivationTypes = {
        {Clamp, {{-1.0f, 1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> shaveCodeGenActivationProfilingTypes = {
        {Cos, {{1.0f}}}};

ShapeMap basic = {{{{1, 50, 1, 1}}, {}}, {{{1, 128, 1, 1}}, {}}};

std::vector<ov::test::InputShape> dynamicBasic = {generateTestShape(256_Dyn), generateTestShape(1, 64_Dyn),
                                                  generateTestShape(1, 8_Dyn, 3072), generateTestShape(1, 50_Dyn, 1, 1),
                                                  generateTestShape(1, 128_Dyn, 1, 1)};
std::vector<ov::test::InputShape> dynamicLarge = {generateTestShape(1, 1, 1, 2333333_Dyn)};

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
const auto largeDynamicCasesHWFP16 = genActLessParamsDyn(activationDynamicTypesLarge, dynamicLarge, ov::element::f16);

const auto basicClampI32 = genActLessParams(
        std::map<ActivationTypes, std::vector<std::vector<float>>>{{Clamp, {{-1.0f, 1.0f}}}},
        std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>({{{{1, 50, 1, 1}}, {}}}), ov::element::i32);

const std::map<ActivationTypes, std::vector<std::vector<float>>> floorActivationTypes = {{Floor, {{-3.0f, 6.0f}}}};
const std::vector<ov::element::Type> floorIntegerElementTypes = {ov::element::i32, ov::element::i64};
const auto basicFloorInteger = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(floorActivationTypes)), ::testing::ValuesIn(floorIntegerElementTypes),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(
                std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>({{{{1, 16, 48, 1}}, {}}})))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const std::map<ActivationTypes, std::vector<std::vector<float>>> reluSignActivationTypes = {
        {Relu, {{1.0f}}},
        {Sign, {{1.0f}}},
};
const std::vector<ov::element::Type> reluSignIntegerElementTypes = {
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64,
};
const auto basicReluSignInteger = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(reluSignActivationTypes)), ::testing::ValuesIn(reluSignIntegerElementTypes),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(
                std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>({{{{1, 16, 48, 1}}, {}}})))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const std::map<ActivationTypes, std::vector<std::vector<float>>> absActivationTypes = {{Abs, {{1.0f}}}};
const std::vector<ov::element::Type> absIntegerElementTypes = {ov::element::i8, ov::element::i16, ov::element::i32};
const auto basicAbsInteger = ::testing::Combine(
        ::testing::ValuesIn(::combineParams(absActivationTypes)), ::testing::ValuesIn(absIntegerElementTypes),
        ::testing::ValuesIn(staticShapesParamTransform(ov::test::utils::combineParams(
                std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>({{{{1, 16, 48, 1}}, {}}})))),
        ::testing::Values(test_utils::TARGET_DEVICE));

// --------------------- 3720+ ---------------------
// ------ NPU SW  ------
INSTANTIATE_TEST_SUITE_P(precommit_Act_fp16, ActivationLayerTest_SW_FP, basicCasesSWFP16,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_fp32, ActivationLayerTest_SW_FP, basicCasesSWFP32,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu, ActivationLayerTest_SW_FP, basicPReluBasicCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Act_PRelu_tiling, ActivationLayerTest_SW_FP, basicTilingPReluCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_PRelu_Slope_param, ActivationLayerTest_SW_FP, basicPReluParamCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_2D, ActivationLayerTest_SW_FP, basicCases2D,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Act_tiling_Sw, ActivationLayerTest_SW_FP, basicTilingCases,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_Floor_Integer, ActivationLayerTest_SW_FP, basicFloorInteger,
                         ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Act_ReluSign_Integer, ActivationLayerTest_SW_FP, basicReluSignInteger,
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
INSTANTIATE_TEST_SUITE_P(precommit_Abs_Int, ShaveCodeGenActivationLayerTest, basicAbsInteger,
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
INSTANTIATE_TEST_SUITE_P(smoke_Act_Large_Dynamic, DynamicActivationLayerTest_HW_FP16, largeDynamicCasesHWFP16,
                         ActivationLayerTest::getTestCaseName);

// Verify Log on fp16 does not produce NaN when input is in the subnormal range
// that NPU FTZ flushes to zero. The kernel clamps to fp16 min-normal (2^-14).
class LogSubnormalLayerTest : public ActivationLayerTestCommon {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& param = function->get_parameters()[0];
        auto tensor = ov::Tensor(param->get_element_type(), targetInputStaticShapes[0]);
        // Use a spread of subnormal fp16 values to cover the full subnormal range:
        //   0x0001 (~5.96e-8) smallest subnormal, 0x0200 (~3.05e-5) mid-range,
        //   1e-6f, 6.0e-5f (just below min-normal 2^-14 ~6.10e-5).
        // All are flushed to 0 by NPU FTZ without the clamp fix.
        static const float kSubnormals[] = {5.96e-8f, 1e-6f, 3.05e-5f, 6.0e-5f};
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            data[i] = ov::float16(kSubnormals[i % 4]);
        }
        inputs.insert({param, tensor});
    }

    void validate() override {
        // Expected: inputs clamped to fp16 min-normal (2^-14) before log, so output ~ log(2^-14).
        const float minNormalF16 = std::ldexp(1.0f, -14);
        const float expectedLog = std::log(minNormalF16);
        const float absTolerance = 1e-2f;

        const auto actualOutputs = get_plugin_outputs();
        ASSERT_FALSE(actualOutputs.empty());
        for (const auto& out : actualOutputs) {
            ASSERT_EQ(out.get_element_type(), ov::element::f16);
            const auto* data = out.data<ov::float16>();
            for (size_t i = 0; i < out.get_size(); ++i) {
                const float value = static_cast<float>(data[i]);
                ASSERT_TRUE(std::isfinite(value)) << "non-finite at index " << i;
                ASSERT_NEAR(value, expectedLog, absTolerance) << "unexpected log result at index " << i;
            }
        }
    }
};

TEST_P(LogSubnormalLayerTest, NPU3720) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}
TEST_P(LogSubnormalLayerTest, NPU4000) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(LogSubnormalLayerTest, NPU5010) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
TEST_P(LogSubnormalLayerTest, NPU5020) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

const auto logSubnormalCases = ::testing::Combine(
        ::testing::Values(std::pair<ActivationTypes, std::vector<float>>{Log, {1.0f}}),
        ::testing::Values(ov::element::f16),
        ::testing::ValuesIn(staticShapesParamTransform(
                ov::test::utils::combineParams(std::map<std::vector<ov::Shape>, std::vector<ov::Shape>>{
                        {{{1, 64, 1, 1}}, {}},
                }))),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_Log_subnormal, LogSubnormalLayerTest, logSubnormalCases,
                         ActivationLayerTest::getTestCaseName);

//
// SCF Tiling
//

const std::map<ActivationTypes, std::vector<std::vector<float>>> scfTilingActivationTypes = {
        {Abs, {{1.0f}}},
        {Sin, {{1.0f}}},
        {Sqrt, {{1.0f}}},
        {Tanh, {{1.0f}}},
        {Mish, {{1.0f}}},
        {Relu, {{1.0f}}},
        {Sigmoid, {{1.0f}}},
        {SoftPlus, {{1.0f}}},
        {Swish, {{1.0f}}},
        {Acos, {{1.0f}}},
        {Acosh, {{1.0f}}},
        {Asin, {{1.0f}}},
        {Asinh, {{1.0f}}},
        {Atan, {{1.0f}}},
        {Atanh, {{1.0f}}},
        {Ceiling, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}},
        {Cos, {{1.0f}}},
        {Cosh, {{1.0f}}},
        {Erf, {{1.0f}}},
        {Exp, {{1.0f}}},
        {Floor, {{1.0f}}},
        {Log, {{1.0f}}},
        {Negative, {{1.0f}}},
        {RoundHalfToEven, {}},
        {Sign, {{1.0f}}},
        {Sinh, {{1.0f}}},
        {Tan, {{1.0f}}},
        {Elu, {{1.0f}}},
        {HSwish, {{1.0f}}},
        {HardSigmoid, {{0.2f, 0.5f}}},
        {HSigmoid, {{1.0f}}},
        {LeakyRelu, {{0.01f}}},
        {Selu, {{1.6732f, 1.0507f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> scfTilingPReluTypes = {
        {PReLu, {{0.01f}}},
};

const auto scfTilingCases = genActLessParams(scfTilingActivationTypes, basic, ov::element::f16);
const auto scfTilingPReluCases = genActLessParams(scfTilingPReluTypes, preluTiling, ov::element::f16);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Act_SCFTiling, ActivationLayerTest_SCFTiling, scfTilingCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_PReLu_SCFTiling, ActivationLayerTest_SCFTiling, scfTilingPReluCases,
                         ActivationLayerTest::getTestCaseName);

}  // namespace
