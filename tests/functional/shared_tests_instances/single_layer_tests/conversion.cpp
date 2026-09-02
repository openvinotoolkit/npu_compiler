//
// Copyright (C) 2019-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/conversion.hpp"
#include <openvino/core/type/element_type.hpp>
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using namespace ov::element;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConversionLayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConversionSpecifyInputLayerTest);

class ConversionLayerTestCommon : public ConversionLayerTest, virtual public VpuOv2LayerTest {};
class ConversionLayerTestCommon_HW : public ConversionLayerTest, virtual public VpuOv2LayerTest {};

class ShaveCodeGenConversionLayerTest : public ConversionLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

TEST_P(ConversionLayerTestCommon_HW, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConversionLayerTestCommon_HW, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenConversionLayerTest, NPU4000) {
    setReferenceSoftwareMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ConversionLayerTestCommon_HW, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ConversionLayerTestCommon_HW, NPU5020_HW) {
    setDefaultHardwareMode();
    setBatchCompilerMode("unroll");
    run(Platform::NPU5020);
}

TEST_P(ShaveCodeGenConversionLayerTest, NPU5010) {
    setReferenceSoftwareMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(ConversionLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConversionLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ConversionLayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

class ConversionLayerTest_HostCompile : public ConversionLayerTestCommon {};

TEST_P(ConversionLayerTest_HostCompile, NPU4000_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ConversionLayerTest_HostCompile, NPU5010_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

class ConversionLayerTest_HostCompile_Interpreter : public ConversionLayerTestCommon {
    void configure_model() override {
        // E#220008: failed to legalize operation 'arith.divui' that was explicitly marked illegal
        VpuOv2LayerTest::configuration[ov::intel_npu::compilation_mode_params.name()] = "auto-unrolling-mode=disabled";
    }
};

TEST_P(ConversionLayerTest_HostCompile_Interpreter, NPU4000_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ConversionLayerTest_HostCompile_Interpreter, NPU5010_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

class ConversionLayerTest_SCFTiling : public ConversionLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "scf-tiling=true";
        // E-190336 for MC support
        configuration["NPU_TILES"] = "1";
    }
};

TEST_P(ConversionLayerTest_SCFTiling, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConversionLayerTest_SCFTiling, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ConversionTypes> conversionOpTypes = {
        ConversionTypes::CONVERT,
        ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<ov::Shape>> inShape = {{{1, 2, 3, 5}}};

const std::vector<std::vector<ov::Shape>> inShapeFP16To4 = {{{1, 2, 2, 16}}, {{1, 2, 3, 5}}};

const std::vector<std::vector<ov::Shape>> inShape1 = {{{1, 2, 3, 4}}};

const std::vector<std::vector<ov::Shape>> inShapeTiling = {{{2000, 2000}}};

const std::vector<std::vector<ov::Shape>> inShapeOdd = {{{1, 1, 1, 111}}};

const std::vector<ov::element::Type> netPrecisions = {f32, f16, u8, i8, i32};

const auto genParams = [](std::vector<ov::element::Type> srcPrec, std::vector<ov::element::Type> dstPrec, auto inShapes,
                          bool includeConvertLike = true) {
    std::vector<ConversionTypes> cvtOpTypes = {ConversionTypes::CONVERT};
    if (includeConvertLike) {
        // For some types, 'ConvertLike' triggers:
        // "Exception from src/plugins/template/backend/ops/convert.cpp:119:
        //  Unhandled data type ... in evaluate_node()"
        cvtOpTypes.push_back(ConversionTypes::CONVERT_LIKE);
    }
    return ::testing::Combine(::testing::ValuesIn(cvtOpTypes),                                      // Conversion type
                              ::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),  // Input shapes
                              ::testing::ValuesIn(srcPrec),                                         // Input type
                              ::testing::ValuesIn(dstPrec),                                         // Output type
                              ::testing::Values(test_utils::TARGET_DEVICE));
};

const auto configParams = genParams(netPrecisions, netPrecisions, inShape);
const auto configParamsBF16ToF16 = genParams({bf16}, {f16}, inShape);
const auto configParamsI32ToU64 = genParams({i32}, {u64}, inShape);
const auto configParamsF64ToI64 = genParams({f64}, {i64}, inShape);
const auto configParamsI64ToF64 = genParams({i64}, {f64}, inShape);
const auto configParamsU4Tiling = genParams({u4}, {f16, u8, i8}, inShapeTiling);
const auto configParamsU2Tiling = genParams({u2}, {f16}, inShapeTiling, false);
const auto configParamsI4Tiling = genParams({i4}, {i8, f16}, inShapeTiling);
const auto configParamsU4OddShape = genParams({u4}, {f16, u8, i8}, inShapeOdd);
const auto configParamsNF4ToF16 = genParams({nf4}, {f16}, inShape1, false);
const auto configParamsU16ToF16 = genParams({u16}, {f16}, inShapeOdd);
const auto configParamsF16ToU16 = genParams({f16}, {u16}, inShapeOdd);
const auto configParamsF8ToF16 = genParams({f8e4m3, f8e5m2}, {f16}, inShapeOdd, false);
const auto configParamsF16ToF8 = genParams({f16}, {f8e4m3, f8e5m2}, inShapeOdd, false);
const auto configParamsF32ToF16Tiling = genParams({f32}, {f16}, inShapeTiling);
const auto configParamsF16ToU4 = genParams({f16}, {u4}, inShapeFP16To4, false);
const auto configParamsF16ToI4 = genParams({f16}, {i4}, inShapeFP16To4, false);

const std::vector<std::vector<ov::test::InputShape>> dynamicShapes = {
        {generateTestShape(std::vector<BoundedDim>{1, 16, 2048_Dyn, 2048}, hostCompileSmallShapesLimitationCallback)},
        {generateTestShape(std::vector<BoundedDim>{1, 16, 2048_Dyn, 2048_Dyn},
                           hostCompileSmallShapesLimitationCallback)},
        {generateTestShape(std::vector<BoundedDim>{1, 3, 2048_Dyn, 2048}, hostCompileSmallShapesLimitationCallback)},
        {generateTestShape(std::vector<BoundedDim>{1, 3, 2048_Dyn, 2048_Dyn},
                           hostCompileSmallShapesLimitationCallback)},
        {generateTestShape(std::vector<BoundedDim>{1, 1, 2048_Dyn, 2048}, hostCompileSmallShapesLimitationCallback)},
        {generateTestShape(std::vector<BoundedDim>{1, 1, 2048_Dyn, 2048_Dyn},
                           hostCompileSmallShapesLimitationCallback)},
};
const auto configParamsF32ToF16 =
        ::testing::Combine(::testing::ValuesIn({ConversionTypes::CONVERT}),  // Conversion type
                           ::testing::ValuesIn(dynamicShapes),               // Input shapes
                           ::testing::ValuesIn({f32}),                       // Input type
                           ::testing::ValuesIn({f16}),                       // Output type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsI32ToU64Dyn = ::testing::Combine(
        ::testing::ValuesIn({ConversionTypes::CONVERT}),  // Conversion type
        ::testing::ValuesIn(
                std::vector<std::vector<ov::test::InputShape>>{{generateTestShape(2048_Dyn)}}),  // Input shapes
        ::testing::ValuesIn({i32}),                                                              // Input type
        ::testing::ValuesIn({u64}),                                                              // Output type
        ::testing::Values(test_utils::TARGET_DEVICE));

// ------ HW ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion, ConversionLayerTestCommon_HW, configParams,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_i4_Conversion, ConversionLayerTestCommon_HW, configParamsI4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_u4_Conversion, ConversionLayerTestCommon_HW, configParamsU4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_u2_Conversion, ConversionLayerTestCommon_HW, configParamsU2Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_bf16_Conversion, ConversionLayerTestCommon_HW, configParamsBF16ToF16,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_f64_i64_Conversion, ConversionLayerTestCommon_HW, configParamsF64ToI64,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_i64_f64_Conversion, ConversionLayerTestCommon_HW, configParamsI64ToF64,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_u16_f16_Conversion, ConversionLayerTestCommon_HW, configParamsU16ToF16,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_f16_u16_Conversion, ConversionLayerTestCommon_HW, configParamsF16ToU16,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_f8_f16_Conversion, ConversionLayerTestCommon_HW, configParamsF8ToF16,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_f16_f8_Conversion, ConversionLayerTestCommon_HW, configParamsF16ToF8,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_nf4_f16_Conversion, ConversionLayerTestCommon_HW, configParamsNF4ToF16,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_i32_u64_Conversion, ConversionLayerTestCommon_HW, configParamsI32ToU64,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_i32_u64_Dynamic_Conversion, ConversionLayerTestCommon_HW, configParamsI32ToU64Dyn,
                         ConversionLayerTest::getTestCaseName);

// ------ ShaveCodeGen ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ShaveCodeGen_Conversion, ShaveCodeGenConversionLayerTest, configParams,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_bf16_ShaveCodeGen_Conversion, ShaveCodeGenConversionLayerTest,
                         configParamsBF16ToF16, ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_f64_i64_ShaveCodeGen_Conversion, ShaveCodeGenConversionLayerTest, configParamsF64ToI64,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_i64_f64_ShaveCodeGen_Conversion, ShaveCodeGenConversionLayerTest, configParamsI64ToF64,
                         ConversionLayerTest::getTestCaseName);

// Tracking number [E#128077]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_u4_odd_Conversion, ConversionLayerTestCommon_HW,
                         configParamsU4OddShape, ConversionLayerTest::getTestCaseName);

// ------ SW ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_f16_u4_Conversion, ConversionLayerTestCommon, configParamsF16ToU4,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_f16_i4_Conversion, ConversionLayerTestCommon, configParamsF16ToI4,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion, ConversionLayerTestCommon, configParams,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_i4_Conversion, ConversionLayerTestCommon, configParamsI4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_u4_Conversion, ConversionLayerTestCommon, configParamsU4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_u4_odd_Conversion, ConversionLayerTestCommon, configParamsU4OddShape,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_bf16_Conversion, ConversionLayerTestCommon, configParamsBF16ToF16,
                         ConversionLayerTest::getTestCaseName);

// ------ HostCompile ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion_HostCompile, ConversionLayerTest_HostCompile, configParamsF32ToF16,
                         ConversionLayerTest::getTestCaseName);

// ------ HostCompile_Interpreter ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion_HostCompile_Interpreter,
                         ConversionLayerTest_HostCompile_Interpreter, configParamsF32ToF16,
                         ConversionLayerTest::getTestCaseName);

// ------ SCFTiling ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion_SCFTiling, ConversionLayerTest_SCFTiling,
                         configParamsF32ToF16Tiling, ConversionLayerTest::getTestCaseName);
}  // namespace
