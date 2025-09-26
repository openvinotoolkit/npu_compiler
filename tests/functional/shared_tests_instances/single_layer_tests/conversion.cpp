// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/conversion.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

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
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ConversionLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
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

const std::vector<std::vector<ov::Shape>> inShapeTiling = {{{2000, 2000}}};

const std::vector<std::vector<ov::Shape>> inShapeOdd = {{{1, 1, 1, 111}}};

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16, ov::element::u8,
                                                      ov::element::i8, ov::element::i32};

const auto configParamsBF16ToF16 =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                              // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),  // Input shapes
                           ::testing::Values(ov::element::bf16),                                // Input type
                           ::testing::Values(ov::element::f16),                                 // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));
const auto configParams =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                              // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),  // Input shapes
                           ::testing::ValuesIn(netPrecisions),                                  // Input type
                           ::testing::ValuesIn(netPrecisions),                                  // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsF64ToI64 =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                              // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),  // Input shapes
                           ::testing::Values(ov::element::f64),                                 // Input type
                           ::testing::Values(ov::element::i64),                                 // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsI64ToF64 =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                              // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),  // Input shapes
                           ::testing::Values(ov::element::i64),                                 // Input type
                           ::testing::Values(ov::element::f64),                                 // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsU4Tiling =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                    // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeTiling)),  // Input shapes
                           ::testing::Values(ov::element::u4),                                        // Input type
                           ::testing::ValuesIn({ov::element::f16, ov::element::u8, ov::element::i8}),  // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));                              // Device name

const auto configParamsU2Tiling =
        ::testing::Combine(::testing::Values(ConversionTypes::CONVERT),                               // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeTiling)),  // Input shapes
                           ::testing::Values(ov::element::u2),                                        // Input type
                           ::testing::ValuesIn({ov::element::f16}),                                   // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));                             // Device name

const auto configParamsI4Tiling =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                    // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeTiling)),  // Input shapes
                           ::testing::Values(ov::element::i4),                                        // Input type
                           ::testing::ValuesIn({ov::element::i8, ov::element::f16}),                  // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsU4OddShape =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                 // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeOdd)),  // Input shapes
                           ::testing::Values(ov::element::u4),                                     // Input type
                           ::testing::ValuesIn({ov::element::f16, ov::element::u8, ov::element::i8}),  // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsU16ToF16 =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                 // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeOdd)),  // Input shapes
                           ::testing::Values(ov::element::u16),                                    // Input type
                           ::testing::Values(ov::element::f16),                                    // Convert type
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsF16ToU16 =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                 // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeOdd)),  // Input shapes
                           ::testing::Values(ov::element::f16),                                    // Input type
                           ::testing::Values(ov::element::u16),                                    // Convert type
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

}  // namespace
