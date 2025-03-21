//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/eltwise.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using ov::test::utils::EltwiseTypes;
using ov::test::utils::InputLayerType;
using ov::test::utils::OpType;

namespace ov {
namespace test {

class EltwiseLayerTestCommon : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

class EltwiseLayerTestF32Common : public EltwiseLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

class EltwiseEmptyShapeInputLayerTest : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};
class EltwiseIntegerLayerTest : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(EltwiseLayerTestCommon, NPU3720_SW) {
    abs_threshold = 0.6;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestCommon, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        // [Tracking number: E#82236]
        if (netPrecisions == ov::element::i32) {
            skip << "Type is not supported";
        }
    });

    abs_threshold = 0.6;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.6;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseLayerTestF32Common, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseLayerTestF32Common, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestF32Common, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        if (netPrecisions == ov::element::f32) {
            skip << "FP32 operations will be converted to IE.scaleshift in AdjustScaleShiftForDWConv in HW Mode. "
                    "IE.scaleshift is a NCE task which do not support FP32";
        }
    });

    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
TEST_P(EltwiseEmptyShapeInputLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseEmptyShapeInputLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
void setCommonSkipCompilationCallback(EltwiseIntegerLayerTest* test) {
    test->setSkipCompilationCallback([test](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(test->GetParam());
        const auto netPrecisions = std::get<4>(test->GetParam());

        // Define sets of unsupported types for specific precisions
        static const std::unordered_set<EltwiseTypes> unsupportedTypesForU16 = {
                EltwiseTypes::SUBTRACT, EltwiseTypes::FLOOR_MOD, EltwiseTypes::MULTIPLY, EltwiseTypes::DIVIDE,
                EltwiseTypes::POWER};
        static const std::unordered_set<EltwiseTypes> unsupportedTypesForU8 = {
                EltwiseTypes::MULTIPLY, EltwiseTypes::DIVIDE, EltwiseTypes::POWER};
        static const std::unordered_set<EltwiseTypes> unsupportedTypesForI16 = {EltwiseTypes::FLOOR_MOD};

        // Check if the current combination of precision and eltwiseType is unsupported
        bool isUnsupported = (netPrecisions == ov::element::u16 && unsupportedTypesForU16.count(eltwiseType)) ||
                             (netPrecisions == ov::element::u8 && unsupportedTypesForU8.count(eltwiseType)) ||
                             (netPrecisions == ov::element::i16 && unsupportedTypesForI16.count(eltwiseType));

        if (isUnsupported) {
            skip << eltwiseType << " SingleLayerTest is not enabled with precision: " << netPrecisions;
        }
    });
}

TEST_P(EltwiseIntegerLayerTest, NPU3720_SW) {
    abs_threshold = 0.6;
    setCommonSkipCompilationCallback(this);
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseIntegerLayerTest, NPU4000_SW) {
    abs_threshold = 0.6;
    setCommonSkipCompilationCallback(this);
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

namespace {

using namespace ov::test;

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f16,
        ov::element::i32,
};

std::vector<ov::test::ElementType> netPrecisionsF16 = {
        ov::element::f16,
};

std::vector<ov::test::ElementType> netPrecisionsF32 = {
        ov::element::f32,
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::VECTOR,
        ov::test::utils::OpType::SCALAR,
};

//
// Test supported Eltwise types + Tiling
//

std::set<EltwiseTypes> eltwiseTypes = {EltwiseTypes::ADD,       EltwiseTypes::MULTIPLY,     EltwiseTypes::SUBTRACT,
                                       EltwiseTypes::DIVIDE,    EltwiseTypes::SQUARED_DIFF, EltwiseTypes::POWER,
                                       EltwiseTypes::FLOOR_MOD, EltwiseTypes::MOD};

std::set<EltwiseTypes> eltwiseTypesF32 = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::POWER,
                                          EltwiseTypes::DIVIDE, EltwiseTypes::SUBTRACT};

std::vector<std::vector<ov::Shape>> bigShape = {{{1, 10, 256, 256}, {1, 10, 256, 256}}};

const auto typesParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypes, EltwiseLayerTestCommon, typesParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto typesParamsF32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
        ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes), ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisionsF32), ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypesF32, EltwiseLayerTestF32Common, typesParamsF32,
                         EltwiseLayerTestF32Common::getTestCaseName);

//
// Test Eltwise input broadcast
//

std::set<EltwiseTypes> broadcastTestEltwiseTypes = {EltwiseTypes::ADD};

std::vector<std::vector<ov::Shape>> broadcastTestInputShape = {{{1, 320, 128, 128}, {1, 320, 1, 1}}};

const auto broadcastTestParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(broadcastTestInputShape)),
                           ::testing::ValuesIn(broadcastTestEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisionsF16),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_InputBroadcastEltwise, EltwiseLayerTestCommon, broadcastTestParams,
                         EltwiseLayerTestCommon::getTestCaseName);

std::set<EltwiseTypes> scalarInput2broadcastTestEltwiseTypes = {EltwiseTypes::DIVIDE};
std::vector<std::vector<ov::Shape>> scalarInput2broadcastTestInputShape = {
        {{1, 1, 1, 512}, {1, 1, 1, 1}},
        {{1, 1, 512, 1}, {1, 1, 1, 1}},
        {{1, 512, 1, 1}, {1, 1, 1, 1}},
};
const auto scalarInput2BroadcastTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(scalarInput2broadcastTestInputShape)),
        ::testing::ValuesIn(scalarInput2broadcastTestEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisionsF16), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined), ::testing::Values(ov::test::utils::DEVICE_NPU),
        ::testing::Values(ov::test::Config{}));
INSTANTIATE_TEST_SUITE_P(precommit_scalarInput2BroadcastEltwise, EltwiseLayerTestCommon,
                         scalarInput2BroadcastTestParams, EltwiseLayerTestCommon::getTestCaseName);

//
// Test Eltwise batch input
//

std::set<EltwiseTypes> batchInputTestEltwiseTypes = {EltwiseTypes::ADD};

std::vector<std::vector<ov::Shape>> batchInputTestInputShape = {{{361, 4, 48, 48}, {361, 4, 48, 48}}};

const auto batchInputTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(batchInputTestInputShape)),
        ::testing::ValuesIn(batchInputTestEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisionsF16), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined), ::testing::Values(ov::test::utils::DEVICE_NPU),
        ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_BatchInputEltwise, EltwiseLayerTestCommon, batchInputTestParams,
                         EltwiseLayerTestCommon::getTestCaseName);

//
// Scalar mode
//

std::vector<std::vector<ov::Shape>> inShapesScalar = {
        {{10}},              // 1D
        {{1, 9}},            // NC
        {{1, 128, 32}},      // CHW
        {{1, 3, 224, 224}},  // NCHW
};

const auto scalarParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesND, EltwiseLayerTestCommon, scalarParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto scalarParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesNDF32, EltwiseLayerTestF32Common, scalarParamsF32,
                         EltwiseLayerTestF32Common::getTestCaseName);

//
// Vector mode
//

std::vector<std::vector<ov::Shape>> inShapesVector = {
        {{24}, {24}},                          // 1D
        {{1, 9}, {1, 1}},                      // NC + scalar
        {{1, 128, 32}, {1, 128, 32}},          // CHW, eltwise
        {{1, 128, 32}, {1, 128, 1}},           // CHW, input1 != input2, broadcast over W
        {{1, 128, 32}, {1, 1, 32}},            // CHW, input1 != input2, broadcast over H
        {{1, 128, 32}, {1, 1, 1}},             // CHW + scalar
        {{1, 3, 224, 224}, {1, 3, 224, 224}},  // NCHW, eltwise
        {{1, 3, 224, 224}, {1, 1, 1, 1}},      // NCHW + scalar
        {{1, 3, 224, 224}, {1, 3, 1, 1}},      // NCHW, broadcast over HW
        {{2, 3, 224, 224}, {1, 1, 1, 224}},    // NCHW, N != 1, broadcast over NCH
};

const auto vectorParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesND, EltwiseLayerTestCommon, vectorParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto vectorParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesNDF32, EltwiseLayerTestF32Common, vectorParamsF32,
                         EltwiseLayerTestF32Common::getTestCaseName);

//
//  This case to test the support for empty shape input for Add and Multiply ops
//
std::set<EltwiseTypes> eltwise0DInputOps = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY};

std::vector<std::vector<ov::Shape>> eltwise0DInputShape = {
        {{}},  // 0D
};

const auto vectorParamsEmptyShapeInput =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(eltwise0DInputShape)),
                           ::testing::ValuesIn(eltwise0DInputOps), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_0DInputTest, EltwiseEmptyShapeInputLayerTest, vectorParamsEmptyShapeInput,
                         EltwiseEmptyShapeInputLayerTest::getTestCaseName);

//
// Bitwise
//

std::vector<std::vector<ov::Shape>> bitwiseInput = {{{1, 1, 256, 56}, {1, 1, 256, 56}},
                                                    {{1, 1, 256, 56}, {1, 1, 256, 1}}};

std::vector<ov::test::ElementType> bitwiseNetPrecisions = {ov::element::i32};

std::set<EltwiseTypes> bitwiseTypes = {EltwiseTypes::BITWISE_AND, EltwiseTypes::BITWISE_OR, EltwiseTypes::BITWISE_XOR};

const auto bitwiseParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwiseInput)),
                           ::testing::ValuesIn(bitwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(bitwiseNetPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_Bitwise, EltwiseLayerTestCommon, bitwiseParams,
                         EltwiseLayerTestCommon::getTestCaseName);

std::vector<std::vector<ov::Shape>> bitwiseNotInput = {{{1, 1, 256, 56}, {}}};

const auto bitwiseNotParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwiseNotInput)),
                           ::testing::Values(EltwiseTypes::BITWISE_NOT), ::testing::Values(InputLayerType::CONSTANT),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(bitwiseNetPrecisions),
                           ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_BitwiseNot, EltwiseLayerTestCommon, bitwiseNotParams,
                         EltwiseLayerTestCommon::getTestCaseName);

//
// Test Unsigned Integer data types
//

std::vector<std::vector<ov::Shape>> inShape = {{{1, 5, 16, 32}, {1, 5, 16, 32}}};

std::vector<ov::test::ElementType> netPrecisionsUnsigned = {ov::element::u8, ov::element::u16, ov::element::u32,
                                                            ov::element::u64};

std::set<EltwiseTypes> eltwiseTypesUnsigned = {EltwiseTypes::ADD,    EltwiseTypes::SUBTRACT, EltwiseTypes::MULTIPLY,
                                               EltwiseTypes::DIVIDE, EltwiseTypes::POWER,    EltwiseTypes::FLOOR_MOD,
                                               EltwiseTypes::MOD};

const auto typesParamsUnsigned = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
        ::testing::ValuesIn(eltwiseTypesUnsigned), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsUnsigned),
        ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_Unsigned, EltwiseIntegerLayerTest, typesParamsUnsigned,
                         EltwiseIntegerLayerTest::getTestCaseName);

//
// Test Integer data types
//

std::vector<ov::test::ElementType> netPrecisionsInteger = {ov::element::i8, ov::element::i16, ov::element::i32};

std::set<EltwiseTypes> eltwiseTypesInteger = {EltwiseTypes::FLOOR_MOD, EltwiseTypes::MOD};

const auto typesParamsInteger = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
        ::testing::ValuesIn(eltwiseTypesInteger), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsInteger),
        ::testing::Values(ov::element::undefined), ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_Signed, EltwiseIntegerLayerTest, typesParamsInteger,
                         EltwiseIntegerLayerTest::getTestCaseName);

}  // namespace
