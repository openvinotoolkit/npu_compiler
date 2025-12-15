//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/strided_slice.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class StridedSliceLayerTestCommon : public StridedSliceLayerTest, virtual public VpuOv2LayerTest {};

class StridedSliceNCELayerTest : public StridedSliceLayerTestCommon {};
class StridedSliceTilingLayerTest : public StridedSliceLayerTestCommon {};
class StridedSliceSubByteCopyLayerTest : public StridedSliceLayerTestCommon {};

TEST_P(StridedSliceLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(StridedSliceNCELayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(StridedSliceTilingLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(StridedSliceSubByteCopyLayerTest, NPU4000_HW) {
    // Tracking number [C#163362]
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "OpenVINO error after inference, 'Tensor data with element type u4, is not representable as pointer "
                "to i8'";
    });
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(StridedSliceLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(StridedSliceLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::StridedSliceLayerTestCommon;
using ov::test::StridedSliceNCELayerTest;
using ov::test::StridedSliceSubByteCopyLayerTest;
using ov::test::StridedSliceTilingLayerTest;

namespace {

std::vector<ov::test::StridedSliceSpecificParams> precommit_tests = {
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{32, 20}})),
         {2, 10},
         {32, 20},
         {1, 2},
         {0, 1},
         {1, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{32, 32, 32}})),
         {0, 0},
         {0, 0},
         {1, 2},
         {1, 1},
         {1, 1},
         {0, 0},
         {0, 1},
         {0, 0}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{2, 5, 32, 32}})),
         {0, 0, 0, 20},
         {1, 2, 30, 30},
         {1, 1, 2, 1},
         {0, 0, 0, 1},
         {0, 1, 0, 1},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{5, 16, 16, 16}})),
         {1, 4, 5, 10},
         {0, 0, 0, 0},
         {2, 7, 5, 3},
         {0, 0, 0, 0},
         {1, 1, 1, 1},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 48, 32, 32}})),
         {0, 16, 0, 20},
         {1, 32, 32, 30},
         {1, 1, 1, 2},
         {1, 0, 1, 0},
         {1, 0, 1, 0},
         {},
         {},
         {}},
};

std::vector<ov::test::StridedSliceSpecificParams> nce_tests = {
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{5, 16, 16, 16}})),
         {1, 4, 5, 10},
         {0, 0, 0, 0},
         {2, 7, 5, 3},
         {0, 0, 0, 0},
         {1, 1, 1, 1},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 48, 32, 32}})),
         {0, 16, 0, 20},
         {1, 32, 32, 30},
         {1, 1, 1, 2},
         {1, 0, 1, 0},
         {1, 0, 1, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 416, 416}})),
         {0, 0, 0, 0},
         {1, 3, 416, 416},
         {1, 1, 2, 2},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 416, 416}})),
         {0, 0, 1, 0},
         {1, 3, 416, 416},
         {1, 1, 2, 2},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 416, 416}})),
         {0, 0, 0, 1},
         {1, 3, 416, 416},
         {1, 1, 2, 2},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 416, 416}})),
         {0, 0, 1, 1},
         {1, 3, 416, 416},
         {1, 1, 2, 2},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {},
         {},
         {}},
};

std::vector<ov::test::StridedSliceSpecificParams> tests_5d = {
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 5, 20, 32, 32}})),
         {0, 0, 0, 0, 0},
         {1, 5, 20, 32, 32},
         {1, 1, 1, 1, 2},
         {0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 1, 1, 51, 1}})),
         {0, 0, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {1, 1, 1, 2, 1},
         {1, 1, 0, 1, 1},
         {1, 1, 0, 1, 1},
         {0, 0, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {0, 0, 0, 0, 0}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 5, 20, 32, 32}})),
         {0, 0, 0, 0, 0},
         {1, 5, 20, 32, 32},
         {1, 2, 2, 2, 2},
         {0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 8, 8, 8, 8}})),
         {0, 0, 0, 0, 0},
         {1, 8, 8, 8, 8},
         {1, 2, 2, 2, 2},
         {0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 8, 8, 8, 8}})),
         {0, 2, 3, 1, 1},
         {1, 8, 8, 8, 8},
         {1, 2, 2, 2, 2},
         {0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0},
         {},
         {},
         {}},
};

std::vector<ov::test::StridedSliceSpecificParams> tiling_tests = {
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 8, 80, 1280}})),
         {0, 0, 0, 0},
         {0, 0, 2147483647, 0},  // The 2147483647 value from ends is supported beacause of ResolveStridedSlice pass.
         {1, 1, 4, 1},
         {1, 1, 0, 1},
         {1, 1, 0, 1},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 640, 640}})),
         {0, 0, 0, 0},
         {0, 0, 2147483647, 0},  // The 2147483647 value from ends is supported beacause of ResolveStridedSlice pass.
         {1, 1, 2, 1},
         {1, 1, 1, 1},
         {1, 1, 0, 1},
         {},
         {},
         {}},
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 640, 640}})),
         {0, 0, 0, 0},
         {0, 0, 0, 2147483647},  // The 2147483647 value from ends is supported beacause of ResolveStridedSlice pass.
         {1, 1, 1, 2},
         {1, 1, 1, 1},
         {1, 1, 1, 0},
         {},
         {},
         {}},
};

std::vector<ov::test::StridedSliceSpecificParams> testSubByteTypeCase = {
        {ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{{1, 2, 3, 12}}})),
         {0, 0, 0, 0},
         {1, 2, 3, 12},
         {1, 1, 1, 3},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {},
         {},
         {}},
};

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_StridedSlice, StridedSliceLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(precommit_tests), ::testing::ValuesIn(modelTypes),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         StridedSliceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StridedSlice_5D, StridedSliceLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(tests_5d), ::testing::ValuesIn(modelTypes),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         StridedSliceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_StridedSlice, StridedSliceNCELayerTest,
                         ::testing::Combine(::testing::ValuesIn(nce_tests), ::testing::ValuesIn(modelTypes),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         StridedSliceNCELayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_StridedSlice, StridedSliceTilingLayerTest,
                         ::testing::Combine(::testing::ValuesIn(tiling_tests), ::testing::ValuesIn(modelTypes),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         StridedSliceTilingLayerTest::getTestCaseName);

// Sub-byte case with non-byte aligned strides and testing builtin SW.Kernel Copy
// Tracking number [E#160558] - Direct access to Network I/O params
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_StridedSlice_subByteTypeCase, StridedSliceSubByteCopyLayerTest,
                         ::testing::Combine(::testing::ValuesIn(testSubByteTypeCase),
                                            ::testing::ValuesIn({ov::element::u4, ov::element::i4}),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         StridedSliceSubByteCopyLayerTest::getTestCaseName);

}  // namespace
