// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/roll.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class RollLayerTestCommon : public RollLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(RollLayerTestCommon, NPU3720) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(RollLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(RollLayerTestCommon, NPU5010) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::RollLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypes = {
        ov::element::u8, ov::element::i32,
        ov::element::f16,  // CPU-plugin has parameter I16, but NPU does not support it. So value from
                           // CPU-plugin I16 is changed to FP16.
};

std::vector<std::vector<ov::Shape>> inputShapes = {
        {{16}},             // testCase1D
        {{17, 19}},         // testCase2DZeroShifts
        {{4, 3}},           // testCase2D
        {{2, 320, 320}},    // testCase3D
        {{3, 11, 6, 4}},    // testCaseNegativeUnorderedAxes4D
        {{2, 16, 32, 32}},  // testCaseRepeatingAxes5D
};

const std::vector<std::vector<int64_t>> shift = {
        {5}, {0, 0}, {1, 2, 1}, {160, 160}, {7, 3}, {16, 15, 10, 2, 1, 7, 2, 8, 1, 1}, {300, 250}};

const std::vector<std::vector<int64_t>> axes = {
        {0}, {0, 1}, {0, 1, 0}, {1, 2}, {-3, -2}, {-1, -2, -3, 1, 0, 3, 3, 2, -2, -3}, {0, 1}};

const auto testRollParams0 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[0])),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(shift[0]), ::testing::Values(axes[0]),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto testRollParams1 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[1])),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(shift[1]), ::testing::Values(axes[1]),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto testRollParams2 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[2])),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(shift[2]), ::testing::Values(axes[2]),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto testRollParams3 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[3])),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(shift[3]), ::testing::Values(axes[3]),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto testRollParams4 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[4])),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(shift[4]), ::testing::Values(axes[4]),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto testRollParams5 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[5])),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(shift[5]), ::testing::Values(axes[5]),
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Roll_Test_Check0, RollLayerTestCommon, testRollParams0,
                         RollLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Roll_Test_Check1, RollLayerTestCommon, testRollParams1,
                         RollLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Roll_Test_Check2, RollLayerTestCommon, testRollParams2,
                         RollLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Roll_Test_Check3, RollLayerTestCommon, testRollParams3,
                         RollLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Roll_Test_Check4, RollLayerTestCommon, testRollParams4,
                         RollLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Roll_Test_Check5, RollLayerTestCommon, testRollParams5,
                         RollLayerTestCommon::getTestCaseName);
}  // namespace
