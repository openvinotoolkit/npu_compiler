//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/cum_sum.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class CumSumLayerTestCommon : public CumSumLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(CumSumLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CumSumLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(CumSumLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<std::vector<ov::Shape>> shapes = {{{5, 14, 5, 7}},
                                                    // Values from real neural networks
                                                    {{1, 1}},
                                                    {{1, 1024}},
                                                    {{1, 128}},
                                                    {{1, 25, 36}},
                                                    {{1, 384}},
                                                    {{1, 5}},
                                                    {{1, 9}},
                                                    {{8, 128}},
                                                    {{8, 384}}};

const std::vector<ov::element::Type> inputPrecision = {ov::element::f16, ov::element::f32};
const std::vector<ov::element::Type> inputLowPrecision = {ov::element::f16, ov::element::i32};

const std::vector<int64_t> axes = {0, 1};
const std::vector<int64_t> negativeAxes = {-2, -1};

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse = {true, false};

const auto testCaseAxis_0 =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}),  // Input shapes
                         testing::Values(inputPrecision[0]),                                      // Model type
                         testing::Values(axes[0]),                                                // Axis
                         testing::Values(exclusive[0]),                                           // Exclusive
                         testing::Values(reverse[1]),                                             // Reverse
                         testing::Values(test_utils::TARGET_DEVICE));                             // Device name

const auto testCasesNegativeAxis = testing::Combine(
        testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}), testing::Values(inputPrecision[1]),
        testing::ValuesIn(negativeAxes), testing::Values(exclusive[1]), testing::Values(reverse[0]),
        testing::Values(test_utils::TARGET_DEVICE));

std::vector<std::vector<ov::Shape>> iShape(shapes.begin() + 1, shapes.end());
const auto testCasesRealNet =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShape)),
                         testing::Values(inputPrecision[0]), testing::Values(axes[1]), testing::Values(exclusive[1]),
                         testing::Values(reverse[1]), testing::Values(test_utils::TARGET_DEVICE));

const auto testCasePrecommit = testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[4]})}),
                                                testing::ValuesIn(inputLowPrecision), testing::ValuesIn(negativeAxes),
                                                testing::ValuesIn(exclusive), testing::ValuesIn(reverse),
                                                testing::Values(test_utils::TARGET_DEVICE));

std::vector<std::vector<ov::test::InputShape>> cumSumDynamicShapes = {
        {{{{1, 1, 1, ov::Dimension(1, 10)}, {{1, 1, 1, 10}, {1, 1, 1, 5}}}},
         {{{1, ov::Dimension(1, 32)}, {{1, 16}, {1, 32}}}},
         {{{ov::Dimension(1, 8), 128}, {{4, 128}, {8, 128}}}}}};

const auto cumSumParamsDynamic = ::testing::Combine(
        ::testing::ValuesIn(cumSumDynamicShapes), ::testing::Values(inputPrecision[0]), ::testing::Values(axes[1]),
        ::testing::Values(exclusive[1]), ::testing::Values(reverse[1]), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_CumSumDynamic, CumSumLayerTestCommon, cumSumParamsDynamic,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_axis_0, CumSumLayerTestCommon, testCaseAxis_0,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_negative_axis, CumSumLayerTestCommon, testCasesNegativeAxis,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_real_net, CumSumLayerTestCommon, testCasesRealNet,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_CumSum, CumSumLayerTestCommon, testCasePrecommit,
                         CumSumLayerTestCommon::getTestCaseName);

}  // namespace
