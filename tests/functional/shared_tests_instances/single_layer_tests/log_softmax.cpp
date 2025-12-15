// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/log_softmax.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LogSoftmaxLayerTestCommon : public LogSoftmaxLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(LogSoftmaxLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogSoftmaxLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(LogSoftmaxLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelType = {ov::element::f16};

std::vector<int64_t> axis2D = {0, 1};

std::vector<std::vector<ov::Shape>> inShapes2D = {{{12, 5}},    {{1, 40}},   {{1, 66}},   {{1, 72}},   {{5, 120}},
                                                  {{5, 59}},    {{64, 29}},  {{1, 2312}}, {{1, 4192}}, {{1, 4335}},
                                                  {{10, 6495}}, {{1200, 5}}, {{2708, 7}}};

std::vector<std::vector<ov::Shape>> inShapes3D = {{{5, 30, 1}}};

std::vector<std::vector<ov::Shape>> inShapes4D = {
        {{1, 10, 7, 4}},
        {{1, 2, 204, 62}},
        {{3, 20, 1, 15}},
        {{1, 48, 160, 80}},
};

std::vector<int64_t> axis3D = {0, 1, 2};
std::vector<int64_t> axis4D = {0, 1, 2, 3};

const auto params2D = testing::Combine(testing::ValuesIn(modelType),
                                       testing::ValuesIn(static_shapes_to_test_representation(inShapes2D)),
                                       testing::ValuesIn(axis2D), testing::Values(test_utils::TARGET_DEVICE));

const auto params3D = testing::Combine(testing::ValuesIn(modelType),
                                       testing::ValuesIn(static_shapes_to_test_representation(inShapes3D)),
                                       testing::ValuesIn(axis3D), testing::Values(test_utils::TARGET_DEVICE));

const auto params4D = testing::Combine(testing::ValuesIn(modelType),
                                       testing::ValuesIn(static_shapes_to_test_representation(inShapes4D)),
                                       testing::ValuesIn(axis4D), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsTiling = testing::Combine(testing::ValuesIn(modelType),
                                           testing::ValuesIn({static_shapes_to_test_representation({inShapes4D[3]})}),
                                           testing::Values(1), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, LogSoftmaxLayerTestCommon, params2D,
                         LogSoftmaxLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_3D, LogSoftmaxLayerTestCommon, params3D,
                         LogSoftmaxLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_4D, LogSoftmaxLayerTestCommon, params4D,
                         LogSoftmaxLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_tiling, LogSoftmaxLayerTestCommon, paramsTiling,
                         LogSoftmaxLayerTestCommon::getTestCaseName);

}  // namespace
