//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reorg_yolo.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/reorg_yolo.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ReorgYoloLayerTestCommon : public ReorgYoloLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<size_t> inputShape;
        ov::element::Type modelType;
        size_t stride;
        std::tie(inputShape, stride, modelType, std::ignore) = this->GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front());
        auto reorgYolo = std::make_shared<ov::op::v0::ReorgYolo>(param, stride);
        VpuOv2LayerTest::function =
                std::make_shared<ov::Model>(reorgYolo->outputs(), ov::ParameterVector{param}, "ReorgYolo");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(ReorgYoloLayerTestCommon, NPU3720_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReorgYoloLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
TEST_P(ReorgYoloLayerTestCommon, NPU5010_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<size_t>> inputShapesA = {
        std::vector<size_t>{1, 64, 26, 26},  // openvino eg
        std::vector<size_t>{1, 4, 4, 4},    std::vector<size_t>{1, 8, 4, 4},    std::vector<size_t>{2, 8, 4, 4},
        std::vector<size_t>{1, 62, 14, 14}, std::vector<size_t>{1, 62, 34, 24}, std::vector<size_t>{1, 24, 34, 62},
        std::vector<size_t>{1, 26, 64, 26},
};

const std::vector<size_t> stridesA = {2};

const std::vector<std::vector<size_t>> inputShapesB = {
        std::vector<size_t>{1, 9, 3, 3},
};

const std::vector<size_t> stridesB = {3};

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const auto paramsA = testing::Combine(testing::ValuesIn(inputShapesA), testing::ValuesIn(stridesA),
                                      testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsB = testing::Combine(testing::ValuesIn(inputShapesB), testing::ValuesIn(stridesB),
                                      testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ReorgYolo_a, ReorgYoloLayerTestCommon, paramsA,
                         ReorgYoloLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReorgYolo_b, ReorgYoloLayerTestCommon, paramsB,
                         ReorgYoloLayerTestCommon::getTestCaseName);

}  // namespace
