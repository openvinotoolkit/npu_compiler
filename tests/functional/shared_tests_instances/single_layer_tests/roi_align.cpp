//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/roi_align.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ROIAlignV9LayerTestCommon : public ROIAlignV9LayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

TEST_P(ROIAlignV9LayerTestCommon, NPU3720_SW) {
    abs_threshold = 0.012;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ROIAlignV9LayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.012;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> modelTypes = {
        ov::element::f16,
};

const std::vector<std::string> poolingMode = {"avg", "max"};

std::vector<std::vector<ov::Shape>> inputShape = {
        {{2, 22, 20, 20}}, {{2, 18, 20, 20}}, {{2, 4, 20, 20}}, {{2, 4, 20, 40}}};

const std::vector<ov::Shape> coordsShape = {{2, 4}};

const std::vector<int> pooledH = {8};

const std::vector<int> pooledW = {8};

const std::vector<float> spatialScale = {0.03125f, 1.f};

const std::vector<int> poolingRatio = {2};

const std::vector<std::string> alignedMode = {"asymmetric", "half_pixel", "half_pixel_for_nn"};

const auto testROIAlignV9Params = testing::Combine(
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)), testing::ValuesIn(coordsShape),
        testing::ValuesIn(pooledH), testing::ValuesIn(pooledW), testing::ValuesIn(spatialScale),
        testing::ValuesIn(poolingRatio), testing::ValuesIn(poolingMode), testing::ValuesIn(alignedMode),
        testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(precommit_ROIAlign, ROIAlignV9LayerTestCommon, testROIAlignV9Params,
                         ROIAlignV9LayerTestCommon::getTestCaseName);

}  // namespace
