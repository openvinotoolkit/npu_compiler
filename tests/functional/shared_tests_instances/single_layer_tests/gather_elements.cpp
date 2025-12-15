// Copyright (C) 2021-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/gather_elements.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/op/gather_elements.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// [E#177479] GatherElements operator is limited to positive indices as it does not support negative indices at this
// time
class GatherElementsLayerTestCommon : public GatherElementsLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        ov::Shape indicesShape;
        std::vector<InputShape> shapes;
        ov::element::Type modelType, indicesType;
        int axis;
        std::tie(shapes, indicesShape, axis, modelType, indicesType, targetDevice) = this->GetParam();
        init_input_shapes(shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front());

        auto axisDim = targetStaticShapes[0][0][axis < 0 ? axis + targetStaticShapes[0][0].size() : axis];
        ov::test::utils::InputGenerateData inData;

        inData.start_from = 0;
        inData.range = axisDim;

        auto indicesNodeTensor = ov::test::utils::create_and_fill_tensor(indicesType, indicesShape, inData);
        auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesNodeTensor);

        auto gatherEl = std::make_shared<ov::op::v6::GatherElements>(param, indicesNode, axis);
        gatherEl->set_friendly_name("GatherElements");

        auto result = std::make_shared<ov::op::v0::Result>(gatherEl);

        function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gatherEl");
    }
};

TEST_P(GatherElementsLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherElementsLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(GatherElementsLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> dPrecisions = {ov::element::f32};

const std::vector<ov::element::Type> iPrecisions = {ov::element::i32};

const std::vector<int> axes_set1 = {-1, 0, 1};
const std::vector<int> axes_set2 = {-2, 1};
const std::vector<int> axes_set3 = {0};

const std::vector<std::vector<ov::Shape>> iShapes = {{{2, 2}}, {{5, 7, 9, 1}}, {{2, 2, 1}}};

const auto GatherElements_PRECOMMIT_set1 = ::testing::Combine(
        testing::ValuesIn({static_shapes_to_test_representation(iShapes[0])}), testing::Values(ov::Shape{2, 2}),
        testing::ValuesIn(axes_set1), testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
        testing::Values(test_utils::TARGET_DEVICE));

const auto GatherElements_PRECOMMIT_set2 = ::testing::Combine(
        testing::ValuesIn({static_shapes_to_test_representation(iShapes[1])}), testing::Values(ov::Shape{5, 7, 9, 1}),
        testing::ValuesIn(axes_set2), testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
        testing::Values(test_utils::TARGET_DEVICE));

const auto GatherElements_PRECOMMIT_set3 = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation(iShapes[2])}), ::testing::Values(ov::Shape{4, 2, 1}),
        ::testing::ValuesIn(axes_set3), ::testing::ValuesIn(dPrecisions), ::testing::ValuesIn(iPrecisions),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set1, GatherElementsLayerTestCommon,
                         GatherElements_PRECOMMIT_set1, GatherElementsLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set2, GatherElementsLayerTestCommon,
                         GatherElements_PRECOMMIT_set2, GatherElementsLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set3, GatherElementsLayerTestCommon,
                         GatherElements_PRECOMMIT_set3, GatherElementsLayerTestCommon::getTestCaseName);

}  // namespace
