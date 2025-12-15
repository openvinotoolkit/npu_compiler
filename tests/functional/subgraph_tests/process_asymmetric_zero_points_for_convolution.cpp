//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

#include "vpux/utils/core/error.hpp"
namespace ov::test::subgraph {

class Conv2dInMixedModeIncorrectZeroPoint : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        VPUX_THROW_UNLESS(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = 1 + i % 7;
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void SetUp() override {
        constexpr size_t filterIn = 4;
        const ov::Shape inputShape = {1, filterIn, 4, 4};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        constexpr size_t filterOut = 4;
        constexpr size_t kernelHeight = 3;
        constexpr size_t kernelWidth = 3;
        const ov::Shape weightShape = ov::Shape{filterOut, filterIn, kernelHeight, kernelWidth};
        std::vector<uint8_t> weightsData(filterIn * filterOut * kernelWidth * kernelHeight);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = 129 + i % 3;
        }
        weightsData.at(0) = 0;
        weightsData.at(1) = 255;
        const auto constLayer_node = ov::opset1::Constant::create(ov::element::u8, weightShape, weightsData);
        constexpr float zp = 120.f;

        auto convert_node = std::make_shared<ov::opset1::Convert>(constLayer_node->output(0), ov::element::f32);

        const auto zero_points = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{zp});
        const auto shift_node = std::make_shared<ov::opset1::Subtract>(convert_node->output(0), zero_points->output(0));

        const auto scales = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto scale_node = std::make_shared<ov::opset1::Multiply>(shift_node->output(0), scales->output(0));

        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                params.at(0), scale_node->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv2d_node)};

        function = std::make_shared<ov::Model>(results, params, "Conv2dInMixedModeIncorrectZeroPoint");
        rel_threshold = 0.5f;
    }
};

//
// Platform test definition
//

TEST_F(Conv2dInMixedModeIncorrectZeroPoint, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-convolution-mixed-precision-decomposition=true";
    run(Platform::NPU3720);
}

TEST_F(Conv2dInMixedModeIncorrectZeroPoint, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-convolution-mixed-precision-decomposition=true";
    run(Platform::NPU4000);
}
TEST_F(Conv2dInMixedModeIncorrectZeroPoint, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-convolution-mixed-precision-decomposition=true";
    run(Platform::NPU5010);
}
}  // namespace ov::test::subgraph
