//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/fake_quantize.hpp"

namespace ov::test::subgraph {

class FqWithTwoAxesLayerTestCommon : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        using inputValueType = ov::element_type_traits<ov::element::f32>::value_type;
        inputValueType* inputData = inputTensor.data<inputValueType>();
        const auto totalSize = ov::shape_size(inputStaticShape);
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = (i % 255) - 127;
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        const ov::Shape inputShape = {4, 8, 16};
        const std::vector<ov::Shape> inferenceShapes = {inputShape};
        const ov::test::InputShape dataShape = {inputShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto fqInLow = ov::opset1::Constant::create(ov::element::f32, {1, 1, 1}, std::vector<float>{-128.f});
        const auto fqInHigh = ov::opset1::Constant::create(ov::element::f32, {1, 1, 1}, std::vector<float>{127.f});

        const ov::Shape fqShape = {inputShape.at(0), 1, inputShape.at(2)};
        using fqValueType = ov::element_type_traits<ov::element::f32>::value_type;
        const auto fqTotalSize = ov::shape_size(fqShape);
        std::vector<fqValueType> fqOutLowData(fqTotalSize, 0.f);
        for (size_t idx = 0; idx < fqOutLowData.size(); idx++) {
            fqOutLowData.at(idx) = -32.0 * (1 + idx % 3);
        }
        const auto fqOutLow = ov::opset1::Constant::create(ov::element::f32, fqShape, fqOutLowData);
        std::vector<fqValueType> fqOutHighData(fqTotalSize, 0.f);
        for (size_t idx = 0; idx < fqOutHighData.size(); idx++) {
            fqOutHighData.at(idx) = -fqOutLowData.at(idx) - 1;
        }
        const auto fqOutHigh = ov::opset1::Constant::create(ov::element::f32, fqShape, fqOutHighData);
        const size_t fqLevels = 256;
        const auto fq = std::make_shared<ov::opset1::FakeQuantize>(
                param->output(0), fqInLow->output(0), fqInHigh->output(0), fqOutLow->output(0), fqOutHigh->output(0),
                fqLevels, ov::op::AutoBroadcastType::NUMPY);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(fq->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "FQWithAxis");
    }
};

//
// Platform test definition
//

TEST_F(FqWithTwoAxesLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(FqWithTwoAxesLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_F(FqWithTwoAxesLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace ov::test::subgraph
