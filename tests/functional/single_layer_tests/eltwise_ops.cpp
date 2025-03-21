//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov;
using namespace element;

namespace ov::test {

template <typename EltwiseOpT>
class Eltwise2InputLayerTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == 1, "Expected 1 inputShapes");
        OPENVINO_ASSERT(funcInputs.size() == 2, "Expected 2 inputs");
        const auto& inputStaticShape = inputShapes[0];
        auto inputTensor1 = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputTensor2 = ov::Tensor{ov::element::f16, inputStaticShape};
        using inputValueType = ov::element_type_traits<ov::element::f16>::value_type;
        inputValueType* inputData1 = inputTensor1.data<inputValueType>();
        inputValueType* inputData2 = inputTensor2.data<inputValueType>();
        const auto totalSize = ov::shape_size(inputStaticShape);

        std::iota(inputData1, inputData1 + totalSize, 0);
        std::iota(inputData2, inputData2 + totalSize, 1);

        inputs = {{funcInputs[0].get_node_shared_ptr(), inputTensor1},
                  {funcInputs[1].get_node_shared_ptr(), inputTensor2}};
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
        const std::vector<ov::Shape> inferenceShapes = {ov::Shape{2, 3}, ov::Shape{2, 3}};
        const ov::test::InputShape dataShape = {ov::Shape{2, 3}, inferenceShapes};
        init_input_shapes({dataShape});

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3});
        auto mul = std::make_shared<EltwiseOpT>(param1, param2);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(mul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                               "EltwiseMultiply");
    }
};

typedef Eltwise2InputLayerTest<ov::op::v1::Multiply> EltwiseMultiplyLayerTest;

TEST_F(EltwiseMultiplyLayerTest, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(EltwiseMultiplyLayerTest, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
typedef Eltwise2InputLayerTest<ov::op::v1::Add> EltwiseAddLayerTest;

TEST_F(EltwiseAddLayerTest, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(EltwiseAddLayerTest, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test
