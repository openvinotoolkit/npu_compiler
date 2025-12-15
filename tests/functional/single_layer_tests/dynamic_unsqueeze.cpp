//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset13_decl.hpp>

#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::test {

PRETTY_PARAM(BoundedShape, ov::test::InputShape);
PRETTY_PARAM(Indices, std::vector<int>);
PRETTY_PARAM(InputType, ov::element::Type);

using UnsqueezeLayerTestParams = std::tuple<BoundedShape, Indices, InputType>;

class DynamicUnsqueezeLayerTest : public testing::WithParamInterface<UnsqueezeLayerTestParams>, public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 10;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& [inputShape, indices, inputType] = GetParam();

        const auto indicesValues = indices.value();
        const auto indicesShape = ov::Shape{indicesValues.size()};
        const auto indicesPartialShape = generateTestShape(indicesShape);

        init_input_shapes({inputShape.value(), indicesPartialShape});

        const auto dataParam = std::make_shared<ov::opset13::Parameter>(inputType.value(), inputDynamicShapes.front());
        const auto axesConstant =
                std::make_shared<ov::opset13::Constant>(ov::element::i64, indicesShape, indicesValues);
        const auto unsqueeze = std::make_shared<ov::opset13::Unsqueeze>(dataParam, axesConstant);

        dataParam->set_friendly_name("input");
        axesConstant->set_friendly_name("axes");

        function = std::make_shared<ov::Model>(unsqueeze->outputs(), ov::ParameterVector{dataParam}, "Unsqueeze");
    }
};

TEST_P(DynamicUnsqueezeLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicUnsqueezeLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<BoundedShape> inShapes = {generateTestShape(1, 1, 10)};
const std::vector<Indices> indices = {Indices({3})};
const std::vector<InputType> inputPrecision = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke, DynamicUnsqueezeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(indices),
                                            ::testing::ValuesIn(inputPrecision)),
                         PrintTestCaseName());

}  // namespace ov::test
