//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset13_decl.hpp>

#include "openvino/op/concat.hpp"

namespace ov::test {

PRETTY_PARAM(BoundedShape, ov::test::InputShape);
PRETTY_PARAM(InputType, ov::element::Type);
PRETTY_PARAM(Axis, int32_t);

using ConcatLayerTestParams = std::tuple<std::vector<BoundedShape>, Axis, InputType>;

class DynamicConcatLayerTest : public testing::WithParamInterface<ConcatLayerTestParams>, public VpuOv2LayerTest {
protected:
    void SetUp() override {
        const auto& [boundedShapes, axis, inputType] = GetParam();

        auto inputShapes = std::vector<ov::test::InputShape>();
        for (const auto& boundedShape : boundedShapes) {
            inputShapes.push_back(boundedShape.value());
        }

        init_input_shapes(inputShapes);

        auto params = ov::ParameterVector();
        auto paramNodes = ov::NodeVector();
        for (auto i = 0; i < static_cast<int>(inputDynamicShapes.size()); i++) {
            const auto dataParam = std::make_shared<ov::opset13::Parameter>(inputType.value(), inputDynamicShapes[i]);
            const auto inputName = std::string("input_").append(std::to_string(i));
            dataParam->set_friendly_name(inputName);
            params.push_back(dataParam);
            paramNodes.push_back(dataParam);
        }

        const auto concat = std::make_shared<ov::opset13::Concat>(paramNodes, axis.value());

        function = std::make_shared<ov::Model>(concat->outputs(), params, "DynamicConcat");
    }

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
};

TEST_P(DynamicConcatLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicConcatLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicConcatLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<std::vector<BoundedShape>> inShapes = {
        {generateTestShape(1, 1, 640_Dyn, 128), generateTestShape(1, 1, 640_Dyn, 128)},
        {generateTestShape(1, 640_Dyn, 1, 128), generateTestShape(1, 1, 1, 128)}};
const std::vector<Axis> axis = {1};
const std::vector<InputType> inputPrecision = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicConcatOptimized, DynamicConcatLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(axis),
                                            ::testing::ValuesIn(inputPrecision)),
                         PrintTestCaseName());

}  // namespace ov::test
