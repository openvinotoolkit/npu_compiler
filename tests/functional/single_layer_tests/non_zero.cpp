//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset10.hpp>

namespace ov::test {

PRETTY_PARAM(BoundedShape, ov::test::InputShape);
PRETTY_PARAM(InputType, ov::element::Type);

using NonZeroLayerTestParams = std::tuple<BoundedShape, InputType>;

class NonZeroLayerTestCommon : public testing::WithParamInterface<NonZeroLayerTestParams>, public VpuOv2LayerTest {
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
        const auto& [inputShape, inputType] = GetParam();

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, shape));
        }

        auto nonZero = std::make_shared<ov::opset10::NonZero>(inputParams[0], ov::element::i64);
        inputParams[0]->set_friendly_name("input");

        auto convertI32 = std::make_shared<ov::op::v0::Convert>(nonZero, ov::element::i32);

        auto results = ov::ResultVector();
        for (size_t i = 0; i < convertI32->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset10::Result>(convertI32->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "NonZero");
    }
};

TEST_P(NonZeroLayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NonZeroLayerTestCommon, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<InputType> inputPrecision = {ov::element::f32, ov::element::i32};

const std::vector<BoundedShape> inShapesStatic = {staticShape(120), staticShape(8, 32), staticShape(4, 8, 20),
                                                  staticShape(1, 3, 3), staticShape(2, 4, 8, 20)};

INSTANTIATE_TEST_SUITE_P(smoke, NonZeroLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(inShapesStatic), ::testing::ValuesIn(inputPrecision)),
                         PrintTestCaseName());

}  // namespace ov::test
