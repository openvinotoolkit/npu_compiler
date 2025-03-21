//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {
using SelectDuplicatedInputTestParams = ov::Shape;

class SelectDuplicatedInputTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SelectDuplicatedInputTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SelectDuplicatedInputTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };

    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
        for (size_t i = 0; i < totalSize; i += 2) {
            inputData[i] = 0.0f;
        }
        for (size_t i = 1; i < totalSize; i += 2) {
            inputData[i] = 1.0f;
        }
        inputs = {{funcInputs[0].get_node_shared_ptr(), inputTensor}};
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.05f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        const auto inputShape = GetParam();
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const ov::Shape constShape({1});
        std::vector<ov::float16> constValues(1, std::numeric_limits<ov::float16>::lowest());
        const auto constTensor = ov::op::v0::Constant::create(ov::element::f16, constShape, constValues);
        const auto conditionTensor = std::make_shared<ov::opset1::Convert>(params[0]->output(0), ov::element::boolean);
        const auto select = std::make_shared<ov::opset1::Select>(conditionTensor->output(0), constTensor->output(0),
                                                                 params[0]->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(select->output(0))};
        function = std::make_shared<ov::Model>(results, params, "SelectDuplicatedInput");
    }
};

//
// Platform test definition
//

TEST_P(SelectDuplicatedInputTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SelectDuplicatedInputTestCommon, NPU4000_HW) {
    // Tracking number [E#116403]
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "Incorrect inference results";
    });
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
INSTANTIATE_TEST_SUITE_P(smoke_SelectDuplicatedInput, SelectDuplicatedInputTestCommon,
                         ::testing::Values(SelectDuplicatedInputTestParams{
                                 1, 1, 1024, 1024  // input shape
                         }),
                         SelectDuplicatedInputTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
