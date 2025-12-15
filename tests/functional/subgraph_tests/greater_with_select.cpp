//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/greater.hpp"
#include "openvino/op/select.hpp"

namespace ov::test::subgraph {
/*
     (in1)    (in2)
     1x129     1x1
       |        |
        --------
           |
       (Greater)   (in2)   (in3)
         1x129       2      1.5
           |         |       |
            -----------------
                     |
                  (Select)
                     |
                  (Result)

*/

struct GreaterWithSelectTestParams {
    ov::Shape inputShape;
    bool genCustomInput;
};
class GreaterWithSelectTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<GreaterWithSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GreaterWithSelectTestParams>& obj) {
        size_t genCustomInput = obj.param.genCustomInput;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "genCustomInput=" << std::boolalpha << obj.param.genCustomInput << sep;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        auto& [inputShape, genCustomInput] = GetParam();

        const auto& funcInputs = function->inputs();
        auto createAndFillTensorInput = [](ov::Shape inputStaticShape, bool genCustomInput) -> ov::Tensor {
            auto inputTensor = ov::Tensor{ov::element::i64, inputStaticShape};
            const auto totalSize =
                    std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
            int64_t customInputValue = genCustomInput ? 0 : 128;
            for (size_t i = 0; i < totalSize; ++i) {
                // force custom input generation
                inputData[i] = customInputValue;
            }
            return inputTensor;
        };
        auto tensorIn = createAndFillTensorInput(inputShapes[0], genCustomInput);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorIn});
    }

    void SetUp() override {
        auto& [inputShape, _] = GetParam();
        init_input_shapes(static_shapes_to_test_representation({ov::Shape{inputShape}}));
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes.front())};

        // Greater with const input (1x129 filled from 0,..128) and 1x1 input
        auto requiredIndices = std::vector<int64_t>(129);
        std::iota(requiredIndices.begin(), requiredIndices.end(), 0);
        const auto constInput = ov::op::v0::Constant::create(ov::element::i64, {1, 129}, requiredIndices);
        const auto greater =
                std::make_shared<ov::op::v1::Greater>(constInput, params[0], ov::op::AutoBroadcastType::NUMPY);

        // Select with tensor input1 and scalar input2 and input3
        const auto scalarIn2Select = ov::op::v0::Constant::create(ov::element::f16, {1}, {2});
        const auto scalarIn3Select = ov::op::v0::Constant::create(ov::element::f16, {1}, {1.5});
        const auto select = std::make_shared<ov::op::v1::Select>(greater, scalarIn2Select, scalarIn3Select,
                                                                 ov::op::AutoBroadcastType::NUMPY);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(select)};

        function = std::make_shared<ov::Model>(results, params, "GreaterWithSelect");
    }
};

TEST_P(GreaterWithSelectTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GreaterWithSelectTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(GreaterWithSelectTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<GreaterWithSelectTestParams> configs = {
        // {shape, input_generator}
        {{1}, true},
        {{1}, false}};

INSTANTIATE_TEST_SUITE_P(smoke_GreaterWithSelect_customInput, GreaterWithSelectTestCommon, ::testing::ValuesIn(configs),
                         GreaterWithSelectTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
