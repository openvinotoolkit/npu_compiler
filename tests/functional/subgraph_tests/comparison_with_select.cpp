//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_enums.hpp>
#include "common_test_utils/node_builders/comparison.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/select.hpp"

namespace ov::test::subgraph {
/*
     (in1)    (in2)
     1x129     1x1
       |        |
        --------
           |
     (Comparison)  (in2)   (in3)
         1x129       2      1.5
           |         |       |
            -----------------
                     |
                  (Select)
                     |
                  (Result)

*/

using ComparisonWithSelectTestParams = std::tuple<ov::test::utils::ComparisonTypes, ov::Shape, bool>;

class ComparisonWithSelectTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ComparisonWithSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonWithSelectTestParams>& obj) {
        auto [comparisonType, inputShape, genCustomInput] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "ComparisonType=" << comparisonType << sep;
        result << "InputShape=" << ov::test::utils::vec2str(inputShape) << sep;
        result << "GenCustomInput=" << std::boolalpha << genCustomInput << sep;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        auto& [_, inputShape, genCustomInput] = GetParam();

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
        auto& [comparisonType, inputShape, _] = GetParam();
        init_input_shapes(static_shapes_to_test_representation({ov::Shape{inputShape}}));
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes.front())};

        // Comparison with const input (1x129 filled from 0,..128) and 1x1 input
        auto requiredIndices = std::vector<int64_t>(129);
        std::iota(requiredIndices.begin(), requiredIndices.end(), 0);
        const auto constInput = ov::op::v0::Constant::create(ov::element::i64, {1, 129}, requiredIndices);
        const auto comparison = ov::test::utils::make_comparison(constInput, params[0], comparisonType);

        // Select with tensor input1 and scalar input2 and input3
        const auto scalarIn2Select = ov::op::v0::Constant::create(ov::element::f16, {1}, {2});
        const auto scalarIn3Select = ov::op::v0::Constant::create(ov::element::f16, {1}, {1.5});
        const auto select = std::make_shared<ov::op::v1::Select>(comparison, scalarIn2Select, scalarIn3Select,
                                                                 ov::op::AutoBroadcastType::NUMPY);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(select)};

        function = std::make_shared<ov::Model>(results, params, "ComparisonWithSelect");
    }
};

TEST_P(ComparisonWithSelectTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonWithSelectTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ComparisonWithSelectTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(ComparisonWithSelectTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<ov::test::utils::ComparisonTypes> comparisonTypes = {
        ov::test::utils::ComparisonTypes::GREATER, ov::test::utils::ComparisonTypes::GREATER_EQUAL,
        ov::test::utils::ComparisonTypes::LESS,    ov::test::utils::ComparisonTypes::LESS_EQUAL,
        ov::test::utils::ComparisonTypes::EQUAL,   ov::test::utils::ComparisonTypes::NOT_EQUAL};

const std::vector<ov::Shape> inputShapes = {{1}};

const std::vector<bool> genCustomInput = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_ComparisonWithSelect_customInput, ComparisonWithSelectTestCommon,
                         ::testing::Combine(::testing::ValuesIn(comparisonTypes), ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(genCustomInput)),
                         ComparisonWithSelectTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
