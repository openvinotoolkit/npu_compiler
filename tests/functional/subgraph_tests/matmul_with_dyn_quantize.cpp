//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace {
struct DynQuantShapes {
    const ov::Shape _input;
    const ov::Shape _weightShape;
    const ov::Shape _scaleShape;
    const bool _transposeB;
};
using DynQuantParams = std::tuple<DynQuantShapes>;
}  // namespace

namespace ov::test::subgraph {

class MatMulWithDynQTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DynQuantParams> {
public:
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
        /* creates subgraph
        weights(as arg)
             |
          Convert   Cst
              \     /
              Subtract  QuantScale(as arg)
                    \     /
                    Multiply
                       |
           Input    Convert
               \    /
               Matmul
                 |
               Output
        */
        const auto& [shapes] = GetParam();

        const std::vector<ov::Shape> inInferenceShapes = {shapes._input};
        const ov::test::InputShape inShape = {shapes._input, inInferenceShapes};
        const std::vector<ov::Shape> wInferenceShapes = {shapes._weightShape};
        const ov::test::InputShape wShape = {shapes._weightShape, wInferenceShapes};
        const std::vector<ov::Shape> scaleInferenceShapes = {shapes._scaleShape};
        const ov::test::InputShape scaleShape = {shapes._scaleShape, scaleInferenceShapes};
        init_input_shapes({inShape, wShape, scaleShape});

        const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto weights = std::make_shared<ov::opset1::Parameter>(ov::element::u4, inputDynamicShapes.at(1));
        const auto quantScale = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(2));

        const auto convert0 = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f16);

        const auto cst = ov::opset1::Constant::create(ov::element::f16, {1, 1}, std::vector<float>{8.f});

        const auto subtract = std::make_shared<ov::opset1::Subtract>(convert0->output(0), cst->output(0));

        const auto mul = std::make_shared<ov::opset1::Multiply>(subtract->output(0), quantScale->output(0));

        const auto convert1 = std::make_shared<ov::opset1::Convert>(mul->output(0), ov::element::f32);

        const auto matmul =
                std::make_shared<ov::opset1::MatMul>(input->output(0), convert1->output(0), false, shapes._transposeB);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{input, weights, quantScale}, "MatMulWithDynQ");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynQuantParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [shapes] = obj.param;
        result << "InShape=" << shapes._input << sep;
        result << "WeightShape=" << shapes._weightShape << sep;
        result << "ScaleShape=" << shapes._scaleShape;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(MatMulWithDynQTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulWithDynQTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
const std::vector<DynQuantShapes> shapes = {
        /*case1=*/{/*_input=*/{1, 1, 4096},
                   /*_weightShape=*/{4096, 4096},
                   /*_scaleShape=*/{4096, 1},
                   /*_transposeB=*/true},
        /*case2=*/
        {/*_input=*/{1, 1024, 4096},
         /*_weightShape=*/{4096, 4096},
         /*_scaleShape=*/{4096, 1},
         /*_transposeB=*/true}};

// Tracking number [E#144857]
INSTANTIATE_TEST_SUITE_P(DISABLED_MatMulWithDynQ, MatMulWithDynQTestCommon,
                         ::testing::Combine(::testing::ValuesIn(shapes)), MatMulWithDynQTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
