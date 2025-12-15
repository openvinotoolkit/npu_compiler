//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace {
struct DynDQShapes {
    const ov::Shape _input;
    const ov::Shape _weightShape;
    const ov::Shape _scaleShape;
    const bool _transposeB;
};
using DynDQParams = std::tuple<ov::element::Type, ov::element::Type, DynDQShapes>;
}  // namespace

namespace ov::test::subgraph {

class MatMulWithDynDQTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DynDQParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compiler_dynamic_quantization.name()] = "YES";
    }

public:
    void SetUp() override {
        /* creates subgraph

        weights(as arg)
                |
         Convert(fp16)   QuantScale(as arg)
                    \     /
                    Multiply
                       |
           Input    Convert(optional)
               \    /
               Matmul
                 |
               Output
        */
        const auto& [_matmulOpType, _weightsType, shapes] = GetParam();

        const std::vector<ov::Shape> inInferenceShapes = {shapes._input};
        const ov::test::InputShape inShape = {shapes._input, inInferenceShapes};
        const std::vector<ov::Shape> wInferenceShapes = {shapes._weightShape};
        const ov::test::InputShape wShape = {shapes._weightShape, wInferenceShapes};
        const std::vector<ov::Shape> scaleInferenceShapes = {shapes._scaleShape};
        const ov::test::InputShape scaleShape = {shapes._scaleShape, scaleInferenceShapes};
        init_input_shapes({inShape, wShape, scaleShape});

        const auto input = std::make_shared<ov::opset1::Parameter>(_matmulOpType, inputDynamicShapes.at(0));
        const auto weights = std::make_shared<ov::opset1::Parameter>(_weightsType, inputDynamicShapes.at(1));
        const auto quantScale = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(2));
        const auto convert0 = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f16);
        const auto mul = std::make_shared<ov::opset1::Multiply>(convert0->output(0), quantScale->output(0));
        const auto convert1 = std::make_shared<ov::opset1::Convert>(mul->output(0), _matmulOpType);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(
                input->output(0), _matmulOpType == ov::element::f16 ? mul->output(0) : convert1->output(0), false,
                shapes._transposeB);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, weights, quantScale},
                                               "MatMulWithDynDQ");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynDQParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [_matmulOpType, _weightsType, shapes] = obj.param;
        result << "WeightsType=" << _weightsType << sep;
        result << "MatmulOpType=" << _matmulOpType << sep;
        result << "InShape=" << shapes._input << sep;
        result << "WeightShape=" << shapes._weightShape << sep;
        result << "ScaleShape=" << shapes._scaleShape;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(MatMulWithDynDQTestCommon, NPU4000_DebugTestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(MatMulWithDynDQTestCommon, NPU5010_DebugTestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<DynDQShapes> testShapes = {
        /*case1=*/{/*_input=*/{1, 1, 4096},
                   /*_weightShape=*/{4096, 4096},
                   /*_scaleShape=*/{4096, 1},
                   /*_transposeB=*/true},
        /*case2=*/{/*_input=*/{1, 64, 8192},
                   /*_weightShape=*/{3072, 8192},
                   /*_scaleShape=*/{3072, 1},
                   /*_transposeB=*/true}};

INSTANTIATE_TEST_SUITE_P(DynDQ_FP16_NF4, MatMulWithDynDQTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::f16), ::testing::Values(ov::element::nf4),
                                            ::testing::ValuesIn(testShapes)),
                         MatMulWithDynDQTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DynDQ_FP32_NF4, MatMulWithDynDQTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::f32), ::testing::Values(ov::element::nf4),
                                            ::testing::ValuesIn(testShapes)),
                         MatMulWithDynDQTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
