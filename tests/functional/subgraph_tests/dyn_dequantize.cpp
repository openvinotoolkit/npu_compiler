//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace {
struct DynDeQuantShapes {
    const ov::Shape _input;
    const ov::Shape _scaleShape;
    ov::element::Type _inputType;
    ov::element::Type _outputType;
};
using DynDeQuantParams = std::tuple<DynDeQuantShapes>;
}  // namespace

namespace ov::test::subgraph {

class DynDQTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DynDeQuantParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compiler_dynamic_quantization.name()] = "YES";
    }

public:
    void SetUp() override {
        /* creates subgraph
        input(i4)
           |
        Convert   Scale
              \     /
              Multiply
                 |
               Output
        */
        const auto& [shapes] = GetParam();
        const std::vector<ov::Shape> inInferenceShapes = {shapes._input};
        const ov::test::InputShape inShape = {shapes._input, inInferenceShapes};
        const std::vector<ov::Shape> scaleInferenceShapes = {shapes._scaleShape};
        const ov::test::InputShape scaleShape = {shapes._scaleShape, scaleInferenceShapes};
        init_input_shapes({inShape, scaleShape});
        const auto input = std::make_shared<ov::opset1::Parameter>(shapes._inputType, inputDynamicShapes.at(0));
        const auto quantScale = std::make_shared<ov::opset1::Parameter>(shapes._outputType, inputDynamicShapes.at(1));
        const auto convert0 = std::make_shared<ov::opset1::Convert>(input->output(0), shapes._outputType);
        const auto mul = std::make_shared<ov::opset1::Multiply>(convert0->output(0), quantScale->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(mul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, quantScale}, "DynDQ");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynDeQuantParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [shapes] = obj.param;
        result << "InShape=" << shapes._input << sep;
        result << "ScaleShape=" << shapes._scaleShape;
        result << "inputType=" << shapes._inputType;
        result << "outputType=" << shapes._outputType;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(DynDQTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynDQTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
const std::vector<DynDeQuantShapes> shapes = {
        {{3, 30, 128}, {3, 30, 1}, ov::element::i4, ov::element::f16},
        {{3, 30, 128}, {3, 1, 128}, ov::element::i4, ov::element::f16},
        {{3, 14, 12}, {3, 1, 12}, ov::element::i4, ov::element::f16},
};

// Tracking number [E#144857]
INSTANTIATE_TEST_SUITE_P(DynDQ, DynDQTestCommon, ::testing::Combine(::testing::ValuesIn(shapes)),
                         DynDQTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
