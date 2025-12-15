//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test::subgraph {

using DynDeQuantParams = std::tuple<ov::Shape,           // input
                                    ov::Shape,           // scale
                                    ov::element::Type,   // inputType
                                    ov::element::Type>;  // outputType

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
        const auto& [inShape, scaleShape, iType, oType] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inShape, scaleShape}));
        const auto input = std::make_shared<ov::opset1::Parameter>(iType, inputDynamicShapes.at(0));
        const auto quantScale = std::make_shared<ov::opset1::Parameter>(oType, inputDynamicShapes.at(1));
        const auto convert0 = std::make_shared<ov::opset1::Convert>(input->output(0), oType);
        const auto mul = std::make_shared<ov::opset1::Multiply>(convert0->output(0), quantScale->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(mul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, quantScale}, "DynDQ");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynDeQuantParams>& obj) {
        const auto& [inShape, scaleShape, iType, oType] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "InShape=" << inShape << sep;
        result << "ScaleShape=" << scaleShape;
        result << "inputType=" << iType;
        result << "outputType=" << oType;
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

TEST_P(DynDQTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<DynDeQuantParams> params = {
        {{3, 30, 128}, {3, 30, 1}, ov::element::i4, ov::element::f16},
        {{3, 30, 128}, {3, 1, 128}, ov::element::i4, ov::element::f16},
        {{3, 14, 12}, {3, 1, 12}, ov::element::i4, ov::element::f16},
        {{16, 8, 32}, {1, 1, 1}, ov::element::nf4, ov::element::f16},
        {{16, 8, 32}, {16, 1, 1}, ov::element::nf4, ov::element::f16},
        {{16, 8, 32}, {1, 1, 32}, ov::element::nf4, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(DynDQ, DynDQTestCommon, ::testing::ValuesIn(params), DynDQTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
