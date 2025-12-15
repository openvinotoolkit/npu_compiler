//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <random>

#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

namespace ov::test::subgraph {

// Test subgraph:
// ```
// Const -> Convert -> Multiply -> Subtract
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// becomes FQ (--weights-dequantize-to-fake-quantize)
// ```
// then:
// ```
// Const - FQ -> Subtract
//   |      |    ^^^^^^^^
//    \    /  converted to add (--convert-subtract-to-add)
//     \  /
// multiplied by -1 (same pass) -> makes wrong FQ input range (e.g. [-255; 0])
// ```
// At the moment the compiler fixes such invalid FQ in a separate pass (see
// --split-fake-quant) to bring the FQ input range (and weights) to "canonical"
// form (so that input range is either [-128; 127] or [0; 255] for 256 levels
// and i8/u8 data types).
//
// This test just makes sure that this strange case with malfunctioning compiler
// behavior passes accuracy.

class InvalidFqCreatedByCompiler : public VpuOv2LayerTest {
public:
    void SetUp() override {
        configuration["NPU_COMPILER_TYPE"] = "MLIR";

        ov::Shape inputShape = {1, 3, 20, 20};
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});

        const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));

        const auto weights = ov::opset1::Constant::create(ov::element::u8, ov::Shape{3, 1, 1}, {8, 249, 42});
        const auto scale = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {0.5});
        const auto convert = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f32);
        const auto multiply = std::make_shared<ov::opset1::Multiply>(
                convert->output(0), scale->output(0), ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));

        const auto subtract = std::make_shared<ov::opset1::Subtract>(
                input->output(0), multiply->output(0), ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(subtract->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "InvalidFqCreatedByCompiler");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "DataType=" << obj.param;
        return result.str();
    };
};

TEST_F(InvalidFqCreatedByCompiler, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(InvalidFqCreatedByCompiler, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_F(InvalidFqCreatedByCompiler, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace ov::test::subgraph
