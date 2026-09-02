//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov::test::subgraph {

class ShapeOfTest : public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        const auto& funcInputs = function->inputs();
        const auto& inputStaticShape = inputShapes.at(0);
        const auto totalSize = ov::shape_size(inputStaticShape);

        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::cos(i);
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto& expected = expectedTensors[0];
        const auto& actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.f;
        const float relThreshold = 0.f;
        ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
    }

    void SetUp() override {
        const std::vector<ov::Shape> inferenceShapes = {{1, 8, 32, 64}};
        const ov::test::InputShape dataShape = {{1, 8, ov::Dimension(1, 128), ov::Dimension(1, 128)}, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto shapeOf = std::make_shared<ov::opset1::ShapeOf>(param->output(0));
        // Convert is necessary because ShapeOf returns I64, which is not supported.
        const auto convert = std::make_shared<ov::opset1::Convert>(shapeOf->output(0), ov::element::i32);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(convert->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "ShapeOf");
    }
};

//
// Platform test definition
//

// TODO: [E#115058] Enable the test after adding dynamic shape support in the plugin
TEST_F(ShapeOfTest, DISABLED_NPU3720_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test::subgraph
