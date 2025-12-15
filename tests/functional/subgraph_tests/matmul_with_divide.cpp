//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"

namespace ov::test::subgraph {

class MatMulWithDivideTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<ov::element::Type> {
    template <ov::element::Type_t T>
    static void populate_data_impl(ov::Tensor& tensor, size_t totalSize) {
        auto data = tensor.data<typename ov::element_type_traits<T>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            data[i] = std::sin(i);
        }
    }

    void populate_data(ov::Tensor& tensor, size_t totalSize) {
        switch (ov::element::Type_t(_elementType)) {
        case ov::element::f16:
            return populate_data_impl<ov::element::f16>(tensor, totalSize);
        case ov::element::f32:
            return populate_data_impl<ov::element::f32>(tensor, totalSize);
        default:
            OPENVINO_THROW_NOT_IMPLEMENTED("This test only makes sense with fp types");
        }
    }

    const ov::element::Type _elementType = GetParam();

public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        populate_data(inputTensor, totalSize);
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

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
        configuration["NPU_COMPILER_TYPE"] = "MLIR";

        // create a subgraph (MatMul -> Divide) that will be lowered into a HW
        // convolution
        constexpr size_t batchSize = 16;
        constexpr size_t numGroups = 3;
        constexpr size_t numColumns = 32;
        constexpr size_t numRows = 64;
        const std::vector<ov::Shape> inferenceShapes = {{batchSize, numColumns * numGroups}};
        const ov::test::InputShape dataShape = {{batchSize, numColumns * numGroups}, inferenceShapes};
        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(_elementType, inputDynamicShapes.at(0));
        const auto matmul = std::make_shared<ov::opset1::MatMul>(param->output(0), param->output(0), false, true);

        // Note: in order to test a MatMul -> Divide at MLIR level, one has to
        // preserve the "full" OV IR - with FQ - otherwise, frontend passes
        // would optimize the OV IR before it actually gets to MLIR -- instead
        // of testing MLIR compiler's accuracy we'd test something else.
        const auto fqData = ov::opset1::Constant::create(_elementType, ov::Shape{1}, {240.0});
        const auto fqInLow = ov::opset1::Constant::create(_elementType, ov::Shape{1}, {0.0});
        const auto fqInHigh = ov::opset1::Constant::create(_elementType, ov::Shape{1}, {255.0});
        const auto fqOutLow = ov::opset1::Constant::create(_elementType, ov::Shape{1}, {-8.0});
        const auto fqOutHigh = ov::opset1::Constant::create(_elementType, ov::Shape{1}, {7.0});
        constexpr size_t fqLevels = 256;
        const auto fq = std::make_shared<ov::opset1::FakeQuantize>(
                fqData->output(0), fqInLow->output(0), fqInHigh->output(0), fqOutLow->output(0), fqOutHigh->output(0),
                fqLevels, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));

        const auto divide = std::make_shared<ov::opset1::Divide>(
                matmul->output(0), fq->output(0), ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(divide->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MatMulWithDivide");
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

TEST_P(MatMulWithDivideTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulWithDivideTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(MatMulWithDivideTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
const std::vector<ov::element::Type> elementTypes = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(MatMulWithDivide, MatMulWithDivideTestCommon, ::testing::ValuesIn(elementTypes),
                         MatMulWithDivideTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
