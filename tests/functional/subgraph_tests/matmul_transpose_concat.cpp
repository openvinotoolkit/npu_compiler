//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::test::subgraph {

using InputShape = std::pair<ov::PartialShape, std::vector<ov::Shape>>;

class MatMulTransposeConcatTestCommon : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == funcInputs.size(),
                        "Input shapes number does not match with inputs number");

        auto create_and_fill_tensor = [](ov::Shape inputStaticShape) -> ov::Tensor {
            auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
            const auto totalSize =
                    std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            for (int64_t i = 0; i < totalSize; i++) {
                inputData[i] = (i % 255) - 127;
            }
            return inputTensor;
        };

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = create_and_fill_tensor(inputShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
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

    std::shared_ptr<ov::Node> buildMatMul(const ov::Output<ov::Node>& param, const ov::Shape& weightsShape, bool transA,
                                          bool transB) {
        const auto weightsSize =
                std::accumulate(weightsShape.cbegin(), weightsShape.cend(), 1, std::multiplies<size_t>());
        std::vector<float> values(weightsSize, 1.f);

        const auto weights = ov::op::v0::Constant::create(ov::element::f32, weightsShape, values);
        return std::make_shared<ov::op::v0::MatMul>(param, weights, transA, transB);
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode =
                std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::opset1::Reshape>(
                std::make_shared<ov::opset1::Reshape>(param, constNode, false));
        return reshape;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

    void SetUp() override {
        std::vector<size_t> lhsInputShapeVec = {1, 1, 512};
        std::vector<size_t> rhsInputShape = {512, 512};
        std::vector<size_t> constantInputShapeVec = {1, 8, 64, 447};

        InputShape lhsInputShape = {{}, std::vector<ov::Shape>{lhsInputShapeVec}};
        InputShape constantInputShape = {{}, std::vector<ov::Shape>{constantInputShapeVec}};
        init_input_shapes({lhsInputShape, constantInputShape});

        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        input1->set_friendly_name("input1");
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1]);
        input2->set_friendly_name("input2");

        const auto matmul = buildMatMul(input1, ov::Shape(rhsInputShape), false, true);

        std::vector<size_t> targetShape{1, 1, 8, 64};
        auto reshape = buildReshape(matmul->output(0), targetShape);

        const auto transpose1 = buildTranspose(reshape, std::vector<int64_t>{0, 2, 1, 3});

        const auto transpose2 = buildTranspose(input2, std::vector<int64_t>{0, 1, 3, 2});

        auto concat = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector({transpose2->output(0), transpose1->output(0)}), 2);

        const auto transposeOut = buildTranspose(concat, std::vector<int64_t>{0, 1, 3, 2});

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(transposeOut->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2}, "MatMulTransposeConcat");
    }
};

class MatMulTransposeConcatTest_NPU3720 : public MatMulTransposeConcatTestCommon {};

TEST_F(MatMulTransposeConcatTest_NPU3720, HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test::subgraph
