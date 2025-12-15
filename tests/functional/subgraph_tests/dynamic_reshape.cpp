//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::test::subgraph {

class NPUDynamicReshapeBaseTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        VPUX_THROW_UNLESS(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = i % 17;
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

        const float absThreshold = 0.01f;
        const float relThreshold = 0.01f;
        ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
    }
};

using DynamicReshapeConfig = std::tuple<std::vector<int64_t>>;

class NPUDynamicReshapeTest :
        public testing::WithParamInterface<DynamicReshapeConfig>,
        public NPUDynamicReshapeBaseTest {
public:
    void SetUp() override {
        // The test builds the following subgraph:
        // Input -> ShapeOf -> ReduceProd -> Divide -> Concat -> Reshape
        //       \_______________________________________________/^
        //
        // With [1, 3, 8, 15] shape, the transformation basically means this:
        // ShapeOf -> ReducerProd = 1 * 3 * 8 * 15 = 360 -> Divide by 2 = 180
        // Concat prefix + 180, where the prefix determines how to reshape the input.
        // Possible reshapes and their meaning:
        // 1. [1, 1, 2, 180] - self-explanatory.
        // 2. [0, 1, 2, 180] - copy the first dimension from the dimensions of the input tensor.
        // 3. [1, 1, -1, 180] - the third dimension is totalSize / remainderSize = 360 / 180 = 2.
        // where totalSize = 1 * 3 * 8 * 15, remainderSize = 1 * 1 * 180.
        const auto inferenceShapes = std::vector<ov::Shape>{{1, 3, 8, 15}};
        const auto dataShape =
                ov::test::InputShape{{1, 3, ov::Dimension(1, 10), ov::Dimension(1, 16)}, inferenceShapes};
        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto shapeOf = std::make_shared<ov::opset1::ShapeOf>(param->output(0));
        const auto reductionAxis = ov::opset1::Constant::create(ov::element::i64, {}, {0});
        const auto reduce =
                std::make_shared<ov::opset1::ReduceProd>(shapeOf->output(0), reductionAxis->output(0), true);
        const auto divideConst = ov::opset1::Constant::create(ov::element::i64, {1}, {2});
        const auto divide = std::make_shared<ov::opset1::Divide>(reduce->output(0), divideConst->output(0));
        const auto& concatValues = std::get<0>(GetParam());
        const auto prefix = ov::opset1::Constant::create(ov::element::i64, {concatValues.size()}, concatValues);
        const auto concat =
                std::make_shared<ov::opset1::Concat>(ov::OutputVector({prefix->output(0), divide->output(0)}), 0);
        const auto reshape = std::make_shared<ov::opset1::Reshape>(param->output(0), concat->output(0), true);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(reshape->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicReshape");
    }
};

using DynamicReshapeFusingConfig = std::tuple<std::vector<int64_t>, std::vector<int64_t>>;

class NPUDynamicReshapeFusingTest :
        public testing::WithParamInterface<DynamicReshapeFusingConfig>,
        public NPUDynamicReshapeBaseTest {
public:
    void SetUp() override {
        const auto inferenceShapes = std::vector<ov::Shape>{{1, 3, 8, 15}};
        const auto dataShape =
                ov::test::InputShape{{1, 3, ov::Dimension(1, 10), ov::Dimension(1, 16)}, inferenceShapes};
        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto shapeOf = std::make_shared<ov::opset1::ShapeOf>(param->output(0));
        const auto reductionAxis = ov::opset1::Constant::create(ov::element::i64, {}, {0});
        const auto reduce =
                std::make_shared<ov::opset1::ReduceProd>(shapeOf->output(0), reductionAxis->output(0), true);
        const auto divideConst = ov::opset1::Constant::create(ov::element::i64, {1}, {2});
        const auto divide = std::make_shared<ov::opset1::Divide>(reduce->output(0), divideConst->output(0));

        const auto& firstConcatValues = std::get<0>(GetParam());
        const auto firstPrefix =
                ov::opset1::Constant::create(ov::element::i64, {firstConcatValues.size()}, firstConcatValues);
        const auto firstConcat =
                std::make_shared<ov::opset1::Concat>(ov::OutputVector({firstPrefix->output(0), divide->output(0)}), 0);
        const auto firstReshape = std::make_shared<ov::opset1::Reshape>(param->output(0), firstConcat->output(0), true);

        const auto& secondConcatValues = std::get<1>(GetParam());
        const auto secondPrefix =
                ov::opset1::Constant::create(ov::element::i64, {secondConcatValues.size()}, secondConcatValues);
        const auto secondConcat =
                std::make_shared<ov::opset1::Concat>(ov::OutputVector({secondPrefix->output(0), divide->output(0)}), 0);
        const auto secondReshape =
                std::make_shared<ov::opset1::Reshape>(firstReshape->output(0), secondConcat->output(0), true);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(secondReshape->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicReshapeFusing");
    }
};

class NPUDynamicReshapeTransposeTest :
        public testing::WithParamInterface<DynamicReshapeFusingConfig>,
        public NPUDynamicReshapeBaseTest {
public:
    void SetUp() override {
        const auto inferenceShapes = std::vector<ov::Shape>{{1, 16, 4, 320}};
        const auto dataShape = ov::test::InputShape{{1, 16, 4, ov::Dimension(1, 320)}, inferenceShapes};
        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto transposeOrder = ov::opset1::Constant::create(ov::element::i64, {4}, {2, 0, 1, 3});
        const auto transpose = std::make_shared<ov::opset1::Transpose>(param->output(0), transposeOrder);

        const auto& firstReshapeValues = std::get<0>(GetParam());
        const auto firstReshapeConst =
                ov::opset1::Constant::create(ov::element::i64, {firstReshapeValues.size()}, firstReshapeValues);
        const auto firstReshape = std::make_shared<ov::opset1::Reshape>(transpose->output(0), firstReshapeConst, true);

        const auto& secondReshapeValues = std::get<1>(GetParam());
        const auto secondReshapeConst =
                ov::opset1::Constant::create(ov::element::i64, {secondReshapeValues.size()}, secondReshapeValues);
        const auto secondReshape =
                std::make_shared<ov::opset1::Reshape>(firstReshape->output(0), secondReshapeConst, true);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(secondReshape->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicReshapeTransposeFusing");
    }
};

//
// Platform test definition
//

TEST_P(NPUDynamicReshapeTest, NPU3720_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NPUDynamicReshapeTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(NPUDynamicReshapeTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(NPUDynamicReshapeFusingTest, NPU3720_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NPUDynamicReshapeFusingTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(NPUDynamicReshapeFusingTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(NPUDynamicReshapeTransposeTest, NPU3720_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NPUDynamicReshapeTransposeTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(NPUDynamicReshapeTransposeTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// DynamicReshape and DynamicReshapeFusing
const std::vector<std::vector<int64_t>> concatValues = {
        {1, 1, 2},
        {1, 1, -1},
        {0, 1, 2},
        {0, 1, -1},
};

const std::vector<std::vector<int64_t>> secondConcatValues = {
        {1, 2, 1}, {1, -1, 1}, {0, 2, 1}, {0, -1, 1}, {-1, 0, 1},
};

// DynamicReshapeTransposeFusing
const std::vector<std::vector<int64_t>> firstReshapeValues = {
        {0, -1, 64},
        {-1, 1, 64},
};

const std::vector<std::vector<int64_t>> secondReshapeValues = {
        {1, -1, 64},
};

INSTANTIATE_TEST_SUITE_P(DynamicReshape, NPUDynamicReshapeTest, ::testing::Combine(::testing::ValuesIn(concatValues)));
INSTANTIATE_TEST_SUITE_P(DynamicReshapeFusing, NPUDynamicReshapeFusingTest,
                         ::testing::Combine(::testing::ValuesIn(concatValues),
                                            ::testing::ValuesIn(secondConcatValues)));
INSTANTIATE_TEST_SUITE_P(DynamicReshapeTranspose, NPUDynamicReshapeTransposeTest,
                         ::testing::Combine(::testing::ValuesIn(firstReshapeValues),
                                            ::testing::ValuesIn(secondReshapeValues)));

}  // namespace ov::test::subgraph
