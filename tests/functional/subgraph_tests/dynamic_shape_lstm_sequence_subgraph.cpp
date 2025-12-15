//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset1_decl.hpp>
#include <openvino/pass/manager.hpp>

#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::test;
using namespace ov::test::utils;

namespace {

using LSTMSubgraphParams =
        std::tuple<std::vector<ov::test::InputShape>, ov::element::Type, ov::op::RecurrentSequenceDirection>;
using OutputVector = std::vector<ov::Output<ov::Node>>;

class LSTMSubgraphNPUTestBase : public testing::WithParamInterface<LSTMSubgraphParams>, public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSubgraphParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "IS=";

        ov::element::Type inputType;
        std::vector<ov::test::InputShape> shapes;
        ov::op::RecurrentSequenceDirection direction;

        std::tie(shapes, inputType, direction) = obj.param;

        for (auto shape : shapes) {
            result << vec2str(shape.second) << sep;
        }
        result << "direction=" << direction << sep;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 100;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }

protected:
    ov::ParameterVector inputParams;

    void SetUp() override {
        const auto& [shapes, typeForInput, direction] = this->GetParam();
        const auto& inputShapeForParameter = shapes[0];

        size_t hidden_size = 128;
        size_t input_size = 64;
        size_t batch_size = 1;

        size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;

        const auto dataShape = ov::test::InputShape{{batch_size, num_directions, hidden_size},
                                                    {{batch_size, num_directions, hidden_size}}};

        std::vector<ov::test::InputShape> inShapes = {inputShapeForParameter, dataShape, dataShape};

        init_input_shapes(inShapes);

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(typeForInput, inputDynamicShapes[0]));

        std::vector<int64_t> targetShape{1, -1, 64};
        auto reshapeConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape.size()}, targetShape);

        inputParams[0]->set_friendly_name("input_0");
        auto reshapedInput = std::make_shared<ov::opset1::Reshape>(inputParams[0], reshapeConst, true);

        auto X = reshapedInput->output(0);
        auto Y = std::make_shared<ov::op::v0::Parameter>(typeForInput,
                                                         ov::Shape{batch_size, num_directions, hidden_size});
        auto Z = std::make_shared<ov::op::v0::Parameter>(typeForInput,
                                                         ov::Shape{batch_size, num_directions, hidden_size});

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
        auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

        createLSTMSequence(X, Y, Z, seq_lengths, typeForInput, hidden_size, input_size, num_directions, direction);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(function);
    }

    virtual void createLSTMSequence(const ov::Output<ov::Node>& X, const std::shared_ptr<ov::op::v0::Parameter>& Y,
                                    const std::shared_ptr<ov::op::v0::Parameter>& Z,
                                    const std::shared_ptr<ov::op::v1::Gather>& seq_lengths,
                                    const ov::element::Type& typeForInput, size_t hidden_size, size_t input_size,
                                    size_t num_directions, ov::op::RecurrentSequenceDirection direction) = 0;
};

class LSTMSubgraphNPUTest : public LSTMSubgraphNPUTestBase {
protected:
    void createLSTMSequence(const ov::Output<ov::Node>& X, const std::shared_ptr<ov::op::v0::Parameter>& Y,
                            const std::shared_ptr<ov::op::v0::Parameter>& Z,
                            const std::shared_ptr<ov::op::v1::Gather>& seq_lengths,
                            const ov::element::Type& typeForInput, size_t hidden_size, size_t input_size,
                            size_t num_directions, ov::op::RecurrentSequenceDirection direction) override {
        auto w_val = std::vector<float>(num_directions * 4 * hidden_size * input_size, 0);
        auto r_val = std::vector<float>(num_directions * 4 * hidden_size * hidden_size, 0);
        auto b_val = std::vector<float>(num_directions * 4 * hidden_size, 0);

        auto W = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, input_size},
                                              w_val);
        auto R = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, hidden_size},
                                              r_val);
        auto B = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size}, b_val);

        auto lstm_sequence =
                std::make_shared<ov::op::v5::LSTMSequence>(X, Y, Z, seq_lengths, W, R, B, hidden_size, direction);

        auto Y_out = std::make_shared<ov::op::v0::Result>(lstm_sequence->output(0));
        auto Ho = std::make_shared<ov::op::v0::Result>(lstm_sequence->output(1));
        auto Co = std::make_shared<ov::op::v0::Result>(lstm_sequence->output(2));

        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        function =
                std::make_shared<ov::Model>(ov::NodeVector{Y_out, Ho, Co}, ov::ParameterVector{inputParams[0], Y, Z});
        function->set_friendly_name("LSTMSequenceSubgraphNPU");
    }
};

class LSTMSubgraphNPUTest2LSTMSeq : public LSTMSubgraphNPUTestBase {
protected:
    void createLSTMSequence(const ov::Output<ov::Node>& X, const std::shared_ptr<ov::op::v0::Parameter>& Y,
                            const std::shared_ptr<ov::op::v0::Parameter>& Z,
                            const std::shared_ptr<ov::op::v1::Gather>& seq_lengths,
                            const ov::element::Type& typeForInput, size_t hidden_size, size_t input_size,
                            size_t num_directions, ov::op::RecurrentSequenceDirection direction) override {
        auto createLSTM = [&](float init_val, const ov::Output<ov::Node>& input, size_t input_size) {
            auto w_val = std::vector<float>(num_directions * 4 * hidden_size * input_size, init_val);
            auto r_val = std::vector<float>(num_directions * 4 * hidden_size * hidden_size, init_val);
            auto b_val = std::vector<float>(num_directions * 4 * hidden_size, init_val);

            auto W = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, input_size},
                                                  w_val);
            auto R = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, hidden_size},
                                                  r_val);
            auto B = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size}, b_val);
            return std::make_shared<ov::op::v5::LSTMSequence>(input, Y, Z, seq_lengths, W, R, B, hidden_size,
                                                              direction);
        };

        auto lstm_sequence_1 = createLSTM(0, X, input_size);
        auto Y_out_1 = std::make_shared<ov::op::v0::Result>(lstm_sequence_1->output(0));
        auto Ho_1 = std::make_shared<ov::op::v0::Result>(lstm_sequence_1->output(1));
        auto Co_1 = std::make_shared<ov::op::v0::Result>(lstm_sequence_1->output(2));

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {2, 0, 1, 3});
        auto transposed_output = std::make_shared<ov::op::v1::Transpose>(lstm_sequence_1->output(0), transpose_order);

        std::vector<int64_t> targetShape1{-1, 1, 256};
        auto reshapeConst1 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape1.size()}, targetShape1);
        auto reshaped_output1 = std::make_shared<ov::op::v1::Reshape>(transposed_output, reshapeConst1, true);

        std::vector<int64_t> targetShape2{1, -1, 256};
        auto reshapeConst2 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape2.size()}, targetShape2);
        auto reshaped_output2 = std::make_shared<ov::op::v1::Reshape>(reshaped_output1, reshapeConst2, true);

        input_size = 256;

        auto lstm_sequence_2 = createLSTM(1, reshaped_output2, input_size);
        auto Y_out_2 = std::make_shared<ov::op::v0::Result>(lstm_sequence_2->output(0));
        auto Ho_2 = std::make_shared<ov::op::v0::Result>(lstm_sequence_2->output(1));
        auto Co_2 = std::make_shared<ov::op::v0::Result>(lstm_sequence_2->output(2));

        auto transposed_output2 = std::make_shared<ov::op::v1::Transpose>(lstm_sequence_2->output(0), transpose_order);

        auto reshaped_output3 = std::make_shared<ov::op::v1::Reshape>(transposed_output2, reshapeConst1, true);
        auto reshaped_output4 = std::make_shared<ov::op::v1::Reshape>(reshaped_output3, reshapeConst2, true);

        function = std::make_shared<ov::Model>(ov::NodeVector{Y_out_2, reshaped_output3, reshaped_output4},
                                               ov::ParameterVector{inputParams[0], Y, Z});
        function->set_friendly_name("LSTMSequenceSubgraphNPU");
    }
};

// Reproducing the problems of the target model
class LSTMSubgraphNPUTest2LSTMSeq_extended :
        public testing::WithParamInterface<LSTMSubgraphParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSubgraphParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "IS=";

        ov::element::Type inputType;
        std::vector<ov::test::InputShape> shapes;
        ov::op::RecurrentSequenceDirection direction;

        std::tie(shapes, inputType, direction) = obj.param;

        for (auto shape : shapes) {
            result << vec2str(shape.second) << sep;
        }
        result << "direction=" << direction << sep;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 100;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }

protected:
    ov::ParameterVector inputParams;

    void SetUp() override {
        const auto& [shapes, typeForInput, direction] = this->GetParam();
        const auto& inputShapeForParameter = shapes[0];

        size_t hidden_size = 128;
        size_t input_size = 64;
        size_t batch_size = 1;

        size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;

        const auto dataShape = ov::test::InputShape{{batch_size, num_directions, hidden_size},
                                                    {{batch_size, num_directions, hidden_size}}};

        std::vector<ov::test::InputShape> inShapes = {inputShapeForParameter, dataShape, dataShape};

        init_input_shapes(inShapes);

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(typeForInput, inputDynamicShapes[0]));

        // Additional operations for the more extended subgraph
        auto add_const =
                ov::op::v0::Constant::create(typeForInput, ov::Shape{1, 16, 1, 1}, std::vector<float>(16, 1.0f));
        auto add_output = std::make_shared<ov::op::v1::Add>(inputParams[0], add_const);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {2, 0, 1, 3});
        auto transposed_output = std::make_shared<ov::op::v1::Transpose>(add_output, transpose_order);

        std::vector<int64_t> targetShape1{-1, 1, 64};
        auto reshapeConst1 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape1.size()}, targetShape1);
        auto reshaped_output1 = std::make_shared<ov::op::v1::Reshape>(transposed_output, reshapeConst1, true);

        std::vector<int64_t> targetShape2{1, -1, 64};
        auto reshapeConst2 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape2.size()}, targetShape2);
        auto reshaped_output2 = std::make_shared<ov::op::v1::Reshape>(reshaped_output1, reshapeConst2, true);

        auto X = reshaped_output2->output(0);
        auto Y = std::make_shared<ov::op::v0::Parameter>(typeForInput,
                                                         ov::Shape{batch_size, num_directions, hidden_size});
        auto Z = std::make_shared<ov::op::v0::Parameter>(typeForInput,
                                                         ov::Shape{batch_size, num_directions, hidden_size});

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
        auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

        createLSTMSequence(X, Y, Z, seq_lengths, typeForInput, hidden_size, input_size, num_directions, direction);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(function);
    }

    void createLSTMSequence(const ov::Output<ov::Node>& X, const std::shared_ptr<ov::op::v0::Parameter>& Y,
                            const std::shared_ptr<ov::op::v0::Parameter>& Z,
                            const std::shared_ptr<ov::op::v1::Gather>& seq_lengths,
                            const ov::element::Type& typeForInput, size_t hidden_size, size_t input_size,
                            size_t num_directions, ov::op::RecurrentSequenceDirection direction) {
        auto createLSTM = [&](float init_val, const ov::Output<ov::Node>& input, size_t input_size) {
            auto w_val = std::vector<float>(num_directions * 4 * hidden_size * input_size, init_val);
            auto r_val = std::vector<float>(num_directions * 4 * hidden_size * hidden_size, init_val);
            auto b_val = std::vector<float>(num_directions * 4 * hidden_size, init_val);

            auto W = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, input_size},
                                                  w_val);
            auto R = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, hidden_size},
                                                  r_val);
            auto B = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size}, b_val);
            return std::make_shared<ov::op::v5::LSTMSequence>(input, Y, Z, seq_lengths, W, R, B, hidden_size,
                                                              direction);
        };

        auto lstm_sequence_1 = createLSTM(0, X, 64);
        auto Y_out_1 = std::make_shared<ov::op::v0::Result>(lstm_sequence_1->output(0));
        auto Ho_1 = std::make_shared<ov::op::v0::Result>(lstm_sequence_1->output(1));
        auto Co_1 = std::make_shared<ov::op::v0::Result>(lstm_sequence_1->output(2));

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {2, 0, 1, 3});
        auto transposed_output2 = std::make_shared<ov::op::v1::Transpose>(lstm_sequence_1->output(0), transpose_order);

        std::vector<int64_t> targetShape3{-1, 1, 256};
        auto reshapeConst3 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape3.size()}, targetShape3);
        auto reshaped_output3 = std::make_shared<ov::op::v1::Reshape>(transposed_output2, reshapeConst3, true);

        std::vector<int64_t> targetShape4{1, -1, 256};
        auto reshapeConst4 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape4.size()}, targetShape4);
        auto reshaped_output4 = std::make_shared<ov::op::v1::Reshape>(reshaped_output3, reshapeConst4, true);

        auto lstm_sequence_2 = createLSTM(1, reshaped_output4, 256);
        auto Y_out_2 = std::make_shared<ov::op::v0::Result>(lstm_sequence_2->output(0));
        auto Ho_2 = std::make_shared<ov::op::v0::Result>(lstm_sequence_2->output(1));
        auto Co_2 = std::make_shared<ov::op::v0::Result>(lstm_sequence_2->output(2));

        auto transposed_output3 = std::make_shared<ov::op::v1::Transpose>(lstm_sequence_2->output(0), transpose_order);

        auto reshaped_output5 = std::make_shared<ov::op::v1::Reshape>(transposed_output3, reshapeConst3, true);
        auto reshaped_output6 = std::make_shared<ov::op::v1::Reshape>(reshaped_output5, reshapeConst4, true);

        auto matmul_weights =
                ov::op::v0::Constant::create(typeForInput, ov::Shape{548, 256}, std::vector<float>(548 * 256, 1.0f));
        auto matmul_output = std::make_shared<ov::op::v0::MatMul>(reshaped_output6, matmul_weights, false, true);

        auto add_bias = ov::op::v0::Constant::create(typeForInput, ov::Shape{1, 1, 548}, std::vector<float>(548, 1.0f));
        auto add_output2 = std::make_shared<ov::op::v1::Add>(matmul_output, add_bias);

        auto softmax_output = std::make_shared<ov::op::v1::Softmax>(add_output2, 2);

        function = std::make_shared<ov::Model>(ov::NodeVector{Y_out_2, Ho_2, Co_2, softmax_output},
                                               ov::ParameterVector{inputParams[0], Y, Z});
        function->set_friendly_name("LSTMSequenceSubgraphNPU_More_Extended");
    }
};

//                    *------------------------*
//                    |     Input parameter    |
//                    |       (dynamic)        |
//                    *------------------------*
//                                 |
//                  *----------------------------*
//      ____________|           Reshape          |___
//     |            *----------------------------*   |
//     |                                             |
//     |    *---------*  *---------*      *------------------*
//     |    |  Const  |  |  Const  |      |     ShapeOf      |
//     |    *---------*  *---------*      *------------------*
//     |         |            |                      |
//     |         |            |           *------------------*
//     |         |            |           |      Gather      |
//     |         |            |           *------------------*
//     |         |            |                      |
//     |         |            |                      |
//     |         |            |                      |
//     |         |_________*-----------------------------------*
//     |___________________|    TensorIterator (body: LSTM)    |
//                         *-----------------------------------*
TEST_P(LSTMSubgraphNPUTest, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(LSTMSubgraphNPUTest, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

//                    *------------------------*
//                    |     Input parameter    |
//                    |       (dynamic)        |
//                    *------------------------*
//                                 |
//                  *----------------------------*
//      ____________|           Reshape          |___
//     |            *----------------------------*   |
//     |                                             |
//     |    *---------*  *---------*      *------------------*
//     |    |  Const  |  |  Const  |      |     ShapeOf      |
//     |    *---------*  *---------*      *------------------*
//     |         |            |                      |
//     |         |            |           *------------------*
//     |         |            |           |      Gather      |
//     |         |            |           *------------------*
//     |         |            |                      |
//     |         |            |                      |
//     |         |            |                      |
//     |         |__________*--------------------------------*
//     |____________________|         LSTMSequence (1st)     |
//                          *--------------------------------*
//                                          |
//                                   *----------------*
//                                   |   Transpose    |
//                                   *----------------*
//                                           |
//                                   *----------------*
//                                   |    Reshape     |
//                                   *----------------*
//                                           |
//                           *-------------------------------*
//                           |         LSTMSequence (2nd)    |
//                           *-------------------------------*
TEST_P(LSTMSubgraphNPUTest2LSTMSeq, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(LSTMSubgraphNPUTest2LSTMSeq, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

//                                      .. same ..
//                                           |
//                           *-------------------------------*
//                           |         LSTMSequence (2nd)    |
//                           *-------------------------------*
//                                           |
//                                   *----------------*
//                                   |   Transpose    |
//                                   *----------------*
//                                           |
//                                   *----------------*
//                                   |    Reshape     |
//                                   *----------------*
//                                           |
//                                   *----------------*
//                                   |    Reshape     |
//                                   *----------------*
//                                           |
//                                   *----------------*
//                                   |    MatMul      |
//                                   *----------------*
//                                           |
//                                   *----------------*
//                                   |      Add       |
//                                   *----------------*
//                                           |
//                                   *----------------*
//                                   |    SoftMax     |
//                                   *----------------*
TEST_P(LSTMSubgraphNPUTest2LSTMSeq_extended, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(LSTMSubgraphNPUTest2LSTMSeq_extended, NPU4000_HW_TestKindSubgraph_1Tile) {
    setDefaultHardwareMode();
    configuration["NPU_TILES"] = "1";
    run(Platform::NPU4000);
}

TEST_P(LSTMSubgraphNPUTest2LSTMSeq_extended, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(LSTMSubgraphNPUTest2LSTMSeq_extended, NPU5010_HW_TestKindSubgraph_1Tile) {
    setDefaultHardwareMode();
    configuration["NPU_TILES"] = "1";
    run(Platform::NPU5010);
}

const std::vector<ov::element::Type> inputType = {ov::element::f32};
const std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                                   ov::op::RecurrentSequenceDirection::REVERSE,
                                                                   ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};

const std::vector<std::vector<ov::test::InputShape>> inShapesShapeOfDataDynamic = {
        {{{ov::Dimension(4, 10), 1, 64}, {{6, 1, 64}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSubgraphNPUTest, LSTMSubgraphNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapesShapeOfDataDynamic),
                                            ::testing::ValuesIn(inputType), ::testing::ValuesIn(direction)),
                         LSTMSubgraphNPUTest::getTestCaseName);

const std::vector<ov::op::RecurrentSequenceDirection> bidirectional = {
        ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
const std::vector<std::vector<ov::test::InputShape>> inShapesShapeOfDataDynamicExtended = {
        {{{1, 16, 4, ov::Dimension(4, 10)}, {{1, 16, 4, 6}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSubgraphNPUTest2LSTMSeq, LSTMSubgraphNPUTest2LSTMSeq,
                         ::testing::Combine(::testing::ValuesIn(inShapesShapeOfDataDynamic),
                                            ::testing::ValuesIn(inputType), ::testing::ValuesIn(bidirectional)),
                         LSTMSubgraphNPUTest2LSTMSeq::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSubgraphNPUTest2LSTMSeq_extended, LSTMSubgraphNPUTest2LSTMSeq_extended,
                         ::testing::Combine(::testing::ValuesIn(inShapesShapeOfDataDynamicExtended),
                                            ::testing::ValuesIn(inputType), ::testing::ValuesIn(bidirectional)),
                         LSTMSubgraphNPUTest2LSTMSeq_extended::getTestCaseName);

}  // namespace
