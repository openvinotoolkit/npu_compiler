//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>
#include "vpu_ov2_layer_test.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/pass/manager.hpp>

#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>

#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov::test;
using namespace ov::test::utils;

namespace {
using ov::test::InputShape;

static std::shared_ptr<ov::Model> makeLSTMSequence(ov::element::Type_t model_type, ov::PartialShape initShape, size_t N,
                                                   size_t I, size_t H,
                                                   ov::op::RecurrentSequenceDirection seq_direction) {
    size_t num_directions = seq_direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
    auto X = std::make_shared<ov::op::v0::Parameter>(model_type, initShape);
    auto Y = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{N, num_directions, H});
    auto Z = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{N, num_directions, H});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
    auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
    auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

    auto w_val = std::vector<float>(num_directions * 4 * H * I, 0);
    auto r_val = std::vector<float>(num_directions * 4 * H * H, 0);
    auto b_val = std::vector<float>(num_directions * 4 * H, 0);
    auto W = ov::op::v0::Constant::create(model_type, ov::Shape{num_directions, 4 * H, I}, w_val);
    auto R = ov::op::v0::Constant::create(model_type, ov::Shape{num_directions, 4 * H, H}, r_val);
    auto B = ov::op::v0::Constant::create(model_type, ov::Shape{num_directions, 4 * H}, b_val);

    auto rnn_sequence = std::make_shared<ov::op::v5::LSTMSequence>(X, Y, Z, seq_lengths, W, R, B, H, seq_direction);
    auto Y_out = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(0));
    auto Ho = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(1));
    auto Co = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(2));
    Y_out->set_friendly_name("Y_out");
    Ho->set_friendly_name("Ho");
    Co->set_friendly_name("Co");

    auto fn_ptr = std::make_shared<ov::Model>(ov::NodeVector{Y_out, Ho, Co}, ov::ParameterVector{X, Y, Z});
    fn_ptr->set_friendly_name("LSTMSequence");
    return fn_ptr;
}

enum class LSTMType { LSTMCell = 0, LSTMSequence = 1 };

using DynamicTensorIteratorNPUParams =
        typename std::tuple<LSTMType,    // LSTM type (LSTMCell, LSTMSequence)
                            InputShape,  // input shapes (N[batch], L[seq_length], I[input_size])
                            int32_t,     // hidden size
                            ov::op::RecurrentSequenceDirection,  // sequence direction
                            std::string,                         // device name
                            ov::element::Type>;                  // type

class DynamicTensorIteratorNPUTest :
        public testing::WithParamInterface<DynamicTensorIteratorNPUParams>,
        virtual public ov::test::SubgraphBaseTest,
        virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicTensorIteratorNPUParams>& obj) {
        LSTMType type;
        InputShape data_shapes;
        int32_t hidden_size;
        ov::op::RecurrentSequenceDirection seq_direction;
        std::string target_device;
        ov::element::Type model_type;
        std::tie(type, data_shapes, hidden_size, seq_direction, target_device, model_type) = obj.param;
        std::ostringstream result;
        result << "TestType=" << (type == LSTMType::LSTMCell ? "LSTMCell" : "LSTMSequence") << "_";
        result << "IS=(";
        result << ov::test::utils::partialShape2str({data_shapes.first}) << "_";
        result << ov::test::utils::vec2str(data_shapes.second) << "_";
        result << ")_";
        result << "hidden_size=" << hidden_size << "_";
        result << "direction=" << seq_direction << "_";
        result << "netPRC=" << model_type << "_";
        result << "targetDevice=" << target_device << "_";
        return result.str();
    }

private:
    InputShape data_shapes;
    ov::op::RecurrentSequenceDirection seq_direction;
    ov::element::Type model_type;
    size_t hidden_size;
    size_t batch_size;
    size_t input_size;
    LSTMType type;

protected:
    void SetUp() override {
        std::tie(type, data_shapes, hidden_size, seq_direction, targetDevice, model_type) = GetParam();
        auto init_shape = data_shapes.first;

        init_input_shapes({data_shapes});

        batch_size = static_cast<size_t>(init_shape[0].get_length());
        input_size = static_cast<size_t>(init_shape[init_shape.size() - 1].get_length());

        function = makeLSTMSequence(model_type, init_shape, batch_size, input_size, hidden_size, seq_direction);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(function);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        size_t numDirections = seq_direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        ov::Shape defaultShape{batch_size, numDirections, hidden_size};
        auto itTargetShape = targetInputStaticShapes.begin();
        int seed = 1;
        for (const auto& param : function->get_parameters()) {
            std::shared_ptr<ov::Node> inputNode = param;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto& node : param->get_output_target_inputs(i)) {
                    std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (itTargetShape != targetInputStaticShapes.end()) {
                            if (nodePtr->get_input_node_ptr(port)->shared_from_this() ==
                                inputNode->shared_from_this()) {
                                ov::Tensor tensor = ov::test::utils::create_and_fill_tensor_real_distribution(
                                        param->get_element_type(), *itTargetShape, -1.0f, 1.0f, seed++);
                                inputs.insert({param, tensor});
                                break;
                            }
                        } else {
                            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor_real_distribution(
                                    param->get_element_type(), defaultShape, -1.0f, 1.0f, seed++);
                            inputs.insert({param, tensor});
                        }
                    }
                }
            }
            if (itTargetShape != targetInputStaticShapes.end()) {
                itTargetShape++;
            }
        }
    }
};

TEST_P(DynamicTensorIteratorNPUTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = 0.0001f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicTensorIteratorNPUTest, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

std::vector<InputShape> input_shapes = {
        InputShape(ov::PartialShape({1, ov::Dimension(1, 35), 512}), {{1, 30, 512}, {1, 10, 512}, {1, 5, 512}})};

std::vector<int32_t> hidden_sizes = {128, 512};

std::vector<ov::element::Type> model_types = {ov::element::f32};

std::vector<ov::op::RecurrentSequenceDirection> recurent_sequence_direction = {
        ov::op::RecurrentSequenceDirection::FORWARD, ov::op::RecurrentSequenceDirection::REVERSE,
        ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicTensorIterator_LSTMSequence, DynamicTensorIteratorNPUTest,
                         testing::Combine(testing::ValuesIn({LSTMType::LSTMSequence}), testing::ValuesIn(input_shapes),
                                          testing::ValuesIn(hidden_sizes),
                                          testing::ValuesIn(recurent_sequence_direction),
                                          testing::Values<std::string>(test_utils::TARGET_DEVICE),
                                          testing::ValuesIn(model_types)),
                         DynamicTensorIteratorNPUTest::getTestCaseName);

}  // namespace
