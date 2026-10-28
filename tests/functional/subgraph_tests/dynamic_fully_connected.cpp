//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/shape_of.hpp"

#include "openvino/core/type/float16.hpp"

#include <random>
#include <sstream>
#include "pretty_test_arguments.hpp"

namespace ov::test::subgraph {

using DynamicFullyConnectedParams = std::tuple<ov::test::InputShape,  // input [rows(dyn), inChannels]
                                               size_t,                // outChannels
                                               ov::element::Type>;    // precision

class DynamicFullyConnectedTest :
        public testing::WithParamInterface<DynamicFullyConnectedParams>,
        public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto inputTensor = ov::Tensor(funcInputs[0].get_element_type(), inputShapes[0]);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        const auto elementType = funcInputs[0].get_element_type();

        if (elementType == ov::element::f32) {
            auto* data = inputTensor.data<float>();
            for (size_t i = 0; i < inputTensor.get_size(); ++i) {
                data[i] = dist(rng);
            }
        } else if (elementType == ov::element::f16) {
            auto* data = inputTensor.data<ov::float16>();
            for (size_t i = 0; i < inputTensor.get_size(); ++i) {
                data[i] = ov::float16(dist(rng));
            }
        }

        inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        const auto& [dataShape, outChannels, type] = this->GetParam();

        // f16 accumulates rounding over inChannels products, so allow a small tolerance while keeping f32 tight.
        rel_threshold = 0.01f;
        abs_threshold = 0.05f;

        init_input_shapes({dataShape});
        const auto inChannels = static_cast<size_t>(dataShape.first[1].get_length());

        const auto param = std::make_shared<ov::opset1::Parameter>(type, inputDynamicShapes.at(0));

        std::vector<float> weightsValues(outChannels * inChannels);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.06f, 0.06f);
        for (auto& v : weightsValues) {
            v = dist(rng);
        }
        const auto weights = ov::opset1::Constant::create(type, ov::Shape{outChannels, inChannels}, weightsValues);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(param->output(0), weights->output(0), false, true);

        std::vector<float> biasValues(outChannels);
        std::mt19937 biasRng(123);
        std::uniform_real_distribution<float> biasDist(0.0f, 0.4f);
        for (auto& v : biasValues) {
            v = biasDist(biasRng);
        }
        const auto bias = ov::opset1::Constant::create(type, ov::Shape{1, outChannels}, biasValues);
        const auto add = std::make_shared<ov::opset1::Add>(matmul->output(0), bias->output(0));

        // Second consumer keeps the FullyConnected result dynamic through ConvertFCToConv (see the class note).
        const auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(add->output(0), ov::element::i64);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(add->output(0)),
                                              std::make_shared<ov::opset1::Result>(shapeOf->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicFullyConnected");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicFullyConnectedParams>& obj) {
        const auto& [dataShape, outChannels, type] = obj.param;
        std::ostringstream result;
        result << "RowsBound_" << dataShape.first[0].get_max_length() << "_InChannels_"
               << dataShape.first[1].get_length() << "_OutChannels_" << outChannels << "_Type_" << type;
        return result.str();
    }
};

TEST_P(DynamicFullyConnectedTest, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicFullyConnectedTest, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<ov::test::InputShape> smokeInShapes = {
        generateTestShape(14_Dyn, 128),
        generateTestShape(32_Dyn, 64),
};

const std::vector<size_t> smokeOutChannels = {64};

const std::vector<ov::element::Type> smokePrecisions = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicFullyConnected, DynamicFullyConnectedTest,
                         ::testing::Combine(::testing::ValuesIn(smokeInShapes), ::testing::ValuesIn(smokeOutChannels),
                                            ::testing::ValuesIn(smokePrecisions)),
                         DynamicFullyConnectedTest::getTestCaseName);

}  // namespace ov::test::subgraph
