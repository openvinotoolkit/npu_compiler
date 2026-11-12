//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <limits>
#include <vector>

#include "openvino/op/matmul.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/opsets/opset1_decl.hpp"

namespace ov::test::subgraph {

class FuseOneHotSelectSubgraphTest : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape indicesShape{1, 16};
        const ov::Shape maskShape{1, 16, 1};
        init_input_shapes(static_shapes_to_test_representation({indicesShape, maskShape}));

        const auto indices = std::make_shared<ov::opset1::Parameter>(ov::element::i32, inputDynamicShapes.at(0));
        const auto mask = std::make_shared<ov::opset1::Parameter>(ov::element::boolean, inputDynamicShapes.at(1));

        const auto depth = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {8});
        const auto onValue = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
        const auto offValue = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
        const auto oneHot = std::make_shared<ov::op::v16::OneHot>(
                indices, depth, onValue, offValue, 2, ov::op::v16::OneHot::NegativeIndicesMode::IGNORE_NEGATIVE);

        const auto nanValue =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {std::numeric_limits<float>::quiet_NaN()});
        const auto select =
                std::make_shared<ov::opset1::Select>(mask, nanValue, oneHot, ov::op::AutoBroadcastType::NUMPY);
        const auto reshapeShape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {16, 8});
        const auto reshape = std::make_shared<ov::opset1::Reshape>(select, reshapeShape, false);

        const std::vector<float> weightValues = {
                0.5f,  1.0f,  1.5f,  2.0f,  2.5f,  3.0f,  3.5f,  4.0f,  2.0f,  4.0f,  6.0f,
                8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 6.0f,  12.0f, 18.0f, 24.0f, 30.0f, 36.0f,
                42.0f, 48.0f, 12.0f, 24.0f, 36.0f, 48.0f, 60.0f, 72.0f, 84.0f, 96.0f,
        };
        const auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, weightValues);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(reshape, weights, false, true);
        const ov::ResultVector results{std::make_shared<ov::opset1::Result>(matmul)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{indices, mask}, "FuseOneHotSelect");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        OPENVINO_ASSERT(targetInputStaticShapes.size() == 2, "Expected indices and mask inputs");

        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 2, "Expected indices and mask parameters");

        ov::Tensor indicesTensor{ov::element::i32, targetInputStaticShapes[0]};
        auto* indicesData = indicesTensor.data<int32_t>();
        for (size_t i = 0; i < ov::shape_size(targetInputStaticShapes[0]); ++i) {
            indicesData[i] = i == 7 ? -1 : (i == 8 ? 8 : static_cast<int32_t>((i * 5 + 3) % 8));
        }

        ov::Tensor maskTensor{ov::element::boolean, targetInputStaticShapes[1]};
        auto* maskData = maskTensor.data<bool>();
        for (size_t i = 0; i < ov::shape_size(targetInputStaticShapes[1]); ++i) {
            maskData[i] = i % 5 == 0;
        }

        inputs = {
                {funcInputs[0].get_node_shared_ptr(), indicesTensor},
                {funcInputs[1].get_node_shared_ptr(), maskTensor},
        };
    }
};

TEST_F(FuseOneHotSelectSubgraphTest, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(FuseOneHotSelectSubgraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(FuseOneHotSelectSubgraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(FuseOneHotSelectSubgraphTest, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_F(FuseOneHotSelectSubgraphTest, NPU6010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

}  // namespace ov::test::subgraph
