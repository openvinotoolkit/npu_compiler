//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for E#163853 (K2 accuracy regression, PR #28182).
//
// The OptimizeTileOp IE pass folds Tile(bias_1xCx1x1) → Add(act, tile_out) when the
// resulting broadcast-Add is cheaper than a DPU Eltwise on equal-sized tensors.
// When the Add carries a post_op activation (e.g. ReLU), the pre-fix code either
// skipped the fold or incorrectly dropped the activation, producing wrong outputs
// for negative values. This test verifies the activation is preserved after the fold.
//
// Model graph:
//   param(1xCxHxW) ──► GroupConv(const_filter) ──► Add ──► ReLU ──► result
//   const_bias(1xCx1x1) ──► Tile([1,1,H,W]) ─────────────┘
//
// Using a constant filter and constant bias triggers the 'isSpatialBroadcastFromGroupConv'
// guard in FoldTileOpRewriter, forcing the Tile fold regardless of tensor size.
// Input values in [-1, 1] combined with filter=+0.5 and bias=-0.3 guarantee that
// roughly half the outputs are negative before ReLU, so a missing activation is
// immediately visible as a numerical mismatch against the CPU reference.

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/tile.hpp>

namespace ov::test::subgraph {

class TileBroadcastAddReluTest : public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        // Values in [-1, 1]: after GroupConv (weight=+0.5) and bias (-0.3), roughly half
        // the outputs fall below zero. If ReLU is dropped the negative values survive
        // and the comparison against the CPU reference fails.
        ov::test::utils::InputGenerateData inData;
        inData.start_from = -1.0;
        inData.range = 2.0;
        inData.resolution = 32768;
        VpuOv2LayerTest::inputs.insert(
                {funcInputs[0].get_node_shared_ptr(),
                 ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), targetShapes[0], inData)});
    }

protected:
    void SetUp() override {
        constexpr int64_t C = 64, H = 32, W = 32;
        const ov::Shape inputShape{1, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W)};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);

        // 1x1 depthwise GroupConv: groups=C, filter [C, 1, 1, 1, 1], weight=+0.5.
        // Output = input * 0.5, so input in [-1,1] → output in [-0.5, 0.5].
        const ov::Shape filterShape{static_cast<size_t>(C), 1, 1, 1, 1};
        auto filter = ov::op::v0::Constant::create(ov::element::f32, filterShape, std::vector<float>(C, 0.5f));

        auto groupConv = std::make_shared<ov::op::v1::GroupConvolution>(param, filter, ov::Strides{1, 1},
                                                                        ov::CoordinateDiff{0, 0},
                                                                        ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

        // Constant bias 1xCx1x1 = -0.3 → post-Add range ≈ [-0.8, 0.2], about 60% negative.
        const ov::Shape biasShape{1, static_cast<size_t>(C), 1, 1};
        auto bias = ov::op::v0::Constant::create(ov::element::f32, biasShape, std::vector<float>(C, -0.3f));

        // Tile bias from [1,C,1,1] to [1,C,H,W].
        // In the IE dialect this becomes: Const::DeclareOp → TileOp → AddOp.
        auto repeats = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, H, W});
        auto tile = std::make_shared<ov::op::v0::Tile>(bias, repeats);

        // Add → ReLU.  The IE canonicalization pass fuses ReLU into AddOp's post_op
        // attribute.  FoldTileOpRewriter then strips the post_op, removes the Tile,
        // and re-inserts a standalone ReLU — the path that was broken before this fix.
        auto add = std::make_shared<ov::op::v1::Add>(groupConv, tile);
        auto relu = std::make_shared<ov::op::v0::Relu>(add);

        function = std::make_shared<ov::Model>(relu->outputs(), ov::ParameterVector{param}, "TileBroadcastAddRelu");
    }
};

TEST_F(TileBroadcastAddReluTest, NPU3720_HW_TestKindSubgraph) {
    abs_threshold = 0.05f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(TileBroadcastAddReluTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = 0.05f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(TileBroadcastAddReluTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = 0.05f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(TileBroadcastAddReluTest, NPU5020_HW_TestKindSubgraph) {
    abs_threshold = 0.05f;
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test::subgraph
