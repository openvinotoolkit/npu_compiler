//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using ov::test::InputShape;

class DynamicPadSubgraphTest : virtual public VpuOv2LayerTest {
protected:
    void SetUp() override {
        // EfficientDet model subgraph testing PadOp functionality
        InputShape inputShape = InputShape{{Dimension(1, 100), 1}, std::vector<Shape>{{100, 1}}};
        outType = inType = ov::element::f32;

        init_input_shapes({inputShape});

        ParameterVector functionParams;
        functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front()));
        functionParams.front()->set_friendly_name("input");

        // Will introduce Slice opset8 to complete the subgraph -> Tracking number [E#194126]
        // Squeeze to remove dimension
        auto squeezeAxes =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{1});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(functionParams.front(), squeezeAxes);

        // ShapeOf to get dynamic shape
        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(squeeze, ov::element::i64);

        // ConvertLike to convert shape to i32
        auto convertLikeRef =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
        auto convertLike = std::make_shared<ov::op::v1::ConvertLike>(shapeOf, convertLikeRef);

        // Subtract: 100 - dynamic_size
        auto subtractConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{100});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(subtractConst, convertLike);

        // Pad with dynamic padding
        auto padsBegin =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
        auto padValue = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
        auto pad = std::make_shared<ov::op::v1::Pad>(squeeze, padsBegin, subtract, padValue, ov::op::PadMode::CONSTANT);

        const auto results = ov::ResultVector{std::make_shared<ov::op::v0::Result>(pad)};
        function = std::make_shared<ov::Model>(results, functionParams, "EfficientDetPadSubgraph");
    }
};

TEST_F(DynamicPadSubgraphTest, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(DynamicPadSubgraphTest, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(DynamicPadSubgraphTest, NPU5020_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_F(DynamicPadSubgraphTest, NPU6010_HW_TestKindSubgraph) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

}  // namespace ov::test::subgraph
