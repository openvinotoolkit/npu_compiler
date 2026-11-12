//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/abs.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softplus.hpp"

namespace ov::test {

class AbsMultiplySoftPlusTestCommon : public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;

        const auto inputShape = Shape{1, 16, 720, 80};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        // splat -1.0 — guarantees Abs(x) * (-1) <= 0 for all x.
        const auto negOne =
                ov::op::v0::Constant::create(ov::element::f16, Shape{1, 1, 1, 1}, std::vector<float>{-1.0f});

        const auto absOp = std::make_shared<ov::op::v0::Abs>(input);
        const auto mulOp = std::make_shared<ov::op::v1::Multiply>(absOp, negOne);
        const auto softPlusOp = std::make_shared<ov::op::v4::SoftPlus>(mulOp);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(softPlusOp)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "AbsMultiplySoftPlusTest");
    }
};

TEST_F(AbsMultiplySoftPlusTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(AbsMultiplySoftPlusTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(AbsMultiplySoftPlusTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(AbsMultiplySoftPlusTestCommon, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_F(AbsMultiplySoftPlusTestCommon, NPU6010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

}  // namespace ov::test
