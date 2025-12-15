// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/clamp.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reduce_l2.hpp"

namespace ov::test {

class ReduceL2ClampDivideTestCommon : public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;

        const auto inputShape = Shape{1024, 6, 64, 30};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        const auto reduceL2 = std::make_shared<ov::op::v4::ReduceL2>(
                input, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), true);
        const auto clamp = std::make_shared<ov::op::v0::Clamp>(reduceL2, 1e-9, std::numeric_limits<float>::max());
        const auto divide = std::make_shared<ov::op::v1::Divide>(input, clamp);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(divide)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "ReduceL2ClampDivideTest");
    }
};

TEST_F(ReduceL2ClampDivideTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(ReduceL2ClampDivideTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(ReduceL2ClampDivideTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test
