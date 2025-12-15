//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "openvino/op/prelu.hpp"

namespace ov::test {

class PRelu2InputsSubGraphTest : public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f32;

        const ov::Shape inputShape0{1, 24, 256, 256};
        const ov::Shape inputShape1{1, 1, 1, 1};

        init_input_shapes(static_shapes_to_test_representation({inputShape0, inputShape1}));

        const auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto input2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));

        auto postOp = std::make_shared<ov::op::v0::PRelu>(input1, input2);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(postOp)};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2}, "PRelu2InputsSubGraphTest");
    }
};

TEST_F(PRelu2InputsSubGraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(PRelu2InputsSubGraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test
