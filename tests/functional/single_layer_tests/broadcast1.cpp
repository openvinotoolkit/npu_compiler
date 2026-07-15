//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/broadcast.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test {

class BroadcastOpset1LayerTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        for (const auto& shape : targetInputStaticShapes) {
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 2000, 0, 32768);
            inputs[function->get_parameters().at(0)] = tensor;
        }
    }

protected:
    void SetUp() override {
        const ov::Shape inputShape{18, 1, 1};
        const std::vector<int32_t> targetShapeVals{18, 2, 64};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front());
        auto targetShapeConst = ov::op::v0::Constant::create(ov::element::i32, {3}, targetShapeVals);

        auto broadcast =
                std::make_shared<ov::op::v1::Broadcast>(param, targetShapeConst, ov::op::AutoBroadcastType::NUMPY);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(broadcast)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "BroadcastOpset1");
    }
};

TEST_F(BroadcastOpset1LayerTest, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_F(BroadcastOpset1LayerTest, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_F(BroadcastOpset1LayerTest, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_F(BroadcastOpset1LayerTest, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test
