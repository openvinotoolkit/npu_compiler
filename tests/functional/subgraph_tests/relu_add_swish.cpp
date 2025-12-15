//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/swish.hpp"

namespace ov::test::subgraph {

class ReluAddSwishTest : public VpuOv2LayerTest {
public:
    void SetUp() override {
        const ov::Shape inShape{1, 256, 56, 56};
        init_input_shapes(static_shapes_to_test_representation({inShape}));
        const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const auto weightTotalSize = ov::shape_size(inShape);
        std::vector<float> weightsData(weightTotalSize, 0);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = i % 32;
        }
        const auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f16, inShape, weightsData);
        const auto relu = std::make_shared<ov::op::v0::Relu>(param);
        const auto add = std::make_shared<ov::op::v1::Add>(relu, weights);
        const auto swish = std::make_shared<ov::op::v4::Swish>(add);

        const auto results = ov::ResultVector{std::make_shared<ov::op::v0::Result>(swish)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "ReluAddSwish");
    }
};

TEST_F(ReluAddSwishTest, NPU5010_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test::subgraph
