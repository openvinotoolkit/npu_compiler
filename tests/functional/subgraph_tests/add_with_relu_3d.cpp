//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"

namespace ov::test::subgraph {

class AddRelu3DTests : public VpuOv2LayerTest {
public:
    void SetUp() override {
        const ov::Shape staticShape{16, 32, 64};
        const ov::PartialShape dynamicShape{16, 32, ov::Dimension(1, 128)};
        const std::vector<ov::Shape> inferenceShapes = {staticShape};
        const ov::test::InputShape dataShape = {dynamicShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const ov::Shape weightShape{1, 32, 1};
        const auto weightTotalSize = ov::shape_size(weightShape);
        std::vector<float> weightsData(weightTotalSize, 0);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = i % 32;
        }
        const auto weights = ov::opset1::Constant::create(ov::element::f16, weightShape, weightsData);
        const auto add = std::make_shared<ov::opset1::Add>(param, weights);
        const auto relu = std::make_shared<ov::opset1::Relu>(add);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(relu)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "AddRelu3D");
    }
};

TEST_F(AddRelu3DTests, NPU3720_HW_TestKindSubgraph) {
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(AddRelu3DTests, NPU4000_HW_TestKindSubgraph) {
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(AddRelu3DTests, NPU5010_HW_TestKindSubgraph) {
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_F(AddRelu3DTests, NPU5020_HW_TestKindSubgraph) {
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test::subgraph
