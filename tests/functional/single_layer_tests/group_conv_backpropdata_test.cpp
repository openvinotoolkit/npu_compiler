// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class GroupConvBackpropDataInputFilterLayerTest : public VpuOv2LayerTest {
public:
    void SetUp() override {
        const ov::Shape staticInputShape{1, 64, 64, 64};
        const std::vector<ov::Shape> inferenceInputShapes = {staticInputShape};
        const ov::test::InputShape dataShape = {staticInputShape, inferenceInputShapes};

        const ov::Shape staticFilterShape{64, 1, 1, 4, 4};
        const std::vector<ov::Shape> inferenceFilterShapes = {staticFilterShape};
        const ov::test::InputShape filterShape = {staticFilterShape, inferenceFilterShapes};

        init_input_shapes({dataShape, filterShape});

        auto input = std::make_shared<ov::opset6::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        auto filter = std::make_shared<ov::opset6::Parameter>(ov::element::f16, inputDynamicShapes.at(1));

        ov::Strides strides = {2, 2};
        ov::CoordinateDiff pads_begin = {1, 1};
        ov::CoordinateDiff pads_end = {1, 1};
        ov::Strides dilations = {1, 1};

        // Example of GroupConvolutionBackpropData using non-constant inputs
        auto group_conv = std::make_shared<ov::opset6::GroupConvolutionBackpropData>(
                input, filter, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

        auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(group_conv)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, filter},
                                               "GroupConvBackpropDataInputFilter");
    }
};

TEST_F(GroupConvBackpropDataInputFilterLayerTest, NPU3720_HW) {
    // The threshold is marked because the test runs with fp16 precision
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(GroupConvBackpropDataInputFilterLayerTest, NPU4000_HW) {
    // The threshold is marked because the test runs with fp16 precision
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// NPU5010
TEST_F(GroupConvBackpropDataInputFilterLayerTest, NPU5010_HW) {
    // The threshold is marked because the test runs with fp16 precision
    abs_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test::subgraph
