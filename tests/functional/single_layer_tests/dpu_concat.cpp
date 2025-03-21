// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <vpu_ov2_layer_test.hpp>

using namespace ov::test;
namespace {

class DpuConcatLayerTest : public VpuOv2LayerTest {
protected:
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        const ov::Shape input_1Shape{1, 384, 1, 1};
        const ov::Shape input_2Shape{1, 384, 1, 1};
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_1Shape, input_2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        ov::OutputVector concatInputs;
        for (size_t i = 0; i < params.size(); i++) {
            concatInputs.push_back(params[i]);
        }

        const auto concat = std::make_shared<ov::opset1::Concat>(concatInputs, 2);
        const auto results = ov::ResultVector{std::make_shared<ov::opset3::Result>(concat->output(0))};
        function = std::make_shared<ov::Model>(results, params, "DpuConcat");
    }
};

TEST_F(DpuConcatLayerTest, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(DpuConcatLayerTest, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace
