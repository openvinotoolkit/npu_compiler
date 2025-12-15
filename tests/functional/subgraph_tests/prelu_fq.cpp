//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <openvino/opsets/opset1_decl.hpp>
#include <openvino/opsets/opset3_decl.hpp>
#include <openvino/opsets/opset7_decl.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/prelu.hpp"

namespace ov::test {

class FqPreluSubGraphTest : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 16, 320, 320};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const size_t dataLevels = 256;
        const std::vector<float> outDataLow = {-1.18359375f};
        const std::vector<float> outDataHigh = {1.84765625f};
        const auto outDataFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {},
                                                                   outDataLow, outDataHigh, outDataLow, outDataHigh);

        const auto negativeSlope = ov::opset1::Constant::create(ov::element::f16, {1}, std::vector<ov::float16>{0.1});
        auto postOp = std::make_shared<ov::opset7::PRelu>(outDataFq, negativeSlope);

        const ov::ResultVector results{std::make_shared<ov::opset3::Result>(postOp)};
        function = std::make_shared<ov::Model>(results, params, "FqPreluSubGraphTest");
        rel_threshold = 0.6f;
    }
};
TEST_F(FqPreluSubGraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_F(FqPreluSubGraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(FqPreluSubGraphTest, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

class PreluFqSubGraphTest : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 16, 320, 320};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const auto negativeSlope = ov::opset1::Constant::create(ov::element::f16, {1}, std::vector<ov::float16>{0.1});
        auto postOp = std::make_shared<ov::opset7::PRelu>(params[0], negativeSlope);

        const size_t dataLevels = 256;
        const std::vector<float> outDataLow = {-1.18359375f};
        const std::vector<float> outDataHigh = {1.84765625f};
        const auto outDataFq = ov::test::utils::make_fake_quantize(postOp, ov::element::f16, dataLevels, {}, outDataLow,
                                                                   outDataHigh, outDataLow, outDataHigh);

        const ov::ResultVector results{std::make_shared<ov::opset3::Result>(outDataFq)};
        function = std::make_shared<ov::Model>(results, params, "PreluFqSubGraphTest");
        rel_threshold = 0.6f;
    }
};
TEST_F(PreluFqSubGraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_F(PreluFqSubGraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(PreluFqSubGraphTest, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test
