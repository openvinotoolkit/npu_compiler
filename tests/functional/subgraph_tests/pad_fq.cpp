//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <openvino/opsets/opset1_decl.hpp>
#include <openvino/opsets/opset3_decl.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/pad.hpp"

namespace ov::test {

class PadFqSubGraphTest : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 2, 1, 480};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        ov::ParameterVector params{
                std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        auto padsBeginConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                     std::vector<int64_t>{0, 0, 0, 0});
        auto padsEndConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                   std::vector<int64_t>{0, 0, 0, 480});
        float argPadValue = 0.0f;
        auto argPadValueConst = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, &argPadValue);
        auto pad = std::make_shared<ov::op::v12::Pad>(params[0], padsBeginConst, padsEndConst, argPadValueConst,
                                                      ov::op::PadMode::CONSTANT);
        const size_t dataLevels = 256;
        const std::vector<float> outDataLow = {-39.12467956542969};
        const std::vector<float> outDataHigh = {39.43274688720703};
        const auto outDataFq = ov::test::utils::make_fake_quantize(pad, ov::element::f32, dataLevels, {}, outDataLow,
                                                                   outDataHigh, outDataLow, outDataHigh);
        const ov::ResultVector results{std::make_shared<ov::opset3::Result>(outDataFq)};
        function = std::make_shared<ov::Model>(results, params, "PadFqSubGraphTest");
    }
};

TEST_F(PadFqSubGraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_F(PadFqSubGraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(PadFqSubGraphTest, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test
