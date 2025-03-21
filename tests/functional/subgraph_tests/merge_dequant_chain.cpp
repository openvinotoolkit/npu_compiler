//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov::test::subgraph {

using MergeDequantChainTestParams = std::tuple<ov::Shape>;

class MergeDequantChainTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MergeDequantChainTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MergeDequantChainTestParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        const auto& [shape] = obj.param;
        result << "InputShape=" << shape << sep;
        return result.str();
    }

    void SetUp() override {
        const auto& [inputShape] = GetParam();
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));
        ov::ParameterVector params{std::make_shared<ov::opset6::Parameter>(ov::element::f32, ov::Shape(inputShape))};

        const size_t dataLevels = 256;
        const std::vector<float> inDataLow = {0.0f};
        const std::vector<float> inDataHigh = {255.0f};

        auto fqNode = ov::test::utils::make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, inDataLow,
                                                          inDataHigh, inDataLow, inDataHigh);
        auto convertNode = std::make_shared<ov::opset1::Convert>(fqNode, ov::element::u8);

        auto nextConvertNode1 = std::make_shared<ov::opset1::Convert>(convertNode, ov::element::f32);

        const auto zeroPoints1 = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto shiftNode1 = std::make_shared<ov::opset1::Subtract>(nextConvertNode1, zeroPoints1);

        const auto scales1 = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto scaleNode1 = std::make_shared<ov::opset1::Multiply>(shiftNode1, scales1);

        auto nextConvertNode2 = std::make_shared<ov::opset1::Convert>(convertNode, ov::element::f32);

        const auto zeroPoints2 = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto shiftNode2 = std::make_shared<ov::opset1::Subtract>(nextConvertNode2, zeroPoints2);

        const auto scales2 = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto scaleNode2 = std::make_shared<ov::opset1::Multiply>(shiftNode2, scales2);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scaleNode1),
                                       std::make_shared<ov::op::v0::Result>(scaleNode2)};
        function = std::make_shared<ov::Model>(results, params, "MergeDequantChain");
    }
};

TEST_P(MergeDequantChainTestCommon, NPU3720_HW) {
    rel_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MergeDequantChainTestCommon, NPU4000_HW) {
    rel_threshold = 0.5f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

ov::Shape inputShape = {1, 4, 16, 16};

INSTANTIATE_TEST_SUITE_P(smoke_MergeDequantChain, MergeDequantChainTestCommon,
                         ::testing::Combine(::testing::Values(inputShape)),
                         MergeDequantChainTestCommon::getTestCaseName);

}  // namespace
