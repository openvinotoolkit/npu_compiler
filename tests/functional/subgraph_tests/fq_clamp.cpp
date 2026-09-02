//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::test::utils;

namespace ov::test {
namespace LayerTestsDefinitions {

using outFQAndClampRangesType = std::vector<std::pair<float, float>>;

using FQClampTestParams = std::tuple<ov::element::Type,  // inType
                                     ov::element::Type,  // outType
                                     outFQAndClampRangesType>;

class FQClampSubGraphTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FQClampTestParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 100, -50, 1, 1);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        outFQAndClampRangesType outFQAndClampRanges;
        std::tie(inType, outType, outFQAndClampRanges) = GetParam();
        rel_threshold = 0.5f;

        const ov::Shape inputShape{1, 16, 20, 20};
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        // Create out FQ
        auto outFQRanges = outFQAndClampRanges[0];
        const size_t dataLevels = 256;
        const std::vector<float> outDataLow = {outFQRanges.first};
        const std::vector<float> outDataHigh = {outFQRanges.second};
        const auto outFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, outDataLow,
                                                               outDataHigh, outDataLow, outDataHigh);

        // Create Clamp
        const ov::Shape convOutShape{1, 32, 20, 20};
        auto clampRanges = outFQAndClampRanges[1];
        std::vector<float> constantsValue{clampRanges.first, clampRanges.second};
        auto clamp = ov::test::utils::make_activation(outFq, ov::element::f16, ov::test::utils::Clamp, convOutShape,
                                                      constantsValue);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clamp)};
        function = std::make_shared<ov::Model>(results, params, "FQClamp");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<FQClampTestParams> obj) {
        ov::element::Type ip;
        ov::element::Type op;
        outFQAndClampRangesType outFQAndClampRanges;
        std::tie(ip, op, outFQAndClampRanges) = obj.param;

        auto outFQRanges = outFQAndClampRanges[0];
        auto clampRanges = outFQAndClampRanges[1];

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "outFQ={" << outFQRanges.first << ", " << outFQRanges.second << ", " << outFQRanges.first << ", "
               << outFQRanges.second << "}_";
        result << "clamp={" << clampRanges.first << ", " << clampRanges.second << "}_";
        return result.str();
    }
};

class FQClampSubGraphTest_NPU3720 : public FQClampSubGraphTestCommon {};

TEST_P(FQClampSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<outFQAndClampRangesType> outFQAndClampRanges = {
        {{0.f, 10.f}, {0.f, 5.f}},
        {{-20.15748f, 20.0f}, {-5.f, 5.f}},
        {{-20.f, .0f}, {-5.f, 0.f}},
};

const std::vector<ov::element::Type> inPrecisions = {ov::element::f16};

const std::vector<ov::element::Type> outPrecisions = {
        // Convert layer will be inserted because of FP32 output, that allows:
        // - Propagate Dequantize through the Clamp, since if there is Return after the Clamp, then we cannot do
        // it(E#35846)
        // - Avoid an error in ov::float16::ie_abs (C#101214)
        ov::element::f32};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inPrecisions), ::testing::ValuesIn(outPrecisions),
                                           ::testing::ValuesIn(outFQAndClampRanges));

INSTANTIATE_TEST_SUITE_P(precommit_FQClamp, FQClampSubGraphTest_NPU3720, basicCases,
                         FQClampSubGraphTest_NPU3720::getTestCaseName);

}  // namespace
}  // namespace ov::test
