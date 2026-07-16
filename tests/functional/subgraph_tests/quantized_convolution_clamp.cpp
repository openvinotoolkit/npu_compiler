//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"

using namespace ov::test::utils;

namespace ov::test {
namespace LayerTestsDefinitions {

// This test aims for:
//   - Check that Dequantize layer is propagated through the Clamp
//   - Check that Convolution's ppe has right value of clamp_high and clamp_low attributes,
//     not just min/max value of int type
//
//       [input]
//          |
//         FQ
//          |
//        (conv) --- FQ -- [filter]
//          |
//          FQ
//          |
//        Clamp
//          |
//       [output]
//

using outFQAndClampRangesType = std::vector<std::pair<float, float>>;

using QuantizedConvClampTestParams = std::tuple<ov::element::Type,  // inType
                                                ov::element::Type,  // outType
                                                outFQAndClampRangesType>;

class QuantizedConvClampSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedConvClampTestParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto data_size = shape_size(targetInputStaticShapes[0]);
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 100, -50, 1, 1);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        outFQAndClampRangesType outFQAndClampRanges;
        std::tie(inType, outType, outFQAndClampRanges) = GetParam();
        rel_threshold = 0.5f;

        const ov::Shape inputShape{1, 16, 20, 20};
        const ov::Shape weightsShape{32, 16, 1, 1};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};

        // Create FQ for input
        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {-127};
        const std::vector<float> dataInHigh = {128};
        const std::vector<float> dataOutLow = {-127};
        const std::vector<float> dataOutHigh = {128};
        const auto dataFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, dataInLow,
                                                                dataInHigh, dataOutLow, dataOutHigh);

        // Create FQ for weights
        const auto weightsU8 = ov::op::v0::Constant::create(ov::element::u8, weightsShape, std::vector<uint8_t>{1});
        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const size_t weightsLevels = 255;
        const auto weightsInLow =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{-127.0f});
        const auto weightsInHigh =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{127.0f});

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = -127.0f;
            perChannelHigh[i] = 127.0f;
        }

        const auto weightsOutLow =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape[0], 1, 1, 1}, perChannelLow);
        const auto weightsOutHigh =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape[0], 1, 1, 1}, perChannelHigh);

        const auto weightsFq = std::make_shared<ov::op::v0::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh,
                                                                          weightsOutLow, weightsOutHigh, weightsLevels);

        // Create Convolution
        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(dataFq, weightsFq, strides, pads_begin, pads_end, dilations);

        // Create out FQ
        auto outFQRanges = outFQAndClampRanges[0];
        const std::vector<float> outDataLow = {outFQRanges.first};
        const std::vector<float> outDataHigh = {outFQRanges.second};
        const auto outFq = ov::test::utils::make_fake_quantize(conv, ov::element::f32, dataLevels, {}, outDataLow,
                                                               outDataHigh, outDataLow, outDataHigh);

        // Create Clamp
        const ov::Shape convOutShape{1, 32, 20, 20};
        auto clampRanges = outFQAndClampRanges[1];
        std::vector<float> constantsValue{clampRanges.first, clampRanges.second};
        auto clamp = ov::test::utils::make_activation(outFq, ov::element::f16, ov::test::utils::Clamp, convOutShape,
                                                      constantsValue);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clamp)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedConvClamp");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedConvClampTestParams> obj) {
        ov::element::Type ip;
        ov::element::Type op;
        outFQAndClampRangesType outFQAndClampRanges;
        std::tie(ip, op, outFQAndClampRanges) = obj.param;

        auto outFQRanges = outFQAndClampRanges[0];
        auto clampRanges = outFQAndClampRanges[1];

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputPrec=" << ip << "}" << sep;
        result << "OutputPrec=" << op << "}" << sep;
        result << "outFQ={" << outFQRanges.first << ", " << outFQRanges.second << ", " << outFQRanges.first << ", "
               << outFQRanges.second << "}" << sep;
        result << "clamp={" << clampRanges.first << ", " << clampRanges.second << "}" << sep;
        return result.str();
    }
};

class QuantizedConvClampSubGraphTest_NPU3720 : public QuantizedConvClampSubGraphTestCommon {};

TEST_P(QuantizedConvClampSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

class QuantizedConvClampSubGraphTest_NPU4000 : public QuantizedConvClampSubGraphTestCommon {};

TEST_P(QuantizedConvClampSubGraphTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

class QuantizedConvClampSubGraphTest_NPU5010 : public QuantizedConvClampSubGraphTestCommon {};

TEST_P(QuantizedConvClampSubGraphTest_NPU5010, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<outFQAndClampRangesType> outFQAndClampRanges = {
        // FQ range > clamp range
        {{0.f, 10.f}, {0.f, 5.f}},
        {{-20.15748f, 20.0f}, {-5.f, 5.f}},
        {{-20.f, .0f}, {-5.f, 0.f}},
        // FQ range < clamp range
        {{-20.15748f, 20.0f}, {-25.f, 25.f}},
        // clamp range == FQ range
        {{-20.15748f, 20.0f}, {-20.15748f, 20.f}},
        // clamp range != FQ range
        {{-20.f, 0.0f}, {-10.f, 10.f}},
};

const std::vector<ov::element::Type> inPrecisions = {ov::element::f16};

const std::vector<ov::element::Type> outrecisions = {
        // Convert layer will be inserted because of FP32 output, that allows:
        // - Propagate Dequantize through the Clamp, since if there is Return after the Clamp, then we cannot do
        // it(E#35846)
        // - Avoid an error in ov::float16::ie_abs (C#101214)
        ov::element::f32};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inPrecisions), ::testing::ValuesIn(outrecisions),
                                           ::testing::ValuesIn(outFQAndClampRanges));

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedConvClamp, QuantizedConvClampSubGraphTest_NPU3720, basicCases,
                         QuantizedConvClampSubGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedConvClamp, QuantizedConvClampSubGraphTest_NPU4000, basicCases,
                         QuantizedConvClampSubGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedConvClamp, QuantizedConvClampSubGraphTest_NPU5010, basicCases,
                         QuantizedConvClampSubGraphTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test
