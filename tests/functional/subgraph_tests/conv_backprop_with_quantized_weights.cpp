//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

namespace ov::test::subgraph {

struct ConvBackpropWithQuantizedWeightsParams {
    ov::Shape inDims;  // NCHW
    ov::Shape wDims;   // [C_in, C_out, kH, kW] (ConvolutionBackpropData weights layout)
    std::vector<uint64_t> strides;
    std::vector<int64_t> padsBegin;
    std::vector<int64_t> padsEnd;
    std::vector<uint64_t> dilations;
};

//        [input: u8, NCHW]
//              |
//    (dequantize: scale=0.012297759, zp=86)
//              |
//        [ConvTranspose] --- (dequantize weights: per-tensor scale=0.00035209375, zp=127)
//              |
//           (+ bias)
//              |
//    (quantize/dequantize: scale=0.0044795363, zp=131)
//              |
//           [output]
//

class ConvBackpropWithQuantizedWeightsTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvBackpropWithQuantizedWeightsParams> {
    void SetUp() override {
        const auto testParams = GetParam();
        const ov::Shape inputShape = testParams.inDims;
        const ov::Shape weightsShape = testParams.wDims;
        const auto outChannels = weightsShape.at(1);

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        // Activation fake-quantize: u8 quantized activation -> f16, scale=0.012297759, zero_point=86
        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {-1.057607f};
        const std::vector<float> dataInHigh = {2.078342f};
        const auto activationFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {},
                                                                      dataInLow, dataInHigh, dataInLow, dataInHigh);

        // Weights dequantize chain: u8 quantized weights -> f16, per-tensor scale=0.00035209375, zero_point=127.
        const auto weightsRaw = ov::op::v0::Constant::create(ov::element::u8, weightsShape,
                                                             std::vector<uint8_t>(ov::shape_size(weightsShape), 140));
        const auto weightsConverted = std::make_shared<ov::op::v0::Convert>(weightsRaw, ov::element::f16);
        const auto weightsZp = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, std::vector<float>{127.0f});
        const auto weightsShifted = std::make_shared<ov::op::v1::Subtract>(weightsConverted, weightsZp);
        const auto weightsScale =
                ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, std::vector<float>{0.00035209375f});
        const auto weightsDequantized = std::make_shared<ov::op::v1::Multiply>(weightsShifted, weightsScale);

        // ConvTranspose
        const ov::Strides strides = testParams.strides;
        const ov::CoordinateDiff padsBegin = testParams.padsBegin;
        const ov::CoordinateDiff padsEnd = testParams.padsEnd;
        const ov::Strides dilations = testParams.dilations;
        const auto autoPad = ov::op::PadType::EXPLICIT;
        const auto convTranspose = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                activationFq, weightsDequantized, strides, padsBegin, padsEnd, dilations, autoPad,
                ov::CoordinateDiff{0, 0});

        // Bias: per-channel f16 constant.
        std::vector<float> biasValues(outChannels);
        for (size_t i = 0; i < outChannels; ++i) {
            biasValues[i] = static_cast<float>(i) * 4.3162786e-06f * 10.0f;
        }
        const auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, outChannels, 1, 1}, biasValues);
        const auto biasedConvTranspose = std::make_shared<ov::op::v1::Add>(convTranspose, bias);

        // Output quantize/dequantize chain: u8, scale=0.0044795363, zero_point=131.
        const std::vector<float> outLow = {-0.586839f};
        const std::vector<float> outHigh = {0.555462f};
        const auto outputFq = ov::test::utils::make_fake_quantize(biasedConvTranspose, ov::element::f16, dataLevels, {},
                                                                  outLow, outHigh, outLow, outHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "ConvBackpropWithQuantizedWeightsTest");
        rel_threshold = 0.02f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvBackpropWithQuantizedWeightsParams>& obj) {
        const auto& p = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "in={" << p.inDims.at(0) << "," << p.inDims.at(1) << "," << p.inDims.at(2) << "," << p.inDims.at(3)
               << "}" << sep;
        return result.str();
    };
};

class ConvBackpropWithQuantizedWeightsTest_NPU5010 : public ConvBackpropWithQuantizedWeightsTest {};

TEST_P(ConvBackpropWithQuantizedWeightsTest_NPU5010, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

//        [input: u8, NCHW]
//              |
//    (dequantize: scale=0.012297759, zp=86)
//              |
//        [ConvTranspose] --- (dequantize weights: per-channel scale, zp=0)
//              |
//           (+ bias)
//              |
//    (quantize/dequantize: scale=0.0044795363, zp=131)
//              |
//           [output]
//
class ConvBackpropWithPerChannelWeightsNoZpTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvBackpropWithQuantizedWeightsParams> {
    void SetUp() override {
        const auto testParams = GetParam();
        const ov::Shape inputShape = testParams.inDims;
        const ov::Shape weightsShape = testParams.wDims;
        const auto outChannels = weightsShape.at(1);

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        // Activation fake-quantize: u8 quantized activation -> f16, scale=0.012297759, zero_point=86
        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {-1.057607f};
        const std::vector<float> dataInHigh = {2.078342f};
        const auto activationFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {},
                                                                      dataInLow, dataInHigh, dataInLow, dataInHigh);

        // Weights dequantize chain: u8 quantized weights -> f16, per-channel scale (no zero-point, i.e. zp=0).
        const auto weightsRaw = ov::op::v0::Constant::create(ov::element::u8, weightsShape,
                                                             std::vector<uint8_t>(ov::shape_size(weightsShape), 140));
        const auto weightsConverted = std::make_shared<ov::op::v0::Convert>(weightsRaw, ov::element::f16);
        std::vector<float> weightsScaleValues(outChannels);
        for (size_t i = 0; i < outChannels; ++i) {
            weightsScaleValues[i] = 0.00035209375f + static_cast<float>(i) * 1e-6f;
        }
        const auto weightsScale =
                ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, outChannels, 1, 1}, weightsScaleValues);
        const auto weightsDequantized = std::make_shared<ov::op::v1::Multiply>(weightsConverted, weightsScale);

        // ConvTranspose
        const ov::Strides strides = testParams.strides;
        const ov::CoordinateDiff padsBegin = testParams.padsBegin;
        const ov::CoordinateDiff padsEnd = testParams.padsEnd;
        const ov::Strides dilations = testParams.dilations;
        const auto autoPad = ov::op::PadType::EXPLICIT;
        const auto convTranspose = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                activationFq, weightsDequantized, strides, padsBegin, padsEnd, dilations, autoPad,
                ov::CoordinateDiff{0, 0});

        // Bias: per-channel f16 constant.
        std::vector<float> biasValues(outChannels);
        for (size_t i = 0; i < outChannels; ++i) {
            biasValues[i] = static_cast<float>(i) * 4.3162786e-06f * 10.0f;
        }
        const auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, outChannels, 1, 1}, biasValues);
        const auto biasedConvTranspose = std::make_shared<ov::op::v1::Add>(convTranspose, bias);

        // Output quantize/dequantize chain: u8, scale=0.0044795363, zero_point=131.
        const std::vector<float> outLow = {-0.586839f};
        const std::vector<float> outHigh = {0.555462f};
        const auto outputFq = ov::test::utils::make_fake_quantize(biasedConvTranspose, ov::element::f16, dataLevels, {},
                                                                  outLow, outHigh, outLow, outHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "ConvBackpropWithPerChannelWeightsNoZpTest");
        rel_threshold = 0.02f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvBackpropWithQuantizedWeightsParams>& obj) {
        const auto& p = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "in={" << p.inDims.at(0) << "," << p.inDims.at(1) << "," << p.inDims.at(2) << "," << p.inDims.at(3)
               << "}" << sep;
        return result.str();
    };
};

class ConvBackpropWithPerChannelWeightsNoZpTest_NPU5010 : public ConvBackpropWithPerChannelWeightsNoZpTest {};

TEST_P(ConvBackpropWithPerChannelWeightsNoZpTest_NPU5010, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

// The spatial extent (64x64 in -> 128x128 out) is chosen so that outputH * outputW
// (128*128 = 16384) meets the MEM_LIMIT threshold in ConvBackpropDataToMultipleConvConversion
// (legalize_convbackpropdata.cpp), which is what selects the 3x3 kernel-split ("use3x3") optimization on arch >
// NPU40XX.
const auto convBackpropWithQuantizedWeightsSmallSize =
        ::testing::ValuesIn<ConvBackpropWithQuantizedWeightsParams>({ConvBackpropWithQuantizedWeightsParams{
                ov::Shape{1, 32, 64, 64},  // in dims
                ov::Shape{32, 32, 4, 4},   // weights dims (C_in, C_out, kH, kW)
                {2, 2},                    // strides
                {1, 1},                    // pads_begin
                {1, 1},                    // pads_end
                {1, 1}                     // dilations
        }});

INSTANTIATE_TEST_SUITE_P(smoke_ConvBackpropWithQuantizedWeights, ConvBackpropWithQuantizedWeightsTest_NPU5010,
                         convBackpropWithQuantizedWeightsSmallSize,
                         ConvBackpropWithQuantizedWeightsTest_NPU5010::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvBackpropWithPerChannelWeightsNoZp, ConvBackpropWithPerChannelWeightsNoZpTest_NPU5010,
                         convBackpropWithQuantizedWeightsSmallSize,
                         ConvBackpropWithPerChannelWeightsNoZpTest_NPU5010::getTestCaseName);

}  // namespace
