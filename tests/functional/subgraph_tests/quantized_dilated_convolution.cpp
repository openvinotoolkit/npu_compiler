// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <openvino/opsets/opset14.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "openvino/opsets/opset1.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::test;

namespace {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//       [input]
//          |
//     (Dequantize)
//          |
//  (Dilated GroupConv) --- (Scale) -- (Shift) -- [filter]
//          |
//      (Dequantize)
//

using QuantizedDilatedConvTestParams = std::tuple<ov::Shape,          // input shape
                                                  std::vector<float>  // dilations
                                                  >;
class QuantizedDilatedConvSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedDilatedConvTestParams> {
    void SetUp() override {
        ov::Shape inputShape;
        std::vector<float> dataFQRanges;
        std::tie(inputShape, dataFQRanges) = GetParam();
        rel_threshold = 0.1f;

        const size_t IC = inputShape[1];
        const size_t KY = 3;
        const size_t KX = 3;

        const ov::Shape weightsShape = ov::Shape{IC, 1, 1, KY, KX};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {dataFQRanges.at(0)};
        const std::vector<float> dataInHigh = {dataFQRanges.at(1)};
        const std::vector<float> dataOutLow = {dataFQRanges.at(2)};
        const std::vector<float> dataOutHigh = {dataFQRanges.at(3)};
        const auto dataFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, dataInLow,
                                                                dataInHigh, dataOutLow, dataOutHigh);

        std::vector<uint8_t> weightsData(IC * 1 * KX * KY);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = 128 + i % 3;
        }
        weightsData.at(0) = 0;
        weightsData.at(1) = 255;
        const auto constLayerNode = ov::opset1::Constant::create(ov::element::u8, weightsShape, weightsData);
        const float zp = 128.f;

        auto convertNode = std::make_shared<ov::opset1::Convert>(constLayerNode->output(0), ov::element::f32);

        const auto zeroPoints = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{zp});
        const auto shiftNode = std::make_shared<ov::opset1::Subtract>(convertNode->output(0), zeroPoints->output(0));

        const auto scales = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto scaleNode = std::make_shared<ov::opset1::Multiply>(shiftNode->output(0), scales->output(0));

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {2, 2};
        const ov::CoordinateDiff pads_end = {2, 2};
        const ov::Strides dilations = {2, 2};

        const auto conv = std::make_shared<ov::opset1::GroupConvolution>(dataFq, scaleNode, strides, pads_begin,
                                                                         pads_end, dilations);
        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {255.0f};
        const auto outFq = ov::test::utils::make_fake_quantize(conv, ov::element::f32, dataLevels, {}, outDataLow,
                                                               outDataHigh, outDataLow, outDataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedDilatedConv");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedDilatedConvTestParams> obj) {
        ov::Shape inputShape;
        std::vector<float> fqRanges;
        std::tie(inputShape, fqRanges) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InShape="
               << "inputShape={" << inputShape.at(0) << ", " << inputShape.at(1) << ", " << inputShape.at(2) << ", "
               << inputShape.at(3) << "}_" << sep;
        result << "FQ={" << fqRanges.at(0) << ", " << fqRanges.at(1) << ", " << fqRanges.at(2) << ", " << fqRanges.at(3)
               << "}" << sep;
        return result.str();
    }
};

class QuantizedDilatedConvSubGraphTest_NPU4000 : public QuantizedDilatedConvSubGraphTestCommon {};

TEST_P(QuantizedDilatedConvSubGraphTest_NPU4000, HW) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-experimental-se-ptrs-operations=true";
    // E148533 - Dilated Conv is not accurate on multiple clusters when quantized
    configuration[ov::intel_npu::tiles.name()] = 1;
    run(Platform::NPU4000);
}

std::vector<std::vector<float>> fqRangesM = {{0.0f, 255.0f, 0.0f, 255.0f}};

std::vector<ov::Shape> inputSizesM = {{1, 64, 16, 16}, {1, 64, 32, 32}};

const auto basicCasesM = ::testing::Combine(::testing::ValuesIn(inputSizesM), ::testing::ValuesIn(fqRangesM));

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedConv, QuantizedDilatedConvSubGraphTest_NPU4000, basicCasesM,
                         QuantizedDilatedConvSubGraphTestCommon::getTestCaseName);

}  // namespace
