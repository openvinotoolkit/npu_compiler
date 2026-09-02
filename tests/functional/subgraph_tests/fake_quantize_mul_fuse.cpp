//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test {

// FakeQuantizeMulFusion transformation replaces:
//       [input]    [Const]
//          |          |
//         (FQ)    (convert)
//          |          |
//        (conv) - (multiply)
//          |
//         (FQ)
//          |
//       [output]
// with:
//       [input]
//          |
//         (FQ)   [Const]
//          |        |
//        (conv) - (FQ)
//          |
//         (FQ)
//          |
//       [output]

using FakeQuantizeMulFuseTestParams = std::tuple<ov::element::Type,  // inPrc
                                                 ov::element::Type,  // outPrc
                                                 std::vector<float>  // fqRanges
                                                 >;
class FakeQuantizeMulFuseSubGraphTest1Common :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<FakeQuantizeMulFuseTestParams> {
    void SetUp() override {
        std::vector<float> dataFQRanges;
        std::tie(inType, outType, dataFQRanges) = GetParam();
        rel_threshold = 0.1f;

        const ov::Shape inputShape{1, 3, 10, 10};
        const ov::Shape weightsShape{4, 3, 1, 1};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {dataFQRanges.at(0)};
        const std::vector<float> dataInHigh = {dataFQRanges.at(1)};
        const std::vector<float> dataOutLow = {dataFQRanges.at(2)};
        const std::vector<float> dataOutHigh = {dataFQRanges.at(3)};
        const auto dataFq = ov::test::utils::make_fake_quantize(params[0], inType, dataLevels, {}, dataInLow,
                                                                dataInHigh, dataOutLow, dataOutHigh);

        std::vector<float> scalesVal{0.53, 0.13, 0.32, 0.51};
        const auto scales = ov::op::v0::Constant::create(inType, ov::Shape{weightsShape[0], 1, 1, 1}, scalesVal);

        std::vector<int8_t> weightsVal{-108, -120, -124, 8, 4, -106, -88, 113, -74, 54, 127, 0};
        const auto weights = ov::op::v0::Constant::create(ov::element::i8, weightsShape, weightsVal);
        const auto convert = std::make_shared<ov::op::v0::Convert>(weights, outType);

        const auto mul = std::make_shared<ov::op::v1::Multiply>(convert, scales);
        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(dataFq, mul, strides, pads_begin, pads_end, dilations);

        const auto outFq = ov::test::utils::make_fake_quantize(conv, outType, dataLevels, {}, dataInLow, dataInHigh,
                                                               dataOutLow, dataOutHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "FakeQuantizeMulFuse");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeMulFuseTestParams> obj) {
        ov::element::Type ip;
        ov::element::Type op;
        std::vector<float> fqRanges;
        std::tie(ip, op, fqRanges) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "FQ={" << fqRanges.at(0) << ", " << fqRanges.at(1) << ", " << fqRanges.at(2) << ", " << fqRanges.at(3)
               << "}_";
        return result.str();
    }
};

class FakeQuantizeMulFuseSubGraphTest1_NPU3720 : public FakeQuantizeMulFuseSubGraphTest1Common {};

// NOTE: Without fusion of Dequantize: NCE (quantized output: int8) -> Dequantize -> float
//       - NCE Op rounds 0.49 to 0. Dequantize(0) = 0.0f. Grid-aligned.
//       - e.g. NCEOp with rounding -> Dequantize
//
//       With fusion of Dequantize: NCE (float output: mixed-precision) -> Clamp(0, max)
//       - No rounding because of mixed-precision
//       - e.g. NCEOp without rounding -> Dequantize
// This is the trade-off with Mixed Precision. Clamp can never replace: Round -> Dequantize.
// Adjusting abs_threshold to account for +/- 0.5 variance.
TEST_P(FakeQuantizeMulFuseSubGraphTest1_NPU3720, HW) {
    abs_threshold = 0.5;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

std::vector<std::vector<float>> fqRangesM = {{0.0f, 255.0f, 0.0f, 255.0f}};

const std::vector<ov::element::Type> netPrecisionsM = {ov::element::f32};

const std::vector<ov::element::Type> netOutputPrecisionsM = {ov::element::f32};

const auto basicCasesM = ::testing::Combine(::testing::ValuesIn(netPrecisionsM),
                                            ::testing::ValuesIn(netOutputPrecisionsM), ::testing::ValuesIn(fqRangesM));

INSTANTIATE_TEST_SUITE_P(precommit_FakeQuantizeMulFuse, FakeQuantizeMulFuseSubGraphTest1_NPU3720, basicCasesM,
                         FakeQuantizeMulFuseSubGraphTest1_NPU3720::getTestCaseName);

}  // namespace ov::test
