//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

namespace ov::test {

// This test aims for:
//   - Check NCE weights with dequantize convert to fakeQuantize
//   - Check multiply(scales) has negative value can convert to positive value
// From:
//       [input]
//          |
//         (FQ)
//          |
//        (conv) --- (multiply) -- (subtract) -- (convert) -- [filter]
//          |
//       [output]
//          |
//         (FQ)
// To:
//       [input]
//          |
//         (FQ)
//          |
//        (conv) --- (FQ) -- (convert) -- [filter]
//          |
//       [output]
//          |
//         (FQ)

using HandleFakeQuantHasNegativeScalesTestParams = std::tuple<std::vector<int8_t>,  // weights values
                                                              std::vector<float>,   // zeroPoint values
                                                              std::vector<float>    // scale values
                                                              >;
class HandleFakeQuantHasNegativeScalesSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<HandleFakeQuantHasNegativeScalesTestParams> {
    void SetUp() override {
        std::vector<int8_t> weightsValues;
        std::vector<float> zeroPointValues;
        std::vector<float> scaleValues;
        std::tie(weightsValues, zeroPointValues, scaleValues) = GetParam();

        const ov::Shape inputShape{1, 3, 32, 32};
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        // Create input FQ
        const size_t dataLevels = 256;
        const std::vector<float> defaultFQLow = {0.0f};
        const std::vector<float> defaultFQHigh = {255.0f};
        const auto inputFq = ov::test::utils::make_fake_quantize(
                params[0], ov::element::f32, dataLevels, {}, defaultFQLow, defaultFQHigh, defaultFQLow, defaultFQHigh);

        // Create weights and convert
        const ov::Shape weightsShape{4, 3, 1, 1};
        const auto weights = ov::op::v0::Constant::create(ov::element::i8, weightsShape, weightsValues);
        const auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);

        // Create zeroPoint
        const ov::Shape zeroPointShape{weightsShape[0], 1, 1, 1};
        const auto zeroPoint = ov::op::v0::Constant::create(ov::element::f32, zeroPointShape, zeroPointValues);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zeroPoint);

        // Create scales
        const ov::Shape scalesShape{weightsShape[0], 1, 1, 1};
        const auto scales = ov::op::v0::Constant::create(ov::element::f32, scalesShape, scaleValues);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scales);

        // create Conv
        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(inputFq, multiply, strides, pads_begin, pads_end, dilations);

        // Create output FQ
        const auto outputFq = ov::test::utils::make_fake_quantize(conv, ov::element::f32, dataLevels, {}, defaultFQLow,
                                                                  defaultFQHigh, defaultFQLow, defaultFQHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "HandleFakeQuantHasNegativeScales");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<HandleFakeQuantHasNegativeScalesTestParams> obj) {
        std::vector<int8_t> weightsValues;
        std::vector<float> zeroPointValues;
        std::vector<float> scaleValues;
        std::tie(weightsValues, zeroPointValues, scaleValues) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "zeroPointValues={" << zeroPointValues.at(0) << ", " << zeroPointValues.at(1) << ", "
               << zeroPointValues.at(2) << ", " << zeroPointValues.at(3) << "}" << sep;
        result << "scaleValues={" << scaleValues.at(0) << ", " << scaleValues.at(1) << ", " << scaleValues.at(2) << ", "
               << scaleValues.at(3) << "}" << sep;
        return result.str();
    }
};

class HandleFakeQuantHasNegativeScalesSubGraphTest_NPU3720 :
        public HandleFakeQuantHasNegativeScalesSubGraphTestCommon {};

class HandleFakeQuantHasNegativeScalesSubGraphTest_NPU4000 :
        public HandleFakeQuantHasNegativeScalesSubGraphTestCommon {};

// NOTE: Without fusion of Dequantize: NCE (quantized output: int8) -> Dequantize -> float
//       - NCE Op rounds 0.49 to 0. Dequantize(0) = 0.0f. Grid-aligned.
//       - e.g. NCEOp with rounding -> Dequantize
//
//       With fusion of Dequantize: NCE (float output: mixed-precision) -> Clamp(0, max)
//       - No rounding because of mixed-precision
//       - e.g. NCEOp without rounding -> Dequantize
// This is the trade-off with Mixed-Precision. Clamp can never replace: Round -> Dequantize.
// Adjusting abs_threshold to account for +/- 0.5 variance.
TEST_P(HandleFakeQuantHasNegativeScalesSubGraphTest_NPU3720, HW) {
    rel_threshold = 0.5;
    abs_threshold = 0.5;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(HandleFakeQuantHasNegativeScalesSubGraphTest_NPU4000, HW) {
    rel_threshold = 0.5;
    abs_threshold = 0.5;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

std::vector<std::vector<int8_t>> weightsValues = {{-56, 96, -32, 8, 4, -54, -88, 68, -74, 54, -26, 34}};
std::vector<std::vector<float>> zeroPointValues = {{0.0f, 0.0f, 0.0f, 0.0f}};
std::vector<std::vector<float>> scaleValues = {{0.0121f, -0.0213f, -0.0135f, 0.0317f}};
const auto basicCases = ::testing::Combine(::testing::ValuesIn(weightsValues), ::testing::ValuesIn(zeroPointValues),
                                           ::testing::ValuesIn(scaleValues));

INSTANTIATE_TEST_SUITE_P(precommit_HandleFakeQuantHasNegativeScales,
                         HandleFakeQuantHasNegativeScalesSubGraphTest_NPU3720, basicCases,
                         HandleFakeQuantHasNegativeScalesSubGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_HandleFakeQuantHasNegativeScales,
                         HandleFakeQuantHasNegativeScalesSubGraphTest_NPU4000, basicCases,
                         HandleFakeQuantHasNegativeScalesSubGraphTestCommon::getTestCaseName);

}  // namespace ov::test
