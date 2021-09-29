// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class KmbQuantizedChannelMajorConvSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<LayerTestsUtils::TargetDevice> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 3, 64, 64};
        const InferenceEngine::SizeVector weightsShape{48, 3, 3, 3};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {},
                                                              dataLow, dataHigh, dataLow, dataHigh);

        const auto weightsU8 =
                ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 254, 0);

        const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

        const size_t weightsLevels = 255;

        const auto weightsInLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {254.0f}, false);

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = 0.0f;
            perChannelHigh[i] = 1.0f;
        }

        const auto weightsOutLow = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelHigh, false);

        const auto weightsFq = std::make_shared<ngraph::opset2::FakeQuantize>(
                weightsFP32, weightsInLow, weightsInHigh, weightsOutLow, weightsOutHigh, weightsLevels);

        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(dataFq, weightsFq, strides, pads_begin,
                                                                        pads_end, dilations);
        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {255.0f};
        const auto outFq = ngraph::builder::makeFakeQuantize(conv, ngraph::element::f32, dataLevels, {}, outDataLow, outDataHigh, outDataLow, outDataHigh);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outFq)};
        function = std::make_shared<ngraph::Function>(results, params, "KmbQuantizedChannelMajorConv");

        targetDevice = GetParam();
        threshold = 0.1f;
    }
};

TEST_P(KmbQuantizedChannelMajorConvSubGraphTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbQuantizedChannelMajorConvSubGraphTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbQuantizedChannelMajorConvSubGraphTest,
                        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

}  // namespace
