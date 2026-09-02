//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"

namespace ov::test::subgraph {

using ConvBackpropData2dWithBiasTestParams = std::tuple<ov::Shape>;  // input shape

class ConvBackpropData2dWithBiasTestCommon :
        public testing::WithParamInterface<ConvBackpropData2dWithBiasTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvBackpropData2dWithBiasTestParams> obj) {
        std::vector<size_t> inputShape = std::get<0>(obj.param);

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShape={" << inputShape.at(0) << ", " << inputShape.at(1) << ", " << inputShape.at(2) << ", "
               << inputShape.at(3) << "}_";
        return result.str();
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    std::shared_ptr<ov::Node> buildConvBackpropData2D(const ov::Output<ov::Node>& param, size_t channelNum) {
        const ov::Shape inputShape = param.get_shape();

        const auto weightsSize = inputShape.at(1) * channelNum * 2 * 2;
        std::vector<float> values(weightsSize, 1.f);
        const auto weightsShape = ov::Shape{inputShape.at(1), channelNum, 2, 2};
        const auto weights = ov::op::v0::Constant::create(ov::element::f32, weightsShape, values);

        const ov::Strides strides{2, 2};
        const ov::Strides dilations{1, 1};
        const ov::CoordinateDiff paddingBegin{0, 0};
        const ov::CoordinateDiff paddingEnd{0, 0};
        const ov::CoordinateDiff outputPadding{1, 1};
        const ov::op::PadType autoPad = ov::op::PadType::EXPLICIT;

        auto convBackprop = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                param, weights->output(0), strides, paddingBegin, paddingEnd, dilations, autoPad, outputPadding);

        return convBackprop;
    }

    std::shared_ptr<ov::Node> buildBias(const ov::Output<ov::Node>& param, size_t channelNum) {
        const ov::Shape inputShape = param.get_shape();

        std::vector<float> biases(channelNum, 1.0);
        for (std::size_t i = 0; i < biases.size(); i++) {
            biases.at(i) = i * 0.25f;
        }
        auto bias_weights_node =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, channelNum, 1, 1}, biases.data());

        auto bias_node = std::make_shared<ov::op::v1::Add>(param, bias_weights_node->output(0));

        return bias_node;
    }

    void SetUp() override {
        std::vector<size_t> inputShapeVec = std::get<0>(GetParam());

        InputShape inputShape = {{}, std::vector<ov::Shape>({inputShapeVec})};
        init_input_shapes({inputShape});

        const auto channelNum = 16;

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        input->set_friendly_name("input");

        const auto convBackprop = buildConvBackpropData2D(input, channelNum);
        const auto bias = buildBias(convBackprop->output(0), channelNum);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(bias)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "ConvBackpropData2dWithBiasTest");
    }
};

class ConvBackpropData2dWithBiasTest_NPU3720 : public ConvBackpropData2dWithBiasTestCommon {};

TEST_P(ConvBackpropData2dWithBiasTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

const std::vector<ov::Shape> inputShapes = {
        {1, 3, 64, 64},
        {1, 16, 64, 64},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvBackpropData2dWithBias, ConvBackpropData2dWithBiasTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapes)),
                         ConvBackpropData2dWithBiasTest_NPU3720::getTestCaseName);

}  // namespace
