//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::test::subgraph {

using ConvBackpropData2dWithTransposeTestParams = std::tuple<ov::Layout,           // output layout
                                                             std::vector<int64_t>  // transpose permutation
                                                             >;

class ConvBackpropData2dWithTransposeTestCommon :
        public testing::WithParamInterface<ConvBackpropData2dWithTransposeTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvBackpropData2dWithTransposeTestParams> obj) {
        ov::Layout outLayout;
        std::vector<int64_t> permOrder;
        std::tie(outLayout, permOrder) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "outLayout={" << outLayout.to_string() << "}" << sep;
        result << "permOrder={" << permOrder.at(0) << ", " << permOrder.at(1) << ", " << permOrder.at(2) << ", "
               << permOrder.at(3) << "}_";
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

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

    void SetUp() override {
        std::vector<size_t> inputShapeVec = {1, 16, 64, 64};
        InputShape inputShape = {{}, std::vector<ov::Shape>({inputShapeVec})};
        init_input_shapes({inputShape});

        ov::Layout outLayout;
        std::vector<int64_t> perm;
        std::tie(outLayout, perm) = GetParam();

        const auto channelNum = 16;

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        input->set_friendly_name("input");

        const auto convBackprop = buildConvBackpropData2D(input, channelNum);
        const auto transpose = buildTranspose(convBackprop->output(0), perm);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(transpose)};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "ConvBackpropData2dWithTransposeTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.output().tensor().set_layout(outLayout);
        preProc.output().model().set_layout("NCHW");
        function = preProc.build();
    }
};

class ConvBackpropData2dWithTransposeTest_NPU3720 : public ConvBackpropData2dWithTransposeTestCommon {};

TEST_P(ConvBackpropData2dWithTransposeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

const std::vector<ov::Layout> outLayouts = {
        ov::Layout("NHWC"),
        ov::Layout("NCHW"),
};

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {2, 0, 1, 3},
        {3, 0, 1, 2}, {1, 2, 0, 3}, {2, 1, 0, 3}, {3, 1, 0, 2}, {1, 2, 3, 0}, {2, 1, 3, 0}, {3, 1, 2, 0},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvBackpropData2dWithTranspose, ConvBackpropData2dWithTransposeTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(outLayouts), ::testing::ValuesIn(transposes)),
                         ConvBackpropData2dWithTransposeTest_NPU3720::getTestCaseName);

}  // namespace
