//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"

namespace ov::test {

class AvgPoolWithConvTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ov::Shape, size_t>> {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;

        const auto inputShape = std::get<0>(GetParam());
        const auto kernelSize = std::get<1>(GetParam());

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        // AvgPool with KxK kernel & KxK strides
        auto avgPool = std::make_shared<ov::op::v1::AvgPool>(params.at(0), ov::Strides{kernelSize, kernelSize},
                                                             ov::Shape{0, 0}, ov::Shape{0, 0},
                                                             ov::Shape{kernelSize, kernelSize}, true);

        // Conv with 3x3 kernel
        size_t outputChannelNum = 48;
        const auto conv = buildConv(avgPool->output(0), outputChannelNum);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, params, "AvgPoolWithConvTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.output().tensor().set_layout("NHWC");
        preProc.output().model().set_layout("NCHW");
        function = preProc.build();

        rel_threshold = 0.1f;
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param, size_t outputChannelNum) {
        const ov::Shape& inputShape = param.get_shape();
        const auto weightsSize = inputShape.at(1) * outputChannelNum * 3 * 3;
        std::vector<float> values(weightsSize, 1.0f);
        const auto weightsShape = ov::Shape{outputChannelNum, inputShape.at(1), 3, 3};
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);

        const ov::Strides strides = ov::Strides(std::vector<size_t>{2, 2});
        const ov::CoordinateDiff padsBegin = ov::CoordinateDiff(std::vector<ptrdiff_t>{1, 1});
        const ov::CoordinateDiff padsEnd = ov::CoordinateDiff(std::vector<ptrdiff_t>{1, 1});
        const ov::Strides dilations = ov::Strides(std::vector<size_t>{1, 1});
        auto conv2dNode = std::make_shared<ov::op::v1::Convolution>(param, weights->output(0), strides, padsBegin,
                                                                    padsEnd, dilations);

        return conv2dNode;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<ov::Shape, size_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(AvgPoolWithConvTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(AvgPoolWithConvTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(AvgPoolWithConvTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
const std::vector<ov::Shape> testShapes = {
        {1, 3, 640, 640},

};

const std::vector<ov::Shape> testShapes2 = {
        {1, 1, 1024, 1024},

};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool2x2WithConv, AvgPoolWithConvTestCommon,
                         ::testing::Combine(::testing::ValuesIn(testShapes), ::testing::Values(2)),
                         AvgPoolWithConvTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool8x8WithConv, AvgPoolWithConvTestCommon,
                         ::testing::Combine(::testing::ValuesIn(testShapes2), ::testing::Values(8)),
                         AvgPoolWithConvTestCommon::getTestCaseName);

}  // namespace ov::test
