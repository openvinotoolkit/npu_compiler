//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convolution.hpp"
#include "openvino/op/tanh.hpp"

namespace ov::test {

class ConvWithTanhTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<
                std::tuple<ov::Shape, std::vector<size_t>, std::vector<ptrdiff_t>, std::vector<ptrdiff_t>>> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }
    void SetUp() override {
        const auto& [inShape, kernelSize, padBegin, padEnd] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const auto conv = buildConv(params.at(0), kernelSize, padBegin, padEnd);
        const auto tanh = std::make_shared<ov::op::v0::Tanh>(conv->output(0));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(tanh)};

        function = std::make_shared<ov::Model>(results, params, "ConvWithTanhTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& data, const std::vector<size_t>& kernelSize,
                                        const std::vector<ptrdiff_t>& padBegin, const std::vector<ptrdiff_t>& padEnd) {
        const ov::Shape& inputShape = data.get_shape();
        const auto weightsShape = ov::Shape{16, inputShape.at(1), kernelSize[0], kernelSize[1]};
        const auto weightsSize = shape_size(weightsShape);
        std::vector<float> values(weightsSize, 1.f);
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);
        return std::make_shared<ov::op::v1::Convolution>(
                data, weights->output(0), ov::Strides(std::vector<size_t>{1, 1}), ov::CoordinateDiff(padBegin),
                ov::CoordinateDiff(padEnd), ov::Strides(std::vector<size_t>{1, 1}));
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<
                    std::tuple<ov::Shape, std::vector<size_t>, std::vector<ptrdiff_t>, std::vector<ptrdiff_t>>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};
TEST_P(ConvWithTanhTest, NPU5010_HW) {
    abs_threshold = 0.0001;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<ov::Shape> inputShapes = {
        {1, 3, 16, 16},
        {1, 3, 32, 32},
};

const std::vector<std::vector<size_t>> kernelSizes = {
        {1, 1},
        {3, 3},
        {7, 7},
};

const std::vector<std::vector<ptrdiff_t>> padsBegin = {{0, 0}, {1, 1}};

const std::vector<std::vector<ptrdiff_t>> padsEnd = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_convTanh, ConvWithTanhTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(kernelSizes),
                                            ::testing::ValuesIn(padsBegin), ::testing::ValuesIn(padsEnd)),
                         ConvWithTanhTest::getTestCaseName);

}  // namespace ov::test
