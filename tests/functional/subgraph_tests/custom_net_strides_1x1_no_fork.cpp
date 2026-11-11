//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/opsets/opset1_decl.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"

#include "pretty_test_arguments.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace ov::test {

using CustomNetStrides1x1NoForkParams = std::tuple<ov::test::InputShape, ov::element::Type>;

class CustomNetStrides1x1NoForkTest :
        public testing::WithParamInterface<CustomNetStrides1x1NoForkParams>,
        public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& inputStaticShape = targetInputStaticShapes[0];

        ov::Tensor inputTensor =
                ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), inputStaticShape, 3, 0);

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        const auto& [dataShape, type] = this->GetParam();

        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(type, inputDynamicShapes.at(0));

        constexpr size_t channels = 16;

        auto x = buildConvBias(param, channels);
        x = buildConvBiasRelu(x, channels);
        x = buildConvBiasRelu(x, channels);
        x = buildConvBiasRelu(x, channels);
        x = buildConvBiasRelu(x, channels);
        x = buildConvBiasRelu(x, channels);
        x = buildConvBiasRelu(x, channels);

        const auto scale =
                ov::op::v0::Constant::create(type, ov::Shape{1, channels, 1, 1}, std::vector<float>(channels, 1.5f));
        const auto scaled = std::make_shared<ov::op::v1::Multiply>(x, scale);
        // Residual self-add matches the original graph (both inputs come from the Multiply).
        const auto residual = std::make_shared<ov::op::v1::Add>(scaled, scaled);

        auto out = buildConvBias(residual, channels);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(out->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "CustomNet");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout("NCHW");
        preProc.input().model().set_layout("NCHW");
        preProc.output().tensor().set_layout("NCHW");
        preProc.output().model().set_layout("NCHW");
        function = preProc.build();
    }

    std::shared_ptr<ov::Node> buildConv1x1(const ov::Output<ov::Node>& input, size_t outputChannels) {
        const auto inputShape = input.get_partial_shape();
        VPUX_THROW_UNLESS(inputShape[1].is_static(), "Conv input C dim can't be dynamic");
        const auto inputC = inputShape[1].get_length();

        const auto weightsShape = ov::Shape{outputChannels, vpux::checked_cast<uint64_t>(inputC), 1, 1};
        // Keep the repeated convolutions numerically stable. Unit weights multiply values by inputC at every layer,
        // making the FP16 reference overflow to infinity before the compiler behavior can be validated.
        const auto weightValue = 1.f / static_cast<float>(inputC);
        std::vector<float> weightsValues(ov::shape_size(weightsShape), weightValue);
        const auto weights = ov::op::v0::Constant::create(input.get_element_type(), weightsShape, weightsValues);

        const ov::Strides strides{1, 1};
        const ov::Strides dilations{1, 1};
        const ov::CoordinateDiff padsBegin{0, 0};
        const ov::CoordinateDiff padsEnd{0, 0};

        return std::make_shared<ov::op::v1::Convolution>(input, weights->output(0), strides, padsBegin, padsEnd,
                                                         dilations);
    }

    std::shared_ptr<ov::Node> buildConvBias(const ov::Output<ov::Node>& input, size_t outputChannels) {
        const auto conv = buildConv1x1(input, outputChannels);
        const auto bias = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{1, outputChannels, 1, 1},
                                                       std::vector<float>(outputChannels, 0.1f));
        return std::make_shared<ov::op::v1::Add>(conv, bias);
    }

    std::shared_ptr<ov::Node> buildConvBiasRelu(const ov::Output<ov::Node>& input, size_t outputChannels) {
        const auto bias = buildConvBias(input, outputChannels);
        return std::make_shared<ov::op::v0::Relu>(bias);
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomNetStrides1x1NoForkParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        const auto& [dataShape, type] = obj.param;

        result << "InferShapes" << sep;
        for (auto i : dataShape.second) {
            result << i << sep;
        }
        result << "Type_" << type;
        return result.str();
    }
};

TEST_P(CustomNetStrides1x1NoForkTest, NPU4000_HC_TestKindSubgraph) {
    abs_threshold = 0.02f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(CustomNetStrides1x1NoForkTest, NPU5010_HC_TestKindSubgraph) {
    abs_threshold = 0.02f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(CustomNetStrides1x1NoForkTest, NPU6010_HC_TestKindSubgraph) {
    abs_threshold = 0.02f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU6010);
}

const std::vector<ov::test::InputShape> inShapes = {
        generateTestShape(1, 16, 1080, 1920),
        generateTestShape(std::vector<BoundedDim>{1, 16, 1080_Dyn, 1920_Dyn}, hostCompileSmallShapesLimitationCallback),
};

const std::vector<ov::element::Type> inPrecisions = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_CustomNetStrides1x1NoFork, CustomNetStrides1x1NoForkTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inPrecisions)),
                         CustomNetStrides1x1NoForkTest::getTestCaseName);

}  // namespace ov::test
