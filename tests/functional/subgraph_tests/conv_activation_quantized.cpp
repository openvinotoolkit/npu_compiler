//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/node_builders/activation.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "common/quantization_utils.hpp"

#include "openvino/op/convolution.hpp"

namespace ov::test {

struct ConvQuantParams {
    std::optional<FakeQuantizeParams> inputQuant;
    std::optional<FakeQuantizeParams> weightsQuant;
    std::optional<FakeQuantizeParams> outputQuant;
};

using ConvActivationQuantParams = std::tuple<utils::ActivationTypes, ConvQuantParams>;
using ConvSwishQuantParams = std::tuple<float, ConvQuantParams>;

//      [input]
//         |
//    FakeQuantize (optional)
//         |
//    Convolution -- FakeQuantize -- [weights]
//         |          (optional)
//     Activation
//         |
//    FakeQuantize (optional)
//         |
//      [output]

class ConvWithActivationQuantizedTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvActivationQuantParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }

    void SetUp() override {
        rel_threshold = 0.004f;
        const auto& [activationType, quantParams] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({ov::Shape{1, 16, 16, 16}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto lastOutput = params.at(0)->get_default_output();
        if (quantParams.inputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.inputQuant)
                                 ->get_default_output();
        }

        auto weights = buildWeights(lastOutput.get_shape(), 0.5f);
        if (quantParams.weightsQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.weightsQuant)
                                 ->get_default_output();
        }

        lastOutput = buildConv(lastOutput, weights);

        lastOutput = activationType == utils::Swish
                             ? utils::make_activation(lastOutput, ov::element::f16, activationType, ov::Shape{}, {1.f})
                             : utils::make_activation(lastOutput, ov::element::f16, activationType);

        if (quantParams.outputQuant.has_value()) {
            const auto& outputQuant = quantParams.outputQuant.value();
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.outputQuant)
                                 ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutput)};

        function = std::make_shared<ov::Model>(results, params, "ConvWithActivationQuantizedTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildWeights(const ov::Shape& shape, float value) const {
        const auto weightsSize = 16 * shape.at(1) * 1 * 1;
        const auto weightsShape = ov::Shape{16, shape.at(1), 1, 1};
        return ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>(weightsSize, value))
                ->get_default_output();
    }

    ov::Output<ov::Node> buildConv(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& weights) const {
        return std::make_shared<ov::op::v1::Convolution>(input, weights, ov::Strides(std::vector<size_t>{1, 1}),
                                                         ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                                                         ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                                                         ov::Strides(std::vector<size_t>{1, 1}))
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvActivationQuantParams>& obj) {
        const auto& [activationType, quantParams] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << utils::testKind(__FILE__);

        if (quantParams.inputQuant.has_value()) {
            result << sep << "IQ=" << *quantParams.inputQuant;
        }
        if (quantParams.weightsQuant.has_value()) {
            result << sep << "WQ=" << *quantParams.weightsQuant;
        }
        if (quantParams.outputQuant.has_value()) {
            result << sep << "OQ=" << *quantParams.outputQuant;
        }
        result << sep << "ActivationType=" << activationType;

        return result.str();
    };
};

class ConvWithSwishQuantizedTest : public VpuOv2LayerTest, public testing::WithParamInterface<ConvSwishQuantParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }

    void SetUp() override {
        rel_threshold = 0.002f;
        const auto& [beta, quantParams] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({ov::Shape{1, 16, 16, 16}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto lastOutput = params.at(0)->get_default_output();
        if (quantParams.inputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.inputQuant)
                                 ->get_default_output();
        }

        auto weights = buildWeights(lastOutput.get_shape(), 0.5f);
        if (quantParams.weightsQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.weightsQuant)
                                 ->get_default_output();
        }

        lastOutput = buildConv(lastOutput, weights);

        lastOutput = utils::make_activation(lastOutput, ov::element::f16, utils::Swish, ov::Shape{}, {beta});

        if (quantParams.outputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.outputQuant)
                                 ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutput)};

        function = std::make_shared<ov::Model>(results, params, "ConvWithSwishQuantizedTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildWeights(const ov::Shape& shape, float value) const {
        const auto weightsSize = 16 * shape.at(1) * 1 * 1;
        const auto weightsShape = ov::Shape{16, shape.at(1), 1, 1};
        return ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>(weightsSize, value))
                ->get_default_output();
    }

    ov::Output<ov::Node> buildConv(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& weights) const {
        return std::make_shared<ov::op::v1::Convolution>(input, weights, ov::Strides(std::vector<size_t>{1, 1}),
                                                         ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                                                         ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                                                         ov::Strides(std::vector<size_t>{1, 1}))
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvSwishQuantParams>& obj) {
        const auto& [beta, quantParams] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << utils::testKind(__FILE__);

        if (quantParams.inputQuant.has_value()) {
            result << sep << "IQ=" << *quantParams.inputQuant;
        }
        if (quantParams.weightsQuant.has_value()) {
            result << sep << "WQ=" << *quantParams.weightsQuant;
        }
        if (quantParams.outputQuant.has_value()) {
            result << sep << "OQ=" << *quantParams.outputQuant;
        }
        result << sep << "Beta=" << beta;

        return result.str();
    };
};

const std::vector<utils::ActivationTypes> activations = {utils::Tanh, utils::Sigmoid, utils::Gelu, utils::Exp};

const std::vector<float> betas = {1.0f, 1.7f, 10.0f};

const std::vector<ConvQuantParams> quantParams = {
        ConvQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}), std::nullopt, std::nullopt},
        ConvQuantParams{std::nullopt, FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}), std::nullopt},
        ConvQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}),
                        FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}), std::nullopt},

        ConvQuantParams{std::nullopt, std::nullopt, FakeQuantizeParams({-1.f}, {1.f}, {-1.f}, {1.f})},
        ConvQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}),
                        FakeQuantizeParams({0.f}, {255.f}, {0.f}, {100.f}),
                        FakeQuantizeParams({-1.f}, {1.f}, {-1.f}, {1.f})},
        ConvQuantParams{std::nullopt,
                        FakeQuantizeParams({0.f}, {0.5f},
                                           {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, 0.0f},
                                           {0.53f, 0.47f, 0.51f, 0.52f, 0.48f, 0.49f, 0.51f, 0.48f, 0.52f, 0.51f, 0.53f,
                                            0.47f, 0.51f, 0.49f, 0.5f, 0.49f}),
                        std::nullopt},
};

INSTANTIATE_TEST_SUITE_P(smoke_convActivation, ConvWithActivationQuantizedTest,
                         ::testing::Combine(::testing::ValuesIn(activations), ::testing::ValuesIn(quantParams)),
                         ConvWithActivationQuantizedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_convActivation, ConvWithSwishQuantizedTest,
                         ::testing::Combine(::testing::ValuesIn(betas), ::testing::ValuesIn(quantParams)),
                         ConvWithSwishQuantizedTest::getTestCaseName);
}  // namespace ov::test
