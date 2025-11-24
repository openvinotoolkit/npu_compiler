//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/node_builders/activation.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "common/quantization_utils.hpp"

#include "openvino/op/avg_pool.hpp"

namespace ov::test {

struct AvgPoolQuantParams {
    std::optional<FakeQuantizeParams> inputQuant;
    std::optional<FakeQuantizeParams> outputQuant;
};

enum class ErrorType : uint8_t { ABSOLUTE, RELATIVE };
struct Activation {
    utils::ActivationTypes activationType;
    ErrorType errorType;
    float threshold{0};
};

using AvgPoolWithActivationQuantTestParams = std::tuple<Activation, AvgPoolQuantParams>;
using AvgPoolWithSwishQuantTestParams = std::tuple<float, AvgPoolQuantParams>;

//      [input]
//         |
//    FakeQuantize (optional)
//         |
//      AvgPool
//         |
//     Activation
//         |
//    FakeQuantize (optional)
//         |
//      [output]

class AvgPoolWithActivationQuantizedTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<AvgPoolWithActivationQuantTestParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }

    void SetUp() override {
        const auto& [activation, quantParams] = GetParam();
        const auto& activationType = activation.activationType;
        const auto& errorType = activation.errorType;
        const auto& threshold = activation.threshold;

        init_input_shapes(static_shapes_to_test_representation({ov::Shape{1, 16, 16, 16}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto lastOutput = params.at(0)->get_default_output();
        if (quantParams.inputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.inputQuant)
                                 ->get_default_output();
        }

        lastOutput = buildAvgPool(lastOutput);

        if (threshold != 0) {
            if (errorType == ErrorType::ABSOLUTE) {
                abs_threshold = threshold;
                rel_threshold = disable_threshold;
            } else {
                rel_threshold = threshold;
                abs_threshold = disable_threshold;
            }
        }

        lastOutput = activationType == utils::Swish
                             ? utils::make_activation(lastOutput, ov::element::f16, activationType, ov::Shape{}, {1.f})
                             : utils::make_activation(lastOutput, ov::element::f16, activationType);

        if (quantParams.outputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.outputQuant)
                                 ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutput)};

        function = std::make_shared<ov::Model>(results, params, "AvgPoolWithActivationQuantizedTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildAvgPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::AvgPool>(input, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                                                     ov::Shape{2, 2}, true)
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<AvgPoolWithActivationQuantTestParams>& obj) {
        const auto& [activation, quantParams] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << utils::testKind(__FILE__);

        if (quantParams.inputQuant.has_value()) {
            result << sep << "IQ=" << *quantParams.inputQuant;
        }
        if (quantParams.outputQuant.has_value()) {
            result << sep << "OQ=" << *quantParams.outputQuant;
        }
        result << sep << "ActivationType=" << activation.activationType;

        return result.str();
    };
};

class AvgPoolWithSwishQuantizedTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<AvgPoolWithSwishQuantTestParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }

    void SetUp() override {
        const auto& [beta, quantParams] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({ov::Shape{1, 16, 16, 16}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto lastOutput = params.at(0)->get_default_output();
        if (quantParams.inputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.inputQuant)
                                 ->get_default_output();
        }

        lastOutput = buildAvgPool(lastOutput);

        lastOutput = utils::make_activation(lastOutput, ov::element::f16, utils::Swish, ov::Shape{}, {beta});

        if (quantParams.outputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.outputQuant)
                                 ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutput)};

        function = std::make_shared<ov::Model>(results, params, "AvgPoolWithSwishQuantizedTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildAvgPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::AvgPool>(input, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                                                     ov::Shape{2, 2}, true)
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<AvgPoolWithSwishQuantTestParams>& obj) {
        const auto& [beta, quantParams] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << utils::testKind(__FILE__);

        if (quantParams.inputQuant.has_value()) {
            result << sep << "IQ=" << *quantParams.inputQuant;
        }
        if (quantParams.outputQuant.has_value()) {
            result << sep << "OQ=" << *quantParams.outputQuant;
        }
        result << sep << "Beta=" << beta;

        return result.str();
    };
};

const std::vector<Activation> activations = {{utils::Tanh, ErrorType::ABSOLUTE},
                                             {utils::Sigmoid, ErrorType::ABSOLUTE},
                                             {utils::Gelu, ErrorType::ABSOLUTE},
                                             {utils::Exp, ErrorType::RELATIVE, 0.03f}};

const std::vector<float> betas = {1.0f, 1.7f, 10.0f};

const std::vector<AvgPoolQuantParams> quantParams = {
        AvgPoolQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}), std::nullopt},
        AvgPoolQuantParams{std::nullopt, FakeQuantizeParams({-1.f}, {1.f}, {-1.f}, {1.f})},
        AvgPoolQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}),
                           FakeQuantizeParams({-1.f}, {1.f}, {-1.f}, {1.f})},
        AvgPoolQuantParams{FakeQuantizeParams({0.f}, {0.5f},
                                              {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f},
                                              {0.53f, 0.47f, 0.51f, 0.52f, 0.48f, 0.49f, 0.51f, 0.48f, 0.52f, 0.51f,
                                               0.53f, 0.47f, 0.51f, 0.49f, 0.5f, 0.49f}),
                           std::nullopt},
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolActivation, AvgPoolWithActivationQuantizedTest,
                         ::testing::Combine(::testing::ValuesIn(activations), ::testing::ValuesIn(quantParams)),
                         AvgPoolWithActivationQuantizedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolSwish, AvgPoolWithSwishQuantizedTest,
                         ::testing::Combine(::testing::ValuesIn(betas), ::testing::ValuesIn(quantParams)),
                         AvgPoolWithSwishQuantizedTest::getTestCaseName);
}  // namespace ov::test
