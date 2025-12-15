//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/node_builders/activation.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "common/quantization_utils.hpp"

#include "openvino/op/max_pool.hpp"

namespace ov::test {

struct MaxPoolQuantParams {
    std::optional<FakeQuantizeParams> inputQuant;
    std::optional<FakeQuantizeParams> outputQuant;
};

using MaxPoolWithActivationQuantTestParams = std::tuple<utils::ActivationTypes, MaxPoolQuantParams>;
using MaxPoolWithSwishQuantTestParams = std::tuple<float, MaxPoolQuantParams>;

//      [input]
//         |
//    FakeQuantize (optional)
//         |
//      MaxPool
//         |
//     Activation
//         |
//    FakeQuantize (optional)
//         |
//      [output]

class MaxPoolWithActivationQuantizedTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MaxPoolWithActivationQuantTestParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }

    void SetUp() override {
        const auto& [activationType, quantParams] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({ov::Shape{1, 16, 16, 16}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto lastOutput = params.at(0)->get_default_output();
        if (quantParams.inputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.inputQuant)
                                 ->get_default_output();
        }

        lastOutput = buildMaxPool(lastOutput);

        lastOutput = activationType == utils::Swish
                             ? utils::make_activation(lastOutput, ov::element::f16, activationType, ov::Shape{}, {1.f})
                             : utils::make_activation(lastOutput, ov::element::f16, activationType);

        if (quantParams.outputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.outputQuant)
                                 ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutput)};

        function = std::make_shared<ov::Model>(results, params, "MaxPoolWithActivationQuantizedTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildMaxPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::MaxPool>(input, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                                                     ov::Shape{2, 2})
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxPoolWithActivationQuantTestParams>& obj) {
        const auto& [activationType, quantParams] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << utils::testKind(__FILE__);

        if (quantParams.inputQuant.has_value()) {
            result << sep << "IQ=" << *quantParams.inputQuant;
        }
        if (quantParams.outputQuant.has_value()) {
            result << sep << "OQ=" << *quantParams.outputQuant;
        }
        result << sep << "ActivationType=" << activationType;

        return result.str();
    };
};

class MaxPoolWithSwishQuantizedTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MaxPoolWithSwishQuantTestParams> {
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

        lastOutput = buildMaxPool(lastOutput);

        lastOutput = utils::make_activation(lastOutput, ov::element::f16, utils::Swish, ov::Shape{}, {beta});

        if (quantParams.outputQuant.has_value()) {
            lastOutput = utils::makeFakeQuantize(lastOutput, ov::element::f16, 256, *quantParams.outputQuant)
                                 ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutput)};

        function = std::make_shared<ov::Model>(results, params, "MaxPoolWithSwishQuantizedTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        preProc.output().tensor().set_layout(ov::Layout("NHWC"));
        preProc.output().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildMaxPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::MaxPool>(input, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
                                                     ov::Shape{2, 2})
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxPoolWithSwishQuantTestParams>& obj) {
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

TEST_P(MaxPoolWithActivationQuantizedTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(MaxPoolWithSwishQuantizedTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<utils::ActivationTypes> activations = {utils::Tanh, utils::Sigmoid, utils::Gelu, utils::Exp,
                                                         utils::HSwish};

const std::vector<float> betas = {1.0f, 1.7f, 10.0f};

const std::vector<MaxPoolQuantParams> quantParams = {
        MaxPoolQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}), std::nullopt},
        MaxPoolQuantParams{std::nullopt, FakeQuantizeParams({-1.f}, {1.f}, {-1.f}, {1.f})},
        MaxPoolQuantParams{FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f}),
                           FakeQuantizeParams({-1.f}, {1.f}, {-1.f}, {1.f})},
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolActivation, MaxPoolWithActivationQuantizedTest,
                         ::testing::Combine(::testing::ValuesIn(activations), ::testing::ValuesIn(quantParams)),
                         MaxPoolWithActivationQuantizedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolSwish, MaxPoolWithSwishQuantizedTest,
                         ::testing::Combine(::testing::ValuesIn(betas), ::testing::ValuesIn(quantParams)),
                         MaxPoolWithSwishQuantizedTest::getTestCaseName);
}  // namespace ov::test
