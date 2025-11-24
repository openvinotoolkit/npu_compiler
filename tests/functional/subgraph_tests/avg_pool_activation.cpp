//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <common_test_utils/node_builders/activation.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/avg_pool.hpp"

namespace ov::test {

enum class ErrorType : uint8_t { ABSOLUTE, RELATIVE };

struct AvgPoolWithActivationTestParams {
    utils::ActivationTypes activationType;
    ErrorType errorType;
    float threshold{0.0f};
    float swishBeta{0.0f};
};

class AvgPoolWithActivationTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<AvgPoolWithActivationTestParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-sprlut=true";
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& inputStaticShape = targetInputStaticShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();

        // Testing the whole fp16 range
        constexpr uint16_t fp16Exponent = 0x7C00;
        for (size_t i = 0; i < totalSize; i++) {
            auto ui = static_cast<uint16_t>(i);
            if ((ui & fp16Exponent) == fp16Exponent) {  // Skip infinity/nans
                inputData[i] = 0.f;
            } else {
                inputData[i] = ov::float16::from_bits(ui);
            }
        }

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        auto& [activationType, errorType, threshold, swishBeta] = GetParam();

        const auto inShape = ov::Shape{1, 16, 16, 256};
        init_input_shapes(static_shapes_to_test_representation({inShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto avgPool = std::make_shared<ov::op::v1::AvgPool>(params.at(0), ov::Strides{1, 1}, ov::Shape{0, 0},
                                                             ov::Shape{0, 0}, ov::Shape{1, 1}, true);

        if (errorType == ErrorType::ABSOLUTE) {
            abs_threshold = threshold;
            rel_threshold = disable_threshold;
        } else {
            rel_threshold = threshold;
            abs_threshold = disable_threshold;
        }

        const auto activation = activationType == utils::Swish
                                        ? utils::make_activation(avgPool->output(0), ov::element::f16, activationType,
                                                                 ov::Shape{}, {swishBeta})
                                        : utils::make_activation(avgPool->output(0), ov::element::f16, activationType);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(activation)};

        function = std::make_shared<ov::Model>(results, params, "AvgPoolWithActivationTest");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolWithActivationTestParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "ActivationType=" << obj.param.activationType;
        result << sep << "SwishBeta=" << obj.param.swishBeta;
        return result.str();
    }
};

const std::vector<AvgPoolWithActivationTestParams> activations = {

        {utils::Tanh, ErrorType::ABSOLUTE, 0.0001f},
        {utils::Sigmoid, ErrorType::ABSOLUTE, 0.0001f},
        {utils::Gelu, ErrorType::ABSOLUTE, 0.00013f},
        {utils::Swish, ErrorType::ABSOLUTE, 0.00014f, 1.0f},
        {utils::Swish, ErrorType::ABSOLUTE, 0.00014f, 1.7f},
        {utils::Swish, ErrorType::ABSOLUTE, 0.00014f, 10.0f},
        {utils::Exp, ErrorType::RELATIVE, 0.003f}

};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolWithActivation, AvgPoolWithActivationTest, ::testing::ValuesIn(activations),
                         AvgPoolWithActivationTest::getTestCaseName);

}  // namespace ov::test
