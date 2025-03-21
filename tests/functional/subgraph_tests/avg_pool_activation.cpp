//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/node_builders/activation.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "private_properties.hpp"

namespace ov::test {

struct AvgPoolWithActivationTestParams {
    utils::ActivationTypes activationType;
    float threshold;
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
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = ov::float16::from_bits(static_cast<uint16_t>(i));
        }

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        auto& [activationType, threshold] = GetParam();
        abs_threshold = threshold;

        const auto inShape = ov::Shape{1, 16, 16, 256};
        init_input_shapes(static_shapes_to_test_representation({inShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto avgPool = std::make_shared<ov::op::v1::AvgPool>(params.at(0), ov::Strides{1, 1}, ov::Shape{0, 0},
                                                             ov::Shape{0, 0}, ov::Shape{1, 1}, true);
        const auto activation = utils::make_activation(avgPool->output(0), ov::element::f16, activationType);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(activation)};

        function = std::make_shared<ov::Model>(results, params, "AvgPoolWithActivationTest");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolWithActivationTestParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "ActivationType=" << obj.param.activationType;
        return result.str();
    }
};

const std::vector<AvgPoolWithActivationTestParams> activations = {{utils::Tanh, 0.0001f}, {utils::Sigmoid, 0.0001f}};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolWithActivation, AvgPoolWithActivationTest, ::testing::ValuesIn(activations),
                         AvgPoolWithActivationTest::getTestCaseName);

}  // namespace ov::test
