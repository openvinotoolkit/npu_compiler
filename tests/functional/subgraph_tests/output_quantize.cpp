//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <common_test_utils/node_builders/fake_quantize.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

#include <openvino/core/type/element_type.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>

namespace ov::test {

class OutputQuantizeTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<bool> {
    void configure_model() override {
        configuration[ov::intel_npu::qdq_optimization.name()] = "YES";
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& inputStaticShape = targetInputStaticShapes[0];

        const float startFrom = 6.0f;
        const uint32_t range = 1;
        const int32_t resolution = 1000;

        ov::Tensor inputTensor = ov::test::utils::create_and_fill_tensor(
                funcInputs[0].get_element_type(), inputStaticShape, range, startFrom, resolution);

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }
    void SetUp() override {
        const auto withReshape = GetParam();
        const ov::Shape inputShape{1, 1, 32, 32};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0])};

        const size_t dataLevels = 65536;
        const std::vector<float> inDataLow = {-7.0f};
        const std::vector<float> inDataHigh = {7.0f};
        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {65535.0f};

        const auto dataFq = ov::test::utils::make_fake_quantize(params[0]->output(0), ov::element::f32, dataLevels, {},
                                                                inDataLow, inDataHigh, outDataLow, outDataHigh);
        auto convertToInt = std::make_shared<ov::op::v0::Convert>(dataFq, ov::element::u16)->get_default_output();
        if (withReshape) {
            convertToInt = std::make_shared<ov::op::v1::Reshape>(
                    convertToInt, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 32, 1, 32}), false);
        }
        const auto convertToFloat = std::make_shared<ov::op::v0::Convert>(convertToInt, ov::element::f32);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(
                convertToFloat, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {32768.0f}));
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(
                subtract, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.000213623f}));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convertToInt),
                                       std::make_shared<ov::op::v0::Result>(multiply)};
        function = std::make_shared<ov::Model>(results, params, "OutputQuantizeTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "WithReshape=" << obj.param << sep;
        return result.str();
    };
};

TEST_P(OutputQuantizeTestCommon, NPU4000_HW) {
    rel_threshold = 0.0003f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(OutputQuantizeTestCommon, NPU5010_HW) {
    rel_threshold = 0.0003f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(OutputQuantizeTestCommon, NPU5020_HW) {
    rel_threshold = 0.0003f;
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_P(OutputQuantizeTestCommon, NPU6010_HW) {
    rel_threshold = 0.0003f;
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

const std::vector<bool> withReshape = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_outputQuantize, OutputQuantizeTestCommon, ::testing::ValuesIn(withReshape),
                         OutputQuantizeTestCommon::getTestCaseName);

}  // namespace ov::test
