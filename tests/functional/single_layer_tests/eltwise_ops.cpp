//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_op/eltwise.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov;
using namespace element;

namespace ov::test {

template <typename EltwiseOpT>
class Eltwise2InputLayerTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == 1, "Expected 1 inputShapes");
        OPENVINO_ASSERT(funcInputs.size() == 2, "Expected 2 inputs");
        const auto& inputStaticShape = inputShapes[0];
        auto inputTensor1 = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputTensor2 = ov::Tensor{ov::element::f16, inputStaticShape};
        using inputValueType = ov::element_type_traits<ov::element::f16>::value_type;
        inputValueType* inputData1 = inputTensor1.data<inputValueType>();
        inputValueType* inputData2 = inputTensor2.data<inputValueType>();
        const auto totalSize = ov::shape_size(inputStaticShape);

        std::iota(inputData1, inputData1 + totalSize, 0);
        std::iota(inputData2, inputData2 + totalSize, 1);

        inputs = {{funcInputs[0].get_node_shared_ptr(), inputTensor1},
                  {funcInputs[1].get_node_shared_ptr(), inputTensor2}};
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

    void SetUp() override {
        const std::vector<ov::Shape> inferenceShapes = {ov::Shape{2, 3}, ov::Shape{2, 3}};
        const ov::test::InputShape dataShape = {ov::Shape{2, 3}, inferenceShapes};
        init_input_shapes({dataShape});

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3});
        auto mul = std::make_shared<EltwiseOpT>(param1, param2);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(mul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                               "EltwiseMultiply");
    }
};

typedef Eltwise2InputLayerTest<ov::op::v1::Multiply> EltwiseMultiplyLayerTest;

TEST_F(EltwiseMultiplyLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(EltwiseMultiplyLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_F(EltwiseMultiplyLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
typedef Eltwise2InputLayerTest<ov::op::v1::Add> EltwiseAddLayerTest;
class EltwiseAddLayerTest_HostCompile : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

TEST_F(EltwiseAddLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(EltwiseAddLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseAddLayerTest_HostCompile, NPU4000_HC) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "Host Pipeline does not support inference yet: C#164943";
    });

    setHostCompileMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

typedef Eltwise2InputLayerTest<ov::op::v1::Subtract> EltwiseSubtractLayerTest;

TEST_F(EltwiseAddLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(EltwiseSubtractLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(EltwiseAddLayerTest_HostCompile, NPU5010_HC) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "Host Pipeline does not support inference yet: C#164943";
    });

    setHostCompileMode();
    setMLIRCompilerType();
    run(Platform::NPU5010);
}

const std::vector<std::vector<ov::test::InputShape>> dynamicShapes = {
        {generateTestShape(1, 16, 1280_Dyn, 1280)},
        {generateTestShape(1, 16, 1280_Dyn, 1280_Dyn)},
        {generateTestShape(1, 3, 1280_Dyn, 1280)},
        {generateTestShape(1, 3, 1280_Dyn, 1280_Dyn)},
};

const auto dynamicAddParams =
        ::testing::Combine(::testing::ValuesIn(dynamicShapes), ::testing::ValuesIn({utils::EltwiseTypes::ADD}),
                           ::testing::ValuesIn({utils::InputLayerType::PARAMETER}),
                           ::testing::ValuesIn({utils::OpType::VECTOR}), ::testing::ValuesIn({ElementType::f16}),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapes, EltwiseAddLayerTest_HostCompile, dynamicAddParams,
                         EltwiseAddLayerTest_HostCompile::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> staticShapes = {{generateTestShape(1, 16, 1280, 1280)},
                                                                     {generateTestShape(1, 3, 1280, 1280)}};
const auto staticAddParams =
        ::testing::Combine(::testing::ValuesIn(staticShapes), ::testing::ValuesIn({utils::EltwiseTypes::ADD}),
                           ::testing::ValuesIn({utils::InputLayerType::PARAMETER}),
                           ::testing::ValuesIn({utils::OpType::VECTOR}), ::testing::ValuesIn({ElementType::f16}),
                           ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(smoke, EltwiseAddLayerTest_HostCompile, staticAddParams,
                         EltwiseAddLayerTest_HostCompile::getTestCaseName);

}  // namespace ov::test
