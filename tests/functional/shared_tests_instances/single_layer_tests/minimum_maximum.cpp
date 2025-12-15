//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/opsets/opset1.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class MaxMinLayerTestCommon : public MaxMinLayerTest, virtual public VpuOv2LayerTest {};
class ShaveCodeGenMaxMinLayerTestCommon : public MaxMinLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};
class MaxMinLayerTestDynamic : public MaxMinLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& inputStaticShape = targetInputStaticShapes[0];

        const int32_t startFrom = 0;
        const int32_t range = 10;

        ov::Tensor inputTensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                         inputStaticShape, range, startFrom);

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }
    void SetUp() override {
        const auto& [dataShape, opType, type, inputType, _] = this->GetParam();
        init_input_shapes({dataShape});

        const auto& partialShape = dataShape[0].first;
        std::vector<size_t> constantShape;

        for (size_t i = 0; i < partialShape.rank().get_length(); ++i) {
            if (partialShape[i].is_static()) {
                constantShape.push_back(partialShape[i].get_length());
            } else {
                constantShape.push_back(1);
            }
        }
        const auto cst = std::make_shared<ov::opset1::Constant>(type, Shape(constantShape), std::vector<float>{0.5f});
        const auto param = std::make_shared<ov::opset1::Parameter>(type, inputDynamicShapes.at(0));

        std::shared_ptr<ov::Node> minMaxOp;
        if (opType == MinMaxOpType::MINIMUM) {
            minMaxOp = std::make_shared<ov::opset1::Minimum>(param, cst);
        } else {
            minMaxOp = std::make_shared<ov::opset1::Maximum>(param, cst);
        }

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(minMaxOp->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicMaxMin");
    }
};

TEST_P(MaxMinLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(MaxMinLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenMaxMinLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

TEST_P(MaxMinLayerTestDynamic, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(MaxMinLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(MaxMinLayerTestDynamic, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ShaveCodeGenMaxMinLayerTestCommon, NPU5010) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> modelTypes = {
        ov::element::f16,
};

const std::vector<MinMaxOpType> opType = {
        MinMaxOpType::MINIMUM,
        MinMaxOpType::MAXIMUM,
};

const std::vector<InputLayerType> inputType = {InputLayerType::CONSTANT};

const std::vector<std::vector<ov::Shape>> inShapes3D = {{{1, 2, 4}, {1}}};
const std::vector<std::vector<ov::Shape>> inShapes4D = {{{1, 64, 32, 32}, {1, 64, 32, 32}}, {{1, 1, 1, 3}, {1}}};
const std::vector<std::vector<ov::Shape>> inShapesGeneric = {{{1, 1, 16, 32}, {1, 1, 16, 32}}, {{32}, {1}}};

const auto params0 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes4D)),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(test_utils::TARGET_DEVICE));

const auto params1 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes3D)),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(test_utils::TARGET_DEVICE));

const auto params2 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapesGeneric)),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(test_utils::TARGET_DEVICE));

const auto params3 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(
                                              std::vector<std::vector<ov::Shape>>({{{1, 1, 1, 3}, {1}}}))),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test0, MaxMinLayerTestCommon, params0, MaxMinLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test1, MaxMinLayerTestCommon, params1, MaxMinLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test2, MaxMinLayerTestCommon, params2, MaxMinLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test3, MaxMinLayerTestCommon, params3, MaxMinLayerTestCommon::getTestCaseName);

const std::vector<ov::test::InputShape> params_dynamic = {
        generateTestShape(1, 3, 128_Dyn, 128_Dyn),
};

const auto paramsDynamic = testing::Combine(::testing::Values(params_dynamic), ::testing::ValuesIn(opType),
                                            ::testing::ValuesIn(modelTypes), ::testing::ValuesIn(inputType),
                                            ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test_dynamic, MaxMinLayerTestDynamic, paramsDynamic,
                         MaxMinLayerTestDynamic::getTestCaseName);

// Avoid shapes that would require tiling for ShaveCodeGen for now.
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test4, ShaveCodeGenMaxMinLayerTestCommon, params2,
                         ShaveCodeGenMaxMinLayerTestCommon::getTestCaseName);

}  // namespace
