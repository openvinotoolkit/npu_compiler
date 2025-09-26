//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/comparison.hpp"
#include "common_test_utils/node_builders/comparison.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ComparisonLayerTestCommon : public ComparisonLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        ComparisonTypes comparisonOpType;
        InputLayerType secondInputType;
        ov::element::Type modelType;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, comparisonOpType, secondInputType, modelType, std::ignore, additionalConfig) =
                this->GetParam();

        init_input_shapes(inputShapes);

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(modelType, inputShapes[0].second[0])};

        std::shared_ptr<ov::Node> secondInput;
        if (secondInputType == InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape(inputShapes[1].second[0]));
            secondInput = param;
            inputs.push_back(param);
        } else {
            auto tensor = create_and_fill_tensor(modelType, ov::Shape(inputShapes[1].second[0]));
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto comparisonNode = make_comparison(inputs[0], secondInput, comparisonOpType);
        auto convertedComparisonNode = std::make_shared<ov::op::v0::Convert>(comparisonNode, modelType);
        function = std::make_shared<ov::Model>(convertedComparisonNode, inputs, "Comparison");
    }
};

class ComparisonLayerTestDynamic : public ComparisonLayerTest, virtual public VpuOv2LayerTest {
    // A copy of ComparisonLayerTest, because compiler does not support boolean type (it is represented as u8).
    // We cannot use ComparisonLayerTestCommon since it handles inputShapes and inputDynamicShapes differently,
    // which prevents its use with dynamic shapes. The SetUp override can be removed once the compiler supports the
    // boolean type as-is (#109588).
    void SetUp() override {
        std::vector<InputShape> shapes;
        InputLayerType secondInputType;
        std::map<std::string, std::string> additionalConfig;
        ov::element::Type modelType;
        ov::test::utils::ComparisonTypes comparisonOpType;
        std::tie(shapes, comparisonOpType, secondInputType, modelType, targetDevice, additionalConfig) =
                this->GetParam();
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        init_input_shapes(shapes);

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes[0])};

        std::shared_ptr<ov::Node> secondInput;
        if (secondInputType == InputLayerType::PARAMETER) {
            secondInput = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes[1]);
            inputs.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(secondInput));
        } else {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(modelType, targetStaticShapes.front()[1]);
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto comparisonNode = ov::test::utils::make_comparison(inputs[0], secondInput, comparisonOpType);

        // Workaround added - Convert node to avoid working with boolean as output.
        auto convertedComparisonNode = std::make_shared<ov::op::v0::Convert>(comparisonNode, modelType);
        function = std::make_shared<ov::Model>(convertedComparisonNode, inputs, "Comparison");
    }
};

class ComparisonLayerTest_Tiling : public ComparisonLayerTestCommon {};
class ShaveCodeGenComparisonLayerTestCommon : public ComparisonLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

TEST_P(ComparisonLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTest_Tiling, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTestDynamic, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ComparisonLayerTestDynamic, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenComparisonLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
std::vector<ComparisonTypes> comparisonOpTypes_MLIR = {
        ComparisonTypes::EQUAL,     ComparisonTypes::LESS,    ComparisonTypes::LESS_EQUAL,
        ComparisonTypes::NOT_EQUAL, ComparisonTypes::GREATER, ComparisonTypes::GREATER_EQUAL,
};

std::vector<ComparisonTypes> comparisonOpTypesDynamic_MLIR = {
        ComparisonTypes::LESS,
        ComparisonTypes::LESS_EQUAL,
};

std::vector<InputLayerType> secondInputTypes = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

std::map<std::string, std::string> additionalConfig = {};

auto input_shape_converter = [](const std::vector<std::pair<ov::Shape, ov::Shape>>& shapes) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& shape : shapes) {
        result.push_back({shape.first, shape.second});
    }
    return result;
};

std::map<ov::Shape, std::vector<ov::Shape>> inputShapes = {
        {{5}, {{1}}},
        {{10, 1}, {{1, 50}}},
        {{1, 16, 32}, {{1, 16, 32}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> precommit_inShapes = {{{1, 16, 32}, {{1, 1, 32}}},
                                                                  {{1, 1, 5, 5}, {{1, 1, 1, 1}}}};

std::map<ov::Shape, std::vector<ov::Shape>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

// Only 4D dynamic shapes are supported (3D and 2D require ConvertShapeTo4D support for this layer #163161)
// Test 3 cases - broadcast with max dim, no broadcast and broadcast where input is not a max dim.
std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic_4D = {
        {{{1, 1, ov::Dimension(1, 10), 200}, {{1, 1, 10, 200}, {1, 1, 10, 200}, {1, 1, 5, 200}}},
         {{1, 1, ov::Dimension(1, 10), 200}, {{1, 1, 1, 200}, {1, 1, 10, 200}, {1, 1, 1, 200}}}},
};

std::vector<ov::element::Type> precision = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

auto inputShapesComparisonParams = input_shape_converter(combineParams(inputShapes));
const auto comparison_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesComparisonParams)),
                           ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
                           ::testing::ValuesIn(precision), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

auto inputShapesPrecommit = input_shape_converter(combineParams(precommit_inShapes));
const auto precommit_comparison_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesPrecommit)),
                           ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
                           ::testing::ValuesIn(precision), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

auto inputShapesTiling = input_shape_converter(combineParams(tiling_inShapes));
const auto tiling_comparison_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesTiling)),
                           ::testing::Values(ComparisonTypes::EQUAL), ::testing::ValuesIn(secondInputTypes),
                           ::testing::Values(ov::element::f16), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additionalConfig));

const auto comparison_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(in_shapes_dynamic_4D), ::testing::ValuesIn(comparisonOpTypesDynamic_MLIR),
        ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(precision),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_Comparison, ComparisonLayerTestCommon, comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Comparison, ComparisonLayerTestCommon, precommit_comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Comparison, ComparisonLayerTestCommon, tiling_comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

// ShaveCodeGen tests
INSTANTIATE_TEST_SUITE_P(smoke_Comparison, ShaveCodeGenComparisonLayerTestCommon, comparison_params,
                         ShaveCodeGenComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Comparison, ShaveCodeGenComparisonLayerTestCommon, precommit_comparison_params,
                         ShaveCodeGenComparisonLayerTestCommon::getTestCaseName);

// E-152367: ShaveCodeGen Tiling support
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_tiling_Comparison, ShaveCodeGenComparisonLayerTestCommon,
                         tiling_comparison_params, ShaveCodeGenComparisonLayerTestCommon::getTestCaseName);

//  Dynamic shapes cases
INSTANTIATE_TEST_SUITE_P(smoke_ComparisonDynamic, ComparisonLayerTestDynamic, comparison_params_dynamic,
                         ComparisonLayerTestDynamic::getTestCaseName);

}  // namespace
