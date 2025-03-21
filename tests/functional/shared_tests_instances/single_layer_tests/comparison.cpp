//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/comparison.hpp"
#include <vector>
#include "common/functions.h"
#include "common_test_utils/node_builders/comparison.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

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

class ComparisonLayerTest_Tiling : public ComparisonLayerTestCommon {};

class ComparisonLayerTest_FP32 : public ComparisonLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(ComparisonLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTest_FP32, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTest_Tiling, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ComparisonLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ComparisonLayerTest_FP32, NPU4000_SW) {
    setReferenceSoftwareMode();
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

std::map<ov::Shape, std::vector<ov::Shape>> precommit_inShapes = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

std::vector<ov::element::Type> precision = {
        ov::element::f16,
        ov::element::i32,
};

auto inputShapesComparisonParams = input_shape_converter(combineParams(inputShapes));
const auto comparison_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesComparisonParams)),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::ValuesIn(precision), ::testing::Values(DEVICE_NPU), ::testing::Values(additionalConfig));

auto inputShapesPrecommit = input_shape_converter(combineParams(precommit_inShapes));
const auto precommit_comparison_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesPrecommit)),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::ValuesIn(precision), ::testing::Values(DEVICE_NPU), ::testing::Values(additionalConfig));

auto inputShapesTiling = input_shape_converter(combineParams(tiling_inShapes));
const auto tiling_comparison_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesTiling)),
        ::testing::Values(ComparisonTypes::EQUAL), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(ov::element::f16), ::testing::Values(DEVICE_NPU), ::testing::Values(additionalConfig));

const auto precommit_comparison_params_FP32 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesPrecommit)),
        ::testing::Values(ComparisonTypes::LESS_EQUAL), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(ov::element::f32), ::testing::Values(DEVICE_NPU), ::testing::Values(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_Comparison, ComparisonLayerTestCommon, comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Comparison, ComparisonLayerTestCommon, precommit_comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Comparison, ComparisonLayerTestCommon, tiling_comparison_params,
                         ComparisonLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Comparison_FP32, ComparisonLayerTestCommon, precommit_comparison_params_FP32,
                         ComparisonLayerTestCommon::getTestCaseName);

}  // namespace
