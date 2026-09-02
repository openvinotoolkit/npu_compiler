//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/logical.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/op/convert.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LogicalLayerTest);

class LogicalLayerTestCommon : public LogicalLayerTest, virtual public VpuOv2LayerTest {};
class LogicalLayerTestHW : public LogicalLayerTestCommon {};
class ShaveCodeGenLogicalLayerTestCommon : public LogicalLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

class DynamicLogicalLayerTest : public LogicalLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        const auto& [shapes, logicalOpType, secondInputType, modelType, _, additionalConfig] = this->GetParam();
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        init_input_shapes(shapes);

        ov::ParameterVector inputs{
                std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[0])};

        std::shared_ptr<ov::Node> secondInput;
        if (secondInputType == InputLayerType::PARAMETER) {
            secondInput = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[1]);
            inputs.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(secondInput));
        } else {
            ov::Tensor tensor =
                    ov::test::utils::create_and_fill_tensor(ov::element::boolean, targetStaticShapes.front()[1]);
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        std::shared_ptr<ov::Node> logicalNode;
        if (logicalOpType == ov::test::utils::LogicalTypes::LOGICAL_AND) {
            logicalNode = std::make_shared<ov::op::v1::LogicalAnd>(inputs[0], secondInput);
        } else if (logicalOpType == ov::test::utils::LogicalTypes::LOGICAL_OR) {
            logicalNode = std::make_shared<ov::op::v1::LogicalOr>(inputs[0], secondInput);
        } else {
            OPENVINO_THROW("Unsupported logical operation type");
        }

        auto convertedLogicalNode = std::make_shared<ov::op::v0::Convert>(logicalNode, modelType);
        function = std::make_shared<ov::Model>(convertedLogicalNode, inputs, "Logical");
    }
};

TEST_P(LogicalLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogicalLayerTestHW, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicLogicalLayerTest, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogicalLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicLogicalLayerTest, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenLogicalLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(LogicalLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicLogicalLayerTest, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ShaveCodeGenLogicalLayerTestCommon, NPU5010) {
    setReferenceSoftwareMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(LogicalLayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

TEST_P(DynamicLogicalLayerTest, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<std::vector<ov::Shape>> combineShapes(
        const std::map<ov::Shape, std::vector<ov::Shape>>& input_shapes_static) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& input_shape : input_shapes_static) {
        for (auto& item : input_shape.second) {
            result.push_back({input_shape.first, item});
        }

        if (input_shape.second.empty()) {
            result.push_back({input_shape.first, {}});
        }
    }
    return result;
}

std::vector<InputLayerType> secondInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::vector<ov::element::Type> modelTypes = {
        ov::element::boolean,
};

std::map<std::string, std::string> additional_config = {};

std::set<LogicalTypes> supportedTypes = {
        LogicalTypes::LOGICAL_OR,
        LogicalTypes::LOGICAL_XOR,
        LogicalTypes::LOGICAL_AND,
};

std::vector<ov::test::utils::LogicalTypes> logicalOpTypesDynamic = {
        ov::test::utils::LogicalTypes::LOGICAL_AND,
        ov::test::utils::LogicalTypes::LOGICAL_OR,
};

std::map<ov::Shape, std::vector<ov::Shape>> inShapes = {
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},   {{1, 16, 32}, {{1, 16, 32}}}, {{1, 28, 300, 1}, {{1, 1, 300, 28}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}}}, {{2, 200}, {{2, 200}}},

};

std::map<ov::Shape, std::vector<ov::Shape>> precommit_inShapes = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> inShapesNot = {
        {{1, 2, 4}, {}},
};

std::map<ov::Shape, std::vector<ov::Shape>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic = {
        {{{1, 11, ov::Dimension(1, 10)}, {{1, 11, 1}, {1, 11, 10}, {1, 11, 5}}},
         {{1, 11, ov::Dimension(1, 10)}, {{1, 11, 1}, {1, 11, 10}, {1, 11, 5}}}},
};

const auto logical_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(inShapes))),
        ::testing::ValuesIn(supportedTypes), ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config));

const auto precommit_logical_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(precommit_inShapes))),
        ::testing::ValuesIn(supportedTypes), ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config));

const auto precommit_logical_params_not =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(inShapesNot))),
                           ::testing::Values(LogicalTypes::LOGICAL_NOT), ::testing::Values(InputLayerType::CONSTANT),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto tiling_logical_params =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(tiling_inShapes))),
                           ::testing::Values(LogicalTypes::LOGICAL_OR), ::testing::ValuesIn(secondInputTypes),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto logical_params_dynamic =
        ::testing::Combine(::testing::ValuesIn(in_shapes_dynamic), ::testing::ValuesIn(logicalOpTypesDynamic),
                           ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(modelTypes),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config));

// [Tracking number E#109588]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_logical, LogicalLayerTestCommon, logical_params,
                         LogicalLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_logical, LogicalLayerTestCommon, precommit_logical_params,
                         LogicalLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_logical_not, LogicalLayerTestCommon, precommit_logical_params_not,
                         LogicalLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_tiling, LogicalLayerTestHW, tiling_logical_params,
                         LogicalLayerTest::getTestCaseName);

// [Tracking number E#109588]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_logical, ShaveCodeGenLogicalLayerTestCommon, logical_params,
                         ShaveCodeGenLogicalLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_logical, ShaveCodeGenLogicalLayerTestCommon,
                         precommit_logical_params, ShaveCodeGenLogicalLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_logical_not, ShaveCodeGenLogicalLayerTestCommon,
                         precommit_logical_params_not, ShaveCodeGenLogicalLayerTestCommon::getTestCaseName);
// [Tracking number E#109588 and E#152367]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_tiling, ShaveCodeGenLogicalLayerTestCommon, tiling_logical_params,
                         ShaveCodeGenLogicalLayerTestCommon::getTestCaseName);

// [Tracking number E#185715  and E#185715]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_LogicalDynamic, DynamicLogicalLayerTest, logical_params_dynamic,
                         DynamicLogicalLayerTest::getTestCaseName);

}  // namespace
