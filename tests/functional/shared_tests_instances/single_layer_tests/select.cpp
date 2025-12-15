//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "single_op_tests/select.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/select.hpp"

namespace ov {
namespace test {

class SelectLayerTestCommon : public SelectLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<InputShape> inputShapes(3);
        ov::element::Type inputType;
        ov::op::AutoBroadcastSpec broadcast;
        std::tie(inputShapes, inputType, broadcast, targetDevice) = this->GetParam();
        init_input_shapes(inputShapes);

        ov::ParameterVector inputs;
        for (auto&& shape : inputDynamicShapes) {
            inputs.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, shape));
        }
        ov::OutputVector selectInputs;
        auto boolInput = std::make_shared<ov::op::v0::Convert>(inputs[0], ov::element::boolean);
        selectInputs.push_back(boolInput);
        for (size_t i = 1; i < inputDynamicShapes.size(); i++) {
            selectInputs.push_back(inputs[i]);
        }

        auto select =
                std::make_shared<ov::op::v1::Select>(selectInputs[0], selectInputs[1], selectInputs[2], broadcast);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(select)};
        function = std::make_shared<ov::Model>(results, inputs, "select");
    }
};

class ShaveCodeGenSelectLayerTestCommon : public SelectLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

TEST_P(SelectLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(SelectLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenSelectLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

TEST_P(SelectLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(ShaveCodeGenSelectLayerTestCommon, NPU5010) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::SelectLayerTestCommon;
using ov::test::ShaveCodeGenSelectLayerTestCommon;

namespace {
const std::vector<ov::element::Type> inputTypes = {ov::element::f16, ov::element::i32, ov::element::i64};

const std::vector<std::vector<ov::Shape>> inShapes = {
        {{10, 2, 1, 1}, {10, 2, 1, 1}, {1, 2, 1, 1}},        {{1, 1, 1, 32}, {1, 1, 1, 1}, {1, 4, 16, 32}},
        {{1, 1, 1, 32}, {1, 4, 16, 32}, {1, 1, 1, 1}},       {{1, 1, 1, 1024}, {1, 1, 1, 1}, {1, 1, 1, 1024}},
        {{1, 1, 1, 1024}, {1, 1, 1, 1024}, {1, 1, 1, 1}},    {{1, 1, 1, 1024}, {1, 1, 1, 1024}, {1, 1, 1, 1024}},
        {{1, 1, 2, 4, 4}, {1, 1, 1, 1, 1}, {1, 1, 2, 4, 4}},
};

const auto selectTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), ::testing::ValuesIn(inputTypes),
        ::testing::Values(ov::op::AutoBroadcastType::NUMPY), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Select, SelectLayerTestCommon, selectTestParams,
                         SelectLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Select, ShaveCodeGenSelectLayerTestCommon, selectTestParams,
                         ShaveCodeGenSelectLayerTestCommon::getTestCaseName);
}  // namespace
