//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include "single_op_tests/select.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

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

class SelectLayerTest_NPU3720 : public SelectLayerTestCommon {};
class SelectLayerTest_NPU4000 : public SelectLayerTestCommon {};

TEST_P(SelectLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(SelectLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::SelectLayerTest_NPU3720;
using ov::test::SelectLayerTest_NPU4000;

namespace {
const std::vector<ov::element::Type> inputTypes = {
        ov::element::f16,
};

const std::vector<std::vector<ov::Shape>> shapes = {
        {{1}, {1}, {1}},
        {{8}, {8}, {8}},
        {{4, 5}, {4, 5}, {4, 5}},
        {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
};

const std::vector<std::vector<ov::Shape>> shapesHighDims = {
        {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
        {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
};

const std::vector<std::vector<ov::Shape>> inShapes = {
        {{10, 2, 1, 1}, {10, 2, 1, 1}, {1, 2, 1, 1}},     {{1, 1, 1, 32}, {1, 1, 1, 1}, {1, 4, 16, 32}},
        {{1, 1, 1, 32}, {1, 4, 16, 32}, {1, 1, 1, 1}},    {{1, 1, 1, 1024}, {1, 1, 1, 1}, {1, 1, 1, 1024}},
        {{1, 1, 1, 1024}, {1, 1, 1, 1024}, {1, 1, 1, 1}}, {{1, 1, 1, 1024}, {1, 1, 1, 1024}, {1, 1, 1, 1024}}};

const auto selectTestParams0 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)), ::testing::ValuesIn(inputTypes),
        ::testing::Values(ov::op::AutoBroadcastType::NONE), ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto selectTestParams_highDims =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapesHighDims)),
                           ::testing::ValuesIn(inputTypes), ::testing::Values(ov::op::AutoBroadcastType::NONE),
                           ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto selectTestParams1 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), ::testing::ValuesIn(inputTypes),
        ::testing::Values(ov::op::AutoBroadcastType::NUMPY), ::testing::Values(ov::test::utils::DEVICE_NPU));

// --------- NPU3720 ---------

INSTANTIATE_TEST_SUITE_P(smoke_Select, SelectLayerTest_NPU3720, selectTestParams1,
                         SelectLayerTest_NPU3720::getTestCaseName);

// --------- NPU4000 ---------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Select, SelectLayerTest_NPU4000, selectTestParams1,
                         SelectLayerTest_NPU4000::getTestCaseName);

}  // namespace
