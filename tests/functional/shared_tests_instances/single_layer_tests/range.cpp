// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/range.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class RangeLayerTestCommon : public RangeLayerTest, virtual public VpuOv2LayerTest {
    // Latest 'RangeLayerTest::SetUp' builds 'start,stop,step' as non-CONST inputs, thus unable to infer output shape.
    // So using older SetUp that builds CONST inputs.
    // `Parameter` op was also needed to be added to the result of `Range` and `Select` op was used as an interface
    // between `Parameter` and `Range` ops.
    void SetUp() override {
        ov::element::Type modelType;
        float start, stop, step;
        std::tie(start, stop, step, modelType, std::ignore) = GetParam();

        auto constants = std::vector<std::shared_ptr<ov::op::v0::Constant>>{
                std::make_shared<ov::op::v0::Constant>(modelType, ov::Shape(), start),
                std::make_shared<ov::op::v0::Constant>(modelType, ov::Shape(), stop),
                std::make_shared<ov::op::v0::Constant>(modelType, ov::Shape(), step)};

        constants[0]->set_friendly_name("start");
        constants[1]->set_friendly_name("stop");
        constants[2]->set_friendly_name("step");

        auto range = std::make_shared<ov::op::v4::Range>(constants[0], constants[1], constants[2], modelType);
        constants.push_back(
                std::make_shared<ov::op::v0::Constant>(ov::element::boolean, range->output(0).get_shape(), false));

        std::vector<ov::test::InputShape> inputShape =
                ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{range->output(0).get_shape()});

        VpuOv2LayerTest::init_input_shapes(inputShape);

        auto params = ov::ParameterVector{
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::targetStaticShapes.front().at(0)),
        };

        auto select = std::make_shared<ov::op::v1::Select>(constants[3], params[0], range);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(select)};
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(results, params, "Range");
    }

    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }

    void infer() override {
        // Cancel latest implementation, rely on default behavior
        VpuOv2LayerTest::infer();
    }
};

// Do not support dynamic shapes yet, but created tests for validation when will support.
// Tests will be disabled until this feature is implemented.
// [Tracking number E#113199]
class RangeLayerTestDynamic : public RangeLayerTestCommon {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        float start, stop, step;
        tie(start, stop, step, std::ignore, std::ignore) = GetParam();

        auto inputMap = utils::getInputMap();
        auto itTargetShape = targetInputStaticShapes.begin();
        auto params = VpuOv2LayerTest::function->get_parameters();
        std::vector<float> paramsValues = {start, stop, step};
        for (size_t idx_params = 0; idx_params < params.size(); idx_params++) {
            std::shared_ptr<ov::Node> inputNode = params[idx_params];
            for (size_t i = 0; i < params[idx_params]->get_output_size(); i++) {
                for (const auto& node : params[idx_params]->get_output_target_inputs(i)) {
                    std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                    auto it = inputMap.find(nodePtr->get_type_info());
                    ASSERT_NE(it, inputMap.end());
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                            ov::Tensor tensor{params[idx_params]->get_element_type(), params[idx_params]->get_shape()};
                            *tensor.data<float>() = paramsValues[idx_params];
                            VpuOv2LayerTest::inputs.insert({params[idx_params], tensor});
                            break;
                        }
                    }
                }
            }
            itTargetShape++;
        }
    }
    void SetUp() override {
        ov::element::Type modelType;
        float start, stop, step;
        tie(start, stop, step, modelType, std::ignore) = GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({ov::Shape()}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape()),
                                   std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape()),
                                   std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape())};
        params[0]->set_friendly_name("start");
        params[1]->set_friendly_name("stop");
        params[2]->set_friendly_name("step");

        auto range = std::make_shared<ov::op::v4::Range>(params[0], params[1], params[2], modelType);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(range)};
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(results, params, "Range");
    }
};

TEST_P(RangeLayerTestDynamic, NPU3720_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(RangeLayerTestCommon, NPU3720_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(RangeLayerTestCommon, NPU4000_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
}  // namespace test

}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f32};

const std::vector<float> start = {2.0f, 1.0f};
const std::vector<float> stop = {23.0f, 15.0f};
const std::vector<float> step = {3.0f, 4.5f};

const auto testRangePositiveStepParams = ::testing::Combine(testing::ValuesIn(start),  // start
                                                            testing::ValuesIn(stop),   // stop
                                                            testing::ValuesIn(step),   // positive step
                                                            testing::ValuesIn(modelTypes), testing::Values(DEVICE_NPU));

const auto testRangeNegativeStepParams = ::testing::Combine(testing::Values(23.0f),  // start
                                                            testing::Values(2.0f),   // stop
                                                            testing::Values(-3.0f),  // negative step
                                                            testing::ValuesIn(modelTypes), testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Range, RangeLayerTestCommon, testRangePositiveStepParams,
                         RangeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_negative_Range, RangeLayerTestCommon, testRangeNegativeStepParams,
                         RangeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Range, RangeLayerTestDynamic, testRangePositiveStepParams,
                         RangeLayerTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_negative_Range, RangeLayerTestDynamic, testRangeNegativeStepParams,
                         RangeLayerTestDynamic::getTestCaseName);

}  // namespace
