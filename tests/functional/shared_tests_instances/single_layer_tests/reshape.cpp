// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reshape.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/reshape.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ReshapeLayerTestCommon : public ReshapeLayerTest, virtual public VpuOv2LayerTest {
private:
    void SetUp() override {
        std::vector<size_t> inputShape;
        ov::element::Type modelType;
        std::vector<int64_t> outFormShapes;
        bool specialZero;
        std::tie(specialZero, modelType, inputShape, outFormShapes, std::ignore) = this->GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front());
        auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{outFormShapes.size()},
                                                                 outFormShapes);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(param, const_node, specialZero);
        VpuOv2LayerTest::function =
                std::make_shared<ov::Model>(reshape->outputs(), ov::ParameterVector{param}, "Reshape");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(ReshapeLayerTestCommon, NPU3720_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReshapeLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
TEST_P(ReshapeLayerTestCommon, NPU5010_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const auto paramCollapse1 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 1, 1, 100}, {1, 100, 1, 1}})),
        ::testing::Values(std::vector<int64_t>({0, 100})), ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramCollapse2 = ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                                               ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                                               ::testing::Values(std::vector<int64_t>({1, 0, 100})),
                                               ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramExpand1 = ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                                             ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                                             ::testing::Values(std::vector<int64_t>({1, 0, 100})),
                                             ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramExpand2 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes), ::testing::Values(std::vector<size_t>({1, 100})),
        ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})), ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramExpand3 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes), ::testing::Values(std::vector<size_t>({1, 2, 100})),
        ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})), ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramGeneric1 = ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                                              ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                                              ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
                                              ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramGeneric2 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes), ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
        ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, ReshapeLayerTestCommon, paramCollapse1,
                         ReshapeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, ReshapeLayerTestCommon, paramCollapse2,
                         ReshapeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, ReshapeLayerTestCommon, paramExpand1,
                         ReshapeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, ReshapeLayerTestCommon, paramExpand2,
                         ReshapeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, ReshapeLayerTestCommon, paramExpand3,
                         ReshapeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, ReshapeLayerTestCommon, paramGeneric1,
                         ReshapeLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric2, ReshapeLayerTestCommon, paramGeneric2,
                         ReshapeLayerTestCommon::getTestCaseName);

}  // namespace
