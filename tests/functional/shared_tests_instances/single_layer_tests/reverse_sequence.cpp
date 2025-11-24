//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reverse_sequence.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/reverse_sequence.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ReverseSequenceLayerTestCommon : public ReverseSequenceLayerTest, virtual public VpuOv2LayerTest {
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
    void SetUp() override {
        ov::element::Type modelType;
        int64_t batchAxisIdx;
        int64_t seqAxisIdx;
        std::vector<size_t> inputShape;
        std::vector<size_t> secondInputShape;
        InputLayerType secondaryInputType;

        std::tie(batchAxisIdx, seqAxisIdx, inputShape, secondInputShape, secondaryInputType, modelType, std::ignore) =
                GetParam();

        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape, secondInputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front())};
        auto second_data_type = ov::element::i32;  // according to the specification
        std::shared_ptr<ov::Node> secondary_input;
        if (InputLayerType::CONSTANT == secondaryInputType) {
            auto tensor = create_and_fill_tensor(second_data_type, secondInputShape);
            secondary_input = std::make_shared<ov::op::v0::Constant>(tensor);
        } else if (InputLayerType::PARAMETER == secondaryInputType) {
            secondary_input = std::make_shared<ov::op::v0::Parameter>(second_data_type, ov::Shape(secondInputShape));
            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(secondary_input));
        } else {
            throw std::runtime_error("Unsupported input type");
        }

        auto reverse =
                std::make_shared<ov::op::v0::ReverseSequence>(params[0], secondary_input, batchAxisIdx, seqAxisIdx);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(reverse->outputs(), params, "ReverseSequence");
    }
};

TEST_P(ReverseSequenceLayerTestCommon, NPU3720_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReverseSequenceLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::f16, ov::element::u8};
const std::vector<int64_t> batchAxisIndices = {0};
const std::vector<int64_t> batchAxisInner = {2};
const std::vector<int64_t> seqAxisIndices = {1};
const std::vector<int64_t> seqAxisIndices4D = {1, 2, 3};
const std::vector<std::vector<size_t>> inputShapes = {{3, 10}, {3, 10, 12}, {3, 10, 11, 20}};
const std::vector<std::vector<size_t>> reversSeqLengthsVecShapes = {{3}};
const std::vector<std::vector<size_t>> reversSeqLengthsVecShapesInner = {{11}};

const auto configSmokeParams = ::testing::Combine(
        ::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices), ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(reversSeqLengthsVecShapes), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParams4D = ::testing::Combine(
        ::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices4D), ::testing::Values(inputShapes[2]),
        ::testing::ValuesIn(reversSeqLengthsVecShapes), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(netPrecisions[0]), ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsInnerBatch = ::testing::Combine(
        ::testing::ValuesIn(batchAxisInner), ::testing::ValuesIn(seqAxisIndices), ::testing::Values(inputShapes[2]),
        ::testing::ValuesIn(reversSeqLengthsVecShapesInner), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(netPrecisions[0]), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReverseSequence, ReverseSequenceLayerTestCommon, configSmokeParams,
                         ReverseSequenceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequence_4D, ReverseSequenceLayerTestCommon, configParams4D,
                         ReverseSequenceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequence_InnerBatch, ReverseSequenceLayerTestCommon, configParamsInnerBatch,
                         ReverseSequenceLayerTestCommon::getTestCaseName);

}  // namespace
