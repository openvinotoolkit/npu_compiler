// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/shuffle_channels.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class ShuffleChannelsLayerTestCommon : public ShuffleChannelsLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(ShuffleChannelsLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ShuffleChannelsLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using ov::test::ShuffleChannelsLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> inputShapes = {{{3, 4, 9, 5}}, {{2, 16, 24, 15}}, {{1, 32, 12, 25}}};

const std::vector<ov::test::shuffleChannelsSpecificParams> shuffleParameters = {
        std::make_tuple(1, 2),  std::make_tuple(-3, 2), std::make_tuple(2, 3),
        std::make_tuple(-2, 3), std::make_tuple(3, 5),  std::make_tuple(-1, 5)};

const auto params0 = testing::Combine(testing::ValuesIn(shuffleParameters), testing::ValuesIn(modelTypes),
                                      testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                                      testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels, ShuffleChannelsLayerTestCommon, params0,
                         ShuffleChannelsLayerTestCommon::getTestCaseName);

}  // namespace

namespace {  // conformance scenarios

const std::vector<std::vector<ov::Shape>> inShapes = {
        {{1, 116, 28, 28}}, {{1, 232, 14, 14}}, {{1, 464, 7, 7}},  {{1, 32, 28, 28}}, {{1, 64, 14, 14}},
        {{1, 128, 7, 7}},   {{1, 24, 28, 28}},  {{1, 48, 14, 14}}, {{1, 96, 7, 7}},
};

const std::vector<std::tuple<int, int>> shParams = {
        std::make_tuple(1, 2)  // axis=1, group=2
};

const auto params1 = testing::Combine(testing::ValuesIn(shParams), testing::Values(ov::element::f16),
                                      testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                      testing::Values(test_utils::TARGET_DEVICE));

const auto precommit_params = testing::Combine(testing::ValuesIn(shParams), testing::Values(ov::element::f16),
                                               testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                       std::vector<std::vector<ov::Shape>>({{{1, 4, 3, 2}}}))),
                                               testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(conform_ShuffleChannels, ShuffleChannelsLayerTestCommon, params1,
                         ShuffleChannelsLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(conform_precommit_ShuffleChannels, ShuffleChannelsLayerTestCommon, precommit_params,
                         ShuffleChannelsLayerTestCommon::getTestCaseName);

}  // namespace
