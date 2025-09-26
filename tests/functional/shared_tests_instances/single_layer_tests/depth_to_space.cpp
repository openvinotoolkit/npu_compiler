//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/depth_to_space.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class DepthToSpaceLayerTestCommon : public DepthToSpaceLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(DepthToSpaceLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DepthToSpaceLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DepthToSpaceLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;
using ov::op::v0::DepthToSpace;

namespace {
const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::u8,
        ov::element::f16,
};

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> modes = {
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

const auto DepthToSpaceBS2_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 4, 3, 3}})}),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto DepthToSpaceBS3_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 9, 3, 3}})}),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(3),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto smoke_DepthToSpaceBS4_with_tiling =
        ::testing::Combine(::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 48, 80, 80}})}),
                           ::testing::Values(ov::element::f16), ::testing::ValuesIn(modes), ::testing::Values(4),
                           ::testing::Values(test_utils::TARGET_DEVICE));

std::vector<std::vector<ov::Shape>> inputShapesBS4 = {
        {{1, 16, 5, 4}}, {{1, 16, 9, 7}}, {{1, 128, 5, 4}}, {{1, 128, 9, 7}}};
const auto smoke_DepthToSpaceBS4 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS4)), ::testing::Values(ov::element::f16),
        ::testing::Values(DepthToSpace::DepthToSpaceMode::DEPTH_FIRST), ::testing::Values(4),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto DepthToSpaceBS2_with_large_height =
        ::testing::Combine(::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 4, 300, 3}})),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS2, DepthToSpaceLayerTestCommon, DepthToSpaceBS2_PRECOMMIT,
                         DepthToSpaceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS3, DepthToSpaceLayerTestCommon, DepthToSpaceBS3_PRECOMMIT,
                         DepthToSpaceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS4, DepthToSpaceLayerTestCommon, smoke_DepthToSpaceBS4,
                         DepthToSpaceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_tiling, DepthToSpaceLayerTestCommon, smoke_DepthToSpaceBS4_with_tiling,
                         DepthToSpaceLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_large_height, DepthToSpaceLayerTestCommon,
                         DepthToSpaceBS2_with_large_height, DepthToSpaceLayerTestCommon::getTestCaseName);

}  // namespace
