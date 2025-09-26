//
// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/group_normalization.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GroupNormalizationLayerTestCommon : public GroupNormalizationTest, virtual public VpuOv2LayerTest {};

TEST_P(GroupNormalizationLayerTestCommon, NPU3720_SW) {
    abs_threshold = 0.08;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupNormalizationLayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.08;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {ov::element::f16, ov::element::f32};

// static shapes
const std::vector<ov::Shape> staticInputShapes = {{4, 4, 4},      {3, 8, 8},      {2, 4, 8},
                                                  {4, 4, 16, 16}, {1, 4, 16, 16}, {16, 16, 16, 16}};

const std::vector<int64_t> numGroups = {2, 4};
const std::vector<double> epsilon = {0.0001};

std::vector<ov::AnyMap> additionalConfig = {{}};

const auto groupNormalizationParams =
        testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::Values(ov::element::dynamic),
                         ::testing::Values(ov::element::dynamic),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(staticInputShapes)),
                         ::testing::ValuesIn(numGroups), ::testing::ValuesIn(epsilon),
                         ::testing::Values(test_utils::TARGET_DEVICE), ::testing::ValuesIn(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GroupNormalizationStatic, GroupNormalizationLayerTestCommon,
                         groupNormalizationParams, GroupNormalizationLayerTestCommon::getTestCaseName);

}  // anonymous namespace
