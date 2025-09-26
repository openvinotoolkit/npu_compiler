//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/space_to_batch.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class SpaceToBatchLayerTestCommon : public SpaceToBatchLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(SpaceToBatchLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(SpaceToBatchLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using ov::test::SpaceToBatchLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> shapes = {{{1, 12, 10}}, {{2, 8, 8, 3}}, {{2, 8, 8, 3, 3}}};

const auto precommit_SpaceToBatch_3D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 1, 8}), ::testing::Values(std::vector<int64_t>{0, 0, 2}),
        ::testing::Values(std::vector<int64_t>{0, 0, 4}),
        ::testing::ValuesIn({ov::test::static_shapes_to_test_representation({shapes[0]})}),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto precommit_SpaceToBatch_4D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 6, 4, 1}), ::testing::Values(std::vector<int64_t>{0, 1, 0, 0}),
        ::testing::Values(std::vector<int64_t>{0, 3, 0, 0}),
        ::testing::ValuesIn({ov::test::static_shapes_to_test_representation({shapes[1]})}),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto precommit_SpaceToBatch_5D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 6, 4, 1, 1}), ::testing::Values(std::vector<int64_t>{0, 1, 0, 0, 0}),
        ::testing::Values(std::vector<int64_t>{0, 3, 0, 0, 0}),
        ::testing::ValuesIn({ov::test::static_shapes_to_test_representation({shapes[2]})}),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToBatch_3D, SpaceToBatchLayerTestCommon, precommit_SpaceToBatch_3D,
                         SpaceToBatchLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToBatch_4D, SpaceToBatchLayerTestCommon, precommit_SpaceToBatch_4D,
                         SpaceToBatchLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToBatch_5D, SpaceToBatchLayerTestCommon, precommit_SpaceToBatch_5D,
                         SpaceToBatchLayerTestCommon::getTestCaseName);

}  // namespace
