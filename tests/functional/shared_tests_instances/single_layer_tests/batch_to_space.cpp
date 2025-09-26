//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/batch_to_space.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class BatchToSpaceLayerTestCommon : public BatchToSpaceLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(BatchToSpaceLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(BatchToSpaceLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test

}  // namespace ov

using ov::test::BatchToSpaceLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> shapes = {{{16, 12, 10}}, {{4, 4, 4, 4}}, {{48, 3, 3, 1, 3}}};

const auto precommit_BatchToSpace_3D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 1, 4}), ::testing::Values(std::vector<int64_t>{0, 0, 2}),
        ::testing::Values(std::vector<int64_t>{0, 0, 2}),
        ::testing::ValuesIn({ov::test::static_shapes_to_test_representation({shapes[0]})}),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto precommit_BatchToSpace_4D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 2, 1, 2}), ::testing::Values(std::vector<int64_t>{0, 0, 1, 0}),
        ::testing::Values(std::vector<int64_t>{0, 0, 0, 1}),
        ::testing::ValuesIn({ov::test::static_shapes_to_test_representation({shapes[1]})}),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto precommit_BatchToSpace_5D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 2, 4, 3, 1}), ::testing::Values(std::vector<int64_t>{0, 0, 1, 0, 0}),
        ::testing::Values(std::vector<int64_t>{0, 0, 1, 0, 0}),
        ::testing::ValuesIn({ov::test::static_shapes_to_test_representation({shapes[2]})}),
        ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchToSpace_3D, BatchToSpaceLayerTestCommon, precommit_BatchToSpace_3D,
                         BatchToSpaceLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchToSpace_4D, BatchToSpaceLayerTestCommon, precommit_BatchToSpace_4D,
                         BatchToSpaceLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchToSpace_5D, BatchToSpaceLayerTestCommon, precommit_BatchToSpace_5D,
                         BatchToSpaceLayerTestCommon::getTestCaseName);

}  // namespace
