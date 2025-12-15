//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/tile.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class TileLayerTestCommon : public TileLayerTest, virtual public VpuOv2LayerTest {};
class TileLayerTest_tiling : public TileLayerTestCommon {};

TEST_P(TileLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(TileLayerTest_tiling, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TileLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(TileLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::TileLayerTest_tiling;
using ov::test::TileLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16, ov::element::u8};

const auto tileParams = ::testing::Combine(
        ::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 3, 2}, {3, 2, 1, 5}, {1, 3, 2, 1}})),
        ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                std::vector<std::vector<ov::Shape>>({{{1, 4, 3, 2}}, {{4, 3, 2, 1}}, {{4, 3, 2, 5}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto tileParamsPrecommit = ::testing::Combine(
        ::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{2, 3, 1}})), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{3, 4, 2}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Tile, TileLayerTestCommon, tileParams, TileLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Tile, TileLayerTestCommon, tileParamsPrecommit,
                         TileLayerTestCommon::getTestCaseName);

// NPU3720 - tiling

// case 1: tile on two dimensions
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_1, TileLayerTest_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 2, 3,
                                                                                           3}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 1, 2880, 50}}}))),  // input_shape
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TileLayerTest_tiling::getTestCaseName);

// case 2: repeats values aren't 1
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_2, TileLayerTest_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{2, 2, 3,
                                                                                           3}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{3, 2, 723, 25}}}))),  // input_shape
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TileLayerTest_tiling::getTestCaseName);

// case 3: repeats values may be 1
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_3, TileLayerTest_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{3, 1, 3,
                                                                                           2}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{2, 3, 360, 50}}}))),  // input_shape
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TileLayerTest_tiling::getTestCaseName);

// case 4: tiling dim not divisible
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_4, TileLayerTest_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 1, 1,
                                                                                           14}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 300, 2744, 1}}}))),  // input_shape
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TileLayerTest_tiling::getTestCaseName);

// model case: tensor<1x32x1x1xf16> -> tensor<1x32x1x65536xf16> , NHWC
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_5, TileLayerTest_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 1, 1,
                                                                                           65536}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 32, 1, 1}}}))),  // input_shape
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TileLayerTest_tiling::getTestCaseName);

// case 6: INT32 tiling on two dimensions
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_6, TileLayerTest_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 1, 256,
                                                                                           256}})),  // repeats_values
                           ::testing::Values(ov::element::i32),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 32, 1, 1}}}))),  // input_shape
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TileLayerTest_tiling::getTestCaseName);

}  // namespace
