//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/space_to_depth.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SpaceToDepthLayerTest);

class SpaceToDepthLayerTestCommon : public SpaceToDepthLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

class SpaceToDepthLayerTest_NPU3720 : public SpaceToDepthLayerTestCommon {};

TEST_P(SpaceToDepthLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SpaceToDepthLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SpaceToDepthLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(SpaceToDepthLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(SpaceToDepthLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(SpaceToDepthLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(SpaceToDepthLayerTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_P(SpaceToDepthLayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using ov::test::SpaceToDepthLayerTest_NPU3720;
using ov::test::SpaceToDepthLayerTestCommon;

namespace {
const std::vector<ov::element::Type> inputTypes = {ov::element::f32,
                                                   ov::element::f16,  // value from CPU-plugin I16 is changed for FP16
                                                   ov::element::u8};

const std::vector<ov::op::v0::SpaceToDepth::SpaceToDepthMode> modes = {
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

const auto SpaceToDepthBS2_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 2, 3 * 4, 3 * 4}}}))),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto SpaceToDepthBS3_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 2, 3 * 3, 3 * 3}}}))),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(3),
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS2, SpaceToDepthLayerTestCommon, SpaceToDepthBS2_PRECOMMIT,
                         SpaceToDepthLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS3, SpaceToDepthLayerTestCommon, SpaceToDepthBS3_PRECOMMIT,
                         SpaceToDepthLayerTestCommon::getTestCaseName);

// DEPTH_FIRST with N=1 lowers to the NCHW->NHWC SpaceToDepth DMA on NPU3720; N>1 must fall back.
const auto makeSpaceToDepthDepthFirstParams = [](std::vector<std::vector<ov::Shape>> shapes, size_t blockSize) {
    return ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                              ::testing::Values(ov::element::f16, ov::element::u8),
                              ::testing::Values(ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST),
                              ::testing::Values(blockSize), ::testing::Values(test_utils::TARGET_DEVICE));
};

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepth_DepthFirst_BS2, SpaceToDepthLayerTest_NPU3720,
                         makeSpaceToDepthDepthFirstParams({{{1, 16, 32, 48}}, {{2, 3, 16, 16}}}, 2),
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepth_DepthFirst_BS3, SpaceToDepthLayerTest_NPU3720,
                         makeSpaceToDepthDepthFirstParams({{{1, 5, 12, 18}}}, 3),
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepth_DepthFirst_BS4, SpaceToDepthLayerTest_NPU3720,
                         makeSpaceToDepthDepthFirstParams({{{1, 3, 64, 160}}, {{2, 3, 32, 32}}}, 4),
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepth_DepthFirst_BS16, SpaceToDepthLayerTest_NPU3720,
                         makeSpaceToDepthDepthFirstParams({{{1, 3, 224, 224}}, {{1, 1, 16, 64}}}, 16),
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

}  // namespace
