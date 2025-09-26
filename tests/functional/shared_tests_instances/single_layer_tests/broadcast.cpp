//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/broadcast.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class BroadcastLayerTestCommon : public BroadcastLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(BroadcastLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(BroadcastLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

// NUMPY MODE

const std::vector<ov::element::Type> inputPrecision = {ov::element::f16, ov::element::f32};

std::vector<std::vector<ov::Shape>> inShapesNumpy = {{{3, 1}}, {{1, 4, 1}}};

std::vector<std::vector<size_t>> targetShapesNumpy = {{2, 3, 6}, {1, 4, 4}};

const auto numpyBroadcastParams1 =
        ::testing::Combine(::testing::Values(targetShapesNumpy[0]),          // target shape
                           ::testing::Values(ov::AxisSet{}),                 // not used in numpy mode
                           ::testing::Values(ov::op::BroadcastType::NUMPY),  // broadcast mode
                           ::testing::Values(static_shapes_to_test_representation(inShapesNumpy[0])),  // Input shape
                           ::testing::ValuesIn(inputPrecision),                                        // Model type
                           ::testing::Values(test_utils::TARGET_DEVICE));                              // Device name

const auto numpyBroadcastParams2 =
        ::testing::Combine(::testing::Values(targetShapesNumpy[1]),                                    // target shape
                           ::testing::Values(ov::AxisSet{}),                                           // axes mapping
                           ::testing::Values(ov::op::BroadcastType::NUMPY),                            // broadcast mode
                           ::testing::Values(static_shapes_to_test_representation(inShapesNumpy[1])),  // Input shape
                           ::testing::ValuesIn(inputPrecision),                                        // Model type
                           ::testing::Values(test_utils::TARGET_DEVICE));                              // Device name

// BIDIRECTIONAL MODE

std::vector<std::vector<ov::Shape>> inShapesBidi = {{{4, 1}}, {{1, 4, 1}}, {{4, 1, 1}}};

std::vector<std::vector<size_t>> targetShapesBidi = {{2, 1, 4}, {1, 4, 4}, {1, 1, 2, 2}};

const auto bidirectionalBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[0]), ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
        ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(static_shapes_to_test_representation(inShapesBidi[0])), ::testing::ValuesIn(inputPrecision),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto bidirectionalBroadcastParams2 =
        ::testing::Combine(::testing::Values(targetShapesBidi[1]), ::testing::Values(ov::AxisSet{}),
                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                           ::testing::Values(static_shapes_to_test_representation(inShapesBidi[1])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(test_utils::TARGET_DEVICE));

const auto bidirectionalBroadcastParams3 =
        ::testing::Combine(::testing::Values(targetShapesBidi[2]), ::testing::Values(ov::AxisSet{}),
                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                           ::testing::Values(static_shapes_to_test_representation(inShapesBidi[2])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(test_utils::TARGET_DEVICE));

// EXPLICIT MODE

std::vector<std::vector<ov::Shape>> inShapesExplicit = {{{3, 1}}, {{2, 4}}};

std::vector<std::vector<size_t>> targetShapesExplicit = {{2, 3, 1}, {2, 3, 4}};

std::vector<ov::AxisSet> axes = {{1, 2}, {0, 2}};

const auto explicitBroadcastParams1 =
        ::testing::Combine(::testing::Values(targetShapesExplicit[0]), ::testing::Values(axes[0]),
                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                           ::testing::Values(static_shapes_to_test_representation(inShapesExplicit[0])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(test_utils::TARGET_DEVICE));

const auto explicitBroadcastParams2 =
        ::testing::Combine(::testing::Values(targetShapesExplicit[1]), ::testing::Values(axes[1]),
                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                           ::testing::Values(static_shapes_to_test_representation(inShapesExplicit[1])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_NumpyBroadcastCheck1, BroadcastLayerTestCommon, numpyBroadcastParams1,
                         BroadcastLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NumpyBroadcastCheck2, BroadcastLayerTestCommon, numpyBroadcastParams2,
                         BroadcastLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BidirectionalBroadcastCheck1, BroadcastLayerTestCommon,
                         bidirectionalBroadcastParams1, BroadcastLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_BidirectionalBroadcastCheck2, BroadcastLayerTestCommon, bidirectionalBroadcastParams2,
                         BroadcastLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_BidirectionalBroadcastCheck3, BroadcastLayerTestCommon, bidirectionalBroadcastParams3,
                         BroadcastLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ExplicitBroadcastCheck1, BroadcastLayerTestCommon, explicitBroadcastParams1,
                         BroadcastLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ExplicitBroadcastCheck2, BroadcastLayerTestCommon, explicitBroadcastParams2,
                         BroadcastLayerTestCommon::getTestCaseName);

}  // namespace
