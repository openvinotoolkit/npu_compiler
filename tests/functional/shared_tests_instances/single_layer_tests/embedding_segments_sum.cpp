
// Copyright (C) 2022-2025 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/embedding_segments_sum.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class EmbeddingSegmentsSumLayerTestCommon : public EmbeddingSegmentsSumLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(EmbeddingSegmentsSumLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EmbeddingSegmentsSumLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::i32, ov::element::f16, ov::element::f32,
                                                      ov::element::u8};

const std::vector<ov::element::Type> indPrecisions = {ov::element::i32};

const std::vector<std::vector<ov::Shape>> embTableShape = {{{5, 6, 4}}, {{10, 35, 8}}};
const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}};
const std::vector<std::vector<size_t>> segmentIds = {{0, 1, 2, 3, 4}};
const std::vector<size_t> numSegments = {7};
const std::vector<size_t> defaultIndex = {0};
const std::vector<bool> withWeights = {false, true};
const std::vector<bool> withDefaultIndex = {false, true};

const auto params = testing::Combine(::testing::ValuesIn(indices), ::testing::ValuesIn(segmentIds),
                                     ::testing::ValuesIn(numSegments), ::testing::ValuesIn(defaultIndex),
                                     ::testing::ValuesIn(withWeights), ::testing::ValuesIn(withDefaultIndex));

INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingSegmentsSumCheck1, EmbeddingSegmentsSumLayerTestCommon,
                         ::testing::Combine(params,
                                            ::testing::ValuesIn(static_shapes_to_test_representation(embTableShape)),
                                            ::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(indPrecisions),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         EmbeddingSegmentsSumLayerTestCommon::getTestCaseName);

}  // namespace
