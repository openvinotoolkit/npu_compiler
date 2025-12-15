
// Copyright (C) 2022-2025 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/embedding_bag_offsets_sum.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class EmbeddingBagOffsetsSumLayerTestCommon : public EmbeddingBagOffsetsSumLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(EmbeddingBagOffsetsSumLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EmbeddingBagOffsetsSumLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(EmbeddingBagOffsetsSumLayerTestCommon, NPU5010_HW) {
    abs_threshold = 0.06;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::i32,
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
};

const std::vector<ov::element::Type> indPrecisions = {
        ov::element::i32,
};

const std::vector<std::vector<ov::Shape>> emb_table_shape = {{{10, 35, 8}}, {{5, 6}}};
const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}};
const std::vector<std::vector<size_t>> offsets = {{0, 2}};
const std::vector<size_t> default_index = {0};
const std::vector<bool> with_weights = {true, false};
const std::vector<bool> with_default_index = {true, false};

const auto EmbeddingBagOffsetsSumParams1 = ::testing::Combine(
        ::testing::ValuesIn(indices), ::testing::ValuesIn(offsets), ::testing::ValuesIn(default_index),
        ::testing::ValuesIn(with_weights), ::testing::ValuesIn(with_default_index));

INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingBagOffsetsSum, EmbeddingBagOffsetsSumLayerTestCommon,
                         ::testing::Combine(EmbeddingBagOffsetsSumParams1,
                                            ::testing::ValuesIn(static_shapes_to_test_representation(emb_table_shape)),
                                            ::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(indPrecisions),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         EmbeddingBagOffsetsSumLayerTestCommon::getTestCaseName);

}  // namespace
