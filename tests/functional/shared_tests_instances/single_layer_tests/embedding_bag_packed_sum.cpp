//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/embedding_bag_packed_sum.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class EmbeddingBagPackedSumLayerTestCommon : public EmbeddingBagPackedSumLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(EmbeddingBagPackedSumLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EmbeddingBagPackedSumLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(EmbeddingBagPackedSumLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<ov::Shape>> embTableShape = {{{5, 10}}};
const std::vector<std::vector<size_t>> indices = {{0, 2}, {1, 2}, {3, 4}};
const std::vector<bool> withWeights = {true, false};
const auto params = ::testing::Combine(::testing::Values(indices), ::testing::ValuesIn(withWeights));

const ov::element::Type embeddingTablePrecision = ov::element::f16;
const ov::element::Type indicesPrecisions = ov::element::i32;

INSTANTIATE_TEST_SUITE_P(smoke_precommit_EmbeddingBagPackedSum, EmbeddingBagPackedSumLayerTestCommon,
                         ::testing::Combine(params,
                                            ::testing::ValuesIn(static_shapes_to_test_representation(embTableShape)),
                                            ::testing::Values(embeddingTablePrecision),
                                            ::testing::Values(indicesPrecisions),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         EmbeddingBagPackedSumLayerTest::getTestCaseName);

}  // namespace
