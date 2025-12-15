//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include <stdint.h>
#include <array>

#include <gtest/gtest.h>
#include <vector>
#include <vpux_headers/relocations.hpp>

#include "common_test_utils/test_assertions.hpp"

using namespace elf;
using namespace elf::relocations;

namespace {

struct ReduceDmaTestsParams {
    std::string testName;
    uint32_t dmaShapes[DMA_SYMBOL_MAX_TENSOR_DIMENSIONS];
    uint32_t dmaStrides[DMA_SYMBOL_MAX_TENSOR_DIMENSIONS];
    std::array<uint32_t, DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> expectedReducedDmaShapes;
    std::array<uint32_t, DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> expectedReducedDmaStrides;
    uint32_t dmaSize;
};

class ReduceDmaTest : public testing::TestWithParam<ReduceDmaTestsParams> {};

TEST_P(ReduceDmaTest, ReduceResultsAreCorrect) {
    auto params = GetParam();

    std::array<uint32_t, DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> reducedDmaShapes{1, 1, 1, 1, 1, 1};
    std::array<uint32_t, DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> reducedDmaStrides{0, 0, 0, 0, 0, 0};

    reduceDmaDims(params.dmaShapes, params.dmaStrides, params.dmaSize, reducedDmaShapes, reducedDmaStrides);

    for (size_t idx = 0; idx < DMA_SYMBOL_MAX_TENSOR_DIMENSIONS; idx++) {
        ASSERT_EQ(reducedDmaShapes[idx], params.expectedReducedDmaShapes[idx]);
        ASSERT_EQ(reducedDmaStrides[idx], params.expectedReducedDmaStrides[idx]);
    }
}

INSTANTIATE_TEST_SUITE_P(ReduceDmaTestSuite, ReduceDmaTest,
                         testing::Values(ReduceDmaTestsParams{"CompactDmaByteElement",
                                                              {6, 2, 4, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              {48, 1, 1, 1, 1, 1},
                                                              {1, 0, 0, 0, 0, 0},
                                                              1},
                                         ReduceDmaTestsParams{"ContinuousDmaSlicedOnInnermostDimByteElement",
                                                              {3, 2, 4, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              {3, 2, 4, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              1},
                                         ReduceDmaTestsParams{"ContinuousDmaSlicedOnOuterDimByteElement",
                                                              {6, 2, 2, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              {24, 1, 1, 1, 1, 1},
                                                              {1, 48, 0, 0, 0, 0},
                                                              1},
                                         ReduceDmaTestsParams{"ContinuousDmaSlicedOnMiddleDimByteElement",
                                                              {6, 2, 4, 1, 10, 1},
                                                              {1, 6, 12, 48, 96, 960},
                                                              {48, 10, 1, 1, 1, 1},
                                                              {1, 96, 960, 0, 0, 0},
                                                              1},
                                         ReduceDmaTestsParams{"StridedDmaSlicedOnInnermostDimByteElement",
                                                              {6, 2, 4, 1, 1, 1},
                                                              {2, 12, 24, 96, 0, 0},
                                                              {1, 6, 2, 4, 1, 1},
                                                              {1, 2, 12, 24, 96, 0},
                                                              1},
                                         ReduceDmaTestsParams{"CompactDmaNonByteElement",
                                                              {6, 2, 4, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              {192, 1, 1, 1, 1, 1},
                                                              {4, 0, 0, 0, 0, 0},
                                                              4},
                                         ReduceDmaTestsParams{"ContinuousDmaSlicedOnInnermostDimNonByteElement",
                                                              {3, 2, 4, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              {12, 2, 4, 1, 1, 1},
                                                              {4, 24, 48, 192, 0, 0},
                                                              4},
                                         ReduceDmaTestsParams{"ContinuousDmaSlicedOnOuterDimNonByteElement",
                                                              {6, 2, 2, 1, 1, 1},
                                                              {1, 6, 12, 48, 0, 0},
                                                              {96, 1, 1, 1, 1, 1},
                                                              {4, 192, 0, 0, 0, 0},
                                                              4},
                                         ReduceDmaTestsParams{"ContinuousDmaSlicedOnMiddleDimNonByteElement",
                                                              {6, 2, 4, 1, 10, 1},
                                                              {1, 6, 12, 48, 96, 960},
                                                              {192, 10, 1, 1, 1, 1},
                                                              {4, 384, 3840, 0, 0, 0},
                                                              4},
                                         ReduceDmaTestsParams{"StridedDmaSlicedOnInnermostDimNonByteElement",
                                                              {6, 2, 4, 1, 1, 1},
                                                              {2, 12, 24, 96, 0, 0},
                                                              {4, 6, 2, 4, 1, 1},
                                                              {4, 8, 48, 96, 384, 0},
                                                              4}),
                         [](const testing::TestParamInfo<ReduceDmaTestsParams>& info) {
                             return info.param.testName;
                         });

struct TileAddressCalculationParams {
    std::string testName;
    uint32_t tileOffsets[DMA_SYMBOL_MAX_TENSOR_DIMENSIONS];
    uint32_t strides[DMA_SYMBOL_MAX_TENSOR_DIMENSIONS];
    uint64_t baseAddress;
    uint64_t expectedAddress;
    uint32_t elementSize;
};

class TileAddressCalculationTest : public testing::TestWithParam<TileAddressCalculationParams> {};

TEST_P(TileAddressCalculationTest, ResultsAreCorrect) {
    auto params = GetParam();

    auto actualAddress =
            calculateDmaAddress(params.baseAddress, params.tileOffsets, params.strides, params.elementSize);

    ASSERT_EQ(actualAddress, params.expectedAddress);
}

INSTANTIATE_TEST_SUITE_P(TileAddressCalculationTestSuite, TileAddressCalculationTest,
                         testing::Values(TileAddressCalculationParams{"NoOffsets",
                                                                      {0, 0, 0, 0, 0, 0},
                                                                      {1, 2, 3, 4, 5, 6},
                                                                      0xA0000000,
                                                                      0xA0000000,
                                                                      1},
                                         TileAddressCalculationParams{"OffsetOnFirstDimByteElement",
                                                                      {2, 0, 0, 0, 0, 0},
                                                                      {2, 4, 8, 16, 0, 0},
                                                                      0xA0000000,
                                                                      0xA0000004,
                                                                      1},
                                         TileAddressCalculationParams{"OffsetOnMultipleDimsByteElement",
                                                                      {2, 0, 4, 0, 0, 0},
                                                                      {2, 4, 8, 16, 0, 0},
                                                                      0xA0000000,
                                                                      0xA0000024,
                                                                      1},
                                         TileAddressCalculationParams{"OffsetOnFirstDimNonByteElement",
                                                                      {2, 0, 0, 0, 0, 0},
                                                                      {1, 2, 3, 4, 5, 6},
                                                                      0xA0000000,
                                                                      0xA0000008,
                                                                      4},
                                         TileAddressCalculationParams{"OffsetOnMultipleDimsNonByteElement",
                                                                      {2, 0, 4, 3, 0, 0},
                                                                      {2, 4, 8, 16, 0, 0},
                                                                      0xA0000000,
                                                                      0xA0000150,
                                                                      4}),
                         [](const testing::TestParamInfo<TileAddressCalculationParams>& info) {
                             return info.param.testName;
                         });

}  // namespace
