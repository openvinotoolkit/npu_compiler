//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

#include <gtest/gtest.h>

namespace {

using namespace vpux;

struct DmaCanonicalizationTestParams {
    MemShape dmaBufferShape;
    MemStrides dmaBufferStrides;
    Shape argumentShape;
    llvm::SmallVector<int64_t> tileOffsets;
    llvm::SmallVector<int64_t> expectedDmaShape;
    llvm::SmallVector<int64_t> expectedDmaStride;
    llvm::SmallVector<int64_t> expectedOffsets;
};

using DmaCanonicalizationTest = ::testing::TestWithParam<DmaCanonicalizationTestParams>;

TEST_P(DmaCanonicalizationTest, Test) {
    auto params = GetParam();
    llvm::SmallVector<int64_t> canonicalTileOffset(params.expectedOffsets.size(), 0);
    llvm::SmallVector<int64_t> canonicalDmaShape(params.expectedDmaShape.size(), 1);
    llvm::SmallVector<int64_t> canonicalDmaStride(params.expectedDmaStride.size(), 0);

    ELF::getCanonicalDmaForm(params.dmaBufferShape, params.dmaBufferStrides, params.argumentShape, params.tileOffsets,
                             canonicalDmaShape, canonicalDmaStride, canonicalTileOffset);

    EXPECT_EQ(canonicalDmaShape, params.expectedDmaShape);
    EXPECT_EQ(canonicalDmaStride, params.expectedDmaStride);
    EXPECT_EQ(canonicalTileOffset, params.expectedOffsets);
}

INSTANTIATE_TEST_SUITE_P(DmaCanonicalizationTestSuite, DmaCanonicalizationTest,
                         testing::Values(DmaCanonicalizationTestParams{{1, 4, 1, 6},  // Unit expansion only
                                                                       {Bit(96), Bit(12), Bit(12), Bit(1)},
                                                                       {8, 12},
                                                                       {6, 0, 4, 0, 0, 0},
                                                                       {6, 4, 1, 1, 1, 1},
                                                                       {1, 12, 0, 0, 0, 0},
                                                                       {6, 4, 0, 0, 0, 0}},
                                         DmaCanonicalizationTestParams{{4, 6},  // Unit contraction
                                                                       {Bit(12), Bit(1)},
                                                                       {8, 1, 12, 1},
                                                                       {6, 4, 0, 0, 0, 0},
                                                                       {1, 6, 1, 4, 1, 1},
                                                                       {1, 1, 1, 12, 0, 0},
                                                                       {0, 6, 0, 4, 0, 0}},
                                         DmaCanonicalizationTestParams{{1, 4, 1, 6},  // Both expansion and contraction
                                                                       {Bit(96), Bit(12), Bit(12), Bit(1)},
                                                                       {8, 1, 12, 1},
                                                                       {6, 0, 4, 0, 0, 0},
                                                                       {1, 6, 1, 4, 1, 1},
                                                                       {1, 1, 12, 12, 0, 0},
                                                                       {0, 6, 0, 4, 0, 0}}));
}  // namespace
