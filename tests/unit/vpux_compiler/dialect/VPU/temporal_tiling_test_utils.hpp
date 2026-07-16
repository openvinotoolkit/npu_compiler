//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/isolated_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/prefetch_tiling.hpp"

#include <gtest/gtest.h>

namespace vpux::VPU::test {

// Helper to create a PeakMemorySizeBuckets with specified values
inline PeakMemorySizeBuckets makeBuckets(int64_t shared, int64_t individual, int64_t prefetch = 0) {
    return {Byte(shared), Byte(individual), Byte(prefetch)};
}

// Thin test wrappers exposing protected virtual methods for unit testing.
// These delegate directly to the real implementations.

class TestableIsolatedTiling : public IsolatedTiling {
public:
    using IsolatedTiling::classifyOperandSize;
    using IsolatedTiling::computeTileOverallCost;
    using IsolatedTiling::computeTilePeakMemory;
};

class TestablePipelineTiling : public PipelineTiling {
public:
    using PipelineTiling::classifyOperandSize;
    using PipelineTiling::computeTileOverallCost;
    using PipelineTiling::computeTilePeakMemory;
};

class TestablePrefetchTiling : public PrefetchTiling {
public:
    using PrefetchTiling::classifyOperandSize;
    using PrefetchTiling::computeTileOverallCost;
    using PrefetchTiling::computeTilePeakMemory;
};

// Parameterized cost formula test case with explicit tile-position flags
struct CostFormulaWithFlagsTestCase {
    uint64_t stallCost;
    uint64_t dmaInCost;
    uint64_t computeCost;
    uint64_t dmaOutCost;
    bool isFirstTile;
    bool isLastTile;
    uint64_t expectedResult;
};

// Parameterized operand classification test case
struct ClassifyOperandTestCase {
    Byte inputSize;
    size_t operandIndex;
    size_t numOperands;
    bool isActivationShared;
    bool isWeightShared;
    // expected bucket deltas after classifyOperandSize
    int64_t expectedSharedDelta;
    int64_t expectedIndividualDelta;
    int64_t expectedPrefetchDelta;
};

// Parameterized peak memory test case
struct PeakMemoryTestCase {
    PeakMemorySizeBuckets buckets;
    Byte weightTableSize;
    Byte sparsityMapSize;
    bool isWeightShared;
    bool isWeightTableShared;
    Byte expectedPeak;
};

}  // namespace vpux::VPU::test
