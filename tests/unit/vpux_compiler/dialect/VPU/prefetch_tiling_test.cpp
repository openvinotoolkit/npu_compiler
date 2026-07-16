//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "temporal_tiling_test_utils.hpp"

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_PrefetchTiling.*"

using namespace vpux;
using namespace vpux::VPU;
using namespace vpux::VPU::test;

// ============================================================================
// getName
// ============================================================================
TEST(MLIR_PrefetchTiling, GetName) {
    PrefetchTiling scenario;
    EXPECT_EQ(scenario.getName(), "PREFETCH");
}

// ============================================================================
// computeTileOverallCost: stall + max(dmaIn, compute) + dmaOut
// ============================================================================
class MLIR_PrefetchTiling_CostFormula : public testing::TestWithParam<CostFormulaWithFlagsTestCase> {};

TEST_P(MLIR_PrefetchTiling_CostFormula, CostFormulaIsPrefetch) {
    const auto& tc = GetParam();
    TestablePrefetchTiling scenario;
    const auto calculated = scenario.computeTileOverallCost(tc.stallCost, tc.dmaInCost, tc.computeCost, tc.dmaOutCost,
                                                            tc.isFirstTile, tc.isLastTile);
    EXPECT_EQ(calculated, tc.expectedResult);
}

INSTANTIATE_TEST_SUITE_P(PrefetchCostCases, MLIR_PrefetchTiling_CostFormula,
                         testing::Values(
                                 // All zero
                                 CostFormulaWithFlagsTestCase{0, 0, 0, 0, false, false, 0},
                                 // Compute-dominant: 0 + max(200,300) + 100 = 400
                                 CostFormulaWithFlagsTestCase{0, 200, 300, 100, false, false, 400},
                                 // DMA-in dominant: 50 + max(500,100) + 200 = 750
                                 CostFormulaWithFlagsTestCase{50, 500, 100, 200, false, false, 750},
                                 // DMA-out additive: 0 + max(100,100) + 400 = 500
                                 CostFormulaWithFlagsTestCase{0, 100, 100, 400, false, false, 500},
                                 // Stall dominant: 1000 + max(1,1) + 1 = 1002
                                 CostFormulaWithFlagsTestCase{1000, 1, 1, 1, false, false, 1002},
                                 // All equal: 10 + max(100,100) + 100 = 210
                                 CostFormulaWithFlagsTestCase{10, 100, 100, 100, false, false, 210}));

// ============================================================================
// computeTileOverallCost: isFirstTile=true collapses to serial sum;
// isLastTile is always ignored
// ============================================================================
class MLIR_PrefetchTiling_CostFormulaWithFlags : public testing::TestWithParam<CostFormulaWithFlagsTestCase> {};

TEST_P(MLIR_PrefetchTiling_CostFormulaWithFlags, CostFormulaRespectsFirstTileFlag) {
    const auto& tc = GetParam();
    TestablePrefetchTiling scenario;
    const auto calculated = scenario.computeTileOverallCost(tc.stallCost, tc.dmaInCost, tc.computeCost, tc.dmaOutCost,
                                                            tc.isFirstTile, tc.isLastTile);
    EXPECT_EQ(calculated, tc.expectedResult);
}

INSTANTIATE_TEST_SUITE_P(PrefetchCostWithFlagsCases, MLIR_PrefetchTiling_CostFormulaWithFlags,
                         testing::Values(
                                 // isFirstTile=true: serial sum stall+dmaIn+compute+dmaOut
                                 CostFormulaWithFlagsTestCase{0, 200, 300, 100, true, false, 600},
                                 CostFormulaWithFlagsTestCase{50, 500, 100, 200, true, false, 850},
                                 CostFormulaWithFlagsTestCase{10, 100, 100, 100, true, false, 310},
                                 // isLastTile flag is ignored: same result as {false, false}
                                 CostFormulaWithFlagsTestCase{0, 200, 300, 100, false, true, 400},
                                 CostFormulaWithFlagsTestCase{50, 500, 100, 200, false, true, 750},
                                 // isFirstTile=true takes priority over isLastTile=true
                                 CostFormulaWithFlagsTestCase{0, 200, 300, 100, true, true, 600}));

// ============================================================================
// classifyOperandSize: Prefetch routes shared to shared bucket,
// non-shared inputs to both prefetch AND individual
// ============================================================================
class MLIR_PrefetchTiling_ClassifyOperand : public testing::TestWithParam<ClassifyOperandTestCase> {};

TEST_P(MLIR_PrefetchTiling_ClassifyOperand, CorrectBucketRouting) {
    const auto& tc = GetParam();
    TestablePrefetchTiling scenario;
    PeakMemorySizeBuckets buckets;
    scenario.classifyOperandSize(buckets, tc.inputSize, tc.operandIndex, tc.numOperands, tc.isActivationShared,
                                 tc.isWeightShared);
    EXPECT_EQ(buckets.shared.count(), tc.expectedSharedDelta);
    EXPECT_EQ(buckets.individual.count(), tc.expectedIndividualDelta);
    EXPECT_EQ(buckets.prefetch.count(), tc.expectedPrefetchDelta);
}

INSTANTIATE_TEST_SUITE_P(ClassifyCases, MLIR_PrefetchTiling_ClassifyOperand,
                         testing::Values(
                                 // Activation shared -> shared bucket only
                                 ClassifyOperandTestCase{Byte(1024), 0, 3, true, false, 1024, 0, 0},
                                 // Activation not shared -> individual + prefetch (operandIndex < numOperands-1)
                                 ClassifyOperandTestCase{Byte(1024), 0, 3, false, false, 0, 1024, 1024},
                                 // Weight shared -> shared bucket only
                                 ClassifyOperandTestCase{Byte(2048), 1, 3, false, true, 2048, 0, 0},
                                 // Weight not shared -> individual + prefetch (operandIndex=1 < 2=numOperands-1)
                                 ClassifyOperandTestCase{Byte(2048), 1, 3, false, false, 0, 2048, 2048},
                                 // Output (idx=2, last of 3) -> individual only (operandIndex NOT < numOperands-1)
                                 ClassifyOperandTestCase{Byte(512), 2, 3, false, false, 0, 512, 0},
                                 ClassifyOperandTestCase{Byte(512), 2, 3, true, true, 0, 512, 0},
                                 // 2-operand op: idx=0, not shared -> individual + prefetch
                                 ClassifyOperandTestCase{Byte(1024), 0, 2, false, false, 0, 1024, 1024},
                                 // 2-operand op: idx=1 (last) -> individual only
                                 ClassifyOperandTestCase{Byte(512), 1, 2, false, false, 0, 512, 0}));

// ============================================================================
// computeTilePeakMemory: individual + shared + prefetch
// Weight table/sparsity: routed to shared or individual based on flags
// ============================================================================
class MLIR_PrefetchTiling_PeakMemory : public testing::TestWithParam<PeakMemoryTestCase> {};

TEST_P(MLIR_PrefetchTiling_PeakMemory, PeakIsPrefetchFormula) {
    const auto& tc = GetParam();
    TestablePrefetchTiling scenario;
    const auto result = scenario.computeTilePeakMemory(tc.buckets, tc.weightTableSize, tc.sparsityMapSize,
                                                       tc.isWeightShared, tc.isWeightTableShared);
    EXPECT_EQ(result, tc.expectedPeak);
}

INSTANTIATE_TEST_SUITE_P(
        PeakCases, MLIR_PrefetchTiling_PeakMemory,
        testing::Values(
                // Pure individual, no prefetch: 4096 + 0 + 0 = 4096
                PeakMemoryTestCase{makeBuckets(0, 4096, 0), Byte(0), Byte(0), false, false, Byte(4096)},
                // Individual + prefetch: 2048 + 0 + 1024 = 3072
                PeakMemoryTestCase{makeBuckets(0, 2048, 1024), Byte(0), Byte(0), false, false, Byte(3072)},
                // All three buckets: 512 + 2048 + 1024 = 3584
                PeakMemoryTestCase{makeBuckets(512, 2048, 1024), Byte(0), Byte(0), false, false, Byte(3584)},
                // Weight table shared -> shared
                PeakMemoryTestCase{makeBuckets(0, 2048, 512), Byte(256), Byte(0), false, true, Byte(2048 + 256 + 512)},
                // Weight table not shared -> individual
                PeakMemoryTestCase{makeBuckets(0, 2048, 512), Byte(256), Byte(0), false, false, Byte(2048 + 256 + 512)},
                // Sparsity map shared -> shared
                PeakMemoryTestCase{makeBuckets(0, 2048, 512), Byte(0), Byte(128), true, false, Byte(2048 + 128 + 512)},
                // Sparsity map not shared -> individual
                PeakMemoryTestCase{makeBuckets(0, 2048, 512), Byte(0), Byte(128), false, false, Byte(2048 + 128 + 512)},
                // All overhead shared
                PeakMemoryTestCase{makeBuckets(512, 1024, 256), Byte(256), Byte(128), true, true,
                                   Byte(1024 + 512 + 256 + 256 + 128)},
                // Zero everything
                PeakMemoryTestCase{makeBuckets(0, 0, 0), Byte(0), Byte(0), false, false, Byte(0)}));

// ============================================================================
// getScheduleStrategy: PrefetchTiling throws (TODO)
// ============================================================================
TEST(MLIR_PrefetchTiling, GetScheduleStrategyThrows) {
    PrefetchTiling scenario;
    ComputeRegion emptyRegion(std::make_unique<SchedulingLoop>());
    EXPECT_ANY_THROW(scenario.getScheduleStrategy(emptyRegion, 8192));
}

// ============================================================================
// classifyOperandSize accumulation: mixed sharing in a 3-operand op
// ============================================================================
TEST(MLIR_PrefetchTiling, ClassifyOperandAccumulatesCorrectly) {
    TestablePrefetchTiling scenario;
    PeakMemorySizeBuckets buckets;
    // Activation shared: goes to shared only
    scenario.classifyOperandSize(buckets, Byte(1024), 0, 3, true, false);
    // Weight not shared (idx=1 < numOperands-1=2): individual + prefetch
    scenario.classifyOperandSize(buckets, Byte(2048), 1, 3, false, false);
    // Output (idx=2, last): individual only
    scenario.classifyOperandSize(buckets, Byte(512), 2, 3, false, false);
    EXPECT_EQ(buckets.shared, Byte(1024));
    EXPECT_EQ(buckets.individual, Byte(2048 + 512));
    EXPECT_EQ(buckets.prefetch, Byte(2048));
}

// ============================================================================
// Prefetch vs Pipeline: prefetch adds prefetch bucket, pipeline does not
// ============================================================================
TEST(MLIR_PrefetchTiling, PrefetchBucketDiffersFromPipeline) {
    TestablePrefetchTiling prefetch;
    TestablePipelineTiling pipeline;
    PeakMemorySizeBuckets prefetchBuckets;
    PeakMemorySizeBuckets pipelineBuckets;

    // Non-shared weight operand (idx=1, 3 operands)
    prefetch.classifyOperandSize(prefetchBuckets, Byte(2048), 1, 3, false, false);
    pipeline.classifyOperandSize(pipelineBuckets, Byte(2048), 1, 3, false, false);

    // Both put it in individual
    EXPECT_EQ(prefetchBuckets.individual, pipelineBuckets.individual);
    // Only prefetch puts it in the prefetch bucket
    EXPECT_EQ(prefetchBuckets.prefetch, Byte(2048));
    EXPECT_EQ(pipelineBuckets.prefetch, Byte(0));
}
