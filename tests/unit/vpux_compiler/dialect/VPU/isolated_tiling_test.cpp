//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "temporal_tiling_test_utils.hpp"

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_IsolatedTiling.*"

using namespace vpux;
using namespace vpux::VPU;
using namespace vpux::VPU::test;

// ============================================================================
// getName
// ============================================================================
TEST(MLIR_IsolatedTiling, GetName) {
    IsolatedTiling scenario;
    EXPECT_EQ(scenario.getName(), "ISOLATED");
}

// ============================================================================
// computeTileOverallCost: stall + dmaIn + compute + dmaOut (serial sum)
// ============================================================================
class MLIR_IsolatedTiling_CostFormula : public testing::TestWithParam<CostFormulaWithFlagsTestCase> {};

TEST_P(MLIR_IsolatedTiling_CostFormula, CostFormulaIsSerialSum) {
    const auto& tc = GetParam();
    TestableIsolatedTiling scenario;
    const auto result = scenario.computeTileOverallCost(tc.stallCost, tc.dmaInCost, tc.computeCost, tc.dmaOutCost,
                                                        tc.isFirstTile, tc.isLastTile);
    EXPECT_EQ(result, tc.expectedResult);
}

INSTANTIATE_TEST_SUITE_P(SerialCostCases, MLIR_IsolatedTiling_CostFormula,
                         testing::Values(CostFormulaWithFlagsTestCase{0, 0, 0, 0, false, false, 0},
                                         CostFormulaWithFlagsTestCase{100, 200, 300, 400, false, false, 1000},
                                         CostFormulaWithFlagsTestCase{1, 1, 1, 1, false, false, 4},
                                         CostFormulaWithFlagsTestCase{0, 500, 0, 500, false, false, 1000},
                                         CostFormulaWithFlagsTestCase{1000, 0, 0, 0, false, false, 1000}));

// ============================================================================
// classifyOperandSize: shared activation/weight go to shared bucket;
// all other operands go to individual
// ============================================================================
class MLIR_IsolatedTiling_ClassifyOperand : public testing::TestWithParam<ClassifyOperandTestCase> {};

TEST_P(MLIR_IsolatedTiling_ClassifyOperand, AllGoesToIndividual) {
    const auto& tc = GetParam();
    TestableIsolatedTiling scenario;
    PeakMemorySizeBuckets buckets;
    scenario.classifyOperandSize(buckets, tc.inputSize, tc.operandIndex, tc.numOperands, tc.isActivationShared,
                                 tc.isWeightShared);
    EXPECT_EQ(buckets.shared.count(), tc.expectedSharedDelta);
    EXPECT_EQ(buckets.individual.count(), tc.expectedIndividualDelta);
    EXPECT_EQ(buckets.prefetch.count(), tc.expectedPrefetchDelta);
}

INSTANTIATE_TEST_SUITE_P(ClassifyCases, MLIR_IsolatedTiling_ClassifyOperand,
                         testing::Values(
                                 // Activation shared -> shared bucket
                                 ClassifyOperandTestCase{Byte(1024), 0, 3, true, false, 1024, 0, 0},
                                 // Activation not shared -> individual
                                 ClassifyOperandTestCase{Byte(1024), 0, 3, false, false, 0, 1024, 0},
                                 // Weight shared -> shared bucket
                                 ClassifyOperandTestCase{Byte(2048), 1, 3, false, true, 2048, 0, 0},
                                 // Weight not shared -> individual
                                 ClassifyOperandTestCase{Byte(2048), 1, 3, false, false, 0, 2048, 0},
                                 // Output operand (idx=2, last) -> always individual
                                 ClassifyOperandTestCase{Byte(512), 2, 3, false, false, 0, 512, 0},
                                 // Single operand, not shared -> individual
                                 ClassifyOperandTestCase{Byte(256), 0, 1, false, false, 0, 256, 0}));

// ============================================================================
// computeTilePeakMemory: individual + shared + overhead (weightTable/sparsityMap
// routed to shared or individual based on flags; no prefetch bucket)
// ============================================================================
class MLIR_IsolatedTiling_PeakMemory : public testing::TestWithParam<PeakMemoryTestCase> {};

TEST_P(MLIR_IsolatedTiling_PeakMemory, PeakIsIndividualPlusOverhead) {
    const auto& tc = GetParam();
    TestableIsolatedTiling scenario;
    const auto result = scenario.computeTilePeakMemory(tc.buckets, tc.weightTableSize, tc.sparsityMapSize,
                                                       tc.isWeightShared, tc.isWeightTableShared);
    EXPECT_EQ(result, tc.expectedPeak);
}

INSTANTIATE_TEST_SUITE_P(
        PeakCases, MLIR_IsolatedTiling_PeakMemory,
        testing::Values(
                // Pure individual, no overhead
                PeakMemoryTestCase{makeBuckets(0, 4096), Byte(0), Byte(0), false, false, Byte(4096)},
                // Individual + weight table + sparsity map
                PeakMemoryTestCase{makeBuckets(0, 4096), Byte(256), Byte(128), false, false, Byte(4096 + 256 + 128)},
                // Shared + individual: both contribute to peak
                PeakMemoryTestCase{makeBuckets(1000, 2048), Byte(0), Byte(0), false, false, Byte(3048)},
                // Zero everything
                PeakMemoryTestCase{makeBuckets(0, 0), Byte(0), Byte(0), false, false, Byte(0)}));

// ============================================================================
// getScheduleStrategy: IsolatedTiling throws (TODO, to remove when implemented)
// ============================================================================
TEST(MLIR_IsolatedTiling, GetScheduleStrategyThrows) {
    IsolatedTiling scenario;
    ComputeRegion emptyRegion(std::make_unique<SchedulingLoop>());
    EXPECT_ANY_THROW(scenario.getScheduleStrategy(emptyRegion, 8192));
}

// ============================================================================
// Multiple classifyOperandSize calls accumulate in individual bucket
// ============================================================================
TEST(MLIR_IsolatedTiling, ClassifyOperandSizeAccumulates) {
    TestableIsolatedTiling scenario;
    PeakMemorySizeBuckets buckets;
    // Simulate 3-operand op: activation + weights + output
    scenario.classifyOperandSize(buckets, Byte(1024), 0, 3, false, false);
    scenario.classifyOperandSize(buckets, Byte(2048), 1, 3, false, false);
    scenario.classifyOperandSize(buckets, Byte(512), 2, 3, false, false);
    EXPECT_EQ(buckets.individual, Byte(1024 + 2048 + 512));
    EXPECT_EQ(buckets.shared, Byte(0));
    EXPECT_EQ(buckets.prefetch, Byte(0));
}
