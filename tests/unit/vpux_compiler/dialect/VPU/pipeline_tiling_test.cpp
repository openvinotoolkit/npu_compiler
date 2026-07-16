//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "temporal_tiling_test_utils.hpp"

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_PipelineTiling.*"

using namespace vpux;
using namespace vpux::VPU;
using namespace vpux::VPU::test;

// ============================================================================
// getName
// ============================================================================
TEST(MLIR_PipelineTiling, GetName) {
    PipelineTiling scenario;
    EXPECT_EQ(scenario.getName(), "PIPELINE");
}

// ============================================================================
// computeTileOverallCost: stall + max(max(dmaIn, compute), dmaOut)
// ============================================================================
class MLIR_PipelineTiling_CostFormula : public testing::TestWithParam<CostFormulaWithFlagsTestCase> {};

TEST_P(MLIR_PipelineTiling_CostFormula, CostFormulaIsDoubleBuffered) {
    const auto& tc = GetParam();
    TestablePipelineTiling scenario;
    const auto calculated = scenario.computeTileOverallCost(tc.stallCost, tc.dmaInCost, tc.computeCost, tc.dmaOutCost,
                                                            tc.isFirstTile, tc.isLastTile);
    EXPECT_EQ(calculated, tc.expectedResult);
}

INSTANTIATE_TEST_SUITE_P(PipelineCostCases, MLIR_PipelineTiling_CostFormula,
                         testing::Values(
                                 // All zero
                                 CostFormulaWithFlagsTestCase{0, 0, 0, 0, false, false, 0},
                                 // Compute-dominant: stall=0, max(max(200,300),100) = 300
                                 CostFormulaWithFlagsTestCase{0, 200, 300, 100, false, false, 300},
                                 // DMA-in dominant: stall=50, max(max(500,100),200) = 500 -> 550
                                 CostFormulaWithFlagsTestCase{50, 500, 100, 200, false, false, 550},
                                 // DMA-out dominant: stall=0, max(max(100,100),400) = 400
                                 CostFormulaWithFlagsTestCase{0, 100, 100, 400, false, false, 400},
                                 // Stall dominant: stall=1000, max(max(1,1),1) = 1 -> 1001
                                 CostFormulaWithFlagsTestCase{1000, 1, 1, 1, false, false, 1001},
                                 // All equal: stall=10, max(max(100,100),100) = 100 -> 110
                                 CostFormulaWithFlagsTestCase{10, 100, 100, 100, false, false, 110}));

// ============================================================================
// computeTileOverallCost: isFirstTile and isLastTile flags each select a
// distinct formula; middle tiles (both false) are already covered above
// ============================================================================
class MLIR_PipelineTiling_CostFormulaWithFlags : public testing::TestWithParam<CostFormulaWithFlagsTestCase> {};

TEST_P(MLIR_PipelineTiling_CostFormulaWithFlags, CostFormulaRespectsPositionFlags) {
    const auto& tc = GetParam();
    TestablePipelineTiling scenario;
    const auto calculated = scenario.computeTileOverallCost(tc.stallCost, tc.dmaInCost, tc.computeCost, tc.dmaOutCost,
                                                            tc.isFirstTile, tc.isLastTile);
    EXPECT_EQ(calculated, tc.expectedResult);
}

INSTANTIATE_TEST_SUITE_P(
        PipelineCostWithFlagsCases, MLIR_PipelineTiling_CostFormulaWithFlags,
        testing::Values(
                // isFirstTile=true: DMA-in is serial, compute and DMA-out compete
                // formula: stall + dmaIn + max(compute, dmaOut)
                CostFormulaWithFlagsTestCase{0, 200, 300, 100, true, false, 500},   // 0+200+max(300,100)=500
                CostFormulaWithFlagsTestCase{50, 200, 300, 400, true, false, 650},  // 50+200+max(300,400)=650
                CostFormulaWithFlagsTestCase{0, 100, 100, 100, true, false, 200},   // 0+100+max(100,100)=200
                // isLastTile=true: DMA-in and compute compete, DMA-out is serial
                // formula: stall + max(dmaIn, compute) + dmaOut
                CostFormulaWithFlagsTestCase{0, 200, 300, 100, false, true, 400},    // 0+max(200,300)+100=400
                CostFormulaWithFlagsTestCase{50, 200, 300, 400, false, true, 750},   // 50+max(200,300)+400=750
                CostFormulaWithFlagsTestCase{0, 100, 100, 100, false, true, 200}));  // 0+max(100,100)+100=200

// ============================================================================
// classifyOperandSize: Pipeline routes shared activation/weight to shared bucket
// ============================================================================
class MLIR_PipelineTiling_ClassifyOperand : public testing::TestWithParam<ClassifyOperandTestCase> {};

TEST_P(MLIR_PipelineTiling_ClassifyOperand, CorrectBucketRouting) {
    const auto& tc = GetParam();
    TestablePipelineTiling scenario;
    PeakMemorySizeBuckets buckets;
    scenario.classifyOperandSize(buckets, tc.inputSize, tc.operandIndex, tc.numOperands, tc.isActivationShared,
                                 tc.isWeightShared);
    EXPECT_EQ(buckets.shared.count(), tc.expectedSharedDelta);
    EXPECT_EQ(buckets.individual.count(), tc.expectedIndividualDelta);
    EXPECT_EQ(buckets.prefetch.count(), tc.expectedPrefetchDelta);
}

INSTANTIATE_TEST_SUITE_P(ClassifyCases, MLIR_PipelineTiling_ClassifyOperand,
                         testing::Values(
                                 // Activation shared -> shared bucket
                                 ClassifyOperandTestCase{Byte(1024), 0, 3, true, false, 1024, 0, 0},
                                 // Activation not shared -> individual bucket
                                 ClassifyOperandTestCase{Byte(1024), 0, 3, false, false, 0, 1024, 0},
                                 // Weight shared -> shared bucket
                                 ClassifyOperandTestCase{Byte(2048), 1, 3, false, true, 2048, 0, 0},
                                 // Weight not shared -> individual bucket
                                 ClassifyOperandTestCase{Byte(2048), 1, 3, false, false, 0, 2048, 0},
                                 // Output (idx=2, last) -> always individual
                                 ClassifyOperandTestCase{Byte(512), 2, 3, false, false, 0, 512, 0},
                                 ClassifyOperandTestCase{Byte(512), 2, 3, true, true, 0, 512, 0}));

// ============================================================================
// computeTilePeakMemory: individual*2 + shared
// Weight table/sparsity: routed to shared or individual based on flags
// ============================================================================
class MLIR_PipelineTiling_PeakMemory : public testing::TestWithParam<PeakMemoryTestCase> {};

TEST_P(MLIR_PipelineTiling_PeakMemory, PeakIsDoubleBuffered) {
    const auto& tc = GetParam();
    TestablePipelineTiling scenario;
    const auto result = scenario.computeTilePeakMemory(tc.buckets, tc.weightTableSize, tc.sparsityMapSize,
                                                       tc.isWeightShared, tc.isWeightTableShared);
    EXPECT_EQ(result, tc.expectedPeak);
}

INSTANTIATE_TEST_SUITE_P(
        PeakCases, MLIR_PipelineTiling_PeakMemory,
        testing::Values(
                // Pure individual: 4096*2 = 8192
                PeakMemoryTestCase{makeBuckets(0, 4096), Byte(0), Byte(0), false, false, Byte(8192)},
                // Shared only: shared = 1024, individual*2 = 0 -> 1024
                PeakMemoryTestCase{makeBuckets(1024, 0), Byte(0), Byte(0), false, false, Byte(1024)},
                // Mixed: shared=1024, individual=2048 -> 2048*2 + 1024 = 5120
                PeakMemoryTestCase{makeBuckets(1024, 2048), Byte(0), Byte(0), false, false, Byte(5120)},
                // Weight table shared: goes to shared bucket
                PeakMemoryTestCase{makeBuckets(0, 2048), Byte(256), Byte(0), false, true, Byte(2048 * 2 + 256)},
                // Weight table not shared: goes to individual -> (2048+256)*2 = 4608
                PeakMemoryTestCase{makeBuckets(0, 2048), Byte(256), Byte(0), false, false, Byte((2048 + 256) * 2)},
                // Sparsity map shared: goes to shared
                PeakMemoryTestCase{makeBuckets(0, 2048), Byte(0), Byte(128), true, false, Byte(2048 * 2 + 128)},
                // Sparsity map not shared: goes to individual -> (2048+128)*2 = 4352
                PeakMemoryTestCase{makeBuckets(0, 2048), Byte(0), Byte(128), false, false, Byte((2048 + 128) * 2)},
                // All overhead shared
                PeakMemoryTestCase{makeBuckets(512, 1024), Byte(256), Byte(128), true, true,
                                   Byte(1024 * 2 + 512 + 256 + 128)},
                // Zero everything
                PeakMemoryTestCase{makeBuckets(0, 0), Byte(0), Byte(0), false, false, Byte(0)}));

// ============================================================================
// getScheduleStrategy: PipelineTiling throws (TODO)
// ============================================================================
TEST(MLIR_PipelineTiling, GetScheduleStrategyThrows) {
    PipelineTiling scenario;
    ComputeRegion emptyRegion(std::make_unique<SchedulingLoop>());
    EXPECT_ANY_THROW(scenario.getScheduleStrategy(emptyRegion, 8192));
}

// ============================================================================
// classifyOperandSize accumulation with mixed sharing flags
// ============================================================================
TEST(MLIR_PipelineTiling, ClassifyOperandAccumulatesCorrectly) {
    TestablePipelineTiling scenario;
    PeakMemorySizeBuckets buckets;
    // Activation shared: goes to shared
    scenario.classifyOperandSize(buckets, Byte(1024), 0, 3, true, false);
    // Weight not shared: goes to individual
    scenario.classifyOperandSize(buckets, Byte(2048), 1, 3, false, false);
    // Output: goes to individual
    scenario.classifyOperandSize(buckets, Byte(512), 2, 3, false, false);
    EXPECT_EQ(buckets.shared, Byte(1024));
    EXPECT_EQ(buckets.individual, Byte(2048 + 512));
    EXPECT_EQ(buckets.prefetch, Byte(0));
}
