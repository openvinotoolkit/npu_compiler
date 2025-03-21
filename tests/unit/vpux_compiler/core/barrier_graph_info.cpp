//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_graph_info.hpp"

#include <gtest/gtest.h>

using namespace vpux;
using BarrierGraphInfoTests = ::testing::Test;

/**
 *
 *  DMA0      DMA1
 *    \        /
 *       bar0
 *    /        \
 *  DPU2      DMA3
 *    \        /
 *       bar1
 *    /    |    \
 *  SHV4  DPU5   DMA6
 *    \    |    /
 *       bar2
 *    /        \
 *  DPU7      DMA8
 *    \        /
 *       bar3
 *    /        \
 *  DPU9      DMA10
 *    \        /
 *       bar4
 *    /        \
 *  SHV11       DMA12
 */
std::pair<BarrierInfoMaps, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>> barrierMapWithMultiTaskQueueTypes() {
    BarrierInfoMaps inputBarrierMaps;
    inputBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {},   // task 1
            {0},  // task 2
            {0},  // task 3
            {1},  // task 4
            {1},  // task 5
            {1},  // task 6
            {2},  // task 7
            {2},  // task 8
            {3},  // task 9
            {3},  // task 10
            {4},  // task 11
            {4},  // task 12
    };

    inputBarrierMaps.barrierConsumerMap = {
            {2, 3},     // barrier 0
            {4, 5, 6},  // barrier 1
            {7, 8},     // barrier 2
            {9, 10},    // barrier 3
            {11, 12},   // barrier 4
    };

    inputBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {2},  // task 5
            {2},  // task 6
            {3},  // task 7
            {3},  // task 8
            {4},  // task 9
            {4},  // task 10
            {},   // task 11
            {},   // task 12
    };

    inputBarrierMaps.barrierProducerMap = {
            {0, 1},     // barrier 0
            {2, 3},     // barrier 1
            {4, 5, 6},  // barrier 2
            {7, 8},     // barrier 3
            {9, 10},    // barrier 4
    };

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_ACT, 0};
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeMap;
    taskQueueTypeMap[dmaType] = {0, 1, 3, 6, 8, 10, 12};
    taskQueueTypeMap[dpuType] = {2, 5, 7, 9};
    taskQueueTypeMap[shvType] = {4, 11};

    return std::make_pair(inputBarrierMaps, taskQueueTypeMap);
}

TEST_F(BarrierGraphInfoTests, CheckParentAndChildBarriers) {
    auto [barrierConfig, taskQueueTypeMap] = barrierMapWithMultiTaskQueueTypes();
    BarrierGraphInfoTest graphInfoTest(taskQueueTypeMap, barrierConfig);

    for (auto barInd : irange(barrierConfig.barrierProducerMap.size())) {
        BarrierGraphInfo::BarrierSet expectedParentBarriers, expectedChildrenBarriers;
        if (barInd != 0) {
            expectedParentBarriers.insert(barInd - 1);
        }
        if (barInd != barrierConfig.barrierProducerMap.size() - 1) {
            expectedChildrenBarriers.insert(barInd + 1);
        }
        auto parentBarriers = graphInfoTest.getParentBarrier(barInd);
        ASSERT_EQ(parentBarriers, expectedParentBarriers);
        auto childrenBarriers = graphInfoTest.getChildrenBarrier(barInd);
        ASSERT_EQ(childrenBarriers, expectedChildrenBarriers);
    }
}

TEST_F(BarrierGraphInfoTests, CheckTaskExecutionStep) {
    auto [barrierConfig, taskQueueTypeMap] = barrierMapWithMultiTaskQueueTypes();
    BarrierGraphInfoTest graphInfoTest(taskQueueTypeMap, barrierConfig);
    auto taskExecutionSteps = graphInfoTest.getExecutionStepTaskBatch();
    std::map<size_t, std::set<size_t>> expectedResults;
    expectedResults[0] = {0, 1};     // step 0 -> task [0, 1]
    expectedResults[1] = {2, 3};     // step 1 -> task [2, 3]
    expectedResults[2] = {4, 5, 6};  // step 2 -> task [4, 5, 6]
    expectedResults[3] = {7, 8};     // step 3 -> task [7, 8]
    expectedResults[4] = {9, 10};    // step 4 -> task [9, 10]
    expectedResults[5] = {11, 12};   // step 5 -> task [11, 12]
    ASSERT_EQ(expectedResults.size(), taskExecutionSteps.size());
    for (size_t i = 0; i < expectedResults.size(); i++) {
        ASSERT_EQ(expectedResults[i], std::set<size_t>(taskExecutionSteps[i].begin(), taskExecutionSteps[i].end()));
    }
}

TEST_F(BarrierGraphInfoTests, CheckBarrierLongestHopQueueType) {
    auto [barrierConfig, taskQueueTypeMap] = barrierMapWithMultiTaskQueueTypes();
    BarrierGraphInfoTest graphInfoTest(taskQueueTypeMap, barrierConfig);
    auto barrierLongestHopTaskType = graphInfoTest.getBarrierLongestQueueType();
    SmallVector<VPURT::TaskQueueType> expectedResults(5);
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_ACT, 0};

    expectedResults[0] = dpuType;  // bar0: DPU  DPU2 is the first op in DPU queue
    expectedResults[1] = shvType;  // bar1: SHV  SHV4-> bar2 -> bar3 -> bar4 -> SHV11
    expectedResults[2] = dpuType;  // bar2: DPU  DPU7 -> bar3 -> DPU9
    expectedResults[3] = dpuType;  // bar3: DPU  DPU7 -> bar3 -> DPU9
    expectedResults[4] = shvType;  // bar4: SHV  SHV4-> bar2 -> bar3 -> bar4 -> SHV11
    ASSERT_EQ(barrierLongestHopTaskType, expectedResults);
}

TEST_F(BarrierGraphInfoTests, CheckBarrierFirstAndLastExecutionStep) {
    auto [barrierConfig, taskQueueTypeMap] = barrierMapWithMultiTaskQueueTypes();
    BarrierGraphInfoTest graphInfoTest(taskQueueTypeMap, barrierConfig);
    auto firstExecutionStep = graphInfoTest.getBarrierFirstExecutionStep();
    auto lastExecutionStep = graphInfoTest.getBarrierLastExecutionStep();
    SmallVector<size_t> expectedFirstStep(barrierConfig.barrierProducerMap.size());
    SmallVector<size_t> expectedLastStep(barrierConfig.barrierProducerMap.size());
    for (auto barInd : irange(barrierConfig.barrierProducerMap.size())) {
        // first step is equal to its producer's min execution step
        expectedFirstStep[barInd] = barInd;
        // last step is equal to its consumer's max execution step
        expectedLastStep[barInd] = barInd + 1;
    }
    ASSERT_EQ(firstExecutionStep, expectedFirstStep);
    ASSERT_EQ(lastExecutionStep, expectedLastStep);
}
