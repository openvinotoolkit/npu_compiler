//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

using namespace vpux;
using ExecutionGroupAnalysisTests = ::testing::Test;

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
std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> buildBarrierMapWithMultiTaskQueueTypes() {
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_ACT, 0};
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeMap;

    taskQueueTypeMap[dpuType] = {2, 5, 7, 9};
    taskQueueTypeMap[shvType] = {4, 11};
    return taskQueueTypeMap;
}

std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> buildBarrierMapWithMultiTaskQueueTypesMultiTile() {
    // Creating TaskQueueTypes with different IDs to simulate different tiles
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeMap;

    // DPU tasks across multiple tiles (different TaskQueueType IDs for different tiles)
    taskQueueTypeMap[{VPU::ExecutorKind::DPU, 0}] = {1, 2, 3, 4};      // Tile 0
    taskQueueTypeMap[{VPU::ExecutorKind::DPU, 1}] = {5, 6, 7, 8};      // Tile 1
    taskQueueTypeMap[{VPU::ExecutorKind::DPU, 2}] = {15, 16, 17, 18};  // Tile 2

    // SHAVE tasks across multiple tiles
    taskQueueTypeMap[{VPU::ExecutorKind::SHAVE_ACT, 0}] = {9, 10};   // Tile 0
    taskQueueTypeMap[{VPU::ExecutorKind::SHAVE_ACT, 1}] = {11, 12};  // Tile 1
    taskQueueTypeMap[{VPU::ExecutorKind::SHAVE_ACT, 2}] = {13, 14};  // Tile 2

    return taskQueueTypeMap;
}

TEST_F(ExecutionGroupAnalysisTests, CheckDPUGroups) {
    auto taskQueueTypeMap = buildBarrierMapWithMultiTaskQueueTypes();

    ExecutionGroupAnalysisTest execGroupAnalysisTest(taskQueueTypeMap, /*maxVariantCount*/ 4, /*maxInvariantCount*/ 2,
                                                     /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                     /*tilesCount*/ 1);

    auto dpuExecGroups = execGroupAnalysisTest.getDPUExecutionGroups();
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    auto execGroupLists = dpuExecGroups[dpuType];
    ExecutionGroupList expectedDPUList = {{2, 5}, {7, 9}};

    // Check if there are 2 groups
    ASSERT_EQ(execGroupLists.size(), expectedDPUList.size());

    // Check if each group has same number of tasks
    ASSERT_EQ(execGroupLists[0].size(), expectedDPUList[0].size());

    // Check if tasks are in same order
    for (size_t i = 0; i < expectedDPUList.size(); ++i) {
        for (size_t j = 0; j < expectedDPUList[i].size(); ++j) {
            ASSERT_EQ(execGroupLists[i][j], expectedDPUList[i][j]);
        }
    }
}

TEST_F(ExecutionGroupAnalysisTests, CheckSWGroups) {
    auto taskQueueTypeMap = buildBarrierMapWithMultiTaskQueueTypes();
    ExecutionGroupAnalysisTest execGroupAnalysisTest(taskQueueTypeMap, /*maxVariantCount*/ 4, /*maxInvariantCount*/ 2,
                                                     /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                     /*tilesCount*/ 1);

    auto swGroups = execGroupAnalysisTest.getActShvExecutionGroups();
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_ACT, 0};

    auto execGroupLists = swGroups[shvType];
    ExecutionGroupList expectedSWList = {{4, 11}};

    // Check if there is 1 group
    ASSERT_EQ(execGroupLists.size(), expectedSWList.size());

    // Check if each group has same number of tasks
    ASSERT_EQ(execGroupLists[0].size(), expectedSWList[0].size());

    // Check if tasks are in same order
    for (size_t i = 0; i < expectedSWList.size(); ++i) {
        for (size_t j = 0; j < expectedSWList[i].size(); ++j) {
            ASSERT_EQ(execGroupLists[i][j], expectedSWList[i][j]);
        }
    }
}

TEST_F(ExecutionGroupAnalysisTests, CheckAllTasksGrouped) {
    auto taskQueueTypeMap = buildBarrierMapWithMultiTaskQueueTypesMultiTile();

    ExecutionGroupAnalysisTest execGroupAnalysisTest(taskQueueTypeMap, /*maxVariantCount*/ 4, /*maxInvariantCount*/ 2,
                                                     /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                     /*tilesCount*/ 3);

    // Collect all tasks from the execution groups, grouped by queue type
    std::map<VPURT::TaskQueueType, SmallVector<size_t>> groupedTasksByQueueType;

    // Collect tasks from DPU execution groups
    for (auto& [queueType, execGroupLists] : execGroupAnalysisTest.getDPUExecutionGroups()) {
        for (auto& group : execGroupLists) {
            groupedTasksByQueueType[queueType].insert(groupedTasksByQueueType[queueType].end(), group.begin(),
                                                      group.end());
        }
    }

    // Collect tasks from SHAVE_ACT execution groups
    for (auto& [queueType, execGroupLists] : execGroupAnalysisTest.getActShvExecutionGroups()) {
        for (auto& group : execGroupLists) {
            groupedTasksByQueueType[queueType].insert(groupedTasksByQueueType[queueType].end(), group.begin(),
                                                      group.end());
        }
    }

    // Collect all tasks from the taskQueueTypeMap, grouped by queue type
    std::map<VPURT::TaskQueueType, SmallVector<size_t>> expectedTasksByQueueType;
    for (auto& [queueType, taskList] : taskQueueTypeMap) {
        expectedTasksByQueueType[queueType] = SmallVector<size_t>(taskList.begin(), taskList.end());
    }

    // Check if tasks are correctly grouped for DPU and SHAVE_ACT queue types
    for (auto& [queueType, expectedTasks] : expectedTasksByQueueType) {
        if (queueType.type == VPU::ExecutorKind::DPU || queueType.type == VPU::ExecutorKind::SHAVE_ACT) {
            auto& groupedTasks = groupedTasksByQueueType[queueType];

            // Sort both lists for comparison
            std::sort(groupedTasks.begin(), groupedTasks.end());
            std::sort(expectedTasks.begin(), expectedTasks.end());

            // Check if the tasks are grouped correctly for the given queue type
            ASSERT_EQ(groupedTasks, expectedTasks);
        }
    }
}

TEST_F(ExecutionGroupAnalysisTests, CheckGroupIndexForTask) {
    auto taskQueueTypeMap = buildBarrierMapWithMultiTaskQueueTypesMultiTile();

    ExecutionGroupAnalysisTest execGroupAnalysisTest(taskQueueTypeMap, /*maxVariantCount*/ 4, /*maxInvariantCount*/ 2,
                                                     /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                     /*tilesCount*/ 3);

    // Verify group indices for tasks in DPU execution groups
    for (auto& [queueType, execGroupLists] : execGroupAnalysisTest.getDPUExecutionGroups()) {
        for (size_t groupIdx = 0; groupIdx < execGroupLists.size(); ++groupIdx) {
            const auto& group = execGroupLists[groupIdx];
            for (const auto& taskIdx : group) {
                auto result = execGroupAnalysisTest.getGroupIndexForTask(taskIdx, queueType);
                ASSERT_TRUE(result.has_value());
                ASSERT_EQ(result.value(), groupIdx);
            }
        }
    }

    // Verify group indices for tasks in SHAVE_ACT execution groups
    for (auto& [queueType, execGroupLists] : execGroupAnalysisTest.getActShvExecutionGroups()) {
        for (size_t groupIdx = 0; groupIdx < execGroupLists.size(); ++groupIdx) {
            const auto& group = execGroupLists[groupIdx];
            for (const auto& taskIdx : group) {
                auto result = execGroupAnalysisTest.getGroupIndexForTask(taskIdx, queueType);
                ASSERT_TRUE(result.has_value());
                ASSERT_EQ(result.value(), groupIdx);
            }
        }
    }

    // Verify tasks not in any group return nullopt
    size_t nonExistentTaskIdx = 99999;  // Arbitrary value not in any group
    auto result = execGroupAnalysisTest.getGroupIndexForTask(nonExistentTaskIdx);
    ASSERT_FALSE(result.has_value());
}
