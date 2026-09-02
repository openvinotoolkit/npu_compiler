//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/execution_group_analysis.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <gtest/gtest.h>
#include <algorithm>

using namespace vpux;

namespace {

using BarrierPagesSplitTests = ::testing::Test;

/**
 * HW FIFO (DMA): t0 t1 t3
 * HW FIFO (DPU): t2 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1  t2
 *          |  \ |
 *          b1  \|
 *          |    |
 * ------   t3   |
 *          |    b2
 *          |    |
 * Page1    |    t4
 *            \  |
 *              b3
 *               |
 * ------       t5
 *               |
 *              b4
 * Page2         |
 *              t6
 *               |
 *              b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphSimple() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {5}      // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {5}      // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckSplitForGraphSimple) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphSimple();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    ASSERT_TRUE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());
    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);

    ASSERT_EQ(page0BoundaryTasks.size(), 3);
    EXPECT_EQ(page0BoundaryTasks[0], 1);
    EXPECT_EQ(page0BoundaryTasks[1], 2);
    EXPECT_EQ(page0BoundaryTasks[2], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2
 * HW FIFO (DPU): t1 t3 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1  t2  <- Long dep: t2->b4. Need to be legalized to t2->b3
 *          |    |
 *          b1   |
 *          |    |
 * ------   t3   |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t4   |
 *          |    |
 *          b3   |
 *          |    |
 * ------   t5   |
 *           \  /
 *            b4
 * Page2      |
 *            t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLongDepOnTaskUpdateBarrierSide() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {4},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {3},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskUpdateBarrierSide) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithLongDepOnTaskUpdateBarrierSide();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 2 that requires legalization
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 2);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2
 * HW FIFO (DPU): t1 t3 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1  t2  <- Long dep: t2->b4. Need to be legalized to t2->b3
 *          |    |
 *          b1   |
 *          |    |
 * ------   t3   |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t4   |
 *          |    |
 *          b3   |
 *        / |    |
 * ------ | t5   | <- t5 does not update any barrier from next page but since
 *        |     /     there is no such task in the graph, that works on Page1/Page2
 *         \  b4      boundary, page split legalization, which requires at least one boundary task,
 * Page2    \ |       will create t5->b4 dependency
 *            t6   <- long dep: b3->t6. Need to be legalized. It is redundant due to b3->t5->b4->t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLongDepAndNoBoundaryTaskInOnePage() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {4},  // task 2
            {2},  // task 3
            {3},  // task 4
            {},   // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},     // task 0
            {0},    // task 1
            {0},    // task 2
            {1},    // task 3
            {2},    // task 4
            {3},    // task 5
            {3, 4}  // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {3},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepAndNoBoundaryTaskInOnePage) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithLongDepAndNoBoundaryTaskInOnePage();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 2 and 6 that requires legalization
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 2);
    EXPECT_EQ(tasksToLegalize[0], 2);
    EXPECT_EQ(tasksToLegalize[1], 6);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t7
 * HW FIFO (DPU): t1 t3 t4 t5 t6
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   t2   <-  t2->b4 need to be legalized by connecting t2 to b3
 *          |      |
 *          b1     |
 *          | \    |
 * ------   t3 \   |
 *          |   |  |
 *          b2  |  |
 * Page1    |   |  |
 *          t4  |  |
 *          |   |  |
 *          b3  |  |
 *          |   |  |
 *          t5  |  |
 *             /   |
 *            /    |
 *           /     |
 * ------   t6    /    <- b1->t6 needs to be removed
 *           \   /
 *            b4
 * Page2      |
 *            t7
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskUpdateBarrierSideBoundaryTaskDifferentBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {4},  // task 2
            {2},  // task 3
            {3},  // task 4
            {},   // task 5
            {4},  // task 6
            {5}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {1},  // task 6
            {4}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 7};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {3},  // task 2
            {2},  // task 3
            {3},  // task 4
            {},   // task 5
            {4},  // task 6
            {5}   // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {},   // task 6
            {4}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskUpdateBarrierSideBoundaryTaskDifferentBarrier) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskUpdateBarrierSideBoundaryTaskDifferentBarrier();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 6);

    // Check that it is taskInd = 2 that requires legalization
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 2);
    EXPECT_EQ(tasksToLegalize[0], 2);
    EXPECT_EQ(tasksToLegalize[1], 6);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t5
 * HW FIFO (DPU): t1 t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0   <- legalize b0->t4 dep - remove it
 *           /  \
 * Page0    t1   |
 *          |    |
 *          b1   |
 *          |    |
 * ------   t2   |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t3   |
 *         /    /
 *        |    /
 *        |  t4
 *        \  |
 *          b3
 *          |
 * ------   t5
 *  Page2   |
 *          b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLongDepOnTaskWaitBarrierSide() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {0},  // task 4
            {3}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {},   // task 4
            {3}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSide) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithLongDepOnTaskWaitBarrierSide();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 1);
    EXPECT_EQ(page0BoundaryTasks[0], 2);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 4);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t5
 * HW FIFO (DPU): t1 t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   |
 *          |    |
 *          b1   |     <- b1->t5 dependency needs to be legalized - removed and
 *          | \  |        replaced with t1->b3 dependency
 * ------   t2 | t3
 *          |  | |
 * Page1    b2 | b3
 *          |   \|
 * ------   t4   t5
 * Page2     \  /
 *            b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLongDepOnTaskWaitBarrierSideWith2WaitBarriers() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},     // task 0
            {0},    // task 1
            {1},    // task 2
            {0},    // task 3
            {2},    // task 4
            {1, 3}  // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 3},  // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {4}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {0},  // task 3
            {2},  // task 4
            {3}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWith2WaitBarriers) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWith2WaitBarriers();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 2);
    EXPECT_EQ(page1BoundaryTasks[0], 4);
    EXPECT_EQ(page1BoundaryTasks[1], 5);

    // Check that it is taskInd = 5 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 5);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t1 t5
 * HW FIFO (DPU): t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t2   t1
 *          |    |
 *          |    b1   <- b1->t4 dependency needs to be legalize by creating
 *          |    |       t1->b2 dependency
 * ------   |    |
 *          |    |
 *          b2   |
 * Page1    |    |
 *          t3   |
 *         /    /
 *        |    /
 *        |  t4
 *        \  |
 *          b3
 *          |
 * ------   t5
 *  Page2   |
 *          b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskWaitBarrierSideWithBoundaryTaskBarrierShared() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {2},  // task 3
            {1},  // task 4
            {3}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {2},  // task 3
            {},   // task 4
            {3}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWithBoundaryTaskBarrierShared) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWithBoundaryTaskBarrierShared();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 1);
    EXPECT_EQ(page0BoundaryTasks[0], 2);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 4);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2
 * HW FIFO (DPU): t1 t3 t4 t5 t6 t7
 *
 * ------   t0
 *          |
 *          b0
 *          |  \
 * Page0    t1   t2
 *          |    |
 *          b1   |
 *          |    |
 *          t3   |
 *          |    |
 * ------   t4   |
 *          |    |
 * Page1    |    |
 *          |    b2  <- b2->t7 dep needs to be legalized by connecting t2->b3
 *          b3   |
 *          |    |
 * ------   t5   |
 *          |    |
 *          b4   |
 * Page2    |    |
 *          t6   /
 *         /    /
 *        |    /
 *        |  t7
 *        \  |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPage() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {},   // task 3
            {3},  // task 4
            {4},  // task 5
            {},   // task 6
            {5}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {},   // task 4
            {3},  // task 5
            {4},  // task 6
            {2}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 4, 5, 6, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {},      // task 3
            {3},     // task 4
            {4},     // task 5
            {},      // task 6
            {5}      // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {},   // task 4
            {3},  // task 5
            {4},  // task 6
            {}    // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPage) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPage();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 4);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 5);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 7);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0
 * HW FIFO (DPU): t1 t2 t3 t4 t5 t6
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |  \
 *          b1  \
 *          |    |
 *          t2   |
 *          |    |
 * ------   t3   |
 *          |    |
 * Page1    |    |
 *          |    b2  -> b2->t6 needs to be legalized by removing it
 *          b3   |
 *          |    |
 * ------   t4   |
 *          |    |
 *          b4   |
 * Page2    |    |
 *          t5   /
 *         /    /
 *        |    /
 *        |  t6
 *        \  |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPageOnSameFifo() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {},      // task 2
            {3},     // task 3
            {4},     // task 4
            {},      // task 5
            {5}      // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {},   // task 3
            {3},  // task 4
            {4},  // task 5
            {2}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4, 5, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {},      // task 2
            {3},     // task 3
            {4},     // task 4
            {},      // task 5
            {5}      // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {},   // task 3
            {3},  // task 4
            {4},  // task 5
            {}    // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierOnSameFifo) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLongDepOnTaskWaitBarrierSideWithProducerFromEarlierPageOnSameFifo();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_FALSE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 1);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 1);
    EXPECT_EQ(page1BoundaryTasks[0], 4);

    // Check that it is taskInd = 4 that requires legalization - long dep on wait barrier side
    auto tasksToLegalize = barrierPagesSplitHandlerTest.getTasksWithNonAdjacentPageDependencyToLegalize();
    ASSERT_EQ(tasksToLegalize.size(), 1);
    EXPECT_EQ(tasksToLegalize[0], 6);

    // Modify graph
    barrierPagesSplitHandlerTest.legalizeNonAdjacentPageDependencies(tasksToLegalize);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t5
 * HW FIFO (DPU): t1 t3 t6
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |
 *          b1
 *          |  \
 * ------   t3  t2
 *          |
 *          b2
 *          |
 * Page1    t4
 *          |
 *          b3
 *          | \
 * ------   t5  t6
 *          |
 * Page2    b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLastTaskOnFifoInPageWithNoUpdateBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {},   // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {},   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {3}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {4}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {3}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLastTaskOnFifoInPageWithNoUpdateBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithLastTaskOnFifoInPageWithNoUpdateBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    ASSERT_TRUE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());
    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto lastTaskTypePerPageWithNoUpdBar = barrierPagesSplitHandlerTest.getLastTasksOnFifoPerPageWithNoUpdBar();
    ASSERT_EQ(lastTaskTypePerPageWithNoUpdBar.size(), 2);
    EXPECT_EQ(lastTaskTypePerPageWithNoUpdBar[0], 2);
    EXPECT_EQ(lastTaskTypePerPageWithNoUpdBar[1], 6);

    barrierPagesSplitHandlerTest.addUpdateBarriersForLastTaskOnFifoInPage(lastTaskTypePerPageWithNoUpdBar);

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);

    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 2);
    EXPECT_EQ(page1BoundaryTasks[0], 5);
    EXPECT_EQ(page1BoundaryTasks[1], 6);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t3 t5 t6
 * HW FIFO (DPU): t1 t4
 *
 * ------   t0
 *          |
 *          b0
 *          |  \
 * Page0    t1  t2
 *          |  / |
 *          b1   t3 <- last task on FIFO before sync task - needs to have update barrier: t3->b1
 *          |
 * ------   t4 (sync task)
 *          |
 *          b2
 *          |
 * Page1    t5
 *          |
 *          b3
 *          |
 * ------   t6
 *          |
 * Page2    b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithLastTaskOnFifoInPageWithNoUpdateBarBeforeSyncTask() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {1},  // task 2
            {},   // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {},   // task 3
            {1},  // task 4
            {2},  // task 5
            {3}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {4};

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 3, 5, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {},   // task 3
            {1},  // task 4
            {2},  // task 5
            {3}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithLastTaskOnFifoInPageWithNoUpdateBarBeforeSyncTask) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphWithLastTaskOnFifoInPageWithNoUpdateBarBeforeSyncTask();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    ASSERT_TRUE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());
    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto lastTaskTypePerPageWithNoUpdBar = barrierPagesSplitHandlerTest.getLastTasksOnFifoPerPageWithNoUpdBar();
    ASSERT_TRUE(!lastTaskTypePerPageWithNoUpdBar.empty());
    EXPECT_EQ(lastTaskTypePerPageWithNoUpdBar[0], 3);

    barrierPagesSplitHandlerTest.addUpdateBarriersForLastTaskOnFifoInPage(lastTaskTypePerPageWithNoUpdBar);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t8
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   |
 *          |    |
 *          b1   |
 *          |    |
 * ------   t3   t2
 *          |    |
 * Page1    b3   b2
 *          |    |
 * ------   t5   t4  -> Create t2->b3 and t3->b2 deps
 *          |    |
 * Page2    b5   b4
 *          |    |
 * ------   t7   t6  -> Create t4->b5 and t5->b4 deps
 *           \  /
 *            b6
 * Page3      |
 *            t8
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithBoundaryTaskDepsToLegalize() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeBoundaryTaskDeps) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithBoundaryTaskDepsToLegalize();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    EXPECT_FALSE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto boundaryTaskPairsMissingDepInBetween = barrierPagesSplitHandlerTest.getBoundaryTaskPairsMissingDepInBetween();

    // Expected missing deps in graph
    //  3 -> 4
    //  2 -> 5
    //  5 -> 6
    //  4 -> 7
    EXPECT_EQ(boundaryTaskPairsMissingDepInBetween.size(), 4);

    auto it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                        std::make_pair<size_t, size_t>(3, 4));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(2, 5));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(5, 6));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(4, 7));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    barrierPagesSplitHandlerTest.legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t8
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------      t0
 *             |
 *             b0
 *           /    \
 * Page0    t1     |
 *          |      |
 *          b1     |
 *          |      |
 * ------   t3    t2  <- Create t3->b2 deps
 *          |  /   |
 * Page1    b3    b2
 *          |      |
 * ------   t5     t4  <- Create t4->b5 and t5->b4 deps
 *          |      |
 * Page2    b5     b4
 *          |      |
 * ------   t7     t6
 *           \    /
 *             b6
 * Page3       |
 *             t8
 *             |
 *             b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithBoundaryTasksWithMultipleUpdateBars() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeBoundaryTaskDepsWithMultipleUpdateBars) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithBoundaryTasksWithMultipleUpdateBars();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    EXPECT_FALSE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 2);
    EXPECT_EQ(page1BoundaryTasks[0], 4);
    EXPECT_EQ(page1BoundaryTasks[1], 5);

    auto page2BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(2);
    ASSERT_EQ(page2BoundaryTasks.size(), 2);
    EXPECT_EQ(page2BoundaryTasks[0], 6);
    EXPECT_EQ(page2BoundaryTasks[1], 7);

    auto boundaryTaskPairsMissingDepInBetween = barrierPagesSplitHandlerTest.getBoundaryTaskPairsMissingDepInBetween();

    // Expected missing deps in graph
    //  3 -> 4
    //  5 -> 6
    //  4 -> 7
    EXPECT_EQ(boundaryTaskPairsMissingDepInBetween.size(), 3);

    auto it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                        std::make_pair<size_t, size_t>(3, 4));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(5, 6));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    it = std::find(boundaryTaskPairsMissingDepInBetween.begin(), boundaryTaskPairsMissingDepInBetween.end(),
                   std::make_pair<size_t, size_t>(4, 7));
    EXPECT_TRUE(it != boundaryTaskPairsMissingDepInBetween.end());

    barrierPagesSplitHandlerTest.legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t2 t3 t6 t7 t8
 * HW FIFO (DPU): t0 t1 t4 t5
 *
 * ------   t0
 *            \
 *             b0
 *           /  |
 * Page0    t1  t2
 *          |     \
 *          b1     \
 *            \     |
 * ------       t3  |
 *            /    /
 * Page1    b3    b2
 *        /     /
 *        |    /
 *        | t4
 *        |    \
 *        \     \
 * ------  t5  /|
 *            / |
 * Page2    b5  b4
 *          |   |
 *          |   t6
 *           \    \
 *            \    \
 * ------       t7  |
 *              |  /
 *              b6
 * Page3        |
 *              t8
 *              |
 *              b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithMultipleBoundaryTasksOnSameFifo() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4, 5},  // task 4
            {},      // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {2, 3, 6, 7, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {0, 1, 4, 5};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4},     // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeMultipleBoundaryTasksOnSameFifo) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithMultipleBoundaryTasksOnSameFifo();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    EXPECT_FALSE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto page0BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(0);
    ASSERT_EQ(page0BoundaryTasks.size(), 2);
    EXPECT_EQ(page0BoundaryTasks[0], 2);
    EXPECT_EQ(page0BoundaryTasks[1], 3);

    auto page1BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(1);
    ASSERT_EQ(page1BoundaryTasks.size(), 2);
    EXPECT_EQ(page1BoundaryTasks[0], 4);
    EXPECT_EQ(page1BoundaryTasks[1], 5);

    auto page2BoundaryTasks = barrierPagesSplitHandlerTest.getFirstAndLastBoundaryTasksForPage(2);
    ASSERT_EQ(page2BoundaryTasks.size(), 2);
    EXPECT_EQ(page2BoundaryTasks[0], 6);
    EXPECT_EQ(page2BoundaryTasks[1], 7);

    auto boundaryTaskPairsMissingDepInBetween = barrierPagesSplitHandlerTest.getBoundaryTaskPairsMissingDepInBetween();

    barrierPagesSplitHandlerTest.legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0
 * HW FIFO (DPU): t1 t2 t3 t4 t5
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |
 *          b1  <- b1 barrier has no consumer. It is redundant and can be removed
 *                 Remove t1->b1 dependency
 * ------   t2
 *          |
 *          b2
 * Page1    |
 *          t3
 *
 *          b3 <- b3 barrier has no producer. Create dependency from boundary task t2 to b3
 *          |
 * ------   t4
 *          |
 *          b4
 * Page2    |
 *          t5
 *          |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithWithBarriersWithNoProducerOrNoConsumer() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {},   // task 3
            {4},  // task 4
            {5}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {},   // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4, 5};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {},      // task 1
            {2, 3},  // task 2
            {},      // task 3
            {4},     // task 4
            {5}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {},   // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeGraphWithBarriersWithNoProducerOrNoConsumer) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithWithBarriersWithNoProducerOrNoConsumer();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize);

    // Original graph does not have valid split to pages
    ASSERT_TRUE(barrierPagesSplitHandlerTest.areNoDepsGoingBeyondNeighborPage());

    barrierPagesSplitHandlerTest.ensureBarrierHasProducer();
    auto foundRedundantBarriers = barrierPagesSplitHandlerTest.cleanupRedundantBarriers();
    EXPECT_TRUE(foundRedundantBarriers);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t5
 * HW FIFO (DPU): t1 t2 t3 t4
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |
 *          b1
 *          |
 * ------   t2
 *          |
 *          b2
 * Page1    |
 *          t3
 *          |
 *          b3
 *          |
 * ------   t4
 *          |
 *          b4
 * Page2    |
 *          t5
 *          |
 *          b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphGetBarrierDmaLocation() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetBarrierDmaLocation) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphGetBarrierDmaLocation();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    // Verify the split is valid afterwards
    ASSERT_TRUE(barrierPagesSplitHandlerTest.isSplitToPagesValid());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();
    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 2);
}

/**
 * HW FIFO (DMA): t0 t1 t2 t3 t5
 * HW FIFO (DPU): t4
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |  \
 *          |  b1
 *          |   |
 * ------   |  t2    <- Add t2->b2 dependency
 *          |   |
 *          b2  |
 *          |   |
 *  Page1   t3  |    <- Remove t3->b4 and change it to t4->b4. Slot for inserting barrier DMA
 *          |  \|       prepared between b2 and b3
 *          |   b3
 *          |   |
 * ------   |   t4
 *          |   |
 *          b4  |
 * Page2    |   |
 *          t5  |
 *           \ /
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereStartBarIsEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {3},     // task 2
            {3, 4},  // task 3
            {5},     // task 4
            {5}      // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 2, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2, 3},  // task 2
            {3},     // task 3
            {4, 5},  // task 4
            {5}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartBarIsEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToLegalizeForBarrierDmaWhereStartBarIsEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 2);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t1 t2 t3 t5
 * HW FIFO (DPU): t4
 *
 * ------   t0
 *          |
 *          b0
 *          |
 * Page0    t1
 *          |  \
 *          |  b1
 *          |   |
 * ------   |  t2    <- Add t2->b2 dependency
 *          |   |
 *          b2  |
 *          |   |
 *  Page1   t3  |    <- Slot for inserting barrier DMA prepared between b2 and b3
 *             \|
 *              b3
 *              |
 * ------       t4
 *            / |
 *          b4  |
 * Page2    |   |
 *          t5  |
 *           \ /
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphToLegalizeForBarrierDmaWhereStartBarIsEndBarAndLastBarIsSingle() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {3},     // task 2
            {3},     // task 3
            {4, 5},  // task 4
            {5}      // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 2, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2, 3},  // task 2
            {3},     // task 3
            {4, 5},  // task 4
            {5}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartBarIsEndBarAndLastBarIsSingle) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereStartBarIsEndBarAndLastBarIsSingle();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 2);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t1 t2 t3 t4
 *
 * ------   t0
 *          |
 *          b0
 * Page0    |
 *          t1
 *          |
 *          b1
 *          |
 * ------   t2
 *          |  \
 *          b2  b3    <- b2 is picked as page start barrier but its also a page end barrier
 * Page1    |   |        Configure b3 to be page end barrier and insert barrier DMA between b2 and b3
 *          |   t3
 *          |
 * ------   t4
 *          |
 * Page2    b4
 * ------
 */
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereOnlyEndBarIsStartBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {},      // task 3
            {4}      // task 4
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {3},  // task 3
            {2}   // task 4
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {},      // task 3
            {4}      // task 4
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {3},  // task 3
            {3}   // task 4
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereOnlyEndBarIsStartBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereOnlyEndBarIsStartBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 2);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t8
 * HW FIFO (DPU): t1 t3 t5 t6 t7
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   \
 *          |     |
 *          b1    |
 *          |     |
 * ------   t3    t2
 *          |  /\ |
 * Page1    b3    b2    <- b2->t4 dep to be removed and replaced with b3->t4.  Slot for inserting barrier DMA
 *          |     |        prepared between b2 and b3
 * ------   t5    t4
 *          |  \/ |
 * Page2    b5    b4    <- b4->t6 dep to be removed and replaced with b5->t6.  Slot for inserting barrier DMA
 *          |     |        prepared between b4 and b5
 * ------   t7    t6
 *           \  /
 *            b6
 * Page3      |
 *            t8
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereAllStartBarsAreEndBars() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 6, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2, 3},  // task 2
            {2, 3},  // task 3
            {4, 5},  // task 4
            {4, 5},  // task 5
            {6},     // task 6
            {6},     // task 7
            {7}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {3},  // task 4
            {3},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereAllStartBarsAreEndBars) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereAllStartBarsAreEndBars();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 3);

    auto barProgDmaPosPage2 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(2);

    ASSERT_TRUE(barProgDmaPosPage2.valid);
    ASSERT_EQ(barProgDmaPosPage2.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage2.waitBars[0], 4);
    ASSERT_EQ(barProgDmaPosPage2.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage2.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage2.insertAfter, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t7
 * HW FIFO (DPU): t1 t3 t5 t8 t9
 *
 * ------     t0
 *            |
 *            b0
 *           /  \
 * Page0    t1   \
 *          |     |
 *          b1    |
 *          |     |
 * ------   t3    t2
 *            \ /
 *             b2       <- No start only barrier. Remove t4->b5 dependency and create t5->b5
 *             |           and b3->t6. Slot for inserting barrier DMA between b2 and b3
 * Page1       t4
 *            / | \
 *           /  b3 \
 *          |   |   |
 *          |  t5   |
 *          |   |   |
 * ------   |   |  t6
 *          |    \  |
 * Page2    b5    b4    <- Slot for inserting barrier DMA between b4 and b5
 *          |  /\  |
 * ------   t7    t8
 *           \  /
 *            b6
 * Page3      |
 *            t9
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphToLegalizeForBarrierDmaWhereAllStartBarsAreEndBarsWithReiterationNeeded() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2},     // task 3
            {3, 5},  // task 4
            {4},     // task 5
            {4},     // task 6
            {6},     // task 7
            {6},     // task 8
            {7},     // task 9
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {0},     // task 2
            {1},     // task 3
            {2},     // task 4
            {3},     // task 5
            {},      // task 6
            {4, 5},  // task 7
            {4, 5},  // task 8
            {6}      // task 9
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 7};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 8, 9};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2},     // task 3
            {3},     // task 4
            {4, 5},  // task 5
            {4},     // task 6
            {6},     // task 7
            {6},     // task 8
            {7},     // task 9
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {0},     // task 2
            {1},     // task 3
            {2},     // task 4
            {3},     // task 5
            {3},     // task 6
            {4, 5},  // task 7
            {4, 5},  // task 8
            {6}      // task 9
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereAllStartBarsAreEndBarsWithReiterationNeeded) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereAllStartBarsAreEndBarsWithReiterationNeeded();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 3);

    auto barProgDmaPosPage2 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(2);

    ASSERT_TRUE(barProgDmaPosPage2.valid);
    ASSERT_EQ(barProgDmaPosPage2.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage2.waitBars[0], 4);
    ASSERT_EQ(barProgDmaPosPage2.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage2.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage2.insertAfter, 6);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *           /   \
 * ------   t3    t4
 *          |  \   | \
 *          b3    b4  |   <- remove t4->b4 dependency and add t4->b3.
 *          |      |  |      Slot for inserting barrier DMA prepared between b3 and b4&b5
 *  Page1   t5     |  |
 *          |      |  |
 *          b5 <------
 *          |      |
 * ------   t6    t7
 *          |   /
 *   Page2  b6
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereNotAllStartBarsAreEndBars() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3, 4},  // task 3
            {4, 5},  // task 4
            {5},     // task 5
            {6},     // task 6
            {6}      // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {5},  // task 6
            {4}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 3;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},        // task 0
            {1},        // task 1
            {2},        // task 2
            {3, 4},     // task 3
            {3, 4, 5},  // task 4
            {5},        // task 5
            {6},        // task 6
            {6}         // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {5},  // task 6
            {4}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereNotAllStartBarsAreEndBars) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereNotAllStartBarsAreEndBars();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 3);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 2);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 4);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[1], 5);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t4 t6 t8
 * HW FIFO (DPU): t1 t3 t5 t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *           /   \
 * ------   t3    t4
 *          |   /  |
 *          b3     |
 *          |  \   |
 *  Page1   |   t5 |
 *          |    \ |
 *          |      b4
 *          |      |
 *          |      t6
 *          |      |
 *          |      b5   <- create b5->t7 dependency. Barrier DMA to wait on b3 and update b5
 *          |      |
 * ------   t7    t8
 *          |   /
 *   Page2  b6
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereStartBarDependsOnEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {3, 4},  // task 4
            {4},     // task 5
            {5},     // task 6
            {6},     // task 7
            {6}      // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {3},  // task 7
            {5}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 3, 5, 7};

    size_t pageSize = 3;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {3, 4},  // task 4
            {4},     // task 5
            {5},     // task 6
            {6},     // task 7
            {6}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {5}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartBarDependsOnEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereStartBarDependsOnEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 3);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t1 t2 t3 t5 t9
 * HW FIFO (DPU): t4 t6 t7 t8 t10
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *            |
 *            t3
 *            |
 *            b3
 *           /   \
 * ------   t4    t5   <- Create t5->b4 dep
 *          |      |      Slot for inserting barrier DMA prepared between b4 and b4&b7
 *          b4     |
 *          |      |
 *  Page1   t6     |
 *          |      |
 *          b5     |
 *          | \    |
 *          |  t7  |
 *          |   \  /
 *          |    b6
 *          |    |
 *          |    t8
 *          |    |
 *          |    b7
 *          |    |
 * ------   t9   t10
 *          |   /
 *   Page2  b8
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphToLegalizeForBarrierDmaWhereOneOfWaitBarriersWouldDependOnUpdate() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {6},  // task 5
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8},  // task 9
            {8}   // task 10
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6},  // task 8
            {5},  // task 9
            {7}   // task 10
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 2, 3, 5, 9};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {4, 6, 7, 8, 10};

    size_t pageSize = 4;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {4, 6},  // task 5
            {5},     // task 6
            {6},     // task 7
            {7},     // task 8
            {8},     // task 9
            {8}      // task 10
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6},  // task 8
            {5},  // task 9
            {7}   // task 10
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereOneOfWaitBarriersWouldDependOnUpdate) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereOneOfWaitBarriersWouldDependOnUpdate();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 4);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 2);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[1], 7);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t2 t3 t4 t5
 * HW FIFO (DPU): t1
 *
 * ------     t0
 *            |
 *            b0
 *           / \
 * Page0    t1  t2
 *          |    |
 *          b1   |
 *          |    |
 * ------   t3   |  <- both t2 and t3 are boundary tasks but t3 has no update barrier
 *              /
 *            b2  <- Create t3->b2 dep
 *            |      Slot for inserting barrier DMA prepared between b2 and b3
 *  Page1     t4
 *            |
 *            b3
 *            |
 * ------     t5
 *            |
 *  Page2     b4
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToLegalizeForBarrierDmaWhereStartTaskHasNoUpdateBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {},   // task 3
            {3},  // task 4
            {4}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 2, 3, 4, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartTaskHasNoUpdateBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereStartTaskHasNoUpdateBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(
            barrierInfoTest, barrierMapsConfig.taskQueueTypeMap, pageSize, /* barrierFifoDepth */ 1,
            /* numClusters */ 1, /*shvTasksWithDpu*/ {}, /*shvTasksWithDma*/ {},
            /*initializeAndVerifyBoundaryTasksData*/ false);

    barrierPagesSplitHandlerTest.reconfigureAllowDepsChangeDuringInitialization(true);
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.initializeBoundaryTasksData());
    barrierPagesSplitHandlerTest.reconfigureAllowDepsChangeDuringInitialization(false);
    EXPECT_THROW(barrierPagesSplitHandlerTest.ensureBoundaryTasksHaveUpdateBarriers(), vpux::Exception);
    barrierPagesSplitHandlerTest.reconfigureAllowDepsChangeDuringInitialization(true);
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.ensureBoundaryTasksHaveUpdateBarriers());

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);

    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 2);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 3);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 3);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t3 t4
 * HW FIFO (DPU): t1 t2
 *           DMA0  DPU
 * ------     t0
 *            |
 *            b0
 *               \
 *  Page0          t1
 *               / | \
 *            b1   |  \
 *            |    |   |
 * ------     |    |   |
 *            |   /\   |
 *            | b2, b3 |
 *            |   \/   |
 *  Page1     |   t2   | boundary task on page 1 without update barrier
 *            |       /
 *            |     /
 * ------    /     /
 *          |    /
 *          | b4
 *           \|
 *  Page2     t3
 *            |
 *            b5
 *            |
 * ------     t4
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphWithBoundaryTaskWithoutUpdateBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},           // task 0
            {1, 2, 3, 4},  // task 1
            {},            // task 2
            {5},           // task 3
            {},            // task 4
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {2, 3},  // task 2
            {1, 4},  // task 3
            {5},     // task 4
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 3, 4};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig = barrierMapsConfig;
    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},           // task 0
            {1, 2, 3, 4},  // task 1
            {4},           // task 2
            {5},           // task 3
            {},            // task 4
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {2, 3},  // task 2
            {1, 4},  // task 3
            {5},     // task 4
    };

    fillProducersAndConsumers(expectedBarrierMapsConfig);
    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckDepsChangeDuringInitialization) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphWithBoundaryTaskWithoutUpdateBarrier();
    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(
            barrierInfoTest, barrierMapsConfig.taskQueueTypeMap, pageSize, /* barrierFifoDepth */ 1,
            /* numClusters */ 1,
            /*shvTasksWithDpu*/ {}, /*shvTasksWithDma*/ {}, /*initializeAndVerifyBoundaryTasksData*/ false);
    barrierPagesSplitHandlerTest.reconfigureAllowDepsChangeDuringInitialization(false);
    EXPECT_THROW(barrierPagesSplitHandlerTest.initializeBoundaryTasksData(), vpux::Exception);
    barrierPagesSplitHandlerTest.reconfigureAllowDepsChangeDuringInitialization(true);
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.initializeBoundaryTasksData());

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();
    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA): t0 t3 t4
 * HW FIFO (DPU): t1 t2 t5 t6 t7 t8
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *            t1
 *            |
 * Page0      b1
 *            |
 *            t2
 *            |
 *            b2
 *            |
 *            t3
 *            | \
 *            |  \
 *            |   \
 * ------     t4   \
 *            |     \
 *            |     b3   <- Create t4->b3 dependency
 *            |    / |
 *  Page1     |   t5 |
 *            | /    |
 *            b4     |
 *            |      |   <- Barrier DMA to wait on b3 and update b5
 *            t6     |
 *            |      |
 *            b5     |
 *            |      |
 * ------     t8    t7   <- Remove b3->t7 and create b5->t7 dependency
 *            |   /
 *   Page2    b6
 *            |
 * ------
 *
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphToLegalizeForBarrierDmaWhereStartBarDependsOnEndBarAndStartBarToLegalize() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {5},  // task 6
            {6},  // task 7
            {6}   // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {},   // task 4
            {3},  // task 5
            {4},  // task 6
            {3},  // task 7
            {5}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 3, 4};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {1, 2, 5, 6, 7, 8};

    size_t pageSize = 3;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {3, 4},  // task 4
            {4},     // task 5
            {5},     // task 6
            {6},     // task 7
            {6}      // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {},   // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {5}   // task 8
    };

    fillProducersAndConsumers(expectedBarrierMapsConfig);
    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, LegalizeForBarrierDmaWhereStartBarDependsOnEndBarAndStartBarToLegalize) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToLegalizeForBarrierDmaWhereStartBarDependsOnEndBarAndStartBarToLegalize();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    barrierPagesSplitHandlerTest.legalizeForDmaProgrammingBarriers();

    auto barProgDmaPosPage1 = barrierPagesSplitHandlerTest.getDmaProgrammingBarrierPosition(1);
    ASSERT_TRUE(barProgDmaPosPage1.valid);
    ASSERT_EQ(barProgDmaPosPage1.waitBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.waitBars[0], 3);
    ASSERT_EQ(barProgDmaPosPage1.updateBars.size(), 1);
    EXPECT_EQ(barProgDmaPosPage1.updateBars[0], 5);
    EXPECT_EQ(barProgDmaPosPage1.insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();
    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t5 t6 t9
 * HW FIFO (DMA1): t1 t8
 * HW FIFO (DPU): t2 t3 t4 t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page1     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4  <- Page1 has no boundary tasks of DMA0 and DMA1 type
 *            |      both need to be inserted
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *           /   \
 * ------   t6    t7   <- Page2 has boundary tasks only of DMA0 type
 *          |  /   |      DMA1 type dummy DMA needs to be inserted
 *  Page3   b6     b7
 *          |      |
 * ------   t8   t9
 *          |   /
 *  Page4   b8
 *
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDma() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {6},     // task 6
            {6, 7},  // task 7
            {8},     // task 8
            {8}      // task 9
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6},  // task 8
            {7}   // task 9
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 5, 6, 9};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 3, 4, 7};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {6},     // task 6
            {6, 7},  // task 7
            {8},     // task 8
            {8}      // task 9
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6},  // task 8
            {7}   // task 9
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionData) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToInsertDummyDma();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};

    ASSERT_EQ(dummyDmaInsertionData.size(), 3);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 3);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 0);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 3);

    EXPECT_EQ(dummyDmaInsertionData[1].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[1].queueType, dmaType1);
    ASSERT_EQ(dummyDmaInsertionData[1].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[1].waitBars[0], 3);
    ASSERT_EQ(dummyDmaInsertionData[1].updateBars.size(), 0);
    EXPECT_EQ(dummyDmaInsertionData[1].insertAfter, 3);

    EXPECT_EQ(dummyDmaInsertionData[2].pageInd, 2);
    EXPECT_EQ(dummyDmaInsertionData[2].queueType, dmaType1);
    ASSERT_EQ(dummyDmaInsertionData[2].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[2].waitBars[0], 5);
    ASSERT_EQ(dummyDmaInsertionData[2].updateBars.size(), 0);
    EXPECT_EQ(dummyDmaInsertionData[2].insertAfter, 6);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t1 t2 t5
 * HW FIFO (DMA1): t6
 * HW FIFO (DPU): t3 t4
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |  \
 * ------     t2  t3
 *            |    |
 *  Page1     b2   b3     <- Page1 has no boundary tasks of DMA1 type
 *            |    |         and Dummy DMA1 is to be inserted after barrier b2
 * ------     t4  t5         Create t3->b2 dependency to guarantee that
 *            |   /          Dummy DMA1 starts after all Page0 barriers have
 *            b4             been fully consumed
 *            |
 *  Page2     t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDmaWithAdditionalWaitBarLegalization() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 1, 2, 5};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2, 3},  // task 3
            {4},     // task 4
            {4},     // task 5
            {5}      // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithAdditionalWaitBarLegalization) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToInsertDummyDmaWithAdditionalWaitBarLegalization();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType1);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 2);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 0);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t5
 * HW FIFO (DMA1): t1
 * HW FIFO (DPU): t2 t3 t4
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page1     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4 (sync)  <- Page1 has no boundary tasks of DMA0 type
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDmaWithSyncPoint() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {4};

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 3, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4}   // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithSyncPoint) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToInsertDummyDmaWithSyncPoint();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 2);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 3);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 3);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t6
 * HW FIFO (DMA1): t1 t3
 * HW FIFO (DPU): t2 t4 t5
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |  \
 * ------     t2  t3
 *            |   |
 *            b2  |  <- Create t3->b2 dependency and have b2 as wait barrier for dummy DMA
 *            |   |
 *  Page1     t4  |
 *            |  /
 *            b3
 *            |
 * ------     t5 (sync)  <- Page1 has no boundary tasks of DMA0 type
 *            |
 *            b4
 *            |
 *  Page2     t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDmaWithSyncPointAndCommonStartEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {5};

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2, 3},  // task 3
            {3},     // task 4
            {4},     // task 5
            {5}      // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithSyncPointAndCommonStartEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToInsertDummyDmaWithSyncPointAndCommonStartEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 2);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 3);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t7
 * HW FIFO (DMA1): t1 t3 t6
 * HW FIFO (DPU): t2 t4 t5
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 *            t2
 *            |
 *            b2
 *            |  \
 * ------     t3  t4
 *            |   |
 *            b3  |  <- Create t4->b3 dependency and have b3 as wait barrier for dummy DMA
 *            |   |
 *  Page1     t5  |
 *            |   |
 *            b5  b4
 *            |  /
 * ------     t6 (sync)  <- Page1 has no boundary tasks of DMA0 type
 *            |
 *            b6
 *            |
 *  Page2     t7
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps>
graphToInsertDummyDmaWithSyncPointWith2WaitBarsAndCommonStartEndBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {7}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {2},     // task 3
            {2},     // task 4
            {3},     // task 5
            {4, 5},  // task 6
            {6}      // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {6};

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 7};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5};

    size_t pageSize = 3;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {3, 4},  // task 4
            {5},     // task 5
            {6},     // task 6
            {7}      // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {2},     // task 3
            {2},     // task 4
            {3},     // task 5
            {4, 5},  // task 6
            {6}      // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithSyncPointWith2WaitBarsAndCommonStartEndBar) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToInsertDummyDmaWithSyncPointWith2WaitBarsAndCommonStartEndBar();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 3);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 5);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t5
 * HW FIFO (DMA1): t1 t3
 * HW FIFO (DPU): t2 t4
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |  \
 * ------     t2  t3
 *            |   |
 *            b2  b3     <- dummyDMA0 to be inserted between b2 and b3, t3->b2 dep to be created
 *  Page1     |  /
 * ------     t4 (sync)  <- Page1 has no boundary tasks of DMA0 type
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToInsertDummyDmaWithSyncPointThatWaitsOnAllPageBarriers() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5}   // task 5
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {1},     // task 3
            {2, 3},  // task 4
            {4}      // task 5
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {4};

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 5};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {2, 3},  // task 3
            {4},     // task 4
            {5}      // task 5
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {1},     // task 3
            {2, 3},  // task 4
            {4}      // task 5
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, GetAndLegalizeDummyDmaInsertionDataWithSyncPointThatWaitsOnAllPageBarriers) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] =
            graphToInsertDummyDmaWithSyncPointThatWaitsOnAllPageBarriers();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto dummyDmaInsertionData = barrierPagesSplitHandlerTest.getAndLegalizeDummyDmaInsertionData();

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};

    ASSERT_EQ(dummyDmaInsertionData.size(), 1);

    EXPECT_EQ(dummyDmaInsertionData[0].pageInd, 1);
    EXPECT_EQ(dummyDmaInsertionData[0].queueType, dmaType0);
    ASSERT_EQ(dummyDmaInsertionData[0].waitBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].waitBars[0], 2);
    ASSERT_EQ(dummyDmaInsertionData[0].updateBars.size(), 1);
    EXPECT_EQ(dummyDmaInsertionData[0].updateBars[0], 3);
    EXPECT_EQ(dummyDmaInsertionData[0].insertAfter, 3);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t6
 * HW FIFO (DMA1): t1 t3
 * HW FIFO (DPU): t2 t4 t5
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |  \
 * ------     t2  t3
 *            |  /
 *            b2
 *            |
 *  Page1     t4
 *            |
 *            b3
 *            |
 * ------     t5
 *            |
 *            b4
 *            |
 *  Page2     t6
 *            |
 *            b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueAtBootstrapAndStartBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueAtBootstrapAndStartBarrier) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig] =
            graphToCheckEnqueueAtBootstrapAndStartBarrier();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // DPU tasks (3 tasks) enqueued after start barrier
    ASSERT_EQ(enqueueDataVec.size(), 1);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 2);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 1);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4
 * HW FIFO (DMA1): t1 t3 t5
 * HW FIFO (DPU): t6
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3  <- enqueue DPU (t6) after this barrier
 *            |
 * ------     t4
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *            |
 * ------     t6
 *  Page3     |
 *            b6
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, BarrierInfoMaps> graphToCheckEnqueueOfDpuNotAtStartBarrier() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {6};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 2};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuNotAtStartBarrier) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuNotAtStartBarrier();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // DPU task enqueued after barrier 3 - end barrier of Page 1
    ASSERT_EQ(enqueueDataVec.size(), 1);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 1);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 3);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t6
 * HW FIFO (DMA1): t1 t3 t5
 * HW FIFO (DPU): t7 t8
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3 (PID3)  <- enqueue t7 DPU after barrier b3. Enqueue t8 DPU together
 *            |             as there are required dependencies from b3, which is previous
 * ------     t4            barrier instance of b6 - wait barrier of t8, to t7
 *            |
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *            |  \
 * ------     t6 t7
 *            |  /
 *            b6 (PID3)
 *  Page3     |
 *            t8
 *            |
 *            b7
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, BarrierInfoMaps> graphToCheckEnqueueMergeOfDpu() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {7, 8};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 3, 2};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7}   // task 8
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5},  // task 6
            {5},  // task 7
            {6}   // task 8
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueMergeOfDpu) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, expectedBarrierMapsConfig] = graphToCheckEnqueueMergeOfDpu();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // Both DPU tasks enqueued together after barrier 3
    ASSERT_EQ(enqueueDataVec.size(), 1);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 1);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 3);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 1);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 4);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4
 * HW FIFO (DMA1): t1 t3 t6
 * HW FIFO (SHV(withDPU)): t5
 * HW FIFO (DPU): t7
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 * ------     t2
 *            |
 *            b2
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |  \
 * ------     t4  t5
 *            |  /
 *            b4  <- enqueue DPU (t7) after this barrier after SHVwithDPU (t5)
 *            |
 *  Page2     t6
 *            |
 *            b5
 *            |
 * ------     t7
 *  Page3     |
 *            b6
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedDueToShvWithDpu() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {5},  // task 6
            {6}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {7};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {5};

    size_t pageSize = 2;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 2};

    SmallVector<size_t> shvTasksWithDpu = {5};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {5},  // task 6
            {6}   // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedDueToShvWithDpu) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedDueToShvWithDpu();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // SHV tasks enqueued after start barrier
    // DPU task enqueue delayed after SHV task
    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx
    ASSERT_EQ(enqueueDataVec.size(), 2);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[1].pageInd, 2);
    EXPECT_EQ(enqueueDataVec[1].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[1].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[1].waitBars[0], 4);
    EXPECT_EQ(enqueueDataVec[1].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].insertBefore, 6);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t6
 * HW FIFO (DMA1): t1 t3 t8
 * HW FIFO (SHV(withDPU)): t5 t7
 * HW FIFO (DPU): t9
 *
 * ------     t0
 *            | \
 *            b0 b1
 *            | /
 *  Page0     t1
 *            |
 *            b2
 *            |
 * ------     t2
 *            | \
 *            b3 b4
 *            | /
 *  Page1     t3
 *            |
 *            b5
 *            |
 * ------     t4
 *            |
 *            b6
 *            | \
 *            t5 t6
 *            | /
 *            b7
 *            |
 *  Page2     t7
 *            |
 *            b8
 *            |
 * ------     t8 (sync)
 *  Page3     |
 *            b9  <- enqueue DPU (t9) after this barrier after SHVwithDPU (t5 and t7) and sync task (t8)
 *            |
 *            t9
 *            |
 *            b10
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedDueToShvWithDpuWithSyncTask() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0, 1},  // task 0
            {2},     // task 1
            {3, 4},  // task 2
            {5},     // task 3
            {6},     // task 4
            {7},     // task 5
            {7},     // task 6
            {8},     // task 7
            {9},     // task 8
            {10}     // task 9
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0, 1},  // task 1
            {2},     // task 2
            {3, 4},  // task 3
            {5},     // task 4
            {6},     // task 5
            {6},     // task 6
            {7},     // task 7
            {8},     // task 8
            {9}      // task 9
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {8};

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {9};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {5, 7};

    size_t pageSize = 3;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4};

    SmallVector<size_t> shvTasksWithDpu = {5, 7};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0, 1},  // task 0
            {2},     // task 1
            {3, 4},  // task 2
            {5},     // task 3
            {6},     // task 4
            {7},     // task 5
            {7},     // task 6
            {8},     // task 7
            {9},     // task 8
            {10}     // task 9
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0, 1},  // task 1
            {2},     // task 2
            {3, 4},  // task 3
            {5},     // task 4
            {6},     // task 5
            {6},     // task 6
            {7},     // task 7
            {8},     // task 8
            {9}      // task 9
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedDueToShvWithDpuWithSyncTask) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedDueToShvWithDpuWithSyncTask();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // SHV tasks enqueued after start barrier
    // DPU task enqueue delayed after SHV task
    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx
    ASSERT_EQ(enqueueDataVec.size(), 2);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 1);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 5);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 1);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 4);

    EXPECT_EQ(enqueueDataVec[1].pageInd, 3);
    EXPECT_EQ(enqueueDataVec[1].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[1].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[1].waitBars[0], 9);
    EXPECT_EQ(enqueueDataVec[1].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].insertBefore, 9);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t6 t9
 * HW FIFO (DMA1): t1 t3 t5
 * HW FIFO (SHV0_0(withDPU)): t7
 * HW FIFO (SHV0_1(withDPU)): t8
 * HW FIFO (DPU): t10
 *
 * ------      t0
 *             |
 *             b0
 *             |
 *  Page0      t1
 *             |
 *             b1
 *             |
 *             t2
 *             |
 *             b2
 *             |
 * ------      t3
 *             |
 *             b3
 *             |
 *  Page1      t4
 *             |
 *             b4
 *             |
 *             t5
 *             |
 *             b5
 *           / |  \
 * ------  t6  t7  t8    <- t10 (DPU) initial enqueue proposal was b5 but since t7 & t8
 *           \ |   |        are SHV with DPU its enqueue needs to be delayed to be after
 *             b6  b7       both of them. Final t10 enqueue DMA barriers are b6 and b7
 *             |  /
 *  Page2      t9
 *             |
 *             b8
 *             |
 * ------      t10
 *  Page3      |
 *             b9
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedDueToMultipleShvWithDpuInParallel() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8},  // task 9
            {9}   // task 10
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {2},     // task 3
            {3},     // task 4
            {4},     // task 5
            {5},     // task 6
            {5},     // task 7
            {5},     // task 8
            {6, 7},  // task 9
            {8}      // task 10
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType0{config::ExecutorKind::SHAVE_ACT, 0};  // SHV tile 0 list 0
    const VPURT::TaskQueueType shvType1{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4, 6, 9};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {10};
    barrierMapsConfig.taskQueueTypeMap[shvType0] = {7};
    barrierMapsConfig.taskQueueTypeMap[shvType1] = {8};

    size_t pageSize = 3;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3};

    SmallVector<size_t> shvTasksWithDpu = {7, 8};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8},  // task 9
            {9}   // task 10
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {2},     // task 3
            {3},     // task 4
            {4},     // task 5
            {5},     // task 6
            {5},     // task 7
            {5},     // task 8
            {6, 7},  // task 9
            {8}      // task 10
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedDueToMultipleShvWithDpuInParallel) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedDueToMultipleShvWithDpuInParallel();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // SHV tasks enqueued after start barrier
    // DPU task enqueue delayed after SHV task
    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx
    ASSERT_EQ(enqueueDataVec.size(), 3);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[1].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[1].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[1].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[1].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[1].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[2].pageInd, 2);
    EXPECT_EQ(enqueueDataVec[2].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[2].waitBars.size(), 2);
    EXPECT_EQ(enqueueDataVec[2].waitBars[0], 6);
    EXPECT_EQ(enqueueDataVec[2].waitBars[1], 7);
    EXPECT_EQ(enqueueDataVec[2].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[2].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[2].insertBefore, 9);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t6 t7
 * HW FIFO (DMA1): t1 t3
 * HW FIFO (SHV(withDPU)): t5
 * HW FIFO (DPU): t8 t9
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 *            t2
 *            |
 *            b2
 *            |
 *            t3
 *            |
 *            b3
 *          /   \
 *         t4   t5    <- t5 is SHV(DPU) task, t8 enqueue can be happened before, because we do not have dependency
 *         |     |       between t5 and t8.
 *         t6    b4      t9 enqueue must be delayed after t5, proposed enqueue position was insertBefore t6 and wBar b4,
 *         |     |       but this insertion create unexpected dependency between t5 and t8 which was already processed.
 * -----   b5    t7      For avoid this case, we change insertion position - insertBefore t7.
 *         |     |
 *         t8    b6
 *         |     |
 *         b7    t9
 *               |
 *              b8

 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedForAvoidCreatingExtraBarrierDependency() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {},   // task 4
            {4},  // task 5
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8}   // task 9
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {},   // task 6
            {4},  // task 7
            {5},  // task 8
            {6}   // task 9
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4, 6, 7};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {8, 9};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {5};

    size_t pageSize = 5;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 4, 5, 0, 1, 2};

    SmallVector<size_t> shvTasksWithDpu = {5};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {},   // task 4
            {4},  // task 5
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8}   // task 9
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {},   // task 6
            {4},  // task 7
            {5},  // task 8
            {6}   // task 9
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedForAvoidCreatingExtraBarrierDependency) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedForAvoidCreatingExtraBarrierDependency();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // SHV tasks enqueued after start barrier
    // DPU task enqueue delayed after SHV task
    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx
    ASSERT_EQ(enqueueDataVec.size(), 3);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[1].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[1].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[1].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[1].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[1].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[2].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[2].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[2].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[2].waitBars[0], 4);
    EXPECT_EQ(enqueueDataVec[2].startTaskIdx, 1);
    EXPECT_EQ(enqueueDataVec[2].endTaskIdx, 1);
    EXPECT_EQ(enqueueDataVec[2].insertBefore, 7);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t6 t8
 * HW FIFO (DMA1): t1 t3 t7
 * HW FIFO (SHV(withDPU)): t5
 * HW FIFO (DPU): t9 t10
 *
 * ------     t0
 *            |
 *            b0
 *            |
 *  Page0     t1
 *            |
 *            b1
 *            |
 *            t2
 *            |
 *            b2
 *            |
 *            t3
 *            |
 *            b3
 *          /   \
 *         t4   t5    <- t5 is SHV(DPU) task, t9 enqueue can be happened before, because we do not have dependency
 *         |     |       between t5 and t9.
 *         |     b4      t10 enqueue must be delayed after t5, proposed enqueue position was insertBefore t6 and wBar
 *         |     |       b4, but this insertion create unexpected dependency between t5 and t9 which was already
 *         |     |       processed. To avoid this case, we change insertion position - insertBefore t9
 * -----   t6    t7      with wBar b6.
 *         |     |
 *         b5    b6
 *         |     |
 *  Page1  t8    |
 *         |     |
 *         b7    |
 *         |     |
 *         t9    t10
 *           \  /
 *            b8
 *
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedToNextPage() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {},   // task 4
            {4},  // task 5
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8},  // task 9
            {8}   // task 10
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {},   // task 6
            {4},  // task 7
            {5},  // task 8
            {7},  // task 9
            {6}   // task 10
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 7};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {9, 10};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {5};

    size_t pageSize = 5;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 4, 5, 0, 1, 2};

    SmallVector<size_t> shvTasksWithDpu = {5};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {},   // task 4
            {4},  // task 5
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
            {8},  // task 9
            {8}   // task 10
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {},   // task 6
            {4},  // task 7
            {5},  // task 8
            {7},  // task 9
            {6}   // task 10
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedToNextPage) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedToNextPage();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // SHV tasks enqueued after start barrier
    // DPU task enqueue delayed after SHV task
    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx
    ASSERT_EQ(enqueueDataVec.size(), 3);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[1].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[1].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[1].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[1].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[1].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[2].pageInd, 1);
    EXPECT_EQ(enqueueDataVec[2].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[2].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[2].waitBars[0], 6);
    EXPECT_EQ(enqueueDataVec[2].startTaskIdx, 1);
    EXPECT_EQ(enqueueDataVec[2].endTaskIdx, 1);
    EXPECT_EQ(enqueueDataVec[2].insertBefore, 9);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

/**
 * HW FIFO (DMA0): t0 t1 t5 t7
 * HW FIFO (DMA1): t2 t3
 * HW FIFO (SHV(withDPU)): t4
 * HW FIFO (DPU): t6
 *
 * ------   t0
 *          |
 *          b0
 *          |
 *  Page0   t1
 *          |
 *          b1
 *          |
 *          t2
 *          |
 *          b2
 *          |
 * ------   t3
 *          |
 *          b3
 *          |  \
 *  Page1   |   t4 (SHVwithDPUandDMA)
 *          t5  |
 *          |   |
 *          b4  b5
 *          |   |
 * ------   t6  t7  - DPU (t6) depends on DMA (t5) which is going to be blocked
 *  Page2   |  /      by SkipDMA descriptor that will be later added and which will be
 *          b6        under SHVwithDPUandDMA (t4) task. t6 enqueue must be delayed after b5
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu, shvTasksWithDma and
// expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, SmallVector<size_t>, BarrierInfoMaps>
graphToCheckEnqueueOfDpuDelayedDueToShvWithDpuAndDma() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {5},  // task 4
            {4},  // task 5
            {6},  // task 6
            {6}   // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5}   // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN,
                                        getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN,
                                        getDMAQueueIdEncoding(/*port*/ 1, VPUIP::DmaChannelType::DDR)};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 1, 5, 7};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {2, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {6};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {4};

    size_t pageSize = 3;

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 4, 5, 0};

    SmallVector<size_t> shvTasksWithDpu = {4};
    SmallVector<size_t> shvTasksWithDma = {4};

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {5},  // task 4
            {4},  // task 5
            {6},  // task 6
            {6}   // task 7
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5}   // task 7
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, shvTasksWithDma,
                           expectedBarrierMapsConfig);
}

// Create a tuple with BarrierInfoMaps, pageSize, barrierToPidVec, shvTasksWithDpu and shvTasksWithDma.
std::tuple<BarrierInfoMaps, size_t, SmallVector<size_t>, SmallVector<size_t>, SmallVector<size_t>>
graphToCheckEnqueueOfFirstShvSubmitDmaAfterLatestStartBarrierProducer() {
    BarrierInfoMaps barrierMapsConfig;

    // Dedicated layout to differentiate heuristics for SHV enqueue placement:
    // - Current rule: insert after latest producer of start barrier b0 -> task 0 => insertBefore = 1.
    // - Historical rule: insert before earliest consumer of b0 -> task 3 => insertBefore = 3.
    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {7},  // task 1
            {8},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {5},  // task 6
            {4},  // task 7
            {6},  // task 8
            {6}   // task 9
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {},   // task 1
            {7},  // task 2
            {0},  // task 3
            {1},  // task 4
            {2},  // task 5
            {3},  // task 6
            {3},  // task 7
            {4},  // task 8
            {5}   // task 9
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN,
                                        getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN,
                                        getDMAQueueIdEncoding(/*port*/ 1, VPUIP::DmaChannelType::DDR)};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{config::ExecutorKind::SHAVE_ACT, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 3, 7, 9};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 2, 4, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {8};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {6};

    size_t pageSize = 4;
    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 4, 5, 0, 1, 2};
    SmallVector<size_t> shvTasksWithDpu = {6};
    SmallVector<size_t> shvTasksWithDma = {6};

    return std::make_tuple(barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, shvTasksWithDma);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfDpuDelayedDueToShvWithDpuAndDma) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, shvTasksWithDma, expectedBarrierMapsConfig] =
            graphToCheckEnqueueOfDpuDelayedDueToShvWithDpuAndDma();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu, shvTasksWithDma);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    // DMA tasks enqueued at bootstrap without enqueue DMA
    // SHV tasks enqueued after start barrier
    // DPU task enqueue delayed after SHV task
    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx
    ASSERT_EQ(enqueueDataVec.size(), 2);

    EXPECT_EQ(enqueueDataVec[0].pageInd, 0);
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::SHAVE_ACT);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[0].waitBars[0], 0);
    EXPECT_EQ(enqueueDataVec[0].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[0].insertBefore, 1);

    EXPECT_EQ(enqueueDataVec[1].pageInd, 1);
    EXPECT_EQ(enqueueDataVec[1].queueType.type, config::ExecutorKind::DPU);
    ASSERT_EQ(enqueueDataVec[1].waitBars.size(), 1);
    EXPECT_EQ(enqueueDataVec[1].waitBars[0], 5);
    EXPECT_EQ(enqueueDataVec[1].startTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].endTaskIdx, 0);
    EXPECT_EQ(enqueueDataVec[1].insertBefore, 5);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueOfFirstShvSubmitDmaAfterLatestStartBarrierProducer) {
    auto [barrierMapsConfig, pageSize, barrierToPidVec, shvTasksWithDpu, shvTasksWithDma] =
            graphToCheckEnqueueOfFirstShvSubmitDmaAfterLatestStartBarrierProducer();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1,
                                                                 /* numClusters */ 1, shvTasksWithDpu, shvTasksWithDma);

    barrierPagesSplitHandlerTest.initPrevPhysBarrierData(barrierToPidVec);
    ExecutionGroupAnalysisTest execGroupAnalysis(barrierMapsConfig.taskQueueTypeMap, /*maxVariantCount*/ 4,
                                                 /*maxInvariantCount*/ 2,
                                                 /*maxKernelInvocationCount*/ 2, /*maxKernelRangeCount*/ 2,
                                                 /*tilesCount*/ 1);

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};
    auto enqueueDataVec = barrierPagesSplitHandlerTest.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    ASSERT_EQ(enqueueDataVec.size(), 2);

    // Enqueue data ordered by insertBefore, pageInd, queueType, startTaskIdx.
    // This SHV enqueue must be positioned right after the latest producer of start barrier b0 (task 0).
    EXPECT_EQ(enqueueDataVec[0].queueType.type, config::ExecutorKind::SHAVE_ACT);
    EXPECT_EQ(enqueueDataVec[0].queueType.id, 1);
    ASSERT_EQ(enqueueDataVec[0].waitBars.size(), 1);
    const auto startBarrier = enqueueDataVec[0].waitBars[0];
    EXPECT_EQ(startBarrier, 0);

    const auto& startBarrierProducers = barrierMapsConfig.barrierProducerMap[startBarrier];
    ASSERT_FALSE(startBarrierProducers.empty());
    const auto latestStartBarrierProducer =
            *std::max_element(startBarrierProducers.begin(), startBarrierProducers.end());
    const auto currentInsertBefore = latestStartBarrierProducer + 1;

    // Historical heuristic for reference: enqueue before earliest start-barrier consumer.
    const auto& startBarrierConsumers = barrierMapsConfig.barrierConsumerMap[startBarrier];
    ASSERT_FALSE(startBarrierConsumers.empty());
    const auto earliestStartBarrierConsumer =
            *std::min_element(startBarrierConsumers.begin(), startBarrierConsumers.end());
    const auto previousInsertBefore = earliestStartBarrierConsumer;

    EXPECT_NE(previousInsertBefore, currentInsertBefore);
    EXPECT_GT(previousInsertBefore, currentInsertBefore);

    EXPECT_EQ(enqueueDataVec[0].insertBefore, currentInsertBefore);
}

/**
 * HW FIFO (DMA0): t0 t2 t4
 * HW FIFO (DMA1): t1 t3 t5
 * HW FIFO (DPU): t6
 *
 * ------     t0
 *            |
 *            b0 <- start barrier
 *            |
 *  Page0     t1
 *            |
 *            b1 <- enqueue barrier for Page1 boundary tasks and Page2 tasks
 *            |
 * ------     t2
 *            |
 *            b2 <- enqueue barrier for Page2 boundary tasks and Page3 tasks
 *            |
 *  Page1     t3
 *            |
 *            b3
 *            |
 * ------     t4 <- create b2->t4 dependency so that b2 consumption event happens
 *            |     at the page end and can be used as enqueue barrier
 *            b4
 *            |
 *  Page2     t5
 *            |
 *            b5
 *            |
 * ------     t6
 *  Page3     |
 *            b6
 * ------
 */
// Create a tuple with BarrierInfoMaps, pageSize and expectedBarrierMapsConfig
std::tuple<BarrierInfoMaps, size_t, BarrierInfoMaps> graphToCheckEnqueueBarrierOfDpuAndDma() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6}   // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5
            {5}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{config::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{config::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{config::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 2, 4};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {1, 3, 5};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {6};

    size_t pageSize = 2;

    BarrierInfoMaps expectedBarrierMapsConfig;

    expectedBarrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {6}   // task 6
    };

    expectedBarrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2
            {2},     // task 3
            {2, 3},  // task 4
            {4},     // task 5
            {5}      // task 6
    };
    fillProducersAndConsumers(expectedBarrierMapsConfig);

    return std::make_tuple(barrierMapsConfig, pageSize, expectedBarrierMapsConfig);
}

TEST_F(BarrierPagesSplitTests, CheckEnqueueBarrierOfDpuAndDma) {
    auto [barrierMapsConfig, pageSize, expectedBarrierMapsConfig] = graphToCheckEnqueueBarrierOfDpuAndDma();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                                 pageSize, /*_barrierFifoDepth = */ 1);

    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyTaskBarrierPagesAreValid());
    EXPECT_NO_THROW(barrierPagesSplitHandlerTest.verifyNoCyclicDeps());

    EXPECT_TRUE(barrierPagesSplitHandlerTest.areBoundaryTasksFromNeighborPagesDependent());

    auto enqueueBarrierDataVec = barrierPagesSplitHandlerTest.getAndLegalizeEnqueueBarrierData();
    ASSERT_EQ(enqueueBarrierDataVec.size(), 7);

    // Page 0 Task 0 (DMA) - enqueue at bootstrap
    ASSERT_TRUE(!enqueueBarrierDataVec[0].has_value());
    // Page 0 Task 1 (DMA) - enqueue at bootstrap
    ASSERT_TRUE(!enqueueBarrierDataVec[1].has_value());
    // Page 0 Task 2 (DMA) - enqueue at bootstrap
    ASSERT_TRUE(!enqueueBarrierDataVec[2].has_value());
    // Page 1 Task 3 (DMA) - enqueue at bootstrap
    ASSERT_TRUE(!enqueueBarrierDataVec[3].has_value());
    // Page 1 boundary Task 4 (DMA) - enqueue at bar1(Page0)
    ASSERT_TRUE(enqueueBarrierDataVec[4].has_value());
    EXPECT_EQ(enqueueBarrierDataVec[4].value(), 1);
    // Page 2 Task 5 (DMA) - enqueue at bar1(Page0)
    ASSERT_TRUE(enqueueBarrierDataVec[5].has_value());
    EXPECT_EQ(enqueueBarrierDataVec[5].value(), 1);
    // Page 2 boundary Task 6 (DPU) - enqueue at bar2(Page1)
    ASSERT_TRUE(enqueueBarrierDataVec[6].has_value());
    EXPECT_EQ(enqueueBarrierDataVec[6].value(), 2);

    auto testResult = barrierPagesSplitHandlerTest.getBarrierMaps();

    EXPECT_EQ(expectedBarrierMapsConfig.taskUpdateBarriers, testResult.taskUpdateBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.taskWaitBarriers, testResult.taskWaitBarriers);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierProducerMap, testResult.barrierProducerMap);
    EXPECT_EQ(expectedBarrierMapsConfig.barrierConsumerMap, testResult.barrierConsumerMap);
}

}  // namespace
