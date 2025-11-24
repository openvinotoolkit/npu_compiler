//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/interfaces/enqueue_barrier.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/core/barrier_info.hpp"

#include <gtest/gtest.h>

using namespace vpux;

using EnqueueBarrierTests = ::testing::Test;

/**
 * HW FIFO (DMA0): t0 t2 t4 t5 t6
 * HW FIFO (DMA1): t1 t3
 * Barriers: b<VID>(<PID>)
 *
 *        t0
 *         |
 *       b0(0)
 *       /  \
 *      t1  t2
 *      |  \ |
 *    b1(1) b2(2)
 *      |    |
 *      t3  t4
 *        \  |
 *          b3(3)
 *           |
 *          t5
 *           |
 *          b4(0)
 *           |
 *          t6
 */
std::pair<BarrierInfoMaps, SmallVector<size_t>> graphSimple() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {}       // task 6
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

    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType2{VPU::ExecutorKind::DMA_NN, 1};

    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {0, 2, 4, 5, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType2] = {1, 3};

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0};

    return std::make_pair(barrierMapsConfig, barrierToPidVec);
}

TEST_F(EnqueueBarrierTests, CheckEnqueueForGraphSimple) {
    auto [barrierMapsConfig, barrierToPidVec] = graphSimple();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 4 is expected to have no enqueue barrier as it can be enqueued at bootstrap
    // because it uses first instance of barriers
    EXPECT_FALSE(enqueueBarrierHandlerTest.getEnqueueBarrier(4).has_value());

    // Task 5 can be enqueued at barrier 0, which is previous instance of its update barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(5).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(5).value(), 0);

    // Task 6 can be enqueued at barrier 0, which is previous instance of its wait barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(6).value(), 0);
}

TEST_F(EnqueueBarrierTests, CheckEnqueueForGraphSimpleBarrierFifo3) {
    auto [barrierMapsConfig, barrierToPidVec] = graphSimple();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec, /*barrierFifoDepth*/ 3);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 4 is expected to have no enqueue barrier as it can be enqueued at bootstrap
    // because it uses first instance of barriers
    EXPECT_FALSE(enqueueBarrierHandlerTest.getEnqueueBarrier(4).has_value());

    // Task 5 can be enqueued at bootstrap as this is first instance of wait barrier
    // and needed update barrier will be ready in barrier FIFO
    EXPECT_FALSE(enqueueBarrierHandlerTest.getEnqueueBarrier(5).has_value());

    // Task 6 can be enqueued at bootstrap, same as previous task (t5) because
    // at t5 execution previous instance of t6 wait barrier (b0(0)) is guaranteed to be
    // consumed and reconfigured for b4(0)
    EXPECT_FALSE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
}

/**
 * HW FIFO (DMA0): t0 t2 t4 t5 t6
 * HW FIFO (DMA1): t1 t3
 * Barriers: b<VID>(<PID>)
 *
 *        t0
 *         |
 *       b0(0)
 *       /  \
 *      t1  t2
 *      |  \ |
 *    b1(1) b2(2)
 *      |    |
 *      t3  t4
 *        \  |
 *          b3(0)
 *           |
 *          t5
 *           |
 *          t6
 *           |
 *          b4(1)
 */
std::pair<BarrierInfoMaps, SmallVector<size_t>> graphWithTasksWithoutWaitBar() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {},      // task 5
            {4}      // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {}    // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType2{VPU::ExecutorKind::DMA_NN, 1};

    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {0, 2, 4, 5, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType2] = {1, 3};

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 0, 1};

    return std::make_pair(barrierMapsConfig, barrierToPidVec);
}

TEST_F(EnqueueBarrierTests, CheckEnqueueForGraphWithTasksWithoutWaitBar) {
    auto [barrierMapsConfig, barrierToPidVec] = graphWithTasksWithoutWaitBar();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 4 can be enqued at barrier 0, which is previous instance of its update barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(4).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(4).value(), 0);

    // Task 5 can be enqued at barrier 0, which is previous instance of its wait barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(5).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(5).value(), 0);

    // Task 6 can be enqued at barrier 1, which is previous instance of its update barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(6).value(), 1);
}

/**
 * HW FIFO (DMA0): t0 t1
 * HW FIFO (DPU):  t5 t6
 * HW FIFO (SHV):  t2 t3 t4
 * Barriers: b<VID>(<PID>)
 *
 *   t0
 *    |
 *  b0(0)
 *    |
 *   t1
 *    |
 *  b1(2)
 *    |
 *   t2
 *    |
 *  b2(1)
 *    |
 *   t3
 *    |
 *  b3(3)
 *    |
 *   t4
 *    |
 *  b4(1)
 *    |
 *   t5        <- t5(DPU) will be enqueued at b2(1)
 *    |
 *  b5(0)
 *    |
 *   t6        <- t6(DPU) initial enqueue proposal is b1(2)
 *    |           but its enqueue will be delayed to that of t5 - b2(1)
 *  b6(2)
 */
std::pair<BarrierInfoMaps, SmallVector<size_t>> graphDelayEnqueueBasedOnPrevious() {
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

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_NN, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {5, 6};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {2, 3, 4};

    SmallVector<size_t> barrierToPidVec = {0, 2, 1, 3, 1, 0, 2};

    return std::make_pair(barrierMapsConfig, barrierToPidVec);
}

TEST_F(EnqueueBarrierTests, CheckDelayEnqueueBasedOnPrevious) {
    auto [barrierMapsConfig, barrierToPidVec] = graphDelayEnqueueBasedOnPrevious();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 5 will be enqued at barrier 2
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(5).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(5).value(), 2);

    // Task 6 is expected to have its enqueue delayed to that of Task 5 - barrier 2
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(6).value(), 2);
}

/**
 * HW FIFO (DMA0): t0 t1 t2 t3 t6
 * HW FIFO (DMA1): t4 t5 t7
 * Barriers: b<VID>(<PID>)
 *
 *   t0
 *    |
 *  b0(0)
 *    |
 *   t1
 *    |
 *  b1(1)
 *    |
 *   t2
 *    |
 *  b2(2)
 *    |
 *   t3
 *    |
 *  b3(0)
 *    |
 *   t4        <- t4(DMA1) will be enqueued at b0(0)
 *    |
 *   t5        <- t5(DMA1) enqueue barrier first candidate is b1(1)
 *    |           but its enqueue will be delayed until t4(DMA) executes to
 *  b4(1)         guarantee DMA FIFO has no overflow. Final enqueue would be b3(0)
 *    |
 *   t6
 *    |
 *   t7        <- t7(DMA1) enqueue barrier first candidate is b2(2)
 *    |           but its enqueue will be delayed until t5(DMA) executes to
 *  b5(2)         guarantee DMA FIFO has no overflow. Final enqueue would be b4(1)
 */
std::pair<BarrierInfoMaps, SmallVector<size_t>> graphDelayEnqueueBasedOnDmaFifoSize() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {},   // task 4
            {4},  // task 5
            {},   // task 6
            {5},  // task 7
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {2},  // task 3
            {3},  // task 4
            {},   // task 5
            {4},  // task 6
            {}    // task 7
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};

    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {4, 5, 7};
    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 1, 2, 3, 6};

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 0, 1, 2};

    return std::make_pair(barrierMapsConfig, barrierToPidVec);
}

TEST_F(EnqueueBarrierTests, CheckDelayEnqueueBasedOnDmaFifoSize) {
    auto [barrierMapsConfig, barrierToPidVec] = graphDelayEnqueueBasedOnDmaFifoSize();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    // For test purpose make FIFO depth = 1 and disable optimization to make it easier to trigger scenario
    // where FIFO can get overflow
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec, /*barrierFifoDepth*/ 1,
                                                           /*dmaFifoDepth = */ 1,
                                                           /*optimizeAndMergeEnqFlag = */ false);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 4 will be enqueued at barrier 0 (common ancestor of wait and update barrier)
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(4).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(4).value(), 0);

    // Task 5 is expected to have its enqueue delayed
    // to that of Task 4 wait barrier (barrier 3) to guarantee FIFO has no overflow
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(5).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(5).value(), 3);

    // Task 7 is expected to have its enqueue delayed
    // to that of Task 5 update barrier (barrier 4) to guarantee FIFO has no overflow
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(7).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(7).value(), 4);
}

/**
 * HW FIFO (DMA0): t0 t1 t9 t10
 * HW FIFO (DMA1): t2 t3 t4 t6 t8
 * HW FIFO (DPU):  t5 t7
 * Barriers: b<VID>(<PID>)
 *
 *      t0
 *       |
 *      b0(0)
 *     /  \
 *   t1   t2
 *    |    |
 *  b1(1) b2(2)
 *    |    |
 *   t3    t5
 *    |    |
 *  b3(3)  |
 *    |    |
 *   t4    |
 *     \  /
 *     b4(0)
 *     /  \
 *   t6   t7
 *    |   |
 *    |  b5(1)
 *    |   |
 *   t8   t9
 *    |   |
 *    |  b6(3)
 *    |   |
 *    |   t10
 *     \ /
 *    b7(2)
 */
std::pair<BarrierInfoMaps, SmallVector<size_t>> graphDelayEnqueueBasedOnDmaFifoSizeWithReiteration() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {4},  // task 5
            {},   // task 6
            {5},  // task 7
            {7},  // task 8
            {6},  // task 9
            {7},  // task 10
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {1},  // task 3
            {3},  // task 4
            {2},  // task 5
            {4},  // task 6
            {4},  // task 7
            {},   // task 8
            {5},  // task 9
            {6}   // task 10
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 1, 9, 10};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {2, 3, 4, 6, 8};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {5, 7};

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 3, 0, 1, 3, 2};

    return std::make_pair(barrierMapsConfig, barrierToPidVec);
}

TEST_F(EnqueueBarrierTests, CheckDelayEnqueueBasedOnDmaFifoSizeWithReiteration) {
    auto [barrierMapsConfig, barrierToPidVec] = graphDelayEnqueueBasedOnDmaFifoSizeWithReiteration();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    // For test purpose make FIFO depth = 1 and disable optimization to make it easier to trigger scenario
    // where FIFO can get overflow
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec, /*barrierFifoDepth*/ 1,
                                                           /*dmaFifoDepth = */ 1,
                                                           /*optimizeAndMergeEnqFlag = */ false);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 6 will be enqueued at barrier 0 - previous instance of its wait barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(6).value(), 0);

    // Task 8 will be enqueued at barrier 3 as a result of delaying enqueue to not overflow DMA FIFO
    // Original enqueue barrier was 2 but using 3 guarantees that task 4 and 6 was popped from FIFO
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(8).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(8).value(), 3);

    // Task 9 will be enqueued at barrier 3 - common ancestor of its wait and update barrier
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(9).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(9).value(), 3);

    // Task 10 original enqueue proposal was 4 but to not overflow DMA FIFO it was changed
    // to 5 as this guarantees that task 9 was popped from FIFO
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(10).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(10).value(), 5);
}

/**
 * HW FIFO (DMA0): t0 t1 t2 t3 t5
 * HW FIFO (DMA1): t4 t6
 * Barriers: b<VID>(<PID>)
 *
 *   t0
 *    |
 *  b0(2)
 *    |
 *   t1
 *    |
 *  b1(0)
 *    |
 *   t2
 *    |
 *  b2(1)
 *    |
 *   t3
 *    |
 *  b3(0)
 *    |
 *   t4        <- t4(DMA0) will be enqueued at b2(1)
 *    |
 *  b4(1)
 *    |
 *   t5
 *    |
 *  b5(2)
 *    |
 *   t6        <- t6(DMA0) prev wait barrier instance is b0(0)
 *                but its enqueue will be combined with previous DMA task
 *                enqueue as part of optimization
 */
std::pair<BarrierInfoMaps, SmallVector<size_t>> graphOptimizeEnqueueWithPrevious() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5
            {}    // task 6
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

    const VPURT::TaskQueueType dmaType0{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 1};

    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {4, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType0] = {0, 1, 2, 3, 5};

    SmallVector<size_t> barrierToPidVec = {2, 0, 1, 0, 1, 2};

    return std::make_pair(barrierMapsConfig, barrierToPidVec);
}

TEST_F(EnqueueBarrierTests, CheckOptimizeEnqueueWithPrevious) {
    auto [barrierMapsConfig, barrierToPidVec] = graphOptimizeEnqueueWithPrevious();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 4 will be enqued at barrier 0
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(4).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(4).value(), 2);

    // Task 6 is expected to have its enqueue aligned with that of previous task
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(6).value(), 2);
}

TEST_F(EnqueueBarrierTests, CheckOptimizeEnqueueWithPreviousBarrierFifo3) {
    auto [barrierMapsConfig, barrierToPidVec] = graphOptimizeEnqueueWithPrevious();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);

    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec, /*barrierFifoDepth*/ 3);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 4 will be enqued at previous instance of wait barrier (b1(0)). Update
    // barrier will be ready in barrier FIFO
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(4).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(4).value(), 1);

    // Task 6 is expected to have its enqueue aligned with that of previous task
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(6).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(6).value(), 1);
}

/**
 * HW FIFO (DMA0): t0 t1 t4 t6
 * HW FIFO (DPU):  t2 t5
 * HW FIFO (SHV(withDPU)): t3
 * Barriers: b<VID>(<PID>)
 *
 *   t0
 *    |
 *  b0(0)
 *    |
 *   t1
 *    |
 *  b1(1)
 *    |
 *   t2
 *    |
 *  b2(2)
 *    |
 *   t3   <- SHV with DPU
 *    |
 *  b3(0)
 *    |
 *   t4
 *    |
 *  b4(1)
 *    |
 *   t5  <- Based on barriers this DPU could be enqueued at b2, but because
 *    |     later there is SHV task with DPU (t3) enqueue needs to be delayed
 *  b5(2)   to barrier b3, so that when t3 runs, DPU FIFO is not blocked
 *    |
 *   t6
 *    |
 *  b6(0)
 */
std::tuple<BarrierInfoMaps, SmallVector<size_t>, SmallVector<size_t>> graphDelayDpuEnqueueBasedOnShvWithDpu() {
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

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType shvType{VPU::ExecutorKind::SHAVE_NN, 1};  // SHV tile 0 list 1

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 5};
    barrierMapsConfig.taskQueueTypeMap[shvType] = {3};

    SmallVector<size_t> barrierToPidVec = {0, 1, 2, 0, 1, 2, 0};

    SmallVector<size_t> shvTasksWithDpu = {3};

    return std::make_tuple(barrierMapsConfig, barrierToPidVec, shvTasksWithDpu);
}

TEST_F(EnqueueBarrierTests, CheckDelayDpuEnqueueBasedOnShvWithDpu) {
    auto [barrierMapsConfig, barrierToPidVec, shvTasksWithDpu] = graphDelayDpuEnqueueBasedOnShvWithDpu();

    ASSERT_TRUE(barrierToPidVec.size() == barrierMapsConfig.barrierProducerMap.size());

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::EnqueueBarrierHandler enqueueBarrierHandlerTest(barrierInfoTest, barrierMapsConfig.taskQueueTypeMap,
                                                           barrierToPidVec, /*barrierFifoDepth*/ 1, /*dmaFifoDepth*/ 64,
                                                           /*optimizeAndMergeEnqFlag*/ true, /*numClusters*/ 1,
                                                           shvTasksWithDpu);

    const auto res = enqueueBarrierHandlerTest.calculateEnqueueBarriers();
    ASSERT_TRUE(mlir::succeeded(res));

    // Task 5 will be enqueued at barrier 3
    ASSERT_TRUE(enqueueBarrierHandlerTest.getEnqueueBarrier(5).has_value());
    EXPECT_EQ(enqueueBarrierHandlerTest.getEnqueueBarrier(5).value(), 3);
}
