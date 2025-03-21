//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/core/barrier_info.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

using namespace vpux;

using BarrierSimulatorTests = ::testing::Test;

/**
 *        t0
 *         |
 *        b0
 *       /  \
 *      t1  t2
 *      |  \ |
 *      b1  b2
 *     /  \  |
 *    t3    t4
 *          |
 *          b3
 *          |
 *          t5
 *          |
 *          b4
 *          |
 *          t6
 */
BarrierInfoMaps graphToCheckBarrierWlmHandler() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1, 2},  // task 1
            {2},     // task 2
            {},      // task 3
            {3},     // task 4
            {4},     // task 5
            {}       // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {0},     // task 2
            {1},     // task 3
            {1, 2},  // task 4
            {3},     // task 5
            {4}      // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType2{VPU::ExecutorKind::DMA_NN, 1};

    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {0, 2, 4, 5, 6};
    barrierMapsConfig.taskQueueTypeMap[dmaType2] = {1, 3};

    return barrierMapsConfig;
}

TEST_F(BarrierSimulatorTests, CheckBarrierWlmHandler) {
    auto barrierMapsConfig = graphToCheckBarrierWlmHandler();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierWlmHandler barrierWlmHandlerTest(barrierInfoTest);

    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(4, 0));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(4, 1));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(4, 2));
}

/**
 * HW FIFO (DMA): t2 t4 t6
 * HW FIFO (DPU): t0 t1 t3 t5 t7 t8
 *
 *       t0
 *       |
 *       b0
 *       |
 *       t1
 *       |
 *       b1
 *     /  \
 *    t2   t3
 *    |    |
 *    b2   b3
 *    |    |
 *    t4   t5
 *    |    |
 *    b4   b5
 *    |    |
 *    t6   t7
 *     \   /
 *       b6
 *       |
 *      t8
 */
BarrierInfoMaps graphToCheckBarrierWlmHandlerWithDepsTwoBranches() {
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
            {}    // task 8
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6}   // task 8
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {2, 4, 6};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {0, 1, 3, 5, 7, 8};

    return barrierMapsConfig;
}

TEST_F(BarrierSimulatorTests, CheckBarrierWlmHandlerWithDepsTwoBranches) {
    auto barrierMapsConfig = graphToCheckBarrierWlmHandlerWithDepsTwoBranches();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierWlmHandler barrierWlmHandlerTest(barrierInfoTest);

    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(6, 0));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(6, 1));
}

/**
 * HW FIFO (DMA): t0 t1 t3
 * HW FIFO (DPU): t2 t4 t5 t6
 *
 *       t0
 *     /   \
 *    b0   b1
 *    |    |
 *    t1   t2
 *    |    |
 *    |    b2
 *    |    |
 *    t3   t4
 *     \   /
 *       b3
 *       |
 *       t5
 *       |
 *       b4
 *       |
 *       t6
 */
BarrierInfoMaps graphToCheckBarrierWlmHandlerWithDepsTwoBranchesThroughFifo() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0, 1},  // task 0
            {},      // task 1
            {2},     // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {}       // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {},   // task 3
            {2},  // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5, 6};

    return barrierMapsConfig;
}

TEST_F(BarrierSimulatorTests, CheckBarrierWlmHandlerWithDepsTwoBranchesThroughFifo) {
    auto barrierMapsConfig = graphToCheckBarrierWlmHandlerWithDepsTwoBranchesThroughFifo();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierWlmHandler barrierWlmHandlerTest(barrierInfoTest);

    // TODO: To be fixed with E#138120
    // EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(4, 0));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(4, 1));
}

/**
 *
 * HW FIFO : t0 t1 t2 t5 t8 t10 t11
 * HW FIFO : t4 t7 t9
 * HW FIFO : t6
 * HW FIFO : t3
 *
 *        t0
 *         |
 *        b0
 *         |
 *        t1
 *         |
 *        b1
 *         | \
 *        t2  \
 *         |   \
 *        b2    \
 *      /    \   \
 *    t4     t5  t3
 *   /  \    |   |
 *  b4  b5   b6  |
 *  |    |   |   |
 *  t6  t7   t8  |
 *       |   |   |
 *      b7   b8  |
 *       |   |   |
 *      t9   t10 /
 *        \  |  /
 *         \ | /
 *           b3
 *           |
 *          t11
 */
BarrierInfoMaps graphToCheckBarrierWlmHandlerWithMultipleBranches() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0},     // task 0
            {1},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4, 5},  // task 4
            {6},     // task 5
            {},      // task 6
            {7},     // task 7
            {8},     // task 8
            {3},     // task 9
            {3},     // task 10
            {}       // task 11
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {2},  // task 5
            {4},  // task 6
            {5},  // task 7
            {6},  // task 8
            {7},  // task 9
            {8},  // task 10
            {3}   // task 11
    };

    fillProducersAndConsumers(barrierMapsConfig);

    const VPURT::TaskQueueType dpuType1{VPU::ExecutorKind::DPU, 0};
    const VPURT::TaskQueueType dpuType2{VPU::ExecutorKind::DPU, 1};
    const VPURT::TaskQueueType dmaType1{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dmaType2{VPU::ExecutorKind::DMA_NN, 1};

    barrierMapsConfig.taskQueueTypeMap[dpuType1] = {0, 1, 2, 5, 8, 10, 11};
    barrierMapsConfig.taskQueueTypeMap[dpuType2] = {4, 7, 9};
    barrierMapsConfig.taskQueueTypeMap[dmaType1] = {6};
    barrierMapsConfig.taskQueueTypeMap[dmaType2] = {3};

    return barrierMapsConfig;
}

TEST_F(BarrierSimulatorTests, CheckBarrierWlmHandlerWithMultipleBranches) {
    auto barrierMapsConfig = graphToCheckBarrierWlmHandlerWithMultipleBranches();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierWlmHandler barrierWlmHandlerTest(barrierInfoTest);

    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(3, 0));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(3, 1));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(3, 2));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(3, 5));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(3, 5));
}

/**
 * HW FIFO (DMA): t0 t1 t3 t4 t6 t7 t8 t10
 * HW FIFO (DPU): t2 t5 t9
 *
 *      t0
 *     /  \
 *    b0  b1
 *    |    |
 *    t1  t2
 *     \  /
 *      b2
 *      |
 *      t3 (sync point)
 *      |
 *      b3
 *     /  \
 *    t4   t5
 *    |    |
 *    b4   |
 *    |    |
 *    t6   |
 *     \  /
 *      b5
 *      |
 *      t7 (sync point)
 *      |
 *      b6
 *     /   \
 *    t8   t9
 *    |
 *    b7
 *    |
 *    t10
 */
BarrierInfoMaps graphToCheckBarrierWlmHandlerWithGraphSplit() {
    BarrierInfoMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0, 1},  // task 0
            {2},     // task 1
            {2},     // task 2
            {3},     // task 3
            {4},     // task 4
            {5},     // task 5
            {5},     // task 6
            {6},     // task 7
            {7},     // task 8
            {},      // task 9
            {}       // task 10
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
            {6},  // task 9
            {7}   // task 10
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.syncTasksIds = {3, 7};

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};

    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 3, 4, 6, 7, 8, 10};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 5, 9};

    return barrierMapsConfig;
}

TEST_F(BarrierSimulatorTests, CheckBarrierWlmHandlerWithGraphSplit) {
    auto barrierMapsConfig = graphToCheckBarrierWlmHandlerWithGraphSplit();

    BarrierInfoTest barrierInfoTest(barrierMapsConfig);
    VPURT::BarrierWlmHandler barrierWlmHandlerTest(barrierInfoTest);

    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 0));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 1));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 2));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 3));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 4));
    EXPECT_TRUE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 5));
    EXPECT_FALSE(barrierWlmHandlerTest.canBarrierReusePidFromBarrier(7, 6));
}
