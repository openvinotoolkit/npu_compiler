//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

namespace vpux {

// Test helper class for FeasibleMemoryScheduler to access private members
class FeasibleMemorySchedulerTest {
public:
    explicit FeasibleMemorySchedulerTest(FeasibleMemoryScheduler& scheduler): _scheduler(scheduler) {
    }

    size_t getLoopRegionSize() const {
        return _scheduler._loopRegions.size();
    }

    size_t getScheduledLoopRegionSize() const {
        return _scheduler._scheduledLoopRegionInd.size();
    }

    const FeasibleMemoryScheduler::ScheduledOpInfoVec& getScheduledOps() const {
        return _scheduler._scheduledOps;
    }

    // In Loop region, the input buffers are allocated to the same addresses
    bool verifyLoopInputAddress() const;

    // Methods for manipulating the scheduler state for testing purposes

    // Run only the setup phase of init() (skips schedulingLoop) so state can be
    // manipulated before calling scheduleComputeOps() directly.
    void runInitSetup() {
        _scheduler.doInitSetup();
    }

    // Call the private scheduleComputeOps() method directly.
    void runScheduleComputeOps() {
        _scheduler.scheduleComputeOps();
    }

    // Directly insert an op index into _readyDMAOps.
    void injectReadyDMAOp(FeasibleMemoryScheduler::operationIdxType opIdx) {
        _scheduler._readyDMAOps.insert(opIdx);
    }

    // Remove an op from _readyDataOps (used to simulate DATA_IN being scheduled elsewhere).
    void removeFromReadyDataOps(FeasibleMemoryScheduler::operationIdxType opIdx) {
        _scheduler._readyDataOps.erase(opIdx);
    }

    // Mark an op as already scheduled by inserting into _opIdxEndCycleMap.
    void markOpAsScheduled(FeasibleMemoryScheduler::operationIdxType opIdx, size_t cycleEnd) {
        _scheduler._opIdxEndCycleMap[opIdx] = cycleEnd;
    }

    // Mark a buffer as alive in the scan handler.
    void markBufferAlive(mlir::Value buffer) {
        _scheduler._scan.handler().markAsAlive(buffer);
    }

    // Register a buffer producer (used together with markOpAsScheduled so scheduler
    // can locate the producer of a "pre-scheduled" op's output buffer if needed).
    void registerBufferProducer(mlir::Value buffer, FeasibleMemoryScheduler::operationIdxType opIdx) {
        _scheduler._bufferProducer[buffer] = opIdx;
    }

    // Query whether an op is still in _readyDMAOps.
    bool isInReadyDMAOps(FeasibleMemoryScheduler::operationIdxType opIdx) const {
        return _scheduler._readyDMAOps.count(opIdx) > 0;
    }

    // Return whether the op is in loopRegionInd (fix's guard set).
    bool isInLoopRegionInd(FeasibleMemoryScheduler::operationIdxType opIdx) const {
        return _scheduler._computeRegionsSchedule.loopRegionInd.count(opIdx) > 0;
    }

    // Inspect the cycle-begin heap for entries with the given opIdx.
    size_t countInCycleBeginHeap(FeasibleMemoryScheduler::operationIdxType opIdx) const {
        size_t count = 0;
        for (const auto& elem : _scheduler._cycleBeginHeap) {
            if (elem.op_ == opIdx) {
                ++count;
            }
        }
        return count;
    }

private:
    FeasibleMemoryScheduler& _scheduler;
};

}  // namespace vpux
