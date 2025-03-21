//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/schedule_analysis_utils.hpp"

#include "vpux/utils/profiling/reports/ted.hpp"

#include <iomanip>

using namespace vpux;

static const std::map<std::string, int> executorStrToId = {
        {"DMA_NN", 0}, {"DPU", 1}, {"NCE", 2}, {"SHAVE_ACT", 3}, {"SHAVE_NN", 4}};

ExecutorStallCycles vpux::getExecutorStallRegions(ScheduledOpInfoVec& scheduledOps) {
    ExecutorStallCycles executorStalls;
    // Map: Key: pair {executorKind, executorInstance}, Value: cycle
    std::map<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, int64_t> executorCycles;

    for (auto& schedOp : scheduledOps) {
        for (auto execInst : schedOp.executorInstanceMask.set_bits()) {
            auto executorAndInstancePair = std::make_pair(schedOp.queueType, execInst);
            if (schedOp.cycleBegin_ > executorCycles[executorAndInstancePair]) {
                // stall on executor from
                auto executorStall = std::make_pair(executorCycles[executorAndInstancePair], schedOp.cycleBegin_);
                executorStalls[executorAndInstancePair].push_back(executorStall);
            }
            executorCycles[executorAndInstancePair] = schedOp.cycleEnd_;
        }
    }

    return executorStalls;
}

StallCycles vpux::getStallsOnAllExecutorPipelines(ScheduledOpInfoVec& scheduledOps) {
    StallCycles stallsOnAllExecutorPipelines;
    auto executorStalls = getExecutorStallRegions(scheduledOps);

    // loop through executor with least stalls
    auto minStallSizeExecutor = std::min_element(
            executorStalls.begin(), executorStalls.end(),
            [](const std::pair<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, StallCycles>& stall1,
               const std::pair<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, StallCycles>& stall2) {
                return stall1.second.size() < stall2.second.size();
            });

    auto stall = minStallSizeExecutor->second.begin();
    auto targetStall = *stall;
    bool overlapExists = false;

    while (stall != minStallSizeExecutor->second.end()) {
        for (auto& otherExecutor : executorStalls) {
            overlapExists = false;
            for (auto& otherStalls : otherExecutor.second) {
                // other stall before
                if (targetStall.first >= otherStalls.second) {
                    continue;
                }
                // other stall after
                if (targetStall.second <= otherStalls.first) {
                    continue;
                }
                // overlap exists
                overlapExists = true;
                targetStall.first = std::max(targetStall.first, otherStalls.first);
                targetStall.second = std::min(targetStall.second, otherStalls.second);
            }
            if (!overlapExists) {
                // if stall not on on all executors
                break;
            }
        }

        if (overlapExists) {
            // stall can be removed
            stallsOnAllExecutorPipelines.push_back(targetStall);
            // try to find another overlapping range
            targetStall.first = targetStall.second;
            targetStall.second = stall->second;
        } else {
            // move to the next stall
            ++stall;
            // target next stall
            if (stall != minStallSizeExecutor->second.end()) {
                targetStall = *stall;
            }
        }
    }

    return stallsOnAllExecutorPipelines;
}

void vpux::verifyDependenciesPreservedInCycles(AsyncDepsInfo& depsInfo, ScheduledOpInfoVec& scheduledOps) {
    for (auto& schedOp : scheduledOps) {
        auto execOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        auto cycleEnd = getAsyncExecuteCycleEnd(execOp);
        for (auto con : depsInfo.getConsumerOps(schedOp.op_)) {
            auto conExecOp = depsInfo.getExecuteOpAtIndex(con);
            auto cycleBegin = getAsyncExecuteCycleBegin(conExecOp);
            // all consumers must execute after the dependency
            VPUX_THROW_UNLESS(cycleEnd <= cycleBegin, "Dependencies not preserved {0} {1}", schedOp.op_, con);
        }
    }
}

StringRef vpux::getTaskType(const ScheduledOpInfo& op) {
    StringRef taskType = "DPU";
    if (op.isOriginalSpillReadOp()) {
        taskType = "DMA-spill-in";
    } else if (op.isOriginalSpillWriteOp()) {
        taskType = "DMA-spill-out";
    } else if (op.queueType.execKind == VPU::ExecutorKind::DMA_NN) {
        if (op.isDataOp()) {
            taskType = "DMA-in";
        } else {
            taskType = "DMA-out";
        }
    }
    return taskType;
}

// This method will analyse the async.execute operations and provide some scheduling statistics
// regarding stalls and calculate how much they could be reduced
void vpux::printScheduleStatistics(mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo, Logger log,
                                   llvm::ArrayRef<ScheduledOpInfo> scheduledOps) {
    log.setName("schedule-analysis");
    // 1. Print final schedule
    log.trace("Schedule Statistics");
    log = log.nest();
    for (auto& op : scheduledOps) {
        // task type info
        auto taskType = getTaskType(op);
        // cycle info
        auto cycleInfo = std::to_string(op.cycleBegin_) + " -> " + std::to_string(op.cycleEnd_);
        // loc info
        auto execOpLoc = depsInfo.getExecuteOpAtIndex(op.op_).getLoc();
        // per operations statistics
        log.trace("op = {0} type = {1} cycles = {2} name = {3}", op.op_, taskType, cycleInfo, execOpLoc);
    }
    log = log.unnest();

    // 2. Perform schedule analysis on async.execute operations
    // store current cycle for every executor
    log.trace("Schedule Analysis");
    log = log.nest();
    mlir::DenseMap<StringRef, int64_t> executorPipeline;
    mlir::DenseMap<StringRef, int64_t> executorFreeCycles;
    int64_t totalPossibleImprovement = 0;
    int64_t pipelineEndCycle = 0;

    auto firstAsyncExecOp = netFunc.getOps<mlir::async::ExecuteOp>().begin();
    while (firstAsyncExecOp != netFunc.getOps<mlir::async::ExecuteOp>().end()) {
        auto execOp = *firstAsyncExecOp;
        if (!execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
            ++firstAsyncExecOp;
            continue;
        }
        if (!execOp->hasAttr(cycleBegin)) {
            ++firstAsyncExecOp;
            continue;
        }
        if (!execOp->hasAttr(cycleEnd)) {
            ++firstAsyncExecOp;
            continue;
        }
        const auto executor = VPUIP::VPUIPDialect::getExecutor(execOp);
        auto opCycleStart = execOp->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getValue().getSExtValue();
        auto opCycleEnd = execOp->getAttr(cycleEnd).cast<mlir::IntegerAttr>().getValue().getSExtValue();
        pipelineEndCycle = std::max(pipelineEndCycle, opCycleEnd);

        // generic analysis
        auto executorName = executor.getLeafName();
        if (executorPipeline.find(executorName) == executorPipeline.end()) {
            // check if could be executed sooner
            executorPipeline[executorName] = opCycleEnd;
            executorFreeCycles[executorName] = 0;
        } else {
            // check if could be executed earlier
            if (opCycleStart > executorPipeline[executorName]) {
                // free cycles on the pipeline
                executorFreeCycles[executorName] += opCycleStart - executorPipeline[executorName];
                // check if dependency is causing the stall
                auto execOpIdx = depsInfo.getIndex(execOp);
                int64_t eraliestExecutionStartCycle = 0;
                for (auto depIdx : depsInfo.getOpDeps(execOpIdx)) {
                    auto depExecOp = depsInfo.getExecuteOpAtIndex(depIdx);
                    if (!depExecOp->hasAttr(cycleEnd)) {
                        ++firstAsyncExecOp;
                        continue;
                    }
                    auto depOpCycleEnd =
                            depExecOp->getAttr(cycleEnd).cast<mlir::IntegerAttr>().getValue().getSExtValue();
                    eraliestExecutionStartCycle = std::max(eraliestExecutionStartCycle, depOpCycleEnd);
                }
                if (opCycleStart > eraliestExecutionStartCycle) {
                    // operation could be executed earlier
                    log.trace("op = {0} could execute earlier", execOpIdx);
                    // check if it is delaying other pipelines
                    int64_t consumerStartCycle = std::numeric_limits<int64_t>::max();
                    auto consumerExecutorName = executorName;
                    size_t consumerIdx = 0;
                    // find earliest executing consumer
                    for (auto conIdx : depsInfo.getConsumerOps(execOpIdx)) {
                        auto conExecOp = depsInfo.getExecuteOpAtIndex(conIdx);
                        if (!conExecOp->hasAttr(cycleEnd)) {
                            ++firstAsyncExecOp;
                            continue;
                        }
                        auto conOpCycleStart =
                                conExecOp->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getValue().getSExtValue();
                        if (conOpCycleStart < consumerStartCycle) {
                            consumerIdx = conIdx;
                            consumerStartCycle = conOpCycleStart;
                            consumerExecutorName = VPUIP::VPUIPDialect::getExecutor(conExecOp).getLeafName();
                        }
                    }
                    // check if stalls exist
                    if (consumerStartCycle > executorPipeline[consumerExecutorName]) {
                        // consumer op is delayed
                        // check if consumer op could execute earlier
                        if (executorFreeCycles[consumerExecutorName] > 0) {
                            auto possibleImprovement =
                                    std::min(executorFreeCycles[consumerExecutorName],
                                             consumerStartCycle - executorPipeline[consumerExecutorName]);
                            log.trace("op = {0} delays consumer = {1} by {2} cycles", execOpIdx, consumerIdx,
                                      possibleImprovement);
                            totalPossibleImprovement += possibleImprovement;
                        }
                    }
                }
            }
            executorPipeline[executorName] = opCycleEnd;
        }

        ++firstAsyncExecOp;
    }
    log = log.unnest();

    // 3. Print analysis statistics
    log.trace("Schedule Statistics:");
    log = log.nest();
    log.trace("Total Pipeline Cycles      = {0} cycles", pipelineEndCycle);
    log.trace("Total Possible Improvement = {0} cycles", totalPossibleImprovement);
    log.trace("Total Possible Improvement = {0} %",
              (100.0 * totalPossibleImprovement / (pipelineEndCycle != 0 ? pipelineEndCycle : 1)));
    log = log.unnest();

    // NOTE: not fully accurate due to unpredictable DPU order of execution
}

SpillStats vpux::getDynamicSpillingStats(llvm::ArrayRef<ScheduledOpInfo> scheduledOps) {
    size_t spillWriteNum = 0;
    size_t spillWriteFragNum = 0;
    size_t spillWriteOfDataOpNum = 0;
    size_t spillReadNum = 0;
    for (auto& op : scheduledOps) {
        if (op.isSpillWrite()) {
            spillWriteNum++;
            if (op.isSpillWriteFrag()) {
                spillWriteFragNum++;
            }
            if (op.isDataOp()) {
                spillWriteOfDataOpNum++;
            }
        } else if (op.isSpillRead()) {
            spillReadNum++;
        }
    }
    return {spillWriteNum, spillWriteFragNum, spillWriteOfDataOpNum, spillReadNum};
}

void vpux::printSpillingStatistics(Logger log, SpillStats& beforePrefetching, SpillStats& afterPrefetching,
                                   SpillStats& afterOptimizations) {
    log.setName("schedule-analysis");
    log.trace("Dynamic spilling statistics");
    log = log.nest();

    log.trace("Spilling before prefetching:");
    log.nest().trace("SPILL_WRITE = {0}\t SPILL_WRITE_FRAG = {1}\t SPILL_WRITE_DATA = {2}\t SPILL_READ = {3}",
                     beforePrefetching.numOfSpillWrites, beforePrefetching.numOfSpillWritesDueToFrag,
                     beforePrefetching.numOfSpillWritesOfDataOps, beforePrefetching.numOfSpillRead);

    log.trace("Spilling after prefetching:");
    log.nest().trace("SPILL_WRITE = {0}\t SPILL_WRITE_FRAG = {1}\t SPILL_WRITE_DATA = {2}\t SPILL_READ = {3}",
                     afterPrefetching.numOfSpillWrites, afterPrefetching.numOfSpillWritesDueToFrag,
                     afterPrefetching.numOfSpillWritesOfDataOps, afterPrefetching.numOfSpillRead);

    log.trace("Spilling after optimizations:");
    log.nest().trace("SPILL_WRITE = {0}\t SPILL_WRITE_FRAG = {1}\t SPILL_WRITE_DATA = {2}\t SPILL_READ = {3}",
                     afterOptimizations.numOfSpillWrites, afterOptimizations.numOfSpillWritesDueToFrag,
                     afterOptimizations.numOfSpillWritesOfDataOps, afterOptimizations.numOfSpillRead);
    log = log.unnest();
}

void vpux::createTracingJSON(mlir::func::FuncOp& netFunc, StringRef fileName) {
    std::ofstream out_stream(fileName.str());
    VPUX_THROW_UNLESS(out_stream.good(), "File to dump traces to is not created correctly");

    profiling::TraceEventDesc ted;
    ted.pid = 0;

    out_stream << std::setprecision(0) << "{\"traceEvents\":[" << std::endl;

    // store all operation info in struct
    for (auto execOp : netFunc.getOps<mlir::async::ExecuteOp>()) {
        const auto attr = execOp->getAttrOfType<mlir::IntegerAttr>("async-deps-index");
        const auto index = checked_cast<uint32_t>(attr.getValue().getZExtValue());
        auto indexString = std::to_string(index);
        auto executor = VPUIP::VPUIPDialect::getExecutor(execOp).getLeafName().str();
        auto cycleBegin = getAsyncExecuteCycleBegin(execOp);
        auto cycleEnd = getAsyncExecuteCycleEnd(execOp);

        ted.name = std::move(indexString);
        ted.category = executor;
        ted.tid = executorStrToId.at(executor);
        ted.timestamp = cycleBegin;
        ted.duration = cycleEnd - cycleBegin;
        out_stream << ted;
    }

    // add meta information
    out_stream << std::string(R"({"name": "process_name", "ph": "M", "pid": )") << 0 << R"(, "tid": )" << 0
               << R"(, "args": {"name" : "Inference"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << 0 << R"(, "tid": )" << 0
               << R"(, "args": {"name" : "DMA_NN"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << 0 << R"(, "tid": )" << 1
               << R"(, "args": {"name" : "DPU"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << 0 << R"(, "tid": )" << 2
               << R"(, "args": {"name" : "NCE"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << 0 << R"(, "tid": )" << 4
               << R"(, "args": {"name" : "SHAVE_ACT"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << 0 << R"(, "tid": )" << 5
               << R"(, "args": {"name" : "SHAVE_NN"}},)" << std::endl;
    out_stream << "]";
    out_stream << "}" << std::endl;
}
