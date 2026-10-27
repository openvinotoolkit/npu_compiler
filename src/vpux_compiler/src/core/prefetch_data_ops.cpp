//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/prefetch_data_ops.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

PrefetchDataOps::PrefetchDataOps(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo)
        : _log(Logger::global().nest("prefetch-data-ops", 0)), _scheduledOps(initialSchedule), _depsInfo(depsInfo) {
}

void PrefetchDataOps::init() {
    for (const auto& op : _scheduledOps) {
        // store cycle cost for ops
        _operationCycleCost[op.op_] = op.cycleEnd_ - op.cycleBegin_;

        if (op.queueType.execKind == config::ExecutorKind::DMA_NN) {
            if (op.isSpillWrite() || op.isSpillRead()) {
                ++_dynamicSpillCount;
            }
            ++_dmaCount;
            if (op.isDataOp()) {
                // store data ops
                _dataOpIdx.insert(op.op_);
            }
        } else {
            // store compute executor type ops
            _computeExecutorKindOpIdx.insert(op.op_);
        }
    }
}

bool PrefetchDataOps::isScheduled(size_t opIdx) {
    return _operationCycles.find(opIdx) != _operationCycles.end();
}

bool PrefetchDataOps::hasDependencies(size_t opIdx) {
    // order-independent: emptiness check only
    return !_depsInfo.getOpDepsUnordered(opIdx).empty();
}

size_t PrefetchDataOps::getOperationCycleCost(size_t opIdx) {
    auto opCycleCostItr = _operationCycleCost.find(opIdx);
    VPUX_THROW_WHEN(opCycleCostItr == _operationCycleCost.end(), "No cycle cost for '{0}'", opIdx);
    return opCycleCostItr->second;
}

size_t PrefetchDataOps::getNextFreePipelineCycle(config::ExecutorKind executorKind, bool prefetchFIFO) {
    if (prefetchFIFO) {
        return _prefetchPipeline.empty() ? 0 : _prefetchPipeline.rbegin()->getCycleEnd();
    }
    return _executorPipelineCycles[executorKind].empty()
                   ? 0
                   : _executorPipelineCycles[executorKind].rbegin()->getCycleEnd();
}

PrefetchDataOps::CycleInfo PrefetchDataOps::scheduleOp(size_t opIdx, size_t cycleBegin, size_t cycleCost,
                                                       config::ExecutorKind executorKind, bool prefetchFIFO) {
    auto cycleInfo = CycleInfo(opIdx, cycleBegin, cycleCost, executorKind);
    if (prefetchFIFO) {
        _prefetchPipeline.push_back(cycleInfo);
    } else {
        _executorPipelineCycles[executorKind].push_back(cycleInfo);
    }
    _operationCycles[opIdx] = cycleInfo;
    return cycleInfo;
}

bool PrefetchDataOps::dependenciesScheduled(size_t opIdx) {
    // order-independent: all-of predicate over the dependency set
    for (const auto depIdx : _depsInfo.getOpDepsUnordered(opIdx)) {
        if (!isScheduled(depIdx)) {
            return false;
        }
    }
    return true;
}

size_t PrefetchDataOps::getDependencyCycleEnd(size_t opIdx) {
    size_t cycleEnd = std::numeric_limits<size_t>::min();
    // order-independent: max over the dependency set
    for (const auto depIdx : _depsInfo.getOpDepsUnordered(opIdx)) {
        VPUX_THROW_UNLESS(isScheduled(depIdx), "Dependency '{0}' was not scheduled for '{1}'", depIdx, opIdx);
        cycleEnd = std::max(cycleEnd, _operationCycles[depIdx].getCycleEnd());
    }
    return cycleEnd;
}

bool PrefetchDataOps::prefetchPipelineStallsExist() {
    return !_prefetchPipelineStalls.empty();
}

// move DMAs earlier on prefetch DMA FIFO to have more free cycles
void PrefetchDataOps::updatePrefetchPipeline(size_t fromCycle, size_t toCycle, size_t adjustCycles) {
    _log.nest(5).trace("updatePrefetchPipeline from '{0}' to '{1}' cycles '{2}'", fromCycle, toCycle, adjustCycles);

    for (auto& op : _prefetchPipeline | reversed) {
        if (op.getCycleEnd() > toCycle) {
            // op not impacted by stall
            continue;
        }

        if (op.getCycleBegin() < fromCycle) {
            // ops not impacted
            break;
        }

        // adjust cycles
        _log.nest(6).trace("adjusting cycle from '{0}' to '{1}'", op.cycleBegin_, op.cycleBegin_ - adjustCycles);
        op.cycleBegin_ -= adjustCycles;
    }
}

// schedule prefetch DMAs closer to optimal cycle end by reducing stalls
// on prefetch DMA FIFO
void PrefetchDataOps::reducePrefetchPipelineStalls(size_t targetRollback) {
    // increase prefetching
    size_t cyclesFreed = 0;
    auto lastStall = _prefetchPipelineStalls.rbegin();

    // figure out how many stalls need to be eliminated
    while (lastStall != _prefetchPipelineStalls.rend() && cyclesFreed < targetRollback) {
        auto stallCycles = lastStall->getCycleCost();
        cyclesFreed += stallCycles;
        ++lastStall;
    }

    // need to shrink stall in some cases
    cyclesFreed = std::min(cyclesFreed, targetRollback);
    _log.nest(3).trace("Need to get back '{0}' cycles, can get '{1}'", targetRollback, cyclesFreed);

    // eliminate stalls on prefetch DMA FIFO
    auto stallToRemove = _prefetchPipelineStalls.rbegin();
    size_t rangeEnd = std::numeric_limits<size_t>::max();
    while (stallToRemove != lastStall) {
        auto rangeBegin = stallToRemove->getCycleEnd();
        // move DMAs earlier
        updatePrefetchPipeline(rangeBegin, rangeEnd, cyclesFreed);

        rangeEnd = stallToRemove->getCycleBegin();
        cyclesFreed -= stallToRemove->getCycleCost();

        if (cyclesFreed >= stallToRemove->getCycleCost()) {
            // erase stall since all cycles used
            std::advance(stallToRemove, 1);
            _prefetchPipelineStalls.erase(stallToRemove.base());
        }
    }
}

// schedule prefetch DMA operation in prefetch FIFO, can create stalls
// stalls on prefetch FIFO are stored and can be reduced by prefetching DMAs earlier
void PrefetchDataOps::schedulePrefetchDMA(size_t opIdx, size_t optimalCycleEnd) {
    auto earliestScheduleCycle = getNextFreePipelineCycle(config::ExecutorKind::DMA_NN, true);
    const auto cycleCost = getOperationCycleCost(opIdx);
    auto cycleEnd = earliestScheduleCycle + cycleCost;

    if (cycleEnd < optimalCycleEnd) {
        // stall created on prefetch FIFO, store this stall so later if needed it can be removed
        _log.nest(4).trace("Stall created from '{0}' to '{1}' of size '{2}'", earliestScheduleCycle,
                           optimalCycleEnd - cycleCost, optimalCycleEnd - cycleCost - earliestScheduleCycle);
        _prefetchPipelineStalls.push_back(CycleInfo(_cycleInfoStallDummyOp, earliestScheduleCycle,
                                                    optimalCycleEnd - cycleCost - earliestScheduleCycle));
        cycleEnd = optimalCycleEnd;
    } else if (cycleEnd > optimalCycleEnd) {
        // cycle end is after optimal cycle end, try to eliminate stalls on prefetch FIFO
        if (prefetchPipelineStallsExist()) {
            reducePrefetchPipelineStalls(cycleEnd - optimalCycleEnd);
        }
        // always schedule on optimal cycles
        cycleEnd = std::max(cycleCost, optimalCycleEnd);
    }

    // use updated cycles
    auto cycleBegin = cycleEnd - cycleCost;
    auto cycleInfo = scheduleOp(opIdx, cycleBegin, cycleCost, config::ExecutorKind::DMA_NN, true);
    _log.nest().trace("Prefetch DMA '{0}' cycles '{1}'->'{2}' with optimal end '{3}'", cycleInfo.getOpIdx(),
                      cycleInfo.getCycleBegin(), cycleInfo.getCycleEnd(), optimalCycleEnd);
}

// schedule DMA operation in compute DMA FIFO
void PrefetchDataOps::scheduleDMA(size_t opIdx, size_t optimalCycleEnd) {
    // find earliest scheduling cycle for op
    auto earliestScheduleCycle =
            std::max(getNextFreePipelineCycle(config::ExecutorKind::DMA_NN), getDependencyCycleEnd(opIdx));
    // use earliest scheduling cycle or optimal cycle end
    const auto cycleCost = getOperationCycleCost(opIdx);
    if (optimalCycleEnd >= cycleCost) {
        earliestScheduleCycle = std::max(earliestScheduleCycle, optimalCycleEnd - cycleCost);
    }
    // schedule operation
    auto cycleInfo = scheduleOp(opIdx, earliestScheduleCycle, cycleCost, config::ExecutorKind::DMA_NN);
    _log.nest().trace("Schedule DMA '{0}' on cycles '{1}'->'{2}' with optimal end '{3}'", cycleInfo.getOpIdx(),
                      cycleInfo.getCycleBegin(), cycleInfo.getCycleEnd(), optimalCycleEnd);
}

// schedule data ops during compute op
void PrefetchDataOps::prefetchDataOps(size_t dmaTargetEndCycle) {
    // schedule all DMAs until target DMA cycle
    auto nextDMA = _prefetchPipeline.begin();
    while (nextDMA != _prefetchPipeline.end()) {
        const auto opIdx = nextDMA->opIdx_;
        if (!dependenciesScheduled(opIdx)) {
            // can not schedule this DMA
            ++nextDMA;
            continue;
        }

        if (isScheduled(opIdx)) {
            nextDMA = _prefetchPipeline.erase(nextDMA);
            continue;
        }

        if (nextDMA->getCycleBegin() > dmaTargetEndCycle) {
            // no more DMAs to schedule during this scheduling cycle
            break;
        }

        scheduleDMA(opIdx);
        nextDMA = _prefetchPipeline.erase(nextDMA);
    }
}

// schedule dependencies for compute op, distribute into prefetch FIFO and compute DMA FIFO
// if scheduling with defined DMA order schedule all DMAs in the same FIFO
size_t PrefetchDataOps::scheduleDependenciesForCompute(size_t opIdx, config::ExecutorKind executorKind) {
    // find earliest possible cycle for compute op, it will be the optimal end cycle for data ops
    auto dataOpTargetCycleEnd = getNextFreePipelineCycle(executorKind);
    for (const auto dep : _depsInfo.getOpDepsUnordered(opIdx)) {
        if (!isScheduled(dep)) {
            continue;
        }
        dataOpTargetCycleEnd = std::max(dataOpTargetCycleEnd, _operationCycles[dep].getCycleEnd());
    }

    // schedule dependencies for compute op and update cycle end for dependencies
    for (const auto depIdx : _depsInfo.getOpDeps(opIdx)) {
        if (isScheduled(depIdx)) {
            continue;
        } else if (_prefetchOpsDefined || hasDependencies(depIdx)) {
            // schedule DMA on compute DMA FIFO
            scheduleDMA(depIdx, dataOpTargetCycleEnd);
        } else {
            // schedule DMA on prefetch FIFO
            schedulePrefetchDMA(depIdx, dataOpTargetCycleEnd);
        }

        dataOpTargetCycleEnd = std::max(dataOpTargetCycleEnd, _operationCycles[depIdx].getCycleEnd());
    }

    return dataOpTargetCycleEnd;
}

// schedule compute operation, before schedule dependencies for the compute operation
void PrefetchDataOps::scheduleComputeOperation(size_t opIdx, config::ExecutorKind executorKind) {
    _log.trace("Scheduling compute '{0}' on '{1}'", opIdx, executorKind);
    auto computeCycleBegin = scheduleDependenciesForCompute(opIdx, executorKind);
    const auto computeOpCycleCost = getOperationCycleCost(opIdx);

    if (_prefetchOpsDefined && executorKind != config::ExecutorKind::DMA_NN) {
        prefetchDataOps(computeCycleBegin + computeOpCycleCost);
        // all prefetch ops should be during this compute executor op
        auto cycleEnd = computeCycleBegin + computeOpCycleCost;
        cycleEnd = std::max(cycleEnd, getNextFreePipelineCycle(config::ExecutorKind::DMA_NN));
        computeCycleBegin = cycleEnd - computeOpCycleCost;
    }

    auto cycleInfo = scheduleOp(opIdx, computeCycleBegin, computeOpCycleCost, executorKind);
    _log.trace("Scheduled compute '{0}' cycles '{1}'->'{2}' on '{3}'", cycleInfo.getOpIdx(), cycleInfo.getCycleBegin(),
               cycleInfo.getCycleEnd(), cycleInfo.getExecutorKind());
}

// re-iterate the schedule and store cycle cost of ops
// schedule compute operations
void PrefetchDataOps::performCycleScheduling() {
    _log.trace("Performing cycle scheduling, with prefetching = '{0}'", _prefetchOpsDefined);
    for (const auto& op : _scheduledOps) {
        // skip implicit spill and profiling operations
        if (!op.isOriginalOp()) {
            continue;
        }

        if (op.isDataOp() || (_prefetchOpsDefined && isScheduled(op.op_))) {
            // ops scheduled during compute dependency scheduling
            continue;
        } else {
            scheduleComputeOperation(op.op_, op.queueType.execKind);
        }
    }
}

// create a common FIFO for data ops
void PrefetchDataOps::createDataOpPipeline() {
    // create a new pipeline for data ops
    SmallVector<CycleInfo> DataOpPipeline;

    auto dmaFrontRev = _executorPipelineCycles[config::ExecutorKind::DMA_NN].rbegin();
    auto prefetchFrontRev = _prefetchPipeline.rbegin();
    size_t dmaFIFORev = std::numeric_limits<size_t>::max();

    _log.trace("Creating data op prefetch pipeline:");
    while (dmaFrontRev != _executorPipelineCycles[config::ExecutorKind::DMA_NN].rend() ||
           prefetchFrontRev != _prefetchPipeline.rend()) {
        auto dmaCycleBegin = std::numeric_limits<size_t>::min();
        if (dmaFrontRev != _executorPipelineCycles[config::ExecutorKind::DMA_NN].rend()) {
            if (_dataOpIdx.find(dmaFrontRev->getOpIdx()) == _dataOpIdx.end()) {
                ++dmaFrontRev;
                continue;
            }
            dmaCycleBegin = dmaFrontRev->getCycleEnd();
        }

        auto prefetchCycleBegin = std::numeric_limits<size_t>::min();
        if (prefetchFrontRev != _prefetchPipeline.rend()) {
            prefetchCycleBegin = prefetchFrontRev->getCycleBegin();
        }

        if (prefetchCycleBegin >= dmaCycleBegin) {
            auto newCycleEnd = std::min(dmaFIFORev, prefetchFrontRev->getCycleEnd());
            newCycleEnd = std::max(newCycleEnd, prefetchFrontRev->getCycleCost());
            auto newCycleBegin = newCycleEnd - prefetchFrontRev->getCycleCost();
            // schedule prefetch DMA before compute DMA
            _log.nest().trace("Add prefetch DMA: '{0}', with optimal cycles '{1}' -> '{2}'",
                              prefetchFrontRev->getOpIdx(), newCycleBegin, newCycleEnd);
            DataOpPipeline.push_back(
                    CycleInfo(prefetchFrontRev->getOpIdx(), newCycleBegin, prefetchFrontRev->getCycleCost()));
            ++prefetchFrontRev;
        } else {
            auto newCycleEnd = std::min(dmaFIFORev, dmaFrontRev->getCycleEnd());
            newCycleEnd = std::max(newCycleEnd, dmaFrontRev->getCycleCost());
            auto newCycleBegin = newCycleEnd - dmaFrontRev->getCycleCost();
            // schedule compute DMA
            _log.nest().trace("Add compute DMA: '{0}', with optimal cycles '{1}' -> '{2}'", dmaFrontRev->getOpIdx(),
                              newCycleBegin, newCycleEnd);
            DataOpPipeline.push_back(CycleInfo(dmaFrontRev->getOpIdx(), newCycleBegin, dmaFrontRev->getCycleCost()));
            ++dmaFrontRev;
        }

        dmaFIFORev = DataOpPipeline.rbegin()->getCycleBegin();
    }

    std::reverse(DataOpPipeline.begin(), DataOpPipeline.end());
    // use prefetch pipeline as order for all data ops
    _prefetchPipeline = std::move(DataOpPipeline);
    // reset for second scheduling iteration with defined DMA order
    _operationCycles.clear();
    _executorPipelineCycles = {{config::ExecutorKind::DMA_NN, {}},    {config::ExecutorKind::DPU, {}},
                               {config::ExecutorKind::NCE, {}},       {config::ExecutorKind::SHAVE_NN, {}},
                               {config::ExecutorKind::SHAVE_ACT, {}}, {config::ExecutorKind::M2I, {}}};
}

void PrefetchDataOps::sortOps(SmallVector<CycleInfo>& toBeSorted) {
    llvm::sort(toBeSorted.begin(), toBeSorted.end(), [](const CycleInfo& op1, const CycleInfo& op2) {
        // first cycle begin
        if (op1.getCycleBegin() != op2.getCycleBegin()) {
            return op1.getCycleBegin() < op2.getCycleBegin();
        }

        // DMA ops first
        if (op1.getExecutorKind() != op2.getExecutorKind()) {
            if (op1.getExecutorKind() == config::ExecutorKind::DMA_NN) {
                return true;
            }
            if (op2.getExecutorKind() == config::ExecutorKind::DMA_NN) {
                return false;
            }
        }

        // second cycle end
        if (op1.getCycleEnd() != op2.getCycleEnd()) {
            return op1.getCycleEnd() < op2.getCycleEnd();
        }

        // operation index
        return op1.getOpIdx() < op2.getOpIdx();
    });
}

SmallVector<PrefetchDataOps::CycleInfo> PrefetchDataOps::getNewOrder() {
    SmallVector<CycleInfo> sortedComputeAndDMAOps;
    SmallVector<CycleInfo> sortedDataOps;

    // distribute ops into data ops and compute ops with compute DMAs
    for (const auto& op : _operationCycles) {
        if (_dataOpIdx.find(op.first) != _dataOpIdx.end()) {
            sortedDataOps.push_back(op.second);
        } else {
            sortedComputeAndDMAOps.push_back(op.second);
        }
    }

    // sort all operation cycles
    sortOps(sortedDataOps);
    sortOps(sortedComputeAndDMAOps);

    std::set<size_t> scheduledOps;
    const auto dependenciesScheduled = [&](size_t opIdx) {
        for (const auto depIdx : _depsInfo.getOpDepsUnordered(opIdx)) {
            if (scheduledOps.find(depIdx) == scheduledOps.end()) {
                return false;
            }
        }
        return true;
    };

    const auto getNextComputeCycleBegin = [&](CycleInfo* computeItr) {
        auto cycleBegin = std::numeric_limits<size_t>::max();
        auto temp = computeItr;
        size_t computeCount = 0;
        while (temp != sortedComputeAndDMAOps.end() && computeCount < _advanceComputeExecutorKindOpsForPrefetch) {
            if (_computeExecutorKindOpIdx.find(temp->getOpIdx()) != _computeExecutorKindOpIdx.end()) {
                ++computeCount;
            }

            cycleBegin = temp->getCycleBegin();
            ++temp;
        }

        return cycleBegin;
    };

    SmallVector<CycleInfo> sortedOpCycles;
    _log.trace("Defining new order:");
    const auto scheduleOp = [&](CycleInfo* opItr) {
        sortedOpCycles.push_back(*opItr);
        scheduledOps.insert(opItr->getOpIdx());
        _log.nest().trace("opIdx = '{0}', cycles = '{1}' -> '{2}', executor = '{3}'", opItr->getOpIdx(),
                          opItr->getCycleBegin(), opItr->getCycleEnd(), opItr->getExecutorKind());
    };

    // generate new order for operations based on cycles
    auto computeItr = sortedComputeAndDMAOps.begin();
    while (!sortedDataOps.empty() || computeItr != sortedComputeAndDMAOps.end()) {
        auto dataItr = sortedDataOps.begin();
        if (dataItr == sortedDataOps.end()) {
            scheduleOp(computeItr);
            ++computeItr;
            continue;
        }

        if (computeItr == sortedComputeAndDMAOps.end()) {
            scheduleOp(dataItr);
            ++dataItr;
            continue;
        }

        // compare to next compute cycle begin
        const auto nextComputeCycleBegin = getNextComputeCycleBegin(computeItr);
        while (dataItr != sortedDataOps.end()) {
            auto dataCycleBegin = dataItr->getCycleBegin();
            if (dataCycleBegin >= nextComputeCycleBegin) {
                break;
            }

            if (!dependenciesScheduled(dataItr->getOpIdx())) {
                ++dataItr;
                continue;
            }

            // schedule data op
            scheduleOp(dataItr);
            dataItr = sortedDataOps.erase(dataItr);
        }

        // schedule compute op or compute DMA
        scheduleOp(computeItr);
        ++computeItr;
    }

    return sortedOpCycles;
}

void PrefetchDataOps::reorderToPrefetch(ArrayRef<CycleInfo> sortedOpCycles) {
    // reorder IR such that DMAs to prefetch are earlier
    mlir::Operation* prevAsyncOp = nullptr;
    for (const auto& opCycles : sortedOpCycles) {
        _log.trace("Op '{0}' cycles '{1}'->'{2}' on '{3}'", opCycles.opIdx_, opCycles.getCycleBegin(),
                   opCycles.getCycleEnd(), opCycles.getExecutorKind());

        mlir::Operation* asyncOp = _depsInfo.getExecuteOpAtIndex(opCycles.opIdx_);
        VPUX_THROW_WHEN(asyncOp == nullptr, "AsyncOp not located based on index");
        if (prevAsyncOp != nullptr) {
            asyncOp->moveAfter(prevAsyncOp);
        } else {
            mlir::Operation* firstAsyncExecOp = _depsInfo.getExecuteOpAtIndex(0);
            asyncOp->moveBefore(firstAsyncExecOp);
        }
        prevAsyncOp = asyncOp;
    }
}

bool PrefetchDataOps::enableDataOpPrefetching() {
    // store information about ops
    init();

    // avoid regenerating schedule with prefetching if dominated by dynamic spills to save compile time
    // large number of dynamic spills suggests sub-optimal strategy and optimization
    // performant execution requires schedule and DMA optimization
    if (_dynamicSpillCount > _dynamicSpillMinCount && _dynamicSpillCount > _dmaCount * _dynamicSpillRatio) {
        _log.info("Dynamic spills dominate schedule, optimizations needed for prefetching");
        return false;
    }

    // re-schedule operations where data ops and compute DMA ops have their own FIFOs
    // schedule data ops optimally just before the compute if possible, this may create stalls
    // future DMAs may have earlier optimal scheduling cycle than available, in this case
    // try to see in any stall exist on prefetch pipeline and prefetch all DMAs earlier
    performCycleScheduling();

    // merge FIFO for data ops and compute ops to create an ordered FIFO for all DMAs
    // this will be the new order of DMA operations
    createDataOpPipeline();

    // create new schedule which can prefetch data ops
    _prefetchOpsDefined = true;
    performCycleScheduling();

    // based on new schedule generate a new order for operations
    auto newOpOrder = getNewOrder();
    // reorder IR to new order
    reorderToPrefetch(newOpOrder);

    return true;
}
