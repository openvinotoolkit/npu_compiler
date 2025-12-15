//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

using namespace vpux;
using operationIdxType = FeasibleMemoryScheduler::operationIdxType;
//
// Feasible Memory Scheduler
//

// This class will try to produce a feasible memory schedule based on the dependency map provided from
// AsyncDepsInfo and use the LinearScan class to allocate the resources.
// Data and Compute ops, where Data ops are operations moving data to CMX are distinguished in order to
// follow the scheduling of Compute ops along with their dependencies (Data ops). This optimizes CMX usage,
// and allows for feasible CMX schedule to be generated.
// The graph is iterated topologically based on the dependencies from input to output(s).
// In init() ready lists will be populated using operations without dependencies.
// In schedulingLoop() there are two possible scenarios:
// 1. Scheduling the next earliest operation from the start cycle heap, and adding it to the op output table.
// 2. Un-scheduling operations: freeing CMX space and updating dependencies, creating new ready
//      operations which will be allocated at the next available cycle.

FeasibleMemoryScheduler::FeasibleMemoryScheduler(VPU::MemoryKind memKind, VPU::MemoryKind secondLvlMemKind,
                                                 MemLiveRangeInfo& liveRangeInfo, AsyncDepsInfo& depsInfo, Logger log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan,
                                                 config::ArchKind arch, std::shared_ptr<VPUNN::VPUCostModel> costModel,
                                                 int64_t nceClusterCount, int64_t dmaCount,
                                                 bool enableScheduleStatistics, bool optimizeFragmentation,
                                                 bool activelySpillForPrefetching)
        : _log(log),
          _memKind(memKind),
          _secondLvlMemKind(secondLvlMemKind),
          _liveRangeInfo(liveRangeInfo),
          _depsInfo(depsInfo),
          _scan(scan),
          _archKind(arch),
          _costModel(std::move(costModel)),
          _nceClusterCount(nceClusterCount),
          _numDMAPorts(dmaCount),
          _enableScheduleStatistics(enableScheduleStatistics),
          _optimizeFragmentation(optimizeFragmentation),
          _activelySpillForPrefetching(activelySpillForPrefetching) {
    _log.setName("feasible-memory-scheduler-allocator");

    auto dmaChannels = getDMAChannelsWithIndependentLinkAgents(arch);
    for (auto dmaChannel : dmaChannels) {
        QueueType queueType;
        queueType.execKind = VPU::ExecutorKind::DMA_NN;
        queueType.id = getDMAQueueIdEncoding(dmaChannel);
        _executorPipelines[queueType].assign(dmaCount, 1);
    }
}

struct FeasibleMemoryScheduler::ScheduleStateSnapshot {
    FeasibleMemoryScheduler& scheduler;
    bool captured = false;
    bool finalized = false;

    // Captured state variables from the scheduler.
    decltype(scheduler._readyComputeOps) readyComputeOps;
    LinearScan<mlir::Value, LinearScanHandler> scan;
    decltype(scheduler._executorPipelines) executorPipelines;
    decltype(scheduler._cycleBeginHeap) cycleBeginHeap;
    decltype(scheduler._readyDataOps) readyDataOps;
    decltype(scheduler._readySpilledOps) readySpilledOps;
    decltype(scheduler._spillBufferMap) spillBufferMap;

    // Constructor to optionally capture the scheduler's state.
    ScheduleStateSnapshot(FeasibleMemoryScheduler& sched, bool needCapture): scheduler(sched), scan(sched._scan) {
        if (!needCapture) {
            return;
        }
        captured = true;  // Mark the state as captured.
        readyComputeOps = scheduler._readyComputeOps;
        scan = scheduler._scan;
        executorPipelines = scheduler._executorPipelines;
        cycleBeginHeap = scheduler._cycleBeginHeap;
        readyDataOps = scheduler._readyDataOps;
        readySpilledOps = scheduler._readySpilledOps;
        spillBufferMap = scheduler._spillBufferMap;
    }

    // Restores the scheduler's state to the captured snapshot.
    void rollback() {
        if (!captured || finalized) {
            return;  // If the state was not captured or already finalized, do nothing.
        }
        scheduler._readyComputeOps = std::move(readyComputeOps);
        scheduler._scan = std::move(scan);
        scheduler._executorPipelines = std::move(executorPipelines);
        scheduler._cycleBeginHeap = std::move(cycleBeginHeap);
        scheduler._readyDataOps = std::move(readyDataOps);
        scheduler._readySpilledOps = std::move(readySpilledOps);
        scheduler._spillBufferMap = std::move(spillBufferMap);
        finalized = true;
    }
};

bool compareHeapOrderWhenCycleMatch(const FeasibleMemoryScheduler::HeapElement& a,
                                    const FeasibleMemoryScheduler::HeapElement& b) {
    if (a.isPrefetched() && !b.isPrefetched()) {
        return true;
    }
    if (!a.isPrefetched() && b.isPrefetched()) {
        return false;
    }
    if (a.spillBuffer_ != nullptr && b.spillBuffer_ != nullptr && a.spillBuffer_ != b.spillBuffer_) {
        return ValueOrderCmp::compare(a.spillBuffer_, b.spillBuffer_);
    }
    return a.op_ < b.op_;
}

// Sort heap by earliest begin cycle
bool FeasibleMemoryScheduler::CycleBeginMinHeapOrdering::operator()(const HeapElement& a, const HeapElement& b) const {
    if (a.cycleBegin_ != b.cycleBegin_) {
        return a.cycleBegin_ < b.cycleBegin_;
    }
    return compareHeapOrderWhenCycleMatch(a, b);
}

// Sort heap by earliest end cycle
bool FeasibleMemoryScheduler::CycleEndMinHeapOrdering::operator()(const HeapElement& a, const HeapElement& b) const {
    if (a.cycleEnd_ != b.cycleEnd_) {
        return a.cycleEnd_ < b.cycleEnd_;
    }
    return compareHeapOrderWhenCycleMatch(a, b);
}

void FeasibleMemoryScheduler::updateBufferCycleUseAndProducer(size_t opIdx, size_t opCycleEnd, const mlir::Value buffer,
                                                              bool isNewProducer) {
    // update buffer producer
    if (isNewProducer) {
        if (_bufferProducer.find(buffer) == _bufferProducer.end()) {
            _newBufferProducersFromScheduleComputeOps.insert(buffer);
        }

        if (_originalBufferProducersFromScheduleComputeOps.find(buffer) ==
            _originalBufferProducersFromScheduleComputeOps.end()) {
            _originalBufferProducersFromScheduleComputeOps[buffer] = _bufferProducer[buffer];
        }

        _bufferProducer[buffer] = opIdx;
    }
    // update last cycle use of buffer
    auto bufferUseCycleEnd = _bufferLastCycleUse.find(buffer);
    if (bufferUseCycleEnd != _bufferLastCycleUse.end()) {
        if (_originalBufferLastCycleUseFromScheduleComputeOps.find(buffer) ==
            _originalBufferLastCycleUseFromScheduleComputeOps.end()) {
            _originalBufferLastCycleUseFromScheduleComputeOps[buffer] = bufferUseCycleEnd->second;
        }
        bufferUseCycleEnd->second = std::max(bufferUseCycleEnd->second, opCycleEnd);
    } else {
        _bufferLastCycleUse[buffer] = opCycleEnd;
        _newBufferLastCycleUsesFromScheduleComputeOps.insert(buffer);
    }
}

void FeasibleMemoryScheduler::pushToCycleBeginHeap(const HeapElement& elem) {
    _cycleBeginHeap.insert(elem);
    // store as writer of output buffers
    if (elem.isSpillReadOp()) {
        updateBufferCycleUseAndProducer(elem.op_, elem.cycleEnd_, elem.spillBuffer_, true);
    } else if (elem.isOriginalOp()) {
        const auto execOp = _depsInfo.getExecuteOpAtIndex(elem.op_);
        for (auto& buffer : _liveRangeInfo.getOutputBuffers(execOp)) {
            updateBufferCycleUseAndProducer(elem.op_, elem.cycleEnd_, buffer, true);
        }
        for (auto& buffer : _liveRangeInfo.getInputBuffers(execOp)) {
            updateBufferCycleUseAndProducer(elem.op_, elem.cycleEnd_, buffer);
        }
    }
    insertInOpIdxCycleEndMap(elem.op_, elem.cycleEnd_);
}

size_t FeasibleMemoryScheduler::findMinScheduledQueueCycle() {
    // for all scheduled ops find the minimal queue cycle end
    size_t targetCycleEnd = std::numeric_limits<size_t>::max();
    std::map<QueueType, size_t> queueMinCycleEnd;
    for (const auto& op : _cycleEndHeap) {
        for (auto execInst : op.executorInstanceMask_.set_bits()) {
            targetCycleEnd = std::min(targetCycleEnd, _executorPipelines[op.queueType_][execInst]);
        }
    }
    return targetCycleEnd;
}

void FeasibleMemoryScheduler::moveFromCycleBeginToCycleEndHeap() {
    // move ops from cycle begin heap to cycle end heap
    for (auto& nextOp : _cycleBeginHeap) {
        _log.nest(2).trace("Move opIdx '{0}'", nextOp.op_);
        // add op to ScheduledOpVec
        populateScheduledOps(nextOp);
        // move to cycle end heap
        _cycleEndHeap.insert(nextOp);
        // decrease outputs if output operation scheduled
        if (_outputOps.find(nextOp.op_) != _outputOps.end()) {
            _outputOps.erase(nextOp.op_);
        }
    }

    _cycleBeginHeap.clear();
}

VPU::ExecutorKind FeasibleMemoryScheduler::getExecutorType(operationIdxType opIdx) {
    if (_spillBufferMap.find(opIdx) != _spillBufferMap.end()) {
        // spilled operation using DMAs for relocation
        return VPU::ExecutorKind::DMA_NN;
    }
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    return vpux::getExecutorType(execOp);
}

FeasibleMemoryScheduler::QueueType FeasibleMemoryScheduler::getQueueType(operationIdxType opIdx) {
    VPUX_THROW_WHEN(_spillBufferMap.find(opIdx) != _spillBufferMap.end(),
                    "Function does not support spilled operations, opIdx - '{0}'", opIdx);

    QueueType queueType;
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        queueType.execKind = VPUIP::VPUIPDialect::getExecutorKind(execOp);

        if (auto dmaTask = vpux::getDmaTypeOp(execOp)) {
            queueType.id = getDMAQueueIdEncoding(dmaTask.getChannelType());
        }
        return queueType;
    }
    // for now treat all other executors as DPU - same as previous implementation
    queueType.execKind = VPU::ExecutorKind::DPU;
    return queueType;
}

// When getting number of ports needed for a task executing on DMA, this
// function determines if based on buffer type execution would require
// multiple ports
bool areMultipleDmaPortsNeeded(mlir::Value buffer) {
    if (auto distType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(buffer.getType())) {
        auto mode = distType.getDistribution().getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
            return true;
        }
    }
    return false;
}

// TODO: In future it might be desired to create some utility functions to gather information about
// the number of executors given operation requires
size_t FeasibleMemoryScheduler::getOpDemandForExecutorsInstances(operationIdxType opIdx, QueueType queueType) {
    auto numOfExecutors = _executorPipelines[queueType].size();
    VPUX_THROW_WHEN(numOfExecutors == 0, "No executor of given type {0} and id {1}", queueType.execKind, queueType.id);
    if (numOfExecutors < 2) {
        return 1;
    }

    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);

    // Current only for DMA tasks:
    // Check if operation works on DistributedBuffers with SEGMENTED mode. In such case
    // such DMA will be later split into per-cluster DMA tasks (unroll-distributed-ops pass).
    // Here assume that this operation will use all executors
    if (queueType.execKind == VPU::ExecutorKind::DMA_NN) {
        const auto usedBufs = _liveRangeInfo.getUsedBuffers(execOp);
        for (auto& buffer : usedBufs) {
            if (areMultipleDmaPortsNeeded(buffer)) {
                return numOfExecutors;
            }
        }
    }

    return 1;
}

size_t FeasibleMemoryScheduler::getBufferDemandForExecutorsInstances(mlir::Value buffer, QueueType queueType) {
    auto numOfExecutors = _executorPipelines[queueType].size();
    if (numOfExecutors < 2) {
        return 1;
    }

    // Current only for DMA tasks:
    // Check if operation works on DistributedBuffers with SEGMENTED mode. In such case
    // such DMA will be later split into per-cluster DMA tasks. Here assume that this operation
    // will use all executors
    if (queueType.execKind == VPU::ExecutorKind::DMA_NN) {
        if (areMultipleDmaPortsNeeded(buffer)) {
            return numOfExecutors;
        }
    }

    return 1;
}

llvm::BitVector FeasibleMemoryScheduler::getExecutorInstanceMask(size_t numOfNeededInstances, QueueType queueType) {
    auto numOfAllInstances = _executorPipelines[queueType].size();

    VPUX_THROW_UNLESS(numOfAllInstances > 0, "No available instances of given queue type");
    VPUX_THROW_UNLESS(numOfNeededInstances == 1 || numOfNeededInstances == numOfAllInstances,
                      "Number of needed executors ('{0}') is different then number of all instances of executor "
                      "('{1}'). This is not "
                      "yet supported",
                      numOfNeededInstances, numOfAllInstances);

    llvm::BitVector executorMask(checked_cast<uint32_t>(numOfAllInstances));

    if (queueType.execKind == VPU::ExecutorKind::DMA_NN) {
        if (numOfNeededInstances == 1) {
            // Find the executor with lowest cycle
            size_t indexMin = 0;
            size_t cycleMin = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i < numOfAllInstances; i++) {
                if (_executorPipelines[queueType][i] < cycleMin) {
                    indexMin = i;
                    cycleMin = _executorPipelines[queueType][i];
                }
            }

            return executorMask.set(checked_cast<uint32_t>(indexMin));
        } else {
            return executorMask.set(0, checked_cast<uint32_t>(numOfAllInstances));
        }
    }

    return executorMask.set(0);
}

llvm::BitVector FeasibleMemoryScheduler::getExecutorInstanceMaskForOp(operationIdxType opIdx, QueueType queueType) {
    // TODO: If executor is configured in the operation read it directly from
    // operation async.execute. Currently this is not needed but in future
    // might be useful in case task distribution is performed by some earlier pass

    auto numOfNeededInstances = getOpDemandForExecutorsInstances(opIdx, queueType);

    return getExecutorInstanceMask(numOfNeededInstances, queueType);
}

llvm::BitVector FeasibleMemoryScheduler::getExecutorInstanceMaskForBuffer(mlir::Value buffer, QueueType queueType) {
    auto numOfNeededInstances = getBufferDemandForExecutorsInstances(buffer, queueType);

    return getExecutorInstanceMask(numOfNeededInstances, queueType);
}

FeasibleMemoryScheduler::QueueAndCycleType FeasibleMemoryScheduler::getCurrentCycleAndExecutorInstanceMask(
        operationIdxType opIdx, size_t depEndCycle) {
    auto queueType = getQueueType(opIdx);
    auto executorInstanceMask = getExecutorInstanceMaskForOp(opIdx, queueType);
    VPUX_THROW_WHEN(executorInstanceMask.none(), "No executor instance found");

    size_t earliestBeginCycle = depEndCycle;
    for (auto instIndex : executorInstanceMask.set_bits()) {
        earliestBeginCycle = std::max(earliestBeginCycle, _executorPipelines[queueType][instIndex]);
    }

    // check if operation cycle begin delayed by dependencies
    for (const auto& dep : _depsInfo.getOpDeps(opIdx)) {
        earliestBeginCycle = std::max(earliestBeginCycle, _opIdxEndCycleMap[dep]);
    }
    return QueueAndCycleType{queueType, std::move(executorInstanceMask), earliestBeginCycle};
}

FeasibleMemoryScheduler::QueueAndCycleType FeasibleMemoryScheduler::getCurrentCycleAndExecutorInstanceMaskForSpill(
        mlir::Value buffer, EOpType spillType, size_t depEndCycle) {
    QueueType queueType;
    queueType.execKind = VPU::ExecutorKind::DMA_NN;
    if (spillType == EOpType::IMPLICIT_SPILL_READ_OP) {
        queueType.id = getDMAQueueIdEncoding(_secondLvlMemKind, _archKind);
    } else {
        queueType.id = getDMAQueueIdEncoding(_memKind, _archKind);
    }

    auto executorInstanceMask = getExecutorInstanceMaskForBuffer(buffer, queueType);

    VPUX_THROW_WHEN(executorInstanceMask.none(), "No executor instance found");

    size_t earliestBeginCycle = depEndCycle;
    for (auto instIndex : executorInstanceMask.set_bits()) {
        earliestBeginCycle = std::max(earliestBeginCycle, _executorPipelines[queueType][instIndex]);
    }

    return QueueAndCycleType{queueType, std::move(executorInstanceMask), earliestBeginCycle};
}

void FeasibleMemoryScheduler::updateCurrentCycleForExecutor(QueueType queueType, llvm::BitVector executorInstanceMask,
                                                            size_t nextAvailableCycle) {
    for (auto execInst : executorInstanceMask.set_bits()) {
        _executorPipelines[queueType][execInst] = nextAvailableCycle;
    }
}

void FeasibleMemoryScheduler::alignExecutors(size_t nextAvailableCycle) {
    for (auto& pipeline : _executorPipelines) {
        auto numOfInst = pipeline.second.size();
        for (size_t i = 0; i < numOfInst; i++) {
            pipeline.second[i] = std::max(pipeline.second[i], nextAvailableCycle);

            std::string executorInstanceInfo = "";

            if (pipeline.first.execKind == VPU::ExecutorKind::DMA_NN) {
                auto channelTypeAsString = getDMAChannelTypeAsString(pipeline.first.id, _archKind);
                if (channelTypeAsString.size() > 0) {
                    executorInstanceInfo += "_" + channelTypeAsString;
                }
            }

            if (numOfInst > 1) {
                executorInstanceInfo += " [" + std::to_string(i) + "]";
            }

            _log.nest().trace("Aligning executor pipeline {0}{1} = {2}", pipeline.first.execKind, executorInstanceInfo,
                              pipeline.second[i]);
        }
    }
}

size_t FeasibleMemoryScheduler::spilledOperationCycleCost(mlir::Value spilledBuffer) {
    if (_spillBufferCycleCost.find(spilledBuffer) != _spillBufferCycleCost.end()) {
        // reuse calculated cycles
        return _spillBufferCycleCost[spilledBuffer];
    }
    // get and store cost of buffer spill
    _spillBufferCycleCost[spilledBuffer] =
            getDMACost(spilledBuffer, spilledBuffer, _archKind, _costModel, _numDMAPorts);
    return _spillBufferCycleCost[spilledBuffer];
}

size_t FeasibleMemoryScheduler::operationCycleCost(operationIdxType opIdx) {
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (!execOp->hasAttr(cycleCostAttrName)) {
        // operations without cycle cost will have cycle cost = 1
        _log.warning("async.exec {0} has no cycle cost attribute {1}", execOp->getLoc(), cycleCostAttrName);
        return 1;
    }

    return checked_cast<size_t>(
            mlir::cast<mlir::IntegerAttr>(execOp->getAttr(cycleCostAttrName)).getValue().getSExtValue());
}

bool FeasibleMemoryScheduler::isDataOp(operationIdxType opIdx) {
    // Operations moving data to CMX are considered data ops. All others are
    // considered compute operations. This distinguishment is needed to balance
    // CMX memory space and not to fill CMX space with only data operations resulting
    // in not being able to fit the compute operation. Data operations will only be
    // scheduled when needed by the compute operation so that the CMX space can be
    // freed as soon as possible.
    if (getExecutorType(opIdx) != VPU::ExecutorKind::DMA_NN) {
        return false;
    }

    if (_outputOps.find(opIdx) != _outputOps.end()) {
        return false;
    }

    // TODO: E93697 extend DataOp scheduling to schedule its dependencies
    for (const auto& depIdx : _depsInfo.getOpDeps(opIdx)) {
        if (isDataOp(depIdx)) {
            return false;
        }
    }

    if (auto dmaTask = vpux::getDmaTypeOp(_depsInfo.getExecuteOpAtIndex(opIdx))) {
        // DMA from DDR to NN_CMX
        auto srcMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getInput().getType()).getMemoryKind();
        auto dstMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getOutput().getType()).getMemoryKind();
        return (_memKind == dstMemSpace && _memKind != srcMemSpace);
    }

    return false;
}

// Compute operation that depends only on constants and function inputs.
bool FeasibleMemoryScheduler::isNoInputDepComputeOp(operationIdxType opIdx) {
    // #E163065 - hang to be investigated. Limit to SW dequantize only.
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (VPUIP::VPUIPDialect::getExecutorKind(execOp) != VPU::ExecutorKind::SHAVE_ACT) {
        return false;
    }
    auto isSWDequant = false;
    execOp.getBody()->walk([&](mlir::Operation* op) {
        if (auto swKernelOp = mlir::dyn_cast_or_null<VPUIP::SwKernelOp>(op)) {
            if (getSwKernelEntryName(swKernelOp) == "dequantize") {
                isSWDequant = true;
            }
        }
    });
    if (!isSWDequant) {
        return false;
    }

    auto opDeps = _depsInfo.getOpDeps(opIdx);
    for (auto& dep : opDeps) {
        // Depends on data op which has no other dependencies
        if (!_isDataOp[dep] || !_depsInfo.getOpDeps(dep).empty()) {
            return false;
        }
    }

    return true;
}

void FeasibleMemoryScheduler::identifyDataOps() {
    auto opCount = _inDegreeTable.size();
    _isDataOp.resize(opCount);
    for (size_t opIdx = 0; opIdx < opCount; ++opIdx) {
        if (isDataOp(opIdx)) {
            _isDataOp.set(opIdx);
        }
    }
}

bool FeasibleMemoryScheduler::freeMemoryResources(const HeapElement& hElement) {
    auto op = _depsInfo.getExecuteOpAtIndex(hElement.op_);
    // free possible buffers, where this is the last user of the buffer
    bool freeMemoryResources = false;
    for (auto& buffer : _liveRangeInfo.getUsedBuffers(op)) {
        if (_liveRangeInfo.eraseUser(buffer, op) == 0) {
            _log.nest().trace("Mark buffer as dead, '{0}'", buffer);
            _scan.handler().markAsDead(buffer);
            freeMemoryResources = true;
        }
    }
    if (freeMemoryResources) {
        _log.nest().trace("Free non alive buffers");
        _scan.freeNonAlive();
    }
    return freeMemoryResources;
}

bool FeasibleMemoryScheduler::unscheduledOpsOnQueue(const QueueType& queueType) {
    for (auto& op : _cycleEndHeap) {
        if (op.queueType_ != queueType) {
            continue;
        }
        // queue type exists in cycle end heap
        return false;
    }
    return true;
}

void FeasibleMemoryScheduler::distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps) {
    // populate ready lists depending on op type/state
    _log.trace("Distribute new ready ops");
    _log = _log.nest();
    for (auto& readyOpIdx : readyOps) {
        if (_isDataOp[readyOpIdx]) {
            VPUX_THROW_UNLESS(_readyDataOps.find(readyOpIdx) == _readyDataOps.end(),
                              "Operation already in the ready data list '{0}'", readyOpIdx);
            _log.nest().trace("Add to ready data ops '{0}'", readyOpIdx);
            _readyDataOps.insert(readyOpIdx);
            const auto newReadyOps = reduceInDegreeOfAdjacentOperations(readyOpIdx);
            distributeReadyOps(newReadyOps);
        } else {
            const auto queueType = getQueueType(readyOpIdx);
            if (VPUIP::VPUIPDialect::isComputeExecutorKind(queueType.execKind)) {
                VPUX_THROW_UNLESS(_readyComputeOps.find(readyOpIdx) == _readyComputeOps.end(),
                                  "Operation already in ready compute list '{0}'", readyOpIdx);
                _log.nest().trace("Add to ready compute ops '{0}'", readyOpIdx);
                _readyComputeOps.insert(readyOpIdx);
            } else {
                VPUX_THROW_UNLESS(_readyDMAOps.find(readyOpIdx) == _readyDMAOps.end(),
                                  "Operation already in ready compute DMA list '{0}'", readyOpIdx);
                _log.nest().trace("Add to ready DMA ops '{0}'", readyOpIdx);
                _readyDMAOps.insert(readyOpIdx);
            }
        }
    }
    _log = _log.unnest();
}

SmallVector<operationIdxType> FeasibleMemoryScheduler::unlockNewReadyOps(const HeapElement& hElement) {
    if (!hElement.isOriginalOp()) {
        return SmallVector<operationIdxType>{};
    }
    const auto executorType = getExecutorType(hElement.op_);
    if (!VPUIP::VPUIPDialect::isComputeExecutorKind(executorType)) {
        // non compute executor kind consumers unlocked during scheduling
        return SmallVector<operationIdxType>{};
    }
    // propagate through original compute ops, generate new ready ops
    return reduceInDegreeOfAdjacentOperations(hElement.op_);
}

void FeasibleMemoryScheduler::unscheduleAllCompletingOps() {
    // find earliest scheduled queue cycle end
    const auto minScheduledQueueCycle = findMinScheduledQueueCycle();

    // unschedule operations from cycle end heap to target cycle end
    SmallVector<operationIdxType> readyOps = {};
    for (auto& nextOp : llvm::make_early_inc_range(_cycleEndHeap)) {
        if (nextOp.cycleEnd_ > minScheduledQueueCycle) {
            // do not unschedule post target cycle
            break;
        }

        _log.nest(2).trace("Unschedule opIdx '{0}'", nextOp.op_);
        if (freeMemoryResources(nextOp)) {
            // align executors only if memory resources freed
            alignExecutors(nextOp.cycleEnd_);
        }

        // retrieve new ready ops
        const auto newReadyOps = unlockNewReadyOps(nextOp);
        readyOps.insert(readyOps.end(), newReadyOps.begin(), newReadyOps.end());

        // remove op from heap
        _cycleEndHeap.erase(nextOp);
    }

    // distribute ready ops into ready lists
    distributeReadyOps(readyOps);
}

SmallVector<operationIdxType> FeasibleMemoryScheduler::reduceInDegreeOfAdjacentOperations(operationIdxType opIdx) {
    SmallVector<operationIdxType> zeroInDegreeOps;
    // reduce in-degree (number of incoming edges) for consumers of ready data ops
    for (const auto& consumer : _depsInfo.getConsumerOps(opIdx)) {
        if (_inDegreeTable[consumer] < 2) {
            zeroInDegreeOps.push_back(consumer);
            _inDegreeTable.erase(consumer);
        } else {
            VPUX_THROW_UNLESS(_inDegreeTable[consumer] > 0, "Invalid in-degree");
            _inDegreeTable[consumer]--;
        }
    }
    return zeroInDegreeOps;
}

void FeasibleMemoryScheduler::initializeReadyLists() {
    // populate ready lists with operations without dependencies
    SmallVector<operationIdxType> operationsWithNoDependencies;

    for (auto& entry : _inDegreeTable) {
        if (entry.second == 0) {
            operationsWithNoDependencies.push_back(entry.first);
        }
    }

    distributeReadyOps(operationsWithNoDependencies);
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers) {
    // retrieve size of buffers
    SmallVector<BufferOrder> bufferVector;
    // order buffers based on usage type
    for (auto& val : operationBuffers) {
        auto opSize = _scan.handler().getSize(val);

        size_t outDegree = 0;
        if (_bufferOpIdxMap.find(val) != _bufferOpIdxMap.end()) {
            for (auto& opIdx : _bufferOpIdxMap[val]) {
                outDegree += _outDegreeTable[opIdx];
            }
        } else {
            VPUX_THROW("Couldn't find the buffer '{0}' in output async index map", val.getLoc());
        }

        bufferVector.push_back(BufferOrder(val, opSize, outDegree));
    }
    // sort based on buffer qualities
    llvm::sort(bufferVector.begin(), bufferVector.end(), [](const BufferOrder& val1, const BufferOrder& val2) {
        // second outDegree of the buffer/parentOp
        if (val1.outDegree != val2.outDegree) {
            return val1.outDegree > val2.outDegree;
        }

        // third op size
        if (val1.size != val2.size) {
            return val1.size > val2.size;
        }

        // finally position in IR
        const auto parentOp = val1.buffer.getDefiningOp();
        VPUX_THROW_UNLESS(parentOp != nullptr, "Block arguments are not supported");
        return parentOp->isBeforeInBlock(val2.buffer.getDefiningOp());
    });

    // repopulate only with buffers
    SmallVector<mlir::Value> orderedBufs;
    for (auto& buff : bufferVector) {
        orderedBufs.push_back(buff.buffer);
    }
    return orderedBufs;
}

size_t FeasibleMemoryScheduler::scheduleSpilledOpBuffer(operationIdxType opIdx, mlir::Value* buffer) {
    // schedule the spilled dependency
    const auto queueAndCycle = getCurrentCycleAndExecutorInstanceMaskForSpill(*buffer, EOpType::IMPLICIT_SPILL_READ_OP);
    _log.nest().trace("Scheduling spilled op:'{0}' at cycle {1}", opIdx, queueAndCycle.cycle);
    // also store the buffer spilled
    auto spilledReadBuffer = *buffer;
    VPUX_THROW_UNLESS(_readySpilledOps.find(spilledReadBuffer) != _readySpilledOps.end(),
                      "Failed to find spill buffer");
    _readySpilledOps.erase(spilledReadBuffer);
    VPUX_THROW_UNLESS(_spillBufferMap.find(opIdx) != _spillBufferMap.end(), "Failed to find spill opIdx");
    if (_spillBufferMap[opIdx].size() > 1) {
        _spillBufferMap[opIdx].erase(spilledReadBuffer);
    } else {
        _spillBufferMap.erase(opIdx);
    }
    // update representation in scan handler
    _scan.handler().removeDynamicSpill(spilledReadBuffer);
    const auto opCycleCost = spilledOperationCycleCost(spilledReadBuffer);
    const auto nextAvailableCycle = queueAndCycle.cycle + opCycleCost;
    // update current cycle directly
    updateCurrentCycleForExecutor(queueAndCycle.queueType, queueAndCycle.execMask, nextAvailableCycle);
    pushToCycleBeginHeap(
            HeapElement(opIdx, queueAndCycle, opCycleCost, EOpType::IMPLICIT_SPILL_READ_OP, spilledReadBuffer));
    return nextAvailableCycle;
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::getNonAliveBuffersUsedByOperation(operationIdxType opIdx) {
    // retrieve all buffers used by the op which are not alive
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    SmallVector<mlir::Value> operationBuffers;

    for (auto& buffer : usedBuffs) {
        if (_scan.handler().isAlive(buffer)) {
            continue;
        }
        operationBuffers.push_back(buffer);
    }
    return operationBuffers;
}

mlir::DenseSet<mlir::Value> FeasibleMemoryScheduler::getBuffersToAllocateForOp(operationIdxType opIdx) {
    // retrieve non alive buffers
    auto usedBuffers = getNonAliveBuffersUsedByOperation(opIdx);

    mlir::DenseSet<mlir::Value> buffersToAllocate(usedBuffers.begin(), usedBuffers.end());
    for (const auto& dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opIdxEndCycleMap.find(dep) != _opIdxEndCycleMap.end()) {
            // op was scheduled
            continue;
        }

        VPUX_THROW_UNLESS(_readyDataOps.find(dep) != _readyDataOps.end(),
                          "Failed to get buffers - operation not ready '{0}'", dep);
        auto depBuffers = getBuffersToAllocateForOp(dep);
        buffersToAllocate.insert(depBuffers.begin(), depBuffers.end());
    }

    return buffersToAllocate;
}

size_t FeasibleMemoryScheduler::scheduleDependencies(operationIdxType opIdx) {
    // retrieve operation's buffers that need allocation
    for (auto val : getNonAliveBuffersUsedByOperation(opIdx)) {
        _scan.handler().markAsAlive(val);
        if (!_scan.handler().isDynamicSpill(val)) {
            continue;
        }
        // special case for spilled reads
        auto bufferProducer = _bufferProducer.find(val);
        VPUX_THROW_UNLESS(bufferProducer != _bufferProducer.end(), "Failed to find buffer producer for '{0}'", val);
        scheduleSpilledOpBuffer(bufferProducer->second, &val);
    }

    // schedule required dependencies order based on earliest scheduling cycle and IR order
    std::map<size_t, std::set<operationIdxType>> sortedDemandList;
    for (const auto& depIdx : _depsInfo.getOpDeps(opIdx)) {
        if (_opIdxEndCycleMap.find(depIdx) != _opIdxEndCycleMap.end()) {
            // op was scheduled
            continue;
        }

        VPUX_THROW_UNLESS(_readyDataOps.find(depIdx) != _readyDataOps.end(),
                          "Failed to schedule dependencies - operation not ready '{0}'", depIdx);
        const auto cycleBegin = getCurrentCycleAndExecutorInstanceMask(depIdx).cycle;
        sortedDemandList[cycleBegin].insert(depIdx);
    }

    for (auto& entry : sortedDemandList) {
        for (auto& depIdx : entry.second) {
            scheduleOp(depIdx);
            _readyDataOps.erase(depIdx);
        }
    }

    return getEarliestComputeBeginCycle(opIdx);
}

size_t FeasibleMemoryScheduler::scheduleOp(operationIdxType opIdx, EOpType opType) {
    // schedule dependencies
    const auto depEndCycle = scheduleDependencies(opIdx);

    // find schedule cycles for op
    const auto queueAndCycle = getCurrentCycleAndExecutorInstanceMask(opIdx, depEndCycle);
    const auto opCycleCost = operationCycleCost(opIdx);
    const auto nextAvailableCycle = queueAndCycle.cycle + opCycleCost;

    // schedule op
    updateCurrentCycleForExecutor(queueAndCycle.queueType, queueAndCycle.execMask, nextAvailableCycle);
    pushToCycleBeginHeap(HeapElement(opIdx, queueAndCycle, opCycleCost, opType));

    return nextAvailableCycle;
}

size_t FeasibleMemoryScheduler::getOperationLevel(operationIdxType opIdx, bool isSpilled) {
    if (!isSpilled) {
        return _opLevelVec[opIdx];
    }
    // original consumer(s) could have been already scheduled
    auto minRemainingConsumerLevel = std::numeric_limits<size_t>::max();
    for (const auto& consumerIdx : _depsInfo.getConsumerOps(opIdx)) {
        if (_opIdxEndCycleMap.find(consumerIdx) != _opIdxEndCycleMap.end()) {
            // consumer scheduled
            continue;
        }
        minRemainingConsumerLevel = std::min(minRemainingConsumerLevel, _opLevelVec[consumerIdx]);
    }
    return minRemainingConsumerLevel;
}

size_t calculateTotalBuffersSize(const mlir::DenseSet<mlir::Value>& buffers,
                                 LinearScan<mlir::Value, LinearScanHandler>& scan) {
    size_t totalSize = 0;
    auto& handler = scan.handler();
    for (const auto& buffer : buffers) {
        totalSize += handler.getSize(buffer);
    }
    return totalSize;
}

// Check if the total size of the buffers fits into the available free CMX memory
bool fitsIntoFreeCMX(LinearScan<mlir::Value, LinearScanHandler>& scan, const mlir::DenseSet<mlir::Value>& buffers,
                     size_t freeCmx) {
    size_t totalSize = calculateTotalBuffersSize(buffers, scan);
    return totalSize <= freeCmx;
}

// Get CMX demand of operation including all its dependencies
size_t FeasibleMemoryScheduler::getOpCmxDemand(operationIdxType opIdx) {
    auto buffers = getBuffersToAllocateForOp(opIdx);
    return calculateTotalBuffersSize(buffers, _scan);
}

void FeasibleMemoryScheduler::prefetchOps(ArrayRef<std::pair<operationIdxType, size_t>> scheduledOps,
                                          mlir::DenseSet<mlir::Value>& buffersToAllocate,
                                          bool checkMemoryFragmentation) {
    // consider barrier limitations
    std::set<operationIdxType> aliveOperations;
    for (auto& aliveBuffer : _scan.handler().getAliveValues()) {
        const auto aliveOpIdx = _bufferProducer[aliveBuffer];
        if (!_isDataOp[aliveOpIdx]) {
            continue;
        }
        aliveOperations.insert(aliveOpIdx);
    }

    // TODO: E93149 update barrier usage
    auto aliveOperationCount = aliveOperations.size();
    size_t barrierLimit = checked_cast<size_t>(_barrierPerCluster * _nceClusterCount) - scheduledOps.size();

    if (barrierLimit <= aliveOperationCount) {
        _log.nest().trace("Can not prefetch: alive ops '{0}' >= barrier limit '{1}'", aliveOperationCount,
                          barrierLimit);
        return;
    }

    // use IR order for prefetching
    size_t lastScheduledOp = 0;
    size_t lastScheduledCycle = 0;
    size_t lastScheduledLevel = 0;
    for (auto& scheduledOp : scheduledOps) {
        const auto executorType = getExecutorType(scheduledOp.first);
        if (executorType == VPU::ExecutorKind::DMA_NN) {
            continue;
        }
        lastScheduledOp = std::max(lastScheduledOp, scheduledOp.first);
        lastScheduledLevel = std::max(lastScheduledLevel, _opLevelVec[scheduledOp.first]);
        if (operationCycleCost(scheduledOp.first) <= 1) {
            // avoid comparing invalid cycles
            continue;
        }
        lastScheduledCycle = std::max(lastScheduledCycle, scheduledOp.second);
    }

    // limit prefetching candidates to defined level limit
    const auto currLevelLimit = lastScheduledLevel + _prefetchingLevelLimit;

    // find data ops before last scheduled op, IR is reordered such that
    // prefetch data ops are before compute op, sort prefetch candidates based on level
    std::map<size_t, std::set<operationIdxType>> sortedCandidates;
    for (auto& dataOp : _readyDataOps) {
        if (dataOp > lastScheduledOp) {
            continue;
        }
        const auto opLevel = getOperationLevel(dataOp);
        if (opLevel > currLevelLimit) {
            continue;
        }
        _log.nest().trace("Prefetch candidate: '{0}'", dataOp);
        sortedCandidates[opLevel].insert(dataOp);
    }

    // also consider prefetching spill candidates
    for (auto& spillOp : _readySpilledOps) {
        if (spillOp.second > lastScheduledOp) {
            continue;
        }
        const auto opLevel = getOperationLevel(spillOp.second, true);
        if (opLevel > currLevelLimit) {
            continue;
        }
        if (_scan.handler().isAlive(spillOp.first)) {
            continue;
        }
        _log.nest().trace("Prefetch spill candidate: '{0}'", spillOp.second);
        sortedCandidates[opLevel].insert(spillOp.second);
    }

    // try to allocate and schedule prefetch ops
    for (auto& entry : sortedCandidates) {
        for (const auto& opIdx : entry.second) {
            mlir::DenseSet<mlir::Value> operationBuffers;
            size_t scheduleCycle = 0;
            if (_readyDataOps.find(opIdx) != _readyDataOps.end()) {
                operationBuffers = getBuffersToAllocateForOp(opIdx);
                scheduleCycle = getCurrentCycleAndExecutorInstanceMask(opIdx).cycle;
            } else {
                if (_spillBufferMap.find(opIdx) == _spillBufferMap.end()) {
                    bool wasScheduledBefore =
                            std::any_of(_cycleBeginHeap.begin(), _cycleBeginHeap.end(), [=](const HeapElement& elem) {
                                return elem.op_ == opIdx;
                            });
                    VPUX_THROW_UNLESS(wasScheduledBefore, "Failed to find spill candidate '{0}'", opIdx);
                    _log.nest(2).trace("Prefetch '{0}' has been already scheduled", opIdx);
                } else {
                    operationBuffers = _spillBufferMap[opIdx];
                    for (auto& val : operationBuffers) {
                        const auto queueAndCycle =
                                getCurrentCycleAndExecutorInstanceMaskForSpill(val, EOpType::IMPLICIT_SPILL_READ_OP);
                        scheduleCycle = std::max(scheduleCycle, queueAndCycle.cycle);
                    }
                }
            }

            if (operationBuffers.empty()) {
                _log.nest(2).trace("No buffers to allocate for: '{0}'", opIdx);
                continue;
            }

            // avoid barriers for next compute dependencies using level check
            if (lastScheduledCycle != 0 && lastScheduledLevel + 1 < _opLevelVec[opIdx] &&
                scheduleCycle >= lastScheduledCycle) {
                _log.nest(2).trace("Would be scheduled after compute: '{0}'", opIdx);
                return;
            }

            operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
            if (!canAllocBuffers(operationBuffers)) {
                _log.nest(2).trace("Can not fit: '{0}'", opIdx);

                if (checkMemoryFragmentation) {
                    // save the failed prefetch buffer set for later allocation
                    // will try to resolve the fragment by actively spilling
                    if (fitsIntoFreeCMX(_scan, operationBuffers, _scan.totalFreeSize())) {
                        _log.nest(2).trace("Prefetch fails due to memory fragmentation");
                        _prefetchFailedDueToFragmentation = true;
                        ++_prefetchFragmentationFailureCount;
                        _fragmentedBuffers = std::move(operationBuffers);
                    }
                }

                return;
            }

            // need to allocate more buffers
            buffersToAllocate = std::move(operationBuffers);

            if (_readyDataOps.find(opIdx) != _readyDataOps.end()) {
                // schedule prefetch op
                _log.nest().trace("Scheduling prefetch op: '{0}'", opIdx);
                scheduleOp(opIdx, EOpType::ORIGINAL_PREFETCHED_OP);
                _readyDataOps.erase(opIdx);
                ++aliveOperationCount;
            } else {
                // schedule spilled prefetch op
                auto spilledBuffers = _spillBufferMap[opIdx];
                auto spilledBuffersVec = sortUsedBuffers(spilledBuffers);
                for (auto& val : spilledBuffersVec) {
                    _log.trace("Scheduling spill prefetch op: '{0}'", opIdx);
                    // mark the spilled buffer as alive in case other operation
                    // that can be scheduled as part of this prefetching iteration also depends on it
                    _scan.handler().markAsAlive(val);
                    scheduleSpilledOpBuffer(opIdx, &val);
                    ++aliveOperationCount;
                }
            }

            // consider barrier limitations
            if (barrierLimit <= aliveOperationCount) {
                _log.nest().trace("End prefetch: alive ops '{0}' >= barrier limit '{1}'", aliveOperationCount,
                                  barrierLimit);
                return;
            }
        }
    }
}

void FeasibleMemoryScheduler::sortAndAllocateBuffers(mlir::DenseSet<mlir::Value>& buffersToAllocate) {
    _log.nest().trace("Allocate memory for the alive buffers");
    for (auto& val : buffersToAllocate) {
        _log.nest(2).trace("Mark as alive '{0}'", val);
        _scan.handler().markAsAlive(val);
    }

    if (_optimizeFragmentation) {
        auto usedBuffers = sortUsedBuffers(buffersToAllocate);
        VPUX_THROW_UNLESS(_scan.alloc(usedBuffers, false, Partitioner::Direction::Up),
                          "Failed to statically allocate '{0}' memory", _memKind);
    } else {
        VPUX_THROW_UNLESS(_scan.alloc(buffersToAllocate, false, Partitioner::Direction::Up),
                          "Failed to statically allocate '{0}' memory", _memKind);
    }
}

void FeasibleMemoryScheduler::resetScheduleComputeOpsDeltas() {
    _prefetchFailedDueToFragmentation = false;
    _originalBufferProducersFromScheduleComputeOps.clear();
    _newBufferProducersFromScheduleComputeOps.clear();
    _originalOpIdxEndCycleMapFromScheduleComputeOps.clear();
    _newOpsFromScheduleComputeOps.clear();
    _originalBufferLastCycleUseFromScheduleComputeOps.clear();
    _newBufferLastCycleUsesFromScheduleComputeOps.clear();
}

void FeasibleMemoryScheduler::scheduleComputeOps() {
    SmallVector<std::pair<operationIdxType, size_t>> scheduledOps;
    mlir::DenseSet<mlir::Value> buffersToAllocate;
    SmallVector<operationIdxType> computeOpIdxToSchedule;

    // Ops with dependencies on other compute ops
    SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>> computeOps;
    // Ops which depend on constants or function inputs have lower priority
    SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>> noInputDepComputeOps;

    // Prepare status and snapshots for active spilling and fragmentation optimization

    ScheduleStateSnapshot snap(*this, _activelySpillForPrefetching);
    resetScheduleComputeOpsDeltas();

    // find compute ops to schedule
    for (auto& queue : _computeOpOrder) {
        auto firstOpInQueue = queue.second.begin();
        if (firstOpInQueue == queue.second.end()) {
            // no ops on queue left
            continue;
        }
        if (_readyComputeOps.find(*firstOpInQueue) == _readyComputeOps.end()) {
            // operation not ready
            continue;
        }
        if (!unscheduledOpsOnQueue(queue.first)) {
            // need to unschedule ops on queue before scheduling
            continue;
        }

        if (isNoInputDepComputeOp(*firstOpInQueue)) {
            _log.trace("Compute op '{0}' depends only on constants so added to lower prio queue", *firstOpInQueue);
            noInputDepComputeOps.push_back({*firstOpInQueue, queue.first});
        } else {
            // Store compute ops with dependencies to higher prio queue
            computeOps.push_back({*firstOpInQueue, queue.first});
        }
    }

    SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>> computeOpsToErase;

    // Check if compute ops satisfy buffer allocation constraints and select them for scheduling
    auto selectOpsToSchedule = [&](SmallVector<std::pair<operationIdxType, FeasibleMemoryScheduler::QueueType>>& ops) {
        for (auto opIter = ops.begin(); opIter != ops.end();) {
            auto operationBuffers = getBuffersToAllocateForOp(opIter->first);
            operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
            if (!canAllocBuffers(operationBuffers)) {
                // operation does not fit in memory
                opIter++;
                continue;
            }

            // op will be scheduled
            buffersToAllocate = std::move(operationBuffers);
            computeOpIdxToSchedule.push_back(opIter->first);
            _log.trace("Compute op to schedule: '{0}'", opIter->first);
            computeOpsToErase.push_back(*opIter);
            opIter = ops.erase(opIter);
        }
    };

    selectOpsToSchedule(computeOps);
    if (computeOps.empty()) {
        // Schedule noInputDepComputeOps as close as possible to its consumers.
        // Do not schedule them if there are other ready ops which do not fit into CMX yet.
        selectOpsToSchedule(noInputDepComputeOps);
    }

    // schedule compute ops
    for (auto& computeOpIdx : computeOpIdxToSchedule) {
        _log.trace("Scheduling compute op: '{0}'", computeOpIdx);

        auto opCycleEnd = scheduleOp(computeOpIdx);
        _readyComputeOps.erase(computeOpIdx);
        scheduledOps.push_back(std::make_pair(computeOpIdx, opCycleEnd));
    }

    // Return true if an alive buffer can be found to spill to support prefetching
    // and set _evictionCandidateToSupportPrefetch
    auto foundProperSpillCandidateToSupportPrefetch = [&]() {
        auto aliveBuffers = snap.scan.handler().getAliveValues();
        if (aliveBuffers.empty()) {
            return false;
        }

        // `originalBufferProducer` is the stage before running the current `scheduleComputeOps`.
        // So the changes in `_bufferProducer` is reverted.
        // And eviction candidate need to be selected in the alive buffers before the current scheduling
        // because the current scheduling will be redone with the eviction
        auto originalBufferProducer = _bufferProducer;
        for (auto& val : _originalBufferProducersFromScheduleComputeOps) {
            originalBufferProducer[val.first] = val.second;
        }

        for (auto& val : _newBufferProducersFromScheduleComputeOps) {
            originalBufferProducer.erase(val);
        }

        _evictionCandidateToSupportPrefetch =
                chooseCandidateForEvictionToSupportPrefetch(aliveBuffers, originalBufferProducer, snap.scan);
        if (!_evictionCandidateToSupportPrefetch.has_value()) {
            return false;
        }

        return true;
    };

    auto restoreStateBeforeScheduling = [&]() {
        // restore state before scheduling compute ops
        snap.rollback();

        for (auto& val : _originalBufferProducersFromScheduleComputeOps) {
            _bufferProducer[val.first] = val.second;
        }

        for (auto& val : _newBufferProducersFromScheduleComputeOps) {
            _bufferProducer.erase(val);
        }

        for (auto& val : _originalOpIdxEndCycleMapFromScheduleComputeOps) {
            _opIdxEndCycleMap[val.first] = val.second;
        }

        for (auto& val : _newOpsFromScheduleComputeOps) {
            _opIdxEndCycleMap.erase(val);
        }

        for (auto& val : _originalBufferLastCycleUseFromScheduleComputeOps) {
            _bufferLastCycleUse[val.first] = val.second;
        }

        for (auto& val : _newBufferLastCycleUsesFromScheduleComputeOps) {
            _bufferLastCycleUse.erase(val);
        }
    };

    // prefetch data ops
    if (!computeOpIdxToSchedule.empty()) {
        // If fragmentation is detected, _prefetchFailedDueToFragmentation is changed to true
        // And the active spilling logic will be triggered to find an eviction candidate to support prefetching
        prefetchOps(scheduledOps, buffersToAllocate, _activelySpillForPrefetching);
        if (_prefetchFailedDueToFragmentation) {
            // If prefetchOps failed due to fragmentation, try to find a proper spill candidate
            // If candidate found (_prefetchFailedDueToFragmentation is still true),
            // the candidate is saved in `_evictionCandidateToSupportPrefetch`
            _prefetchFailedDueToFragmentation = foundProperSpillCandidateToSupportPrefetch();
        }

        if (_prefetchFailedDueToFragmentation) {
            // If prefetchOps failed due to fragmentation, but successfully found a spill candidate,
            // restore state before scheduling compute ops and early return
            // so the spilling can be performed by `forceSpillingForPrefetch` on `_evictionCandidateToSupportPrefetch`
            // and prefetching should work after that

            ++_prefetchFragmentationFailureFixedCount;
            restoreStateBeforeScheduling();
            return;
        }

        for (auto& val : buffersToAllocate) {
            _scan.handler().markAsAlive(val);
        }
    }

    for (auto opIter = computeOpsToErase.begin(); opIter != computeOpsToErase.end(); opIter++) {
        auto& scheduledQueue = _computeOpOrder[opIter->second];
        scheduledQueue.erase(scheduledQueue.begin());
    }

    // find DMA ops to schedule
    SmallVector<operationIdxType> DMAOpIdxToSchedule;
    for (auto& readyOpIdx : _readyDMAOps) {
        auto operationBuffers = getBuffersToAllocateForOp(readyOpIdx);
        operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());
        if (!canAllocBuffers(operationBuffers)) {
            continue;
        }

        buffersToAllocate = std::move(operationBuffers);
        DMAOpIdxToSchedule.push_back(readyOpIdx);
    }

    // schedule DMA ops
    SmallVector<operationIdxType> readyOps = {};
    for (auto& DMAOpIdx : DMAOpIdxToSchedule) {
        _log.trace("Scheduling DMA op: '{0}'", DMAOpIdx);

        auto opCycleEnd = scheduleOp(DMAOpIdx);
        _readyDMAOps.erase(DMAOpIdx);
        scheduledOps.push_back(std::make_pair(DMAOpIdx, opCycleEnd));

        auto newReadyOps = reduceInDegreeOfAdjacentOperations(DMAOpIdx);
        readyOps.insert(readyOps.end(), newReadyOps.begin(), newReadyOps.end());
    }
    // unlock DMA copy back in ops to be prefetched, so we can achieve:
    // | DPU | DMA-out | DMA-in |
    // | -------- SW ---------- |
    distributeReadyOps(readyOps);

    // prefetch data ops - activation reads
    if (!DMAOpIdxToSchedule.empty()) {
        prefetchOps(scheduledOps, buffersToAllocate, false);
    }

    // TODO: E93150 gather all ops to schedule and sort by which-ever can be scheduled earlier

    // allocate buffers
    sortAndAllocateBuffers(buffersToAllocate);
}

void FeasibleMemoryScheduler::scheduleNonComputeOps() {
    mlir::DenseSet<mlir::Value> buffersToAllocate;

    // schedule operation not belonging to main network compute chain as soon as they become
    // ready so that they execute in the next available cycle since they are not prefetched
    for (auto& readyOpIdx : llvm::make_early_inc_range(_nonComputeChainOps)) {
        // Scheduling such operations can only happen once all input dependencies
        // (both data and compute ops) have already been executed. This is different
        // to standard compute op which as part of its scheduling can force scheduling
        // of needed data ops
        bool areDepsReady = true;
        for (const auto& dep : _depsInfo.getOpDeps(readyOpIdx)) {
            if (_spillBufferMap.find(dep) != _spillBufferMap.end()) {
                areDepsReady = false;
                break;
            }
        }
        if (!areDepsReady) {
            continue;
        }

        auto operationBuffers = getBuffersToAllocateForOp(readyOpIdx);
        operationBuffers.insert(buffersToAllocate.begin(), buffersToAllocate.end());

        if (!canAllocBuffers(operationBuffers)) {
            continue;
        }

        buffersToAllocate = std::move(operationBuffers);

        _log.trace("Scheduling non compute chain op: '{0}'", readyOpIdx);
        scheduleOp(readyOpIdx);
        _nonComputeChainOps.erase(readyOpIdx);
    }

    // allocate buffers
    sortAndAllocateBuffers(buffersToAllocate);
}

void FeasibleMemoryScheduler::insertInOpIdxCycleEndMap(const operationIdxType& opIdx, const size_t& endCycle) {
    auto mapItr = _opIdxEndCycleMap.find(opIdx);
    if (mapItr == _opIdxEndCycleMap.end()) {
        _opIdxEndCycleMap[opIdx] = endCycle;
        _newOpsFromScheduleComputeOps.insert(opIdx);
        return;
    }

    if (mapItr->second < endCycle) {
        if (_originalOpIdxEndCycleMapFromScheduleComputeOps.find(opIdx) ==
            _originalOpIdxEndCycleMapFromScheduleComputeOps.end()) {
            _originalOpIdxEndCycleMapFromScheduleComputeOps[opIdx] = mapItr->second;
        }
        _opIdxEndCycleMap[opIdx] = endCycle;
    }
}

size_t FeasibleMemoryScheduler::getEarliestComputeBeginCycle(operationIdxType opIdx) {
    auto queueAndCycle = getCurrentCycleAndExecutorInstanceMask(opIdx);
    auto earliestComputeBeginCycle = queueAndCycle.cycle;
    // precondition: all producers of used buffers scheduled
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    const auto usedBufs = _liveRangeInfo.getUsedBuffers(op);
    for (auto& buffer : usedBufs) {
        if (_bufferProducer.find(buffer) != _bufferProducer.end()) {
            // use cycle end of latest writing op
            earliestComputeBeginCycle = std::max(_opIdxEndCycleMap[_bufferProducer[buffer]], earliestComputeBeginCycle);
        }
    }
    return earliestComputeBeginCycle;
}

void FeasibleMemoryScheduler::evictActiveOp(EvictionCandidate evictionCandidate) {
    VPUX_THROW_UNLESS(_opIdxEndCycleMap.find(evictionCandidate.bufferWriterIdx_) != _opIdxEndCycleMap.end(),
                      "Attempt to evict a non-scheduled operation");

    _readySpilledOps[evictionCandidate.buffer_] = evictionCandidate.bufferWriterIdx_;
    _spillBufferMap[evictionCandidate.bufferWriterIdx_].insert(evictionCandidate.buffer_);

    _log.nest().trace("Mark dynamically spilled buffer as dead, '{0}'", evictionCandidate.buffer_);
    _scan.handler().markAsDead(evictionCandidate.buffer_);
    _scan.handler().markAsDynamicSpill(evictionCandidate.buffer_);

    _log.nest().trace("Free non alive buffers");
    _scan.freeNonAlive();
}

size_t FeasibleMemoryScheduler::evictionPriority(operationIdxType writerOpIdx, mlir::Value buffer) {
    // TODO: E#21936 add other conditions such as:
    // pipelined, multiple out-degree (prefetch)

    // Eviction priority (highest evicted first):
    // (0) - buffers which are CMX contactable
    // (1) - timestamp op buffers
    // (2) - buffers which are result of computeOp
    // (3) - buffers which are result of dataOp

    for (auto bufferUser : buffer.getUsers()) {
        if (mlir::isa<VPUIP::ConcatViewOp>(bufferUser)) {
            // buffer CMX contactable
            return 0;
        }
    }

    if (!_isDataOp[writerOpIdx]) {
        return 2;
    }

    return 3;
}

size_t FeasibleMemoryScheduler::getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer) {
    size_t outputIdx = 0;

    // Get asyncExecOp result corresponding to given buffer
    auto asyncExecOp = _depsInfo.getExecuteOpAtIndex(opIdx);

    for (auto& outBuffer : _liveRangeInfo.getOutputBuffers(asyncExecOp)) {
        if (outBuffer == buffer) {
            return outputIdx;
        }
        outputIdx++;
    }

    VPUX_THROW("Unable to find async.execOp (opIdx - '{0}') result corresponding to buffer '{1}'", opIdx, buffer);
}

operationIdxType FeasibleMemoryScheduler::getEarliestConsumerIdx(operationIdxType opIdx) {
    auto earliestConsumerIdx = std::numeric_limits<operationIdxType>::max();
    for (const auto& consumerIdx : _depsInfo.getConsumerOps(opIdx)) {
        if (!VPUIP::VPUIPDialect::isComputeExecutorKind(getExecutorType(consumerIdx))) {
            continue;
        }
        if (_opIdxEndCycleMap.find(consumerIdx) != _opIdxEndCycleMap.end()) {
            continue;
        }

        earliestConsumerIdx = std::min(earliestConsumerIdx, consumerIdx);
    }
    return earliestConsumerIdx;
}

// Choose the best candidate for eviction to support prefetching
// The candidate should be the smallest buffer that is not currently in use
// and whose eviction would free enough memory to allow prefetching
// For example, next buffer size to prefetch is 100, but there are only 80 continuous free memory due to fragmentation
//      Total cmx size is 200. The alive buffers are:
//          A [0, 100], B [180, 190], C [210, 215], D [220, 300]
//      Gaps are:
//          80: [100, 180], 20: [190, 210], 5: [215, 220]
//      Buffer size of 100 can't be prefetched due to fragmentation, so C, B, D, A are evicted in order to see if
//      the eviction could fix the fragmentation Eviction of B succeed so B is chosen to be eviction candidate for
//      prefetch
std::optional<FeasibleMemoryScheduler::EvictionCandidate>
FeasibleMemoryScheduler::chooseCandidateForEvictionToSupportPrefetch(
        mlir::DenseSet<mlir::Value>& aliveBuffers, mlir::DenseMap<mlir::Value, operationIdxType>& bufferProducer,
        LinearScan<mlir::Value, LinearScanHandler>& scan) {
    std::optional<FeasibleMemoryScheduler::EvictionCandidate> evictionCandidate = std::nullopt;

    // Sort active buffers by priority (lower priority evicted first)
    auto sortedAliveBuffers = sortUsedBuffers(aliveBuffers);

    // Record the smallest eviction candidate
    size_t evictionCandidateSize = 0;

    // Iterate over all alive buffers to find the best candidate for eviction
    for (const auto buffer : llvm::reverse(sortedAliveBuffers)) {
        // Ensure the buffer has been scheduled
        VPUX_THROW_UNLESS(bufferProducer.find(buffer) != bufferProducer.end(),
                          "Buffer not scheduled yet, invalid eviction candidate");

        auto executeOpIdx = bufferProducer[buffer];
        // Check if the buffer is in use
        bool bufferInUse = false;
        for (auto& nextOp : llvm::make_early_inc_range(_cycleEndHeap)) {
            if (nextOp.op_ == executeOpIdx) {
                bufferInUse = true;
                break;
            }
        }

        if (bufferInUse) {
            // Skip buffers that are currently in use
            continue;
        }

        auto size = scan.handler().getSize(buffer);
        if (evictionCandidate.has_value() && (evictionCandidateSize <= size)) {
            // Skip larger or equal sized buffers if a smaller candidate is already found
            continue;
        }

        // Create a temporary scan and mark the current buffer as dead
        // In tempScan the aliveBuffer and liveRange is temporarily changed to simulate the prefetching with eviction
        // The original scan shouldn't be changed so it can be roll back to the stage
        auto tempScan = scan;
        // Simulate the eviction by marking the buffer as dead in tempScan
        tempScan.handler().markAsDead(buffer);
        tempScan.freeNonAlive();

        // Create a temporary set of operation buffers including the current buffer
        auto tempOperationBuffers = _fragmentedBuffers;
        tempOperationBuffers.insert(buffer);

        auto canAllocBuffers = [&]() {
            if (_optimizeFragmentation) {
                // sort to minimize fragmentation
                auto sortedBuffers = sortUsedBuffers(tempOperationBuffers);
                // are resources available and can be allocated
                return tempScan.canAlloc(sortedBuffers);
            }
            return tempScan.canAlloc(tempOperationBuffers);
        };
        // Check if resources are available and can be allocated with eviction
        // If yes, record the eviction candidate with smaller size
        if (canAllocBuffers()) {
            auto priority = evictionPriority(executeOpIdx, buffer);
            auto earliestConsumerIdx = getEarliestConsumerIdx(executeOpIdx);
            // in special case of multiple output buffers store output idx
            auto outputIdx = getOpBufferOutputIdx(executeOpIdx, buffer);
            evictionCandidate = EvictionCandidate(priority, earliestConsumerIdx, size, executeOpIdx, outputIdx, buffer);
            evictionCandidateSize = size;
        }
    }

    return evictionCandidate;
}

FeasibleMemoryScheduler::EvictionCandidate FeasibleMemoryScheduler::chooseCandidateForEviction(
        const mlir::DenseSet<mlir::Value>& aliveBuffers) {
    // Check if last scheduled op was a SPILL-WRITE. If yes then this is a direct subsequent spill
    // and eviction candidates can be picked up from cache which was prepared during previous search
    // for spill write buffer
    if (!_evictionCandidatesCache.empty() && _scheduledOps.back().isSpillWrite()) {
        auto evictionCandidate = *_evictionCandidatesCache.begin();
        _evictionCandidatesCache.erase(_evictionCandidatesCache.begin());
        return evictionCandidate;
    }
    _evictionCandidatesCache.clear();
    // sort buffers using eviction priority
    for (const auto& buffer : aliveBuffers) {
        VPUX_THROW_UNLESS(_bufferProducer.find(buffer) != _bufferProducer.end(),
                          "Buffer not scheduled yet, invalid eviction candidate");
        auto executeOpIdx = _bufferProducer[buffer];
        auto priority = evictionPriority(executeOpIdx, buffer);
        auto earliestConsumerIdx = getEarliestConsumerIdx(executeOpIdx);
        auto size = _scan.handler().getSize(buffer);
        // in special case of multiple output buffers store output idx
        auto outputIdx = getOpBufferOutputIdx(executeOpIdx, buffer);
        _evictionCandidatesCache.insert(
                EvictionCandidate(priority, earliestConsumerIdx, size, executeOpIdx, outputIdx, buffer));
    }

    // Get eviction candidate with highest priority (beginning of set)
    // Rest will be left in a cache in case of subsequent spilling
    auto evictionCandidate = *_evictionCandidatesCache.begin();
    _evictionCandidatesCache.erase(_evictionCandidatesCache.begin());

    return evictionCandidate;
}

void FeasibleMemoryScheduler::performEvictionAndScheduling(const EvictionCandidate& evictionCandidate) {
    auto spillType = EOpType::IMPLICIT_SPILL_WRITE_OP;

    // Understand if spilling happened due to fragmentation
    // This is the case where operation CMX demand is smaller than
    // total free CMX size but because free contiguous slots are not
    // large enough operation buffers cannot be allocated
    if (_enableScheduleStatistics) {
        size_t freeCmx = _scan.totalFreeSize();
        bool spillingDueToFragmentation = false;

        for (auto& readyOp : _readyComputeOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            if (opTotalSize <= freeCmx) {
                if (!spillingDueToFragmentation) {
                    spillingDueToFragmentation = true;
                    break;
                }
            }
        }
        for (auto& readyOp : _readyDMAOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            if (opTotalSize <= freeCmx) {
                if (!spillingDueToFragmentation) {
                    spillingDueToFragmentation = true;
                    break;
                }
            }
        }

        if (spillingDueToFragmentation) {
            spillType = EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP;
        }

        // TODO: In future scheduler could try to reorder buffers to make necessary
        // space for next operation in case spilling happened due to CMX fragmentation
    }

    _log.nest().trace("Candidate selected for eviction '{0}' '{1}'", evictionCandidate.bufferWriterIdx_,
                      evictionCandidate.buffer_);

    // free the memory space by freeing the op output buffer
    evictActiveOp(evictionCandidate);
    _log.nest().trace("Candidate evicted and spilled");

    // consider spilling operation cycle end
    // TODO: consider last used cycle and next available cycle for spill to avoid stall
    const auto depEndCycle = _bufferLastCycleUse[evictionCandidate.buffer_];
    const auto queueAndCycle =
            getCurrentCycleAndExecutorInstanceMaskForSpill(evictionCandidate.buffer_, spillType, depEndCycle);
    // find operation end cycle
    const auto opCycleCost = spilledOperationCycleCost(evictionCandidate.buffer_);
    const auto nextAvailableCycle = queueAndCycle.cycle + opCycleCost;

    // add with a spilled write state
    pushToCycleBeginHeap(HeapElement(evictionCandidate.bufferWriterIdx_, queueAndCycle, opCycleCost, spillType,
                                     evictionCandidate.buffer_));
    // update current cycle directly
    updateCurrentCycleForExecutor(queueAndCycle.queueType, queueAndCycle.execMask, nextAvailableCycle);

    // memory resource freed, need to align executors to only allocate in future cycles
    alignExecutors(nextAvailableCycle);
}

void FeasibleMemoryScheduler::forceSpillingForPrefetch() {
    _log.trace("Unable to prefetch due to fragmentation, forcing dynamic spill");
    // select a candidate op to be spilled
    VPUX_THROW_UNLESS(_evictionCandidateToSupportPrefetch.has_value(), "Failed to find eviction candidate");
    auto evictionCandidateValue = _evictionCandidateToSupportPrefetch.value();
    performEvictionAndScheduling(evictionCandidateValue);
}

void FeasibleMemoryScheduler::forceScheduleActiveOpEviction() {
    _log.trace("Unable to schedule an operation, forcing dynamic spill");

    // retrieve the alive buffers
    auto aliveBuffers = _scan.handler().getAliveValues();
    size_t freeCmx = _scan.totalFreeSize();
    if (aliveBuffers.empty()) {
        _log.error("Scheduler cannot schedule anything and there is no buffer to spill");
        _log.error("Next operations to schedule:");
        for (auto& nextOp : _computeOpOrder) {
            _log.nest().error("opIdx: {0}, on: {1}", *nextOp.second.begin(), nextOp.first.execKind);
        }
        _log.error("Ready operations:");
        for (auto& readyOp : _readyComputeOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error(
                    "readyComputeOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ",
                    readyOp, opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }
        for (auto& readyOp : _readyDMAOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error("readyDMAOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ",
                              readyOp, opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }
        for (auto& readyOp : _readyDataOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error(
                    "readyDataOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ", readyOp,
                    opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }
        for (auto& readyOp : _nonComputeChainOps) {
            auto opTotalSize = getOpCmxDemand(readyOp);
            auto execOp = _depsInfo.getExecuteOpAtIndex(readyOp);
            _log.nest().error(
                    "nonComputeChainOp: opIdx: {0}, size demand: {1}, available free CMX: {2}, name: {3}, op: {4}, ",
                    readyOp, opTotalSize, freeCmx, execOp.getLoc(), execOp);
        }

        // select a candidate op to be spilled
        cleanUpAndLogSchedule(_scheduledOps);
        VPUX_THROW("Scheduler failure, cannot schedule anything and there is no buffer to spill");
    }

    auto evictionCandidate = chooseCandidateForEviction(aliveBuffers);
    performEvictionAndScheduling(evictionCandidate);
}

void FeasibleMemoryScheduler::createBufferAsyncIdxMap() {
    auto populateMap = [&](mlir::Value buffer, size_t operationIdx,
                           mlir::DenseMap<mlir::Value, SmallVector<size_t>>& bufferOpIdxMap) -> bool {
        auto insertedPair = bufferOpIdxMap.insert({buffer, {operationIdx}});
        if (!insertedPair.second) {
            bufferOpIdxMap[buffer].push_back(operationIdx);
        }
        return true;
    };
    for (auto& asyncDepsPair : _outDegreeTable) {
        auto executeOp = _depsInfo.getExecuteOpAtIndex(asyncDepsPair.first);

        for (auto& buffer : _liveRangeInfo.getOutputBuffers(executeOp)) {
            if (!populateMap(buffer, asyncDepsPair.first, _bufferOpIdxMap)) {
                continue;
            }
        }
    }
}

void FeasibleMemoryScheduler::populateScheduledOps(const HeapElement& scheduledOp) {
    SmallVector<IntervalInfo> outputIntervals;
    SmallVector<IntervalInfo> inputIntervals;
    // store scheduled information
    if (scheduledOp.isSpillWriteOp()) {
        // special case for a spill write with deallocation
        IntervalInfo interval;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(scheduledOp.spillBuffer_));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(scheduledOp.spillBuffer_));
        interval.buffer_ = scheduledOp.spillBuffer_;
        // SPILL WRITE has only input resource
        inputIntervals.push_back(interval);
        // deallocate only after addresses stored
        _log.nest().trace("Deallocate, '{0}'", scheduledOp.spillBuffer_);
        _scan.handler().deallocate(scheduledOp.spillBuffer_);
    } else if (scheduledOp.isSpillReadOp()) {
        IntervalInfo interval;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(scheduledOp.spillBuffer_));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(scheduledOp.spillBuffer_));
        interval.buffer_ = scheduledOp.spillBuffer_;
        outputIntervals.push_back(interval);
    } else {
        auto execOp = _depsInfo.getExecuteOpAtIndex(scheduledOp.op_);
        auto addIntervals = [&](bool isInput, ValueOrderedSet buffers, SmallVector<IntervalInfo>& intervals) {
            for (auto& buffer : buffers) {
                if ((isInput && (_bufferProducer.find(buffer) == _bufferProducer.end())) ||
                    (!isInput && (!scheduledOp.isOriginalOp() && buffer != scheduledOp.spillBuffer_))) {
                    continue;
                }
                IntervalInfo interval;
                // retrieve and store operation addresses
                interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(buffer));
                interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(buffer));
                interval.buffer_ = buffer;
                intervals.push_back(interval);
            }
        };

        addIntervals(true, _liveRangeInfo.getInputBuffers(execOp), inputIntervals);
        addIntervals(false, _liveRangeInfo.getOutputBuffers(execOp), outputIntervals);
    }
    // populate the struct fields
    ScheduledOpInfo scheduled;
    scheduled.op_ = scheduledOp.op_;
    scheduled.opType_ = scheduledOp.opType_;
    scheduled.outputResourceInfo_ = std::move(outputIntervals);
    scheduled.inputResourceInfo_ = std::move(inputIntervals);
    scheduled.cycleBegin_ = scheduledOp.cycleBegin_;
    scheduled.cycleEnd_ = scheduledOp.cycleEnd_;
    scheduled.isDataOp_ = _isDataOp[scheduledOp.op_];
    scheduled.freeCmx_ = _scan.totalFreeSize();
    if (scheduledOp.isSpillOp()) {
        scheduled.queueType.execKind = VPU::ExecutorKind::DMA_NN;
        if (scheduledOp.isSpillWriteOp()) {
            scheduled.queueType.id = getDMAQueueIdEncoding(_memKind, _archKind);
        } else {
            scheduled.queueType.id = getDMAQueueIdEncoding(_secondLvlMemKind, _archKind);
        }
    } else {
        scheduled.queueType = getQueueType(scheduledOp.op_);
    }
    // scheduled.queueType = scheduledOp.isSpillOp() ? VPU::ExecutorKind::DMA_NN : getExecutorType(scheduledOp.op_);
    scheduled.executorInstanceMask = scheduledOp.executorInstanceMask_;
    _scheduledOps.push_back(scheduled);
    insertInOpIdxCycleEndMap(scheduled.op_, scheduled.cycleEnd_);
    _log.trace("Scheduled op: '{0}' during cycles: {1} -> {2}, of type: {3}", scheduled.op_, scheduled.cycleBegin_,
               scheduled.cycleEnd_, scheduled.opTypeName());
}

void FeasibleMemoryScheduler::clearLists() {
    _readyComputeOps.clear();
    _readyDMAOps.clear();
    _readyDataOps.clear();
}

bool FeasibleMemoryScheduler::init() {
    _log.trace("Feasible Memory Scheduler init()");
    _depsInfo.buildConsMap();

    // compute op in/out degree
    _inDegreeTable = _depsInfo.calculateOpInDegreeTable();
    _outDegreeTable = _depsInfo.calculateOpOutDegreeTable();

    // retrieve output ops (ops with no out-degree)
    for (auto& entry : _outDegreeTable) {
        if (entry.second == 0) {
            _outputOps.insert(entry.first);
        }
    }

    identifyDataOps();

    size_t level = 0;
    _opLevelVec.resize(_inDegreeTable.size(), 0);
    for (size_t computeOpIdx = 0; computeOpIdx < _inDegreeTable.size(); ++computeOpIdx) {
        if (_isDataOp[computeOpIdx]) {
            continue;
        }

        const auto queueType = getQueueType(computeOpIdx);
        if (!VPUIP::VPUIPDialect::isComputeExecutorKind(queueType.execKind)) {
            continue;
        }

        _opLevelVec[computeOpIdx] = level;
        for (const auto& depInd : _depsInfo.getOpDeps(computeOpIdx)) {
            if (_opLevelVec[depInd] != 0) {
                continue;
            }
            _opLevelVec[depInd] = level;
        }

        _computeOpOrder[queueType].push_back(computeOpIdx);
        ++level;
    }

    clearLists();
    // TODO: check if input is dag
    initializeReadyLists();
    createBufferAsyncIdxMap();
    schedulingLoop();

    return true;
}

void FeasibleMemoryScheduler::schedulingLoop() {
    // scheduling loop, loop until all output ops are scheduled
    while (!_outputOps.empty()) {
        if (!_cycleBeginHeap.empty()) {
            _log.nest().trace("0. MOVE FROM CYCLE BEGIN TO CYCLE END HEAP");
            // move ops from cycle begin to cycle end heap
            // - populate ScheduledOpInfoVec with op info
            // - decrease any scheduled output ops
            moveFromCycleBeginToCycleEndHeap();
        } else {
            _log.nest().trace("1. UNSCHEDULE OPS FROM CYCLE END HEAP");
            // 1. unschedule all operations from cycle end heap
            //  - free memory of consumed buffers
            //  - unlock new ready operations
            unscheduleAllCompletingOps();

            _log.nest().trace("2. SCHEDULE OPS TO CYCLE BEGIN HEAP");
            // 2. schedule ready operations
            //  - allocate compute operations along with data dependencies
            //  - update executor pipelines
            // 2.1. schedule operation not belonging to main network compute chain
            scheduleNonComputeOps();
            // 2.2 schedule compute operations
            scheduleComputeOps();

            if (_prefetchFailedDueToFragmentation) {
                // optional 2.3 only enabled when _activelySpillForPrefetching is true
                // if prefetch failed due to fragmentation and `_evictionCandidateToSupportPrefetch` was found
                // force spilling of the candidate and try to prefetch again
                _log.nest().trace("DYNAMIC SPILL REQUIRED FOR PREFETCH");
                forceSpillingForPrefetch();
            }

            // 3. if no operation was added to cycle begin heap after scheduling
            //  - unable to schedule an operation, perform dynamic spill
            if (_cycleBeginHeap.empty() && _cycleEndHeap.empty()) {
                _log.nest().trace("3. DYNAMIC SPILL REQUIRED: FORCE DYNAMIC SPILL");
                forceScheduleActiveOpEviction();
            }
        }
    }
}

void FeasibleMemoryScheduler::cleanUpAndLogSchedule(ScheduledOpInfoVec& scheduledOps) {
    // schedule quality based on cycles (cycles start from 1)
    std::map<QueueType, SmallVector<int64_t>> dpuOrDmaQueuesCycles;

    int64_t totalCycles = 0;

    _log.setName("feasible-schedule");
    _log = _log.nest();
    for (const auto& op : scheduledOps) {
        auto execOp = _depsInfo.getExecuteOpAtIndex(op.op_);
        std::string inputResourceInfo = "<none>";
        std::string outputResourceInfo = "<none>";
        std::string executorInstanceInfo = "";
        auto channelTypeAsString = op.queueType.execKind == VPU::ExecutorKind::DMA_NN
                                           ? getDMAChannelTypeAsString(op.queueType.id, _archKind)
                                           : "";

        if (op.queueType.execKind == VPU::ExecutorKind::DMA_NN && channelTypeAsString.size() > 0) {
            executorInstanceInfo += "_" + channelTypeAsString;
        }

        if (_executorPipelines[op.queueType].size() > 1) {
            executorInstanceInfo += " [";
            bool addComma = false;
            for (auto execInd : op.executorInstanceMask.set_bits()) {
                if (addComma) {
                    executorInstanceInfo += ",";
                }
                executorInstanceInfo += std::to_string(execInd);
                addComma = true;
            }

            executorInstanceInfo += "]";
        }

        if (op.hasActiveInputResource()) {
            inputResourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfInputResources(); resourceIdx++) {
                if (op.isActiveInputResource(resourceIdx)) {
                    inputResourceInfo +=
                            "[" + std::to_string(op.beginInputResource(resourceIdx)) + " " +
                            std::to_string(op.endInputResource(resourceIdx)) + "] size = " +
                            std::to_string((op.endInputResource(resourceIdx) - op.beginInputResource(resourceIdx))) +
                            ", ";
                }
            }
        }

        if (op.hasActiveOutputResource()) {
            outputResourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfOutputResources(); resourceIdx++) {
                if (op.isActiveOutputResource(resourceIdx)) {
                    outputResourceInfo +=
                            "[" + std::to_string(op.beginOutputResource(resourceIdx)) + " " +
                            std::to_string(op.endOutputResource(resourceIdx)) + "] size = " +
                            std::to_string((op.endOutputResource(resourceIdx) - op.beginOutputResource(resourceIdx))) +
                            ", ";
                }
            }
        }

        auto cycleInfo = "cycles = " + std::to_string(op.cycleBegin_) + " -> " + std::to_string(op.cycleEnd_);
        _log.trace("op = '{0}'\t executor = '{1}{2}'\t type = '{3}'\t '{4}'\t inputs = '{5}' outputs = '{6}' \t "
                   "free = "
                   "'{7}'\t name = '{8}'",
                   op.op_, op.queueType.execKind, executorInstanceInfo, op.opTypeName(), cycleInfo, inputResourceInfo,
                   outputResourceInfo, op.freeCmx_, execOp.getLoc());

        if (op.queueType.execKind == VPU::ExecutorKind::DMA_NN || op.queueType.execKind == VPU::ExecutorKind::DPU) {
            if (dpuOrDmaQueuesCycles.find(op.queueType) == dpuOrDmaQueuesCycles.end()) {
                dpuOrDmaQueuesCycles[op.queueType].assign(_executorPipelines[op.queueType].size(), 1);
            }

            auto& execCycles = dpuOrDmaQueuesCycles[op.queueType];
            for (auto execInst : op.executorInstanceMask.set_bits()) {
                auto cycleDiff = op.cycleBegin_ - execCycles[execInst];
                if (cycleDiff > 0) {
                    std::string execInstString = "";

                    if (op.queueType.execKind == VPU::ExecutorKind::DMA_NN && channelTypeAsString.size() > 0) {
                        execInstString += "_" + channelTypeAsString;
                    }

                    if (_executorPipelines[op.queueType].size() > 1) {
                        execInstString += " [";
                        execInstString += std::to_string(execInst);
                        execInstString += "]";
                    }

                    _log.nest().trace("--- {0}{1} STALL ({2} cycles) ---", op.queueType.execKind, execInstString,
                                      cycleDiff);
                }
                execCycles[execInst] = op.cycleEnd_;
            }
        }

        totalCycles = std::max(totalCycles, op.cycleEnd_);
    }
    _log = _log.unnest();
    _log.trace("Total Cycles = {0}", totalCycles);
}

FeasibleMemoryScheduler::ScheduledOpInfoVec FeasibleMemoryScheduler::generateSchedule() {
    // start with all buffers requiring allocation
    _scan.handler().markAllBuffersAsDead();

    // start the memory scheduler
    init();

    // sort the operations to be reflected by IR order
    llvm::sort(_scheduledOps.begin(), _scheduledOps.end(), [](const ScheduledOpInfo& op1, const ScheduledOpInfo& op2) {
        // first cycle begin
        if (op1.cycleBegin_ != op2.cycleBegin_) {
            return op1.cycleBegin_ < op2.cycleBegin_;
        }

        // second smaller tasks first
        if (op1.cycleEnd_ != op2.cycleEnd_) {
            return op1.cycleEnd_ < op2.cycleEnd_;
        }

        // operation index
        if (op1.op_ != op2.op_) {
            return op1.op_ < op2.op_;
        }

        // allow self comparison
        return false;
    });

    return _scheduledOps;
}

bool FeasibleMemoryScheduler::canAllocBuffers(mlir::DenseSet<mlir::Value>& buffersToAllocate) {
    if (_optimizeFragmentation) {
        // sort to minimize fragmentation
        auto sortedBuffers = sortUsedBuffers(buffersToAllocate);
        // are resources available and can be allocated
        return _scan.canAlloc(sortedBuffers);
    }
    return _scan.canAlloc(buffersToAllocate);
}

void FeasibleMemoryScheduler::printFragmentFixCountLog() const {
    _log.info("[FeasibleAllocation statistics] fragmentation prefetch count {0}, fixed count {1}",
              _prefetchFragmentationFailureCount, _prefetchFragmentationFailureFixedCount);
}
