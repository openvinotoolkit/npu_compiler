//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

//
// Constructor
//

vpux::BarrierInfo::BarrierInfo(mlir::func::FuncOp func)
        : _log(Logger::global().nest("barrier-info", 0)),
          _func(func),
          _taskIndexAttrName(mlir::StringAttr::get(func->getContext(), "task-index")),
          _barrierIndexAttrName(mlir::StringAttr::get(func->getContext(), "barrier-index")),
          _syncTaskAttrName(mlir::StringAttr::get(func->getContext(), "sync-task")) {
    buildBarrierMaps(func);
}

vpux::BarrierInfo::BarrierInfo(): _log(Logger::global().nest("barrier-info", 0)), _func(nullptr) {
}

//
// clearAttributes
//

void vpux::BarrierInfo::clearAttributes() {
    auto removeAttributeFromRange = [](mlir::StringAttr attrName, auto range) {
        for (auto op : range) {
            VPUX_THROW_UNLESS(op->hasAttr(attrName), "Remove: attribute '{0}' was not set for '{1}' operation at '{2}'",
                              attrName, op->getName(), op->getLoc());
            op->removeAttr(attrName);
        }
    };

    removeAttributeFromRange(_taskIndexAttrName, _allTaskOps);
    removeAttributeFromRange(_barrierIndexAttrName, _allBarrierOps);
}

//
// getIndex (TaskOp)
//

uint32_t vpux::BarrierInfo::getIndex(VPURT::TaskOp taskOp) const {
    const auto attr = taskOp->getAttrOfType<mlir::IntegerAttr>(_taskIndexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                      _taskIndexAttrName, taskOp->getName(), taskOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

//
// getIndex (BarrierOpInterface)
//

uint32_t vpux::BarrierInfo::getIndex(VPURT::BarrierOpInterface barrierOp) const {
    const auto attr = barrierOp->getAttrOfType<mlir::IntegerAttr>(_barrierIndexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                      _barrierIndexAttrName, barrierOp->getName(), barrierOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

//
// getTaskOpAtIndex
//

VPURT::TaskOp vpux::BarrierInfo::getTaskOpAtIndex(size_t opIdx) const {
    VPUX_THROW_UNLESS(getNumOfTasks() > opIdx, "Task: Invalid index '{0}' for _allTaskOps", opIdx);
    VPUX_THROW_UNLESS(getIndex(_allTaskOps[opIdx]) == opIdx, "Task: Index not matching '{0}'", opIdx);
    return _allTaskOps[opIdx];
}

VPURT::TaskQueueType vpux::BarrierInfo::getTaskQueueType(size_t taskInd) const {
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");
    auto taskQueueTypeIt = llvm::find_if(_taskQueueTypeMap, [&](const auto& queueAndTaskBitVecPair) {
        return queueAndTaskBitVecPair.second.test(taskInd);
    });
    VPUX_THROW_WHEN(taskQueueTypeIt == _taskQueueTypeMap.end(), "Can not find task {0} from the mapping", taskInd);
    return taskQueueTypeIt->first;
}

std::optional<size_t> vpux::BarrierInfo::getPrevTaskOnFifo(size_t taskInd) const {
    VPUX_THROW_UNLESS(getNumOfTasks() > taskInd, "Task index '{0}' out of range", taskInd);
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");

    const auto taskQueueType = getTaskQueueType(taskInd);
    VPUX_THROW_WHEN(_taskQueueTypeMap.find(taskQueueType) == _taskQueueTypeMap.end(),
                    "Task queue map not initialized for executor of task {0}", taskInd);

    const auto taskBitVec = _taskQueueTypeMap.at(taskQueueType);

    auto prevTaskInd = taskBitVec.find_prev(taskInd);
    if (prevTaskInd < 0) {
        return std::nullopt;
    }
    return static_cast<size_t>(prevTaskInd);
}

// Return the closest previous task on the same queue that has a wait barrier
// Example:  Bar0 -> Op0 -> Op1 -> Op2
//  When called for Op2 it would return Op0 as Op0 has wait barrier
std::optional<size_t> vpux::BarrierInfo::getPrevTaskOnFifoWithWaitBar(size_t taskInd) const {
    VPUX_THROW_UNLESS(getNumOfTasks() > taskInd, "Task index '{0}' out of range", taskInd);
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");

    const auto taskQueueType = getTaskQueueType(taskInd);
    VPUX_THROW_WHEN(_taskQueueTypeMap.find(taskQueueType) == _taskQueueTypeMap.end(),
                    "Task queue map not initialized for executor of task {0}", taskInd);

    const auto taskBitVec = _taskQueueTypeMap.at(taskQueueType);

    auto currentTaskInd = taskInd;
    do {
        auto prevTaskInd = taskBitVec.find_prev(currentTaskInd);
        if (prevTaskInd < 0) {
            return std::nullopt;
        }
        currentTaskInd = static_cast<size_t>(prevTaskInd);
    } while (_taskWaitBarriers[currentTaskInd].empty());

    return currentTaskInd;
}

//
// getBarrierOpAtIndex
//

VPURT::BarrierOpInterface vpux::BarrierInfo::getBarrierOpAtIndex(size_t opIdx) const {
    VPUX_THROW_UNLESS(_allBarrierOps.size() > opIdx, "Barrier: Invalid index '{0}' for _allBarrierOps", opIdx);
    VPUX_THROW_UNLESS(getIndex(_allBarrierOps[opIdx]) == opIdx, "Barrier: Index not matching '{0}' '{1}'", opIdx,
                      _allBarrierOps[opIdx]);
    return _allBarrierOps[opIdx];
}

//
// getWaitBarriers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getWaitBarriers(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd <= _taskWaitBarriers.size(), "Task not found in _taskWaitBarriers, '{0}'", taskInd);
    return _taskWaitBarriers[taskInd];
}

void vpux::BarrierInfo::dumpBarriers(std::string fileNamePrefix) {
    dumpBarrierDependencies(_taskUpdateBarriers, fileNamePrefix + ".taskUpdateBarriers");
    dumpBarrierDependencies(_taskWaitBarriers, fileNamePrefix + ".taskWaitBarriers");
    dumpBarrierDependencies(_barrierProducerMap, fileNamePrefix + ".barrierProducerMap");
    dumpBarrierDependencies(_barrierConsumerMap, fileNamePrefix + ".barrierConsumerMap");
    // Dump number of barrier slots used per task
    dumpSlots(fileNamePrefix + ".slots");

    // Dump queues
    mlir::DenseSet<vpux::VPU::ExecutorKind> executorKind = {
            VPU::ExecutorKind::DPU,
            VPU::ExecutorKind::DMA_NN,
            VPU::ExecutorKind::SHAVE_ACT,
    };
    auto taskQueueTypeMapOrig = _taskQueueTypeMap;
    clearTaskQueueTypeMap();
    initializeTaskQueueTypeMap(executorKind);
    buildTaskQueueTypeMap();
    dumpQueues(_taskQueueTypeMap, fileNamePrefix + ".taskQueueTypeMap");
    // restore original state of _taskQueueTypeMap to keep the state unchanged
    _taskQueueTypeMap = std::move(taskQueueTypeMapOrig);
}

void vpux::BarrierInfo::dumpBarrierDependencies(const SmallVector<BarrierInfo::TaskSet>& depsMap,
                                                std::string fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name prefix for barriers dump was not provided");
    _log.trace("Dumping barrier dependencies to {0}", fileName);
    std::ofstream ofs(fileName);
    VPUX_THROW_UNLESS(ofs.is_open(), "Cannot open output file {0}", fileName);

    for (const auto& [i, deps] : enumerate(depsMap)) {
        ofs << i << " ";
        for (auto& dep : deps) {
            ofs << dep << " ";
        }
        ofs << std::endl;
    }
    ofs.close();
}

void vpux::BarrierInfo::dumpSlots(std::string fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name prefix for barriers dump was not provided");
    _log.trace("Dumping barrier slots to {0}", fileName);
    std::ofstream ofs(fileName);
    VPUX_THROW_UNLESS(ofs.is_open(), "Cannot open output file {0}", fileName);

    for (const auto& [i, taskInd] : enumerate(irange(getNumOfTasks()))) {
        ofs << i << " " << getNumOfSlotsUsedByTask(getTaskOpAtIndex(taskInd)) << std::endl;
    }
    ofs.close();
}

void vpux::BarrierInfo::dumpQueues(const std::map<VPURT::TaskQueueType, llvm::BitVector>& queueMap,
                                   std::string fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name prefix for barriers dump was not provided");
    _log.trace("Dumping tasks queues to {0}", fileName);
    std::ofstream ofs(fileName);
    VPUX_THROW_UNLESS(ofs.is_open(), "Cannot open output file {0}", fileName);

    for (const auto& [queue, deps] : queueMap) {
        ofs << VPU::stringifyExecutorKind(queue.type).str() << "_" << static_cast<size_t>(queue.id) << "("
            << deps.count() << "): ";
        for (auto dep : deps.set_bits()) {
            ofs << dep << " ";
        }
        ofs << std::endl;
    }
    ofs.close();
}

//
// getUpdateBarriers (by Idn)
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getUpdateBarriers(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd < _taskUpdateBarriers.size(), "Task not found in _taskUpdateBarriers, '{0}'", taskInd);
    return _taskUpdateBarriers[taskInd];
}

//
// getBarrierProducers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierProducers(size_t barrierInd) {
    VPUX_THROW_UNLESS(barrierInd <= _barrierProducerMap.size(), "Barrier not found in _barrierProducerMap, '{0}'",
                      barrierInd);
    return _barrierProducerMap[barrierInd];
}

//
// getBarrierConsumers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierConsumers(size_t barrierInd) {
    VPUX_THROW_UNLESS(barrierInd <= _barrierConsumerMap.size(), "Barrier not found in _barrierConsumerMap, '{0}'",
                      barrierInd);
    return _barrierConsumerMap[barrierInd];
}

BarrierInfo::TaskSet vpux::BarrierInfo::getBarriersUsers(const std::set<int64_t>& barrierInds) {
    BarrierInfo::TaskSet users;
    for (const auto& barrierInd : barrierInds) {
        llvm::set_union(users, getBarrierProducers(barrierInd));
        llvm::set_union(users, getBarrierConsumers(barrierInd));
    }
    return users;
}

//
// getBarrierProducers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierProducers(VPURT::BarrierOpInterface barrierOp) {
    auto barrierInd = getIndex(barrierOp);
    return getBarrierProducers(barrierInd);
}

//
// getBarrierConsumers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierConsumers(VPURT::BarrierOpInterface barrierOp) {
    auto barrierInd = getIndex(barrierOp);
    return getBarrierConsumers(barrierInd);
}

//
// getNumOfSlotsUsed
//

size_t vpux::BarrierInfo::getNumOfSlotsUsed(VPURT::TaskOp op) {
    // This function returns the number of variants used by a task for a barrier.
    // On VPU H/W, a NCE task can be executed (across multiple DPUs) via workloads descriptors (known as variants).
    // An NCE task may have multiple workloads descriptors (which are generated in the NCE DPU workloads pass).
    // Therefore, the number of variants must be verified as they will all update a barrier and
    // contribute to the architecture specific MAX VARIANT COUNT that a barrier has.

    if (op.getExecutorKind() == VPU::ExecutorKind::DPU) {
        const auto module = op->getParentOfType<mlir::ModuleOp>();
        if (VPUIP::supportsPerVariantBarrierConfiguration(module)) {
            return 1;
        }
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op.getInnerTaskOp());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op.getInnerTaskOp())) {
            auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
            return std::distance(swKernelRun.begin(), swKernelRun.end());
        }
        return 1;
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::DMA_NN || op.getExecutorKind() == VPU::ExecutorKind::M2I) {
        return 1;
    }

    // TODO: Analyze and define executor type for funcOp - E#117624
    if (op.getExecutorKind() == VPU::ExecutorKind::UNKNOWN) {
        return 1;
    }

    VPUX_THROW("Unsupported executor: {0}", op.getExecutorKind());
}

size_t vpux::BarrierInfo::getNumOfSlotsUsedByTask(VPURT::TaskOp op) const {
    return getNumOfSlotsUsed(op);
}

//
// getProducerSlotCount
//

size_t vpux::BarrierInfo::getProducerSlotCount(VPURT::BarrierOpInterface barrierOp) {
    size_t producerSlotCount = 0;
    for (const auto& producer : getBarrierProducers(barrierOp)) {
        producerSlotCount += getNumOfSlotsUsed(getTaskOpAtIndex(producer));
    }
    return producerSlotCount;
}

//
// getConsumerSlotCount
//

size_t vpux::BarrierInfo::getConsumerSlotCount(VPURT::BarrierOpInterface barrierOp) {
    size_t consumerSlotCount = 0;
    for (const auto& consumer : getBarrierConsumers(barrierOp)) {
        consumerSlotCount += getNumOfSlotsUsed(getTaskOpAtIndex(consumer));
    }
    return consumerSlotCount;
}

//
// getNumOfBarrierOps
//

size_t vpux::BarrierInfo::getNumOfBarrierOps() const {
    return _allBarrierOps.size();
}

size_t vpux::BarrierInfo::getBarrierMaxVariantSum() const {
    return VPUIP::getBarrierMaxVariantSum(_func);
}

//
// getNumOfTasks
//

size_t vpux::BarrierInfo::getNumOfTasks() const {
    return _allTaskOps.size();
}

//
// resizeBitMap
//

void vpux::BarrierInfo::resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits) {
    bitMap.resize(length);
    for (auto& bit : bitMap) {
        bit.resize(bits, false);
    }
}

void vpux::BarrierInfo::resetBitMap(SmallVector<llvm::BitVector>& bitMap) {
    for (auto& bit : bitMap) {
        bit.reset();
    }
}

SmallVector<size_t> vpux::BarrierInfo::getBarriersForTaskBlock(size_t blockInd, bool blockStartSyncPoint,
                                                               bool blockEndSyncPoint, bool updateBarriers) const {
    VPUX_THROW_WHEN(blockInd >= getControlGraphBlockCount(), "Invalid task block index ({0})", blockInd);
    auto [blockStartInd, blockEndInd] = getControlGraphBlockTaskRange(blockInd, blockStartSyncPoint, blockEndSyncPoint);

    llvm::BitVector barrierInd(getNumOfBarrierOps() + _fifoDependencies.size(),
                               false);  // use of BitVector here is faster than eg std::set
    if (updateBarriers) {
        for (auto taskInd = blockStartInd; taskInd <= blockEndInd; ++taskInd) {
            for (auto barIdx : _taskUpdateBarriers[taskInd]) {
                barrierInd.set(barIdx);
            }
        }
    } else {
        for (auto taskInd = blockStartInd; taskInd <= blockEndInd; ++taskInd) {
            for (auto barIdx : _taskWaitBarriers[taskInd]) {
                barrierInd.set(barIdx);
            }
        }
    }

    auto numOfBarriersInBlock = barrierInd.count();
    if (numOfBarriersInBlock == 0) {
        return {};
    }

    SmallVector<size_t> barriersInBlock;
    barriersInBlock.reserve(numOfBarriersInBlock);
    for (auto idx : barrierInd.set_bits()) {
        barriersInBlock.push_back(idx);
    }
    return barriersInBlock;
}

size_t vpux::BarrierInfo::getControlGraphBlockIndex(size_t taskInd) const {
    return _syncTasksIds.empty() ? 0 : _taskToBlockMap[taskInd];
}

size_t vpux::BarrierInfo::getBarrierBlockIndex(size_t barInd) {
    auto barrierLastProducer = getBarrierLatestProducer(barInd);
    if (isSyncPoint(barrierLastProducer)) {
        return getControlGraphBlockIndex(barrierLastProducer) + 1;
    }
    return getControlGraphBlockIndex(barrierLastProducer);
}

std::optional<size_t> vpux::BarrierInfo::getControlGraphSyncPoint(size_t taskInd) const {
    auto blockInd = getControlGraphBlockIndex(taskInd);
    if (blockInd == _syncTasksIds.size()) {
        return std::nullopt;
    }
    return _syncTasksIds[blockInd];
}

//
//  producersControlsAllConsumers
//

bool vpux::BarrierInfo::producersControlsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                                      const TaskSet& origConsumers,
                                                      ArrayRef<TaskSet> origWaitBarriersMap) {
    // Get new consumers not in original consumers

    auto consumersWithoutDirectControl = llvm::set_difference(newConsumers, origConsumers);
    if (consumersWithoutDirectControl.empty()) {
        return true;
    }
    if (origProducers.empty()) {
        return false;
    }

    auto consumerHasImplicitTaskQueueType = inImplicitQueueTypeDependencyList(consumersWithoutDirectControl);
    if (!consumerHasImplicitTaskQueueType) {
        return false;
    }
    auto producerHasImplicitTaskQueueType = inImplicitQueueTypeDependencyList(origProducers);
    if (!producerHasImplicitTaskQueueType) {
        return false;
    }

    if (*std::max_element(origProducers.begin(), origProducers.end()) >=
        *std::min_element(consumersWithoutDirectControl.begin(), consumersWithoutDirectControl.end())) {
        return false;
    }

    auto anyProducerCanRunParallelWithConsumer = llvm::any_of(origProducers, [&](const auto& producer) {
        return llvm::any_of(consumersWithoutDirectControl, [&](const auto& consumer) {
            return origWaitBarriersMap[producer] == origWaitBarriersMap[consumer];
        });
    });
    return !anyProducerCanRunParallelWithConsumer;
}

//
// inImplicitQueueTypeDependencyList
//

bool vpux::BarrierInfo::inImplicitQueueTypeDependencyList(const TaskSet& taskList) {
    // ensure that _taskQueueTypeMap is build at given time with buildTaskQueueTypeMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");
    auto allTasksAreInImplicitQueueTypeDependencyList = llvm::all_of(taskList, [&](const auto& taskInd) {
        for (const auto& item : _taskQueueTypeMap) {
            if (item.second.test(static_cast<unsigned>(taskInd))) {
                return true;
            }
        }
        return false;
    });
    return allTasksAreInImplicitQueueTypeDependencyList;
}

//
// buildBarrierMaps
//

void vpux::BarrierInfo::buildBarrierMaps(mlir::func::FuncOp func) {
    _log.trace("Collect initial producer maps");

    _allTaskOps = to_small_vector(func.getOps<VPURT::TaskOp>());
    _allBarrierOps = to_small_vector(func.getOps<VPURT::BarrierOpInterface>());

    _log.nest().trace("There are '{0}' VPURT::TaskOp", getNumOfTasks());
    _log.nest().trace("There are '{0}' VPURT::BarrierOpInterface", _allBarrierOps.size());

    // resize barrier maps
    _taskWaitBarriers = SmallVector<TaskSet>(getNumOfTasks(), {});
    _taskUpdateBarriers = SmallVector<TaskSet>(getNumOfTasks(), {});
    _barrierConsumerMap = SmallVector<TaskSet>(_allBarrierOps.size(), {});
    _barrierProducerMap = SmallVector<TaskSet>(_allBarrierOps.size(), {});

    // set index to task ops and barrier ops
    for (const auto& p : _allTaskOps | indexed) {
        p.value()->setAttr(_taskIndexAttrName, getIntAttr(p.value().getContext(), p.index()));
        if (p.value()->hasAttr(_syncTaskAttrName)) {
            _syncTasksIds.push_back(p.index());
        }
    }

    if (!_syncTasksIds.empty()) {
        buildTaskBlockMap();
        // Check block sizes
        size_t firstBlockSize = _syncTasksIds[0] + 1;
        for (size_t i = 1; i < _syncTasksIds.size(); i++) {
            size_t nextBlockSize = _syncTasksIds[i] - _syncTasksIds[i - 1];
            if (nextBlockSize != firstBlockSize) {
                _log.trace("Inferred first block size is '{0}', block '{1}' has {2} tasks", firstBlockSize, i,
                           nextBlockSize);
            }
        }
    }

    for (const auto& p : _allBarrierOps | indexed) {
        p.value()->setAttr(_barrierIndexAttrName, getIntAttr(p.value().getContext(), p.index()));
    }

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op)) {
            addTaskOp(taskOp);
        } else if (auto barrierOp = mlir::dyn_cast<VPURT::BarrierOpInterface>(op)) {
            _log.nest().trace("Found 'VPURT::BarrierOpInterface' Operation at '{0}'", op.getLoc());
            VPUX_THROW_WHEN(barrierOp.getBarrier() == nullptr, "BarrierOpInterface '{0}' has no barrier.", barrierOp);

            // ensure all barrier users are TaskOps
            for (auto* user : barrierOp.getBarrier().getUsers()) {
                VPUX_THROW_WHEN(mlir::dyn_cast<VPURT::TaskOp>(user) == nullptr, "Got non-TaskOp Operation at '{0}'",
                                op.getLoc());
            }
        }
    }
}

void vpux::BarrierInfo::buildTaskBlockMap() {
    VPUX_THROW_WHEN(_syncTasksIds.empty(),
                    "Control graph split was not detected. Building task block map is not necessary.");
    size_t nTasks = getNumOfTasks();
    _taskToBlockMap = SmallVector<size_t>(nTasks);

    size_t blockStartInd = 0;
    size_t blockInd = 0;

    for (auto syncTaskInd : _syncTasksIds) {
        for (size_t taskInd = blockStartInd; taskInd <= syncTaskInd; taskInd++) {
            _taskToBlockMap[taskInd] = blockInd;
        }
        blockInd++;
        blockStartInd = syncTaskInd + 1;
    }
    // assign block index to tasks from last block
    for (size_t taskInd = blockStartInd; taskInd < nTasks; taskInd++) {
        _taskToBlockMap[taskInd] = blockInd;
    }
}

//
// addConsumer
//

void vpux::BarrierInfo::addConsumer(size_t barrierInd, size_t taskInd) {
    _log.trace("Add consumer '{0}' for barrier '{1}'", taskInd, barrierInd);
    _barrierConsumerMap[barrierInd].insert(taskInd);
    _taskWaitBarriers[taskInd].insert(barrierInd);
}

void vpux::BarrierInfo::addConsumer(VPURT::BarrierOpInterface barrierOp, size_t taskInd) {
    addConsumer(getIndex(barrierOp), taskInd);
}

//
// addConsumers
//

void vpux::BarrierInfo::addConsumers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierConsumerMap[barrierInd].insert(taskInd);
        _taskWaitBarriers[taskInd].insert(barrierInd);
    }
}

//
// addProducer
//

void vpux::BarrierInfo::addProducer(size_t barrierInd, size_t taskInd) {
    _log.trace("Add producer '{0}' for barrier '{1}'", taskInd, barrierInd);
    _barrierProducerMap[barrierInd].insert(taskInd);
    _taskUpdateBarriers[taskInd].insert(barrierInd);
}

void vpux::BarrierInfo::addProducer(VPURT::BarrierOpInterface barrierOp, size_t taskInd) {
    addProducer(getIndex(barrierOp), taskInd);
}

//
// addProducers
//

void vpux::BarrierInfo::addProducers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierProducerMap[barrierInd].insert(taskInd);
        _taskUpdateBarriers[taskInd].insert(barrierInd);
    }
}

//
// removeProducers
//
void vpux::BarrierInfo::removeProducers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierProducerMap[barrierInd].erase(taskInd);
        _taskUpdateBarriers[taskInd].erase(barrierInd);
    }
}

void vpux::BarrierInfo::removeProducers(VPURT::BarrierOpInterface barrierOp, const TaskSet& taskInds) {
    const auto barrierInd = getIndex(barrierOp);
    removeProducers(barrierInd, taskInds);
}

//
// removeConsumers
//
void vpux::BarrierInfo::removeConsumers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierConsumerMap[barrierInd].erase(taskInd);
        _taskWaitBarriers[taskInd].erase(barrierInd);
    }
}

void vpux::BarrierInfo::removeConsumers(VPURT::BarrierOpInterface barrierOp, const TaskSet& taskInds) {
    const auto barrierInd = getIndex(barrierOp);
    removeConsumers(barrierInd, taskInds);
}

//
// addTaskOp
//

void vpux::BarrierInfo::addTaskOp(VPURT::TaskOp taskOp) {
    const auto taskInd = getIndex(taskOp);
    _log.trace("Found 'TaskOp' Operation '{0}'", taskInd);

    for (const auto& bar : taskOp.getWaitBarriers()) {
        // Note: can also be VPURT::ConfigureBarrierOp
        auto barrierOp = bar.getDefiningOp<VPURT::BarrierOpInterface>();
        VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid wait barrier op type {0}", bar);
        addConsumer(barrierOp, taskInd);
    }

    for (const auto& bar : taskOp.getUpdateBarriers()) {
        // Note: can also be VPURT::ConfigureBarrierOp
        auto barrierOp = bar.getDefiningOp<VPURT::BarrierOpInterface>();
        VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid wait barrier op type {0}", bar);
        addProducer(barrierOp, taskInd);
    }
}

//
// addNewBarrier
//

size_t vpux::BarrierInfo::addNewBarrier(VPURT::BarrierOpInterface barrierOp) {
    size_t barrierInd = _allBarrierOps.size();
    barrierOp->setAttr(_barrierIndexAttrName, getIntAttr(barrierOp.getContext(), barrierInd));

    _log.trace("Add new barrier '{0}', new barrier size '{1}'", barrierInd, _allBarrierOps.size());
    _allBarrierOps.push_back(barrierOp);

    _barrierConsumerMap.push_back({});
    _barrierProducerMap.push_back({});
    return barrierInd;
}

//
// addNewTaskOp
//

size_t vpux::BarrierInfo::addNewTaskOp(VPURT::TaskOp taskOp) {
    size_t taskOpIdx = getNumOfTasks();
    taskOp->setAttr(_taskIndexAttrName, getIntAttr(taskOp.getContext(), taskOpIdx));

    _log.trace("Add new taskOp '{0}', new taskOps size '{1}'", taskOpIdx, getNumOfTasks());

    _allTaskOps.push_back(taskOp);

    _taskUpdateBarriers.push_back({});
    _taskWaitBarriers.push_back({});

    return taskOpIdx;
}

//
// setWaitBarriers
//

void vpux::BarrierInfo::setWaitBarriers(size_t taskInd, const TaskSet& barriers) {
    // remove previous wait barriers
    for (auto barrierInd : _taskWaitBarriers[taskInd]) {
        _barrierConsumerMap[static_cast<size_t>(barrierInd)].erase(taskInd);
    }

    for (auto barrierInd : barriers) {
        _barrierConsumerMap[barrierInd].insert(taskInd);
    }
    _taskWaitBarriers[taskInd] = barriers;
}

//
// setUpdateBarriers
//

void vpux::BarrierInfo::setUpdateBarriers(size_t taskInd, const TaskSet& barriers) {
    // remove previous update barriers
    for (auto barrierInd : _taskUpdateBarriers[taskInd]) {
        _barrierProducerMap[static_cast<size_t>(barrierInd)].erase(taskInd);
    }

    for (auto barrierInd : barriers) {
        _barrierProducerMap[barrierInd].insert(taskInd);
    }
    _taskUpdateBarriers[taskInd] = barriers;
}

//
// removeProducer
//

void vpux::BarrierInfo::removeProducer(size_t barrierInd, size_t taskInd) {
    _barrierProducerMap[barrierInd].erase(taskInd);
    _taskUpdateBarriers[taskInd].erase(barrierInd);
}

void vpux::BarrierInfo::removeProducer(VPURT::BarrierOpInterface barrierOp, size_t taskInd) {
    removeProducer(getIndex(barrierOp), taskInd);
}

//
// removeConsumer
//
void vpux::BarrierInfo::removeConsumer(size_t barrierInd, size_t taskInd) {
    _barrierConsumerMap[barrierInd].erase(taskInd);
    _taskWaitBarriers[taskInd].erase(barrierInd);
}

void vpux::BarrierInfo::removeConsumer(VPURT::BarrierOpInterface barrierOp, size_t taskInd) {
    removeConsumer(getIndex(barrierOp), taskInd);
}

//
// resetBarrier
//

void vpux::BarrierInfo::resetBarrier(VPURT::BarrierOpInterface barrierOp) {
    const auto barrierInd = getIndex(barrierOp);
    resetBarrier(checked_cast<size_t>(barrierInd));
}

//
// resetBarrier
//

void vpux::BarrierInfo::resetBarrier(size_t barrierInd) {
    _log.trace("Reset barrier '{0}'", barrierInd);

    for (auto taskInd : _barrierProducerMap[barrierInd]) {
        _taskUpdateBarriers[static_cast<size_t>(taskInd)].erase(barrierInd);
    }
    _barrierProducerMap[barrierInd].clear();

    for (auto taskInd : _barrierConsumerMap[barrierInd]) {
        _taskWaitBarriers[static_cast<size_t>(taskInd)].erase(barrierInd);
    }
    _barrierConsumerMap[barrierInd].clear();
}

// Function that will perform split of a control graph with sync points
// present on boundaries of such split
// Example:
//  number of all task: 1200
//  requested split size: 500
//
// Graph will be split into 3 blocks with tasks indexes
// being placed in blocks in following way
//  block 0:    0 -  499
//  block 1:  500 -  999
//  block 2: 1000 - 1199
// Tasks with indexes 499 and 999 are sync points
// meaning that if there was any dependency previously present between blocks
// it will be modified to go through this sync point, example:
//  old dependency:   200 -> Bar -> 600
//  new dependencies: 200 -> newBar1 -> 499  and  499 -> newBar1 -> 600
//
// Goal of this split it to enable faster and less memory hungry optimizations
// on control graph as they can be done within single block between sync tasks
// For this example optimization can be done in 3 ranges: 0 - 499, 499 - 999, 999 - 1199
void vpux::BarrierInfo::splitControlGraphToBlocks(const size_t blockSize) {
    VPUX_THROW_WHEN(blockSize < 2, "Minimal block size is 2, requested size - {1}", blockSize);
    VPUX_THROW_WHEN(!_syncTasksIds.empty(),
                    "Partitioning on control graph was already performed, size - {0}, requested size - {1}",
                    _syncTasksIds[0] + 1, blockSize);

    auto tasksSize = getNumOfTasks();
    if (blockSize >= tasksSize) {
        _log.trace("Not enough tasks to perform split, requested split size - '{0}', num of tasks - '{1}'", blockSize,
                   tasksSize);
        return;
    }

    size_t numOfBlocks = tasksSize / blockSize;
    if (tasksSize % blockSize > 0) {
        numOfBlocks++;
    }
    auto isSyncPoint = [&](size_t taskIdx) {  // use local version of the method as the graph split is not finished.
        if (taskIdx == getNumOfTasks() - 1) {
            return false;
        }
        return (taskIdx + 1) % blockSize == 0;
    };

    _log.trace("Split control graph: num of task - '{0}', blocks size - '{1}', number of blocks - '{2}'", tasksSize,
               blockSize, numOfBlocks);

    auto barriersCount = _allBarrierOps.size();
    std::map<size_t, std::pair<size_t, size_t>> syncTaskWaitAndUpdateBarrierIndMap;

    // STEP 1:
    // Add explicit sync tasks wait and update barriers
    // which will be used to connect tasks with dependencies
    // going outside of block scope
    // If those new barriers are redundant, could have been combined with other existing barrier
    // or need a split because of large number of producer/consumers they will be optimized/processed
    // by follow-up passes
    for (size_t blockInd = 1; blockInd < numOfBlocks; blockInd++) {
        size_t syncTaskInd = blockInd * blockSize - 1;

        auto syncTaskOp = getTaskOpAtIndex(syncTaskInd);
        auto insertionOp = getTaskOpAtIndex((blockInd - 1) * blockSize);

        mlir::OpBuilder builder(insertionOp);
        //      |<-----syncWaitBar
        //  syncTask
        //      |----->syncUpdBar
        auto waitBar = builder.create<VPURT::DeclareVirtualBarrierOp>(syncTaskOp->getLoc());
        auto waitBarInd = addNewBarrier(waitBar);
        addConsumer(waitBarInd, syncTaskInd);

        builder.setInsertionPoint(syncTaskOp);
        auto updateBar = builder.create<VPURT::DeclareVirtualBarrierOp>(syncTaskOp->getLoc());
        auto updateBarInd = addNewBarrier(updateBar);

        addProducer(updateBarInd, syncTaskInd);

        _log.trace("Sync task at index '{0}', wait bar - '{1}', update bar - '{2}'", syncTaskInd, waitBarInd,
                   updateBarInd);

        syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd] = std::make_pair(waitBarInd, updateBarInd);
    }

    std::map<VPURT::TaskQueueType, size_t> firstTaskWithNoWaitBarForQueueInBlockMap;
    std::map<VPURT::TaskQueueType, size_t> lastTaskWithNoUpdateBarForQueueInBlockMap;

    // STEP 2
    // Identify operations in each block with no wait or update barriers that are first or last
    // for given queue:
    // - last op with no update barrier on each queue - link it to wait barrier of next sync task
    // - first op with no wait barrier on each queue - link it to update barrier of previous sync task
    //
    // If they will not have barriers assigned their position might be shifted
    // with respect to sync task during barrier legalization steps in follow-up passes
    for (size_t taskInd = 0; taskInd < tasksSize; taskInd++) {
        size_t taskBlockInd = taskInd / blockSize;
        auto taskUpdateBarriers = _taskUpdateBarriers[taskInd];
        size_t syncTaskInd = taskBlockInd * blockSize + blockSize - 1;

        // If task does not update any barrier store this information for each queue
        // and when moving to next block add a link to sync task. When encountering sync
        // task add connection from those ops to this sync task
        if (taskBlockInd < numOfBlocks - 1) {
            if (taskInd == syncTaskInd) {
                auto syncTaskWaitBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].first;
                for (const auto& queueListTaskPair : lastTaskWithNoUpdateBarForQueueInBlockMap) {
                    // Create new connection
                    // queueLastTaskInd -> syncWaitBarInd -> syncTaskInd
                    //
                    //  0(last op on queue)       0--------->|
                    //                                       |
                    //        syncWaitBar    =>         syncWaitBar
                    //             |                         |
                    //  1(sync) <--|              1(sync) <--|
                    _log.trace("New dep for task with no update barrier: task '{0}' -> sync task '{1}'",
                               queueListTaskPair.second, syncTaskInd);
                    addProducer(syncTaskWaitBarInd, queueListTaskPair.second);
                }
                lastTaskWithNoUpdateBarForQueueInBlockMap.clear();
            } else if (taskUpdateBarriers.empty()) {
                auto taskOp = getTaskOpAtIndex(taskInd);
                const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
                lastTaskWithNoUpdateBarForQueueInBlockMap[taskQueueType] = taskInd;
            }
        }

        // If task does not wait for any barrier and is first on a given queue
        // for this block (e.g. DMA task) then add a dependency from previous syncTask
        // Otherwise after call to orderExecutionTasksAndBarriers might move it to earlier block
        // and destroy assumption about how graph got split
        if (taskBlockInd > 0) {
            if (taskInd == syncTaskInd || taskInd == tasksSize - 1) {
                size_t prevSyncTaskInd = taskBlockInd * blockSize - 1;
                auto prevSyncTaskUpdateBarInd = syncTaskWaitAndUpdateBarrierIndMap[prevSyncTaskInd].second;
                for (const auto& queueListTaskPair : firstTaskWithNoWaitBarForQueueInBlockMap) {
                    // Create new connection
                    // prevSyncTaskInd -> syncUpdBarInd -> queueFirstTaskInd
                    //
                    //  1(sync)---|               1(sync) ---|
                    //            |                          |
                    //       syncUpdBar       =>        syncUpdBar
                    //                                       |
                    //  2(first op on queue)      2<---------|
                    _log.trace("New dep for task with no wait barrier: sync task '{0}' -> task '{1}'", prevSyncTaskInd,
                               queueListTaskPair.second);
                    addConsumer(prevSyncTaskUpdateBarInd, queueListTaskPair.second);
                }
                firstTaskWithNoWaitBarForQueueInBlockMap.clear();
            } else if (_taskWaitBarriers[taskInd].empty()) {
                auto taskOp = getTaskOpAtIndex(taskInd);
                const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
                if (firstTaskWithNoWaitBarForQueueInBlockMap.find(taskQueueType) ==
                    firstTaskWithNoWaitBarForQueueInBlockMap.end()) {
                    firstTaskWithNoWaitBarForQueueInBlockMap[taskQueueType] = taskInd;
                }
            }
        }
    }

    auto getSetWithNoSyncTasks = [&](const TaskSet& taskSet) {
        TaskSet taskSetWithNoSync;
        for (const auto& task : taskSet) {
            if (!isSyncPoint(task)) {
                taskSetWithNoSync.insert(task);
            }
        }

        return taskSetWithNoSync;
    };

    // STEP 3
    // Perform main processing to identify all ops with long dependency -
    // dependencies that go beyond the scope of existing block
    // Example:
    //  block 0:    0 -  499
    //  block 1:  500 -  999
    //  old dependency:   200 -> Bar -> 600
    //  new dependencies: 200 -> syncWaitBar -> 499(syncTask)  and  499(SyncTask) -> syncUpdBar -> 600
    // For this task scan all barriers and check such that within its producers and consumers have tasks from different
    // blocks. If yes then split those tasks and barrier so that tasks from different groups do not share barrier,
    // besides sync tasks
    for (size_t barInd = 0; barInd < barriersCount; ++barInd) {
        // Check barrier consumers and producers and check if they are from different blocks
        llvm::SmallSet<size_t, 16> blocksSet;

        VPUX_THROW_WHEN(_barrierProducerMap[barInd].empty(), "Barrier has no producers");
        VPUX_THROW_WHEN(_barrierConsumerMap[barInd].empty(), "Barrier has no consumers");

        // Gather information about barrier producers and consumers
        // and store them based on blockId
        std::map<size_t, TaskSet> barProdBlockTasksMap;
        for (const auto& taskInd : _barrierProducerMap[barInd]) {
            size_t taskBlockInd = taskInd / blockSize;
            blocksSet.insert(taskBlockInd);
            barProdBlockTasksMap[taskBlockInd].insert(taskInd);
        }

        std::map<size_t, TaskSet> barConsBlockTasksMap;
        for (const auto& taskInd : _barrierConsumerMap[barInd]) {
            size_t taskBlockInd = taskInd / blockSize;
            blocksSet.insert(taskBlockInd);
            barConsBlockTasksMap[taskBlockInd].insert(taskInd);
        }

        if (blocksSet.size() < 2) {
            // Only tasks from single block included
            // no need to split
            continue;
        }

        // Get barrier block assignment, which would correspond to earliest producer block id
        size_t barBlockId = std::min(barProdBlockTasksMap.begin()->first, barConsBlockTasksMap.begin()->first);

        // If this is last block then early exist. No need to analyze any subsequent barriers
        // as barriers are ordered and from that point no any subsequent barrier would need a split
        if (barBlockId == numOfBlocks + 1) {
            break;
        }

        _log.trace("Bar {0} needs a split, bar block id - {0}", barInd, barBlockId);

        // Remove tasks from this barrier which do not belong to this block
        for (const auto& barProdBlockTasksPair : barProdBlockTasksMap) {
            const auto blockId = barProdBlockTasksPair.first;
            if (blockId == barBlockId) {
                continue;
            }
            // Example control flow that will be shown throughout STEP 3
            // how it is modified
            // bar0 - barrier that is rossing boundary of blocks
            // 0p, 2p, 4p - barrier producer (p) tasks
            // 5c - barrier consumer (c) task
            // 1, 3 - sync tasks
            //
            //    0p -------->|         0p --------->|
            //                |                     bar0
            //    1(sync)     |         1(sync)      |
            //                |                      |
            //    2p -------->|         2p           |
            //                |    =>                |
            //    3(sync)     |         3(sync)      |
            //                |                      |
            //    4p -------->|         4p           |
            //               bar0                    |
            //    5c <--------|         5c <---------|
            _log.nest().trace("Remove producer for bar {0} from block {1}", barInd, blockId);
            removeProducers(barInd, barProdBlockTasksPair.second);

            if (blockId != barProdBlockTasksMap.rbegin()->first ||
                barConsBlockTasksMap.find(blockId) == barConsBlockTasksMap.end()) {
                // If this is not the last block produced by this barrier or there are no consumers in this block
                // add producers of this bar to now produce sync wait barrier
                //
                //    0p -------->|         0p ------------------->|
                //               bar0                             bar0
                //    1(sync)     |         1(sync)                |
                //                |                                |
                //    2p          |         2p -------->|          |
                //                |     =>          syncWaitBar    |
                //    3(sync)     |         3(sync) <---|          |
                //                |                                |
                //    4p          |         4p                     |
                //                |                                |
                //    5c <--------|         5c <-------------------|
                //
                size_t nextSyncTaskInd = blockId * blockSize - 1 + blockSize;
                if (nextSyncTaskInd < tasksSize) {
                    size_t syncTaskWaitBarInd = syncTaskWaitAndUpdateBarrierIndMap[nextSyncTaskInd].first;

                    _log.nest(2).trace("Add producers of sync bar {0} for tasks from block {1}", syncTaskWaitBarInd,
                                       blockId);
                    addProducers(syncTaskWaitBarInd, getSetWithNoSyncTasks(barProdBlockTasksPair.second));
                }
            } else {
                // If this is last block produced by this barrier and there are consumers in this block
                // create a new barrier to handle remaining deps that were produced by currently analyzed barInd
                //
                //    0p ------------------>|       0p ------------------->|
                //                         bar0                          bar0
                //    1(sync)               |       1(sync)                |
                //                          |                              |
                //    2p --------->|        |       2p -------->|          |
                //            syncWaitBar   |  =>          syncWaitBar     |
                //    3(sync) <----|        |       3(sync) <---|          |
                //                          |                              |
                //    4p                    |       4p -------->|          |
                //                          |                newBar        |
                //    5c <------------------|       5c <--------|<---------|
                //
                auto taskOp = getTaskOpAtIndex(*barProdBlockTasksPair.second.begin());
                mlir::OpBuilder builder(taskOp);
                auto newBar = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp->getLoc());
                auto newBarInd = addNewBarrier(newBar);

                _log.nest(2).trace("Add producers and consumers for new bar {0} for tasks from block {1}", newBarInd,
                                   blockId);
                addProducers(newBarInd, barProdBlockTasksPair.second);
                addConsumers(newBarInd, barConsBlockTasksMap[blockId]);
            }

            size_t syncTaskInd = blockId * blockSize - 1;
            size_t syncTaskUpdBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].second;

            //    0p -------------->|         0p --------------->|
            //                     bar0                         bar0
            //    1(sync)           |         1(sync)            |
            //                      |             |----->|       |
            //                      |                syncUpdBar  |
            //                      |         |<---------|       |
            //                      |         |                  |
            //    2p -------->|     |         2p ------->|       |
            //          syncWaitBar |    =>        syncWaitBar   |
            //    3(sync) <---|     |         3(sync) <--|       |
            //                      |             |              |
            //                      |             |----->|       |
            //                      |                syncUpdBar  |
            //                      |         |<---------|       |
            //                      |         |                  |
            //    4p --->|          |         4p --->|           |
            //         newBar       |              newBar        |
            //    5c <---|----------|         5c <---|-----------|
            //
            _log.nest(2).trace("Add consumers of sync bar {0} for tasks from block {1}", syncTaskUpdBarInd, blockId);
            addConsumers(syncTaskUpdBarInd, getSetWithNoSyncTasks(barProdBlockTasksPair.second));
        }
        // Remove tasks from this barrier which do not belong to this block
        for (const auto& barConsBlockTasksPair : barConsBlockTasksMap) {
            if (barConsBlockTasksPair.first == barBlockId) {
                continue;
            }
            //    0p ------------------->|        0p ----------------->|
            //                          bar0                          bar0
            //    1(sync)                |        1(sync)
            //                           |
            //    2p -------->|          |        2p ------->|
            //             syncWaitBar   |   =>          syncWaitBar
            //    3(sync) <---|          |        3(sync) <--|
            //       |                   |           |
            //       |------->|          |           |------>|
            //             syncUpdBar    |               syncUpdBar
            //    |<----------|          |        |<---------|
            //    |                      |        |
            //    4p -------->|          |        4p -------->|
            //             newBar        |                 newBar
            //    5c <--------|<---------|        5c <--------|
            //
            _log.nest().trace("Remove consumers of bar {0} from block {1}", barInd, barConsBlockTasksPair.first);
            removeConsumers(barInd, barConsBlockTasksPair.second);
        }

        size_t barSyncTaskInd = barBlockId * blockSize + blockSize - 1;

        // Sync task should be a consumer of this barrier, but only when
        // it was not the earliest producer
        if (*barProdBlockTasksMap[barBlockId].begin() != barSyncTaskInd) {
            // Remove it from producers just in case it was there
            removeProducer(barInd, barSyncTaskInd);
            //    0p ------------------>|     0p ------------------>|
            //                        bar0                        bar0
            //    1(sync)                     1(sync) <-------------|
            //
            //    2p -------->|               2p ------->|
            //            syncWaitBar                 syncWaitBar
            //    3(sync) <---|               3(sync) <--|
            //      |                     =>    |
            //      |-------->|                 |------->|
            //            syncUpdBar                  syncUpdBar
            //    |<----------|               |<---------|
            //    |                           |
            //    4p -------->|               4p ------->|
            //             newBar                     newBar
            //    5c <--------|               5c <-------|
            //
            addConsumer(barInd, barSyncTaskInd);
        }

        // Link tasks from other blocks from this barrier
        // as consumers of sync task barrier
        for (const auto& barConsBlockTasksPair : barConsBlockTasksMap) {
            if (barConsBlockTasksPair.first == barBlockId) {
                continue;
            }
            size_t syncTaskInd = barConsBlockTasksPair.first * blockSize - 1;
            size_t syncTaskUpdBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].second;

            //    0p ------------------->|     0p ------------------>|
            //                         bar0                        bar0
            //    1(sync) <--------------|     1(sync) <-------------|
            //      |                            |
            //      |-->syncUpdBar               |-->syncUpdBar
            //    |<---------|                 |<---------|
            //    |                            |
            //    2p -------->|                2p ------->|
            //           syncWaitBar                 syncWaitBar
            //    3(sync) <---|                3(sync) <--|
            //      |                     =>   |
            //      |-------->|                |-------------->|
            //           syncUpdBar                        syncUpdBar
            //    |<----------|                |<--------------|
            //    |                            |               |
            //    4p -------->|                4p ------->|    |
            //             newBar                      newBar  |
            //    5c <--------|                5c <-------|<---| (in this particular example this dep is redundant
            //                                                    but it will be optimized by later passes)
            //
            _log.nest().trace("Add consumers of sync bar {0} from block {1}", syncTaskUpdBarInd,
                              barConsBlockTasksPair.first);
            addConsumers(syncTaskUpdBarInd, getSetWithNoSyncTasks(barConsBlockTasksPair.second));
        }
    }

    // STEP 4
    // Add explicit dependency between sync tasks
    // If it is not needed because it was already represented implicitly by other
    // dependencies it will be removed by follow-up passes
    for (size_t blockInd = 2; blockInd < numOfBlocks; blockInd++) {
        size_t syncTaskInd = (blockInd - 1) * blockSize - 1;
        size_t nextSyncTaskInd = blockInd * blockSize - 1;

        // Create new connection
        // syncTaskInd -> syncTaskUpdateBarInd -> nextSyncTaskInd
        auto syncTaskUpdateBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].second;

        _log.trace("New dep: sync task '{0}' -> next sync task '{1}' using barrier '{2}'", syncTaskInd, nextSyncTaskInd,
                   syncTaskUpdateBarInd);
        addConsumer(syncTaskUpdateBarInd, nextSyncTaskInd);
    }

    // STEP 5
    // Scan all barriers and perform check and sanitization needed after
    // modifications done in STEP 3
    for (size_t barInd = 0; barInd < barriersCount; ++barInd) {
        // If after modifying control graph there is a barrier with no consumers then it
        // is redundant and related producers should be linked to next sync task
        if (_barrierConsumerMap[barInd].empty() && !_barrierProducerMap[barInd].empty()) {
            _log.trace("Empty consumers for bar - {0}", barInd);
            for (auto taskInd : _barrierProducerMap[barInd]) {
                size_t taskBlockInd = taskInd / blockSize;
                size_t syncTaskInd = taskBlockInd * blockSize + blockSize - 1;
                if (syncTaskInd >= tasksSize) {
                    continue;
                }

                if (taskInd != syncTaskInd) {
                    // Create new connection
                    // taskInd -> syncTaskWaitBarInd -> syncTaskInd
                    auto syncTaskWaitBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].first;
                    addProducer(syncTaskWaitBarInd, taskInd);
                }
            }

            resetBarrier(barInd);
        }

        // Check if after modifying control graph there are no barriers
        // with empty producers and non empty consumers
        if (_barrierProducerMap[barInd].empty()) {
            VPUX_THROW_UNLESS(_barrierConsumerMap[barInd].empty(), "Barrier {0} has no producers but has consumers",
                              barInd);
        }
    }

    // STEP 6
    // Remove not used barriers created for sync tasks
    for (const auto& syncTaskWitAndUpdateBarrierInd : syncTaskWaitAndUpdateBarrierIndMap) {
        const auto waitBarInd = syncTaskWitAndUpdateBarrierInd.second.first;
        const auto updateBarInd = syncTaskWitAndUpdateBarrierInd.second.first;
        if (_barrierProducerMap[waitBarInd].empty()) {
            resetBarrier(waitBarInd);
        }
        if (_barrierConsumerMap[updateBarInd].empty()) {
            resetBarrier(updateBarInd);
        }
    }

    // STEP 7
    // Put dedicated attribute on sync tasks for follow-up passes so that
    // they are aware of the split and can take advantage of it (e.g. perform optimizeBarriers in parts)
    for (size_t blockInd = 1; blockInd < numOfBlocks; blockInd++) {
        size_t syncTaskInd = blockInd * blockSize - 1;
        auto syncTaskOp = getTaskOpAtIndex(syncTaskInd);
        syncTaskOp->setAttr(_syncTaskAttrName, mlir::UnitAttr::get(syncTaskOp->getContext()));
    }
}

bool vpux::BarrierInfo::verifyControlGraphSplit() {
    if (_syncTasksIds.empty()) {
        return true;
    }

    size_t firstBlockSize = _syncTasksIds[0] + 1;
    for (size_t i = 1; i < _syncTasksIds.size(); i++) {
        auto nextBlockSize = _syncTasksIds[i] - _syncTasksIds[i - 1];
        if (nextBlockSize != firstBlockSize) {
            _log.trace("Inferred first block size is '{0}', but block '{1}' has {2} tasks", firstBlockSize, i,
                       nextBlockSize);
            VPUX_THROW_WHEN(nextBlockSize == 0, "Detected 0-sized task block");
        }
    }

    auto getSyncTaskInd = [&](size_t blockInd) {
        return blockInd == _syncTasksIds.size() ? getNumOfTasks() - 1 : _syncTasksIds[blockInd];
    };

    for (size_t taskInd = 0; taskInd < getNumOfTasks(); taskInd++) {
        size_t taskBlockInd = getControlGraphBlockIndex(taskInd);
        size_t syncTaskInd = getSyncTaskInd(taskBlockInd);

        for (const auto& updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            auto barrierConsumers = _barrierConsumerMap[updateBarrierInd];

            for (const auto& childTaskInd : barrierConsumers) {
                size_t childBlockInd = getControlGraphBlockIndex(childTaskInd);

                if (taskInd == syncTaskInd && taskBlockInd + 1 != childBlockInd) {
                    _log.debug("Out of block dep for sync task: {0} (block {1}) -> bar {2} -> {3} (block{4}) ", taskInd,
                               taskBlockInd, updateBarrierInd, childTaskInd, childBlockInd);
                    return false;
                } else if (taskInd != syncTaskInd && taskBlockInd != childBlockInd) {
                    _log.debug("Out of block dep for task: {0} (block {1}) -> bar {2} -> {3} (block{4}) ", taskInd,
                               taskBlockInd, updateBarrierInd, childTaskInd, childBlockInd);
                    return false;
                }
            }
        }
    }
    return true;
}

bool BarrierInfo::verifyBarriersForTaskDescriptorFetch(const ExecutionGroupListMap& executionGroups, bool wlmEnabled) {
    mlir::DenseSet<vpux::VPU::ExecutorKind> executors = {VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::DPU,
                                                         VPU::ExecutorKind::SHAVE_ACT};
    clearTaskQueueTypeMap();
    initializeTaskQueueTypeMap(executors);
    buildTaskQueueTypeMap();

    for (size_t blockIdx = 0; blockIdx < getControlGraphBlockCount(); ++blockIdx) {
        auto [taskControlMapFifo, controlMapOffset] =
                buildTaskControlMap(blockIdx, /* considerTaskFifoDependency */ true);

        for (auto [queue, fifoExecGroups] : executionGroups) {
            int numberOfGroups = static_cast<int>(fifoExecGroups.size());
            for (int grpIdx = 0; grpIdx < numberOfGroups - 1; grpIdx++) {
                if (!hasBarrierDependencyRequiredForDescriptorFetch(grpIdx, blockIdx, fifoExecGroups,
                                                                    taskControlMapFifo, controlMapOffset, wlmEnabled)) {
                    _log.error("Found tasks in execution group {0} ({1}.{2}) without required task-descriptor-fetch "
                               "barrier in "
                               "block {3} ",
                               grpIdx, queue.type, queue.id, blockIdx);
                    return false;
                }
            }
        }
    }
    return true;
}

bool vpux::BarrierInfo::hasBarrierDependencyRequiredForDescriptorFetch(int grpIdx, size_t blockIdx,
                                                                       const ExecutionGroupList& fifoExecGroups,
                                                                       SmallVector<llvm::BitVector>& taskControlMap,
                                                                       size_t controlMapOffset, bool wlmEnabled) {
    int numberOfGroups = static_cast<int>(fifoExecGroups.size());
    VPUX_THROW_UNLESS(grpIdx >= 0 && grpIdx < numberOfGroups, "Invalid group index");

    if (grpIdx + 1 == numberOfGroups) {
        // no barrier is required at the end of last group
        return true;
    }
    auto lastTaskInGroup = *fifoExecGroups[grpIdx].rbegin();
    auto firstTaskInChildGroup = *fifoExecGroups[grpIdx + 1].begin();
    if (getControlGraphBlockIndex(lastTaskInGroup) != blockIdx ||
        getControlGraphBlockIndex(firstTaskInChildGroup) != blockIdx) {
        // barrier would link tasks from different blocks, barrier is not needed
        return true;
    }

    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);

    std::optional<size_t> firstTaskInGrandChildGroupInBlock = std::nullopt;
    if (grpIdx < numberOfGroups - 2) {
        auto firstTaskInGrandChildGroup = fifoExecGroups[grpIdx + 2][0];
        if (inRange(blockStartInd, blockEndInd, firstTaskInGrandChildGroup)) {
            firstTaskInGrandChildGroupInBlock = firstTaskInGrandChildGroup;
        }
    }

    // check if barrier between last task in execution group and first task of next (child) execution group is required
    auto endGrpUpdateBarriers = getUpdateBarriers(lastTaskInGroup);

    // If grand child execution group does not exist, barrier is not needed.
    if (!firstTaskInGrandChildGroupInBlock.has_value()) {
        return true;
    }

    if (endGrpUpdateBarriers.empty()) {
        // barrier is needed when there isn't any at the end of task group
        return false;
    }

    if (wlmEnabled) {
        // For WLM case, if barrier exists at the end of execution group, new barrier is not required.
        return true;
    }

    // For non-WLM case, additional verifications are required.

    auto hasPath = [&](size_t taskA, size_t taskB) {
        return controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskA - controlMapOffset,
                                                        taskB - controlMapOffset, false);
    };

    // If none of the consumers of the execution group update barriers depend on grand child group,
    // then the new barrier is not needed.
    for (auto endGrpUpdateBarrier : endGrpUpdateBarriers) {
        auto barrierConsumers = getBarrierConsumers(endGrpUpdateBarrier);

        bool hasConsumersDependedntOnGrandChildGrp = false;
        for (const auto& consumer : barrierConsumers) {
            if (hasPath(firstTaskInGrandChildGroupInBlock.value(), consumer)) {
                hasConsumersDependedntOnGrandChildGrp = true;
                break;
            }
        }

        if (!hasConsumersDependedntOnGrandChildGrp) {
            return true;
        }
    }

    return false;
}

bool vpux::BarrierInfo::adjustTasksDependenciesToGraphSplitConstraints(const TaskSet& tasks) {
    if (_syncTasksIds.empty()) {
        return false;
    }

    bool dependenciesChanged = false;

    size_t controlMapBlockInd = std::numeric_limits<size_t>::max();
    SmallVector<llvm::BitVector> taskControlMap;
    size_t controlMapOffset = 0;

    auto updateControlMap = [&](size_t taskBlockInd, SmallVector<llvm::BitVector>& taskControlMap,
                                size_t& controlMapOffset) {
        if (controlMapBlockInd != taskBlockInd) {
            std::tie(taskControlMap, controlMapOffset) = buildTaskControlMap(taskBlockInd, false, true);
            controlMapBlockInd = taskBlockInd;
        }
    };

    for (auto taskInd : tasks) {
        size_t taskBlockInd = getControlGraphBlockIndex(taskInd);
        auto taskSyncPoint = getControlGraphSyncPoint(taskInd);

        // check task dependencies through its update barriers
        auto taskUpdateBarriers = getUpdateBarriers(taskInd);
        for (const auto& updateBarrierInd : taskUpdateBarriers) {
            auto barrierBlockIndex = getBarrierBlockIndex(updateBarrierInd);

            if (!taskSyncPoint.has_value()) {
                // the task belongs to last block
                VPUX_THROW_WHEN(barrierBlockIndex != taskBlockInd,
                                "taskInd '{0}' does not have sync point and should not have "
                                "out-of-block consumers",
                                taskInd);
            }

            if (taskBlockInd == barrierBlockIndex) {
                // dependence does not span across blocks
                continue;
            }

            if (isSyncPoint(taskInd) && taskBlockInd + 1 == barrierBlockIndex) {
                // sync-point can update barriers from the following block
                continue;
            }
            _log.trace("Out of block dependence for task: {0} (block {1}) -> bar {2} (block{3}) ", taskInd,
                       taskBlockInd, updateBarrierInd, barrierBlockIndex);
            dependenciesChanged = true;

            /*
              Task (taskInd) to barrier (updateBarrierInd) dependence spans across task blocks.
              Remove out-of-block connection from a producer

              /----------------------------------------------------------------------\
              0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6 - b6 - 7

            */

            // remove out-of-block connection from a producer (0->b6)
            removeProducer(updateBarrierInd, taskInd);

            // Add connection to nearest sync point. This is to assert that the taskInd updates a barrier after it
            // finishes.
            auto syncPointWaitBars = getWaitBarriers(taskSyncPoint.value());
            VPUX_THROW_WHEN(syncPointWaitBars.empty(), "Sync point '{0}' does not have wait barriers.", taskSyncPoint);

            updateControlMap(taskBlockInd, taskControlMap, controlMapOffset);

            if (!isSyncPoint(taskInd) &&
                !controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskInd - controlMapOffset,
                                                          taskSyncPoint.value() - controlMapOffset, false)) {
                // Link out-of-block producer (0) to the nearest sync-point wait barrier (b1).
                // If the task is a sync-point, or a connection already exists, nothing needs to be done.
                for (auto syncPointWaitBar : syncPointWaitBars) {
                    addProducer(syncPointWaitBar, taskInd);
                }
            }
        }

        // check task dependencies through its wait barriers
        auto taskWaitBarriers = getWaitBarriers(taskInd);
        for (const auto& waitBarrierInd : taskWaitBarriers) {
            auto barrierBlockIndex = getBarrierBlockIndex(waitBarrierInd);
            auto barrierLastProducer = getBarrierLatestProducer(waitBarrierInd);

            if (taskBlockInd == barrierBlockIndex) {
                // dependence does not span across blocks
                continue;
            }

            if (isSyncPoint(barrierLastProducer) && taskBlockInd - 1 == barrierBlockIndex) {
                // any task from a given block can wait for update barrier of sync-point from the previous block
                continue;
            }
            _log.trace("Out of block dependence for task: {0} (block {1}) -> bar {2} (block{3}) ", taskInd,
                       taskBlockInd, waitBarrierInd, barrierBlockIndex);

            dependenciesChanged = true;

            /*
              Barrier (waitBarrierInd) to task (taskInd) dependence spans across task blocks.
              Replace out-of-block connection to a consumer

                    /----------------------------------------------------------------------\
              0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6        7
                                                                              \------------/
              with a connection to its preceding sync point, if no such connection exists
            */

            // Remove out-of-block connection to a consumer (b0->7)
            removeConsumer(waitBarrierInd, taskInd);
            // Remove all barrier producers if there are no more consumers for barrier waitBarrierInd.
            // (Task 0 should be removed as producer of b0 if b0 is not consumed by any task).
            if (getBarrierConsumers(waitBarrierInd).empty()) {
                removeProducers(waitBarrierInd, getBarrierProducers(waitBarrierInd));
            }

            // get sync-point preceding current task (5)
            auto prevSyncPoint = getPreviousBlockSyncPoint(taskInd);
            // if prevSyncPoint doesn't exist, task taskInd cannot have out-of-block dependency and we shouldn't be
            // here.
            VPUX_THROW_UNLESS(prevSyncPoint.has_value(), "There's not sync-point preceding task {0}", taskInd);

            updateControlMap(taskBlockInd, taskControlMap, controlMapOffset);

            if (!isSyncPoint(taskInd) &&
                !controlPathExistsBetweenTasksInSameBlock(taskControlMap, prevSyncPoint.value() - controlMapOffset,
                                                          taskInd - controlMapOffset, false)) {
                // Link out-of-block consumer (7) to the preceding sync-point update barrier (b5).
                // If the task is a sync-point, or a connection already exists, nothing needs to be done.
                auto prevSyncPointUpdateBarriers = getUpdateBarriers(prevSyncPoint.value());
                for (const auto& syncPointUpdateBarrierInd : prevSyncPointUpdateBarriers) {
                    addConsumer(syncPointUpdateBarrierInd, taskInd);
                }
            }
        }
    }

    return dependenciesChanged;
}

std::optional<size_t> vpux::BarrierInfo::getPreviousBlockSyncPoint(size_t taskInd) const {
    auto blockInd = getControlGraphBlockIndex(taskInd);
    if (blockInd == 0) {
        return std::nullopt;
    } else if (blockInd == _syncTasksIds.size()) {
        return _syncTasksIds.back();
    }
    return _syncTasksIds[blockInd - 1];
}

void vpux::BarrierInfo::splitBarriersWithExceedingVariantCount(size_t availableSlots, size_t maxSlotsSum,
                                                               size_t maxAvailableSlots) {
    bool maxSlotsSumLimitEnabled = (maxSlotsSum < maxAvailableSlots);
    // split barriers if needed
    splitBarrierProducers(availableSlots, maxSlotsSum, maxSlotsSumLimitEnabled);
    // split barriers if needed
    splitBarrierConsumers(availableSlots, maxSlotsSum, maxSlotsSumLimitEnabled);
}

// split barriers if producer count > AVAILABLE SLOTS
// will produce ceil(NUM PRODUCERS / AVAILABLE SLOTS) barriers
// Note: if maxSlotsSumLimitEnabled is true, the maxAvailableSlots passed will be half of MAX_VARIANT_SUM not
// MAX_VARIANT_COUNT
/*
    x1  x2  ... xn             x1  ... x256   x257 ... xn
     \  \   /   /               \   |   /        \  |  /
         Bar0         -->          Bar0            Bar1
          |                            \          /
          u0                                u0
*/
void vpux::BarrierInfo::splitBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots) {
    // producers will be divided into batches
    const auto& barrierProducers = getBarrierProducers(barrierOp);
    auto barrierProducersCount = barrierProducers.size();
    _log.trace("Got barrier: '{0}' at '{1}' with barrier producer count '{2}'", barrierOp->getName(),
               barrierOp->getLoc(), barrierProducersCount);

    // consumers remain the same for all produced batched barriers
    const auto barrierConsumers = getBarrierConsumers(barrierOp);
    // create batches for producers
    auto producerBatches = createLegalVariantBatches(barrierProducers, availableSlots);

    mlir::OpBuilder builder(barrierOp);

    // create barriers based on batches
    for (const auto& producerBatch : producerBatches) {
        builder.setInsertionPoint(barrierOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(barrierOp->getLoc());
        _log.trace("Created new barrier '{0}'", newBarrier);
        addNewBarrier(newBarrier);

        for (const auto& producer : producerBatch) {
            _log.nest().trace("Add producer '{0}' to new barrier, whose slots are {1}", producer,
                              getNumOfSlotsUsed(getTaskOpAtIndex(producer)));
            addProducer(newBarrier, producer);
        }

        for (const auto& consumer : barrierConsumers) {
            _log.nest().trace("Add consumer '{0}' to new barrier, whose slots are {1}", consumer,
                              getNumOfSlotsUsed(getTaskOpAtIndex(consumer)));
            addConsumer(newBarrier, consumer);
        }
    }

    resetBarrier(barrierOp);
    _log.trace("Barrier successfully replaced with batch of barriers");
}

// split barriers if consumer count > AVAILABLE SLOTS
// will produce ceil(NUM PRODUCERS / AVAILABLE SLOTS) barriers
// Note: if maxSlotsSumLimitEnabled is true, the maxAvailableSlots passed will be half of MAX_VARIANT_SUM not
// MAX_VARIANT_COUNT
/*
          u0                                u0
          |                            /          \
         Bar0         -->          Bar0            Bar1
     /  /   \   \               /   |   \        /  |  \
    x1  x2  ... xn             x1  ... x256   x257 ... xn
*/
void vpux::BarrierInfo::splitBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots) {
    // consumers will be divided into batches
    const auto& barrierConsumers = getBarrierConsumers(barrierOp);
    auto barrierConsumersCount = barrierConsumers.size();
    _log.trace("Got barrier: '{0}' at '{1}' with barrier consumer count '{2}'", barrierOp->getName(),
               barrierOp->getLoc(), barrierConsumersCount);

    // producers remain the same for all produced batched barriers
    const auto barrierProducers = getBarrierProducers(barrierOp);
    // crete batches for consumers
    auto consumerBatches = createLegalVariantBatches(barrierConsumers, availableSlots);

    mlir::OpBuilder builder(barrierOp);

    // create barriers based on batches
    for (const auto& consumerBatch : consumerBatches) {
        builder.setInsertionPoint(barrierOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(barrierOp->getLoc());
        _log.trace("Created new barrier '{0}'", newBarrier);
        addNewBarrier(newBarrier);

        for (const auto& producer : barrierProducers) {
            _log.nest().trace("Add producer '{0}' to new barrier, whose slots are {1}", producer,
                              getNumOfSlotsUsed(getTaskOpAtIndex(producer)));
            addProducer(newBarrier, producer);
        }

        for (const auto& consumer : consumerBatch) {
            _log.nest().trace("Add consumer '{0}' to new barrier, whose slots are {1}", consumer,
                              getNumOfSlotsUsed(getTaskOpAtIndex(consumer)));
            addConsumer(newBarrier, consumer);
        }
    }

    resetBarrier(barrierOp);
    _log.trace("Barrier successfully replaced with batch of barriers");
}

void vpux::BarrierInfo::splitBarrierProducers(size_t availableSlots, size_t maxSlotsSum, bool maxSlotsSumLimitEnabled) {
    _func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto producerNum = getProducerSlotCount(barrierOp);
        auto consumerNum = getConsumerSlotCount(barrierOp);
        if (maxSlotsSumLimitEnabled && (producerNum + consumerNum <= maxSlotsSum)) {
            return;
        }

        if (producerNum <= availableSlots) {
            // valid producer configuration - barrier is legal
            return;
        }

        splitBarrierProducers(barrierOp, availableSlots);
    });
}

void vpux::BarrierInfo::splitBarrierConsumers(size_t availableSlots, size_t maxSlotsSum, bool maxSlotsSumLimitEnabled) {
    _func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto producerNum = getProducerSlotCount(barrierOp);
        auto consumerNum = getConsumerSlotCount(barrierOp);
        if (maxSlotsSumLimitEnabled && (producerNum + consumerNum <= maxSlotsSum)) {
            return;
        }

        if (consumerNum <= availableSlots) {
            // valid consumer configuration - barrier is legal
            return;
        }

        splitBarrierConsumers(barrierOp, availableSlots);
    });
}

void vpux::BarrierInfo::removeSyncTaskAttributes() {
    if (_syncTasksIds.empty()) {
        return;
    }

    for (const auto& syncTaskInd : _syncTasksIds) {
        auto syncTaskOp = getTaskOpAtIndex(syncTaskInd);
        VPUX_THROW_UNLESS(syncTaskOp->hasAttr(_syncTaskAttrName), "Expected sync task attribute at task index '{0}'",
                          syncTaskInd);
        syncTaskOp->removeAttr(_syncTaskAttrName);
    }
}

bool vpux::BarrierInfo::isSyncPoint(size_t taskIdx) {
    auto nTasks = getNumOfTasks();
    VPUX_THROW_WHEN(taskIdx >= nTasks, "Requested task index ({0}) is too large for the number of tasks ({1})", taskIdx,
                    nTasks);

    if (_syncTasksIds.empty() || taskIdx == nTasks - 1) {
        // graph split is not done, and last task cannot be sync point
        return false;
    }

    VPUX_THROW_WHEN(_taskToBlockMap.size() != nTasks, "_taskToBlockMap was not initialized (size {0}) for {1} tasks.",
                    _taskToBlockMap.size(), nTasks);
    return _taskToBlockMap[taskIdx + 1] != _taskToBlockMap[taskIdx];
}

size_t vpux::BarrierInfo::getControlGraphBlockCount() const {
    return _syncTasksIds.size() + 1;
}

std::pair<size_t, size_t> vpux::BarrierInfo::getControlGraphBlockTaskRange(size_t blockInd, bool blockStartSyncPoint,
                                                                           bool blockEndSyncPoint) const {
    VPUX_THROW_WHEN(blockInd >= getControlGraphBlockCount(), "Invalid task block index ({0})", blockInd);
    size_t blockStartInd, blockEndInd;
    if (blockInd == 0) {
        blockStartInd = 0;  // for first block the start index of the block is the index of the first task
    } else {
        blockStartInd = blockStartSyncPoint ? _syncTasksIds[blockInd - 1] : _syncTasksIds[blockInd - 1] + 1;
    }

    if (blockInd == getControlGraphBlockCount() - 1) {
        blockEndInd = getNumOfTasks() - 1;  // for last block the end index of the block is the index of the last task
    } else {
        blockEndInd = blockEndSyncPoint ? _syncTasksIds[blockInd] : _syncTasksIds[blockInd] - 1;
    }

    return std::make_pair(blockStartInd, blockEndInd);
}

SmallVector<std::set<size_t>> vpux::BarrierInfo::findParallelTasksWithBarrierDependence(
        const BarrierInfo::TaskSet& tasks, size_t maxCount) {
    SmallVector<std::set<size_t>> parallelTasks(tasks.size());

    size_t controlMapBlockInd = std::numeric_limits<size_t>::max();
    SmallVector<llvm::BitVector> taskControlMap;
    size_t controlMapOffset = 0;

    auto updateControlMap = [&](size_t taskBlockInd, SmallVector<llvm::BitVector>& taskControlMap,
                                size_t& controlMapOffset) {
        if (controlMapBlockInd != taskBlockInd) {
            std::tie(taskControlMap, controlMapOffset) =
                    buildTaskControlMap(taskBlockInd, /* considerTaskFifoDependency */ true);
            controlMapBlockInd = taskBlockInd;
        }
    };

    buildTaskQueueTypeMap();

    // For every task from the provided set, search for parallel tasks located earlier in the IR.
    for (auto [i, taskIdx] : tasks | indexed) {
        auto taskBlockIdx = getControlGraphBlockIndex(taskIdx);
        auto [fromIdx, _] = getControlGraphBlockTaskRange(taskBlockIdx, /* blockStartSyncPoint */ false,
                                                          /* blockEndSyncPoint */ true);
        auto toIdx = taskIdx;

        // build task control map
        updateControlMap(taskBlockIdx, taskControlMap, controlMapOffset);

        // Scan paths in decreasing task order between the current task and preceding tasks up to the preceding
        // sync-point. It is expected that typically parallel tasks will be located in the close vicinity of a given
        // task.
        for (auto idx : irange(fromIdx, toIdx) | reversed) {
            if (!controlPathExistsBetweenTasksInSameBlock(taskControlMap, idx - controlMapOffset,
                                                          taskIdx - controlMapOffset, false)) {
                if (getUpdateBarriers(idx).empty() && getWaitBarriers(idx).empty()) {
                    continue;
                }
                parallelTasks[i].insert(idx);
                if (parallelTasks[i].size() >= maxCount) {
                    break;
                }
            }
        }
    }

    // If not found, search for parallel tasks located latter in the IR relative to the provided task
    for (auto [i, taskIdx] : tasks | indexed) {
        if (parallelTasks[i].size() >= maxCount) {
            continue;
        }
        auto taskBlockIdx = getControlGraphBlockIndex(taskIdx);
        auto [_, toIdx] = getControlGraphBlockTaskRange(taskBlockIdx, /* blockStartSyncPoint */ false,
                                                        /* blockEndSyncPoint */ true);
        auto fromIdx = taskIdx + 1;  // start from index of the following task

        // build task control map
        updateControlMap(taskBlockIdx, taskControlMap, controlMapOffset);

        // Scan paths in increasing task order between the current task and following tasks up to the next
        // sync-point (or the last task) inclusively. It is expected that typically parallel tasks will be located in
        // the close vicinity of a given task.
        for (auto idx : irange(fromIdx, toIdx + 1)) {
            if (!controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskIdx - controlMapOffset,
                                                          idx - controlMapOffset, false)) {
                if (getUpdateBarriers(idx).empty() && getWaitBarriers(idx).empty()) {
                    continue;
                }
                parallelTasks[i].insert(idx);
                if (parallelTasks[i].size() >= maxCount) {
                    break;
                }
            }
        }
    }

    return parallelTasks;
}

//
// optimizeBarriers
//
void vpux::BarrierInfo::optimizeBarriers(bool checkValidSlotCount, bool considerTaskFifoDependency) {
    if (considerTaskFifoDependency) {
        buildTaskQueueTypeMap();
    }

    // A -> B -> C
    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    // Note: It also will merge barriers which have the same producers but different consumers

    // Barrier are optimized based on order of task ops

    _log.trace("Optimize barriers");
    _log = _log.nest();
    _log.trace("Total tasks count: {0}, total barrier count: {1}", getNumOfTasks(), _allBarrierOps.size());

    auto numBarriersBeforeOptimization = getNumOfBarrierOps();

    // Perform optimization in tasks blocks matching the distribution of synchronization points.
    // For update barriers and wait barriers we need to include sync points on both ends of the block
    // because tasks from within a given range can have updateBarriers whose consumer
    // is the upper bound sync point, and they can have waitBarriers whose producer is the lower bound sync point.
    for (size_t taskBlockIndex = 0; taskBlockIndex < getControlGraphBlockCount(); ++taskBlockIndex) {
        auto [blockStartInd, blockEndInd] = getControlGraphBlockTaskRange(taskBlockIndex);
        _log.trace("Block {0}, task range [{1}, {2}] ({3} tasks)", taskBlockIndex, blockStartInd, blockEndInd,
                   blockEndInd - blockStartInd + 1);

        if (considerTaskFifoDependency) {
            // TODO: E#126579
            auto newDepsCount = createBarrierDependenciesImpliedByFIFO(
                    taskBlockIndex, mlir::DenseSet<vpux::VPU::ExecutorKind>({VPU::ExecutorKind::DMA_NN}));
            _log.trace("Created {0} temporary dependencies", newDepsCount);
        }

        // optimize producers
        _log.trace("Optimize producers / update barriers");
        optimizeBarrierProducers(taskBlockIndex);

        // optimize barriers which have the same producers but different consumers
        _log.trace("Optimize barriers / same producers, different consumers");
        optimizeBarriersWithSameProducers(taskBlockIndex, checkValidSlotCount);

        // optimize consumers
        _log.trace("Optimize consumers / wait barriers");
        optimizeBarrierConsumers(taskBlockIndex);

        auto numBarriersAfterOptimization = getNumOfBarrierOps();
        VPUX_THROW_WHEN(numBarriersAfterOptimization != numBarriersBeforeOptimization,
                        "Number of barriers changed during optimization.");

        if (considerTaskFifoDependency) {
            auto removedDepsCount = removeBarrierDependenciesImpliedByFIFO();
            _log.trace("Removed {0} temporary barriers", removedDepsCount);

            numBarriersAfterOptimization = getNumOfBarrierOps();
            VPUX_THROW_WHEN(numBarriersAfterOptimization != numBarriersBeforeOptimization,
                            "Incorrect number of barriers after removing barrier representation of FIFO dependencies.");
        }
    }

    _log.trace("Total tasks count: {0}, total barrier count: {1}", getNumOfTasks(), _allBarrierOps.size());
    _log = _log.unnest();
}

void vpux::BarrierInfo::optimizeBarrierProducers(size_t blockIdx) {
    // TODO: E#79318 optimize loops
    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
    const auto barriersRangeVec = getBarriersForTaskBlock(blockIdx, /* blockStartSyncPoint */ true,
                                                          /* blockEndSyncPoint */ true, /* updateBarriers */ true);
    if (barriersRangeVec.empty()) {
        return;
    }
    size_t barrierOffset = barriersRangeVec[0];

    blockEndInd = std::min(blockEndInd, getNumOfTasks() - 1);
    VPUX_THROW_WHEN(blockStartInd > blockEndInd, "Lower bound of task optimization range is greater than upper bound");

    auto blockSize = blockEndInd - blockStartInd + 1;
    auto barrierSize = barriersRangeVec.back() - barriersRangeVec.front() + 1;

    SmallVector<llvm::BitVector> updateBarriers;
    resizeBitMap(updateBarriers, blockSize, checked_cast<uint32_t>(barrierSize));

    _log.nest().trace("Collect redundant dependencies");
    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        setBarrierMask(updateBarriers[taskInd - blockStartInd], _taskUpdateBarriers[taskInd], barrierOffset);
        if (taskInd == blockEndInd && isSyncPoint(taskInd)) {
            // ignore updating updateBarriers for upper bound sync point because it only has barriers whose consumers
            // are tasks from outside of the current range.
            continue;
        }
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, childTaskInd)) {
                    updateBarriers[taskInd - blockStartInd] |=
                            updateBarriers[static_cast<size_t>(childTaskInd - blockStartInd)];
                } else {
                    VPUX_THROW("Task {0} has update barriers with a consumer ({1}) from outside of the current "
                               "range [{2}, {3}].",
                               taskInd, childTaskInd, blockStartInd, blockEndInd);
                }
            }
        }
    }

    _log.nest().trace("Remove redundant dependencies");
    for (size_t taskInd = static_cast<unsigned>(blockStartInd); taskInd <= static_cast<unsigned>(blockEndInd);
         ++taskInd) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, childTaskInd)) {
                    updateBarriers[taskInd - blockStartInd].reset(
                            updateBarriers[static_cast<size_t>(childTaskInd - blockStartInd)]);
                }
            }
        }
        TaskSet targetUpdateBarriers;
        for (auto bar : updateBarriers[taskInd - blockStartInd].set_bits()) {
            targetUpdateBarriers.insert(bar + barrierOffset);
        }
        setUpdateBarriers(taskInd, targetUpdateBarriers);
    }
}

void vpux::BarrierInfo::optimizeBarrierConsumers(size_t blockIdx) {
    // TODO: E#79318 optimize loops
    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
    const auto barriersRangeVec = getBarriersForTaskBlock(blockIdx, /* blockStartSyncPoint  */ true,
                                                          /* blockEndSyncPoint */ true, /* updateBarriers */ false);
    if (barriersRangeVec.empty()) {
        return;
    }
    size_t barrierOffset = barriersRangeVec[0];

    blockEndInd = std::min(blockEndInd, getNumOfTasks() - 1);
    VPUX_THROW_WHEN(blockStartInd > blockEndInd, "Lower bound of task optimization range is greater than upper bound");

    auto blockSize = blockEndInd - blockStartInd + 1;
    auto barrierSize = barriersRangeVec.back() - barriersRangeVec.front() + 1;
    // optimize consumers

    SmallVector<llvm::BitVector> waitBarriers;
    resizeBitMap(waitBarriers, blockSize, checked_cast<uint32_t>(barrierSize));

    _log.nest().trace("Collect redundant dependencies");
    for (auto taskInd = blockStartInd; taskInd <= blockEndInd; ++taskInd) {
        setBarrierMask(waitBarriers[taskInd - blockStartInd], _taskWaitBarriers[taskInd], barrierOffset);
        if (taskInd == blockStartInd && isSyncPoint(taskInd)) {
            // ignore updating waitBarriers for lower bound sync point because it only has barriers whose producers
            // are tasks from outside of the current range.
            continue;
        }
        for (auto waitBarrierInd : _taskWaitBarriers[taskInd]) {
            for (auto parentTaskInd : _barrierProducerMap[waitBarrierInd]) {
                if (inRange(blockStartInd, blockEndInd, parentTaskInd)) {
                    waitBarriers[taskInd - blockStartInd] |=
                            waitBarriers[static_cast<size_t>(parentTaskInd - blockStartInd)];
                } else {
                    VPUX_THROW("Task {0} has wait barriers with a producer ({1}) from outside of the current range "
                               "[{2}, {3}].",
                               taskInd, parentTaskInd, blockStartInd, blockEndInd);
                }
            }
        }
    }

    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        for (auto waitBarrierInd : _taskWaitBarriers[taskInd]) {
            for (auto parentTaskInd : _barrierProducerMap[waitBarrierInd]) {
                if (inRange(blockStartInd, blockEndInd, parentTaskInd)) {
                    waitBarriers[taskInd - blockStartInd].reset(
                            waitBarriers[static_cast<size_t>(parentTaskInd - blockStartInd)]);
                }
            }
        }

        TaskSet targetWaitBarriers;
        for (auto bar : waitBarriers[taskInd - blockStartInd].set_bits()) {
            targetWaitBarriers.insert(bar + barrierOffset);
        }
        setWaitBarriers(taskInd, targetWaitBarriers);
    }
}

void vpux::BarrierInfo::optimizeBarriersWithSameProducers(size_t blockIdx, bool checkValidSlotCount) {
    // Collect a vector of barrier indexes for current task range.
    // If a new barrier is created between subsequent calls to optimizeBarrier (eg. during looped linearization),
    // iterating over barriersRangeVec will not imply iterating over largely overlapping barrier index ranges
    // when moving from one task block to another.
    const auto barriersRangeVec =
            getBarriersForTaskBlock(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ false);

    if (barriersRangeVec.empty()) {
        return;
    }
    _log.nest().trace("Barrier range [{0}, {1}]", barriersRangeVec.front(), barriersRangeVec.back());
    size_t maxBarrierIndex = _allBarrierOps.size() + _fifoDependencies.size() - 1;
    VPUX_THROW_WHEN(barriersRangeVec.front() > maxBarrierIndex,
                    "Invalid barrier index for lower limit of barrier range optimization");
    VPUX_THROW_WHEN(barriersRangeVec.back() > maxBarrierIndex,
                    "Invalid barrier index for upper limit of barrier range optimization");

    const auto maxVariantCount = getBarrierMaxVariantSum();

    auto addFunc = [&](size_t sum, size_t taskInd) {
        return sum + getNumOfSlotsUsedByTask(getTaskOpAtIndex(taskInd));
    };

    // Check the slot count is valid or not if the two barriers are merged
    auto legalMergeSlotCount = [&](const size_t& bar1, const size_t& bar2) {
        // Currently used by ReduceBArrierDependencies, this is possible to be done here because
        // SplitExceedingVariantCountBarrier will legalize this in barrierLegalizationPipeline
        if (!checkValidSlotCount) {
            return true;
        }
        size_t slotCount = std::accumulate(_barrierProducerMap[bar1].begin(), _barrierProducerMap[bar1].end(),
                                           static_cast<size_t>(0), addFunc);
        BarrierInfo::TaskSet consumers = _barrierConsumerMap[bar1];
        // Use set operation to remove duplicated consumers
        llvm::set_union(consumers, _barrierConsumerMap[bar2]);
        slotCount = std::accumulate(consumers.begin(), consumers.end(), slotCount, addFunc);
        return slotCount <= maxVariantCount;
    };

    // copy barrier map onto vector for quicker comparisons between different producer sets
    SmallVector<SmallVector<size_t>> blockBarrierProducerMap(barriersRangeVec.size());
    for (size_t ind1 = 0; ind1 < barriersRangeVec.size(); ++ind1) {
        size_t barInd = barriersRangeVec[ind1];
        std::copy(_barrierProducerMap[barInd].begin(), _barrierProducerMap[barInd].end(),
                  std::back_inserter(blockBarrierProducerMap[ind1]));
    }
    _log = _log.nest();

    for (size_t ind1 = 0; ind1 < barriersRangeVec.size(); ++ind1) {
        size_t barInd = barriersRangeVec[ind1];
        for (size_t ind2 = ind1 + 1; ind2 < barriersRangeVec.size(); ++ind2) {
            size_t childBarInd = barriersRangeVec[ind2];
            if (blockBarrierProducerMap[ind1] == blockBarrierProducerMap[ind2]) {
                _log.trace("Same producers for barId '{0}' '{1}'", barInd, childBarInd);
                if (legalMergeSlotCount(barInd, childBarInd)) {
                    for (auto consumerInd : _barrierConsumerMap[childBarInd]) {
                        // move all consumers to one barrier
                        addConsumer(barInd, static_cast<size_t>(consumerInd));
                    }
                    _log.trace("New consumers number - {0}", _barrierConsumerMap[barInd].size());
                    resetBarrier(childBarInd);
                    blockBarrierProducerMap[ind2].clear();
                }
            }
        }
    }
    _log = _log.unnest();
}

bool vpux::BarrierInfo::canMergeBarriersForTasks(const BarrierInfo::TaskSet& producers, size_t availableSlots) {
    size_t producerSlotCount = 0;
    for (const auto& producerInd : producers) {
        producerSlotCount += getNumOfSlotsUsedByTask(getTaskOpAtIndex(producerInd));

        if (producerSlotCount > availableSlots) {
            // exceeding producer slot count
            return false;
        }
    }

    return true;
}

void vpux::BarrierInfo::enableUnevenVariantSplit() {
    _enableUnevenVariantSplit = true;
}

// modify barriers for task such that it is driven by a single wait barrier by merging parallel wait barriers
// or linearizing the producers of parallel wait barriers
bool vpux::BarrierInfo::eliminateParallelWaitBarriers(size_t taskInd, size_t availableSlots,
                                                      bool considerTaskExecutorType) {
    const auto waitBarriers = getWaitBarriers(taskInd);  // parallel wait barriers
    auto log = _log.nest();

    auto getBarriersProducerTasks = [&](const BarrierInfo::TaskSet& waitBarriers) {
        BarrierInfo::TaskSet newBarrierProducers;
        for (auto& waitBarrierInd : waitBarriers) {
            // merge all producers
            const auto& barrierProducers = getBarrierProducers(waitBarrierInd);
            llvm::set_union(newBarrierProducers, barrierProducers);
        }

        return newBarrierProducers;
    };

    auto getBarriersConsumerTasks = [&](const BarrierInfo::TaskSet& parallelWaitBarriers) {
        BarrierInfo::TaskSet parallelConsumers;
        for (auto& waitBarrier : parallelWaitBarriers) {
            llvm::set_union(parallelConsumers, getBarrierConsumers(waitBarrier));
        }
        return parallelConsumers;
    };

    BarrierInfo::TaskSet parallelConsumers = getBarriersConsumerTasks(waitBarriers);

    auto getBarriersConsumerTasksWithWaitBarriers = [&](const BarrierInfo::TaskSet& waitBarriers,
                                                        size_t availableSlotsForConsumer) {
        // select consumers from within parallelConsumers with given waitBarriers
        BarrierInfo::TaskSet parallelConsumersSameWaitBarriers;
        size_t parallelConsumerSlotCount = 0;

        for (auto& consumer : parallelConsumers) {
            if (waitBarriers != getWaitBarriers(consumer)) {
                continue;
            }

            parallelConsumerSlotCount += getNumOfSlotsUsedByTask(getTaskOpAtIndex(consumer));
            if (parallelConsumerSlotCount > availableSlotsForConsumer) {
                break;
            }

            parallelConsumersSameWaitBarriers.insert(consumer);
        }

        return parallelConsumersSameWaitBarriers;
    };

    auto unlinkTaskFromParallelBarriers = [&](const BarrierInfo::TaskSet& tasks,
                                              const BarrierInfo::TaskSet& waitBarriers) {
        // remove consumer task from parallel barriers
        for (auto& waitBarrierInd : waitBarriers) {
            const auto barrierConsumers = getBarrierConsumers(waitBarrierInd);
            const auto waitBarrier = getBarrierOpAtIndex(waitBarrierInd);

            // remove link from parallel barriers
            if (barrierConsumers.size() == tasks.size()) {
                // if only consumer, barrier can be reset
                resetBarrier(waitBarrier);
            } else {
                for (auto& taskInd : tasks) {
                    removeConsumer(waitBarrier, taskInd);
                }
            }
        }
    };

    log.trace("Got '{0}' parallel wait barriers for task '{1}'", waitBarriers.size(), taskInd);

    auto availableSlotsForConsumer = availableSlots;
    const auto totalSlots = 2 * availableSlots;  // by default the slots are evenly divided for producer and consumer
    auto parallelProducers = getBarriersProducerTasks(waitBarriers);
    size_t parallelProducerSlotCount = 0;
    for (auto& producer : parallelProducers) {
        parallelProducerSlotCount += getNumOfSlotsUsedByTask(getTaskOpAtIndex(producer));
    }
    // make full use of the total slots
    // originally the slots are evenly split for producers and consumers,
    // e.g., 64 slots in total, 32 for producer and 32 for consumer
    // when the producer exceeds 32 the excess producers are linearized, leading to DPU stalls
    // try to distribute more slots for producer or consumer and check the feasibility
    if (_enableUnevenVariantSplit && (parallelProducerSlotCount < totalSlots)) {
        // check if every consumer could fit consumer-slots
        // if false, use equal slots for producer and consumer (original logic)
        // else, assign slots for producer and use the remaining slots for consumer
        const bool everyConsumerFitProvidedSlots = llvm::all_of(parallelConsumers, [&](auto consumer) {
            return waitBarriers != getWaitBarriers(consumer) ||
                   getNumOfSlotsUsedByTask(getTaskOpAtIndex(consumer)) <= totalSlots - parallelProducerSlotCount;
        });
        if (everyConsumerFitProvidedSlots) {
            availableSlotsForConsumer = totalSlots - parallelProducerSlotCount;
        }
    }

    auto parallelConsumersToMerge = getBarriersConsumerTasksWithWaitBarriers(waitBarriers, availableSlotsForConsumer);
    VPUX_THROW_WHEN(parallelConsumersToMerge.empty(), "parallel consumers are empty");

    // check if total slot count required by parallelProducers <= available slots for producers
    if (canMergeBarriersForTasks(parallelProducers, totalSlots - availableSlotsForConsumer)) {
        log.nest().trace("Can merge '{0}' parallel producers for task '{1}'", parallelProducers.size(), taskInd);
        mergeLegalParallelProducers(taskInd, parallelProducers, parallelConsumersToMerge);
    } else {
        log.nest().trace("Have to linearize '{0}' parallel producers for task '{1}'", parallelProducers.size(),
                         taskInd);
        linearizeLegalParallelProducers(taskInd, parallelProducers, parallelConsumersToMerge, availableSlots,
                                        considerTaskExecutorType);
    }

    for (auto& id : parallelConsumersToMerge) {
        log.trace("Unlink parallel barriers from task '{0}'", id);
    }
    unlinkTaskFromParallelBarriers(parallelConsumersToMerge, waitBarriers);
    log.trace("Task '{0}' now has one wait barrier", taskInd);
    return true;
}

SmallVector<BarrierInfo::TaskSet> BarrierInfo::createProducerBatches(const BarrierInfo::TaskSet& waitBarriers,
                                                                     size_t availableSlots,
                                                                     bool considerTaskExecutorType) {
    // try to create batches of producers using existing barriers if producers satisfy order constraint
    SmallVector<BarrierInfo::TaskSet> legalBatches;

    auto prevLastUserInd = std::numeric_limits<size_t>::max();  // last user index in the previous batch
    for (auto& waitBarrierInd : waitBarriers) {
        const auto& barrierProducers = getBarrierProducers(waitBarrierInd);
        if (legalBatches.empty()) {
            prevLastUserInd = VPURT::getMaxEntry(barrierProducers);
            legalBatches =
                    canMergeBarriersForTasks(barrierProducers, availableSlots)
                            ? SmallVector<BarrierInfo::TaskSet>{barrierProducers}
                            : createLegalVariantBatches(barrierProducers, availableSlots, considerTaskExecutorType);
            continue;
        }

        const auto firstUserInd = VPURT::getMinEntry(barrierProducers);
        if (prevLastUserInd >= firstUserInd) {
            // producers do not satisfy order constraint
            return SmallVector<BarrierInfo::TaskSet>{};
        }

        prevLastUserInd = VPURT::getMaxEntry(barrierProducers);
        auto currentBatchPlusBarrierProducers = legalBatches.back();
        llvm::set_union(currentBatchPlusBarrierProducers, barrierProducers);

        if (canMergeBarriersForTasks(currentBatchPlusBarrierProducers, availableSlots)) {
            // can add to the same batch
            legalBatches.back() = std::move(currentBatchPlusBarrierProducers);
        } else {
            // possible improvement: if the entire batch doesn't fit, consider filling-in the last batch with a portion
            // of tasks from barrierProducers (E140023).
            // Need to create a new batch
            if (canMergeBarriersForTasks(barrierProducers, availableSlots)) {
                legalBatches.push_back(barrierProducers);
            } else {
                legalBatches.append(
                        createLegalVariantBatches(barrierProducers, availableSlots, considerTaskExecutorType));
            }
        }
    }

    return legalBatches;
}

/*
    Linearize wait barriers if task wait barrier count > 1 by linearizing legal batches of parallel wait barriers

        x..xn    y..ym             x..xn
          |        |              /     \
        Bar0     Bar1     -->   Bar0     Bar1
       /    \   /    \           |        |
    u..ui   o..oj   a..ak      u..ui    y..ym
                                       /     \
                                     Bar2   Bar3
                                      |      |
                                    o..oj   a..ak
*/
void vpux::BarrierInfo::linearizeLegalParallelProducers(size_t taskInd, const BarrierInfo::TaskSet& parallelProducers,
                                                        const BarrierInfo::TaskSet& parallelConsumers,
                                                        size_t availableSlots, bool considerTaskExecutorType) {
    // create legal batches of barrier producers
    auto legalBatches = createProducerBatches(getWaitBarriers(taskInd), availableSlots, considerTaskExecutorType);
    if (legalBatches.empty()) {
        // create new batches of producers that satisfy order constraint
        legalBatches = createLegalVariantBatches(parallelProducers, availableSlots, considerTaskExecutorType);
    }

    // last batch is the current task
    BarrierInfo::TaskSet nextBatch = parallelConsumers;
    auto taskOp = getTaskOpAtIndex(taskInd);
    mlir::OpBuilder builder(taskOp);

    // create a new barrier between batches
    for (const auto& batch : legalBatches | reversed) {
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp.getLoc());
        addNewBarrier(newBarrier);

        for (auto& barrierProducer : batch) {
            addProducer(newBarrier, barrierProducer);
        }

        for (auto& barrierConsumer : nextBatch) {
            addConsumer(newBarrier, barrierConsumer);
        }

        nextBatch = batch;
    }
}

// merge wait barriers if task wait barrier count > 1 by
// creating a new barrier replacing parallel wait barriers
/*
        x..xn    y..ym             x..xn    y..ym
          |        |              /    \   /    \
        Bar0     Bar1     -->   Bar0    Bar1    Bar2
       /    \   /    \           |       |       |
    u..ui   o..oj   a..ak      u..ui   o..oj   a..ak
*/
void vpux::BarrierInfo::mergeLegalParallelProducers(size_t taskInd, const BarrierInfo::TaskSet& parallelProducers,
                                                    const BarrierInfo::TaskSet& parallelConsumers) {
    // create a new barrier for all parallel producers
    auto taskOp = getTaskOpAtIndex(taskInd);
    mlir::OpBuilder builder(taskOp);
    auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp.getLoc());
    addNewBarrier(newBarrier);

    // add all legal producers to new barrier
    for (auto& barrierProducer : parallelProducers) {
        addProducer(newBarrier, barrierProducer);
    }

    // add all parallel consumers with same wait barriers
    for (auto& consumer : parallelConsumers) {
        addConsumer(newBarrier, consumer);
    }
}

void vpux::BarrierInfo::shareWaitAndUpdateBarriers(size_t availableSlots) {
    // This method requires task queue map built with buildTaskQueueTypeMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "TaskQueueTypeMap has not been created");
    for (const auto& entry : _taskQueueTypeMap) {
        // operate per queue
        const auto& dmaQueue = entry.second;
        // handle update barriers
        BarrierInfo::TaskSet tasksNoUpdateBarrier;
        for (const auto& taskInd : dmaQueue.set_bits()) {
            const auto taskUpdateBarriers = getUpdateBarriers(taskInd);
            if (taskUpdateBarriers.empty()) {
                tasksNoUpdateBarrier.insert(taskInd);
            } else if (!tasksNoUpdateBarrier.empty()) {
                linkTasksToBarriers(tasksNoUpdateBarrier, taskUpdateBarriers, false, availableSlots);
                tasksNoUpdateBarrier.clear();
            }
        }

        // handle wait barriers
        BarrierInfo::TaskSet tasksNoWaitBarrier;
        for (auto taskInd = dmaQueue.find_last(); taskInd != -1;) {
            const auto taskWaitBarriers = getWaitBarriers(static_cast<size_t>(taskInd));
            if (taskWaitBarriers.empty()) {
                tasksNoWaitBarrier.insert(static_cast<size_t>(taskInd));
            } else if (!tasksNoWaitBarrier.empty()) {
                linkTasksToBarriers(tasksNoWaitBarrier, taskWaitBarriers, true, availableSlots);
                tasksNoWaitBarrier.clear();
            }
            taskInd = dmaQueue.find_prev(taskInd);
        }
    }
}

// link tasks to barriers and create legal variant batches if needed
void vpux::BarrierInfo::linkTasksToBarriers(const BarrierInfo::TaskSet& tasksToAdd,
                                            const BarrierInfo::TaskSet& newBarriers, bool waitBarriers,
                                            size_t availableSlots) {
    VPUX_THROW_UNLESS(_fifoDependencies.size() == 0,
                      "Cannot create new barriers when FIFO dependencies are created but not represented in IR");
    mlir::OpBuilder builder(getBarrierOpAtIndex(*newBarriers.begin()));
    BarrierInfo::TaskSet newBarrierBatch;

    for (const auto& barrierIdn : newBarriers) {
        BarrierInfo::TaskSet barrierTasks;
        if (waitBarriers) {
            barrierTasks = getBarrierConsumers(barrierIdn);
        } else {
            barrierTasks = getBarrierProducers(barrierIdn);
        }

        llvm::set_union(barrierTasks, tasksToAdd);
        auto batches = createLegalVariantBatches(barrierTasks, availableSlots);
        if (batches.size() > 1) {
            auto insertionPoint = getBarrierOpAtIndex(barrierIdn);
            BarrierInfo::TaskSet prevBatch;
            if (waitBarriers) {
                prevBatch = getBarrierProducers(barrierIdn);
            } else {
                batches.push_back(getBarrierConsumers(barrierIdn));
            }
            for (const auto& batch : batches) {
                if (!prevBatch.empty()) {
                    builder.setInsertionPoint(insertionPoint);
                    const auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(insertionPoint->getLoc());
                    addNewBarrier(newBarrier);
                    const auto newBarrierIdn = getIndex(newBarrier);
                    newBarrierBatch.insert(newBarrierIdn);
                    addProducers(newBarrierIdn, prevBatch);
                    addConsumers(newBarrierIdn, batch);
                }
                prevBatch = batch;
            }
            resetBarrier(barrierIdn);
        } else if (waitBarriers) {
            addConsumers(barrierIdn, tasksToAdd);
            newBarrierBatch.insert(barrierIdn);
        } else {
            addProducers(barrierIdn, tasksToAdd);
            newBarrierBatch.insert(barrierIdn);
        }
    }
}

size_t vpux::BarrierInfo::getBarrierLatestProducer(size_t barInd) {
    auto barProducers = getBarrierProducers(barInd);
    if (barProducers.empty()) {
        VPUX_THROW("Barrier {0} does not have any producers.", barInd);
    }
    std::set<size_t> barProducersOrd(barProducers.begin(), barProducers.end());
    return *barProducersOrd.rbegin();
}

unsigned vpux::BarrierInfo::createBarrierDependenciesImpliedByFIFO(
        size_t blockIdx, std::optional<mlir::DenseSet<vpux::VPU::ExecutorKind>> executorKind) {
    // This method requires task queue map built with buildTaskQueueTypeMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "TaskQueueTypeMap has not been created");
    unsigned newDepsCount = 0;
    auto addTemporaryBarrierDependence = [&](size_t prevIndex, size_t currentIndex) {
        // Add barrier dependence between task with index prevIndex and task with index currentIndex but do not create a
        // new barrier in the IR. This creates a temporary state of mismatch between number of barriers in the IR and
        // number of barriers tracked by the _barrierProducerMap and _barrierConsumerMap maps. This state is used to
        // perform barrier optimizations taking into account FIFO dependencies. These dependencies will be
        // removed after optimization is done for the sake of internal self-consistency.
        size_t tempBarrierInd = _barrierProducerMap.size();
        _barrierProducerMap.push_back({});
        _barrierConsumerMap.push_back({});

        addProducer(tempBarrierInd, prevIndex);
        addConsumer(tempBarrierInd, currentIndex);
        newDepsCount += 2;
        return tempBarrierInd;
    };

    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ false);

    // Go over the FIFOs, if the two tasks are in the same blockRange and does not have barrier dependency
    // add a temporary dependency between them
    for (const auto& entry : _taskQueueTypeMap) {
        if (executorKind.has_value() && !executorKind.value().contains(entry.first.type)) {
            continue;
        }
        const auto& fifoTasks = entry.second;

        if (fifoTasks.count() < 2) {
            continue;
        }
        int prevIndex =
                fifoTasks.find_first_in(static_cast<unsigned>(blockStartInd), static_cast<unsigned>(blockEndInd));
        if (prevIndex == -1) {  // no tasks in this block
            continue;
        }
        int currIndex = fifoTasks.find_next(prevIndex);

        while (currIndex <= fifoTasks.find_last() && currIndex != -1 && static_cast<size_t>(currIndex) <= blockEndInd) {
            size_t commonBarrier = -1;
            // Check if the two tasks do not have barrier dependency already and that they're in the same block
            if (!hasBarrierDependency(static_cast<size_t>(prevIndex), static_cast<size_t>(currIndex), commonBarrier)) {
                // add temporary dependency from task prevIndex to currIndex
                auto newBarrierIdx = addTemporaryBarrierDependence(prevIndex, currIndex);
                _fifoDependencies.emplace_back(newBarrierIdx, prevIndex, currIndex);
            }
            prevIndex = currIndex;
            currIndex = fifoTasks.find_next(currIndex);
        }
    }

    return newDepsCount;
}

unsigned vpux::BarrierInfo::createBarrierDependenciesBetweenExecutionGroups(size_t blockIdx,
                                                                            vpux::VPU::ExecutorKind executorKind,
                                                                            ExecutionGroupListMap& executionGroups) {
    // This method requires task queue map built with buildTaskQueueTypeMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "TaskQueueTypeMap has not been created");
    auto createBarrierDependencies = [&](const size_t producer, const size_t consumer) {
        auto newBarrierIdx = _barrierProducerMap.size();
        _barrierProducerMap.push_back({});
        _barrierConsumerMap.push_back({});

        addProducer(newBarrierIdx, producer);
        addConsumer(newBarrierIdx, consumer);

        auto taskOp = getTaskOpAtIndex(producer);
        mlir::OpBuilder builder(taskOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp->getLoc());
        newBarrier->setAttr(_barrierIndexAttrName, getIntAttr(newBarrier.getContext(), newBarrierIdx));
        _allBarrierOps.push_back(newBarrier);
        _log.trace("Add new barrier '{0}', new barrier size '{1}'", newBarrierIdx, _allBarrierOps.size());

        return newBarrierIdx;
    };

    // build task control map for current block and all executor kinds
    auto [taskControlMapFifo, controlMapOffset] = buildTaskControlMap(blockIdx, /* considerTaskFifoDependency */ true);

    auto hasExecutorKind = [&](vpux::VPU::ExecutorKind executorKind) {
        for (auto [queue, _] : executionGroups) {
            if (queue.type == executorKind) {
                return true;
            }
        }
        return false;
    };

    VPUX_THROW_UNLESS(hasExecutorKind(executorKind), "Execution groups map is not initialized for {0}", executorKind);

    // Insert barrier between execution groups when required
    size_t newBarriersCount = 0;
    for (auto [queue, fifoExecGroups] : executionGroups) {
        if (fifoExecGroups.size() <= 1) {
            continue;  // No dependencies need to be created if there are no groups or all tasks fit into a single
                       // group.
        }

        int numberOfGroups = static_cast<int>(fifoExecGroups.size());
        for (int grpIdx = 0; grpIdx < numberOfGroups - 1; grpIdx++) {
            if (!hasBarrierDependencyRequiredForDescriptorFetch(grpIdx, blockIdx, fifoExecGroups, taskControlMapFifo,
                                                                controlMapOffset, false)) {
                // get producers and consumers for the new barrier
                auto lastTaskInGroup = *fifoExecGroups[grpIdx].rbegin();
                auto firstTaskInChildGroup = *fifoExecGroups[grpIdx + 1].begin();

                createBarrierDependencies(lastTaskInGroup, firstTaskInChildGroup);
                newBarriersCount++;
            }
        }
    }
    _log.trace("New shared barriers in block {0}: {1}", blockIdx, newBarriersCount);
    return newBarriersCount;
}

unsigned vpux::BarrierInfo::removeBarrierDependenciesImpliedByFIFO() {
    unsigned removedDepsCount = 0;

    // Remove barriers implicitly implied by FIFO dependency that were temporarily added between neighbour tasks
    // on the same FIFO
    for (const auto& dep : _fifoDependencies) {
        size_t commonBarrier = -1;
        const auto [barrierIdx, parentTaskIdx, childTaskIdx] = dep;

        if (hasBarrierDependency(parentTaskIdx, childTaskIdx, commonBarrier) && commonBarrier == barrierIdx) {
            removeProducer(barrierIdx, parentTaskIdx);
            removeConsumer(barrierIdx, childTaskIdx);
            removedDepsCount += 2;
        }
    }

    // restore original sizes of barrier producer and consumer maps after removing temporary barriers associated
    // with FIFO
    auto getOptimizedBarrierMap = [&](SmallVector<TaskSet>& barrierMap, bool producerMap) {
        auto numUsersOfTemporaryFIFObarriers =
                std::accumulate(barrierMap.end() - _fifoDependencies.size(), barrierMap.end(), 0U,
                                [](const size_t& a, const TaskSet& b) {
                                    return a + b.size();
                                });
        VPUX_THROW_UNLESS(numUsersOfTemporaryFIFObarriers == 0, "Found {0} for temporary barrier.",
                          producerMap ? "producers" : "consumers");
        SmallVector<TaskSet> optimizedBarrierMap(barrierMap.begin(), barrierMap.end() - _fifoDependencies.size());
        return optimizedBarrierMap;
    };

    _barrierProducerMap = getOptimizedBarrierMap(_barrierProducerMap, /* producerMap */ true);
    _barrierConsumerMap = getOptimizedBarrierMap(_barrierConsumerMap, /* producerMap */ false);

    _fifoDependencies.clear();
    return removedDepsCount;
}

// iteratively merge parallel wait barriers
bool vpux::BarrierInfo::ensureTasksDrivenBySingleBarrier(size_t availableSlots, bool mergeWaitBarriersIteratively,
                                                         bool considerTaskExecutorType) {
    auto getTasksWithParallelWaitBarriers = [&]() {
        BarrierInfo::TaskSet tasksWithParallelWaitBarriers;
        for (size_t taskInd = 0; taskInd < getNumOfTasks(); taskInd++) {
            if (getWaitBarriers(taskInd).size() >= 2) {
                tasksWithParallelWaitBarriers.insert(taskInd);
            }
        }
        return tasksWithParallelWaitBarriers;
    };

    bool modifiedIR = false;
    auto tasksWithParallelWaitBarriers = getTasksWithParallelWaitBarriers();
    size_t iteration = 0;
    while (tasksWithParallelWaitBarriers.size() > 0) {
        _log.trace("Encountered {0} tasks with more than one wait barrier in iteration {1}",
                   tasksWithParallelWaitBarriers.size(), iteration);

        // Reverse "for" loop over tasks would reduce the required number of iterations, but
        // the schedule may not be exactly equivalent (E140023).
        _func->walk([&](VPURT::TaskOp taskOp) {
            const auto taskInd = getIndex(taskOp);
            if (getWaitBarriers(taskInd).size() < 2) {
                // valid configuration
                return;
            }

            modifiedIR |= eliminateParallelWaitBarriers(taskInd, availableSlots, considerTaskExecutorType);
        });

        auto tasksWithParallelWaitBarriersPrevIt = tasksWithParallelWaitBarriers;
        tasksWithParallelWaitBarriers = getTasksWithParallelWaitBarriers();
        VPUX_THROW_UNLESS(tasksWithParallelWaitBarriers != tasksWithParallelWaitBarriersPrevIt,
                          "Merging parallel wait barriers encountered an identical batch of tasks as in previous "
                          "iteration. Cannot enforce single wait barrier per task.");
        iteration++;

        if (!mergeWaitBarriersIteratively) {
            break;
        }
    }

    return modifiedIR;
}

bool vpux::BarrierInfo::inRange(const size_t low, const size_t high, const size_t val) const {
    return val >= low && val <= high;
}

void vpux::BarrierInfo::setBarrierMask(llvm::BitVector& mask, const BarrierInfo::TaskSet& barriers, size_t offset) {
    for (auto bar : barriers) {
        mask.set(static_cast<unsigned>(bar - offset));
    }
}

//
// buildTaskQueueTypeMap
//
void vpux::BarrierInfo::initializeTaskQueueTypeMap(const mlir::DenseSet<vpux::VPU::ExecutorKind>& executorKind) {
    for (auto execKind : executorKind) {
        const auto module = _func->getParentOfType<mlir::ModuleOp>();
        VPURT::TaskQueueType taskQueueType;
        taskQueueType.type = execKind;

        switch (execKind) {
        case VPU::ExecutorKind::DMA_NN: {
            const auto dmaPortNum = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
            auto dmaChannels = getDMAChannelsWithIndependentLinkAgents(VPU::getArch(module));
            for (auto dmaPortIdx : irange(dmaPortNum)) {
                for (auto dmaChannel : dmaChannels) {
                    taskQueueType.id = getDMAQueueIdEncoding(dmaPortIdx, dmaChannel);
                    llvm::BitVector taskList(checked_cast<uint32_t>(getNumOfTasks()));
                    _taskQueueTypeMap.insert(std::make_pair(taskQueueType, taskList));
                }
            }
            break;
        }

        case VPU::ExecutorKind::DPU:
        case VPU::ExecutorKind::SHAVE_ACT: {
            auto numClusters = IE::getTileExecutor(module).getCount();
            for (auto clusterId : irange(numClusters)) {
                taskQueueType.id = clusterId;
                llvm::BitVector taskList(checked_cast<uint32_t>(getNumOfTasks()));
                _taskQueueTypeMap.insert(std::make_pair(taskQueueType, taskList));
            }
            break;
        }

        default: {
            taskQueueType.id = 0;
            llvm::BitVector taskList(checked_cast<uint32_t>(getNumOfTasks()));
            _taskQueueTypeMap.insert(std::make_pair(taskQueueType, taskList));
        }
        }
    }
}

void vpux::BarrierInfo::clearTaskQueueTypeMap() {
    _taskQueueTypeMap.clear();
}

void vpux::BarrierInfo::buildTaskQueueTypeMap() {
    if (_taskQueueTypeMap.empty()) {
        initializeTaskQueueTypeMap({VPU::ExecutorKind::DMA_NN});
    }

    for (const auto& taskOp : _allTaskOps | reversed) {
        auto taskInd = getIndex(taskOp);
        auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
        if (_taskQueueTypeMap.find(taskQueueType) != _taskQueueTypeMap.end()) {
            _taskQueueTypeMap[taskQueueType].set(taskInd);
        }
    }
}

//
// buildTaskControlMap
//

std::pair<SmallVector<llvm::BitVector>, size_t> vpux::BarrierInfo::buildTaskControlMap(
        size_t blockIdx, bool considerTaskFifoDependency, bool ignoreOutOfBlockDependencies) {
    SmallVector<llvm::BitVector> taskControlMap;
    VPUX_THROW_WHEN(blockIdx >= getControlGraphBlockCount(), "Invalid task block index ({0})", blockIdx);

    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);

    VPUX_THROW_UNLESS(blockStartInd <= blockEndInd, "Invalid range of tasks for building task control map [{0}, {1}]",
                      blockStartInd, blockEndInd);

    _log.trace("Build task control map for task range [{0}, {1}]", blockStartInd, blockEndInd);
    auto newTaskControlMapSize = blockEndInd - blockStartInd + 1;
    if (taskControlMap.size() != newTaskControlMapSize) {
        resizeBitMap(taskControlMap, newTaskControlMapSize, checked_cast<uint32_t>(newTaskControlMapSize));
    }
    resetBitMap(taskControlMap);

    unsigned newDeps = 0;
    if (considerTaskFifoDependency) {
        // For every initialized queue in _taskQueueTypeMap, create temporary barriers between tasks on the same FIFO in
        // order to build taskControlMap with information about tasks-to-task dependencies stored exclusively on FIFO.
        // The these temporary barriers are removed afterwards.
        newDeps = createBarrierDependenciesImpliedByFIFO(blockIdx);
    }

    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto consumerIdx : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, consumerIdx)) {
                    taskControlMap[taskInd - blockStartInd].set(static_cast<unsigned>(consumerIdx - blockStartInd));
                }
            }
        }
    }

    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        if (taskInd == blockEndInd && isSyncPoint(taskInd)) {
            continue;
        }
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, childTaskInd)) {
                    taskControlMap[taskInd - blockStartInd] |=
                            taskControlMap[static_cast<size_t>(childTaskInd - blockStartInd)];
                } else {
                    VPUX_THROW_UNLESS(ignoreOutOfBlockDependencies,
                                      "Task {0} has update barriers with a consumer ({1}) from outside of the current "
                                      "range [{2}, {3}].",
                                      taskInd, childTaskInd, blockStartInd, blockEndInd);
                }
            }
        }
    }

    if (considerTaskFifoDependency) {
        auto removedDeps = removeBarrierDependenciesImpliedByFIFO();
        VPUX_THROW_WHEN(newDeps != removedDeps, "Temporary barriers have not been correctly removed");
    }

    return std::make_pair(taskControlMap, blockStartInd);
}

//
// controlPathExistsBetweenTasksInSameBlock
//

bool vpux::BarrierInfo::controlPathExistsBetweenTasksInSameBlock(const SmallVector<llvm::BitVector>& taskControlMap,
                                                                 size_t taskAInd, size_t taskBInd,
                                                                 bool biDirection) const {
    // ensure that taskControlMap is build at given time with buildTaskControlMap() for the correct task block
    VPUX_THROW_WHEN(taskControlMap.empty(), "Task control map not initialized");

    VPUX_THROW_WHEN(taskAInd >= taskControlMap.size(), "taskAInd out of range {0}", taskAInd);
    VPUX_THROW_WHEN(taskBInd >= taskControlMap.size(), "taskBInd out of range {0}", taskBInd);

    if (taskAInd == taskBInd) {
        return true;
    }

    if (biDirection) {
        return taskControlMap[taskAInd][taskBInd] || taskControlMap[taskBInd][taskAInd];
    }
    return taskControlMap[taskAInd][taskBInd];
}

//
// updateIR
//

void vpux::BarrierInfo::updateIR() {
    // update IR using stored dependencies
    for (size_t taskInd = 0; taskInd < getNumOfTasks(); ++taskInd) {
        auto taskOp = getTaskOpAtIndex(taskInd);

        taskOp.getWaitBarriersMutable().clear();
        for (auto barrierInd : _taskWaitBarriers[taskInd]) {
            auto barrierOp = getBarrierOpAtIndex(static_cast<size_t>(barrierInd));
            taskOp.getWaitBarriersMutable().append(barrierOp.getBarrier());
        }

        taskOp.getUpdateBarriersMutable().clear();
        for (auto barrierInd : _taskUpdateBarriers[taskInd]) {
            auto barrierOp = getBarrierOpAtIndex(static_cast<size_t>(barrierInd));
            taskOp.getUpdateBarriersMutable().append(barrierOp.getBarrier());
        }
    }
}

//
// logBarrierInfo
//

void vpux::BarrierInfo::logBarrierInfo() {
    // useful for logging and debugging barrier dependencies
    _log.setName("barrier-info-log");
    _log.trace("Logging BarrierInfo");

    for (size_t taskInd = 0; taskInd < getNumOfTasks(); ++taskInd) {
        _log.nest().trace("Task '{0}'", taskInd);
        for (const auto& barrierOp : getWaitBarriers(taskInd)) {
            _log.nest(2).trace("Task '{0}' waits for '{1}'", taskInd, barrierOp);
        }
        for (const auto& barrierOp : getUpdateBarriers(taskInd)) {
            _log.nest(2).trace("Task '{0}' updates '{1}'", taskInd, barrierOp);
        }
    }
}

/**
 * hasBarrierDependency - Checks for common barriers between two tasks
 *
 *        0
 *        |
 *       bar0
 *        |
 *        1
 *        |
 *       bar1
 *        |
 *        2
 *        |
 *       bar2
 *
 * For the above example for Task 0 and Task 1 the function returns true & bar0
 * Similarly for Task 0 and Task 2 it returns false as there is no direct barrier dependency
 */

bool vpux::BarrierInfo::hasBarrierDependency(size_t taskOneIdx, size_t taskTwoIdx, size_t& commonBarrier) {
    const auto& taskOneUpdateBarriers = getUpdateBarriers(taskOneIdx);
    const auto& taskTwoWaitBarriers = getWaitBarriers(taskTwoIdx);

    auto hasCommonBarrier = [&commonBarrier](const TaskSet& setOne, const TaskSet& setTwo) {
        for (const auto& barrier : setOne) {
            if (setTwo.count(barrier)) {
                commonBarrier = barrier;
                return true;
            }
        }
        return false;
    };

    return hasCommonBarrier(taskOneUpdateBarriers, taskTwoWaitBarriers);
}

//
// createLegalVariantBatches
//

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfo::createLegalVariantBatches(const TaskSet& tasks,
                                                                               size_t availableSlots,
                                                                               bool considerTaskExecutorType) {
    // store batches of tasks
    SmallVector<TaskSet> legalBatches(1);

    // store total slot count used by batch
    size_t totalSlotCount = 0;

    const auto isLegalVariantCountWith = [&](size_t numSlotsUsedByTask) -> bool {
        return (totalSlotCount + numSlotsUsedByTask) <= availableSlots;
    };

    // create batches for new barriers
    auto orderedTasks = std::set<size_t>(tasks.begin(), tasks.end());
    auto curBatchExecutorKind = getTaskOpAtIndex(*orderedTasks.begin()).getExecutorKind();
    const auto isMatchingExecutorKind = [&](VPU::ExecutorKind taskExecutorKind) {
        if (!considerTaskExecutorType) {
            return true;
        }
        return taskExecutorKind == curBatchExecutorKind;
    };

    for (const auto& taskInd : orderedTasks) {
        // find number of slots consumed by this task
        auto numSlotsUsedByTask = getNumOfSlotsUsed(getTaskOpAtIndex(taskInd));
        auto curTaskExecKind = getTaskOpAtIndex(taskInd).getExecutorKind();

        // check if new batch needs to be created
        if (!isLegalVariantCountWith(numSlotsUsedByTask) || !isMatchingExecutorKind(curTaskExecKind)) {
            legalBatches.push_back({});
            totalSlotCount = 0;
            curBatchExecutorKind = curTaskExecKind;
            // TODO: possible improvement - merge batches of same executor type if the slot count permits (E140023)
        }

        legalBatches.rbegin()->insert(taskInd);
        totalSlotCount += numSlotsUsedByTask;
    }

    return legalBatches;
}

//
// haveSameImplicitDependencyTaskQueueType
//

std::optional<VPURT::TaskQueueType> vpux::BarrierInfo::haveSameImplicitDependencyTaskQueueType(
        const TaskSet& taskInds) {
    if (taskInds.empty()) {
        return std::nullopt;
    }
    auto firstTaskInd = *taskInds.begin();
    // ensure that _taskQueueTypeMap is build at given time with buildTaskQueueTypeMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");
    for (const auto& item : _taskQueueTypeMap) {
        // get the task queue type for the first task, then check all the other tasks have same task queue type
        if (!item.second.test(firstTaskInd)) {
            continue;
        }

        for (const auto& taskInd : taskInds) {
            if (!item.second.test(static_cast<unsigned>(taskInd)) || isSyncPoint(taskInd)) {
                return std::nullopt;
            }
        }
        return item.first;
    }
    return std::nullopt;
}

//
// canBarriersBeMerged
//

bool vpux::BarrierInfo::canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                                            const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                                            ArrayRef<TaskSet> origWaitBarriersMap) {
    // two barriers A and B can be merged if
    // 1. any producer of barrier A controls any consumer of barrier B
    if (!producersControlsAllConsumers(barrierProducersA, barrierConsumersB, barrierConsumersA, origWaitBarriersMap)) {
        return false;
    }

    // 2. any producer of barrier B controls any consumer of barrier A
    if (!producersControlsAllConsumers(barrierProducersB, barrierConsumersA, barrierConsumersB, origWaitBarriersMap)) {
        return false;
    }

    return true;
}

//
// getWaitBarriersMap
//

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfo::getWaitBarriersMap() {
    return _taskWaitBarriers;
}

BarrierInfoTest::BarrierInfoTest(mlir::func::FuncOp func): BarrierInfo(func) {
}

BarrierInfoTest::BarrierInfoTest(BarrierInfoMaps& barrierMaps) {
    initializeBarrierMaps(barrierMaps);
}

void BarrierInfoTest::initializeBarrierMaps(BarrierInfoMaps& barrierMaps) {
    BarrierInfo::_barrierProducerMap = toTaskSet(barrierMaps.barrierProducerMap);
    BarrierInfo::_barrierConsumerMap = toTaskSet(barrierMaps.barrierConsumerMap);
    BarrierInfo::_taskUpdateBarriers = toTaskSet(barrierMaps.taskUpdateBarriers);
    BarrierInfo::_taskWaitBarriers = toTaskSet(barrierMaps.taskWaitBarriers);
    BarrierInfo::_allBarrierOps.resize(barrierMaps.barrierProducerMap.size());
    BarrierInfo::_allTaskOps.resize(barrierMaps.taskUpdateBarriers.size());
    BarrierInfo::_syncTasksIds = barrierMaps.syncTasksIds;
    if (!BarrierInfo::_syncTasksIds.empty()) {
        buildTaskBlockMap();
    }
    setTaskQueueTypeMap(barrierMaps.taskQueueTypeMap);
}

void vpux::BarrierInfoTest::setTaskQueueTypeMap(
        const std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps) {
    llvm::BitVector taskList(checked_cast<uint32_t>(getNumOfTasks()), false);
    for (const auto& [taskQueueType, taskIndVec] : taskQueueMaps) {
        _taskQueueTypeMap.insert(std::make_pair(taskQueueType, taskList));
        for (const auto taskInd : taskIndVec) {
            VPUX_THROW_UNLESS(taskInd < getNumOfTasks(), "Invalid task index '{0}', limit is '{1}'", taskInd,
                              getNumOfTasks());
            _taskQueueTypeMap[taskQueueType].set(taskInd);
        }
    }
}

void vpux::BarrierInfoTest::setMaxVariantCountPerBarrier(size_t variantCount) {
    _maxVariantCountPerBarrier = variantCount;
}

size_t vpux::BarrierInfoTest::getNumOfTasks() const {
    return std::max(_allTaskOps.size(),
                    _taskUpdateBarriers.size());  // BarrierInfoTest class may operate without input funcOp
}

size_t vpux::BarrierInfoTest::getBarrierMaxVariantSum() const {
    return _maxVariantCountPerBarrier;
}

size_t vpux::BarrierInfoTest::getNumOfSlotsUsedByTask([[maybe_unused]] VPURT::TaskOp op) const {
    return 1;  // for tests, assume single slot per task (as for eg. VPU::ExecutorKind::DMA_NN)
}

VPURT::TaskOp vpux::BarrierInfoTest::getTaskOpAtIndex([[maybe_unused]] size_t opIdx) const {
    if (_func != nullptr) {
        return BarrierInfo::getTaskOpAtIndex(opIdx);
    }
    return nullptr;
}

BarrierInfoMaps vpux::BarrierInfoTest::getBarrierMaps() {
    BarrierInfoMaps barrierMaps;
    barrierMaps.taskWaitBarriers = toTaskVec(BarrierInfo::_taskWaitBarriers);
    barrierMaps.taskUpdateBarriers = toTaskVec(BarrierInfo::_taskUpdateBarriers);
    barrierMaps.barrierConsumerMap = toTaskVec(BarrierInfo::_barrierConsumerMap);
    barrierMaps.barrierProducerMap = toTaskVec(BarrierInfo::_barrierProducerMap);
    return barrierMaps;
}

BarrierInfoMaps vpux::BarrierInfoTest::optimizeBarrierProducers(size_t blockIdx) {
    BarrierInfo::optimizeBarrierProducers(blockIdx);
    return getBarrierMaps();
}

BarrierInfoMaps vpux::BarrierInfoTest::optimizeBarrierConsumers(size_t blockIdx) {
    BarrierInfo::optimizeBarrierConsumers(blockIdx);
    return getBarrierMaps();
}

BarrierInfoMaps vpux::BarrierInfoTest::optimizeBarriersWithSameProducers(size_t blockIdx, bool checkValidSlotCount) {
    BarrierInfo::optimizeBarriersWithSameProducers(blockIdx, checkValidSlotCount);
    return getBarrierMaps();
}

BarrierInfoMaps vpux::BarrierInfoTest::optimizeBarriers(bool checkValidSlotCount, bool considerTaskFifoDependency) {
    BarrierInfo::optimizeBarriers(checkValidSlotCount, considerTaskFifoDependency);
    return getBarrierMaps();
}

BarrierInfoMaps vpux::getBarrierMaps(BarrierInfo& barrierInfo) {
    BarrierInfoMaps barrierMaps;
    barrierMaps.syncTasksIds = barrierInfo._syncTasksIds;
    barrierMaps.taskWaitBarriers = toTaskVec(barrierInfo._taskWaitBarriers);
    barrierMaps.taskUpdateBarriers = toTaskVec(barrierInfo._taskUpdateBarriers);
    barrierMaps.barrierConsumerMap = toTaskVec(barrierInfo._barrierConsumerMap);
    barrierMaps.barrierProducerMap = toTaskVec(barrierInfo._barrierProducerMap);
    return barrierMaps;
}

SmallVector<BarrierInfo::TaskSet> vpux::toTaskSet(const SmallVector<SmallVector<size_t>>& map) {
    SmallVector<BarrierInfo::TaskSet> convertedMap(map.size());
    for (auto deps : map | indexed) {
        for (auto dep : deps.value()) {
            convertedMap[deps.index()].insert(dep);
        }
    }
    return convertedMap;
}

BarrierMap vpux::toTaskVec(const SmallVector<BarrierInfo::TaskSet>& map) {
    BarrierMap convertedMap(map.size());
    for (auto deps : map | indexed) {
        if (deps.value().size() == 0) {
            continue;
        }
        convertedMap[deps.index()].reserve(deps.value().size());
        for (auto dep : deps.value()) {
            convertedMap[deps.index()].push_back(dep);
        }
        llvm::sort(convertedMap[deps.index()]);
    }
    return convertedMap;
}

/**
 * Calculate barrier producers/consumers map from task update/wait barriers map
 */
void vpux::fillProducersAndConsumers(BarrierInfoMaps& barrierMaps) {
    barrierMaps.barrierProducerMap.clear();
    barrierMaps.barrierConsumerMap.clear();

    // calculate barrier maximal ID
    llvm::DenseSet<size_t> barriers;
    for (auto taskBarriers : barrierMaps.taskUpdateBarriers) {
        barriers.insert(taskBarriers.begin(), taskBarriers.end());
    }
    for (auto taskBarriers : barrierMaps.taskWaitBarriers) {
        barriers.insert(taskBarriers.begin(), taskBarriers.end());
    }
    if (barriers.empty()) {
        return;
    }
    size_t maxBarrierID = *std::max_element(barriers.begin(), barriers.end());

    auto fillBarrierMap = [&](const BarrierMap& barriersPerTaskMap, BarrierMap& tasksPerBarrierMap) {
        // clear producers / consumers
        tasksPerBarrierMap.resize(maxBarrierID + 1);
        for (auto& tasks : tasksPerBarrierMap) {
            tasks.clear();
        }

        // fill-in tasks for each barrier
        for (auto taskBarriers : barriersPerTaskMap | indexed) {
            for (auto barrier : taskBarriers.value()) {
                tasksPerBarrierMap[barrier].push_back(taskBarriers.index());
            }
        }

        // sort maps
        for (auto& tasks : tasksPerBarrierMap) {
            llvm::sort(tasks);
        }
    };

    fillBarrierMap(barrierMaps.taskUpdateBarriers, barrierMaps.barrierProducerMap);
    fillBarrierMap(barrierMaps.taskWaitBarriers, barrierMaps.barrierConsumerMap);
}
