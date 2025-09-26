//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include "vpux/compiler/utils/dma.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

using namespace vpux;

//
// Virtual ID manipulation
//

namespace {

constexpr StringLiteral virtualIdAttrName = "VPURT.virtualId";

void assignVirtualIds(mlir::Operation* parentOp) {
    int64_t vid = 0;

    parentOp->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::ConfigureBarrierOp>(op)) {
            VPUX_THROW_WHEN(vid >= std::numeric_limits<uint32_t>::max(), "Barrier virtual id '{0}' is too large", vid);

            op->setAttr(virtualIdAttrName, getIntAttr(op->getContext(), vid++));
        }
    });
}

void cleanUpVirtualIds(mlir::Operation* parentOp) {
    parentOp->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::ConfigureBarrierOp>(op)) {
            op->removeAttr(virtualIdAttrName);
        }
    });
}

int64_t getVirtualId(mlir::Operation* op) {
    const auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op->getAttr(virtualIdAttrName));
    VPUX_THROW_WHEN(attr == nullptr, "The barrier operation at '{0}' doesn't have attribute '{1}'", op->getLoc(),
                    virtualIdAttrName);

    return attr.getValue().getSExtValue();
}

}  // namespace

//
// VirtualDependencyTracker
//

int64_t vpux::VPURT::VirtualDependencyTracker::add(VPURT::TaskOp taskOp) {
    auto extract = [&](Range& range, mlir::ValueRange barriers) {
        range.first = _ids.size();

        for (const auto bar : barriers) {
            const auto vid = getVirtualId(bar.getDefiningOp());
            _ids.push_back(vid);
            ++range.second;
        }
    };

    Dependency d;
    extract(d.consumer, taskOp.getWaitBarriers());
    extract(d.producer, taskOp.getUpdateBarriers());

    if (d.consumer.second != 0 || d.producer.second != 0) {
        _deps.push_back(d);
        return checked_cast<int64_t>(_deps.size() - 1);
    }

    return 0;
}

int64_t vpux::VPURT::VirtualDependencyTracker::add(ArrayRef<int64_t> waits, ArrayRef<int64_t> posts) {
    auto extract = [&](Range& range, ArrayRef<int64_t> barriers) {
        range.first = _ids.size();
        range.second = barriers.size();
        _ids.insert(_ids.end(), barriers.begin(), barriers.end());
    };

    if (!waits.empty() || !posts.empty()) {
        Dependency d;
        extract(d.consumer, waits);
        extract(d.producer, posts);

        _deps.push_back(d);
        return checked_cast<int64_t>(_deps.size() - 1);
    }

    return 0;
}

int64_t vpux::VPURT::VirtualDependencyTracker::clone(int64_t i) {
    _deps.push_back(dep(i));
    return checked_cast<int64_t>(_deps.size() - 1);
}

void vpux::VPURT::VirtualDependencyTracker::print(Logger log) const {
    log.trace("{0} dependencies:", _deps.size());

    for (const auto& p : _deps | indexed) {
        const auto& d = p.value();
        const auto ind = p.index();

        log.nest().trace("Dependency[{0}]: consumer: {1} - {2}, producer: {3} - {4}", ind, d.consumer.first,
                         d.consumer.second, d.producer.first, d.producer.second);
    }
}

//
// BarrierWlmHandler
//

vpux::VPURT::BarrierWlmHandler::BarrierWlmHandler(BarrierInfo& barrierInfo): _barrierInfo(barrierInfo) {
    _barrierInfo.buildTaskQueueTypeMap();
    _blockIdxOfTaskControlMap = 0;
    _taskControlMapAndOffset = _barrierInfo.buildTaskControlMap(_blockIdxOfTaskControlMap);
}

vpux::VPURT::BarrierWlmHandler::BarrierWlmHandler(BarrierInfoTest& barrierInfoTest): _barrierInfo(barrierInfoTest) {
    // Task queue type is provided as part of barrierInfoTest
    _blockIdxOfTaskControlMap = 0;
    _taskControlMapAndOffset = _barrierInfo.buildTaskControlMap(_blockIdxOfTaskControlMap);
}

// Function which checks for barrier PID reuse in case _wlmPidProgramming is set (WLM)
// PID can be reused only if there is a dependency from prevVid to current vid
// and prevVid is guaranteed to be fully consumed before schedule is at barrier with given vid
bool vpux::VPURT::BarrierWlmHandler::canBarrierReusePidFromBarrier(size_t vid, size_t prevVid) {
    auto prevBarConsumers = _barrierInfo.getBarrierConsumers(prevVid);
    auto barProducers = _barrierInfo.getBarrierProducers(vid);

    // Check if there is a path from all prevBarConsumers to all task from barProducers
    //           |-> prevBarConsumer0 ->  ..  -> barProducer0 -> |
    //  prevBar->|-> prevBarConsumer1 ->  ..  -> barProducer1 -> | -> bar
    //                                        -> barProducer2 -> |
    for (auto& prevBarConsumer : prevBarConsumers) {
        bool isPath = false;
        auto prevBarConsumerBlock = _barrierInfo.getControlGraphBlockIndex(prevBarConsumer);
        for (auto& barProducer : barProducers) {
            auto barProducerBlock = _barrierInfo.getControlGraphBlockIndex(barProducer);
            if (barProducerBlock != prevBarConsumerBlock) {
                // If tasks are from different blocks then they are guaranteed to have a path
                // through a sync task
                isPath = true;
            } else {
                // Control map is stored only for single control graph block to save memory space
                // If processing moves to next block index rebuild control map for it
                // TODO: Check scenario when task control map is rebuild back to block index for which it contained
                // data in some previous iteration
                if (barProducerBlock != _blockIdxOfTaskControlMap) {
                    _blockIdxOfTaskControlMap = barProducerBlock;
                    _taskControlMapAndOffset = _barrierInfo.buildTaskControlMap(_blockIdxOfTaskControlMap);
                }

                auto& [taskControlMap, taskControlMapOffset] = _taskControlMapAndOffset;
                isPath = _barrierInfo.controlPathExistsBetweenTasksInSameBlock(
                        taskControlMap, prevBarConsumer - taskControlMapOffset, barProducer - taskControlMapOffset,
                        true);
            }
            if (!isPath) {
                return false;
            }
        }
    }

    return true;
}

//
// BarrierSimulator
//

vpux::VPURT::BarrierSimulator::BarrierSimulator(mlir::func::FuncOp funcOp) {
    _barrierProducerSlotCount = static_cast<int64_t>(VPUIP::getBarrierMaxVariantCount(funcOp));
    _barrierTotalSlotCount = static_cast<int64_t>(VPUIP::getBarrierMaxVariantSum(funcOp));
    _availableBarriers = VPUIP::getNumAvailableBarriers(funcOp);
    _dmaTasks.resize(VPUIP::getNumberOfIndependentDmaQueues(funcOp));
    assignVirtualIds(funcOp);
    parseBarriers(funcOp);
    parseTasks(funcOp);
    cleanUpVirtualIds(funcOp);
}

vpux::VPURT::BarrierSimulator::BarrierSimulator(mlir::func::FuncOp funcOp, bool wlmFlag, BarrierInfo& barrierInfo)
        : BarrierSimulator(funcOp) {
    if (wlmFlag) {
        configureForWlm(barrierInfo);
    }
    _barrierInfo = barrierInfo;
}

void vpux::VPURT::BarrierSimulator::configureForWlm(vpux::BarrierInfo& barrierInfo) {
    _barrierWlmHandlerOpt = BarrierWlmHandler(barrierInfo);
    _wlmPidProgramming = true;
}

const VPURT::BarrierConfig& vpux::VPURT::BarrierSimulator::getConfig(mlir::Value bar) const {
    const auto it = _barriersMap.find(bar.getDefiningOp());
    VPUX_THROW_WHEN(it == _barriersMap.end(), "Barrier at '{0}' was not covered by BarrierSimulator", bar.getLoc());
    return _barriers[it->second];
}

void vpux::VPURT::BarrierSimulator::parseBarriers(mlir::Operation* parentOp) {
    _isDynamicBarriers = false;
    _wlmPageApproach = false;

    parentOp->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        _isDynamicBarriers = true;

        const auto vid = getVirtualId(barrierOp);
        VPUX_THROW_UNLESS(vid == checked_cast<int64_t>(_barriers.size()),
                          "Got wrong virtual ID for the barrier at '{0}'", barrierOp->getLoc());

        _barriersMap.insert({barrierOp, _barriers.size()});
        _barrierVID.insert({_barriers.size(), barrierOp});
        _barriers.emplace_back(barrierOp->getLoc());

        // Check if barrier page (used for WLM subgraph approach) is configured
        // on all barriers. Store page data to use it later when performing PID assignment
        auto pageOpt = barrierOp.getWlmPage();
        if (pageOpt.has_value()) {
            if (vid == 0) {
                _wlmPageApproach = true;
            }
            _pageToBarVidVecMap[pageOpt.value()].push_back(vid);
        }
        VPUX_THROW_UNLESS(_wlmPageApproach == pageOpt.has_value(),
                          "Misalignment on WLM page approach setting, flag - {0}, barOp - {1}", _wlmPageApproach,
                          barrierOp);
    });

    parentOp->walk([&](VPURT::ConfigureBarrierOp barrierOp) {
        VPUX_THROW_WHEN(_isDynamicBarriers, "Can't have both dynamic and static barriers at the same time");

        const auto vid = getVirtualId(barrierOp);
        VPUX_THROW_UNLESS(vid == checked_cast<int64_t>(_barriers.size()),
                          "Got wrong virtual ID for the barrier at '{0}'", barrierOp->getLoc());

        _barriersMap.insert({barrierOp, _barriers.size()});
        _barriers.emplace_back(barrierOp->getLoc(), barrierOp.getId());

        _usedBarriers = std::max(_usedBarriers, barrierOp.getId() + 1);
    });
}

void vpux::VPURT::BarrierSimulator::parseTasks(mlir::Operation* parentOp) {
    const auto updateBarrierConfigs = [&](VPURT::TaskOp taskOp, const int64_t count = 1) {
        for (const auto bar : taskOp.getWaitBarriers()) {
            const auto v = getVirtualId(bar.getDefiningOp());
            _barriers[v].consumerCount += count;
        }

        for (const auto bar : taskOp.getUpdateBarriers()) {
            const auto v = getVirtualId(bar.getDefiningOp());
            _barriers[v].producerCount += count;
        }
    };

    // Maintain internally information about already encountered DMA FIFO queues IDs
    // which are composed of DMA port and channel. Depending on platform and settings
    // there might be different independent DMA FIFOs to track as part of barrier handling
    std::unordered_map<int64_t, int64_t> dmaQueueIdToIndexMap;

    parentOp->walk([&](VPURT::TaskOp taskOp) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp();

        const auto virtualDep = _vdt.add(taskOp);

        switch (taskOp.getExecutorKind()) {
        case VPU::ExecutorKind::DMA_NN: {
            const auto ports = VPURT::getDMATaskPorts(taskOp);
            auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(wrappedTaskOp);
            VPUX_THROW_WHEN(dmaTask == nullptr, "Not a DMA task");

            SmallVector<DmaTaskIdx> dmaTaskIndexs;
            for (auto& port : ports) {
                // Find index of DMA FIFO that is analyzed by barrier simulator
                // If this DMA task corresponds to a FIFO that was not already encountered
                // then increase the size of tracked DMA FIFOs
                int64_t dmaTasksQueueIndex;
                auto dmaQueueId = getDMAQueueIdEncoding(port, dmaTask.getChannelType());

                auto dmaQueueIt = dmaQueueIdToIndexMap.find(dmaQueueId);
                if (dmaQueueIt == dmaQueueIdToIndexMap.end()) {
                    dmaTasksQueueIndex = dmaQueueIdToIndexMap.size();
                    dmaQueueIdToIndexMap[dmaQueueId] = dmaTasksQueueIndex;
                } else {
                    dmaTasksQueueIndex = dmaQueueIt->second;
                }

                VPUX_THROW_UNLESS(dmaTasksQueueIndex < static_cast<int64_t>(_dmaTasks.size()),
                                  "NNDMAOp queue index '{0}' larger than maximum number of queues '{1}'",
                                  dmaTasksQueueIndex, _dmaTasks.size());
                auto dmaTaskIdx = std::make_pair(port, _dmaTasks[dmaTasksQueueIndex].size());
                dmaTaskIndexs.push_back(dmaTaskIdx);
                _dmaTasks[dmaTasksQueueIndex].emplace_back(virtualDep);
            }
            updateBarrierConfigs(taskOp);
            if (dmaTaskIndexs.size() > 1) {
                _multiQueueDmaTaskStatus.insert({dmaTaskIndexs, false});
            }
            break;
        }

        case VPU::ExecutorKind::DPU: {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");

            int64_t numVariants = BarrierInfo::getNumOfSlotsUsed(taskOp);
            _nceTasks.emplace_back(virtualDep, numVariants);
            updateBarrierConfigs(taskOp, numVariants);
            break;
        }

        case VPU::ExecutorKind::SHAVE_ACT: {
            _actTasks.emplace_back(virtualDep);
            updateBarrierConfigs(taskOp);
            break;
        }

        case VPU::ExecutorKind::M2I: {
            _m2iTasks.emplace_back(virtualDep);
            updateBarrierConfigs(taskOp);
            break;
        }

        // TODO: Analyze and define executor type for funcOp - E#117624
        // Using DMA is a temporary change
        case VPU::ExecutorKind::UNKNOWN: {
            _dmaTasks[0].emplace_back(virtualDep);
            updateBarrierConfigs(taskOp);
            break;
        }

        default:
            VPUX_THROW("Unsupported executor '{0}'", taskOp.getExecutorKind());
        }
    });
}

//
// The limitation is not related to HW capabilities or FIFO depth, but to the fact that the runtime needs to know when a
// workload is completed, in order to replace it with another one in NN CMX.
//
// Since there's no other efficient feedback mechanism from DPU/SNN to LNN, LNN monitors the barrier production of DPU
// tasks and recycles the storage when the corresponding barrier gets produced. The limitation comes from how many
// invariants/variants can be stored in NN CMX at the same time. For single cluster inferences these counts are 64/512,
// while for 4-cluster inferences 128/512. If too many invariants/variants contribute to the same barrier, the runtime
// will not receive the confirmation that it may recycle the storage to bring in the next workloads, hence the deadlock.
//
// Since the storage area is double buffered, and the workloads in question may start at any index in the buffer, it's
// only safe for at most <storage_size / 2 + 1> consecutive invariants/variants to produce the same barrier. So finally,
// the limits are:
//
// On single cluster:
//   32 + 1 invariants
//   256 + 1 variants
// On 4 clusters:
//   64 + 1 invariants
//   256 + 1 variants
//
mlir::LogicalResult vpux::VPURT::BarrierSimulator::checkProducerCount(Logger log) const {
    for (auto vid : irange(_barriers.size())) {
        const auto producerCount = _barriers[vid].producerCount;

        if (producerCount > _barrierProducerSlotCount) {
            log.warning("Barrier {0} at '{1}' has {2} producers (max {3})", vid, _barriers[vid].loc, producerCount,
                        _barrierProducerSlotCount);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPURT::BarrierSimulator::checkProducerAndConsumerCount(Logger log) const {
    for (auto vid : irange(_barriers.size())) {
        const auto producerCount = _barriers[vid].producerCount;
        const auto consumerCount = _barriers[vid].consumerCount;

        if (producerCount + consumerCount > _barrierTotalSlotCount) {
            log.warning("Barrier {0} at '{1}' has {2} producers and {3} consumers (max sum {4})", vid,
                        _barriers[vid].loc, producerCount, consumerCount, _barrierTotalSlotCount);
            return mlir::failure();
        }
    }

    return mlir::success();
}

SmallVector<mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>>
vpux::VPURT::BarrierSimulator::getBarrierBatchesToLegalize() {
    // convert id's to barriers
    SmallVector<mlir::DenseSet<DeclareVirtualBarrierOp>> batches;
    for (const auto& batch : _barrierBatchesToLegalize) {
        mlir::DenseSet<DeclareVirtualBarrierOp> barrierBatch;

        for (const auto& barrierVID : batch) {
            auto barrierOp = mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(_barrierVID[barrierVID]);
            VPUX_THROW_UNLESS(barrierOp != nullptr, "Got invalid barrier type {0}", _barrierVID[barrierVID]);
            if (barrierOp.getBarrier().use_empty()) {
                // barrier will be removed
                continue;
            }
            barrierBatch.insert(barrierOp);
        }

        if (barrierBatch.empty()) {
            // case when all barriers have no use
            continue;
        }
        batches.push_back(barrierBatch);
    }
    return batches;
}

mlir::LogicalResult vpux::VPURT::BarrierSimulator::simulateBarriers(Logger log, std::optional<int64_t> numBarriers,
                                                                    std::optional<bool> barrierLegalization) {
    if (numBarriers.has_value()) {
        VPUX_THROW_UNLESS(numBarriers.value() <= _availableBarriers,
                          "Provided number of barriers '{0}' is greater than HW limitation '{1}'", numBarriers,
                          _availableBarriers);
    }

    auto activeBarrierLimit = checked_cast<size_t>(numBarriers.value_or(_availableBarriers));
    SmallVector<int64_t> toVirtual(activeBarrierLimit, -1);
    SmallVector<int64_t> prevVidOfPhysBar(activeBarrierLimit, -1);

    bool barrierLegalizationSim = barrierLegalization.value_or(false);

    RingBuffer<int64_t> nextReal;
    if (_isDynamicBarriers) {
        nextReal.reset(toVirtual.size());

        while (!nextReal.full()) {
            nextReal.push(checked_cast<int64_t>(nextReal.size()));
        }
    }

    size_t bar = 0;
    SmallVector<size_t> dma(_dmaTasks.size(), 0);
    size_t dpu = 0;
    size_t act = 0;
    size_t m2i = 0;
    bool progressed = false;

    log.trace("Simulating barrier flow ({0}) with {1} barries", _isDynamicBarriers ? "assignment" : "validation",
              toVirtual.size());

    auto isDmaFifoNotComplete = [&]() {
        for (size_t i = 0; i < _dmaTasks.size(); i++) {
            if (dma[i] < _dmaTasks[i].size()) {
                return true;
            }
        }
        return false;
    };

    auto getDmaProgressLog = [&]() {
        std::string dmaLog = "";
        for (size_t i = 0; i < _dmaTasks.size(); i++) {
            dmaLog += " " + std::to_string(dma[i]) + " / " + std::to_string(_dmaTasks[i].size());
        }
        return dmaLog;
    };

    for (; bar < _barriers.size() || isDmaFifoNotComplete() || dpu < _nceTasks.size() || act < _actTasks.size() ||
           m2i < _m2iTasks.size();
         progressed = false) {
        log.nest(1).trace("DMA:{0}; DPU: {1} / {2}; ACT: {3} / {4}; M2I: {5} / "
                          "{6}; BAR: {7} / {8}",
                          getDmaProgressLog(), dpu, _nceTasks.size(), act, _actTasks.size(), m2i, _m2iTasks.size(), bar,
                          _barriers.size());

        // Static vs dynamic barriers need a different loop exit condition
        const auto hasBarriersToMap = [&]() {
            if (_isDynamicBarriers) {
                return !nextReal.empty();
            } else if (!_barrierPidToVidInstancesQueueMap.empty()) {
                // In case for simulation PID to VID mapping is provided check if any free barrier can be mapped
                // Check if any pid is free to map
                for (size_t pid = 0; pid < toVirtual.size(); pid++) {
                    if (toVirtual[pid] == -1 && !_barrierPidToVidInstancesQueueMap[pid].empty()) {
                        return true;
                    }
                }
                return false;
            } else {
                return toVirtual[_barriers[bar].realId] == -1;
            }
        };

        // Map new barriers
        log.nest(2).trace("Map new barriers");
        for (; bar < _barriers.size() && hasBarriersToMap(); ++bar, progressed = true) {
            int64_t real = -1;

            if (_isDynamicBarriers) {
                if (_wlmPidProgramming && _barrierWlmHandlerOpt.has_value()) {
                    // In this mode barrier simulator will iteratively try to pick free PID for given VID
                    // and check if there is a topological dependency with its previous usage - goal is to not
                    // use same PID on parallel branches and guarantee barrier is fully consumed before same
                    // PID is going to be used for next VID
                    auto numOfAvailablePhysBars = nextReal.size();
                    while (numOfAvailablePhysBars) {
                        real = nextReal.front();
                        nextReal.pop();
                        auto prevVid = prevVidOfPhysBar[real];
                        if (prevVid >= 0) {
                            // Check if there is dependency with previous usage of this pid which implies a linear
                            // dependency otherwise barriers may be alive in parallel If unsuccessful try to find next
                            // physial barrier
                            if (!_barrierWlmHandlerOpt.value().canBarrierReusePidFromBarrier(bar, prevVid)) {
                                // Push this PID back to the ring buffer as it cannot be used for this VID
                                nextReal.push(real);
                                // Decrement amount of barriers to still be checked
                                numOfAvailablePhysBars--;
                                continue;
                            }
                        }
                        break;
                    }

                    if (numOfAvailablePhysBars == 0) {
                        // If no PID was found for given VID exit the loop as no progress in barrier
                        // execution can be made as barrier could not be mapped
                        break;
                    }
                } else {
                    real = nextReal.front();
                    nextReal.pop();
                }

                _barriers[bar].realId = real;
            } else if (!_barrierPidToVidInstancesQueueMap.empty()) {
                // In case for simulation PID to VID mapping is provided pick next free barrier from this data
                for (size_t pid = 0; pid < toVirtual.size(); pid++) {
                    if (toVirtual[pid] == -1 && !_barrierPidToVidInstancesQueueMap[pid].empty()) {
                        real = pid;
                        break;
                    }
                }
            } else {
                real = _barriers[bar].realId;
            }

            VPUX_THROW_WHEN(real < 0, "No new barrier was mapped");

            auto vid = bar;

            if (!_barrierPidToVidInstancesQueueMap.empty()) {
                vid = _barrierPidToVidInstancesQueueMap[real].front();
                _barrierPidToVidInstancesQueueMap[real].pop();
                _barrierProgrammingOrder.push_back(vid);
            }

            toVirtual[checked_cast<size_t>(real)] = checked_cast<size_t>(vid);

            log.nest(3).trace("VID[{0}]: PID {1}, producers {2}, consumers {3}, prev VID {4}", vid, real,
                              _barriers[bar].producerCount, _barriers[bar].consumerCount,
                              prevVidOfPhysBar[checked_cast<size_t>(real)]);
            prevVidOfPhysBar[real] = checked_cast<size_t>(vid);
        }

        // condition to create new batch of barriers
        const auto isNewBatchNeeded = [&]() {
            if (_barrierBatchesToLegalize.empty()) {
                return true;
            }

            if (_barrierBatchesToLegalize.rbegin()->empty()) {
                // do not create new batch if last one is empty
                return false;
            }

            bool activeBarrierIsMapped = false;
            for (const auto& activeBar : toVirtual) {
                if (activeBar == -1) {
                    VPUX_THROW_WHEN(_wlmPidProgramming == false, "Found not mapped barrier with exceeding barriers");
                    // Skip barrier if was not mapped yet. This can happen if PID cannot be reused for given VID
                    // because of lack of topological dependency
                    continue;
                }
                if (_barrierBatchesToLegalize.rbegin()->find(activeBar) != _barrierBatchesToLegalize.rbegin()->end()) {
                    // check if last batch contains any current active barriers
                    return false;
                }
                activeBarrierIsMapped = true;
            }

            return activeBarrierIsMapped;
        };

        // store barriers to legalize in batches, count also free barriers (-1) as during
        // iteration barriers can be produced and consumed, but all need to be mapped
        if (barrierLegalizationSim && toVirtual.size() > activeBarrierLimit) {
            if (isNewBatchNeeded()) {
                // create new batch
                _barrierBatchesToLegalize.push_back({});
            }
            for (const auto& activeBar : toVirtual) {
                if (activeBar == -1) {
                    VPUX_THROW_WHEN(_wlmPidProgramming == false, "Found not mapped barrier with exceeding barriers");
                    // Skip barrier if was not mapped yet. This can happen if PID cannot be reused for given VID
                    // because of lack of topological dependency
                    continue;
                }
                _barrierBatchesToLegalize.rbegin()->insert(activeBar);
            }
        }

        // Process DMAs
        log.nest(2).trace("Process DMAs");
        for (size_t e = 0; e < _dmaTasks.size(); ++e) {
            for (; dma[e] < _dmaTasks[e].size(); ++dma[e], progressed = true) {
                auto& dt = _dmaTasks[e][dma[e]];

                auto isDMATaskInMultiQueues = [&]() {
                    return llvm::find_if(_multiQueueDmaTaskStatus, [&](const auto& item) {
                        const auto& multiTaskList = item.first;
                        DmaTaskIdx currentDmaTaskIdx{e, dma[e]};
                        return llvm::find(multiTaskList, currentDmaTaskIdx) != multiTaskList.end();
                    });
                };
                auto iter = isDMATaskInMultiQueues();
                // Check if dma task is processed in other task queue
                if (iter == _multiQueueDmaTaskStatus.end() || !(iter->second)) {
                    const auto status =
                            processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "DMA[" + std::to_string(e) + "]", dma[e],
                                       toVirtual, nextReal, log.nest(3));

                    if (status == Status::Fail) {
                        return mlir::failure();
                    }
                    if (status == Status::Skip) {
                        break;
                    }
                    log.nest(3).trace("DMA[{0}][{1}]: waits: {2}, posts: {3}", e, dma[e], dt.waits, dt.posts);
                    if (iter != _multiQueueDmaTaskStatus.end()) {
                        // mark task as processed
                        iter->second = true;
                    }
                } else if (iter != _multiQueueDmaTaskStatus.end()) {
                    log.trace("skip DMA[{0}][{1}] since it has been processed in other queue", e, dma[e]);
                }
            }
        }

        // Process DPUs
        log.nest(2).trace("Process DPUs");
        for (; dpu < _nceTasks.size(); ++dpu, progressed = true) {
            auto& dt = _nceTasks[dpu];

            const auto status =
                    processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "DPU", dpu, toVirtual, nextReal, log.nest(3));

            if (status == Status::Fail) {
                return mlir::failure();
            }
            if (status == Status::Skip) {
                break;
            }

            log.nest(3).trace("DPU[{0}]: waits: {1}, posts: {2}, variants: {3}", dpu, dt.waits, dt.posts, dt.count);
        }

        // Process Act Kernels
        log.nest(2).trace("Process Act Kernels");
        for (; act < _actTasks.size(); ++act, progressed = true) {
            auto& dt = _actTasks[act];

            const auto status =
                    processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "ACT", act, toVirtual, nextReal, log.nest(3));

            if (status == Status::Fail) {
                return mlir::failure();
            }
            if (status == Status::Skip) {
                break;
            }

            log.nest(3).trace("ACT[{0}]: waits: {1}, posts: {2}", act, dt.waits, dt.posts);
        }

        // Process M2I tasks
        log.nest(2).trace("Process M2I tasks");
        for (; m2i < _m2iTasks.size(); ++m2i, progressed = true) {
            auto& dt = _m2iTasks[m2i];

            const auto status =
                    processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "M2I", m2i, toVirtual, nextReal, log.nest(3));

            if (status == Status::Fail) {
                return mlir::failure();
            }
            if (status == Status::Skip) {
                break;
            }

            log.nest(3).trace("M2I[{0}]: waits: {1}, posts: {2}", m2i, dt.waits, dt.posts);
        }

        if (barrierLegalizationSim) {
            // update progress
            log.nest(1).trace("Barrier Legalization: progressed={0}", progressed);
            auto freeBarriers = checked_cast<size_t>(std::count(toVirtual.begin(), toVirtual.end(), -1));

            const auto updateNextReal = [&]() {
                nextReal.reset(toVirtual.size());
                for (const auto& p : toVirtual | indexed) {
                    if (p.value() == -1) {
                        // find all indices equal to -1
                        nextReal.push(checked_cast<int64_t>(p.index()));
                    }
                }
            };

            if (!progressed) {
                VPUX_THROW_WHEN(!_wlmPidProgramming && freeBarriers > 0,
                                "Simulation not progressed and there are '{0}' free barriers", freeBarriers);
                // increase real barrier size
                toVirtual.push_back(-1);
                prevVidOfPhysBar.push_back(-1);
                log.nest(3).trace("Increased real barrier size to {0} physical barriers", toVirtual.size());

                progressed = true;
                updateNextReal();
            } else if (freeBarriers > 0 && toVirtual.size() > activeBarrierLimit) {
                // decrease exceeding barriers when possible
                log.nest(2).trace("Reduce exceeding active barrier count, there are {0} free barriers", freeBarriers);

                size_t freeBarrierCount = 0;
                for (const auto& p : toVirtual | indexed) {
                    // remap barrier mapping
                    if (p.value() != -1) {
                        // not a free barrier, update _barriers[v].realId
                        log.nest(4).trace("Remap '{0}' from '{1}' -> '{2}'", p.value(), _barriers[p.value()].realId,
                                          p.index() - freeBarrierCount);
                        _barriers[p.value()].realId = checked_cast<int64_t>(p.index() - freeBarrierCount);
                        continue;
                    }

                    if (toVirtual.size() - freeBarrierCount > activeBarrierLimit) {
                        // remaining free barriers will stay
                        ++freeBarrierCount;
                    }
                }

                // remove free barriers exceeding active barrier limit
                std::set<int64_t, std::greater<int64_t>> indexesRemoved;
                auto indexItr = std::find(toVirtual.begin(), toVirtual.end(), -1);
                while (indexItr != toVirtual.end() && toVirtual.size() > activeBarrierLimit) {
                    indexesRemoved.insert(std::distance(toVirtual.begin(), indexItr));
                    toVirtual.erase(indexItr);
                    indexItr = std::find(toVirtual.begin(), toVirtual.end(), -1);
                }

                for (auto indexToRemove : indexesRemoved) {
                    VPUX_THROW_WHEN(static_cast<size_t>(indexToRemove) >= prevVidOfPhysBar.size(),
                                    "Incorrect index to remove - {0} >= {1}", indexToRemove, prevVidOfPhysBar.size());
                    prevVidOfPhysBar.erase(prevVidOfPhysBar.begin() + indexToRemove);
                }
                updateNextReal();
            }
        }

        if (!progressed) {
            log.debug("Barrier simulation blocked at DMA:{0}; DPU: {1} / {2}; ACT: {3} / {4}; "
                      "M2I: {5} / {6}; BAR: {7} / {8}",
                      getDmaProgressLog(), dpu, _nceTasks.size(), act, _actTasks.size(), m2i, _m2iTasks.size(), bar,
                      _barriers.size());

            for (size_t pid = 0; pid < toVirtual.size(); pid++) {
                auto vid = toVirtual[pid];
                if (vid < 0) {
                    log.debug("PID {0} not mapped", pid);
                } else {
                    log.debug("PID {0} VID {1}, producers: {2}, consumers: {3}", pid, vid, _barriers[vid].producerCount,
                              _barriers[vid].consumerCount);
                    VPUX_THROW_UNLESS(_barriers[vid].realId == static_cast<int64_t>(pid),
                                      "Mismatch in barrier config: VID {0}", vid);
                }
            }

            return mlir::failure();
        }
    }

    if (barrierLegalizationSim && !getBarrierBatchesToLegalize().empty()) {
        // fail only when batches contain barriers that are actually used by tasks
        log.trace("Barrier Legalization: barrier batch to legalize {0}", _barrierBatchesToLegalize.size());
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPURT::BarrierSimulator::simulateBarriersForWlmPageApproach(Logger log, int64_t numBarriers) {
    VPUX_THROW_UNLESS(_isDynamicBarriers,
                      "Unexpected dynamic barrier type for barrier simulation with WLM page approach");
    VPUX_THROW_UNLESS(_wlmPageApproach, "Barrier simulator not initialized for WLM page approach");

    // Since during simulation PIDs are not reassigned but provided in a separate data structure
    // turn dynamic barrier flag off
    _isDynamicBarriers = false;

    const auto pageSize = static_cast<size_t>(numBarriers / 2);

    for (size_t pageInd = 0; pageInd < _pageToBarVidVecMap.size(); pageInd++) {
        size_t pidOffset = 0;
        if (pageInd % 2 == 1) {
            pidOffset += pageSize;
        }

        auto barVidVec = _pageToBarVidVecMap[pageInd];
        VPUX_THROW_WHEN(barVidVec.size() > pageSize, "Number of barriers in page {0} not within limit of page size {1}",
                        barVidVec.size(), pageSize);

        for (size_t i = 0; i < barVidVec.size(); i++) {
            // For each page use following barriers assignment
            // Page0 PIDs : [0 - (numBarriers/2 - 1)]
            // Page1 PIDs : [(numBarriers/2) - (numBarriers - 1)]
            // Page2 PIDs : same as Page0
            // Page3 PIDs : same as Page1
            // ...
            auto pid = pidOffset + i;
            auto vid = barVidVec[i];
            _barriers[vid].realId = pid;
            _barrierPidToVidInstancesQueueMap[pid].push(vid);
        }
    }

    _barrierProgrammingOrder.reserve(_barriers.size());

    auto result = simulateBarriers(log, numBarriers);
    _isDynamicBarriers = true;

    return result;
}

// Simulate barriers with provided virtual to physical barrier configuration. Retrieve
// order barriers were reprogrammed. This will later be used to reorder IR
mlir::LogicalResult vpux::VPURT::BarrierSimulator::generateBarrierOrderWithSimulation(
        Logger log, int64_t numBarriers, SmallVector<size_t>& virtualToPhysicalBarrierMapping) {
    VPUX_THROW_UNLESS(_isDynamicBarriers,
                      "Unexpected dynamic barrier type for barrier simulation with manual barrier mapping configured");
    _isDynamicBarriers = false;
    size_t prevVid = 0;
    for (size_t vid = 0; vid < virtualToPhysicalBarrierMapping.size(); vid++) {
        auto pid = virtualToPhysicalBarrierMapping[vid];
        _barriers[vid].realId = pid;
        VPUX_THROW_UNLESS(vid >= prevVid, "Incorrect VID order");
        prevVid = vid;
        _barrierPidToVidInstancesQueueMap[pid].push(vid);
    }
    _barrierProgrammingOrder.reserve(virtualToPhysicalBarrierMapping.size());

    auto result = simulateBarriers(log, numBarriers);
    _isDynamicBarriers = true;

    return result;
}

// Method to update barrier order in IR based on how barriers were reprogrammed during simulation
// Method used if barrier configuration was provided to the simulator and not defined during its execution
void vpux::VPURT::BarrierSimulator::updateBarrierOrderInIr() {
    VPUX_THROW_WHEN(_barrierProgrammingOrder.empty(), "No barriers order vector available");

    VPURT::DeclareVirtualBarrierOp insertPoint = nullptr;
    VPURT::DeclareVirtualBarrierOp finalBarOp = nullptr;
    for (auto bar : _barrierProgrammingOrder) {
        auto barOp = mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(_barrierVID[bar]);
        if (insertPoint) {
            barOp->moveAfter(insertPoint);
        }
        insertPoint = barOp;

        if (barOp.getIsFinalBarrier()) {
            VPUX_THROW_UNLESS(finalBarOp == nullptr, "More then one final barrier: {0} and {1}", finalBarOp, barOp);
            finalBarOp = barOp;
        }
    }

    // Other passes expect final barrier to be at the end of IR barrier declarations block in the IR
    auto lastBarOp = mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(_barrierVID[_barrierProgrammingOrder.back()]);
    if (finalBarOp != nullptr && lastBarOp != finalBarOp) {
        finalBarOp->moveAfter(lastBarOp);
    }
}

VPURT::BarrierSimulator::Status vpux::VPURT::BarrierSimulator::processSim(
        const VirtualDependencyTracker::Dependency& dep, BarrierUserConfig& user, int64_t count, StringRef taskType,
        int64_t index, SmallVectorImpl<int64_t>& toVirtual, RingBuffer<int64_t>& nextReal, Logger log) {
    const auto isBarrierMapped = [&](int64_t v, int64_t r) {
        if (r < 0 || checked_cast<size_t>(r) >= toVirtual.size()) {
            return false;
        }
        if (_isDynamicBarriers) {
            return true;
        }
        return toVirtual[r] == v;
    };

    for (auto i : irange(dep.consumer.second)) {
        const auto v = _vdt.id(dep.consumer.first + i);
        const auto r = _barriers[v].realId;

        if (!isBarrierMapped(v, r)) {
            log.trace("{0}[{1}] is waiting for consumer barrier {2} to be mapped", taskType, index, v);
            return Status::Skip;
        }

        if (_barriers[v].producerCount > 0) {
            log.trace("{0}[{1}] waiting for barrier {2} to be produced, {3} remaining", taskType, index, v,
                      _barriers[v].producerCount);
            return Status::Skip;
        }
    }

    for (auto i : irange(dep.producer.second)) {
        const auto v = _vdt.id(dep.producer.first + i);
        const auto r = _barriers[v].realId;

        if (!isBarrierMapped(v, r)) {
            log.trace("{0}[{1}] is waiting for producer barrier {2} to be mapped", taskType, index, v);
            return Status::Skip;
        }
    }

    user.startAfter = 0;

    for (auto i : irange(dep.consumer.second)) {
        const auto v = _vdt.id(dep.consumer.first + i);
        const auto r = _barriers[v].realId;

        if (isBarrierMapped(v, r)) {
            // barrier not ready to be consumed
            if ((_barriers[v].producerCount != 0) || (_barriers[v].consumerCount < count)) {
                log.error(
                        "Simulate barriers failed - barrier {0} not ready to be consumed (producers {1} consumers {2})",
                        v, _barriers[v].producerCount, _barriers[v].consumerCount);
                return Status::Fail;
            }

            _barriers[v].consumerCount -= count;
            user.waits.insert(r);
            user.startAfter = std::max(user.startAfter, v + 1);

            if (_barriers[v].consumerCount == 0) {
                toVirtual[r] = -1;

                if (_isDynamicBarriers) {
                    nextReal.push(r);
                }
            }
        } else {
            log.error("Virtual barrier {0} still not mapped", v);
            return Status::Fail;
        }
    }

    user.cleanAfter = _barriers.size();

    for (auto i : irange(dep.producer.second)) {
        const auto v = _vdt.id(dep.producer.first + i);
        const auto r = _barriers[v].realId;

        if (isBarrierMapped(v, r)) {
            if (_barriers[v].producerCount < count) {
                log.error("Simulate barriers failed - barrier {0} producer count ({1} is lower then number of tasks "
                          "which are producing it ({2})",
                          v, _barriers[v].producerCount, count);
                return Status::Fail;
            }

            _barriers[v].producerCount -= count;
            user.posts.insert(r);
            user.startAfter = std::max(user.startAfter, v + 1);
            user.cleanAfter = std::min(user.cleanAfter, v);
        } else {
            log.error("Virtual barrier {0} still not mapped", v);
            return Status::Fail;
        }
    }

    return Status::Success;
}

void vpux::VPURT::BarrierSimulator::calculateBarrierBatchesForParallelTasks() {
    auto getBarrierIndBatch = [&](mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>& barrierOpsBatch) {
        std::set<int64_t> barrierIndBatch;
        for (const auto barrierOp : barrierOpsBatch) {
            barrierIndBatch.insert(_barrierInfo.getIndex(barrierOp));
        }
        return barrierIndBatch;
    };

    for (auto& barriersToLegalize : getBarrierBatchesToLegalize()) {
        calculateBarrierBatchesForParallelTasks(getBarrierIndBatch(barriersToLegalize));
    }
}

void vpux::VPURT::BarrierSimulator::calculateBarrierBatchesForParallelTasks(const std::set<int64_t>& activeBarriers) {
    // try to select tasks that can run in parallel to tasks associated with the barriers selected for legalization
    auto activeBarrierUsers = _barrierInfo.getBarriersUsers(activeBarriers);
    auto parallelTasksBatches = _barrierInfo.findParallelTasksWithBarrierDependence(activeBarrierUsers);
    if (_barrierBatchesToLegalize.empty()) {
        _barrierBatchesToLegalize.push_back({});
    }

    // add barriers associated with tasks that can execute in parallel to
    // tasks governed by currently active barriers
    for (auto parallelTasks : parallelTasksBatches) {
        for (auto parallelTask : parallelTasks) {
            for (auto bar : _barrierInfo.getUpdateBarriers(parallelTask)) {
                _barrierBatchesToLegalize.rbegin()->insert(bar);
            }
            for (auto bar : _barrierInfo.getWaitBarriers(parallelTask)) {
                _barrierBatchesToLegalize.rbegin()->insert(bar);
            }
        }
    }
}

void vpux::VPURT::BarrierSimulator::linkNextIds(Logger log) {
    log.trace("Create links to the next config of the same barrier");

    for (auto i : irange(_barriers.size())) {
        auto& current = _barriers[i];
        current.nextSameId = -1;

        for (auto j = i + 1; j < _barriers.size(); ++j) {
            if (_barriers[j].realId == current.realId) {
                current.nextSameId = j;
                break;
            }
        }

        log.nest().trace("VID: {0}, PHYS ID: {1}, NEXT SAME PHYS ID: {2}", i, current.realId, current.nextSameId);
    }
}
