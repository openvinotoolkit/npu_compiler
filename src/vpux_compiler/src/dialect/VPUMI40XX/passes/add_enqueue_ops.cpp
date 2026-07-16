//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include <queue>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDENQUEUEOPS
#define GEN_PASS_DEF_ADDENQUEUEOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class AddEnqueueOpsPass : public VPUMI40XX::impl::AddEnqueueOpsBase<AddEnqueueOpsPass> {
public:
    explicit AddEnqueueOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Go through all enqueue tasks and process whole schedule with respect to barrier consumption events
// and check if no enqueue task chosen barrier is not yet fully consumed at the moment of enqueuement
// what means that it will be consumed by some future tasks not yet enqueued
mlir::LogicalResult verifyEnqueueBarrierIsNotBlockedByFutureTask(
        VPUMI40XX::MappedInferenceOp mpi, SmallVector<VPURegMapped::EnqueueOp>& enquOps,
        SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers,
        SmallVector<SmallVector<mlir::Operation*>>& lastDmaWithNoEnqueue, int64_t tilesCount, Logger log) {
    log.trace("Verify enqueue barrier is not blocked by future task");
    // Build information on all barrier consumer count
    SmallVector<int64_t> barrierConsumerCounter(barriers.size(), -1);
    auto barrierCount = barrierConsumerCounter.size();
    for (auto& barrier : barriers) {
        auto barrierIdx = mlir::cast<VPURegMapped::IndexType>(barrier.getResult().getType()).getValue();
        VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                        "Invalid barrier index - {0} out of possible amount of barriers {1}", barrierIdx, barrierCount);
        VPUX_THROW_WHEN(barrierConsumerCounter[barrierIdx] > -1, "Barrier {0} has already been updated", barrierIdx);

        if (barrier.getConsumerCount().has_value()) {
            barrierConsumerCounter[barrierIdx] = barrier.getConsumerCount().value();
        }
    }

    // Before processing enqueue tasks first traverse all initial DMAs on each FIFO
    // as they do not have corresponding enqueue op. Update barriers they consume
    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask) {
                continue;
            }

            if (lastDmaWithNoEnqueue[tileIdx][listIdx] == nullptr) {
                continue;
            }

            auto firstDmaTaskIdx = mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue();
            auto lastDmaTaskIdx = mlir::cast<vpux::VPURegMapped::IndexType>(
                                          lastDmaWithNoEnqueue[tileIdx][listIdx]->getResult(0).getType())
                                          .getValue();

            log.trace("Process DMAs with no enqueue op - DMA{0}:{1}:{2}-{3}", tileIdx, listIdx, firstDmaTaskIdx,
                      lastDmaTaskIdx);

            mlir::Operation* dmaOp;
            mlir::Operation* nextDmaOp = dmaTask.getDefiningOp();

            do {
                dmaOp = nextDmaOp;

                if (auto executableTaskOp = mlir::dyn_cast<VPURegMapped::DMATypeOpInterface>(dmaOp)) {
                    for (const auto& waitBar : executableTaskOp.waitBarriers()) {
                        auto barrierIdx = mlir::cast<VPURegMapped::IndexType>(waitBar.getType()).getValue();
                        VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                                        "Invalid barrier index - {0} out of possible amount of barriers {1}",
                                        barrierIdx, barrierCount);
                        barrierConsumerCounter[barrierIdx]--;
                        log.nest().trace("DMA '{0}' decrements barrier '{1}' consumer counter to '{2}'",
                                         dmaOp->getResult(0).getType(), barrierIdx, barrierConsumerCounter[barrierIdx]);
                    }
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaOp));
                nextDmaOp = nextDma ? nextDma.getOperation() : nullptr;

            } while (nextDmaOp != nullptr && dmaOp != lastDmaWithNoEnqueue[tileIdx][listIdx]);
        }
    }

    for (auto& enquOp : enquOps) {
        auto enqBarIdx = mlir::cast<VPURegMapped::IndexType>(enquOp.getBarrier().getType()).getValue();

        log.trace("EnqOp '{0}:{1}' at barrier idx '{2}'", enquOp.getIndex().getType(), enquOp.getTaskType(), enqBarIdx);
        if (barrierConsumerCounter[enqBarIdx] != 0) {
            log.warning("Barrier '{0}' for enqueue not yet consumed, remaining counters: '{1}'. Execution blocked at "
                        "enqueue op '{2}'",
                        enqBarIdx, barrierConsumerCounter[enqBarIdx], enquOp);
            return mlir::failure();
        }

        auto nextTaskOpToProcess = mlir::cast<VPURegMapped::TaskOpInterface>(enquOp.getStart().getDefiningOp());
        auto lastTaskOp = mlir::cast<VPURegMapped::TaskOpInterface>(enquOp.getEnd().getDefiningOp());
        VPURegMapped::TaskOpInterface taskOp;
        mlir::DenseSet<VPUMI40XX::ExecutableTaskOpInterface> processedDpuTasks;

        do {
            taskOp = nextTaskOpToProcess;
            log.trace("Process task '{0}:{1}'", taskOp.getTaskType(), taskOp.getIndexType());

            VPUMI40XX::ExecutableTaskOpInterface barrieredOp;
            bool isDpuTask = false;  // check if barrieredOp is a DPU task

            auto taskDecrementsConsumerCounter = [&](VPUMI40XX::ExecutableTaskOpInterface& barrieredOp) {
                if (!isDpuTask) {
                    return true;
                }
                if (!processedDpuTasks.contains(barrieredOp)) {
                    // Since only first DPU variant configures wait barriers,
                    // allow to decrement the consumer counter only for the first encountered DPU variant
                    // of any given invariant.
                    processedDpuTasks.insert(barrieredOp);
                    return true;
                }
                return false;
            };

            if (enquOp.getTaskType() == VPURegMapped::TaskType::DPUVariant) {
                auto dpuVariantOp = mlir::cast<VPUMI40XX::DPUVariantOp>(taskOp.getOperation());
                barrieredOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(
                        dpuVariantOp.getInvariant().getDefiningOp());
                isDpuTask = true;
            } else {
                barrieredOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(taskOp.getOperation());
            }

            if (barrieredOp && taskDecrementsConsumerCounter(barrieredOp)) {
                for (const auto& waitBar : barrieredOp.waitBarriers()) {
                    auto barrierIdx = mlir::cast<VPURegMapped::IndexType>(waitBar.getType()).getValue();
                    VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                                    "Invalid barrier index - {0} out of possible amount of barriers {1}", barrierIdx,
                                    barrierCount);

                    VPUX_THROW_UNLESS(barrierConsumerCounter[barrierIdx] > 0,
                                      "Barrier {0} consumer count cannot be decremented - {1}", barrierIdx,
                                      barrierConsumerCounter[barrierIdx]);
                    barrierConsumerCounter[barrierIdx]--;
                    log.nest().trace("Task '{0}' decrements barrier '{1}' consumer counter to '{2}'",
                                     taskOp.getOperation()->getResult(0).getType(), barrierIdx,
                                     barrierConsumerCounter[barrierIdx]);
                }
            }
            nextTaskOpToProcess = VPUMI40XX::getNextOp(taskOp);
        } while (taskOp != lastTaskOp);
    }

    return mlir::success();
}

mlir::LogicalResult verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder(SmallVector<VPURegMapped::EnqueueOp>& enquOps,
                                                                       Logger log) {
    llvm::DenseMap<VPUMI40XX::HwQueueType, uint32_t> lastTaskPerQueue;

    for (auto& enqu : enquOps) {
        auto tile = mlir::cast<VPURegMapped::IndexType>(enqu.getStart().getType()).getTileIdx();
        auto list = mlir::cast<VPURegMapped::IndexType>(enqu.getStart().getType()).getListIdx();
        auto index = mlir::cast<VPURegMapped::IndexType>(enqu.getStart().getType()).getValue();
        auto taskType = enqu.getTaskType();
        VPUMI40XX::HwQueueType qType({taskType, tile, list});

        if (lastTaskPerQueue.find(qType) != lastTaskPerQueue.end()) {
            if (lastTaskPerQueue[qType] > index) {
                log.warning("Incorrect position for enque {0} in WorkItem list", enqu);
                return mlir::failure();
            }
        }

        lastTaskPerQueue[qType] = index;
    }

    return mlir::success();
}

// For each task that depends also on descriptor fetching (DPU and SHV)
// read preconfigured enqueue barrier and create new enqueue ops
void addPredefinedEnqusForTasksWithFetch(VPUMI40XX::MappedInferenceOp mpi, const VPURegMapped::TaskType primary,
                                         const VPURegMapped::TaskType secondary,
                                         VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder,
                                         int64_t& counter, Logger log, const int64_t tilesCount,
                                         const int64_t listsCount = 1) {
    auto ctx = mpi.getContext();

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < listsCount; listIdx++) {
            auto startVal = mpi.getListHead(primary, tileIdx, listIdx);
            if (!startVal) {
                continue;
            }

            log.trace("Get enqueue barriers for {0}:{0}:{1}:{2}", stringifyTaskType(primary), tileIdx, listIdx);
            log = log.nest();

            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;

            VPURegMapped::TaskOpInterface taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(startVal.getDefiningOp());
            do {
                auto filteredRange = to_small_vector(
                        taskOp.getResult().getUsers() | vpux::filtered([&secondary](mlir::Operation* op) {
                            // filter out the usages that are of the secondaryType
                            auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
                            return taskOp && taskOp.getTaskType() == secondary;
                        }));

                auto firstSecondaryIt = vpux::min_element(filteredRange, VPUMI40XX::taskOpComparator);
                auto lastSecondaryIt = vpux::max_element(filteredRange, VPUMI40XX::taskOpComparator);
                auto firstSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**firstSecondaryIt);
                auto lastSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**lastSecondaryIt);

                auto barrieredOp = VPUMI40XX::getBarrieredOp(taskOp, lastSecondary);

                auto enqueueTarget = barrieredOp.getEnqueueBarrier();

                VPUX_THROW_UNLESS(enqueueTarget, "No enqueue barrier configured for op {0}", barrieredOp);

                // if the previous enqueue's barrier is the same as the target barrier, we can just add this variant
                // range to the previous enqueue. This is made with the assumption that we topologically iterate over
                // the variants list by their listOrder
                if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                    localPreviousEnqu.getEndMutable().assign(lastSecondary->getResult(0));
                    log.trace("Enqueue task {0} with previous task",
                              mlir::cast<VPURegMapped::IndexType>(firstSecondary->getResult(0).getType()).getValue());
                } else {
                    auto index = VPURegMapped::IndexType::get(ctx, counter);
                    mlir::Value previousEnquVal =
                            localPreviousEnqu ? localPreviousEnqu.getResult()
                                              : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                    localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                            taskOp->getLoc(), index, previousEnquVal, enqueueTarget,
                            /*previousTaskIdxOnSameBarrier*/ nullptr, secondary, firstSecondary->getResult(0),
                            lastSecondary->getResult(0));
                    counter++;
                    log.trace("New enqueue for task {0} at barrier {1}",
                              mlir::cast<VPURegMapped::IndexType>(firstSecondary->getResult(0).getType()).getValue(),
                              mlir::cast<VPURegMapped::IndexType>(enqueueTarget.getType()).getValue());
                }

                taskOp = VPUMI40XX::getNextOp(taskOp);
            } while (taskOp);

            globalPreviousEnqu = localPreviousEnqu;
            log = log.unnest();
        }
    }
}

// For each DMA task read preconfigured enqueue barrier and create new enqueue ops
void addPredefinedEnqusForDmas(VPUMI40XX::MappedInferenceOp mpi, const int64_t tilesCount,
                               VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder, int64_t& counter,
                               SmallVector<SmallVector<mlir::Operation*>>& lastDmaWithNoEnqueue, Logger log) {
    auto ctx = mpi.getContext();

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask) {
                continue;
            }

            log.trace("Get enqueue barriers for {0}:{1}:{2}", stringifyTaskType(VPURegMapped::TaskType::DMA), tileIdx,
                      listIdx);
            log = log.nest();

            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;

            while (dmaTask) {
                auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(dmaTask.getDefiningOp());

                if (executableTaskOp != nullptr && executableTaskOp.getEnqueueBarrier() != nullptr) {
                    auto enqueueTarget = executableTaskOp.getEnqueueBarrier();

                    if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                        log.trace("Enqueue task {0} with previous task",
                                  mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                        localPreviousEnqu.getEndMutable().assign(dmaTask);
                    } else {
                        auto index = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(counter));
                        mlir::Value previousEnquVal =
                                localPreviousEnqu ? localPreviousEnqu.getResult()
                                                  : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                        localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                                dmaTask.getLoc(), index, previousEnquVal, enqueueTarget,
                                /*previousTaskIdxOnSameBarrier*/ nullptr, VPURegMapped::TaskType::DMA, dmaTask,
                                dmaTask);

                        counter++;
                        log.trace("New enqueue for task {0} at barrier {1}",
                                  mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue(),
                                  mlir::cast<VPURegMapped::IndexType>(enqueueTarget.getType()).getValue());
                    }
                } else if (localPreviousEnqu) {
                    log.trace("Enqueue task {0} with previous task",
                              mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                    localPreviousEnqu.getEndMutable().assign(dmaTask);
                } else {
                    log.trace("Enqueue task {0} at bootstrap",
                              mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                    lastDmaWithNoEnqueue[tileIdx][listIdx] = dmaTask.getDefiningOp();
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaTask.getDefiningOp()));
                dmaTask = nextDma ? nextDma.getResult() : nullptr;
            }
            log = log.unnest();
        }
    }
}

// Class for representing enqueue ops dependencies graph
// and providing topological order of enqueue ops
// Enqueue ops for same barrier are grouped together in a single node as
// nevertheless later they need to be placed in IR adjacent to each other
// Graph dependencies are constructed based on task types enqueued by given
// group of enqueue ops on barriers.
// Example:
//  EnqGroup[0]    ->  EnqGroup[1]
//    Bar:X              Bar:Y
//    DPU[0][0-1]        DPU[0][2-3]
//    DMA[0][0-2]        SHV[0][0-1]
// In above example EnqGroup[1] depends on EnqGroup[0] because EnqGroup[1]
// enqueues later DPU tasks than EnqGroup[0]
//
class EnqueueOpGroupsGraph {
public:
    struct EnqOpData {
        VPURegMapped::EnqueueOp enqOp;
        uint32_t lastTaskIdx;
    };

    struct EnqueTypeComparator {
        bool operator()(const VPUMI40XX::HwQueueType& lhs, const VPUMI40XX::HwQueueType& rhs) const {
            if (lhs.type == rhs.type) {
                if (lhs.tile == rhs.tile) {
                    return lhs.index < rhs.index;
                }
                return lhs.tile < rhs.tile;
            }
            // As last compare queue type. Since HwQueueType enum uses DMA -> SHV -> DPU ordering
            // but exisitng PWLM logic enqueued tasks in order DPU -> SHV -> DMA. To prevent from
            // performance changes follow such order as well
            return lhs.type > rhs.type;
        }
    };

    struct EnqOpsOnBarrier {
        size_t index;
        uint32_t barrierIdx;
        std::map<VPUMI40XX::HwQueueType, EnqOpData, EnqueTypeComparator> enqDataOnQueue;
    };

    // Graph representation using adjacency list and indegree counts with capability of providing
    // topological order of nodes. Nodes are identified by their index in the adjacency list.
    // Each node stores a single number.
    // Indented usage:
    // - node key - EnqOpsOnBarrier index
    // - node value - barrier index
    // Genererated topological sort will also align on barrier indexes
    class Graph {
    public:
        // Insert new node without any connections
        size_t addNode(const size_t& value) {
            _adjList.push_back(std::make_pair(value, std::vector<size_t>{}));
            _indegree.push_back(0);
            return _adjList.size() - 1;
        }

        // Update internal graph container with a new edge
        void addEdge(const size_t& fromNodeId, const size_t& toNodeId) {
            auto& neighbors = _adjList[fromNodeId].second;
            if (std::find(neighbors.begin(), neighbors.end(), toNodeId) != neighbors.end()) {
                // Edge already exists
                return;
            }
            neighbors.push_back(toNodeId);
            _indegree[toNodeId]++;
        }

        // Return a vector with neighbor nodes for a given node
        std::vector<size_t> getNeighbors(const size_t& nodeId) const {
            return _adjList[nodeId].second;
        }

        size_t getNodeValue(const size_t& nodeId) const {
            return _adjList[nodeId].first;
        }

        std::vector<size_t> getTopologicalOrder(Logger log) {
            auto smallerNodeValCmp = [&](size_t a, size_t b) {
                return getNodeValue(a) < getNodeValue(b);
            };

            std::vector<size_t> nodeOrder;
            std::set<size_t, decltype(smallerNodeValCmp)> readyNodes(smallerNodeValCmp);
            auto indegree = _indegree;
            // Initialize queue with all nodes having indegree 0
            for (size_t i = 0; i < _adjList.size(); ++i) {
                if (indegree[i] == 0) {
                    readyNodes.insert(i);
                }
            }

            // Process nodes and decrement indegrees of traversed nodes
            while (!readyNodes.empty()) {
                int current = *readyNodes.begin();
                readyNodes.erase(readyNodes.begin());
                nodeOrder.push_back(current);

                SmallVector<size_t> nextNodes;
                for (int neighbor : getNeighbors(current)) {
                    indegree[neighbor]--;
                    if (indegree[neighbor] == 0) {
                        readyNodes.insert(neighbor);
                    }
                }
            }

            // If not all nodes are processed, cycle exists
            if (nodeOrder.size() != _adjList.size()) {
                log.warning("Cyclic dependency detected during enqueue ops ordering! Ordered {0} out of {1} nodes.",
                            nodeOrder.size(), _adjList.size());
                if (nodeOrder.size() > 0) {
                    log.warning("  Last processed enqueue barrier: {0}", getNodeValue(nodeOrder.back()));
                }
                return {};
            }

            return nodeOrder;
        }

    private:
        std::vector<std::pair<size_t, std::vector<size_t>>> _adjList;
        std::vector<size_t> _indegree;
    };

    EnqueueOpGroupsGraph(SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers) {
        identifyEnqueueOps(barriers);
        buildEnqueueOpsGraph();
    }

    SmallVector<VPURegMapped::EnqueueOp> getTopologicalOrder(Logger log) {
        SmallVector<VPURegMapped::EnqueueOp> orderedEnqOps;
        auto topoOrder = _enqGraph.getTopologicalOrder(log);
        if (topoOrder.empty()) {
            return orderedEnqOps;
        }
        for (auto nodeId : topoOrder) {
            auto enqOpsOnBarrier = _enqOpsOnBarriers[nodeId];
            VPUX_THROW_WHEN(enqOpsOnBarrier.index != nodeId,
                            "EnqueueOpGroupsGraph internal error: nodeId {0} does not match expected index {1}", nodeId,
                            enqOpsOnBarrier.index);
            for (const auto& [_, enqData] : enqOpsOnBarrier.enqDataOnQueue) {
                orderedEnqOps.push_back(enqData.enqOp);
            }
        }
        return orderedEnqOps;
    }

private:
    void identifyEnqueueOps(SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers) {
        // Analyze enqueue ops in barrier order so that enqueues
        // attached to same barrier are adjacent to each other
        mlir::Value prevEnqBarrier = nullptr;
        for (auto barrier : barriers) {
            for (auto user : barrier.getResult().getUsers()) {
                auto enqOp = mlir::dyn_cast<VPURegMapped::EnqueueOp>(user);
                if (!enqOp) {
                    continue;
                }

                auto enqBar = enqOp.getBarrier();
                auto enqBarIdx = mlir::cast<VPURegMapped::IndexType>(enqBar.getType()).getValue();

                auto tile = mlir::cast<VPURegMapped::IndexType>(enqOp.getEnd().getType()).getTileIdx();
                auto list = mlir::cast<VPURegMapped::IndexType>(enqOp.getEnd().getType()).getListIdx();
                auto endTaskIdx = mlir::cast<VPURegMapped::IndexType>(enqOp.getEnd().getType()).getValue();
                auto taskType = enqOp.getTaskType();
                VPUMI40XX::HwQueueType queueType({taskType, tile, list});
                _numberOfEnqsPerQueue[queueType]++;

                if (prevEnqBarrier == nullptr || prevEnqBarrier != enqBar) {
                    // New barrier encountered
                    EnqOpsOnBarrier enqOpsOnBar;
                    enqOpsOnBar.index = _enqOpsOnBarriers.size();
                    enqOpsOnBar.barrierIdx = enqBarIdx;
                    _enqOpsOnBarriers.push_back(enqOpsOnBar);
                }
                _enqOpsOnBarriers.back().enqDataOnQueue[queueType] = {enqOp, endTaskIdx};
                prevEnqBarrier = enqBar;
            }
        }
    }

    void buildEnqueueOpsGraph() {
        // Build graph by creating nodes which store enqueue group on same barrier
        for (const auto& enqOpsOnBar : _enqOpsOnBarriers) {
            auto nodeId = _enqGraph.addNode(enqOpsOnBar.barrierIdx);
            VPUX_THROW_WHEN(nodeId != enqOpsOnBar.index,
                            "EnqueueOpGroupsGraph internal error: nodeId {0} does not match expected index {1}", nodeId,
                            enqOpsOnBar.index);
        }

        // Build edges between nodes (enqueue groups) for eqch queue type separetely and do this
        // basd on growing index of tasks enqueued on given queue
        for (const auto& queueAndCount : _numberOfEnqsPerQueue) {
            const auto& queue = queueAndCount.first;
            const auto& count = queueAndCount.second;

            // Create a copy vector of enqueue ops
            // This vector will be sorted for a single HW FIFO type to determine edges
            auto sortVec = _enqOpsOnBarriers;
            // Sort enqueue ops following one specific HW FIFO as in given iteration
            // one queue  is processed and edges are added based on that queue
            // Sort enqueue ops on barriers based on last task index if they both enqueue
            // tasks on the provided HW FIFO. If only one of them enqueues tasks on given HW FIFO
            // it is placed first. If none of them enqueues tasks on given HW FIFO order
            // is based on barrier index, but this is just to have deterministic order. This code
            // nevertheless cares only about enqueue ops for given HW FIFO
            std::sort(sortVec.begin(), sortVec.end(), [&](const EnqOpsOnBarrier& a, const EnqOpsOnBarrier& b) {
                auto aQueueIt = a.enqDataOnQueue.find(queue);
                auto aHasQueue = (aQueueIt != a.enqDataOnQueue.end());

                auto bQueueIt = b.enqDataOnQueue.find(queue);
                auto bHasQueue = (bQueueIt != b.enqDataOnQueue.end());
                if (aHasQueue && bHasQueue) {
                    return aQueueIt->second.lastTaskIdx < bQueueIt->second.lastTaskIdx;
                } else if (aHasQueue && !bHasQueue) {
                    return true;
                } else if (!aHasQueue && bHasQueue) {
                    return false;
                }
                return a.barrierIdx < b.barrierIdx;
            });

            for (size_t i = 0; i < count - 1; i++) {
                auto enq = sortVec[i];
                auto nextEnq = sortVec[i + 1];
                _enqGraph.addEdge(enq.index, nextEnq.index);
            }
        }
    }

    mlir::DenseMap<VPUMI40XX::HwQueueType, size_t> _numberOfEnqsPerQueue;
    SmallVector<EnqOpsOnBarrier> _enqOpsOnBarriers;
    Graph _enqGraph;
};

// After inserting enqueue ops for each FIFO in the IR, enqueue ops need to be ordered
// as there is only 1 enqueue ops (WorkItem task) list that will be processed by VPU-FW
// There are following restrictions:
// 1:
// Prepared enqueues order need to always guarantee that for given HW FIFO tasks are not
// enqueued out of order, meaning enqueue for taskFifoX[N] is not placed after enqueue
// for taskFifoX[N+1]
// This constraint is verified by verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder()
//
// 2:
// Enqueue ops for same barrier are grouped together due to constraints on how WorkItem
// tasks are processed. This is only needed if WorkItem links are not supported by FW.
//
// 3:
// For each Enq[j] and Enq[i], such that j > i, tasks enqueued by Enq[j] cannot block barrier consumption
// of any task enqueued by Enq[i]
// Example for incorrect order:
//  Enq[0]: Bar0, taskX[0]
//  Enq[1]: Bar2, taskX[1]
//  Enq[2]: Bar1, taskY[1], taskY[1].wait(Bar2)
//
// Although above order satisfies constraint 1 it will cause a deadlock as Enq[1] happening at Bar2
// will never happen as Bar2 will be consumed by task which is enqueued later at Enq[2]
// In that case Enq[2] should be placed before Enq[1]
//
// This constraint can be satisfied by updating the order of enqueues with order of barriers - smaller
// index first. Barriers are ordered based on consumption order so if Bar[i] depends on Bar[j] it is
// guaranteed to have larger index (i > j)
// Important: Larger index does not always mean barrier dependency, for example Bar[N+1] does not
// need to depend on Bar[N] in schedule if they are on parallel branch
//
// This constraint is verified by verifyEnqueueBarrierIsNotBlockedByFutureTask()
//
// To satisfy both constraints algorithm should process enqueues for tasks following
// their order in given HW FIFO to satisfy constraint 1 and when picking HW FIFO from
// which enqueue to order first should use smaller barrier index to satisfy constraint 3
SmallVector<VPURegMapped::EnqueueOp> getEnqueueOpsOrder(SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers,
                                                        Logger log) {
    // In case WorkItem links are not supported build enqueue ops dependencies graph
    // and get topological order of enqueue ops
    EnqueueOpGroupsGraph enqueueOpGroupsGraph(barriers);
    return enqueueOpGroupsGraph.getTopologicalOrder(log);
}

void AddEnqueueOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();
    const auto shavesCountPerTile =
            config::getAvailableExecutor(parentModule, config::ExecutorKind::SHAVE_ACT).getCount();

    auto barriers = to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());

    VPURegMapped::EnqueueOp globalPreviousEnqu;
    int64_t globalEnquCounter = 0;

    // Store information on last DMAs on each FIFO which does not have a corresponding
    // enqueue op added in this pass.
    SmallVector<SmallVector<mlir::Operation*>> lastDmaWithNoEnqueue(tilesCount, SmallVector<mlir::Operation*>(2));

    _log.trace("Use already configured enqueue barriers by algorithm from VPURT");

    addPredefinedEnqusForTasksWithFetch(mpi, VPURegMapped::TaskType::DPUInvariant, VPURegMapped::TaskType::DPUVariant,
                                        globalPreviousEnqu, builder, globalEnquCounter, _log, tilesCount);
    addPredefinedEnqusForTasksWithFetch(mpi, VPURegMapped::TaskType::ActKernelRange,
                                        VPURegMapped::TaskType::ActKernelInvocation, globalPreviousEnqu, builder,
                                        globalEnquCounter, _log, tilesCount, shavesCountPerTile);

    addPredefinedEnqusForDmas(mpi, tilesCount, globalPreviousEnqu, builder, globalEnquCounter, lastDmaWithNoEnqueue,
                              _log);

    if (globalEnquCounter == 0) {
        _log.trace("No enqueue ops were added to the IR");
        mpi.setWorkItemCount(0);
        return;
    }

    // After inserting enqueue ops for each FIFO in the IR, enqueue ops need to be ordered
    // as there is only 1 enqueue ops (WorkItem task) list that will be processed by VPU-FW
    auto enquOpsOrder = getEnqueueOpsOrder(barriers, _log);
    VPUX_THROW_WHEN(enquOpsOrder.empty(), "getEnqueueOpsOrder failed");

    // Update enqueue ops order in IR
    for (auto& enqu : enquOpsOrder) {
        enqu.getOperation()->moveBefore(mpi.getOperation());
    }

    auto enquOps = to_small_vector(netFunc.getOps<VPURegMapped::EnqueueOp>());
    if (!enquOps.empty()) {
        mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
    }
    mpi.setWorkItemCount(enquOps.size());
    VPUMI40XX::reindexEnqueueOps(enquOps);

    // Verify enqueue ops can be enqueued at given barriers
    if (mlir::failed(verifyEnqueueBarrierIsNotBlockedByFutureTask(mpi, enquOps, barriers, lastDmaWithNoEnqueue,
                                                                  tilesCount, _log))) {
        VPUX_THROW("verifyEnqueueBarrierIsNotBlockedByFutureTask failed");
    }

    // Check if enqueues order for given HW FIFO is not enqueueing tasks
    // for this FIFO out of order - task N needs to be enqueued before task N+1
    if (mlir::failed(verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder(enquOps, _log))) {
        VPUX_THROW("verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder failed");
    }
}

}  // namespace

//
// createAddEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddEnqueueOpsPass(Logger log) {
    return std::make_unique<AddEnqueueOpsPass>(log);
}
