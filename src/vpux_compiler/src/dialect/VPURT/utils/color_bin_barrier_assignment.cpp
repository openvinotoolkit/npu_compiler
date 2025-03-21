//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/utils/color_bin_barrier_assignment.hpp"

#include <llvm/ADT/SetOperations.h>
#include <limits>

using namespace vpux;

// An experimental number to represent the delayed execute step before the reuse of the physical barrier
constexpr size_t BARRIER_GRACE_PERIOD = 5;

// An experimental number to represent the threshold of the virtual barriers count mapped to the minimum barrier bin
constexpr double THRESHOLD_FOR_MIN_BARRIER_BIN = 20;

// Targets which need to delay the reuse of physical barriers, since on those hardware platform, the runtime has to
// reprogram the barrier count before the next reuse
const std::set<VPU::ArchKind> compatibleTargets = {VPU::ArchKind::NPU37XX};

namespace {

size_t getBarrierGracePeriod(VPU::ArchKind arch) {
    if (compatibleTargets.find(arch) != compatibleTargets.end()) {
        return BARRIER_GRACE_PERIOD;
    }
    return 0;
}
}  // namespace

VPURT::BarrierColorBin::BarrierColorBin(size_t numBarriers, VPU::ArchKind arch, Logger log)
        : _numBarriers(numBarriers), _log(log) {
    _gracePeriod = getBarrierGracePeriod(arch);
}

bool VPURT::BarrierColorBin::calculateBinSize(BarrierGraphInfo& BarrierGraphInfo) {
    auto numVirtualBarriers = BarrierGraphInfo.getBarrierInfo().getNumOfBarrierOps();
    VPUX_THROW_UNLESS(numVirtualBarriers > _numBarriers,
                      "Num of virtual barriers {0} is equal or less than physical barrier {1}", numVirtualBarriers,
                      _numBarriers);
    _barrierBinType.resize(numVirtualBarriers);

    std::map<BinType, size_t> barrierCounts;
    auto barrierLongestHopTaskTypeVec = BarrierGraphInfo.getBarrierLongestQueueType();
    auto& barrierInfo = BarrierGraphInfo.getBarrierInfo();
    // Count barrier num by longest hop task type
    for (size_t barrierInd = 0; barrierInd < barrierLongestHopTaskTypeVec.size(); barrierInd++) {
        const auto taskType = barrierLongestHopTaskTypeVec[barrierInd];
        _barrierBinType[barrierInd] = taskType;
        // Barrier with same executor kind are placed in same bin except DMA since we have symmetrical scheduling
        // between DPU and Shave tiles and all use same barriers. Otherwise we should treat all engines and tiles
        // independently as we do it for DMAs
        if (taskType.type != VPU::ExecutorKind::DMA_NN) {
            _barrierBinType[barrierInd].id = 0;
        }
        // Final barrier will not be considered due to runtime error in some cases. When the bin contains final barrier
        // and the barrier is configure with physical barrier in first loop, the runtime might return error during
        // inference
        auto barrierOp = barrierInfo.getBarrierOpAtIndex(barrierInd);
        if (barrierOp.getIsFinalBarrier()) {
            continue;
        }
        barrierCounts[_barrierBinType[barrierInd]]++;
    }
    if (barrierCounts.empty()) {
        return false;
    }

    // Calculate minimum size for each barrier bin
    std::map<BinType, size_t> binBarrierNum;
    for (const auto& [type, _] : barrierCounts) {
        binBarrierNum[type] = getMinBinSize(barrierCounts, type);
    }

    // Num of barrier has been assigned to barrier bins
    size_t numAssignedBarrier =
            std::accumulate(binBarrierNum.begin(), binBarrierNum.end(), 0, [](size_t sum, const auto& entry) {
                return sum + entry.second;
            });

    // Calculate weights for each bin to assign remained available barriers
    std::map<BinType, float> binWeights;
    float totalWeight = 0.f;
    for (const auto& [type, taskNum] : barrierCounts) {
        auto weight = static_cast<float>(taskNum);
        binWeights[type] = weight;
        totalWeight += weight;
    }

    if (totalWeight < std::numeric_limits<float>::epsilon()) {
        return false;
    }
    totalWeight = _numBarriers / totalWeight;
    for (auto& [_, weight] : binWeights) {
        weight = weight * totalWeight;
    }

    SmallVector<BinType> items;
    for (const auto& [binType, _] : binWeights) {
        items.push_back(binType);
    }
    llvm::sort(items, [&](const auto& lhs, const auto& rhs) {
        return binWeights[lhs] > binWeights[rhs];
    });

    for (const auto& type : items) {
        const auto& weight = binWeights[type];
        auto availableBarriers = _numBarriers - numAssignedBarrier;
        auto barriersToBeAssigned = static_cast<size_t>(std::round(weight) - binBarrierNum[type]);

        if (binBarrierNum[type] + barriersToBeAssigned > barrierCounts[type]) {
            barriersToBeAssigned = barrierCounts[type] - binBarrierNum[type];
        }
        if (barriersToBeAssigned > availableBarriers) {
            barriersToBeAssigned = availableBarriers;
        }
        binBarrierNum[type] += barriersToBeAssigned;
        numAssignedBarrier += barriersToBeAssigned;
    }
    auto getVirturalPerPhysicalRatio = [&](const auto& entry) {
        return static_cast<double>(barrierCounts[entry.first]) / entry.second;
    };
    while (numAssignedBarrier < _numBarriers) {
        auto binIter =
                std::max_element(binBarrierNum.begin(), binBarrierNum.end(), [&](const auto& lhs, const auto& rhs) {
                    return getVirturalPerPhysicalRatio(lhs) < getVirturalPerPhysicalRatio(rhs);
                });
        binIter->second++;
        numAssignedBarrier++;
    }

    for (const auto& [binType, barrierNum] : binBarrierNum) {
        _log.trace("Calculate barrier color bin {0}:{1} size: {2}", VPU::stringifyExecutorKind(binType.type),
                   binType.id, barrierNum);
        _assignedBarriers[binType] = SmallVector<std::deque<size_t>>(barrierNum);
        _barrierSelectionCount[binType] = SmallVector<size_t>(barrierNum, 0);
        _barrierSelectionExecutionStep[binType] = SmallVector<size_t>(barrierNum, 0);
        _virtualBarrierSelection[binType] = SmallVector<int64_t>(barrierNum, -1);
        _physicalBarrierList[binType] = SmallVector<size_t>();
    }

    // Order barrier PIDs for given bin type next to each other
    // Example:
    //  DMA0 bins: PID0-3
    //  DMA1 bins: PID4-7
    //  DPU  bins: PID8-11
    // This is not because of any functional requirement but is done to simplify visualization
    // of VID->PID->BinType assignment
    size_t basePid = 0;
    for (const auto& [binType, barrierNum] : binBarrierNum) {
        for (size_t pid = basePid; pid < barrierNum + basePid; pid++) {
            _physicalBarrierList[binType].push_back(pid);
            _log.trace("Physical barrier {0} is in bin {1}:{2}", pid, VPU::stringifyExecutorKind(binType.type),
                       binType.id);
        }
        basePid += barrierNum;
    }

    return true;
}

bool VPURT::BarrierColorBin::findPhysicalBarrierInBin(BarrierGraphInfo& BarrierGraphInfo, size_t barrierInd) {
    _log.trace("try to find available physical barrier for virtual barrier {0}", barrierInd);
    if (_barrierVirtualToPhysicalMapping[barrierInd] != INVALID_BARRIER_PID) {
        // Barrier has been mapped
        _log.trace("virtual barrier {0} is already assigned to physical barrier {1}", barrierInd,
                   _barrierVirtualToPhysicalMapping[barrierInd]);
        return true;
    }

    auto origBinType = _barrierBinType[barrierInd];
    auto binTypeWithPriority = getBinTypeWithPriority(origBinType);
    bool findAvailablePhysicalBarrier = false;
    for (auto binType : binTypeWithPriority) {
        auto index = getFreePhysicalBarrierIndexInBin(barrierInd, binType, BarrierGraphInfo);
        if (!index.has_value()) {
            _log.trace("No available physical barrier in bin {0}:{1} for barrier {2}",
                       VPU::stringifyExecutorKind(binType.type), binType.id, barrierInd);
            continue;
        }

        if (binType != origBinType) {
            _barrierBinType[barrierInd] = binType;
            _log.trace("Change bin type from {0} to {1} for barrier {2}", VPU::stringifyExecutorKind(origBinType.type),
                       VPU::stringifyExecutorKind(binType.type), barrierInd);
        }
        auto indexValue = index.value();

        VPUX_THROW_UNLESS(indexValue < _physicalBarrierList[binType].size(),
                          "Invalid index {0} in barrier bin {1} for barrier {2}", indexValue,
                          VPU::stringifyExecutorKind(binType.type), barrierInd);
        auto physicalId = _physicalBarrierList[binType][indexValue];
        const auto barrierLastStep = _barrierLastExecutionStep[barrierInd] + _gracePeriod;
        _barrierSelectionExecutionStep[binType][indexValue] = barrierLastStep;
        _virtualBarrierSelection[binType][indexValue] = barrierInd;
        _barrierVirtualToPhysicalMapping[barrierInd] = physicalId;
        _assignedBarriers[binType][indexValue].push_back(barrierInd);
        findAvailablePhysicalBarrier = true;
        _log.trace("Assign virtual barrier {0} with physical barrier {1} in bin {2}:{3}", barrierInd, physicalId,
                   VPU::stringifyExecutorKind(binType.type), binType.id);
        break;
    }
    if (!findAvailablePhysicalBarrier) {
        _log.warning("Can not configure physical barrier for barrier {0}", barrierInd);
        return false;
    }
    return true;
}

bool VPURT::BarrierColorBin::assignPhysicalBarrier(BarrierGraphInfo& BarrierGraphInfo, BarrierSimulator& simulator) {
    _barrierVirtualToPhysicalMapping.resize(_barrierBinType.size(), INVALID_BARRIER_PID);
    getBarrierExecutionStepInfo(BarrierGraphInfo);

    for (const auto& [executeStep, taskBatch] : BarrierGraphInfo.getExecutionStepTaskBatch()) {
        _log.trace("handle execution batch {0}", executeStep);

        std::set<size_t> waitBarriersInCurrentBatch;
        std::set<size_t> updateBarriersInCurrentBatch;

        // Step 1. Clear barrier assignment
        for (auto& [binType, _] : _assignedBarriers) {
            clearBarrierAssignment(_assignedBarriers[binType], executeStep);
        }

        // Step 2. Get all wait and update barriers for this batch
        auto& barrierInfo = BarrierGraphInfo.getBarrierInfo();
        for (auto task : taskBatch) {
            for (auto bar : barrierInfo.getWaitBarriers(task)) {
                waitBarriersInCurrentBatch.insert(bar);
            }
            for (auto bar : barrierInfo.getUpdateBarriers(task)) {
                updateBarriersInCurrentBatch.insert(bar);
            }
        }

        // Step 3. Get physical barrier for the barriers
        for (auto& barrierInd : waitBarriersInCurrentBatch) {
            if (!findPhysicalBarrierInBin(BarrierGraphInfo, barrierInd)) {
                return false;
            };
        }
        for (auto& barrierInd : updateBarriersInCurrentBatch) {
            if (!findPhysicalBarrierInBin(BarrierGraphInfo, barrierInd)) {
                return false;
            }
        }
    }

    _barrierOrder = simulator.generateBarrierOrderWithSimulation(_log, _numBarriers, _barrierVirtualToPhysicalMapping);
    if (_barrierOrder.empty()) {
        _log.error("Can not assign physical barrier using color bin due to simulation error!");
        return false;
    }

    return true;
}

size_t VPURT::BarrierColorBin::getPhysicalBarrier(size_t virtualBarrierInd) {
    return _barrierVirtualToPhysicalMapping[virtualBarrierInd];
}

// Reorder barrier ops according to its physical barrier id to avoid hang in runtime.
// For the first N barrier op, in which N is the physical barrier account, the physical barrier id should be 0 to N-1
// with ascending order
void VPURT::BarrierColorBin::reorderBarriers(BarrierGraphInfo& BarrierGraphInfo, mlir::func::FuncOp funcOp) {
    VPUX_THROW_WHEN(_barrierOrder.empty(), "Barrier order data is empty");

    auto insertPoint = &funcOp.getBody().front().front();
    auto& barrierInfo = BarrierGraphInfo.getBarrierInfo();
    VPURT::BarrierOpInterface finalBarOp = nullptr;
    for (auto bar : _barrierOrder) {
        auto barOp = barrierInfo.getBarrierOpAtIndex(bar);
        barOp->moveAfter(insertPoint);
        insertPoint = barOp;

        if (barOp.getIsFinalBarrier()) {
            VPUX_THROW_UNLESS(finalBarOp == nullptr, "More then one final barrier: {0} and {1}", finalBarOp, barOp);
            finalBarOp = barOp;
        }
    }

    // Other passes expect final barrierto be at the end of IR
    // even if it is not the last in terms of barrier reprogramming order
    if (finalBarOp != nullptr && insertPoint != finalBarOp) {
        finalBarOp->moveAfter(insertPoint);
    }
}

size_t VPURT::BarrierColorBin::getMinBinSize(const std::map<BinType, size_t>& barrierCounts, const BinType& binType) {
    size_t minBarrierBinSize = static_cast<size_t>(std::ceil(_numBarriers / (2.0 * barrierCounts.size())));

    // The min barrier size will increase 50% if too much virtual barriers are mapped in the
    // minimum bin
    if (barrierCounts.at(binType) > THRESHOLD_FOR_MIN_BARRIER_BIN) {
        minBarrierBinSize += static_cast<size_t>(std::ceil(minBarrierBinSize / 2.0));
    }
    return std::min(minBarrierBinSize, barrierCounts.at(binType));
}

void VPURT::BarrierColorBin::getBarrierExecutionStepInfo(BarrierGraphInfo& BarrierGraphInfo) {
    _barrierFirstExecutionStep = BarrierGraphInfo.getBarrierFirstExecutionStep();
    _barrierLastExecutionStep = BarrierGraphInfo.getBarrierLastExecutionStep();
}

void VPURT::BarrierColorBin::clearBarrierAssignment(MutableArrayRef<std::deque<size_t>> assignedBarriers,
                                                    const size_t& executionStep) {
    for (auto& barrierList : assignedBarriers) {
        while (barrierList.size()) {
            if (_barrierLastExecutionStep[barrierList.front()] < executionStep) {
                barrierList.pop_front();
            } else {
                break;
            }
        }
    }
}

std::optional<size_t> VPURT::BarrierColorBin::getFreePhysicalBarrierIndexInBin(size_t virtualBarrierId, BinType binType,
                                                                               BarrierGraphInfo& BarrierGraphInfo) {
    auto blacklist = getBlacklistForBarrier(virtualBarrierId, binType, BarrierGraphInfo);

    const auto& barrierExecutionStep = _barrierFirstExecutionStep[virtualBarrierId];
    for (const auto& item : _barrierSelectionExecutionStep[binType] | indexed) {
        const auto& index = item.index();
        const auto& step = item.value();
        if (step > barrierExecutionStep) {
            blacklist.set(index);
        }
    }

    const auto& assignedBarriers = _assignedBarriers[binType];
    const auto& virtualBarrierSelected = _virtualBarrierSelection[binType];
    auto& barrierSelectionCount = _barrierSelectionCount[binType];

    const size_t binSize = assignedBarriers.size();
    size_t candidateIdEnd = binSize;
    VPUX_THROW_UNLESS(barrierSelectionCount.size() == binSize, "Not equal for selection count size and bin size");
    VPUX_THROW_UNLESS(blacklist.size() == binSize, "Not equal for black list size and bin size");
    std::map<size_t, std::deque<size_t>> counts;
    for (auto candidate : irange(candidateIdEnd)) {
        if (!blacklist.test(candidate) && (!assignedBarriers[candidate].size())) {
            // count candidate if it isn't been assigned
            counts[barrierSelectionCount[candidate]].push_back(candidate);
        }
    }
    if (counts.empty()) {
        return std::nullopt;
    }

    // find the physical barrier which has been mapped for least times. If there are multiple barriers which satisfy it,
    // select the one which currently is mapped to the least virtual barrier.

    _log.trace("try to find best physical barrier candidate for virtual barrier {0} from {1}", virtualBarrierId,
               to_small_vector(counts.begin()->second));
    auto barrier = counts.begin()->second.front();
    auto minVid = virtualBarrierId;
    for (auto& bar : counts.begin()->second) {
        if (virtualBarrierSelected[bar] < static_cast<int64_t>(minVid)) {
            minVid = virtualBarrierSelected[bar];
            barrier = bar;
        }
    }
    barrierSelectionCount[barrier]++;

    return barrier;
}

size_t VPURT::BarrierColorBin::getBarrierSelectionCountForBinType(BinType type) {
    return std::accumulate(_barrierSelectionCount[type].begin(), _barrierSelectionCount[type].end(), 0);
}

SmallVector<VPURT::BarrierColorBin::BinType> VPURT::BarrierColorBin::getBinTypeWithPriority(BinType type) {
    SmallVector<BinType> binTypeWithPriority;
    binTypeWithPriority.push_back(type);
    VPUX_THROW_WHEN(_assignedBarriers.empty(), "Assigned Barrier not init");
    SmallVector<std::pair<BinType, size_t>> counts;

    for (const auto& [binType, _] : _assignedBarriers) {
        auto usedCount = getBarrierSelectionCountForBinType(binType);
        counts.push_back({binType, usedCount});
    }
    llvm::sort(counts, [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });
    for (const auto& [binType, _] : counts) {
        if (binType != type) {
            binTypeWithPriority.push_back(binType);
        }
    }
    return binTypeWithPriority;
}

llvm::BitVector VPURT::BarrierColorBin::getBlacklistForBarrier(size_t virtualBarrierId, BinType binType,
                                                               BarrierGraphInfo& BarrierGraphInfo) {
    // barrier' physical id can not be same as its parents or children
    const auto binSize = _assignedBarriers[binType].size();
    llvm::BitVector blacklist(binSize);
    auto addInBlackList = [&](const BarrierGraphInfo::BarrierSet& barriers) {
        for (auto bar : barriers) {
            const auto& barBinType = _barrierBinType[bar];
            if (barBinType != binType) {
                continue;
            }
            if (_barrierVirtualToPhysicalMapping[bar] != INVALID_BARRIER_PID) {
                auto iter = llvm::find(_physicalBarrierList[binType], _barrierVirtualToPhysicalMapping[bar]);
                VPUX_THROW_WHEN(iter == _physicalBarrierList[binType].end(), "Can not find physical barrier {0} in bin",
                                _barrierVirtualToPhysicalMapping[bar]);
                auto indexInBin = std::distance(_physicalBarrierList[binType].begin(), iter);
                blacklist.set(indexInBin);
            }
        }
    };
    auto preBarriers = BarrierGraphInfo.getParentBarrier(virtualBarrierId);
    addInBlackList(preBarriers);

    if (preBarriers.empty()) {
        auto postBarriers = BarrierGraphInfo.getChildrenBarrier(virtualBarrierId);
        addInBlackList(postBarriers);
    }
    BarrierGraphInfo::BarrierSet barriers;
    for (auto bar : preBarriers) {
        llvm::set_union(barriers, BarrierGraphInfo.getParentBarrier(bar));
    }
    addInBlackList(barriers);
    return blacklist;
}
