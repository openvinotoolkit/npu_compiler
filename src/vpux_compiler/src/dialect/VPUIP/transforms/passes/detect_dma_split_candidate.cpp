//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Support/LogicalResult.h>

#include <vpu_cost_model.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DETECTDMASPLITCANDIDATE
#define GEN_PASS_DEF_DETECTDMASPLITCANDIDATE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

using TaskToTaskConfigMap = DenseMap<VPURT::TaskOp, VPURT::TaskConfig>;
using QueueIDToTaskConfigVecMap = DenseMap<int64_t, VPURT::TaskConfigVec>;

namespace {

struct TaskCycles {
    int64_t cycleStart{0};
    int64_t cycleEnd{0};

    TaskCycles(int64_t cs, int64_t ce): cycleStart(cs), cycleEnd(ce) {
    }

    bool isOverlapping(const TaskCycles& other) {
        VPUX_THROW_WHEN(other.cycleEnd - other.cycleStart < 1, "Unexpected task cycles: cycleStart {0}, cycleEnd {1}",
                        other.cycleStart, other.cycleEnd);

        if (other.cycleEnd <= cycleStart || other.cycleStart >= cycleEnd) {
            return false;
        }

        return true;
    }
};

bool isDuplicatedBufferType(NDTypeInterface bufferType) {
    if (const auto distributedType = mlir::dyn_cast_or_null<vpux::VPUIP::DistributedBufferType>(bufferType)) {
        return VPU::isDuplicated(distributedType.getDistribution());
    }

    return false;
}

bool areSupportedTypes(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    const auto inputDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inType);
    const auto outputBufferDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outType);

    // It's supported if both are non-distributed
    if (inputDistributedType == nullptr && outputBufferDistributedType == nullptr) {
        return true;
    }

    if (inputDistributedType != nullptr && outputBufferDistributedType != nullptr) {
        return false;
    }

    // DUPLICATED distribution is supported
    return (isDuplicatedBufferType(inputDistributedType) || isDuplicatedBufferType(outputBufferDistributedType));
}

SmallVector<TaskCycles> findOverlappingTaskCycles(const VPURT::TaskConfigVec& sortedTaskConfigs,
                                                  const VPURT::TaskConfig& current) {
    SmallVector<TaskCycles> result;

    TaskCycles currTaskCycles(current.cycleStart, current.cycleStart + current.cycleCost);

    auto startIt = std::lower_bound(sortedTaskConfigs.begin(), sortedTaskConfigs.end(), current,
                                    [](const VPURT::TaskConfig& task1, const VPURT::TaskConfig& task2) {
                                        return task1.cycleStart + task1.cycleCost < task2.cycleStart;
                                    });

    if (startIt == sortedTaskConfigs.end()) {
        return result;
    }

    for (auto it = startIt; it != sortedTaskConfigs.end(); ++it) {
        auto taskConfig = *it;
        TaskCycles taskCycles(taskConfig.cycleStart, taskConfig.cycleStart + taskConfig.cycleCost);
        if (taskCycles.isOverlapping(currTaskCycles)) {
            result.push_back(taskCycles);
        }

        if (taskConfig.cycleStart > currTaskCycles.cycleEnd) {
            break;
        }
    }

    return result;
}

SmallVector<TaskCycles> mergeTaskCyclesIntervals(ArrayRef<TaskCycles> sortedIntervals) {
    if (sortedIntervals.empty()) {
        return {};
    }

    SmallVector<TaskCycles> merged;
    merged.reserve(sortedIntervals.size());

    merged.push_back(sortedIntervals[0]);

    for (size_t i = 1; i < sortedIntervals.size(); ++i) {
        if (sortedIntervals[i].cycleStart <= merged.back().cycleEnd) {
            merged.back().cycleEnd = std::max(merged.back().cycleEnd, sortedIntervals[i].cycleEnd);
        } else {
            merged.push_back(sortedIntervals[i]);
        }
    }

    return merged;
}

size_t findMaxCycleGap(ArrayRef<TaskCycles> allTaskCycles, TaskCycles currentTaskCycles) {
    auto startPoint = currentTaskCycles.cycleStart;
    auto endPoint = currentTaskCycles.cycleEnd;

    // Max gap is [startPoint, endPoint] when there's no overlapping tasks
    if (allTaskCycles.empty()) {
        return endPoint - startPoint;
    }

    int64_t maxGap = 0;
    // Considering the starting gap
    if (allTaskCycles[0].cycleStart > startPoint) {
        maxGap = allTaskCycles[0].cycleStart - startPoint;
    }

    for (size_t i = 1; i < allTaskCycles.size(); ++i) {
        auto currStart = allTaskCycles[i].cycleStart;
        auto prevEnd = allTaskCycles[i - 1].cycleEnd;
        maxGap = std::max(maxGap, currStart - prevEnd);
    }

    // Considering the ending gap
    auto lastOverlappingTaskEndPoint = allTaskCycles.back().cycleEnd;
    if (endPoint > lastOverlappingTaskEndPoint) {
        maxGap = std::max(maxGap, endPoint - lastOverlappingTaskEndPoint);
    }

    return maxGap;
}

bool compareTaskConfigChronologically(const VPURT::TaskConfig& first, const VPURT::TaskConfig& second) {
    return first.cycleStart < second.cycleStart;
}

void sortTasksAndCreateTaskMap(VPURT::TaskConfigVec& allTasks, QueueIDToTaskConfigVecMap& queueIdToTasksConfigVecMap,
                               TaskToTaskConfigMap& map) {
    std::sort(allTasks.begin(), allTasks.end(), compareTaskConfigChronologically);

    for (const auto& taskConfig : allTasks) {
        auto taskOp = taskConfig.taskOp;
        auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr) {
            continue;
        }
        const auto port = dmaOp.getPortVal();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");

        const auto portValue = port.value();
        const auto channelType = dmaOp.getChannelType();
        auto dmaQueueTaskId = getDMAQueueIdEncoding(portValue, channelType);

        auto insertedPair = queueIdToTasksConfigVecMap.insert({dmaQueueTaskId, {taskConfig}});
        if (!insertedPair.second) {
            queueIdToTasksConfigVecMap[dmaQueueTaskId].push_back(taskConfig);
        }

        map.insert({taskOp, taskConfig});
    }
}

size_t calculateSplitDMACost(VPUIP::NNDMAOp dmaOp, config::ArchKind arch,
                             const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    size_t static constexpr COST_MAX = std::numeric_limits<size_t>::max();

    const auto maybeTileDim = VPUIP::getCopyDMATilingDim(dmaOp);
    if (!maybeTileDim.has_value()) {
        return COST_MAX;
    }
    const auto tileDim = maybeTileDim.value();

    auto inputType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
    auto inElemType = inputType.getElementType();
    if (!VPU::isVPUNNSupportedElementType(inElemType)) {
        return COST_MAX;
    }

    auto outputType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getOutput().getType());
    auto inputShape = inputType.getShape();

    auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(inputType, tileDim);
    SmallVector<int64_t> splitShape = to_small_vector(inputShape);
    splitShape[tileDim.ind()] = secondPartSize;

    const auto nnType = VPU::getVPUNNElementType(inElemType);
    VPUX_THROW_UNLESS(nnType.has_value(), "Unsupported data type: '{0}'", inElemType);

    const auto nnTensor =
            VPUNN::VPUTensor({checked_cast<unsigned int>(Shape(splitShape).totalSize()), 1, 1, 1}, nnType.value());
    return costModel->DMA(VPU::getVPUDeviceType(arch), {nnTensor}, {nnTensor}, getMemoryLocation(inputType),
                          getMemoryLocation(outputType));
}

//
// DetectDMASplitCandidate
//

class DetectDMASplitCandidate final : public VPUIP::impl::DetectDMASplitCandidateBase<DetectDMASplitCandidate> {
public:
    explicit DetectDMASplitCandidate(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// DMA port is idle when there's another DMA task broadcasting data via another DMA port

// Scenario 1: , Port 1 is totally idle:
// Port 0:           |---------- DMA 0 ----------|
// Port 1: |- DMA 1 -|                           |- DMA 2 -|
// Split DMA 0 to make full use of DMA HW engine
// Port 0:           |--- DMA 0 ---|
// Port 1: |- DMA 1 -|--- DMA 0 ---|- DMA 2 -|

// Scenario 2: Port 1 is partially idle:
// Port 0: |---------- DMA 0 ----------|
// Port 1: |- DMA 1 -|
// Split DMA 0 is still beneficial
// Port 0: |--- DMA 0 ---|
// Port 1: |- DMA 1 -||--- DMA 0 ---|

// This pass collects cycle costs of all NNDMA tasks generated by inference simulator, traverses NNDMA tasks and
// analyzes if it's a split candidate:
// 1. Get cycle cost of current NNDMA task
// 2. Collect cycle costs of overlapping DMA tasks on another DMA port with the same channel type
// 3. Analyze the max idle interval of these overlapping tasks
// 4. Calculate the cost of split NNDMA with cost model
// 5. Mark the NNDMA as a split candidate if the max idle interval on another DMA port can fit the cost of split NNDMA

void DetectDMASplitCandidate::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);

    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);
    CycleCostInfo cycleCostInfo(costModel, func);

    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();
    if (dmaPortCount != 2) {
        return;
    }

    VPURT::InferenceExecutionSimulator infSim(_log, func, cycleCostInfo);
    infSim.runSim();
    auto dmaTasks = infSim.getTaskCycleConfig(VPU::ExecutorKind::DMA_NN);

    QueueIDToTaskConfigVecMap queueIdToTasksConfigVecMap;
    TaskToTaskConfigMap taskMap;
    sortTasksAndCreateTaskMap(dmaTasks, queueIdToTasksConfigVecMap, taskMap);

    func->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return;
        }

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr || dmaOp.getCompressCandidateAttr() != nullptr) {
            return;
        }

        mlir::Operation* inputOp = dmaOp.getInput().getDefiningOp();
        if (!mlir::isa_and_nonnull<VPURT::DeclareBufferOp, Const::DeclareOp>(inputOp)) {
            _log.trace("Can't split op because of unsupported source");
            return;
        }

        VPUX_THROW_UNLESS(dmaOp.getPort().has_value(), "DMA at '{0}' has no portId", dmaOp->getLoc());

        const auto dmaPort = dmaOp.getPort().value();
        const auto channelType = dmaOp.getChannelType();

        if (dmaOp.getSplitCandidateAttr() != nullptr) {
            _log.trace("DMA at '{0}' has been assigned with SplitCandidate attribute already", dmaOp->getLoc());
            return;
        }

        const auto outputBuffer = dmaOp.getOutputBuff();
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(outputBuffer.getType());
        if (!areSupportedTypes(inputType, outputType)) {
            return;
        }

        const auto targetTaskConfigIter = taskMap.find(taskOp);
        if (targetTaskConfigIter == taskMap.end()) {
            return;
        }

        // Get current task's cycle costs
        const auto taskConfig = targetTaskConfigIter->second;
        const auto cycleStart = taskConfig.cycleStart;
        const auto cycleEnd = cycleStart + taskConfig.cycleCost;
        struct TaskCycles taskCycles(cycleStart, cycleEnd);

        // Experimental value to avoid trivial DMA split
        // It's not beneficial to split tasks with cost less than 6000 cycles due to overhead
        const int64_t CYCLE_COST_THRESHOLD_TO_AVOID_TRIVIAL_SPLIT = 6000;
        if (taskConfig.cycleCost < CYCLE_COST_THRESHOLD_TO_AVOID_TRIVIAL_SPLIT) {
            return;
        }

        // Find overlapping tasks on another port with the same channel type
        auto taskQueueID = getDMAQueueIdEncoding(dmaPort == 0 ? 1 : 0, channelType);
        if (queueIdToTasksConfigVecMap.find(taskQueueID) == queueIdToTasksConfigVecMap.end()) {
            // no tasks on another port at all
            dmaOp.setSplitCandidate(true);
            return;
        }

        // For example:
        //  - Current task: [21518003, 21533019], duration 15016 on port 0
        // We may get overlapped tasks cycles on another port, like:
        //  - task 1: [21518003, 21519656] duration 1653
        //  - task 2: [21519656, 21521309] duration 1653

        // Port 0: |------------------------- DMA 0 -------------------------|
        // Port 1: |-DMA 1-||-DMA 2-|

        // overlappingTaskCycles:
        //         |-------||-------|
        // mergedOverlappingTaskCycles:
        //         |----------------|
        // maxGap:
        //                          |----------------------------------------|

        // DMA 0 will be marked as NNDMA split candidate when split DMA cost can fit in mapGap
        auto overlappingTaskCycles = findOverlappingTaskCycles(queueIdToTasksConfigVecMap[taskQueueID], taskConfig);

        auto mergedOverlappingTaskCycles = mergeTaskCyclesIntervals(overlappingTaskCycles);

        auto maxGap = findMaxCycleGap(mergedOverlappingTaskCycles, taskCycles);
        auto splitNNDMACost = calculateSplitDMACost(dmaOp, arch, costModel);
        if (maxGap < splitNNDMACost) {
            return;
        }

        dmaOp.setSplitCandidate(true);
        _log.trace("DMA at '{0}' is assigned with SplitCandidate attribute", dmaOp->getLoc());
        _log.trace("Current {0} - {1}, duration {2}, maxGap {3} splitNNDMACost {4}", cycleStart, cycleEnd,
                   taskConfig.cycleCost, maxGap, splitNNDMACost);
        for (const auto& cycles : overlappingTaskCycles) {
            _log.trace("Overlapping task {0} - {1}, duration {2}", cycles.cycleStart, cycles.cycleEnd,
                       cycles.cycleEnd - cycles.cycleStart);
        }
    });
    _log.trace("Done");
}

}  // namespace

//
// createDetectDMASplitCandidatePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDetectDMASplitCandidatePass(Logger log) {
    return std::make_unique<DetectDMASplitCandidate>(log);
}
