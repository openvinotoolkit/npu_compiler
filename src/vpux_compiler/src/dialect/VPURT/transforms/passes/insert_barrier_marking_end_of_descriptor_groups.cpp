//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_INSERTBARRIERTOMARKTHEENDOFDESCRIPTORGROUP
#define GEN_PASS_DEF_INSERTBARRIERTOMARKTHEENDOFDESCRIPTORGROUP
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

/*

This pass ensures existence of barriers required for managing tasks descriptors for WLM schedules.
Memory allocated for tasks descriptors is split between two halves of the buffer dedicated for descriptors.
In order to load descriptors for tasks from CurrentGroup, task descriptors from GrandParentGroup need to be replaced
and the associated tasks need to finish by that time.

If such barrier does not exist, the newly created barrier will be consumed by the last task from
ParentGroup and by next pass this dependency will be extended to insert fetch DMA operation.

               |->BAR->
               |
|  1 --> 2 --> 3   |  4 --> 5 --> 6   |  7 --> 8 --> 9   |
|------------------|------------------|------------------|
|                  |                  |                  |
| GrandParentGroup |   ParentGroup    |   CurrentGroup   |
|     BUF A        |      BUF B       |     BUF A        |

*/
namespace {

class InsertBarrierToMarkTheEndOfDescriptorGroupPass final :
        public VPURT::impl::InsertBarrierToMarkTheEndOfDescriptorGroupBase<
                InsertBarrierToMarkTheEndOfDescriptorGroupPass> {
public:
    explicit InsertBarrierToMarkTheEndOfDescriptorGroupPass(
            std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void insertBarriersForQueue(mlir::OpBuilder& builder, ExecutionGroupListMap& executionGroupListMap,
                                BarrierInfo& barrierInfo, const BlockRange& blockRange, size_t& numOfBarriersInserted,
                                VPURT::TaskQueueType queueType);
    void insertUpdateBarriersForLastTask(mlir::OpBuilder& builder, ExecutionGroupList& executionGroupList,
                                         ExecutionGroup& executionGroup, BarrierInfo& barrierInfo,
                                         const size_t& groupIdx, const BlockRange& blockRange, size_t queueId);
    void insertBarrierDependency(mlir::OpBuilder& builder, ExecutionGroupListMap& executionGroupListMap,
                                 BarrierInfo& barrierInfo, const BlockRange& blockRange, size_t& numOfBarriersInserted);

private:
    std::optional<WorkloadManagementMode> _workloadManagementMode;
    void safeRunOnFunc() final;
};

/**
 * @brief Searches for the next or previous task with barriers (update or wait).
 *
 * @details
 * Assuming we look for adjacent task with barrier for a task at task-index 5
 * When searching forward, the caller guarantees that
 * `currentTaskIdx` does not have any update barriers. Thus, the function looks
 * for tasks starting from the next task (task-index 6) and checks for either update
 * barriers or wait barriers.
 *
 * If a non DMA task on queueId with either a wait or update barrier is found, it
 * is returned; otherwise, the search continues. If no such task exists, the function returns `nullptr`.
 */
VPURT::TaskOp getAdjacentTaskWithBarriers(BarrierInfo& barrierInfo, size_t currentTaskIdx, size_t queueId) {
    auto isNonDMAOnSpecificTile = [&](mlir::Operation* op) -> bool {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
        if (taskOp != nullptr && taskOp.getExecutorKind() != config::ExecutorKind::DMA_NN) {
            size_t taskQueueId = VPURT::getTaskQueueType(taskOp, false).id;
            if (taskQueueId == queueId) {
                return true;
            }
        }
        return false;
    };

    auto currentTaskOp = barrierInfo.getTaskOpAtIndex(currentTaskIdx);
    while (currentTaskOp != nullptr) {
        mlir::Operation* nextOrPrevTaskOp = nullptr;

        nextOrPrevTaskOp = currentTaskOp->getNextNode();
        auto adjacentTaskOp = mlir::dyn_cast<VPURT::TaskOp>(nextOrPrevTaskOp);
        if (adjacentTaskOp == nullptr) {
            break;
        }

        if (!isNonDMAOnSpecificTile(nextOrPrevTaskOp)) {
            currentTaskOp = adjacentTaskOp;
            continue;
        }

        auto adjacentTaskOpIdx = barrierInfo.getIndex(adjacentTaskOp);

        if (!barrierInfo.getUpdateBarriers(adjacentTaskOpIdx).empty() ||
            !barrierInfo.getWaitBarriers(adjacentTaskOpIdx).empty()) {
            return adjacentTaskOp;
        }

        currentTaskOp = adjacentTaskOp;
    }
    return nullptr;
}

/**
 * @brief Inserts an update barrier for last task in execution group
 *
 * @details
 * This will ensure each group we get from ExecutionGroupAnalysis will have an update barrier which includes adding
 * dependencies for the task on tile 0 as well as its sibling task on other available tiles
 *
 * When inserting an update barrier for current execution group we pick the last task of next execution group as
 * consumer task this is done in order to ensure we will always have current group's descriptor replaced before the next
 * group has finished execution
 *
 *                        CurrentGroup
 *               /->BAR1->FetchDma->BAR2->\
 * |  ........ LastOp  | .............. LastOp | FirstOp .........|
 * |                   |                       |                  |
 * |  GrandParentGroup |           ParentGroup |   CurrentGroup   |
 * |      BUF A        |              BUF B    |     BUF A        |
 */

void InsertBarrierToMarkTheEndOfDescriptorGroupPass::insertUpdateBarriersForLastTask(
        mlir::OpBuilder& builder, ExecutionGroupList& executionGroupList, ExecutionGroup& executionGroup,
        BarrierInfo& barrierInfo, const size_t& groupIdx, const BlockRange& blockRange, size_t queueId) {
    auto nextExecutionGroup = executionGroupList[groupIdx + 1];
    VPUX_THROW_WHEN(nextExecutionGroup.empty(), "nextExecutionGroup is empty");
    auto newBarrierOp = VPURT::createNewBarrier(builder, barrierInfo, nullptr, nullptr, nullptr);

    size_t producerTaskIdx = executionGroup[executionGroup.size() - 1];
    size_t consumerTaskIdx = nextExecutionGroup[nextExecutionGroup.size() - 1];

    // If we cannot insert dependency between task of consecutive groups use the next available task with barriers
    if (!VPURT::inSameTaskBlock(producerTaskIdx, consumerTaskIdx, blockRange)) {
        auto nextTaskWithUpdateBarriers = getAdjacentTaskWithBarriers(barrierInfo, producerTaskIdx, queueId);
        consumerTaskIdx = barrierInfo.getIndex(nextTaskWithUpdateBarriers);
    }

    barrierInfo.addProducer(newBarrierOp, producerTaskIdx);
    barrierInfo.addConsumer(newBarrierOp, consumerTaskIdx);
}

void InsertBarrierToMarkTheEndOfDescriptorGroupPass::insertBarrierDependency(
        mlir::OpBuilder& builder, ExecutionGroupListMap& executionGroupListMap, BarrierInfo& barrierInfo,
        const BlockRange& blockRange, size_t& numOfBarriersInserted) {
    for (auto& [queueType, executionGroup] : executionGroupListMap) {
        if (queueType.type != config::ExecutorKind::SHAVE_ACT && queueType.type != config::ExecutorKind::DPU) {
            continue;
        }
        insertBarriersForQueue(builder, executionGroupListMap, barrierInfo, blockRange, numOfBarriersInserted,
                               queueType);
    }
}

void InsertBarrierToMarkTheEndOfDescriptorGroupPass::insertBarriersForQueue(
        mlir::OpBuilder& builder, ExecutionGroupListMap& executionGroupListMap, BarrierInfo& barrierInfo,
        const BlockRange& blockRange, size_t& numOfBarriersInserted, VPURT::TaskQueueType queueType) {
    auto executionGroupListForTile = executionGroupListMap[queueType];

    for (size_t groupIdx = 0; groupIdx < executionGroupListForTile.size(); ++groupIdx) {
        auto executionGroup = executionGroupListForTile[groupIdx];

        // Last Execution group can have last task without update barriers
        auto lastTaskHasUpdateBarriers = VPURT::lastTaskInGroupHasMandatoryUpdateBarrier(executionGroup, barrierInfo);
        if (!lastTaskHasUpdateBarriers && groupIdx != executionGroupListForTile.size() - 1) {
            ++numOfBarriersInserted;
            insertUpdateBarriersForLastTask(builder, executionGroupListForTile, executionGroup, barrierInfo, groupIdx,
                                            blockRange, queueType.id);
        }
    }
}

/**
 * @brief Inserts a wait and update barrier for required execution groups
 *
 * @details
 * ExecutionGroupAnalysis gives us ExecutionGroups for DP/SHV tasks
 * Group Exec ops will create actual ExecutionGroupOp such that the earliest wait barrier
 * among the tasks in a group will be used as wait barrier for the whole group
 *
 * Similarly the latest update barriers among the tasks in a group will be used as update barriers of the group
 *
 * These barriers mark start and end of ExecutionGroup. In case we do not have barriers for first/last task we can end
 * up in situation where the ExecutionGroupOp will not have either wait or update or any barriers. In such cases we
 * can't find the starting or ending point of execution group which is crucial to insert FetchOp
 *
 * This pass ensures a wait barrier for first task and an update barrier for last task of execution group
 */
void InsertBarrierToMarkTheEndOfDescriptorGroupPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    // createAddPlaceholderFetchDMAsPass inserts placeholder FetchDMAs
    if (_workloadManagementMode.has_value() &&
        _workloadManagementMode.value() == WorkloadManagementMode::FWLM_V1_PAGES) {
        return;
    }

    mlir::OpBuilder builder(netFunc);

    auto taskOps = netFunc.getOps<VPURT::TaskOp>();
    auto numOfTasks = static_cast<size_t>(std::distance(taskOps.begin(), taskOps.end()));
    if (numOfTasks == 0) {
        _log.info("Network has no tasks. Legalization skipped");
        return;
    }
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    BlockRange blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, _log, true);
    barrierInfo.buildTaskQueueTypeMap();

    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    auto listOfDPUExecutionGroups = execGroupAnalysis.getDPUExecutionGroups();

    // logExecutionGroupTasks needs some data processing which can be avoided if correct log level is not set
    if (_log.isActive(LogLevel::Trace)) {
        execGroupAnalysis.logExecutionGroupTasks(_log);
    }

    // Used for logging the inserted barrier information
    size_t numBarriersInsertedForDPUTasks = 0;
    size_t numBarriersInsertedForSHVTasks = 0;

    // Barrier Info will reorder the barriers later
    builder.setInsertionPoint(*taskOps.begin());
    insertBarrierDependency(builder, listOfDPUExecutionGroups, barrierInfo, blockRange, numBarriersInsertedForDPUTasks);

    auto listOfSWExecutionGroups = execGroupAnalysis.getActShvExecutionGroups();
    insertBarrierDependency(builder, listOfSWExecutionGroups, barrierInfo, blockRange, numBarriersInsertedForSHVTasks);

    _log.info("Inserted '{0}' barriers for DPU group and '{1}' barriers for SHV group legalization",
              numBarriersInsertedForDPUTasks, numBarriersInsertedForSHVTasks);

    // Reorder barriers in production order, this will also verify the schedule
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, _log);

    execGroupAnalysis = ExecutionGroupAnalysis(netFunc);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    VPUX_THROW_UNLESS(VPURT::verifyBarriersForTaskDescriptorFetch(barrierInfo, netFunc, _workloadManagementMode),
                      "Encountered execution group without required barrier for task descriptor fetch.");

    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}

}  // namespace

//
// createInsertBarrierToMarkTheEndOfDescriptorGroupPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass(
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<InsertBarrierToMarkTheEndOfDescriptorGroupPass>(workloadManagementMode, log);
}
