//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDBOOTSTRAPWORKITEMS
#define GEN_PASS_DEF_ADDBOOTSTRAPWORKITEMS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class AddBootstrapWorkItemsPass : public VPUMI40XX::impl::AddBootstrapWorkItemsBase<AddBootstrapWorkItemsPass> {
public:
    explicit AddBootstrapWorkItemsPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    WorkloadManagementMode _workloadManagementMode;
};

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps) {
    if (enquOps.size() == 0) {
        return;
    }

    auto ctx = enquOps[0].getContext();
    auto index = [&ctx](auto taskIdx) {
        return VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(taskIdx));
    };

    enquOps[0].getResult().setType(index(0));
    enquOps[0].getPreviousTaskIdxMutable().clear();

    for (size_t i = 1; i < enquOps.size(); i++) {
        auto enqu = enquOps[i];
        enqu.getResult().setType(index(i));
        enqu.getPreviousTaskIdxMutable().assign(enquOps[i - 1]);
    }

    return;
}

bool hasEnqueue(VPURegMapped::TaskOpInterface task, std::optional<int64_t> firstTaskIdxWithEnqueueDma) {
    // Check if task is enqueued by enqueue DMA (Full WLM)
    if (firstTaskIdxWithEnqueueDma.has_value()) {
        auto taskIdx = mlir::cast<VPURegMapped::IndexType>(task.getResult().getType()).getValue();
        if (taskIdx >= firstTaskIdxWithEnqueueDma.value()) {
            return true;
        }
    }

    // Check if task is enqueued by EnqueueOp (Partial WLM)
    auto users = task.getResult().getUsers();
    auto enquIt = llvm::find_if(users, [](mlir::Operation* user) {
        return mlir::isa<VPURegMapped::EnqueueOp>(user);
    });
    return enquIt != users.end();
}

int64_t addEnqueueForOp(mlir::MLIRContext* ctx, mlir::func::FuncOp netFunc, mlir::Value listHead,
                        const VPURegMapped::TaskType taskType, VPURegMapped::EnqueueOp firstEnqueueOp,
                        std::optional<int64_t> firstTaskIdxWithEnqueueDma) {
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    int64_t bootstrapWorkItems = 0;
    if (!listHead) {
        return bootstrapWorkItems;
    }

    auto curTask = mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp());
    if (!hasEnqueue(curTask, firstTaskIdxWithEnqueueDma)) {
        auto startTask = curTask;
        auto endTask = curTask;
        while (auto nextTask = VPUMI40XX::getNextOp(endTask)) {
            if (!hasEnqueue(nextTask, firstTaskIdxWithEnqueueDma)) {
                endTask = nextTask;
            } else {
                break;
            }
        }
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(0));
        auto bootstrapEnqueue = builder.create<VPURegMapped::EnqueueOp>(
                startTask->getLoc(), trivialIndexType, nullptr, nullptr, /*previousTaskIdxOnSameBarrier*/ nullptr,
                taskType, startTask->getResult(0), endTask->getResult(0));
        if (firstEnqueueOp) {
            bootstrapEnqueue.getOperation()->moveBefore(
                    mlir::cast<VPURegMapped::EnqueueOp>(firstEnqueueOp).getOperation());
        }

        bootstrapWorkItems++;
    }
    return bootstrapWorkItems;
}

void AddBootstrapWorkItemsPass::safeRunOnFunc() {
    auto ctx = &(getContext());
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();
    const auto dmaExecutorCount = config::getAvailableExecutor(parentModule, VPU::ExecutorKind::DMA_NN).getCount();

    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt.getValue();
    }

    // Check if there are any Enqueue DMAs present in the schedule
    mlir::DenseMap<VPUMI40XX::HwQueueType, int64_t> firstEnqueueDmaPerHwQueue;

    if (_workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        auto dmaTile0List0Task = mpi.getListHead(VPURegMapped::TaskType::DMA, 0, 0).getDefiningOp<VPUMI40XX::NNDMAOp>();
        do {
            auto enqueueDmaAttr = dmaTile0List0Task.getEnqueueDmaAttr();
            if (enqueueDmaAttr.has_value()) {
                auto taskType = VPUMI40XX::convertExecutorKindToExecutableTaskType(
                        enqueueDmaAttr.value().getTargetExecutorKindAttr().getValue());
                auto tileIdx = static_cast<uint32_t>(enqueueDmaAttr.value().getTileIdx().getValue().getSExtValue());
                auto listIdx = static_cast<uint32_t>(enqueueDmaAttr.value().getListIdx().getValue().getSExtValue());
                auto hwQueue = VPUMI40XX::HwQueueType{taskType, tileIdx, listIdx};

                if (firstEnqueueDmaPerHwQueue.find(hwQueue) == firstEnqueueDmaPerHwQueue.end()) {
                    auto firstTaskIdx = enqueueDmaAttr.value().getStartTaskIdx().getValue().getSExtValue();
                    firstEnqueueDmaPerHwQueue[hwQueue] = firstTaskIdx;
                    _log.trace("Found Enqueue DMA for task type {0} on tile {1}, list {2} with first task index {3}",
                               taskType, tileIdx, listIdx, firstTaskIdx);
                }
            }
            dmaTile0List0Task = VPUMI40XX::getNextOp(dmaTile0List0Task);
        } while (dmaTile0List0Task);
    }

    VPURegMapped::EnqueueOp firstEnqueue = nullptr;
    if (mpi.getWorkItemTasks()) {
        firstEnqueue = mlir::cast<VPURegMapped::EnqueueOp>(mpi.getWorkItemTasks().getDefiningOp());
    }

    VPUX_THROW_WHEN(firstEnqueue != nullptr && !firstEnqueueDmaPerHwQueue.empty(),
                    "Enqueue ops should not yet be present if there are enqueue DMAs");

    int totalNumberBootstrapWorkItems = 0;

    uint32_t shavesCountPerTile = 0;
    auto actInvosCount = parseIntArrayOfArrayAttr<int64_t>(mpi.getActKernelInvocationsCount());
    llvm::for_each(actInvosCount, [&](auto actInvosCountForTile) {
        shavesCountPerTile = std::max(shavesCountPerTile, static_cast<uint32_t>(actInvosCountForTile.size()));
    });

    uint32_t dmaCountPerTile = 0;
    auto dmasCount = parseIntArrayOfArrayAttr<int64_t>(mpi.getDmaCount());
    llvm::for_each(dmasCount, [&](auto dmasCountForTile) {
        dmaCountPerTile = std::max(dmaCountPerTile, static_cast<uint32_t>(dmasCountForTile.size()));
    });

    const mlir::DenseSet<std::pair<VPURegMapped::TaskType, uint32_t>> taskTypesWithListCountPerTile = {
            {{VPURegMapped::TaskType::DMA, dmaCountPerTile},
             {VPURegMapped::TaskType::DPUVariant, 1},
             {VPURegMapped::TaskType::ActKernelInvocation, shavesCountPerTile}}};

    for (const auto& [taskType, listCount] : taskTypesWithListCountPerTile) {
        // In VPURegMapped.Index, DMAs are represented as <tile:list:index>,
        // but they are not strictly tied to tiles. Since multiple DMA ports
        // may be used, relying only on tilesCount would miss enqueue additions
        // for DMAs (e.g., on tile 1).
        const uint32_t dmaExecutorsOrTilesToProcess =
                (taskType == VPURegMapped::TaskType::DMA) ? dmaExecutorCount : tilesCount;

        for (uint32_t listIdx = 0; listIdx < listCount; listIdx++) {
            for (uint32_t dmaExecutorOrTileIdx = 0; dmaExecutorOrTileIdx < dmaExecutorsOrTilesToProcess;
                 dmaExecutorOrTileIdx++) {
                _log.trace("Check task type {0} on list {1}, tile {2} if bootstrap work items are needed", taskType,
                           listIdx, dmaExecutorOrTileIdx);

                auto curHead = mpi.getListHead(taskType, dmaExecutorOrTileIdx, listIdx);

                std::optional<int64_t> firstTaskIdxWithEnqueueDma;
                auto hwQueue = VPUMI40XX::HwQueueType{taskType, dmaExecutorOrTileIdx, listIdx};
                if (_workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
                    auto firstTaskIdxWithEnqueueDmaIt = firstEnqueueDmaPerHwQueue.find(hwQueue);
                    if (firstTaskIdxWithEnqueueDmaIt != firstEnqueueDmaPerHwQueue.end()) {
                        firstTaskIdxWithEnqueueDma = firstTaskIdxWithEnqueueDmaIt->second;
                    }
                }

                auto bootstrapWorkItems =
                        addEnqueueForOp(ctx, netFunc, curHead, taskType, firstEnqueue, firstTaskIdxWithEnqueueDma);
                _log.nest().trace("Added {0} bootstrap work items", bootstrapWorkItems);
                totalNumberBootstrapWorkItems += bootstrapWorkItems;
            }
        }
    }

    auto enquOps = to_small_vector(netFunc.getOps<VPURegMapped::EnqueueOp>());
    if (!enquOps.empty()) {
        reindexEnqueueOps(enquOps);
        mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
        mpi.setWorkItemCount(enquOps.size());
        mpi.setBootsrapWorkItemsCountAttr(builder.getI64IntegerAttr(totalNumberBootstrapWorkItems));
    } else {
        VPUX_THROW("We expect at least one enqueue operation in the function.");
    }
}

}  // namespace

//
// createAddBootstrapWorkItemsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddBootstrapWorkItemsPass(
        WorkloadManagementMode workloadManagementMode, Logger log) {
    return std::make_unique<AddBootstrapWorkItemsPass>(workloadManagementMode, log);
}
