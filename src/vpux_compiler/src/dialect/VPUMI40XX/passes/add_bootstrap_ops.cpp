//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDBOOTSTRAPOPS
#define GEN_PASS_DEF_ADDBOOTSTRAPOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class AddBootstrapOpsPass : public VPUMI40XX::impl::AddBootstrapOpsBase<AddBootstrapOpsPass> {
public:
    explicit AddBootstrapOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
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

void addBootstrapBarriers(mlir::MLIRContext* ctx, mlir::func::FuncOp netFunc) {
    int64_t bootstrapID = 0;
    mlir::Value first;
    int64_t numberOfAvailablePhysicalBarriers = VPUIP::getNumAvailableBarriers(netFunc);
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    std::vector<bool> initialized(numberOfAvailablePhysicalBarriers, false);
    for (auto op : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(bootstrapID));
        auto pid = op.getId();
        if (initialized[pid])
            continue;

        auto bootsTrapTask = builder.create<VPUMI40XX::BootstrapOp>(op.getLoc(), trivialIndexType, op->getResult(0));
        if (bootstrapID == 0) {
            first = bootsTrapTask;
        }
        if (bootstrapID == numberOfAvailablePhysicalBarriers) {
            break;
        }
        ++bootstrapID;
        initialized[pid] = true;
    }
    if (first) {
        mpi.getBootstrapTasksMutable().assign(first);
        mpi.setBootstrapTasksCountAttr(
                builder.getI64IntegerAttr(std::min(numberOfAvailablePhysicalBarriers, bootstrapID)));
    }
}

bool hasEnqueue(VPURegMapped::TaskOpInterface task) {
    auto users = task.getResult().getUsers();
    auto enquIt = llvm::find_if(users, [](mlir::Operation* user) {
        return mlir::isa<VPURegMapped::EnqueueOp>(user);
    });
    return enquIt != users.end();
}

int64_t addEnqueueForOp(mlir::MLIRContext* ctx, mlir::func::FuncOp netFunc, mlir::Value listHead,
                        const VPURegMapped::TaskType taskType, VPURegMapped::EnqueueOp firstEnqueue) {
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    int64_t bootstrapWorkItems = 0;
    if (!listHead) {
        return bootstrapWorkItems;
    }

    auto curTask = mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp());
    if (!hasEnqueue(curTask)) {
        auto startTask = curTask;
        auto endTask = curTask;
        while (auto nextTask = VPUMI40XX::getNextOp(endTask)) {
            if (!hasEnqueue(nextTask)) {
                endTask = nextTask;
            } else {
                break;
            }
        }
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(0));
        auto bootstrapEnqueue = builder.create<VPURegMapped::EnqueueOp>(
                startTask->getLoc(), trivialIndexType, nullptr, nullptr, /*previousTaskIdxOnSameBarrier*/ nullptr,
                taskType, startTask->getResult(0), endTask->getResult(0));
        if (firstEnqueue) {
            bootstrapEnqueue.getOperation()->moveBefore(
                    mlir::cast<VPURegMapped::EnqueueOp>(firstEnqueue).getOperation());
        }

        bootstrapWorkItems++;
    }
    return bootstrapWorkItems;
}

void AddBootstrapOpsPass::safeRunOnFunc() {
    auto ctx = &(getContext());
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    addBootstrapBarriers(ctx, netFunc);

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    VPURegMapped::EnqueueOp firstEnqueue = nullptr;
    if (mpi.getWorkItemTasks()) {
        firstEnqueue = mlir::cast<VPURegMapped::EnqueueOp>(mpi.getWorkItemTasks().getDefiningOp());
    }

    int totalNumberBootstrapworkItems = 0;

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto curHead = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            totalNumberBootstrapworkItems +=
                    addEnqueueForOp(ctx, netFunc, curHead, VPURegMapped::TaskType::DMA, firstEnqueue);
        }
    }

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        auto curVariantHead = mpi.getListHead(VPURegMapped::TaskType::DPUVariant, tileIdx);
        totalNumberBootstrapworkItems +=
                addEnqueueForOp(ctx, netFunc, curVariantHead, VPURegMapped::TaskType::DPUVariant, firstEnqueue);

        auto curActKernelHead = mpi.getListHead(VPURegMapped::TaskType::ActKernelInvocation, tileIdx);
        totalNumberBootstrapworkItems += addEnqueueForOp(ctx, netFunc, curActKernelHead,
                                                         VPURegMapped::TaskType::ActKernelInvocation, firstEnqueue);
    }

    auto enquOps = to_small_vector(netFunc.getOps<VPURegMapped::EnqueueOp>());
    if (!enquOps.empty()) {
        reindexEnqueueOps(enquOps);
        mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
        mpi.setWorkItemCount(enquOps.size());
        mpi.setBootsrapWorkItemsCountAttr(builder.getI64IntegerAttr(totalNumberBootstrapworkItems));
    } else {
        VPUX_THROW("We expect at least one enqueue operation in the function.");
    }
}

}  // namespace

//
// createAddBootstrapOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddBootstrapOpsPass(Logger log) {
    return std::make_unique<AddBootstrapOpsPass>(log);
}
