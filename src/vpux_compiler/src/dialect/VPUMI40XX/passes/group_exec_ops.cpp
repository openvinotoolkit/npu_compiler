//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_GROUPEXECUTIONOPS
#define GEN_PASS_DEF_GROUPEXECUTIONOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

template <class PredicateT, class... Args>
struct FilterRangeTag final {
    FilterRangeTag(Args... args): pred(args...){};

    PredicateT pred;
};

struct isSecondaryTaskTypeFilter {
    isSecondaryTaskTypeFilter(VPURegMapped::TaskType taskType): taskType_(taskType) {
    }

    bool operator()(mlir::Operation* op) const {
        auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
        return taskOp && taskOp.getTaskType() == taskType_;
    }

    VPURegMapped::TaskType taskType_;
};

VPURegMapped::TaskOpInterface getNextOp(VPURegMapped::TaskOpInterface op) {
    auto users = op.getResult().getUsers();
    auto nexOpIt = llvm::find_if(users, [&op](mlir::Operation* user) {
        auto nextTask = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
        return nextTask && (nextTask.getTaskType() == op.getTaskType()) && (nextTask.getPreviousTask() == op);
    });

    op = nexOpIt != users.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nexOpIt) : nullptr;
    return op;
}

VPURegMapped::TaskOpInterface getNextUntil(VPURegMapped::TaskOpInterface start, VPURegMapped::TaskType secondary,
                                           int64_t maxPrimaryCount, int64_t maxSecondaryCount) {
    if (maxPrimaryCount == 1) {
        return start;
    }

    auto isTaskOpOfType = [&secondary](mlir::Operation* op) -> bool {
        auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
        return taskOp && taskOp.getTaskType() == secondary;
    };

    int64_t primaryCount = 1;
    int64_t secondaryCount = llvm::count_if(start.getResult().getUsers(), isTaskOpOfType);

    auto next = start;
    do {
        start = next;
        next = getNextOp(start);

        if (next) {
            secondaryCount += llvm::count_if(next.getResult().getUsers(), isTaskOpOfType);
            primaryCount++;
        }
    } while (next && (primaryCount <= maxPrimaryCount) && (secondaryCount <= maxSecondaryCount));

    return start;
}

void moveOpsIntoBlock(int idx, VPURegMapped::TaskOpInterface from, VPURegMapped::TaskOpInterface to,
                      VPURegMapped::ExecutionGroupOp group,
                      llvm::function_ref<bool(mlir::OpOperand&)> replaceCondition = nullptr) {
    auto block = &group.getTasks().front();

    for (auto token = from; token != to; token = getNextOp(token)) {
        token.getOperation()->moveBefore(block->getTerminator());
    }
    // also move the last op
    to.getOperation()->moveBefore(block->getTerminator());

    if (!group.getPreviousTaskIdx().empty()) {
        auto newArg =
                block->insertArgument(checked_cast<unsigned>(idx), from.getPreviousValue().getType(), from.getLoc());
        from.setPreviousTask(newArg);
    }

    if (replaceCondition) {
        from.getResult().replaceUsesWithIf(group.getStartIndexes()[idx], replaceCondition);
        to.getResult().replaceUsesWithIf(group.getEndIndexes()[idx], replaceCondition);
    } else {
        from.getResult().replaceAllUsesWith(group.getStartIndexes()[idx]);
        to.getResult().replaceAllUsesWith(group.getEndIndexes()[idx]);
    }

    return;
}

// TODO: ned to figure out a clean way to get barriers purely from taskOpInterface
VPUMI40XX::ExecutableTaskOpInterface getBarrieredOp(VPURegMapped::TaskOpInterface primary,
                                                    VPURegMapped::TaskOpInterface secondary) {
    if (primary.getTaskType() == VPURegMapped::TaskType::DPUInvariant) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(primary.getOperation());
    } else if (primary.getTaskType() == VPURegMapped::TaskType::ActKernelRange) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(secondary.getOperation());
    } else {
        VPUX_THROW("Unknown TaskType for pair {0} {1}", primary.getResult(), secondary.getResult());
        return nullptr;
    }

    return nullptr;
}

size_t getMetadataSize(VPURegMapped::TaskType taskType, VPU::ArchKind archKind) {
    // TODO: E109456
    return VPURegMapped::getDefaultTaskListCount(taskType, archKind) / 2;
}

void groupExecOps(VPUMI40XX::MappedInferenceOp mpi, int64_t tilesCount, const VPURegMapped::TaskType primary,
                  const VPURegMapped::TaskType secondary) {
    auto archKind = VPU::getArch(mpi.getOperation());
    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        auto startingVal = mpi.getListHead(primary, tileIdx);
        if (!startingVal)
            continue;

        mlir::OpBuilder groupBuilder(startingVal.getDefiningOp());

        auto traveler = mlir::dyn_cast_or_null<VPURegMapped::TaskOpInterface>(startingVal.getDefiningOp());
        if (!traveler)
            continue;

        auto taskOpCompare = [](mlir::Operation* lhs, mlir::Operation* rhs) {
            auto lhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(lhs);
            auto rhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(rhs);
            return lhsTask.getIndexType().getValue() < rhsTask.getIndexType().getValue();
        };

        mlir::ValueRange previosGroup;
        do {
            auto minPrimary = traveler;
            auto maxPrimary = getNextUntil(traveler, secondary, getMetadataSize(primary, archKind),
                                           getMetadataSize(secondary, archKind));

            auto minSecondaryIt = vpux::min_element(
                    minPrimary.getResult().getUsers() | details::FilterRangeTag<isSecondaryTaskTypeFilter>{secondary},
                    taskOpCompare);
            auto maxSecondaryIt = vpux::max_element(
                    maxPrimary.getResult().getUsers() | details::FilterRangeTag<isSecondaryTaskTypeFilter>{secondary},
                    taskOpCompare);

            auto minSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(*minSecondaryIt);
            auto maxSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(*maxSecondaryIt);

            auto waitBarrs = getBarrieredOp(minPrimary, minSecondary).waitBarriers();
            auto updateBarrs = getBarrieredOp(maxPrimary, maxSecondary).updateBarriers();
            maxSecondary->setAttr(VPUMI40XX::lastSecondaryTaskInExecutionGroup,
                                  mlir::UnitAttr::get(maxSecondary->getContext()));

            // Here we have implicit insert and create logic because
            // GroupOp inherits the SSA values of the first and last variant and invariant,
            // It cannot be placed anywhere, it has to "replace" the whole variants /invariants range even positionally.
            // Position of inserting point is doesn't matter here
            // We don't have restrictions that it should be after the maxVariant
            // It can also be before maxVariant, or before firstInvariant, or anywhere in-between
            groupBuilder.setInsertionPointAfter(maxSecondary.getOperation());
            auto group = groupBuilder.create<VPURegMapped::ExecutionGroupOp>(
                    maxPrimary.getLoc(), mlir::TypeRange({minPrimary.getIndexType(), minSecondary.getIndexType()}),
                    mlir::TypeRange({maxPrimary.getIndexType(), maxSecondary.getIndexType()}), previosGroup, waitBarrs,
                    updateBarrs, primary);

            previosGroup = group.getEndIndexes();

            // add the terminator
            mlir::Block* block = &group.getTasks().emplaceBlock();
            auto terminatorBuilder = mlir::OpBuilder::atBlockEnd(block);
            terminatorBuilder.create<VPURegMapped::GroupYieldOp>(
                    group.getLoc(), mlir::ValueRange({minPrimary.getResult(), minSecondary.getResult()}),
                    mlir::ValueRange({maxPrimary.getResult(), maxSecondary.getResult()}));

            // need to travel now to the next elem in chain before we move all ops into the new container
            traveler = getNextOp(maxPrimary);

            moveOpsIntoBlock(0, minPrimary, maxPrimary, group, [&group, &secondary](mlir::OpOperand& link) {
                bool isOwnerGroup = link.getOwner()->getParentOp() != group.getOperation();
                auto ownerTaskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(link.getOwner());
                bool isOwnerSecodanry = ownerTaskOp ? (ownerTaskOp.getTaskType() == secondary) : false;

                return isOwnerGroup && !isOwnerSecodanry;
            });

            moveOpsIntoBlock(1, minSecondary, maxSecondary, group, [&group](mlir::OpOperand& link) {
                return link.getOwner()->getParentOp() != group.getOperation();
            });
        } while (traveler);
    }
}

class GroupExecutionOpsPass : public VPUMI40XX::impl::GroupExecutionOpsBase<GroupExecutionOpsPass> {
public:
    explicit GroupExecutionOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void GroupExecutionOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    groupExecOps(mpi, tilesCount, VPURegMapped::TaskType::DPUInvariant, VPURegMapped::TaskType::DPUVariant);
    groupExecOps(mpi, tilesCount, VPURegMapped::TaskType::ActKernelRange, VPURegMapped::TaskType::ActKernelInvocation);

    return;
}
}  // namespace

//
// createUnrollFetchTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createGroupExecutionOpsPass(Logger log) {
    return std::make_unique<GroupExecutionOpsPass>(log);
}
