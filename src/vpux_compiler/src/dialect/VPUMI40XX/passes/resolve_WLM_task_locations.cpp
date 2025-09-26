//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/shave.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_RESOLVEWLMTASKLOCATION
#define GEN_PASS_DEF_RESOLVEWLMTASKLOCATION
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class ResolveWLMTaskLocationPass final :
        public VPUMI40XX::impl::ResolveWLMTaskLocationBase<ResolveWLMTaskLocationPass> {
public:
    explicit ResolveWLMTaskLocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    // Key: tileIdx, Value: current offset
    llvm::DenseMap<int64_t, int64_t> _offsetTrackers;
};

void ResolveWLMTaskLocationPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();
    const auto availableShaveEnginesPerTile =
            config::getAvailableExecutor(parentModule, VPU::ExecutorKind::SHAVE_ACT).getCount();

    auto archKind = config::getArch(netFunc);
    const llvm::DenseMap<VPURegMapped::TaskType, size_t> sizes = {
            {VPURegMapped::TaskType::DPUInvariant, VPU::getConstraint(netFunc, VPU::METADATA_MAX_INVARIANT_COUNT) / 2},
            {VPURegMapped::TaskType::DPUVariant, VPU::getConstraint(netFunc, VPU::METADATA_MAX_VARIANT_COUNT) / 2},
            {VPURegMapped::TaskType::ActKernelInvocation,
             VPU::getConstraint(netFunc, VPU::METADATA_MAX_KERNEL_INVOCATION_COUNT) / 2},
            {VPURegMapped::TaskType::ActKernelRange,
             VPU::getConstraint(netFunc, VPU::METADATA_MAX_KERNEL_RANGE_COUNT) / 2}};

    auto getSize = [&sizes](VPURegMapped::TaskType type) -> size_t {
        auto mapIt = sizes.find(type);
        VPUX_THROW_WHEN(mapIt == sizes.end(), "Task Type not registered");

        return mapIt->getSecond();
    };

    auto populate = [this, &netFunc, &archKind](mlir::OpBuilder builder, VPURegMapped::TaskType taskType,
                                                size_t tileIdx, size_t listIdx,
                                                size_t count) -> std::vector<mlir::Value> {
        std::vector<mlir::Value> taskBuffers;
        auto ctx = builder.getContext();
        int64_t& offsetTracker = _offsetTrackers[tileIdx];
        for (size_t i = 0; i < count; ++i) {
            auto index = VPURegMapped::IndexType::get(ctx, static_cast<uint32_t>(tileIdx), listIdx,
                                                      static_cast<uint32_t>(i));
            auto offsetAttr = mlir::IntegerAttr::get(getUInt64Type(ctx), offsetTracker);

            auto binarySize = VPUMI40XX::getTaskBinarySize(taskType, archKind);
            auto taskBuffer =
                    builder.create<VPUMI40XX::DeclareTaskBufferOp>(netFunc.getLoc(), index, taskType, offsetAttr);
            taskBuffers.push_back(taskBuffer.getResult());
            offsetTracker += static_cast<int64_t>(binarySize);
        }

        return taskBuffers;
    };

    auto solveGroupOps = [&mpi, &populate, &getSize](VPURegMapped::TaskType taskType, size_t tileIdx,
                                                     size_t listIdx = 0) -> void {
        auto listHead = mpi.getListHead(taskType, tileIdx, listIdx);
        if (!listHead) {
            return;
        }

        auto groupOp = mlir::cast<VPURegMapped::ExecutionGroupOp>(listHead.getDefiningOp());

        mlir::OpBuilder builder(listHead.getDefiningOp());
        auto groupSize = getSize(taskType);

        auto taskBuffers = populate(builder, taskType, tileIdx, listIdx, groupSize * 2);

        size_t groupCtr = 0;

        while (groupOp) {
            auto taskOps = groupOp.getOps<VPURegMapped::TaskOpInterface>();
            auto numberOfOpsInGroup = std::count_if(std::begin(taskOps), std::end(taskOps), [taskType](auto op) {
                return op.getTaskType() == taskType;
            });
            int offsetFromStart = groupSize - numberOfOpsInGroup;
            size_t taskCtr = 0;
            for (auto execTaskOp : taskOps) {
                if (execTaskOp.getTaskType() != taskType) {
                    continue;
                }
                execTaskOp.setTaskLocation(taskBuffers[groupCtr + taskCtr + offsetFromStart]);
                taskCtr++;
            }
            groupCtr = (groupCtr + groupSize) % (groupSize * 2);
            groupOp = VPUMI40XX::getNextGroup(groupOp);
        }
    };

    // Order of task descriptors in CMX are important, they must me in order
    // Order must be constructed respecting the structure presented at TaskBufferLayoutOp tblgen definition
    // DPUInvariant->DPUVariant->ActKernelRange->ActKernelInvocation
    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        solveGroupOps(VPURegMapped::TaskType::DPUInvariant, tileIdx);
        solveGroupOps(VPURegMapped::TaskType::DPUVariant, tileIdx);
        for (int64_t listIdx = 0; listIdx < availableShaveEnginesPerTile; listIdx++) {
            solveGroupOps(VPURegMapped::TaskType::ActKernelRange, tileIdx, listIdx);
            solveGroupOps(VPURegMapped::TaskType::ActKernelInvocation, tileIdx, listIdx);
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> VPUMI40XX::createResolveWLMTaskLocationPass(Logger log) {
    return std::make_unique<ResolveWLMTaskLocationPass>(log);
}
