//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_REORDERMPIOPS
#define GEN_PASS_DEF_REORDERMPIOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;
using namespace npu40xx;

namespace {
class ReorderMPIOpsPass : public VPUMI40XX::impl::ReorderMPIOpsBase<ReorderMPIOpsPass> {
public:
    explicit ReorderMPIOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <typename OpT, typename Functor = vpux::FuncRef<bool(OpT)>>
mlir::Operation* linearizeOps(
        mlir::func::FuncOp func, mlir::OpBuilder& builder, Functor&& condition = [](OpT) {
            return true;
        }) {
    auto ops = func.getOps<OpT>();

    mlir::Operation* lastOp = nullptr;
    for (auto op : llvm::make_early_inc_range(ops)) {
        if (!condition(op)) {
            continue;
        }

        lastOp = builder.clone(*op.getOperation());

        op.replaceAllUsesWith(lastOp);
        op.erase();

        builder.setInsertionPointAfter(lastOp);
    }

    return lastOp;
}

template <VPURT::BufferSection SEC>
bool buffSec(VPURT::DeclareBufferOp op) {
    return op.getSection() == SEC;
}

template <int ENGINE_ID>
bool engineId(VPUMI40XX::NNDMAOp op) {
    return op.getPort() == ENGINE_ID;
}

template <VPURegMapped::TaskType TASK_TYPE>
auto taskType(size_t tileIndex, size_t listIndex = 0) {
    auto condition = [tileIndex, listIndex](VPUMI40XX::DeclareTaskBufferOp operation) {
        const auto index = mlir::cast<vpux::VPURegMapped::IndexType>(operation.getIndex().getType());
        return operation.getTaskType() == TASK_TYPE && index.getTileIdx() == tileIndex &&
               index.getListIdx() == listIndex;
    };
    return condition;
}

void ReorderMPIOpsPass::safeRunOnFunc() {
    auto func = getOperation();

    auto builder = mlir::OpBuilder::atBlockBegin(&func.getBody().front());

    // when ordering these function calls take care of the dominance relationships.
    // Ideally this should be a canonicalizer-like thing, but re-ordering ops via successive pattern rewrites//
    // seems overkill.
    // Since VPUMI40XX is a highly controlled dialect in terms of the forms it can take, manual implicit good
    // ordering of what to linearize first to not break dominance should be oke.

    // order of DeclareTaskBuffer is important as it must be aligned with firmware expectations
    // tile0: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
    // tile1: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
    // ...
    for (auto tileIndex : irange(nn_public::VPU_MAX_TILES)) {
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(func, builder,
                                                     taskType<VPURegMapped::TaskType::DPUInvariant>(tileIndex));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(func, builder,
                                                     taskType<VPURegMapped::TaskType::DPUVariant>(tileIndex));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(func, builder,
                                                     taskType<VPURegMapped::TaskType::ActKernelRange>(tileIndex, 0));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(func, builder,
                                                     taskType<VPURegMapped::TaskType::ActKernelRange>(tileIndex, 1));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(
                func, builder, taskType<VPURegMapped::TaskType::ActKernelInvocation>(tileIndex, 0));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(
                func, builder, taskType<VPURegMapped::TaskType::ActKernelInvocation>(tileIndex, 1));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(func, builder,
                                                     taskType<VPURegMapped::TaskType::DMA>(tileIndex, 0));
        linearizeOps<VPUMI40XX::DeclareTaskBufferOp>(func, builder,
                                                     taskType<VPURegMapped::TaskType::DMA>(tileIndex, 1));
    }

    linearizeOps<Const::DeclareOp>(func, builder);

    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::NetworkInput>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::NetworkOutput>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::ProfilingOutput>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::DDR>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::CMX_NN>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::MAC_Accumulators>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::Register>);

    linearizeOps<VPUMI40XX::DeclareKernelTextOp>(func, builder);
    linearizeOps<VPUMI40XX::DeclareKernelEntryOp>(func, builder);
    linearizeOps<VPUMI40XX::DeclareKernelArgsOp>(func, builder);
    linearizeOps<VPUMI40XX::KernelParamsOp>(func, builder);

    linearizeOps<VPUMI40XX::ConfigureBarrierOp>(func, builder);

    linearizeOps<VPUMI40XX::ActKernelRangeOp>(func, builder);
    linearizeOps<VPUMI40XX::ActKernelInvocationOp>(func, builder);

    linearizeOps<VPUMI40XX::DPUInvariantOp>(func, builder);
    linearizeOps<VPUMI40XX::DPUVariantOp>(func, builder);

    linearizeOps<VPURegMapped::ViewTaskRangeOp>(func, builder);

    linearizeOps<VPUMI40XX::NNDMAOp>(func, builder, engineId<0>);
    linearizeOps<VPUMI40XX::NNDMAOp>(func, builder, engineId<1>);
}

}  // namespace

//
// reorderMappedInferenceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::reorderMappedInferenceOpsPass(Logger log) {
    return std::make_unique<ReorderMPIOpsPass>(log);
}
