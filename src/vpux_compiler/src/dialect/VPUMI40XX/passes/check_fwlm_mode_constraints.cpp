//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_CHECKFWLMMODECONSTRAINTS
#define GEN_PASS_DEF_CHECKFWLMMODECONSTRAINTS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class CheckFWLMModeConstraintsPass :
        public VPUMI40XX::impl::CheckFWLMModeConstraintsBase<CheckFWLMModeConstraintsPass> {
public:
    explicit CheckFWLMModeConstraintsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isSHVEnqueueDMA(VPUMI40XX::NNDMAOp dmaOp) {
    auto declBuff = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.getOutputBuffs()[0].getDefiningOp());
    if (!declBuff) {
        return false;
    }
    const auto offset = declBuff.getByteOffset() - VPUMI40XX::NNCMX_SHV_CMX_CTRL_BASE;
    return llvm::is_contained(VPUMI40XX::SHV_FIFO_OFFSETS, offset);
}

bool isDPUEnqueueDMA(VPUMI40XX::NNDMAOp dmaOp) {
    auto declBuff = mlir::dyn_cast<VPURT::DeclareBufferOp>(dmaOp.getOutputBuffs()[0].getDefiningOp());
    if (!declBuff) {
        return false;
    }
    const auto offset = declBuff.getByteOffset() - VPUMI40XX::NNCMX_DPU_CMX_CTRL_BASE;
    return llvm::is_contained(VPUMI40XX::DPU_FIFO_OFFSETS, offset);
}

void CheckFWLMModeConstraintsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    if (mpi.getWorkloadManagementBarrierProgrammingMode() !=
        VPURegMapped::WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED) {
        VPUX_THROW("Unsupported execution mode detected");
    }

    size_t enqueueDMACounter = 0;

    size_t barrierOpsWithRealPageId = 0;

    bool isThereShaveorDPUTask = false;

    for (auto& op : netFunc.getOps()) {
        if (auto dmaOp = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(op)) {
            if (isSHVEnqueueDMA(dmaOp) || isDPUEnqueueDMA(dmaOp)) {
                enqueueDMACounter++;
            }
        } else if (auto enqueueOp = mlir::dyn_cast<VPURegMapped::EnqueueOp>(op)) {
            VPUX_THROW_WHEN(enqueueOp.getTaskType() != VPURegMapped::TaskType::DMA, "Unsupported type for enqueue op");
        } else if (auto barOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(op)) {
            auto pageNum = barOp.getWlmPage().value_or(-1);
            if (pageNum >= 0) {
                barrierOpsWithRealPageId++;
            }
        } else if (auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op)) {
            if (taskOp.getTaskType() == VPURegMapped::TaskType::ActKernelInvocation ||
                taskOp.getTaskType() == VPURegMapped::TaskType::DPUVariant) {
                isThereShaveorDPUTask = true;
            }
        }
    }

    VPUX_THROW_WHEN(barrierOpsWithRealPageId == 0, "No barrier with real page ID");

    if (isThereShaveorDPUTask) {
        VPUX_THROW_WHEN(enqueueDMACounter == 0, "Expecting to have enqueue DMAs in presence of SHAVE or DPU tasks");
    }
}

}  // namespace

//
// createDumpStatisticsOfWlmOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createCheckFWLMModeConstraintsPass(Logger log) {
    return std::make_unique<CheckFWLMModeConstraintsPass>(log);
}
