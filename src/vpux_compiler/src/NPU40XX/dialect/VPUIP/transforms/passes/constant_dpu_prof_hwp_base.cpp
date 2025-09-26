//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/profiling/common.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_CONSTANTDPUPROFHWPBASE
#define GEN_PASS_DEF_CONSTANTDPUPROFHWPBASE
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

//
//  ConstantDpuProfHwpBasePass
//

class ConstantDpuProfHwpBasePass final :
        public VPUIP::arch40xx::impl::ConstantDpuProfHwpBaseBase<ConstantDpuProfHwpBasePass> {
public:
    explicit ConstantDpuProfHwpBasePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConstantDpuProfHwpBasePass::safeRunOnFunc() {
    auto func = getOperation();
    auto* ctx = func->getContext();
    mlir::OpBuilder builder(func.getBody());

    func->walk([&](VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
        auto profBuffer = nceClusterTaskOp.getProfilingData();

        if (profBuffer == nullptr) {
            return;
        }
        _log.trace("Update workloadIds for NCETask '{0}'", nceClusterTaskOp->getLoc());

        auto profDeclBuff = profBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_WHEN(profDeclBuff == nullptr, "Profiling buffer of '{0}' is not defined by DeclareBufferOp",
                        nceClusterTaskOp->getLoc());

        // Get how workload offsets need to be adjusted and set base address to 0
        auto workloadIdOffset =
                static_cast<int64_t>(profDeclBuff.getByteOffset() / VPUIP::HW_DPU_PROFILING_SIZE_BYTES_40XX);

        if (!profBuffer.hasOneUse()) {
            // If buffer is shared need to clone it to not interfere with
            // other usages (for example DMA)
            builder.setInsertionPointAfter(profDeclBuff);
            auto* newOp = builder.clone(*profDeclBuff);
            profDeclBuff = mlir::dyn_cast<VPURT::DeclareBufferOp>(newOp);
            nceClusterTaskOp.getProfilingDataMutable().assign(profDeclBuff.getBuffer());
        }
        profDeclBuff.setByteOffsetAttr(vpux::getIntAttr(ctx, 0));

        // Workaround: The dummy DPU task injected by insert-delay-dpu-variant will get the same
        // workload_id as the next DPU task in the same NCEClusterTask. This is intentional and makes the dummy task
        // write its profiling data in the buffer that's been allocated for the next task, with the next (non-dummy) DPU
        // task overwriting the dummy task's profiling data.
        // TODO: E#160727
        nceClusterTaskOp.walk([&](VPUIP::DPUTaskOp dpuTaskOp) {
            VPUX_THROW_UNLESS(dpuTaskOp.getWorkloadId().has_value(),
                              "NCEClusterTask at '{0}' does not have workload ids configured",
                              nceClusterTaskOp->getLoc());
            // Adjust workload_id to represent total offset in CMX
            auto newWorkloadId = workloadIdOffset + dpuTaskOp.getWorkloadId().value();
            dpuTaskOp.setWorkloadIdAttr(vpux::getIntAttr(ctx, newWorkloadId));
        });
    });
}

}  // namespace

//
// createConstantDpuProfHwpBasePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createConstantDpuProfHwpBasePass(Logger log) {
    return std::make_unique<ConstantDpuProfHwpBasePass>(log);
}
