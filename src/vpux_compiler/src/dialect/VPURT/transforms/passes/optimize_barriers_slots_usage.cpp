//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/shave.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_OPTIMIZEBARRIERSSLOTSUSAGE
#define GEN_PASS_DEF_OPTIMIZEBARRIERSSLOTSUSAGE
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class OptimizeBarriersSlotsUsagePass final :
        public VPURT::impl::OptimizeBarriersSlotsUsageBase<OptimizeBarriersSlotsUsagePass> {
public:
    explicit OptimizeBarriersSlotsUsagePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeBarriersSlotsUsagePass::safeRunOnFunc() {
    auto func = getOperation();

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorsKind{VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::DPU};

    if (VPU::isFifoPerShaveEngineEnabled(func)) {
        executorsKind.insert(VPU::ExecutorKind::SHAVE_ACT);
    }

    barrierInfo.initializeTaskQueueTypeMap(executorsKind);
    barrierInfo.buildTaskQueueTypeMap();

    barrierInfo.removeRedundantBarrierProducersAndConsumers(true);

    barrierInfo.updateIR();
    barrierInfo.clearAttributes();

    VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
}
}  // namespace

//
// createOptimizeBarriersSlotsUsagePass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createOptimizeBarriersSlotsUsagePass(Logger log) {
    return std::make_unique<OptimizeBarriersSlotsUsagePass>(log);
}
