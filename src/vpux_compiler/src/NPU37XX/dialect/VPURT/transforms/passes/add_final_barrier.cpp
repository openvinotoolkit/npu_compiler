//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPURT::arch37xx {
#define GEN_PASS_DECL_ADDFINALBARRIER
#define GEN_PASS_DEF_ADDFINALBARRIER
#include "vpux/compiler/NPU37XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch37xx

using namespace vpux;

namespace {

class AddFinalBarrierPass final : public VPURT::arch37xx::impl::AddFinalBarrierBase<AddFinalBarrierPass> {
public:
    explicit AddFinalBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddFinalBarrierPass::safeRunOnFunc() {
    auto func = getOperation();

    auto findInsertPoint = [&]() {
        auto barrierOpsIt = func.getOps<VPURT::DeclareVirtualBarrierOp>();
        if (barrierOpsIt.empty()) {
            return &func.getBody().front().front();
        }
        auto numOfBarrierOp = std::distance(barrierOpsIt.begin(), barrierOpsIt.end());
        auto lastBarrierOpIter = barrierOpsIt.begin();
        for (auto i : irange(numOfBarrierOp - 1)) {
            VPUX_UNUSED(i);
            lastBarrierOpIter++;
        }
        auto lastBarrierOp = *lastBarrierOpIter;
        return lastBarrierOp.getOperation();
    };
    auto insertPoint = findInsertPoint();
    mlir::OpBuilder builder(func);
    builder.setInsertionPointAfter(insertPoint);
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), "finishing_barrier"));
    auto barrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(loc, mlir::UnitAttr::get(&getContext()));
    auto finalBarrier = barrierOp.getBarrier();

    for (auto taskQueueFirstAndLastOp : vpux::VPURT::getTaskQueuesFirstAndLastOp(func)) {
        auto& lastOpInQueue = taskQueueFirstAndLastOp.second.second;
        if (lastOpInQueue.getUpdateBarriers().empty()) {
            _log.trace("Add finishing barrier for {0}", lastOpInQueue->getLoc());
            lastOpInQueue.getUpdateBarriersMutable().assign(finalBarrier);
        }
    }
    VPURT::verifyBarrierSlots(func, _log);
}
}  // namespace

//
// createAddFinalBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch37xx::createAddFinalBarrierPass(Logger log) {
    return std::make_unique<AddFinalBarrierPass>(log);
}
