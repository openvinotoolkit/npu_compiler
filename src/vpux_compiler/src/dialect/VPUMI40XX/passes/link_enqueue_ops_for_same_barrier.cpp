//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_LINKENQUEUEOPSFORSAMEBARRIER
#define GEN_PASS_DEF_LINKENQUEUEOPSFORSAMEBARRIER
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class LinkEnqueueOpsForSameBarrierPass :
        public VPUMI40XX::impl::LinkEnqueueOpsForSameBarrierBase<LinkEnqueueOpsForSameBarrierPass> {
public:
    explicit LinkEnqueueOpsForSameBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LinkEnqueueOpsForSameBarrierPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    // After Enqueue ops list is in final form configure links to enqueue ops happening
    // on the same barrier. This is needed to create links between WorkItem so that
    // they can be marked ready on same barrier event even if they are not positioned
    // one after the other
    mlir::DenseMap<size_t, mlir::Value> prevEnqOnBar;

    netFunc.walk([&](VPURegMapped::EnqueueOp enqOp) {
        auto bar = enqOp.getBarrier();
        if (bar == nullptr) {
            return;
        }

        auto barIdx = mlir::cast<VPURegMapped::IndexType>(bar.getType()).getValue();

        if (prevEnqOnBar.find(barIdx) != prevEnqOnBar.end()) {
            enqOp.getPreviousTaskIdxOnSameBarrierMutable().assign(prevEnqOnBar[barIdx]);
        }

        prevEnqOnBar[barIdx] = enqOp.getIndex();
    });
}

}  // namespace

//
// createLinkEnqueueOpsForSameBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createLinkEnqueueOpsForSameBarrierPass(Logger log) {
    return std::make_unique<LinkEnqueueOpsForSameBarrierPass>(log);
}
