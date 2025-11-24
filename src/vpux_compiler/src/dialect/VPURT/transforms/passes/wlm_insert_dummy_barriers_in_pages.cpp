//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_WLMINSERTDUMMYBARRIERSINPAGES
#define GEN_PASS_DEF_WLMINSERTDUMMYBARRIERSINPAGES
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class WlmInsertDummyBarriersInPagesPass final :
        public VPURT::impl::WlmInsertDummyBarriersInPagesBase<WlmInsertDummyBarriersInPagesPass> {
public:
    explicit WlmInsertDummyBarriersInPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WlmInsertDummyBarriersInPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (config::getWorkloadManagementStatus(module) != WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForLegalization();
    auto dummyBarriersInsertionDataVec = barrierPagesSplitHandler.getDummyBarriersInsertionData();

    if (dummyBarriersInsertionDataVec.empty()) {
        return;
    }

    _log.trace("Insert {0} dummy barriers in pages", dummyBarriersInsertionDataVec.size());

    mlir::OpBuilder builder(func);

    for (const auto& dummyBarrierInsertionData : dummyBarriersInsertionDataVec) {
        auto pageInd = dummyBarrierInsertionData.pageInd;
        auto insertAfter = dummyBarrierInsertionData.insertAfter;
        auto producer = dummyBarrierInsertionData.producer;
        auto consumer = dummyBarrierInsertionData.consumer;

        auto insertionPointOp = barrierInfo.getBarrierOpAtIndex(insertAfter);

        builder.setInsertionPointAfter(insertionPointOp);
        auto newBarrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(insertionPointOp->getLoc());
        barrierInfo.addNewBarrier(newBarrierOp);
        auto newBarrierIdx = barrierInfo.getIndex(newBarrierOp);
        barrierInfo.addProducer(newBarrierIdx, producer);
        barrierInfo.addConsumer(newBarrierIdx, consumer);

        newBarrierOp.setWlmPage(pageInd);

        _log.trace("In page {0} after barrier {1} insert new barrier {2} with producer {3} and consumer {4}", pageInd,
                   insertAfter, newBarrierIdx, producer, consumer);
    }
    barrierInfo.updateIR();
    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmInsertDummyBarriersInPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createWlmInsertDummyBarriersInPagesPass(Logger log) {
    return std::make_unique<WlmInsertDummyBarriersInPagesPass>(log);
}
