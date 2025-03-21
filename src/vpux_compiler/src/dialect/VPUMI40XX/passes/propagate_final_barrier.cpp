//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_PROPAGATEFINALBARRIER
#define GEN_PASS_DEF_PROPAGATEFINALBARRIER
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class PropagateFinalBarrierPass : public VPUMI40XX::impl::PropagateFinalBarrierBase<PropagateFinalBarrierPass> {
public:
    explicit PropagateFinalBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagateFinalBarrierPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());

    auto barrierOps = netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>();
    auto finalBarrierIt = llvm::find_if(barrierOps, [](VPUMI40XX::ConfigureBarrierOp barrOp) {
        return barrOp.getIsFinalBarrier();
    });

    VPUX_THROW_WHEN(finalBarrierIt == barrierOps.end(), "Could not find a final barrier");

    auto finalBarrier = *finalBarrierIt;
    mpi.setFinalBarrierIdAttr(builder.getI64IntegerAttr(finalBarrier.getType().getValue()));
}

}  // namespace

//
// createAddBootstrapOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createPropagateFinalBarrierPass(Logger log) {
    return std::make_unique<PropagateFinalBarrierPass>(log);
}
