//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNWRAPCLUSTERTILING
#define GEN_PASS_DEF_UNWRAPCLUSTERTILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnwrapClusterTilingPass
//

class UnwrapClusterTilingPass final : public VPUIP::impl::UnwrapClusterTilingBase<UnwrapClusterTilingPass> {
public:
    explicit UnwrapClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnwrapClusterTilingPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::OpBuilder builder(&func.getBody().front().front());

    func->walk([&](VPUIP::NCEClusterTilingOp clusterOp) {
        auto innerOp = clusterOp.getInnerTaskOp();
        _log.trace("Unwrap NCEClusterTilingOp with '{0}' - '{1}'", innerOp->getName(), clusterOp->getLoc());

        mlir::IRMapping mapper;
        builder.setInsertionPointAfter(clusterOp);

        for (auto operand : innerOp->getOperands()) {
            if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                mapper.map(operand, clusterOp.getOperand(blockArg.getArgNumber()));
            }
        }
        auto newOp = builder.clone(*innerOp, mapper);

        VPUX_THROW_UNLESS(newOp->getNumResults() == clusterOp->getNumResults(),
                          "Mismatching number fo results: '{0}' != '{1}'", newOp->getNumResults(),
                          clusterOp->getNumResults());

        for (const auto& p : clusterOp->getResultTypes() | indexed) {
            newOp->getResult(checked_cast<unsigned int>(p.index())).setType(p.value());
        }

        clusterOp->replaceAllUsesWith(newOp);
        innerOp->erase();
        clusterOp->erase();
    });
}

}  // namespace

//
// createUnwrapClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnwrapClusterTilingPass(Logger log) {
    return std::make_unique<UnwrapClusterTilingPass>(log);
}
