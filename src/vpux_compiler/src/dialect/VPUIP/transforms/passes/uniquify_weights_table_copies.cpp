//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNIQUIFYWEIGHTSTABLECOPIES
#define GEN_PASS_DEF_UNIQUIFYWEIGHTSTABLECOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UniquifyWeightsTableCopiesPass
//

class UniquifyWeightsTableCopiesPass final :
        public VPUIP::impl::UniquifyWeightsTableCopiesBase<UniquifyWeightsTableCopiesPass> {
public:
    explicit UniquifyWeightsTableCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void UniquifyWeightsTableCopiesPass::safeRunOnFunc() {
    auto func = getOperation();

    mlir::DenseSet<Const::DeclareOp> wtConstants{};
    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (!nceOp.getIsZeroOffsetWeightsTable()) {
            _log.trace("Got NCEClusterTask op without IsZeroOffsetWeightsTable set.");
            return;
        }

        auto wtCopyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(nceOp.getWeightTable().getDefiningOp());
        if (wtCopyOp == nullptr) {
            _log.trace("Got non copy WT input.");
            return;
        }
        if (!wtCopyOp->hasOneUse()) {
            // We decided to only treat the single-use case for now
            _log.trace("Got CopyOp with 0 or multiple users.");
            return;
        }

        auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(wtCopyOp->getOperand(0).getDefiningOp());
        if (cstOp == nullptr || cstOp->hasOneUse() || cstOp.use_empty()) {
            _log.trace("Got non constant input or constant input without users or single use.");
            return;
        }

        wtConstants.insert(cstOp);
    });

    for (auto constDeclare : wtConstants) {
        VPUIP::CopyOp firstCopyUser = nullptr;
        mlir::DenseSet<VPUIP::CopyOp> optimizableCopyOps{};

        for (auto user : constDeclare->getUsers()) {
            if (auto copyUser = mlir::dyn_cast<VPUIP::CopyOp>(user)) {
                if (!copyUser->hasOneUse()) {
                    // We decided to only treat the single-use case for now
                    _log.trace("Got CopyOp with 0 or multiple users.");
                    continue;
                }
                if (auto nceClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(*copyUser->user_begin())) {
                    if (!nceClusterTaskOp.getIsZeroOffsetWeightsTable()) {
                        continue;
                    }
                }
                if (firstCopyUser == nullptr) {
                    firstCopyUser = copyUser;
                }
                if (copyUser->isBeforeInBlock(firstCopyUser)) {
                    firstCopyUser = copyUser;
                }
                if (copyUser.getType() != firstCopyUser.getType()) {
                    continue;
                }
                optimizableCopyOps.insert(copyUser);
            }
        }
        for (auto copyOp : optimizableCopyOps) {
            if (copyOp == firstCopyUser) {
                continue;
            }
            copyOp->replaceAllUsesWith(firstCopyUser);
            copyOp->erase();
        }
    }
}

}  // namespace

//
// createUniquifyWeightsTableCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUniquifyWeightsTableCopiesPass(Logger log) {
    return std::make_unique<UniquifyWeightsTableCopiesPass>(log);
}
