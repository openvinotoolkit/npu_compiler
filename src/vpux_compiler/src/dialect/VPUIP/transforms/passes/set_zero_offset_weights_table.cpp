//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SETZEROOFFSETWEIGHTSTABLE
#define GEN_PASS_DEF_SETZEROOFFSETWEIGHTSTABLE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
using namespace VPUIP;

namespace {

//
// SetZeroOffsetWeightsTablePass
//

class SetZeroOffsetWeightsTablePass final :
        public VPUIP::impl::SetZeroOffsetWeightsTableBase<SetZeroOffsetWeightsTablePass> {
public:
    explicit SetZeroOffsetWeightsTablePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SetZeroOffsetWeightsTablePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (!config::isWeightsTableReuseEnabled(nceOp)) {
            _log.trace("Skipping relocation of weights table for reuse because the function is not supported {0}",
                       func->getLoc());
            return;
        }
        auto weightsTable = nceOp.getWeightTable();
        if (weightsTable == nullptr) {
            return;
        }
        if (nceOp.getWeightsSparsityMap() != nullptr) {
            return;
        }
        if (nceOp.getTaskType() != NCETaskType::CONV) {
            return;
        }
        _log.trace("Got '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());

        nceOp.setIsZeroOffsetWeightsTable(true);
    });
}

}  // namespace

//
// createSetZeroOffsetWeightsTablePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetZeroOffsetWeightsTablePass(Logger log) {
    return std::make_unique<SetZeroOffsetWeightsTablePass>(log);
}
