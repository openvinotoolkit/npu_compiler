//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

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
    explicit SetZeroOffsetWeightsTablePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SetZeroOffsetWeightsTablePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        auto weightsTable = nceOp.getWeightTable();
        if (weightsTable == nullptr) {
            return;
        }
        auto wtShape = getShape(weightsTable);
        if (wtShape.size() == DimsGroups5D::Filter::numDims) {
            // TODO: support WT reuse for Group Matmul
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
