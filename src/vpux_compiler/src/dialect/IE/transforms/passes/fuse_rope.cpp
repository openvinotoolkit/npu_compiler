//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEROPE
#define GEN_PASS_DEF_FUSEROPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseRoPEPass
//

class FuseRoPEPass final : public IE::impl::FuseRoPEBase<FuseRoPEPass> {
public:
    explicit FuseRoPEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

// Match pattern
// Input --------> IE.Multiply --------------------------------------------------
//       |                                                                      |
//       --> IE.StridedSlice -> IE.Multiply                                     |   -> IE.Add
//       |                                 | ---> IE.Concat ---> IE.Multiply ----
//       --> IE.StridedSlice ---------------                                              ^
//   |                                                                                    |
//   |                                                                                    |
//    -------------------------------------------------------------------------------------

void FuseRoPEPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk([&](IE::AddOp addOp) {
        auto mulOp1 = addOp.getOperand(0).getDefiningOp<IE::MultiplyOp>();
        auto mulOp2 = addOp.getOperand(1).getDefiningOp<IE::MultiplyOp>();
        if (!mulOp1 || !mulOp2) {
            return;
        }
        auto input_cos = mulOp1.getOperand(1);
        auto input_sin = mulOp2.getOperand(1);

        auto concatOp = mulOp2.getOperand(0).getDefiningOp<IE::ConcatOp>();
        if (!concatOp || concatOp.getInputs().size() != 2) {
            return;
        }
        auto mulOp3 = concatOp.getOperand(0).getDefiningOp<IE::MultiplyOp>();
        auto stridedSliceOp2 = concatOp.getOperand(1).getDefiningOp<IE::StridedSliceOp>();
        if (!mulOp3 || !stridedSliceOp2) {
            return;
        }
        auto stridedSliceOp1 = mulOp3.getOperand(0).getDefiningOp<IE::StridedSliceOp>();
        if (!stridedSliceOp1) {
            return;
        }
        auto tensorType = stridedSliceOp1->getOperand(0).getType().dyn_cast<mlir::RankedTensorType>();
        if (!tensorType || tensorType.getRank() != 4) {
            return;
        }
        auto shape = tensorType.getShape();
        if (shape[2] == 1) {
            return;
        }
        _log.trace("RoPE pattern matched for operation {0} at {1}", addOp->getName(), addOp->getLoc());
        auto builder = mlir::OpBuilder(addOp);
        auto ropeOp = builder.create<IE::RoPEOp>(appendLoc(addOp->getLoc(), "rope"), stridedSliceOp1->getOperand(0),
                                                 input_cos, input_sin);
        addOp->replaceAllUsesWith(ropeOp);
    });
}

}  // namespace

//
// createFuseRoPEPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseRoPEPass(Logger log) {
    return std::make_unique<FuseRoPEPass>(log);
}
