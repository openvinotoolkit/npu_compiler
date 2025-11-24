//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEREDUCEMEANSQUARE
#define GEN_PASS_DEF_FUSEREDUCEMEANSQUARE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseReduceMeanSquare
//

class FuseReduceMeanSquarePass final : public IE::impl::FuseReduceMeanSquareBase<FuseReduceMeanSquarePass> {
public:
    explicit FuseReduceMeanSquarePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Match pattern
// Input -> IE.Power -> IE.ReduceMean -> IE.Sqrt -> IE.Divide -> IE.Multiply(Cst_Scale)
//                                                       ^
//   |                                                   |
//    ----------------------------------------------------

// Match pattern: Power(^2) -> ReduceMean -> Sqrt
// Replace with: ReduceMeanSquare (which computes sqrt(mean(x^2)))

mlir::Operation* getPowerOp(mlir::Operation* op) {
    // detect Pow(x,2) or Multiply(x,x)
    if (auto power = mlir::dyn_cast<IE::PowerOp>(op)) {
        auto constOp = power.getInput2().getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
            return nullptr;
        }
        const auto coefContent = constOp.getContent();
        const auto coefValue = coefContent.getSplatValue<double>();
        return (coefValue == 2.0) ? op : nullptr;
    } else if (auto mul = mlir::dyn_cast<IE::MultiplyOp>(op)) {
        return mul.getInput1() == mul.getInput2() ? op : nullptr;
    }
    return nullptr;
}

IE::ReduceMeanSquareOp createReduceMeanSquareOp(mlir::OpBuilder& builder, mlir::Value input, mlir::ArrayAttr axesAttr,
                                                mlir::UnitAttr keepDimsAttr) {
    auto loc = input.getLoc();

    return builder.create<IE::ReduceMeanSquareOp>(appendLoc(loc, "_reduce_mean_square"), input, nullptr, axesAttr,
                                                  keepDimsAttr);
}

void FuseReduceMeanSquarePass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk([&](mlir::Operation* op) {
        auto powerOp = getPowerOp(op);

        if (powerOp == nullptr) {
            return;
        }
        _log.trace("Got square operation {0} at {1}", powerOp->getName(), powerOp->getLoc());

        if (!powerOp->hasOneUse()) {
            return;
        }

        auto reduceMeanOp = mlir::dyn_cast_or_null<IE::ReduceMeanOp>(*powerOp->getUsers().begin());
        if (reduceMeanOp == nullptr) {
            return;
        }

        if (!reduceMeanOp->hasOneUse()) {
            return;
        }

        auto sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(*reduceMeanOp->getUsers().begin());
        if (sqrtOp == nullptr) {
            return;
        }

        _log.trace("ReduceMeanSquare pattern matched");

        auto axesAttr = reduceMeanOp.getAxesValueAttr();
        auto keepDimsAttr = reduceMeanOp.getKeepDimsAttr();

        auto builder = mlir::OpBuilder(sqrtOp);
        auto reduceMeanSquareOp = createReduceMeanSquareOp(builder, powerOp->getOperand(0), axesAttr, keepDimsAttr);

        // ReduceMeanSquare computes sqrt(mean(x^2)) directly
        sqrtOp->replaceAllUsesWith(reduceMeanSquareOp);
    });
}

}  // namespace

//
// createReduceMeanSquarePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseReduceMeanSquarePass(Logger log) {
    return std::make_unique<FuseReduceMeanSquarePass>(log);
}
