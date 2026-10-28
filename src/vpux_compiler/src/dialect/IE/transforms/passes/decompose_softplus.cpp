//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSESOFTPLUS
#define GEN_PASS_DEF_DECOMPOSESOFTPLUS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SoftPlusDecomposition
//
// SoftPlus(x) = log(1 + exp(x)), if x < threshold (While return x directly if x >= threshold).
// And threshold = 11.0f in SW.Kernel.
//
// Decomposition is only valid when the input is statically bounded to (-inf, threshold).
//
// For example, current pattern is:
//     Abs -> Multiply(splat <= 0) -> SoftPlus
// Since Abs >= 0 and the multiplier is <= 0, then the product is <= 0 < threshold.
// Then, it could be decomposed into:
//     Abs -> Multiply(splat <= 0) -> Exp -> Add(const=1.0) -> Log.
//
// The supported patterns can be extended in the future.
//

static bool isNonPositiveViaAbsNeg(IE::SoftPlusOp softPlusOp, Logger log) {
    auto mulOp = softPlusOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (mulOp == nullptr) {
        return false;
    }

    auto checkAbsAndConst = [&](mlir::Value maybeAbs, mlir::Value maybeConst) -> bool {
        if (maybeAbs.getDefiningOp<IE::AbsOp>() == nullptr) {
            return false;
        }
        const auto splatVal = Const::getSplatValue<float>(maybeConst);
        if (mlir::failed(splatVal)) {
            return false;
        }
        if (splatVal.value() > 0.0f) {
            log.trace("Multiply const {0} is positive, skipping SoftPlusOp at '{1}'", splatVal.value(),
                      softPlusOp->getLoc());
            return false;
        }
        return true;
    };

    return checkAbsAndConst(mulOp.getInput1(), mulOp.getInput2()) ||
           checkAbsAndConst(mulOp.getInput2(), mulOp.getInput1());
}

// Central guard: return true if SoftPlus input is statically bounded to (-inf, threshold)
static bool isSoftPlusInputBounded(IE::SoftPlusOp softPlusOp, Logger log) {
    if (isNonPositiveViaAbsNeg(softPlusOp, log)) {
        log.trace("SoftPlusOp at '{0}': input bounded via Abs+Multiply(<=0)", softPlusOp->getLoc());
        return true;
    }

    // Additional cases can be added here.

    return false;
}

void decomposeSoftPlus(IE::SoftPlusOp origOp, Logger log) {
    log.trace("Decomposing SoftPlusOp at '{0}'", origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    const auto loc = origOp.getLoc();
    const auto input = origOp.getInput();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());

    // Step 1: Create ExpOp
    auto expOp = IE::ExpOp::create(builder, appendLoc(loc, "exp"), input);

    // Step 2: Create AddOp for "1 + exp(x)"
    const SmallVector<int64_t> oneShape = {1};
    const auto oneStorageType = mlir::RankedTensorType::get(oneShape, inputType.getElementType());
    auto oneConst = Const::createDenseConst<float>(builder, appendLoc(loc, "one"), oneStorageType, 1.0f);
    const auto numpyBroadcast = IE::AutoBroadcastTypeAttr::get(builder.getContext(), IE::AutoBroadcastType::NUMPY);
    auto addOp = IE::AddOp::create(builder, appendLoc(loc, "add_one"), expOp.getOutput(), oneConst, numpyBroadcast,
                                   nullptr, nullptr, nullptr, nullptr);

    // Step 3: Create LogOp for "Log(1 + exp(x))"
    auto logOp = IE::LogOp::create(builder, appendLoc(loc, "log"), addOp.getOutput());

    origOp.getOutput().replaceAllUsesWith(logOp.getOutput());
    origOp.erase();
}

//
// DecomposeSoftPlusPass
//

class DecomposeSoftPlusPass final : public IE::impl::DecomposeSoftPlusBase<DecomposeSoftPlusPass> {
public:
    explicit DecomposeSoftPlusPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DecomposeSoftPlusPass::safeRunOnFunc() {
    auto func = getOperation();
    func.walk([&](IE::SoftPlusOp origOp) {
        if (!isSoftPlusInputBounded(origOp, _log)) {
            _log.trace("Skipping SoftPlusOp at '{0}': input not statically bounded", origOp->getLoc());
            return;
        }
        decomposeSoftPlus(origOp, _log);
    });
}

}  // namespace

//
// createDecomposeSoftPlusPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeSoftPlusPass(Logger log) {
    return std::make_unique<DecomposeSoftPlusPass>(log);
}
