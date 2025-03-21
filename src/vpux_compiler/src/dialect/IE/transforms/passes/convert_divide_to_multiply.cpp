//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDIVIDETOMULTIPLY
#define GEN_PASS_DEF_CONVERTDIVIDETOMULTIPLY
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class ConvertDivideToMultiplyPass final : public IE::impl::ConvertDivideToMultiplyBase<ConvertDivideToMultiplyPass> {
public:
    explicit ConvertDivideToMultiplyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Value createNewFQ(mlir::PatternRewriter& rewriter, IE::FakeQuantizeOp origFqOp, mlir::Value input, float inLow,
                        float inHigh, float outLow, float outHigh) {
    auto inLowConstType =
            origFqOp.getInputLow().getDefiningOp<Const::DeclareOp>().getType().cast<mlir::RankedTensorType>();
    auto outLowConstType =
            origFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>().getType().cast<mlir::RankedTensorType>();

    rewriter.setInsertionPoint(origFqOp);
    auto newInLowConst = Const::createConst(rewriter, origFqOp->getLoc(), inLowConstType, ArrayRef(inLow));
    auto newInHighConst = Const::createConst(rewriter, origFqOp->getLoc(), outLowConstType, ArrayRef(inHigh));
    auto newOutLowConst = Const::createConst(rewriter, origFqOp->getLoc(), inLowConstType, ArrayRef(outLow));
    auto newOutHighConst = Const::createConst(rewriter, origFqOp->getLoc(), outLowConstType, ArrayRef(outHigh));

    auto newFakeQuantizeOp = rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
            origFqOp, origFqOp.getType(), input, newInLowConst, newInHighConst, newOutLowConst, newOutHighConst,
            origFqOp.getLevelsAttr(), origFqOp.getLowFpTypeAttr(), origFqOp.getAutoBroadcastAttr());
    return newFakeQuantizeOp.getOutput();
}

mlir::FailureOr<mlir::Value> replaceWithNewFakeQuantizeOp(mlir::PatternRewriter& rewriter, Const::DeclareOp constOp,
                                                          IE::FakeQuantizeOp fakeQuantize) {
    const auto inLowSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getInputLow());
    const auto inHighSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getInputHigh());
    const auto outLowSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getOutputLow());
    const auto outHighSplat = vpux::Const::template getSplatValue<float>(fakeQuantize.getOutputHigh());

    if (mlir::failed(inLowSplat) || mlir::failed(inHighSplat) || mlir::failed(outLowSplat) ||
        mlir::failed(outHighSplat)) {
        return mlir::failure();
    }

    const auto inLowVal = inLowSplat.value();
    const auto inHighVal = inHighSplat.value();
    const auto outLowVal = outLowSplat.value();
    const auto outHighVal = outHighSplat.value();

    auto newCstAttr = constOp.transformContentAttr().scalarMultInverse().get();
    rewriter.setInsertionPoint(constOp);
    auto newCstOp = rewriter.create<Const::DeclareOp>(constOp->getLoc(), newCstAttr.getType(), std::move(newCstAttr));

    // Get inversed values for new input low/high params for FQ
    const auto inversedConstContent = newCstOp.getContent();
    const auto inversedConstContentVals = inversedConstContent.getValues<float>();

    // New FQ range should contain 0 so new params are calculated as:
    // newInputLow = min(0, min(inversedConstContentVals))
    // newInputHigh = max(0, max(inversedConstContentVals))
    // e.g. origInput = [1, 2, 3] inLow = 0, inHigh = 3
    // inversedInput = [1; 0.5; 0.33] newInLow = 0, newInHigh = 1
    const auto minInversedVal = std::min_element(inversedConstContentVals.begin(), inversedConstContentVals.end());
    const auto maxInversedVal = std::max_element(inversedConstContentVals.begin(), inversedConstContentVals.end());
    const auto newInLowVal = std::min(0.f, *minInversedVal);
    const auto newInHighVal = std::max(0.f, *maxInversedVal);

    // To get new output range we have to quantize original values, inverse them and find min/max:
    // quantizedConstContentVal[Max|Min] = (val[Max|Min] - in_low) / (in_high - in_low) * (out_high - out_low) +
    // out_low
    // inversedQuantizedConstContentVal[Max|Min] = 1 / quantizedConstContentVal[Min|Max]
    // newInputLow = min(0, min(inversedQuantizedConstContentValMin))
    // newInputHigh = max(0, max(inversedQuantizedConstContentValMax))
    auto getQuantizedMinMaxVal = [&](float inversedVal) {
        const float origVal = 1.f / inversedVal;
        const float quantizedVal = (origVal - inLowVal) / (inHighVal - inLowVal) * (outHighVal - outLowVal) + outLowVal;
        VPUX_THROW_WHEN(quantizedVal == 0.f, "Cannot divide by zero");
        return 1.f / quantizedVal;
    };

    const auto newOutLowVal = std::min(0.f, getQuantizedMinMaxVal(*minInversedVal));
    const auto newOutHighVal = std::max(0.f, getQuantizedMinMaxVal(*maxInversedVal));

    return createNewFQ(rewriter, fakeQuantize, newCstOp.getOutput(), newInLowVal, newInHighVal, newOutLowVal,
                       newOutHighVal);
}

class ConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    ConstRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Checks if all users of the operation are Divide ops,
// they operate on floating point type and the operation is the second input of the Divide ops
bool isOpEligible(mlir::Operation* operation) {
    return llvm::all_of(operation->getUsers(), [&](auto user) {
        if (auto divideOp = mlir::dyn_cast<IE::DivideOp>(user); divideOp != nullptr) {
            const auto elementType = divideOp.getOutput().getType().getElementType();
            const bool floatDivision = mlir::isa<mlir::FloatType>(elementType);
            return divideOp.getInput2().getDefiningOp() == operation && floatDivision;
        }
        return false;
    });
}

// Replaces this pattern:
//
//            const.Declare
//           |      |   ..  |
//  IE.Divide  IE.Divide .. IE.Divide
//
// with
//
//       const.Declare' [#const.ScalarMultInverse]
//         |           |     ..     |
//  IE.Multiply  IE.Multiply ..  IE.Multiply
//
mlir::LogicalResult ConstRewriter::matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Const.Declare at '{0}'", origOp->getLoc());
    if (!isOpEligible(origOp)) {
        _log.trace("Ignore: IE.Const op has no IE.Divide user that is a floating point division and has IE.Const "
                   "as a second input");
        return mlir::failure();
    }
    auto newCstAttr = origOp.transformContentAttr().scalarMultInverse().get();
    auto newCstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newCstAttr.getType(), std::move(newCstAttr));

    for (auto userOp : origOp->getUsers()) {
        auto divideOp = mlir::cast<IE::DivideOp>(userOp);
        rewriter.replaceOpWithNewOp<IE::MultiplyOp>(divideOp, divideOp.getInput1(), newCstOp,
                                                    divideOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr,
                                                    nullptr);
    }
    rewriter.eraseOp(origOp);
    return mlir::success();
}

class ConstFakeQuantizeRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    ConstFakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isConstFakeQuantizeEligible(Const::DeclareOp origOp) {
    if (!origOp->hasOneUse()) {
        return false;
    }

    auto fakeQuantize = mlir::dyn_cast<IE::FakeQuantizeOp>(*origOp->getUsers().begin());
    if (fakeQuantize == nullptr) {
        return false;
    }
    // Checks if origOp is actually the first input of FakeQuantize
    if (fakeQuantize.getInput().getDefiningOp() != origOp) {
        return false;
    }

    return isOpEligible(fakeQuantize);
}

// Replaces this pattern:
//
//            const.Declare
//                 |
//           IE.FakeQuantize
//          |      |     .. |
//  IE.Divide  IE.Divide .. IE.Divide
//
// with
//
//           const.Declare' [#const.ScalarMultInverse]
//                     |
//           IE.FakeQuantize' (with new in/out low/high params)
//           |         |     ..    |
//  IE.Multiply  IE.Multiply ..  IE.Multiply
//
mlir::LogicalResult ConstFakeQuantizeRewriter::matchAndRewrite(Const::DeclareOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Const.Declare at '{0}'", origOp->getLoc());
    if (!isConstFakeQuantizeEligible(origOp)) {
        return mlir::failure();
    }

    // Casting is safe because it was already checked
    auto fakeQuantize = mlir::cast<IE::FakeQuantizeOp>(*origOp->getUsers().begin());
    const auto maybeNewFq = replaceWithNewFakeQuantizeOp(rewriter, origOp, fakeQuantize);
    if (mlir::failed(maybeNewFq)) {
        _log.trace("Ignore: IE.FakeQuantize input/output low/high params are not splat values");
        return mlir::failure();
    }

    const auto newFqOp = maybeNewFq.value();
    for (auto userOp : newFqOp.getDefiningOp()->getUsers()) {
        auto divideOp = mlir::cast<IE::DivideOp>(userOp);
        // Insertion point was changed in replaceWithNewFakeQuantizeOp, we have to reset it manually here
        rewriter.setInsertionPoint(divideOp);
        rewriter.replaceOpWithNewOp<IE::MultiplyOp>(divideOp, divideOp.getInput1(), newFqOp,
                                                    divideOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr,
                                                    nullptr);
    }
    rewriter.eraseOp(origOp);
    return mlir::success();
}

void ConvertDivideToMultiplyPass::safeRunOnFunc() {
    static_assert(IE::DivideOp::hasTrait<IE::EltwiseOp>(),
                  "This pass cannot replace IE.Divide with IE.Multiply when division is not element-wise: the "
                  "reciprocal must be calculated differently");

    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConstRewriter>(&ctx, _log);
    patterns.add<ConstFakeQuantizeRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDivideToMultiplyPass(Logger log) {
    return std::make_unique<ConvertDivideToMultiplyPass>(log);
}
