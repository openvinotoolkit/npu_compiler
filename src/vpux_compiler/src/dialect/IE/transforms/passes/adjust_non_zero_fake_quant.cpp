//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTNONZEROFAKEQUANT
#define GEN_PASS_DEF_ADJUSTNONZEROFAKEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// AdjustFakeQuant
//

class AdjustFakeQuant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    AdjustFakeQuant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool doesRangeMeetRequirement(float inLowValue, float inHighValue) {
    if (inLowValue >= inHighValue) {
        return false;
    }

    if (inLowValue > 0 || inHighValue < 0) {
        float newRange;
        if (inLowValue > 0) {
            newRange = inHighValue;
        }

        if (inHighValue < 0) {
            newRange = -inLowValue;
        }

        if (newRange / (inHighValue - inLowValue) <= IE::QUANT_RANGE_RATIO) {
            return true;
        }
    }

    return false;
}

mlir::LogicalResult AdjustFakeQuant::matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), fakeQuantizeOp->getLoc());
    auto ctx = fakeQuantizeOp->getContext();

    if (!IE::isPerTensorFQ({fakeQuantizeOp})) {
        return mlir::failure();
    }

    if (!IE::hasStaticLowAndHighValues(fakeQuantizeOp)) {
        return mlir::failure();
    }
    const auto inLowValue = IE::getConst(fakeQuantizeOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
    const auto inHighValue = IE::getConst(fakeQuantizeOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
    const auto outLowValue = IE::getConst(fakeQuantizeOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
    const auto outHighValue = IE::getConst(fakeQuantizeOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];

    if (!isFloatEqual(inLowValue, outLowValue) || !isFloatEqual(inHighValue, outHighValue)) {
        return mlir::failure();
    }

    if (!doesRangeMeetRequirement(inLowValue, inHighValue)) {
        return mlir::failure();
    }

    // now the range is inLowValue > 0 or inHighValue < 0
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(fakeQuantizeOp.getInput().getType()).getElementType();
    const auto fqArgType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);
    auto zeroConstInput = IE::createFQConst(ctx, fakeQuantizeOp->getLoc(), 0.0, fqArgType, rewriter);
    auto zeroConstOutput = IE::createFQConst(ctx, fakeQuantizeOp->getLoc(), 0.0, fqArgType, rewriter);

    mlir::Value newFqOut;
    if (inLowValue > 0) {
        newFqOut = rewriter.create<IE::FakeQuantizeOp>(fakeQuantizeOp->getLoc(), fakeQuantizeOp.getInput(),
                                                       zeroConstInput, fakeQuantizeOp.getInputHigh(), zeroConstOutput,
                                                       fakeQuantizeOp.getOutputHigh(), fakeQuantizeOp.getLevelsAttr(),
                                                       fakeQuantizeOp.getLowFpTypeAttr(),
                                                       fakeQuantizeOp.getAutoBroadcastAttr())
                           .getOutput();
    } else if (inHighValue < 0) {
        newFqOut = rewriter.create<IE::FakeQuantizeOp>(
                                   fakeQuantizeOp->getLoc(), fakeQuantizeOp.getInput(), fakeQuantizeOp.getInputLow(),
                                   zeroConstInput, fakeQuantizeOp.getOutputLow(), zeroConstOutput,
                                   fakeQuantizeOp.getLevelsAttr(), fakeQuantizeOp.getLowFpTypeAttr(),
                                   fakeQuantizeOp.getAutoBroadcastAttr())
                           .getOutput();
    } else {
        VPUX_THROW("FakeQuant min max does not meet requirement");
    }

    const auto clampLowAttr = getFPAttr(ctx, inLowValue);
    const auto clampHighAttr = getFPAttr(ctx, inHighValue);

    _log.trace("[{0}] Adjust Fake quant prameter for Operation '{1}'", getDebugName(), fakeQuantizeOp->getLoc());
    rewriter.replaceOpWithNewOp<IE::ClampOp>(fakeQuantizeOp, newFqOut, clampLowAttr, clampHighAttr);
    return mlir::success();
}

//
// AdjustNonZeroFakeQuantPass
//

class AdjustNonZeroFakeQuantPass final : public IE::impl::AdjustNonZeroFakeQuantBase<AdjustNonZeroFakeQuantPass> {
public:
    explicit AdjustNonZeroFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustNonZeroFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustFakeQuant>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustNonZeroFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustNonZeroFakeQuantPass(Logger log) {
    return std::make_unique<AdjustNonZeroFakeQuantPass>(log);
}
