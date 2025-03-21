//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPADTOCONCAT
#define GEN_PASS_DEF_CONVERTPADTOCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ReplacePadWithConstAndConcat
//

class ReplacePadWithConstAndConcat final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    ReplacePadWithConstAndConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
        setDebugName("ReplacePadWithConstAndConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp origPadOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReplacePadWithConstAndConcat::matchAndRewrite(IE::PadOp origPadOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::PadOp Operation '{0}'", origPadOp->getLoc());

    if (origPadOp.getMode() != IE::PadMode::CONSTANT) {
        return mlir::failure();
    }

    auto padsBegin = vpux::IE::extractPads(origPadOp.getPadsBeginAttrAttr(), _log);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }

    auto padsEnd = vpux::IE::extractPads(origPadOp.getPadsEndAttrAttr(), _log);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(origPadOp.getPadValueAttr().has_value(), "IE::PadOp has getPadValueAttr() == nullptr {0}",
                      origPadOp->getLoc());
    const auto padValue = origPadOp.getPadValueAttr().value().convertToDouble();

    const auto inputType = origPadOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape().raw();
    const auto outputShape = origPadOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    auto midInput = origPadOp.getInput();
    const auto padsBeginValue = padsBegin.value();
    const auto padsEndValue = padsEnd.value();
    VPUX_THROW_UNLESS(padsBeginValue.size() == inputShape.size() && padsEndValue.size() == inputShape.size(),
                      "`IE::PadOp` {0} shape size {1} mismatch with input size {2}", origPadOp.getLoc(),
                      padsBeginValue.size(), inputShape.size());

    for (const auto reversedAxis : irange(inputShape.size()) | reversed) {
        if (padsBeginValue[reversedAxis] == 0 && padsEndValue[reversedAxis] == 0) {
            continue;
        }

        SmallVector<mlir::Value> valueRange;

        auto constShape = SmallVector<int64_t>(inputShape.size(), 0);
        for (const auto& ind : irange(inputShape.size())) {
            constShape[ind] = ind < reversedAxis ? inputShape[ind] : outputShape[ind];
        }
        _log.nest().trace("Insert ConstOp convert from padsBegin index: {0}", reversedAxis);
        if (padsBeginValue[reversedAxis] != 0) {
            constShape[reversedAxis] = padsBeginValue[reversedAxis];
            valueRange.push_back(vpux::IE::createPaddingConstForConcat(
                    constShape, takeOpLoc(origPadOp, StringLiteral("pad_begin_{0}"), reversedAxis), inputType, padValue,
                    rewriter));
        }

        valueRange.push_back(midInput);

        _log.nest().trace("Insert ConstOp convert from padsEnd index: {0}", reversedAxis);
        if (padsEndValue[reversedAxis] != 0) {
            constShape[reversedAxis] = padsEndValue[reversedAxis];
            valueRange.push_back(vpux::IE::createPaddingConstForConcat(
                    constShape, takeOpLoc(origPadOp, StringLiteral("pad_end_{0}"), reversedAxis), inputType, padValue,
                    rewriter));
        }

        auto concat = rewriter.create<IE::ConcatOp>(takeOpLoc(origPadOp, StringLiteral("concat_{0}"), reversedAxis),
                                                    valueRange, reversedAxis);
        _log.nest().trace("Insert ConcatOp {0}", concat.getLoc());
        midInput = concat.getOutput();
    }

    rewriter.replaceOp(origPadOp, midInput);

    return mlir::success();
}

//
// ConvertPadToConcat
//

class ConvertPadToConcatPass final : public IE::impl::ConvertPadToConcatBase<ConvertPadToConcatPass> {
public:
    explicit ConvertPadToConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertPadToConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReplacePadWithConstAndConcat>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSupportFusePadOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPadToConcatPass(Logger log) {
    return std::make_unique<ConvertPadToConcatPass>(log);
}
