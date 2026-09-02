//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPOWERTOMULT
#define GEN_PASS_DEF_CONVERTPOWERTOMULT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// PowerToMultRewriter
//

constexpr int64_t MAX_CONVERSION_EXPONENT = 3;

bool shouldConvertPowerOp(IE::PowerOp powerOp) {
    // Check if given PowerOp has constant single value exponent
    // with small value. If yes then such PowerOp should be converted
    // to Mult
    auto cstOp = powerOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (cstOp == nullptr) {
        return false;
    }

    // Exponent must be a scalar or tensor with all elements equal
    const auto& constAttr = cstOp.getContentAttr();
    if (!constAttr.isSplat()) {
        return false;
    }

    const auto exponent = constAttr.fold().getSplatValue<float>();
    auto isIntegerExponent = isFloatEqual(std::floor(exponent), exponent);
    auto isConversionBeneficial = (exponent >= 1.0f) && (exponent <= static_cast<float>(MAX_CONVERSION_EXPONENT));
    return isIntegerExponent && isConversionBeneficial;
}

class PowerToMultRewriter final : public mlir::OpRewritePattern<IE::PowerOp> {
public:
    PowerToMultRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PowerOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::PowerOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PowerToMultRewriter::matchAndRewrite(IE::PowerOp powerOp, mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertPowerOp(powerOp)) {
        return mlir::failure();
    }

    _log.trace("Got PowerOp for conversion to MultOp - '{0}'", powerOp->getLoc());

    auto cstOp = powerOp.getInput2().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(cstOp == nullptr, "PowerOp exponent input is not a constant");

    auto constContent = cstOp.getContentAttr().fold();

    auto exponent = checked_cast<int64_t>(constContent.getSplatValue<float>());
    VPUX_THROW_UNLESS(exponent <= MAX_CONVERSION_EXPONENT,
                      "Only exponents less than or equal to {1} are supported for conversion to multiplication. Given "
                      "exponent: {0}",
                      exponent, MAX_CONVERSION_EXPONENT);

    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto createMultiplyOp = [&](mlir::Value input1, mlir::Value input2, int64_t idx) {
        const auto newLoc = takeOpLoc(powerOp, "exponent_{0}", idx);
        return rewriter
                .create<IE::MultiplyOp>(newLoc, input1, input2, broadcastType,
                                        /*post_op=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
                                        /*inputPadding=*/nullptr)
                .getResult();
    };

    auto input1 = powerOp.getInput1();
    for (auto idx = 0; idx < exponent - 1; idx++) {
        input1 = createMultiplyOp(input1, powerOp.getInput1(), idx);
    }

    rewriter.replaceOp(powerOp, input1);

    return mlir::success();
}

//
// ConvertPowerToMultPass
//

class ConvertPowerToMultPass final : public IE::impl::ConvertPowerToMultBase<ConvertPowerToMultPass> {
public:
    explicit ConvertPowerToMultPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertPowerToMultPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PowerToMultRewriter>(&ctx, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createConvertPowerToMultPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPowerToMultPass(Logger log) {
    return std::make_unique<ConvertPowerToMultPass>(log);
}
