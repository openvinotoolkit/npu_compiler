//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

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

class PowerToMultRewriter final : public mlir::OpRewritePattern<IE::PowerOp> {
public:
    PowerToMultRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PowerOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::PowerOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PowerToMultRewriter::matchAndRewrite(IE::PowerOp powerOp, mlir::PatternRewriter& rewriter) const {
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
        const auto newLoc = takeOpLoc(powerOp, StringLiteral("exponent_{0}"), idx);
        return rewriter
                .create<IE::MultiplyOp>(newLoc, input1, input2, broadcastType,
                                        /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_channels=*/nullptr,
                                        /*input_channels=*/nullptr)
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

    const auto isLegalOp = [&](IE::PowerOp powerOp) {
        // Check if given PowerOp has constant single value exponent
        // with small value. If yes then such PowerOp should be converted
        // to Mult
        if (auto cstOp = powerOp.getInput2().getDefiningOp<Const::DeclareOp>()) {
            // Exponent must be a scalar or tensor with all elements equal
            const auto& constAttr = cstOp.getContentAttr();
            if (!constAttr.isSplat()) {
                return true;
            }

            const auto exponent = constAttr.fold().getSplatValue<float>();
            auto isIntegerExponent = isFloatEqual(std::floor(exponent), exponent);
            auto isConversionBeneficial = (static_cast<int64_t>(exponent) <= MAX_CONVERSION_EXPONENT);
            if (isIntegerExponent && isConversionBeneficial) {
                return false;
            }
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::PowerOp>(isLegalOp);
    target.addLegalOp<IE::MultiplyOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PowerToMultRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPowerToMultPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPowerToMultPass(Logger log) {
    return std::make_unique<ConvertPowerToMultPass>(log);
}
