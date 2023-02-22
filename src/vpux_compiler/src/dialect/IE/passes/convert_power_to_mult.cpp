//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// PowerToMultRewriter
//

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

    auto cstOp = powerOp.input2().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(cstOp == nullptr, "PowerOp exponent input is not a constant");

    auto constAttr = cstOp.contentAttr().fold();

    auto exponent = constAttr.getSplatValue<double>();
    VPUX_THROW_UNLESS(exponent == 2, "For now only exponent equal to 2 is supported for conversion to multiplication");

    // TODO: Exponents of N > 2 could also be replaced by a chain/tree of multiplication
    // operations. Such transformation would require further testing to understand from
    // which point it is more performant to leave it as a IE.Power later handled UPA

    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    rewriter.replaceOpWithNewOp<IE::MultiplyOp>(powerOp, powerOp.input1(), powerOp.input1(), broadcastType,
                                                /*post_op=*/nullptr);

    return mlir::success();
}

//
// ConvertPowerToMultPass
//

class ConvertPowerToMultPass final : public IE::ConvertPowerToMultBase<ConvertPowerToMultPass> {
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
        if (auto cstOp = powerOp.input2().getDefiningOp<Const::DeclareOp>()) {
            // Exponent constant must be a scalar or tensor with
            // all elements equal
            auto constAttr = cstOp.contentAttr().fold();
            if (!constAttr.isSplat()) {
                return true;
            }

            // Currently only convert power to multiply operation
            // only when exponent equals to 2
            if (constAttr.getSplatValue<double>() == 2) {
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

    auto func = getFunction();
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
