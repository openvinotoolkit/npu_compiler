//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// SplitFakeQuantPass
//

class SplitFakeQuantPass final : public IE::SplitFakeQuantBase<SplitFakeQuantPass> {
public:
    explicit SplitFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class UseQuantDequant;
    class UseConstDequant;

private:
    void safeRunOnFunc() final;
};

//
// UseQuantDequant
//

class SplitFakeQuantPass::UseQuantDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseQuantDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitFakeQuantPass::UseQuantDequant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize Operation '{0}'", origOp->getLoc());
    auto innerLog = _log.nest();

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        innerLog.trace("Got non constant parameters");
        return mlir::failure();
    }

    const auto outLowAttr = outLowConst.getContent();
    const auto outHighAttr = outHighConst.getContent();

    if (inLowConst.getContent() != outLowAttr || inHighConst.getContent() != outHighAttr) {
        innerLog.trace("Input/output parameters mismatch");
        return mlir::failure();
    }

    innerLog.trace("Try to use quantize/dequantize pair");

    const auto realType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto qElemType = getQuantizedType(outLowConst, outHighConst, origOp.levels(), realElemType, origOp.getLoc());
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = mlir::RankedTensorType::getChecked(origOp.getLoc(), realType.getShape(), qElemType);
    if (qType == nullptr) {
        return mlir::failure();
    }

    auto quantOp = rewriter.create<mlir::quant::QuantizeCastOp>(origOp.getLoc(), qType, origOp.input());
    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, realType, quantOp.getResult());

    return mlir::success();
}

//
// UseConstDequant
//

class SplitFakeQuantPass::UseConstDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseConstDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitFakeQuantPass::UseConstDequant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize Operation '{0}'", origOp->getLoc());
    auto innerLog = _log.nest();

    auto inConst = origOp.input().getDefiningOp<ConstantInterface>();
    if (inConst == nullptr) {
        innerLog.trace("Got non constant input");
        return mlir::failure();
    }

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        innerLog.trace("Got non constant parameters");
        return mlir::failure();
    }

    const auto inLowAttr = inLowConst.getContent();
    const auto inHighAttr = inHighConst.getContent();

    if (!inLowAttr.isSplat() || !inHighAttr.isSplat()) {
        innerLog.trace("Input min/max are not splat values");
        return mlir::failure();
    }

    // TODO: should we check the inLowAttr/inHighAttr values some how?

    innerLog.trace("Try to use constant dequantize");

    const auto realType = inConst.getActualType();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto qElemType = getQuantizedType(outLowConst, outHighConst, origOp.levels(), realElemType, origOp.getLoc());
    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = mlir::RankedTensorType::getChecked(origOp.getLoc(), realType.getShape(), qElemType);
    if (qType == nullptr) {
        return mlir::failure();
    }

    auto newInOp = rewriter.create<IE::ConstantOp>(inConst->getLoc(), qType, inConst.getContent());
    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, origOp.getType(), newInOp.output());

    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::ConstantOp>();
    target.addLegalOp<mlir::quant::QuantizeCastOp>();
    target.addLegalOp<mlir::quant::DequantizeCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UseQuantDequant>(&ctx, _log);
    patterns.insert<UseConstDequant>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitFakeQuantPass(Logger log) {
    return std::make_unique<SplitFakeQuantPass>(log);
}
