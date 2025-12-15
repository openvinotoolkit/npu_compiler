//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSECONVERTWITHQDQ
#define GEN_PASS_DEF_FUSECONVERTWITHQDQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertQuantizeRewriter
//

class ConvertQuantizeRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    ConvertQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("ConvertQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertQuantizeRewriter::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto convertOp = quantizeOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr) {
        return mlir::failure();
    }

    auto inElemType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType()).getElementType();
    if (!inElemType.isInteger(CHAR_BIT)) {
        return mlir::failure();
    }

    auto originDstType = quantizeOp.getDstElemType();
    // We don't support quantile type now
    if (mlir::isa_and_nonnull<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(
                originDstType)) {
        return mlir::failure();
    }

    if (!mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(originDstType)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(quantizeOp, convertOp.getInput(), originDstType);
    return mlir::success();
}

//
// DequantizeConvertRewriter
//

class DequantizeConvertRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    DequantizeConvertRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("DequantizeConvertRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeConvertRewriter::matchAndRewrite(IE::ConvertOp convertOp,
                                                               mlir::PatternRewriter& rewriter) const {
    auto dequantizeOp = convertOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto outElemType = mlir::cast<vpux::NDTypeInterface>(convertOp.getType()).getElementType();
    if (!outElemType.isUnsignedInteger(CHAR_BIT)) {
        return mlir::failure();
    }

    _log.trace("Fusing operations: '{0}' and '{1}'", dequantizeOp->getName(), convertOp->getName());

    auto originDstType = convertOp.getDstElemType();

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(convertOp, dequantizeOp.getInput(), originDstType);

    return mlir::success();
}

//
// FuseConvertWithQDQPass
//

class FuseConvertWithQDQPass final : public IE::impl::FuseConvertWithQDQBase<FuseConvertWithQDQPass> {
public:
    explicit FuseConvertWithQDQPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseConvertWithQDQPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertQuantizeRewriter>(&ctx, _log);
    patterns.add<DequantizeConvertRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createFuseConvertWithQDQPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseConvertWithQDQPass(Logger log) {
    return std::make_unique<FuseConvertWithQDQPass>(log);
}
