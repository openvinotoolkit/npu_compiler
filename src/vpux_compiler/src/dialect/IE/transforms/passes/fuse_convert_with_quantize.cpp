//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSECONVERTWITHQUANTIZE
#define GEN_PASS_DEF_FUSECONVERTWITHQUANTIZE
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

    auto inElemType = convertOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!inElemType.isInteger(CHAR_BIT)) {
        return mlir::failure();
    }

    const double inDataScale = 1.0;
    const int64_t inDataZP = 0;
    mlir::quant::QuantizedType dstType;
    auto originDstType = quantizeOp.getDstElemType();
    if (const auto uniformType = originDstType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        dstType = mlir::quant::UniformQuantizedType::getChecked(
                quantizeOp.getLoc(), uniformType.isSigned(), uniformType.getStorageType(),
                uniformType.getExpressedType(), inDataScale, inDataZP, uniformType.getStorageTypeMin(),
                uniformType.getStorageTypeMax());
    } else if (const auto perAxisType = originDstType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        dstType = mlir::quant::UniformQuantizedPerAxisType::getChecked(
                quantizeOp.getLoc(), perAxisType.isSigned(), perAxisType.getStorageType(),
                perAxisType.getExpressedType(), SmallVector<double>(perAxisType.getScales().size(), inDataScale),
                SmallVector<int64_t>(perAxisType.getScales().size(), inDataZP), perAxisType.getQuantizedDimension(),
                perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
    } else {
        VPUX_THROW("Unsupported Quantized Type {0}", originDstType);
    }
    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(quantizeOp, convertOp.getInput(), dstType);
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

    auto outElemType = convertOp.getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!outElemType.isUnsignedInteger(CHAR_BIT)) {
        return mlir::failure();
    }

    _log.trace("Fusing operations: '{1}' and '{2}'", dequantizeOp->getName(), convertOp->getName());

    auto originDstType = convertOp.getDstElemType();

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(convertOp, dequantizeOp.getInput(), originDstType);

    return mlir::success();
}

//
// FuseConvertWithQuantizePass
//

class FuseConvertWithQuantizePass final : public IE::impl::FuseConvertWithQuantizeBase<FuseConvertWithQuantizePass> {
public:
    explicit FuseConvertWithQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseConvertWithQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertQuantizeRewriter>(&ctx, _log);
    patterns.add<DequantizeConvertRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createFuseConvertWithQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseConvertWithQuantizePass(Logger log) {
    return std::make_unique<FuseConvertWithQuantizePass>(log);
}
