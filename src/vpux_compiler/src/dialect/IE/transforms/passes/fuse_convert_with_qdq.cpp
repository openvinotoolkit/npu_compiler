//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

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
    if (!inElemType.isInteger(8)) {
        return mlir::failure();
    }

    auto originDstType = quantizeOp.getDstElemType();
    // We don't support quantile type now
    if (mlir::isa_and_present<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(
                originDstType)) {
        const auto quantized = mlir::cast<mlir::quant::QuantizedType>(originDstType);
        if (mlir::isa<vpux::type::QuantileType>(quantized.getStorageType())) {
            return mlir::failure();
        }
    }

    if (!mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(originDstType)) {
        return mlir::failure();
    }

    // Block the fold when the Convert input signedness does not match the
    // quantized output signedness. QuantizeCast is a no-op bit reinterpretation,
    // so folding Convert(si8->f16) + Quantize(f16->u8) into QuantizeCast(si8->u8)
    // is incorrect because si8 and u8 have different value semantics for the
    // same byte pattern (e.g., byte 0xFF: si8=-1 vs u8=255).
    auto intType = mlir::cast<mlir::IntegerType>(inElemType);
    auto qType = mlir::cast<mlir::quant::QuantizedType>(originDstType);
    if (intType.isSigned() != qType.isSigned()) {
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
    if (!outElemType.isUnsignedInteger(8)) {
        return mlir::failure();
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
    auto quantizedElemType = inType.getElementType();

    // Fusion is mathematically equivalent only if the dequantize input has identity quantization parameters
    if (!IE::hasIdentityQuantizationParams(quantizedElemType)) {
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
    collectOpsAndApplyPatterns(func, std::move(patterns));
}
}  // namespace

//
// createFuseConvertWithQDQPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseConvertWithQDQPass(Logger log) {
    return std::make_unique<FuseConvertWithQDQPass>(log);
}
