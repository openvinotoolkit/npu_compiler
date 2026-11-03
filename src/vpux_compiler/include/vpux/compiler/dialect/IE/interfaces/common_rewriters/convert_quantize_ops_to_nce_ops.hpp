//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>

#include <utility>

namespace vpux::IE {

//
// QuantizeDequantizeToAvgPool
//

template <class ConcreteOp>
class QuantizeDequantizeToAvgPool final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeDequantizeToAvgPool(mlir::MLIRContext* ctx, const std::function<bool(ConcreteOp)>& canSkipConversion,
                                Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _canSkipConversion(canSkipConversion), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(ConcreteOp)> _canSkipConversion;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult QuantizeDequantizeToAvgPool<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    if (_canSkipConversion(originOp)) {
        return mlir::failure();
    }

    const auto input = originOp.getInput();
    const auto inNDType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inElemType = inNDType.getElementType();
    if (mlir::isa_and_present<IE::DequantizeOp>(originOp) && isInt8Quantized(inElemType)) {
        _log.trace("SI8 Dequantize conversion to NCE is not supported.");
        return mlir::failure();
    }

    auto newPooling = IE::createIdentityAvgPool(originOp.getInput(), originOp.getType(), rewriter, originOp->getLoc());
    rewriter.replaceOp(originOp, newPooling->getResult(0));
    return mlir::success();
}

//
// QuantizeToAddRewriter
//

class QuantizeToAddRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToAddRewriter(mlir::MLIRContext* ctx, const std::function<bool(IE::QuantizeOp)>& canSkipConversion,
                          Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _canSkipConversion(canSkipConversion), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::QuantizeOp)> _canSkipConversion;
    Logger _log;
};

//
// DequantizeToAddRewriter
//

class DequantizeToAddRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToAddRewriter(mlir::MLIRContext* ctx, const std::function<bool(IE::DequantizeOp)>& canSkipConversion,
                            Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _canSkipConversion(canSkipConversion), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::DequantizeOp)> _canSkipConversion;
    Logger _log;
};

//
// DequantizeToDwRewriter
//

class DequantizeToDwRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToDwRewriter(mlir::MLIRContext* ctx, const std::function<bool(IE::DequantizeOp)>& canSkipConversion,
                           Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _canSkipConversion(canSkipConversion), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp DequantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::DequantizeOp)> _canSkipConversion;
    Logger _log;
};

//
// QuantizeToDwRewriter
//

class QuantizeToDwRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToDwRewriter(mlir::MLIRContext* ctx, const std::function<bool(IE::QuantizeOp)>& canSkipConversion,
                         Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _canSkipConversion(canSkipConversion), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::QuantizeOp)> _canSkipConversion;
    Logger _log;
};

//
// DequantizeToConvRewriter
//

class DequantizeToConvRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToConvRewriter(mlir::MLIRContext* ctx, const std::function<bool(IE::DequantizeOp)>& canSkipConversion,
                             Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _canSkipConversion(canSkipConversion), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::DequantizeOp)> _canSkipConversion;
    Logger _log;
};

}  // namespace vpux::IE
