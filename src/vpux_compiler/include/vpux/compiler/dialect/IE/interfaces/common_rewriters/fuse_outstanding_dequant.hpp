//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

class DequantizeWithNCEReLUxRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeWithNCEReLUxRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
        setDebugName("DequantizeWithNCEReLUxRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class DequantizeWithNCEClampRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeWithNCEClampRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
        setDebugName("DequantizeWithNCEClampRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
