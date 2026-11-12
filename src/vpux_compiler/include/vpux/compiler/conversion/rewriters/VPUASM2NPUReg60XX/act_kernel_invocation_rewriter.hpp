//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg60xx {

class ActKernelInvocationRewriter final : public mlir::OpRewritePattern<VPUASM::ActKernelInvocationOp> {
public:
    ActKernelInvocationRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::ActKernelInvocationOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("ActKernelInvocation_VPUASM2NPUReg60XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActKernelInvocationOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
