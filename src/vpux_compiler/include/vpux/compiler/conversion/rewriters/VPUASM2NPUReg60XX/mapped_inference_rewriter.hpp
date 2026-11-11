//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg60xx {

class MappedInferenceRewriter final : public mlir::OpRewritePattern<VPUASM::MappedInferenceOp> {
public:
    MappedInferenceRewriter(mlir::MLIRContext* ctx, const Logger&, ELF::SymbolReferenceMap&)
            : mlir::OpRewritePattern<VPUASM::MappedInferenceOp>(ctx) {
        setDebugName("MappedInference_VPUASM2NPUReg60XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::MappedInferenceOp origOp, mlir::PatternRewriter& rewriter) const final;
};
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
