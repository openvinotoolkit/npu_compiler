//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg50xx {

class MappedInferenceVersionRewriter final : public mlir::OpRewritePattern<VPUASM::MappedInferenceVersionOp> {
public:
    MappedInferenceVersionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::MappedInferenceVersionOp>(ctx), _log(log) {
        setDebugName("MappedInferenceVersion_VPUASM2NPUReg50XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
