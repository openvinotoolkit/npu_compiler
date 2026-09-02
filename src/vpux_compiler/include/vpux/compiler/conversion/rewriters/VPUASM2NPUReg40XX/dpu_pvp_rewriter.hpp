//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg40xx {

class DpuPVPRewriter final : public mlir::OpRewritePattern<VPUASM::DpuPVPOp> {
public:
    DpuPVPRewriter(mlir::MLIRContext* ctx, Logger log, ELF::MainOp elfMain)
            : mlir::OpRewritePattern<VPUASM::DpuPVPOp>(ctx), _log(log), _elfMain(elfMain) {
        setDebugName("DpuPVP_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::DpuPVPOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::MainOp _elfMain;
};
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
