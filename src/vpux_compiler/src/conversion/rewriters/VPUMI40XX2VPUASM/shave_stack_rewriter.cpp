//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/shave_stack_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> ShaveStackRewriter::symbolize(VPUMI40XX::ShaveStackFrameBuffOp op, SymbolMapper&,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto newOp = VPUASM::ShaveStackFrameBuffOp::create(rewriter, op.getLoc(), symName, op.getStackSize());
    rewriter.eraseOp(op);
    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
