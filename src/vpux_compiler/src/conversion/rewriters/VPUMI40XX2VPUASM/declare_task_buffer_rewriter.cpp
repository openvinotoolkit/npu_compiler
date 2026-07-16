//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DeclareTaskBufferRewriter::symbolize(
        VPUMI40XX::DeclareTaskBufferOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op.getResult()).getRootReference();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto newOp = rewriter.create<VPUASM::DeclareTaskBufferOp>(op.getLoc(), symName, taskIdx, op.getTaskTypeAttr(),
                                                              op.getOffsetAttr());

    rewriter.eraseOp(op);
    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
