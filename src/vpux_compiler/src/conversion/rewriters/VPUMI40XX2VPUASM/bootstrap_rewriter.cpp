//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/bootstrap_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> BootstrapRewriter::symbolize(VPUMI40XX::BootstrapOp op, SymbolMapper&,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();
    int barrierId = mlir::cast<vpux::VPURegMapped::IndexType>(op.getBarrier().getType()).getValue();
    auto newOp = VPUASM::BootstrapOp::create(rewriter, op.getLoc(), symName, barrierId);
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
