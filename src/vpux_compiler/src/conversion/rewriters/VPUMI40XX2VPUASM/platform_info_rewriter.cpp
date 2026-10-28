//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/platform_info_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> PlatformInfoRewriter::symbolize(VPUMI40XX::PlatformInfoOp op, SymbolMapper&,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto newOp = VPUASM::PlatformInfoOp::create(rewriter, op.getLoc(), symName);
    rewriter.eraseOp(op);
    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
