//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_version_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> MappedInferenceVersionRewriter::symbolize(
        VPUMI40XX::MappedInferenceVersionOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();
    auto newOp = VPUASM::MappedInferenceVersionOp::create(rewriter, op.getLoc(), symName, op.getMajor(), op.getMinor(),
                                                          op.getPatch());
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
