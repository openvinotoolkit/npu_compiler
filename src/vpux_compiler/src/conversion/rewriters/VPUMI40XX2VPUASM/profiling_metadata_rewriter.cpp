//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/profiling_metadata_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> ProfilingMetadataRewriter::symbolize(
        VPUMI40XX::ProfilingMetadataOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    mlir::StringAttr symName = findSym(result).getRootReference();

    auto newOp = rewriter.create<VPUASM::ProfilingMetadataOp>(op.getLoc(), symName, op.getMetadataAttr());
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
