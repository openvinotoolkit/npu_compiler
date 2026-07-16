//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/view_task_range_rewriter.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> ViewTaskRangeRewriter::symbolize(VPURegMapped::ViewTaskRangeOp op,
                                                                      SymbolMapper& mapper,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    mapper[op.getResult()] = mapper[op.getFirst()];

    rewriter.eraseOp(op);
    return SymbolizationResult();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
