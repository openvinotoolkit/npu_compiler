//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_range_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelRangeRewriter::symbolize(VPUMI40XX::ActKernelRangeOp op, SymbolMapper&,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());

    auto kernelTaskType = op.getKernelTaskType();
    bool isCacheOp = VPUIP::isCacheOpTaskType(kernelTaskType, /*includePrefetch=*/false);

    mlir::SymbolRefAttr kernelTextAttr = isCacheOp ? nullptr : findSym(op.getKernelTextIndex());
    mlir::SymbolRefAttr kernelEntryAttr = isCacheOp ? nullptr : findSym(op.getKernelEntryIndex());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto newOp = rewriter.create<VPUASM::ActKernelRangeOp>(op.getLoc(), symName, taskIdx, taskLocation, kernelTextAttr,
                                                           kernelEntryAttr, kernelTaskType);
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
