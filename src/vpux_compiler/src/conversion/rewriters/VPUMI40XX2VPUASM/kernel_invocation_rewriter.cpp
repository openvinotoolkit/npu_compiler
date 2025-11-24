//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_invocation_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelInvocationRewriter::symbolize(
        VPUMI40XX::ActKernelInvocationOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());
    auto kernelParams = findSym(op.getKernelParams());

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto oldKernelRange = op.getRangeIndex().getDefiningOp<VPUMI40XX::ActKernelRangeOp>();
    auto kernelIndexAttr =
            mlir::IntegerAttr::get(vpux::getUInt64Type(getContext()), oldKernelRange.getIndexType().getValue());

    auto kernelTaskType = oldKernelRange.getKernelTaskType();
    bool isCacheOp = VPUIP::isCacheOpTaskType(kernelTaskType, /*includePrefetch=*/false);

    auto kernelData = isCacheOp ? nullptr : findSym(oldKernelRange.getKernelArgsIndex());
    auto kernelRange = findSym(oldKernelRange.getTaskLocation());

    mlir::SymbolRefAttr profilingData = nullptr;
    if (auto profBuffer = op.getProfilingData()) {
        profilingData = findSym(profBuffer);
    }

    const auto findNextInvocationSymRef = [&](auto op) -> mlir::SymbolRefAttr {
        // optimize for worst-case scenario when invocation isn't linked to anything
        // by searching just 2 next ops instead of whole chain
        // op can be linked either to immediate next or one after it in case of disabled fifo per shave engine
        static constexpr auto maxDistanceToLinkedOp = size_t{2};
        auto candidate = VPUMI40XX::getNextOp(op);
        for ([[maybe_unused]] auto _ : irange(maxDistanceToLinkedOp)) {
            if (!candidate) {
                return nullptr;
            }

            if (auto taskLink = candidate.getTaskLink(); taskLink.has_value() && taskLink.value() == op.getType()) {
                return findSym(candidate.getTaskLocation());
            }

            candidate = VPUMI40XX::getNextOp(candidate);
        }

        return nullptr;
    };

    mlir::SymbolRefAttr nextLink = findNextInvocationSymRef(op);

    auto newOp = rewriter.create<VPUASM::ActKernelInvocationOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, kernelRange, kernelData, kernelParams, waitAttr,
            updateAttr, profilingData, op.getTileAttr(), op.getStartAfterAttr(), op.getCleanAfterAttr(),
            kernelIndexAttr);

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> KernelInvocationRewriter::getSymbolicNames(
        VPUMI40XX::ActKernelInvocationOp op, size_t) {
    return createSymbolicName(op);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
