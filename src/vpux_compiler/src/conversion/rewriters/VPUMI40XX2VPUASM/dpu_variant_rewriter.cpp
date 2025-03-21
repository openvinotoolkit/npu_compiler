//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DPUVariantRewriter::symbolize(VPUMI40XX::DPUVariantOp op, SymbolMapper&,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());
    auto invariantSym = findSym(op.getInvariant());

    auto optionalSym = [&](mlir::Value val) -> mlir::SymbolRefAttr {
        auto sym = val ? findSym(val) : nullptr;
        return sym;
    };

    auto opUses = op.getResult().getUses();
    mlir::SymbolRefAttr nextLink = nullptr;
    auto nextVariantIt = llvm::find_if(opUses, [](mlir::OpOperand& operand) -> bool {
        auto user = mlir::dyn_cast<VPUMI40XX::DPUVariantOp>(operand.getOwner());
        return user && user.getPreviousTask() == operand.get();
    });
    auto nextVariant =
            nextVariantIt != opUses.end() ? mlir::cast<VPUMI40XX::DPUVariantOp>(nextVariantIt->getOwner()) : nullptr;
    if (nextVariant && nextVariant.getTaskLink().has_value()) {
        assert(nextVariant.getTaskLink().value() == op.getType());
        nextLink = findSym(nextVariant.getTaskLocation());
    }

    auto linkedOp = op.getInvariant().getDefiningOp();
    auto linkedInvariantOp = mlir::dyn_cast<VPUMI40XX::DPUInvariantOp>(linkedOp);
    auto invariantTaskLocation = findSym(linkedInvariantOp.getTaskLocation());

    auto weights = optionalSym(op.getWeights());
    auto weightTable = optionalSym(op.getWeightTable());
    auto weightTableDataPtr = optionalSym(op.getWeightTableDataPtr());
    auto weightTableSpPtr = optionalSym(op.getWeightTableSpPtr());
    auto weightTableScale = optionalSym(op.getWeightTableScale());
    auto weightTableBias = optionalSym(op.getWeightTableBias());
    auto weightZeroPoints = optionalSym(op.getWeightZeroPoints());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto newOp = rewriter.create<VPUASM::DPUVariantOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, invariantSym, invariantTaskLocation, weights,
            weightTable, weightTableDataPtr, weightTableSpPtr, weightTableScale, weightTableBias, weightZeroPoints,
            op.getNceTaskTypeAttr(), op.getInStartAttr(), op.getInEndAttr(), op.getStartAttr(), op.getEndAttr(),
            op.getPadAttr(), op.getMpeModeAttr(), op.getClusterIdAttr(), op.getHaloRegionsAttr(),
            op.getWorkloadIdAttr(), op.getSprLutReadAttr(), op.getPalletLutReadAttr(), op.getForceInvReadAttr());

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
