//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <algorithm>

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

    // Find the single DPUVariantOp whose previousTask operand points back to this op's result
    auto users = op.getResult().getUsers();
    auto it = std::find_if(users.begin(), users.end(), [&](mlir::Operation* user) {
        auto variantOp = mlir::dyn_cast<VPUMI40XX::DPUVariantOp>(user);
        return variantOp && variantOp.getPreviousTask() == op.getResult();
    });
    VPUMI40XX::DPUVariantOp nextVariant = it != users.end() ? mlir::cast<VPUMI40XX::DPUVariantOp>(*it) : nullptr;

    mlir::SymbolRefAttr nextLink = nullptr;
    if (nextVariant && nextVariant.getTaskLink().has_value()) {
        assert(nextVariant.getTaskLink().value() == op.getType());
        nextLink = findSym(nextVariant.getTaskLocation());
    }

    auto linkedInvariantOp = op.getInvariant().getDefiningOp<VPUMI40XX::DPUInvariantOp>();
    auto invariantTaskLocation = findSym(linkedInvariantOp.getTaskLocation());

    auto weights = optionalSym(op.getWeights());
    auto weightTable = optionalSym(op.getWeightTable());
    auto weightTableDataPtr = optionalSym(op.getWeightTableDataPtr());
    auto weightTableSpPtr = optionalSym(op.getWeightTableSpPtr());
    auto weightTableScale = optionalSym(op.getWeightTableScale());
    auto weightTableBias = optionalSym(op.getWeightTableBias());
    auto weightTableAlpha = optionalSym(op.getWeightTableAlpha());
    auto weightZeroPoints = optionalSym(op.getWeightZeroPoints());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto newOp = rewriter.create<VPUASM::DPUVariantOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, invariantSym, invariantTaskLocation, weights,
            weightTable, weightTableDataPtr, weightTableSpPtr, weightTableScale, weightTableBias, weightTableAlpha,
            weightZeroPoints, op.getNceTaskTypeAttr(), op.getInStartAttr(), op.getInEndAttr(), op.getStartAttr(),
            op.getEndAttr(), op.getPadAttr(), op.getMpeModeAttr(), op.getClusterIdAttr(), op.getHaloRegionsAttr(),
            op.getWorkloadIdAttr(), op.getSprLutRead(), op.getPalletLutRead(), op.getForceInvRead(),
            op.getVariantPrimitiveIdAttr(), op.getWeightTableOffsetAttr());

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
