//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux {
namespace VPUIPDPU {
DPUVariantRewriter::DPUVariantRewriter(
        mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap,
        VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode)
        : mlir::OpRewritePattern<VPUASM::DPUVariantOp>(ctx),
          _log(log),
          _symRefMap(symRefMap),
          _npu5PPEBackwardsCompatibilityMode(npu5PPEBackwardsCompatibilityMode) {
    setDebugName("DPUInvariant_VPUIPDPURewriter");
}
mlir::LogicalResult DPUVariantRewriter::matchAndRewrite(VPUASM::DPUVariantOp op,
                                                        mlir::PatternRewriter& rewriter) const {
    auto variant = rewriter.create<VPUIPDPU::DPUVariantOp>(
            op.getLoc(), op.getSymNameAttr(), op.getTaskIndexAttr(), op.getTaskLocationAttr(), op.getNextLinkAttr(),
            op.getInvariantTaskLocationAttr(), op.getWeightsAttr(), op.getWeightTableAttr(),
            op.getWeightTableDataPtrAttr(), op.getWeightTableSpPtrAttr(), op.getWeightTableScaleAttr(),
            op.getWeightTableBiasAttr(), op.getWeightZeroPointsAttr(), op.getNceTaskTypeAttr(), op.getWorkloadIdAttr(),
            op.getVariantPrimitiveIdAttr());

    auto& variantRegion = variant.getRegion();
    auto varBlock = rewriter.createBlock(&variantRegion);

    auto dpuVariantExpandIface = mlir::dyn_cast<VPUASM::DPUVariantExpandOpInterface>(op.getOperation());
    if (dpuVariantExpandIface == nullptr) {
        _log.error("Missing expand DPU variant configuration interface for arch {0}",
                   stringifyArchKind(config::getArch(op)).str());
        return mlir::failure();
    }

    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);

        if (dpuVariantExpandIface.expandGeneralConfig(rewriter, _log).failed()) {
            return mlir::failure();
        }
        if (dpuVariantExpandIface.expandIDUConfig(rewriter, _log, _symRefMap, _npu5PPEBackwardsCompatibilityMode)
                    .failed()) {
            return mlir::failure();
        }

        if (dpuVariantExpandIface.expandPPEConfig(rewriter, _log, _symRefMap, _npu5PPEBackwardsCompatibilityMode)
                    .failed()) {
            return mlir::failure();
        }
        if (dpuVariantExpandIface.expandODUConfig(rewriter, _log, varBlock, _symRefMap).failed()) {
            return mlir::failure();
        }
    }

    auto invariant = mlir::cast<VPUASM::DPUInvariantOp>(_symRefMap.lookupSymbol(op.getInvariant()));
    rewriter.create<VPUIPDPU::BarrierCfgOp>(op.getLoc(), invariant.getWaitBarriers(), invariant.getUpdateBarriers(),
                                            invariant.getStartAfterAttr(), invariant.getCleanAfterAttr());

    auto taskIndex = op.getIndexType().getValue();
    auto isFirstVariant = invariant.getFirstVariantIndex() == taskIndex;
    auto isLastVariant = invariant.getVariantCount() > 1 && invariant.getLastVariantIndex() == taskIndex;
    rewriter.create<VPUIPDPU::DPUGroupOp>(op.getLoc(), invariant.getIndexType(), invariant.getVariantCount(),
                                          isFirstVariant, isLastVariant);

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
