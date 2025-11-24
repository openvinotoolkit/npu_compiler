//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/move_view_ops_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux::VPU::VF::v1 {

mlir::LogicalResult MoveViewOpsRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto isOpWeightsFromVFOperandIndex = [](mlir::Operation* op, size_t operandIdx) -> bool {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr) {
            return false;
        }
        if (auto opWeights = llvm::cast_if_present<mlir::BlockArgument>(nceOp.getWeightsOperand())) {
            return opWeights.getArgNumber() == operandIdx;
        }
        return false;
    };
    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());

    for (auto vfOperand : vfOp->getOperands() | indexed) {
        auto parentOp = vfOperand.value().getDefiningOp<VPU::TilingViewLikeOpInterface>();

        // E-163016 remove is VFSupported flag when scf and current algorithm is aligned
        if (parentOp == nullptr || !VPU::isPureViewOp(parentOp) || VPU::onlySupportPartialTilingDims(parentOp) ||
            !parentOp.isVFSupported()) {
            continue;
        }

        // Exclude weights moving for non-SOC tiling
        // As only under SOC case, the producer op for weights (if exists) need to be tiled and merged into VF
        if (llvm::any_of(vfOp.getBody()->getArgument(vfOperand.index()).getUsers(), [&](auto user) {
                return isOpWeightsFromVFOperandIndex(user, vfOperand.index()) &&
                       (tilingStrategy[Dims4D::Act::C.ind()] == 1);
            })) {
            continue;
        }

        if (llvm::all_of(parentOp->getOperands(), [](auto value) {
                return mlir::isa_and_nonnull<mlir::BlockArgument>(value) ||
                       mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
            })) {
            continue;
        }

        auto newVFOp = fuseOpsInBlock(rewriter, vfOp, parentOp);
        rewriter.replaceOp(vfOp, newVFOp.getResult(0));
        return mlir::success();
    }
    return mlir::failure();
}
}  // namespace vpux::VPU::VF::v1
