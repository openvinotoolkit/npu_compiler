//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/rewriters/expand_with_layer_rewriter.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux {
namespace IE {
void swapExpandWithReorder(mlir::PatternRewriter& rewriter, IE::ExpandOp expandOp, mlir::Operation* origReorderOp) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(expandOp);

    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp->getLoc(), origReorderOp->getOperand(0),
                                                     expandOp.getPadsBeginAttr(), expandOp.getPadsEndAttr());

    mlir::IRMapping mapper;
    mapper.map(origReorderOp->getOperand(0), newExpandOp.getOutput());
    mlir::Operation* newOp = rewriter.clone(*origReorderOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
    rewriter.replaceOp(expandOp, newOp->getResults());
}

//
//  The beneficial pattern:
//
//     input               input
//       |                   |
//     Reorder             Expand
//       |                   |
//     Expand   ==>        Reorder
//       |                   |
//     Slice(s)            Slice(s)
//       |                   |
//     Reorder(s)          Reorder(s)
//       |                   |
//     output              output
//
//  It's worth to swap parent Reorder and Expand,  the swapped Reorder will be handled by follow-up optimizations.
//
mlir::LogicalResult ExpandWithLayer::matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const {
    auto* ctx = origExpandOp->getContext();
    auto layerOp = origExpandOp.getInput().getDefiningOp();
    if (layerOp == nullptr) {
        return mlir::failure();
    }
    if (!_isBeneficalToSwap(origExpandOp, layerOp)) {
        return mlir::failure();
    }

    _log.trace("Got '{0}' at '{1}' -> Expand at '{2}' pair", layerOp->getName(), layerOp->getLoc(),
               origExpandOp->getLoc());

    const auto isExpand = [](mlir::Operation* reorderUser) -> bool {
        return mlir::isa<IE::ExpandOp>(reorderUser);
    };

    if (!llvm::all_of(layerOp->getUsers(), isExpand)) {
        return matchFailed(_log.nest(), rewriter, origExpandOp,
                           "Reorder has more than one user and they are heterogeneous");
    }

    // If after swap the op cannot support by PermuteDMA, will not swap
    // Input (1x1x512x512, NCWH) -> Reorder (1x1x512x512, NHWC) -> Expand (1x16x512x512, NHWC)
    // After Swap the Reorder cannot convert to PermuteDMA
    // Input (1x1x512x512, NCWH) -> Expand (1x16x512x512, NCWH) -> Reorder (1x16x512x512, NHWC)
    auto expandOutType = origExpandOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto newOrderInType = expandOutType.changeDimsOrder(DimsOrder::fromValue(layerOp->getOperand(0)));
    auto newOrderOutType = expandOutType.changeDimsOrder(DimsOrder::fromValue(layerOp->getResult(0)));
    auto memPerm = getPermutationFromOrders(newOrderInType.getDimsOrder(), newOrderOutType.getDimsOrder(), ctx);
    auto unsupportPermuteDMA = [&]() -> bool {
        const auto inShape = newOrderInType.getShape();
        return newOrderInType.getRank() == 4 &&
               memPerm == mlir::AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 3, 2, 1}, ctx) &&
               inShape[Dims4D::Act::C] > 1 && inShape[Dims4D::Act::H] > 1 && inShape[Dims4D::Act::W] > 1;
    };

    if (unsupportPermuteDMA()) {
        return mlir::failure();
    }

    for (auto* reorderUser : llvm::make_early_inc_range(layerOp->getUsers())) {
        auto expandOp = mlir::cast<IE::ExpandOp>(reorderUser);
        swapExpandWithReorder(rewriter, expandOp, layerOp);
    }

    return mlir::success();
}

}  // namespace IE
}  // namespace vpux
