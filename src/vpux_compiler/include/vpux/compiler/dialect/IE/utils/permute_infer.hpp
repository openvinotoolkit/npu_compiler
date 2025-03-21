//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool strictInfer);

template <typename PermOpPrev, typename PermOp>
mlir::LogicalResult fusePermutations(PermOp permuteOp, mlir::PatternRewriter& rewriter) {
    auto prevPermuteOp = mlir::dyn_cast_or_null<PermOpPrev>(permuteOp.getInput().getDefiningOp());
    if (prevPermuteOp == nullptr) {
        return mlir::failure();
    }

    // For the case with mempermute having a mempermute and op X as user
    // having sequantial mempermutes yields more performance than parallel, thus don't fuse them
    // If all users are mempermute, then fusing them is better
    auto checkAllUsersMemPerm = llvm::none_of(prevPermuteOp->getUsers(), [](auto user) {
        return !mlir::isa<IE::MemPermuteOp>(user);
    });

    if (mlir::isa<IE::MemPermuteOp>(prevPermuteOp) && !prevPermuteOp->hasOneUse() && !checkAllUsersMemPerm) {
        return mlir::failure();
    }

    auto prevMemPerm = prevPermuteOp.getMemPerm();
    auto memPerm = permuteOp.getMemPerm();
    auto newMemPerm = memPerm.compose(prevMemPerm);

    const auto canFuseIntoPermuteCastOp =
            mlir::isa<IE::PermuteCastOp>(prevPermuteOp) && mlir::isa<IE::PermuteCastOp>(permuteOp);
    if (canFuseIntoPermuteCastOp) {
        rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(permuteOp, permuteOp.getType(), prevPermuteOp.getInput(),
                                                       permuteOp.getDstOrderAttr(),
                                                       mlir::AffineMapAttr::get(newMemPerm));
    } else {
        rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(permuteOp, permuteOp.getType(), prevPermuteOp.getInput(),
                                                      permuteOp.getDstOrderAttr(),
                                                      mlir::AffineMapAttr::get(newMemPerm));
    }

    return mlir::success();
}

mlir::ArrayAttr permuteBounds(mlir::MLIRContext* ctx, vpux::BoundedTypeInterface boundedTensor, DimsOrder srcOrder,
                              DimsOrder dstOrder, mlir::AffineMap memPerm);
