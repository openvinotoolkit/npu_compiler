//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

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
    auto newLoc = takeOpLoc(permuteOp, llvm::formatv("_memperm_{0}", DimsOrder::fromAffineMap(newMemPerm)));
    if (canFuseIntoPermuteCastOp) {
        rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(permuteOp, permuteOp.getType(), prevPermuteOp.getInput(),
                                                       permuteOp.getDstOrderAttr(),
                                                       mlir::AffineMapAttr::get(newMemPerm))
                ->setLoc(newLoc);

    } else {
        rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(permuteOp, permuteOp.getType(), prevPermuteOp.getInput(),
                                                      permuteOp.getDstOrderAttr(), mlir::AffineMapAttr::get(newMemPerm))
                ->setLoc(newLoc);
    }

    return mlir::success();
}
