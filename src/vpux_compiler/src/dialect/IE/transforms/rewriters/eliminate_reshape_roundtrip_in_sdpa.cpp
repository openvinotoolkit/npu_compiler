//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// EliminateReshapeRoundtripInSDPA
//

// In GQA, batch-op processing first shrinks MatMul over grouped heads, then
// unrolls new per-group heads to keep the SoftMax stage operating with a
// consistent head count. This keeps the SDPA chain layout-friendly for later
// optimization.
//
// Around that transformed region, the Concat result is often reshaped to the
// expanded head view for SoftMax (and optional mask Add), then reshaped back
// before per-head Slice users. When those two Reshapes are exact inverses,
// fold the round-trip so the chain stays in the shrunk domain.
//
// The mask is typically causal and usually shaped as [1, 1, T, K]
// Here T is query token length and K is KV-cache length.
//
// This pattern applies a GQA regroup by H_per_G:
//   - head count in shrunk domain: Hs_shrunk = Hs_expanded / H_per_G
//   - token axis in shrunk domain: Ts_shrunk = Ts * H_per_G
//
// To preserve Add semantics after removing the reshape round-trip, the mask is
// tiled on the regrouped axis, e.g. IE.Tile(mask, [1, 1, H_per_G, 1]) when
// regroup happens on H.
//
// Before:
//     Concat(unrolled K MatMul outputs)   [B, Hs_shrunk, Ts_shrunk, K]
//                  |
//               Reshape                   [B, Hs_expanded, Ts, K]
//                  |
//               Add(mask)                 mask is [B, 1, Ts, K]
//                  |
//               SoftMax                   (axis = innermost, size = K)
//                  |
//               Reshape                   [B, Hs_shrunk, Ts_shrunk, K]
//                  |
//         Slice / downstream unrolled V MatMul users
//
// After:
//     Concat(unrolled K MatMul outputs)   [B, Hs_shrunk, Ts_shrunk, K]
//                  |
//               Add(mask')                mask' = IE.Tile(mask, [1, 1, H_per_G, 1]) -> [B, 1, Ts_shrunk, K]
//                  |
//               SoftMax                   (axis = innermost, size = K)
//                  |
//         Slice / downstream unrolled V MatMul users

bool isInverseReshape(IE::ReshapeOp lhs, IE::ReshapeOp rhs) {
    return getShape(lhs.getInput()) == getShape(rhs.getOutput()) &&
           getShape(lhs.getOutput()) == getShape(rhs.getInput());
}

mlir::Operation* getSingleUser(mlir::Value value) {
    if (!value.hasOneUse()) {
        return nullptr;
    }
    return *value.getUsers().begin();
}

template <typename OpType>
OpType getSingleUserAs(mlir::Value value) {
    return mlir::dyn_cast_or_null<OpType>(getSingleUser(value));
}

// Builds a mask value that Add-broadcasts against `shrunkShape` with the same
// semantics as the original Add against `expandedShape`. Returns `nullptr` when
// unsupported.
//
// For each axis:
//   - unchanged (expanded == shrunk): mask must be 1 or match the dim.
//   - growing   (shrunk > expanded):  mask must match expandedDim; tiled by the ratio.
//   - shrinking (expanded > shrunk):  mask must be 1 (broadcast semantics are preserved).
//
// When no axis needs tiling the mask is returned as-is (fast path).
mlir::Value buildShrunkMask(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value mask,
                            ShapeRef expandedShape, ShapeRef shrunkShape) {
    const auto maskShape = getShape(mask);
    if (maskShape.size() != expandedShape.size()) {
        return nullptr;
    }

    SmallVector<int64_t> repeats(maskShape.size(), 1);
    for (const auto i : irange(maskShape.size())) {
        const auto d = Dim(i);
        const auto expandedDim = expandedShape[d];
        const auto shrunkDim = shrunkShape[d];
        const auto maskDim = maskShape[d];

        if (expandedDim == shrunkDim) {
            if (maskDim != 1 && maskDim != expandedDim) {
                return nullptr;
            }
        } else if (shrunkDim > expandedDim && shrunkDim % expandedDim == 0) {
            // Growing axis: tile mask to cover shrunkDim.
            if (maskDim != expandedDim) {
                return nullptr;
            }
            repeats[i] = shrunkDim / expandedDim;
        } else if (expandedDim > shrunkDim && expandedDim % shrunkDim == 0) {
            // Shrinking axis: mask must be 1 to preserve broadcast semantics.
            if (maskDim != 1) {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    return rewriter.createOrFold<IE::TileOp>(appendLoc(loc, "mask_tile"), mask,
                                             getIntArrayAttr(rewriter.getContext(), repeats));
}

class EliminateReshapeRoundtripInSDPA final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    EliminateReshapeRoundtripInSDPA(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ReshapeOp>(ctx, benefit), _log(log) {
        setDebugName("EliminateReshapeRoundtripInSDPA");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EliminateReshapeRoundtripInSDPA::matchAndRewrite(IE::ReshapeOp outerRe,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}'", getDebugName(), outerRe->getLoc());

    // Restrict to SDPA chains where reshapes round-trip with unrolled MatMuls:
    // (AffinedMatMuls) -> Concat -> outerRe -> [Add] -> SoftMax -> innerRe -> Slices (unrolled MatMuls).
    // The Concat and Slices prove that MatMul unroll has occurred.
    auto concatOp = outerRe.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, outerRe, "Outer Reshape input is not produced by Concat");
    }

    // Walk outerRe -> [Add] -> SoftMax -> innerRe, requiring single-use links.
    auto* singleUserOp = getSingleUser(outerRe.getOutput());
    if (singleUserOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, outerRe, "Outer Reshape has multiple users");
    }

    auto addOp = mlir::dyn_cast<IE::AddOp>(singleUserOp);
    mlir::Value maskVal;
    if (addOp != nullptr) {
        if (!addOp.getOutput().hasOneUse()) {
            return matchFailed(_log.nest(), rewriter, outerRe, "Add has multiple users");
        }
        if (addOp.getInput1() == outerRe.getOutput()) {
            maskVal = addOp.getInput2();
        } else if (addOp.getInput2() == outerRe.getOutput()) {
            maskVal = addOp.getInput1();
        } else {
            return matchFailed(_log.nest(), rewriter, outerRe, "Add operand mismatch");
        }
        singleUserOp = *addOp->getUsers().begin();
    }

    auto softmaxOp = mlir::dyn_cast<IE::SoftMaxOp>(singleUserOp);
    if (softmaxOp == nullptr || !softmaxOp.getOutput().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, outerRe, "No single-use SoftMax on the chain");
    }

    auto innerRe = getSingleUserAs<IE::ReshapeOp>(softmaxOp.getOutput());
    if (innerRe == nullptr || !isInverseReshape(outerRe, innerRe)) {
        return matchFailed(_log.nest(), rewriter, outerRe, "No inverse Reshape after SoftMax");
    }

    // After MatMul unroll (which happens in earlier phase), innerRe output should be consumed by Slices.
    // Verify all users are Slice and count them.
    const auto sliceUsers = innerRe->getUsers();
    if (sliceUsers.empty() || !llvm::all_of(sliceUsers, [](mlir::Operation* op) {
            return mlir::isa<IE::SliceOp>(op);
        })) {
        return matchFailed(_log.nest(), rewriter, outerRe,
                           "Inner Reshape user is not a Slice (unroll may not have occurred)");
    }
    const auto sliceUserCount = llvm::range_size(sliceUsers);

    // Verify that concat input count matches slice count. Both should equal the shrink factor.
    const auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() != sliceUserCount) {
        return matchFailed(_log.nest(), rewriter, outerRe, "Concat input count ({0}) does not match Slice count ({1})",
                           concatInputs.size(), sliceUserCount);
    }

    const auto expandedShape = getShape(outerRe.getOutput());
    const auto shrunkShape = getShape(outerRe.getInput());
    if (expandedShape.size() != shrunkShape.size() || expandedShape.empty()) {
        return matchFailed(_log.nest(), rewriter, outerRe, "Reshape pair must preserve non-zero rank");
    }
    const auto rank = static_cast<int64_t>(expandedShape.size());

    // Require SoftMax on the innermost dim with unchanged size, so the axis
    // attribute can be reused verbatim in the shrunk domain.
    const auto axis = softmaxOp.getAxisInd() < 0 ? softmaxOp.getAxisInd() + rank : softmaxOp.getAxisInd();
    if (axis < 0 || axis >= rank) {
        return matchFailed(_log.nest(), rewriter, outerRe, "SoftMax axis is out of range");
    }
    if (axis != rank - 1 || expandedShape[Dim(axis)] != shrunkShape[Dim(axis)]) {
        return matchFailed(_log.nest(), rewriter, outerRe, "SoftMax axis is not the innermost / dim size changed");
    }

    mlir::Value shrunkMask;
    if (addOp != nullptr) {
        shrunkMask = buildShrunkMask(rewriter, addOp->getLoc(), maskVal, expandedShape, shrunkShape);
        if (shrunkMask == nullptr) {
            return matchFailed(_log.nest(), rewriter, outerRe, "Mask cannot be moved into shrunk domain");
        }
    }

    _log.nest().trace("Folding Reshape -> (Add) -> SoftMax -> Reshape into the shrunk domain");

    mlir::Value softmaxInput = outerRe.getInput();
    if (addOp != nullptr) {
        softmaxInput = rewriter.create<IE::AddOp>(
                takeOpLoc(addOp, "shrunk"), softmaxInput, shrunkMask, addOp.getAutoBroadcastAttr(),
                addOp.getPostOpAttr(), addOp.getClampAttr(), addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());
    }

    auto newSoftmaxOp = rewriter.create<IE::SoftMaxOp>(takeOpLoc(softmaxOp, "shrunk"), softmaxInput,
                                                       softmaxOp.getAxisIndAttr(), softmaxOp.getPadSizeAttr(),
                                                       softmaxOp.getDstElemTypeAttr(), softmaxOp.getMaskAwareAttr());

    rewriter.replaceOp(innerRe, newSoftmaxOp.getOutput());
    return mlir::success();
}

}  // namespace

void vpux::IE::registerEliminateReshapeRoundtripInSDPARewriters(RewriterRegistry& registry, Logger log,
                                                                ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                size_t index) {
    const auto benefit = benefitLevels[index];
    registry.registerRewriterSet("eliminate-reshape-roundtrip-in-sdpa-set", [&registry, log, benefit]() {
        registry.registerRewriter<EliminateReshapeRoundtripInSDPA>("eliminate-reshape-roundtrip-in-sdpa", benefit, log);
    });
}
