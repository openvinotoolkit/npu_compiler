//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"

#include <mlir/IR/AffineMap.h>

namespace vpux {
namespace IE {

bool checkMatMul(IE::MatMulOp origOp) {
    const auto is4DShape = [](ShapeRef shape) {
        return shape.size() == 4;
    };
    auto lhs = origOp.getInput1();
    auto rhs = origOp.getInput2();

    auto lhsShape = getShape(lhs);
    auto rhsShape = getShape(rhs);

    if (!is4DShape(lhsShape) || !is4DShape(rhsShape)) {
        return false;
    }

    using namespace Dims4D::Act;

    // Right now it's expected to be the case when transposeA = false and transposeB = true
    if (!IE::isMatmulWithRHSTransposition(origOp)) {
        return false;
    }
    if (lhsShape[N] != rhsShape[N] || lhsShape[C] != rhsShape[C] || lhsShape[W] != rhsShape[W]) {
        return false;
    }

    // Shrinking groups is beneficial only when the LHS sequence-length dimension (H) is small.
    //
    // When the optimization fires the head count on the group axis decreases by the group
    // size, while H grows proportionally.  This reshapes the surrounding SDPA chain into:
    //
    //   MatMul (reduced heads) -> Add (original heads) -> Softmax (original heads)
    //                          -> MatMul (reduced heads)
    //
    // Whether this is a net win depends on how the downstream MatMul is lowered:
    //   - Grouped MatMul:  neutral; no structural change to the schedule.
    //   - Unrolled MatMul: the head-count mismatch across the chain breaks vertical
    //                      fusion within the SDPA pattern, which can regress performance.
    //
    // For prefill workloads H is large (typically 512/1024), so the unrolled-MatMul path
    // is more likely to dominate and shrinking is harmful.
    //
    // For generation workloads the MatMul is memory-bandwidth bound rather than
    // compute bound, so eliminating the Broadcast reduces memory traffic enough to
    // outweigh any VF disruption.  Two generation sub-cases exist:
    //   - Standard decode:        H == 1 (one new token per step).
    //   - Speculative decoding:   H is small but greater than 1 (a short draft
    //                             sequence that is verified in a single forward pass).
    // The threshold below covers both sub-cases.
    // For large H (prefill), only proceed when the grouped-MatMul cost model judges
    // the transformation profitable; this avoids triggering the unrolled-MatMul path
    // that would break vertical fusion across the SDPA chain.
    constexpr int64_t MAX_LHS_H_FOR_SHRINK = 32;
    return lhsShape[H] <= MAX_LHS_H_FOR_SHRINK || isGroupedMatMulBeneficial(origOp, lhsShape, rhsShape);
}

bool checkSwapLast2DimsTranspose(IE::TransposeOp transposeOp) {
    if (transposeOp == nullptr) {
        return false;
    }
    const auto orderAttr = transposeOp.getOrderValue();
    if (!orderAttr.has_value()) {
        return false;
    }
    const auto affineMap = orderAttr.value();
    const unsigned rank = affineMap.getNumDims();
    if (rank < 2 || affineMap.getNumResults() != rank) {
        return false;
    }
    for (unsigned i = 0; i < rank; ++i) {
        const auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(affineMap.getResult(i));
        if (!dimExpr) {
            return false;
        }
        const unsigned expected = (i == rank - 2) ? rank - 1 : (i == rank - 1) ? rank - 2 : i;
        if (dimExpr.getPosition() != expected) {
            return false;
        }
    }
    return true;
}

bool checkAffineReshape(IE::AffineReshapeOp affineReshapeOp) {
    if (affineReshapeOp == nullptr) {
        return false;
    }

    auto inputShape = getShape(affineReshapeOp.getInput());
    auto outputShape = getShape(affineReshapeOp.getOutput());

    // ensure the input/output shape of AffineReshapeOp to be 5/4D
    if (inputShape.size() != 5 || outputShape.size() != 4) {
        return false;
    }

    // Since Matmul actually happens on 2D tensors, we should only check the last 2 dims
    return (inputShape[Dims5D::Act::H] == outputShape[Dims4D::Act::H] &&
            inputShape[Dims5D::Act::W] == outputShape[Dims4D::Act::W]);
}

bool checkBroadCast(IE::BroadcastOp broadcastOp) {
    if (broadcastOp == nullptr) {
        return false;
    }

    const auto is5DShape = [](ShapeRef shape) {
        return shape.size() == 5;
    };
    auto inputShape = getShape(broadcastOp.getInput());
    auto outputShape = getShape(broadcastOp.getOutput());
    if (!is5DShape(inputShape) || !is5DShape(outputShape)) {
        return false;
    }

    auto broadCastDim = IE::getDiffInOutSizeDims(inputShape, outputShape);
    if (broadCastDim.size() != 1) {
        return false;
    }

    // BroadcastOp should broadcast 5D tensor on the first spatial dim (d2)
    return broadCastDim.front() == Dims5D::Act::getSpatialDim(0) && inputShape[broadCastDim.front()] == 1;
}

bool shouldShrinkMatmulGroups(IE::MatMulOp matmulOp) {
    return matchShrinkMatmulGroupsPattern(matmulOp).has_value();
}

std::optional<MatchedShrinkPattern> matchShrinkMatmulGroupsPattern(IE::MatMulOp matmulOp) {
    if (!checkMatMul(matmulOp)) {
        return std::nullopt;
    }

    auto rhs = matmulOp.getInput2();

    // Walk from rhs: peel optional outer Transpose, then AffineReshape.
    IE::AffineReshapeOp reshapeOp = nullptr;
    IE::TransposeOp outerTransposeOp = rhs.getDefiningOp<IE::TransposeOp>();
    if (outerTransposeOp == nullptr) {
        reshapeOp = rhs.getDefiningOp<IE::AffineReshapeOp>();
    } else {
        if (!checkSwapLast2DimsTranspose(outerTransposeOp)) {
            return std::nullopt;
        }
        reshapeOp = outerTransposeOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    }

    if (!checkAffineReshape(reshapeOp)) {
        return std::nullopt;
    }

    // Walk from AffineReshape input: peel optional inner Transpose, then BroadcastOp.
    IE::TransposeOp innerTransposeOp = nullptr;
    IE::BroadcastOp broadCastOp = reshapeOp.getInput().getDefiningOp<IE::BroadcastOp>();
    if (broadCastOp == nullptr) {
        innerTransposeOp = reshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
        if (innerTransposeOp != nullptr) {
            if (!checkSwapLast2DimsTranspose(innerTransposeOp)) {
                return std::nullopt;
            }
            broadCastOp = innerTransposeOp.getInput().getDefiningOp<IE::BroadcastOp>();
        }
    }

    if (!checkBroadCast(broadCastOp)) {
        return std::nullopt;
    }

    return MatchedShrinkPattern{broadCastOp, innerTransposeOp, reshapeOp, outerTransposeOp};
}

}  // namespace IE
}  // namespace vpux
