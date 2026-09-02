// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/Utils/IndexingUtils.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOPSTOMATMUL
#define GEN_PASS_DEF_FUSEOPSTOMATMUL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Minimum M*N*K product to justify the MatMul rewrite.
constexpr int64_t MIN_MATMUL_ELEMENTS = 4096;
// Minimum number of elements to justify the CumSum→MatMul rewrite.
constexpr int64_t MIN_CUMSUM_MATMUL_ELEMENTS = 4096;
// Maximum axis length (L) for the CumSum→MatMul rewrite.
constexpr int64_t MAX_CUMSUM_MATMUL_L = 256;

struct BranchInfo {
    IE::TransposeOp transposeOp;
    IE::AffineReshapeOp reshapeOp;
    Shape newReshapeOutShape;
    // mapping from reshape output dims to transpose input dims
    SmallVector<SmallVector<int64_t>> newAffineMapping;
    SmallVector<int64_t> outDimsChanged;
    int64_t newAxis;
};

// Result of analyzing a dual-axis broadcast Multiply for MatMul conversion.
// Both rewriters share this common validation: NUMPY broadcast, element type, rank >= 3,
// exactly one broadcast-from-1 dim on each input, batch dimensions and M/N derivation.
struct DualAxisBroadcastInfo {
    int64_t lhsBroadcastDim;  // dim where LHS=1, RHS>1 (N axis)
    int64_t rhsBroadcastDim;  // dim where RHS=1, LHS>1 (M axis)
    int64_t batchSize;
    SmallVector<int64_t> batchDims;
    int64_t M;  // mulOutShape[rhsBroadcastDim]
    int64_t N;  // mulOutShape[lhsBroadcastDim]
    ShapeRef mulOutShape;
    size_t rank;
};

// Validates that mulOp is a dual-axis broadcast Multiply suitable for MatMul conversion.
// Returns std::nullopt when any precondition fails.
static std::optional<DualAxisBroadcastInfo> analyzeDualAxisBroadcastMultiply(IE::MultiplyOp mulOp) {
    // Require NUMPY broadcast
    if (mulOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        return std::nullopt;
    }

    // Guard element type: IE.MatMul only supports F16/F32/F64/SI32/quant.
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(mulOp.getOutput().getType()).getElementType();
    const bool isSupportedType = elemType.isF16() || elemType.isF32() || elemType.isF64() || elemType.isInteger(32) ||
                                 mlir::isa<mlir::quant::QuantizedType>(elemType);
    if (!isSupportedType) {
        return std::nullopt;
    }

    // Get shapes and verify rank >= 3 with matching dimensionality
    const auto mulOutShape = getShape(mulOp.getOutput());
    const auto lhsShape = getShape(mulOp.getInput1());
    const auto rhsShape = getShape(mulOp.getInput2());
    const auto rank = mulOutShape.size();

    if (rank < 3 || lhsShape.size() != rank || rhsShape.size() != rank) {
        return std::nullopt;
    }

    // Detect dual-axis broadcast: exactly one dim where LHS=1 and one where RHS=1
    int64_t lhsBroadcastDim = -1;
    int64_t rhsBroadcastDim = -1;

    for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
        const bool lhsIs1 = (lhsShape[Dim(i)] == 1 && mulOutShape[Dim(i)] > 1);
        const bool rhsIs1 = (rhsShape[Dim(i)] == 1 && mulOutShape[Dim(i)] > 1);
        if (lhsIs1 && !rhsIs1) {
            if (lhsBroadcastDim != -1) {
                return std::nullopt;  // multiple broadcast dims on LHS
            }
            lhsBroadcastDim = i;
        } else if (rhsIs1 && !lhsIs1) {
            if (rhsBroadcastDim != -1) {
                return std::nullopt;  // multiple broadcast dims on RHS
            }
            rhsBroadcastDim = i;
        }
    }

    if (lhsBroadcastDim == -1 || rhsBroadcastDim == -1) {
        return std::nullopt;
    }

    // Compute batch dimensions and their product
    int64_t batchSize = 1;
    SmallVector<int64_t> batchDims;
    for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
        if (i != lhsBroadcastDim && i != rhsBroadcastDim) {
            batchDims.push_back(i);
            batchSize *= mulOutShape[Dim(i)];
        }
    }

    const auto M = mulOutShape[Dim(rhsBroadcastDim)];
    const auto N = mulOutShape[Dim(lhsBroadcastDim)];

    return DualAxisBroadcastInfo{
            lhsBroadcastDim, rhsBroadcastDim, batchSize, std::move(batchDims), M, N, mulOutShape, rank};
}

// Returns true when a batched MatMul with the given dimensions can be lowered to VPU.NCE.MatMul on the current target.
static bool isMatMulBeneficialAsNCEMatMul(IE::MultiplyOp mulOp, int64_t batchSize, int64_t M, int64_t N, int64_t K,
                                          bool enableGroupedMatMul) {
    if (!enableGroupedMatMul) {
        return false;
    }

    if (M * N * K < MIN_MATMUL_ELEMENTS) {
        return false;
    }

    const auto moduleOp = mulOp->getParentOfType<mlir::ModuleOp>();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(mulOp.getOutput().getType());

    const Shape input5DShape{batchSize, static_cast<int64_t>(1), K, M, static_cast<int64_t>(1)};
    const Shape filter5DShape{batchSize, N, K, static_cast<int64_t>(1), static_cast<int64_t>(1)};
    const Shape output5DShape{batchSize, static_cast<int64_t>(1), N, M, static_cast<int64_t>(1)};

    const auto emptyLogCb = [](const formatv_object_base&) {};
    return VPU::isNCEMatMulSupported(outputType.changeShape(input5DShape), outputType.changeShape(filter5DShape),
                                     outputType.changeShape(output5DShape), moduleOp, emptyLogCb,
                                     /*checkLayout=*/false, /*checkChannelAlignment=*/false);
}

// Returns the indices of output dimensions whose source input dimensions
// are different from their original positions.
//
// For a pure transpose, the result contains all output indices `i` where
// perm[i] != i.
//
// Example:
// Consider a transpose from:
//   input shape : d0 x d1 x d2 x d3
//   output shape: d0 x d2 x d3 x d1
//
// The permutation map is:
//   perm = (d0, d2, d3, d1)  // i.e., [0, 2, 3, 1]
//
// Index mapping:
//   output dim 0 <- input dim 0   (unchanged)
//   output dim 1 <- input dim 2   (changed)
//   output dim 2 <- input dim 3   (changed)
//   output dim 3 <- input dim 1   (changed)
//
// Therefore, the returned changed dims are:
//   [1, 2, 3]
//
static SmallVector<int64_t> getChangedDimsByTranspose(mlir::AffineMap perm) {
    SmallVector<int64_t> changed;
    for (unsigned i = 0; i < perm.getNumResults(); ++i) {
        auto expr = perm.getResult(i);
        if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
            const unsigned pos = dimExpr.getPosition();
            if (pos != i) {
                changed.push_back(static_cast<int64_t>(i));
            }
        } else {
            // Non-trivial mapping (not a pure permutation) — return empty to indicate unsupported
            return {};
        }
    }
    return changed;
}

// Returns the output dimensions that correspond to input dimensions unchanged
// (i.e., not split, merged, or resized) by the given AffineReshapeOp.
//
// Example:
// Case 1: inShape=[1, 4, 64, 256, 128], outShape=[1, 4, 64, 256, 128, 1], and dim_mapping=[[0], [1], [2], [3], [4, 5]]
// With inDims=[2, 3], would return [2, 3]
//
// Case 2: inShape=[1, 4, 64, 64, 256], outShape=[1, 4, 1, 64, 64, 256], and dim_mapping=[[0], [1, 2], [3], [4], [5]]
// With inDims=[3, 4], would return [4, 5]
//
// Case 3: inShape=[1, 4, 64, 256], outShape=[1, 4, 64, 128, 2], and dim_mapping=[[0], [1], [2], [3, 4]]
// With inDims=[3], would return [] since in-dim 3 is split into out-dim 3 and 4
//
// Case 4: inShape=[1, 4, 64, 256], outShape=[1, 4, 64, 256], and dim_mapping=[[0], [1], [2], [3]]
// With inDims=[0, 1, 2, 3], would return [0, 1, 2, 3]
//
// Case 5: inShape=[1, 4, 64, 256], outShape=[1, 4, 256, 64], and dim_mapping=[[0], [1], [3], [2]]
// With inDims=[2, 3], would return [] since in-dims 2 and 3 are permuted
//
// Case 6: inShape=[1, 4, 64, 256], outShape=[1, 4, 128, 128], and dim_mapping=[[0], [1], [2, 3], [2, 3]]
// With inDims=[2, 3], would return [] since in-dims 2 and 3 are merged
//
// Case 7: inShape=[1, 4, 64, 256], outShape=[1, 4, 128, 128], and dim_mapping=[[0], [1], [2], [3]]
// With inDims=[2, 3], would return [] since in-dims 2 and 3 are resized
//
static SmallVector<int64_t> areDimsNotChangedByReshape(IE::AffineReshapeOp affine, ArrayRef<int64_t> inDims) {
    const auto affineInShape = getShape(affine.getInput());
    const auto affineOutShape = getShape(affine.getOutput());
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affine.getDimMapping());

    int64_t prevOutDim = -1;
    SmallVector<int64_t> outDims;
    for (auto inDim : inDims) {
        if (inDim < 0 || static_cast<size_t>(inDim) >= dimMapping.size()) {
            return {};
        }
        const auto inDimLen = affineInShape[Dim(inDim)];
        const auto& mappedOutDims = dimMapping[static_cast<size_t>(inDim)];
        for (auto mappedOut : mappedOutDims) {
            const auto outDimLen = affineOutShape[Dim(mappedOut)];
            if (outDimLen == 1 && inDimLen != 1) {
                continue;
            }
            if (outDimLen != inDimLen) {
                return {};
            }
            if (prevOutDim != -1 && mappedOut <= prevOutDim) {
                return {};
            }
            prevOutDim = mappedOut;
            outDims.push_back(mappedOut);
        }
    }
    return outDims;
}

class BroadcastMultiplyReduceSumToMatMulRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    BroadcastMultiplyReduceSumToMatMulRewriter(mlir::MLIRContext* ctx, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
        setDebugName("BroadcastMultiplyReduceSumToMatMulRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult BroadcastMultiplyReduceSumToMatMulRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum at '{1}'", getDebugName(), origOp->getLoc());

    // 1. The input to ReduceSum must be a single-use broadcast Multiply
    auto mulOp = origOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (mulOp == nullptr || !mulOp->hasOneUse()) {
        return mlir::failure();
    }

    // 2. Validate dual-axis broadcast Multiply (NUMPY, element type, rank, broadcast dims)
    const auto info = analyzeDualAxisBroadcastMultiply(mulOp);
    if (!info.has_value()) {
        return mlir::failure();
    }

    const auto lhsBroadcastDim = info->lhsBroadcastDim;
    const auto rhsBroadcastDim = info->rhsBroadcastDim;
    const auto mulOutShape = info->mulOutShape;
    const auto rank = info->rank;
    const auto lhsShape = getShape(mulOp.getInput1());
    const auto rhsShape = getShape(mulOp.getInput2());

    // 3. The ReduceSum must reduce over exactly one contraction (K) dimension that is
    //    neither of the two broadcast dims (those are M and N).
    const auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue());
    if (axes.size() != 1) {
        return mlir::failure();
    }

    // Normalize the axis: convert negative values to positive and validate bounds.
    int64_t reduceAxis = axes.front();
    if (reduceAxis < 0) {
        reduceAxis += static_cast<int64_t>(rank);
    }
    if (reduceAxis < 0 || reduceAxis >= static_cast<int64_t>(rank)) {
        return mlir::failure();
    }
    if (reduceAxis == lhsBroadcastDim || reduceAxis == rhsBroadcastDim) {
        return mlir::failure();
    }

    // Both inputs must have the same size on the K (contraction) dimension.
    if (lhsShape[Dim(reduceAxis)] != rhsShape[Dim(reduceAxis)]) {
        return mlir::failure();
    }

    // 4b. When reduceAxis is not the last dim, compute the permutation that moves K to last.
    //     All downstream steps use the "effective" (post-permutation) dim indices and shape.
    const bool needKToLastTranspose = (reduceAxis != static_cast<int64_t>(rank) - 1);
    SmallVector<int64_t> kToLastPerm;
    for (size_t i = 0; i < rank; ++i) {
        if (static_cast<int64_t>(i) != reduceAxis) {
            kToLastPerm.push_back(static_cast<int64_t>(i));
        }
    }
    kToLastPerm.push_back(static_cast<int64_t>(reduceAxis));

    // Map original dim index d → its new position after kToLastPerm.
    const auto remapDim = [&](int64_t d) -> int64_t {
        for (size_t i = 0; i < kToLastPerm.size(); ++i) {
            if (kToLastPerm[i] == d) {
                return static_cast<int64_t>(i);
            }
        }
        VPUX_THROW("remapDim: dim {0} not found in kToLastPerm", d);
    };

    const int64_t effectiveLhsBroadcastDim = needKToLastTranspose ? remapDim(lhsBroadcastDim) : lhsBroadcastDim;
    const int64_t effectiveRhsBroadcastDim = needKToLastTranspose ? remapDim(rhsBroadcastDim) : rhsBroadcastDim;
    const int64_t effectiveReduceAxis = static_cast<int64_t>(rank) - 1;

    // Virtual output shape after kToLastPerm (used for M/N/K/batchSize derivation).
    const auto effectiveOutShapeVec = mlir::applyPermutation(mulOutShape.raw(), kToLastPerm);

    // Compute the batch size using effective dims.
    int64_t batchSize = 1;
    SmallVector<int64_t> batchDims;
    for (size_t i = 0; i < rank; ++i) {
        if (static_cast<int64_t>(i) != effectiveLhsBroadcastDim &&
            static_cast<int64_t>(i) != effectiveRhsBroadcastDim && static_cast<int64_t>(i) != effectiveReduceAxis) {
            batchDims.push_back(static_cast<int64_t>(i));
            batchSize *= effectiveOutShapeVec[i];
        }
    }

    const auto M = effectiveOutShapeVec[effectiveRhsBroadcastDim];
    const auto N = effectiveOutShapeVec[effectiveLhsBroadcastDim];
    const auto K = effectiveOutShapeVec[effectiveReduceAxis];

    _log.trace("[{0}] Detected outer product + reduce: batch={1}, M={2}, N={3}, K={4}", getDebugName(), batchSize, M, N,
               K);

    // 5. Guard: only convert when the resulting IE.MatMul can be lowered to VPU.NCE.MatMul.
    if (!isMatMulBeneficialAsNCEMatMul(mulOp, batchSize, M, N, K, _enableGroupedMatMul)) {
        _log.trace("[{0}] NCE.MatMul not supported for batchSize={1}, skipping conversion", getDebugName(), batchSize);
        return mlir::failure();
    }

    // 6. Build the MatMul
    const auto origLoc = origOp->getLoc();
    const auto ctx = rewriter.getContext();

    // 6a. If reduceAxis is not the last dim, pre-transpose both inputs to move K to last.
    //     The broadcast dim (size=1) in each input is also moved, which is harmless.
    mlir::Value lhsInput = mulOp.getInput1();
    mlir::Value rhsInput = mulOp.getInput2();
    if (needKToLastTranspose) {
        SmallVector<unsigned> kToLastPermUnsigned;
        kToLastPermUnsigned.reserve(kToLastPerm.size());
        for (const auto dim : kToLastPerm) {
            kToLastPermUnsigned.push_back(checked_cast<unsigned>(dim));
        }
        const auto kToLastPermAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(kToLastPermUnsigned, ctx));
        lhsInput = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "lhs_k_to_last"), lhsInput, nullptr,
                                                    kToLastPermAttr)
                           .getOutput();
        rhsInput = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "rhs_k_to_last"), rhsInput, nullptr,
                                                    kToLastPermAttr)
                           .getOutput();
    }

    // Check whether the input dims are already in the order needed for a direct reshape.
    // A Transpose is needed when this order differs from the identity [0,1,...,rank-1].
    const auto buildPerm = [&](int64_t mOrNDim, int64_t broadcastDim) -> SmallVector<unsigned> {
        SmallVector<unsigned> perm;
        for (const auto d : batchDims) {
            perm.push_back(checked_cast<unsigned>(d));
        }
        perm.push_back(checked_cast<unsigned>(mOrNDim));
        perm.push_back(checked_cast<unsigned>(broadcastDim));
        perm.push_back(checked_cast<unsigned>(effectiveReduceAxis));
        return perm;
    };

    // Build a transposed+reshaped 3D tensor [batchSize, mOrN, K] from the given input.
    const auto prepareInput = [&](mlir::Value input, int64_t mOrNDim, int64_t broadcastDim, int64_t mOrN,
                                  StringRef tag) -> mlir::Value {
        mlir::Value val = input;
        const auto perm = buildPerm(mOrNDim, broadcastDim);
        const auto permMap = mlir::AffineMap::getPermutationMap(perm, ctx);
        const auto inMemShape = MemShape(getShape(val).raw());
        if (!isTrivialPermute(inMemShape, permMap)) {
            val = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, (tag + "_transpose").str()), val, nullptr,
                                                   mlir::AffineMapAttr::get(permMap))
                          .getOutput();
        }
        // Reshape to [batchSize, mOrN, K], squeezing the broadcast-1 dim.
        const auto reshapeShape = SmallVector<int64_t>{batchSize, mOrN, K};
        return rewriter
                .create<IE::ReshapeOp>(appendLoc(origLoc, (tag + "_reshape").str()), val,
                                       getIntArrayAttr(ctx, reshapeShape))
                .getOutput();
    };

    // LHS: [batch, M, K]
    auto lhsReshaped = prepareInput(lhsInput, effectiveRhsBroadcastDim, effectiveLhsBroadcastDim, M, "matmul_lhs");
    // RHS: [batch, N, K]
    auto rhsReshaped = prepareInput(rhsInput, effectiveLhsBroadcastDim, effectiveRhsBroadcastDim, N, "matmul_rhs");

    // MatMul: [batch, M, K] @ [batch, N, K]^T -> [batch, M, N]
    auto matmulOp = rewriter.create<IE::MatMulOp>(appendLoc(origLoc, "broadcast_mul_reduce_as_matmul"), lhsReshaped,
                                                  rhsReshaped,
                                                  /*transpose_a=*/false, /*transpose_b=*/true);

    // 7. Reshape + optional Transpose output to match ReduceSum output shape.
    // The MatMul output [batchSize, M, N] is expanded back to (rank-1) dims in effective space,
    // then an inverse Transpose restores the original dim order (excluding K which was reduced).
    mlir::Value outValue = matmulOp.getOutput();
    const auto outPerm = buildPerm(effectiveRhsBroadcastDim, effectiveLhsBroadcastDim);
    const SmallVector<int64_t> outPermI64(outPerm.begin(), outPerm.end());
    const bool anyTransposed = !mlir::isIdentityPermutation(outPermI64);
    if (anyTransposed) {
        // Permuted shape: [batchDim sizes in batchDims order, M, N]
        SmallVector<int64_t> permutedShape;
        for (auto d : batchDims) {
            permutedShape.push_back(effectiveOutShapeVec[d]);
        }
        permutedShape.push_back(M);
        permutedShape.push_back(N);
        outValue = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "matmul_out_expand"), outValue,
                                                  getIntArrayAttr(ctx, permutedShape))
                           .getOutput();

        SmallVector<unsigned> currentOrder;
        currentOrder.reserve(batchDims.size() + 2);
        for (const auto d : batchDims) {
            currentOrder.push_back(checked_cast<unsigned>(d));
        }
        currentOrder.push_back(checked_cast<unsigned>(effectiveRhsBroadcastDim));
        currentOrder.push_back(checked_cast<unsigned>(effectiveLhsBroadcastDim));

        const auto currentOrderMap = mlir::AffineMap::getPermutationMap(currentOrder, ctx);
        const auto invPermMap = mlir::inversePermutation(currentOrderMap);

        if (!invPermMap.isIdentity()) {
            const auto invPermAttr = mlir::AffineMapAttr::get(invPermMap);
            outValue = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "matmul_out_transpose"), outValue, nullptr,
                                                        invPermAttr)
                               .getOutput();
        }
    }

    // Final reshape to the exact ReduceSum output shape (restores keep_dims size-1 if needed).
    const auto reducedShape = getShape(origOp.getOutput());
    auto outReshaped = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "matmul_out_reshape"), outValue,
                                                      getIntArrayAttr(ctx, reducedShape));

    rewriter.replaceOp(origOp, outReshaped.getOutput());

    _log.trace("[{0}] Replaced broadcast Multiply + ReduceSum with MatMul at '{1}'", getDebugName(), origLoc);
    return mlir::success();
}

//
// OuterProductMultiplyToMatMulRewriter
//
// Converts a standalone dual-axis broadcast Multiply (outer product) into an IE.MatMul.
// Pattern: [B,M,1] * [B,1,N] -> [B,M,N]  =>  MatMul([B,M,1], [B,N,1], transpose_b=true)
// Unlike BroadcastMultiplyReduceSumToMatMulRewriter, no ReduceSum consumer is required.
//

class OuterProductMultiplyToMatMulRewriter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    OuterProductMultiplyToMatMulRewriter(mlir::MLIRContext* ctx, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
        setDebugName("OuterProductMultiplyToMatMulRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult OuterProductMultiplyToMatMulRewriter::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Multiply at '{1}'", getDebugName(), mulOp->getLoc());

    // Skip if the sole user is a ReduceSum — BroadcastMultiplyReduceSumToMatMulRewriter
    // handles the combined Multiply+ReduceSum → MatMul fusion more efficiently (K > 1).
    // Multi-use with a ReduceSum user is still converted: the outer product rewrite is
    // cheap (Reshape only, no Transpose), and the dedicated rewriter requires single use.
    if (mulOp->hasOneUse() && mlir::isa<IE::ReduceSumOp>(*mulOp->getUsers().begin())) {
        return mlir::failure();
    }

    // Skip inputs with explicit non-default layout order (e.g. NHWC).
    // The Reshape/Transpose sequence below assumes the default dims order for the given rank;
    // applying it to non-default-order tensors produces type mismatches that fail to legalize
    // downstream.
    const auto lhsOrder = DimsOrder::fromValue(mulOp.getInput1());
    const auto rhsOrder = DimsOrder::fromValue(mulOp.getInput2());
    if (lhsOrder != DimsOrder::fromNumDims(lhsOrder.numDims()) ||
        rhsOrder != DimsOrder::fromNumDims(rhsOrder.numDims())) {
        return mlir::failure();
    }

    // Validate dual-axis broadcast Multiply (NUMPY, element type, rank, broadcast dims)
    const auto info = analyzeDualAxisBroadcastMultiply(mulOp);
    if (!info.has_value()) {
        return mlir::failure();
    }

    const auto lhsBroadcastDim = info->lhsBroadcastDim;
    const auto rhsBroadcastDim = info->rhsBroadcastDim;
    const auto batchSize = info->batchSize;
    const auto& batchDims = info->batchDims;
    const auto M = info->M;
    const auto N = info->N;
    const auto mulOutShape = info->mulOutShape;

    // Skip when batch dims and broadcast dims are interleaved (e.g. [B,M,N,B]).
    // Interleaved ordering requires Transpose ops for both inputs and the output.
    // For outer product (K=1) the MatMul computation is trivial, so the Transpose
    // overhead negates the benefit of DPU acceleration.
    SmallVector<int64_t> dimOrder;
    dimOrder.append(batchDims.begin(), batchDims.end());
    dimOrder.push_back(rhsBroadcastDim);
    dimOrder.push_back(lhsBroadcastDim);
    if (!llvm::is_sorted(dimOrder)) {
        _log.trace("[{0}] Interleaved dim ordering requires Transpose, skipping", getDebugName());
        return mlir::failure();
    }

    _log.trace("[{0}] Detected outer product: batch={1}, M={2}, N={3}", getDebugName(), batchSize, M, N);

    // Verify NCE.MatMul can handle this (K=1 for outer product)
    if (!isMatMulBeneficialAsNCEMatMul(mulOp, batchSize, M, N, /*K=*/1, _enableGroupedMatMul)) {
        _log.trace("[{0}] NCE.MatMul not beneficial for batchSize={1}", getDebugName(), batchSize);
        return mlir::failure();
    }

    const auto origLoc = mulOp->getLoc();
    const auto ctx = rewriter.getContext();

    // The interleaved-dims guard above ensures [batchDims..., M_dim, N_dim] is monotonically
    // increasing, so Reshape alone suffices — no Transpose is needed.

    // LHS: reshape to [batchSize, M, 1]
    auto lhsVal = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "outer_lhs_reshape"), mulOp.getInput1(),
                                                 getIntArrayAttr(ctx, SmallVector<int64_t>{batchSize, M, 1}))
                          .getOutput();

    // RHS: reshape to [batchSize, N, 1]
    auto rhsVal = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "outer_rhs_reshape"), mulOp.getInput2(),
                                                 getIntArrayAttr(ctx, SmallVector<int64_t>{batchSize, N, 1}))
                          .getOutput();

    // MatMul: [batchSize, M, 1] @ [batchSize, N, 1]^T = [batchSize, M, N]
    auto matmulOp = rewriter.create<IE::MatMulOp>(appendLoc(origLoc, "outer_product_as_matmul"), lhsVal, rhsVal,
                                                  /*transpose_a=*/false, /*transpose_b=*/true);

    // Reshape MatMul output [batchSize, M, N] back to the original Multiply output shape.
    // The early guard ensures dim ordering is monotonically increasing (not interleaved),
    // so no output Transpose is needed — a single Reshape suffices.
    auto finalReshape = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "outer_out_reshape"), matmulOp.getOutput(),
                                                       getIntArrayAttr(ctx, mulOutShape.raw()));
    rewriter.replaceOp(mulOp, finalReshape.getOutput());

    _log.trace("[{0}] Replaced outer product Multiply with MatMul at '{1}'", getDebugName(), origLoc);
    return mlir::success();
}

//
// PropagateTransposesRewriter
//
// Full matched subgraph example (Before rewrite):
//
//    Input1                        Input2
//      |                             |
//   Transpose                     Transpose
//      |                             |
//   AffineReshape                AffineReshape
//      |                             |
//   -----------------------------------------
//  |               Multiply                  |
//   -----------------------------------------
//                    |
//                ReduceSum
//                    |
//
// After rewrite the pre-Transpose ops are removed and new AffineReshape ops are created
// directly from the original Transpose inputs.
//
//    Input1                        Input2
//      |                             |
//   AffineReshape                AffineReshape
//      |                             |
//   -----------------------------------------
//  |               Multiply                  |
//   -----------------------------------------
//                    |
//                ReduceSum
//                    |
//
// This rewrite is applicable when follwoing conditions are met:
// 1. Tranpose ops on both branches have the same permutation and swap exactly two adjacent dimensions.
// 2. AffineReshape ops on both branches do not change the swapped dimensions (e.g. by splitting/merging or changing
// their size).
// 3. The ReduceSum reduces over exactly one of the two dimensions that are swapped by the Transpose.
// This ensures that the Transpose ops can be safely removed without introducing additional Transpose ops to restore the
// original dim order.

class PropagateTransposeRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    PropagateTransposeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("PropagateTransposeRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::optional<BranchInfo> checkInputBranch(mlir::Value input, int64_t axis) const;
    IE::AffineReshapeOp createNewReshapeOp(BranchInfo& inputBranch, mlir::PatternRewriter& rewriter) const;
    bool checkTwoInputBranches(BranchInfo& input1, BranchInfo& input2) const;

    Logger _log;
};

std::optional<BranchInfo> PropagateTransposeRewriter::checkInputBranch(mlir::Value input, int64_t axis) const {
    // Check both inputs follow Transpose -> AffineReshape
    auto reshapeOp = input.getDefiningOp<IE::AffineReshapeOp>();
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
        return std::nullopt;
    }

    auto transOp = reshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transOp == nullptr || !transOp->hasOneUse()) {
        return std::nullopt;
    }

    if (!transOp.getOrderValue().has_value()) {
        return std::nullopt;
    }

    // The changed dims are adjacent.
    const auto origTransPerm = transOp.getOrderValue().value();
    const auto inDimsChanged = getChangedDimsByTranspose(origTransPerm);
    if (inDimsChanged.size() != 2 || std::abs(inDimsChanged[0] - inDimsChanged[1]) != 1) {
        return std::nullopt;
    }
    // The reshape must not change the changed dims (e.g. by splitting/merging or changing their size).
    auto outDimsMapping = areDimsNotChangedByReshape(reshapeOp, inDimsChanged);
    if (outDimsMapping.size() != 2 ||
        (outDimsMapping[0] - inDimsChanged[0]) != (outDimsMapping[1] - inDimsChanged[1])) {
        return std::nullopt;
    }

    auto reshapeOutShape = getShape(reshapeOp.getOutput());
    auto newReshapeOutShape = reshapeOutShape.toValues();
    std::swap(newReshapeOutShape[Dim(outDimsMapping[0])], newReshapeOutShape[Dim(outDimsMapping[1])]);

    auto newInputShape = getShape(transOp.getInput());
    auto newReshapeMap = IE::getReassociationMap(newInputShape, ShapeRef(newReshapeOutShape));
    if (mlir::failed(newReshapeMap)) {
        _log.trace("Failed to get reassociation map for new AffineReshape, {0} => {1}", newInputShape,
                   newReshapeOutShape);
        return std::nullopt;
    }

    const auto axisInChangedDims = llvm::is_contained(outDimsMapping, axis);
    if (!axisInChangedDims) {
        return std::nullopt;
    }

    // If the reduced axis is one of the changed dims, we need to update it to point to the new position after dimension
    // swap.
    auto newAxis = axis == outDimsMapping[0] ? outDimsMapping[1] : outDimsMapping[0];

    return BranchInfo{
            transOp, reshapeOp, std::move(newReshapeOutShape), newReshapeMap.value(), std::move(outDimsMapping),
            newAxis};
}

bool PropagateTransposeRewriter::checkTwoInputBranches(BranchInfo& input1, BranchInfo& input2) const {
    auto& trans1 = input1.transposeOp;
    auto& aff1 = input1.reshapeOp;
    auto& trans2 = input2.transposeOp;
    auto& aff2 = input2.reshapeOp;

    // Require both input transposes to have identical permutation
    if (trans1.getOrderValue().value() != trans2.getOrderValue().value()) {
        return false;
    }
    if (DimsOrder::fromValue(trans1.getInput()) != DimsOrder::fromValue(trans2.getInput())) {
        return false;
    }

    if (DimsOrder::fromValue(aff1.getOutput()) != DimsOrder::fromValue(aff2.getOutput())) {
        return false;
    }
    if (input1.outDimsChanged != input2.outDimsChanged || input1.newAxis != input2.newAxis) {
        return false;
    }
    return true;
}

IE::AffineReshapeOp PropagateTransposeRewriter::createNewReshapeOp(BranchInfo& inputBranch,
                                                                   mlir::PatternRewriter& rewriter) const {
    const auto newReshapeOutShapeAttr = getIntArrayAttr(rewriter.getContext(), inputBranch.newReshapeOutShape);
    const auto newReshapeMapAttr = getIntArrayOfArray(rewriter.getContext(), inputBranch.newAffineMapping);
    return rewriter.create<IE::AffineReshapeOp>(appendLoc(inputBranch.reshapeOp->getLoc(), "reshape_input"),
                                                inputBranch.transposeOp.getInput(), newReshapeMapAttr,
                                                newReshapeOutShapeAttr);
}

mlir::LogicalResult PropagateTransposeRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum at '{1}'", getDebugName(), origOp->getLoc());

    if (origOp.getKeepDims() || origOp.getInputPaddingAttr() != nullptr || origOp.getOutputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    const auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue());
    if (axes.size() != 1) {
        return mlir::failure();
    }
    auto axis = axes.front();

    auto mulOp = origOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (mulOp == nullptr || !mulOp->hasOneUse()) {
        return mlir::failure();
    }

    if (mulOp.getInputPaddingAttr() != nullptr || mulOp.getOutputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    const auto branchInfo1 = checkInputBranch(mulOp.getInput1(), axis);
    if (!branchInfo1.has_value()) {
        return mlir::failure();
    }
    const auto branchInfo2 = checkInputBranch(mulOp.getInput2(), axis);
    if (!branchInfo2.has_value()) {
        return mlir::failure();
    }

    auto inputBranch1Info = branchInfo1.value();
    auto inputBranch2Info = branchInfo2.value();
    if (!checkTwoInputBranches(inputBranch1Info, inputBranch2Info)) {
        return mlir::failure();
    }

    // Build new Multiply that consumes original transpose inputs (i.e., remove pre-transposes)
    const auto origLoc = origOp->getLoc();
    const auto ctx = rewriter.getContext();

    auto in1NewReshapeOp = createNewReshapeOp(inputBranch1Info, rewriter);
    auto in2NewReshapeOp = createNewReshapeOp(inputBranch2Info, rewriter);

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            appendLoc(mulOp->getLoc(), "multiply_in"), in1NewReshapeOp.getOutput(), in2NewReshapeOp.getOutput(),
            mulOp.getAutoBroadcastAttr(), mulOp.getPostOpAttr(), mulOp.getClampAttr(), nullptr, nullptr);

    const auto newAxesAttr = getIntArrayAttr(ctx, ArrayRef<int64_t>{inputBranch1Info.newAxis});
    auto newReduceSumOp = rewriter.create<IE::ReduceSumOp>(appendLoc(origLoc, "changed"), newMultiply.getOutput(),
                                                           newAxesAttr, origOp.getKeepDimsAttr(), nullptr, nullptr);

    rewriter.replaceOp(origOp, newReduceSumOp.getOutput());

    _log.trace("[{0}] Removed input Transpose->AffineReshape chains and replaced Multiply at '{1}'", getDebugName(),
               origLoc);
    return mlir::success();
}

//
// CumSumToMatMulRewriter
//

class CumSumToMatMulRewriter final : public mlir::OpRewritePattern<IE::CumSumOp> {
public:
    CumSumToMatMulRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::CumSumOp>(ctx), _log(log) {
        setDebugName("CumSumToMatMulRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::CumSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isSupportedCumSum(IE::CumSumOp origOp) const;
    mlir::Value buildTriangularWeights(mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t L,
                                       mlir::Type elemType, bool exclusive, bool reverse) const;

    Logger _log;
};

bool CumSumToMatMulRewriter::isSupportedCumSum(IE::CumSumOp origOp) const {
    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inputElemType = inputType.getElementType();
    if (!mlir::isa<mlir::Float32Type, mlir::Float16Type>(inputElemType)) {
        _log.trace("Unsupported element type");
        return false;
    }

    const auto inputTensorType = mlir::cast<mlir::RankedTensorType>(origOp.getInput().getType());
    if (!inputTensorType.hasStaticShape()) {
        _log.trace("Dynamic shapes are not supported");
        return false;
    }

    const auto rank = inputType.getRank();
    if (rank < 2) {
        _log.trace("Input rank must be >= 2");
        return false;
    }

    if (!origOp.getAxisValue().has_value()) {
        _log.trace("Dynamic axis not supported (requires canonicalization first)");
        return false;
    }

    auto axis = origOp.getAxisValue().value();
    if (axis < 0) {
        axis += rank;
    }

    const auto inputShape = inputType.getShape();
    const auto L = inputShape[Dim(axis)];

    int64_t batchSize = 1;
    for (int64_t i = 0; i < axis; ++i) {
        batchSize *= inputShape[Dim(i)];
    }
    int64_t trailingSize = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        trailingSize *= inputShape[Dim(i)];
    }

    if (batchSize * L * trailingSize < MIN_CUMSUM_MATMUL_ELEMENTS) {
        _log.trace("Too few elements ({0}), skipping conversion", batchSize * L * trailingSize);
        return false;
    }

    if (L > MAX_CUMSUM_MATMUL_L) {
        _log.trace("Axis length L={0} exceeds limit {1}, skipping conversion", L, MAX_CUMSUM_MATMUL_L);
        return false;
    }

    const auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    if (L % alignment != 0) {
        _log.trace("Axis length L={0} is not a multiple of alignment {1}, skipping conversion", L, alignment);
        return false;
    }

    return true;
}

mlir::Value CumSumToMatMulRewriter::buildTriangularWeights(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                           int64_t L, mlir::Type elemType, bool exclusive,
                                                           bool reverse) const {
    const Shape weightShape = {L, L};
    SmallVector<float> weights(L * L, 0.0f);

    for (int64_t k = 0; k < L; ++k) {
        for (int64_t i = 0; i < L; ++i) {
            bool active = false;
            if (!reverse && !exclusive) {
                active = (k <= i);
            } else if (!reverse && exclusive) {
                active = (k < i);
            } else if (reverse && !exclusive) {
                active = (k >= i);
            } else {
                active = (k > i);
            }
            if (active) {
                weights[k * L + i] = 1.0f;
            }
        }
    }

    const auto weightType = mlir::RankedTensorType::get(weightShape.raw(), elemType);
    return Const::createFloatConst(rewriter, loc, weightType, ArrayRef<float>(weights));
}

mlir::LogicalResult CumSumToMatMulRewriter::matchAndRewrite(IE::CumSumOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto origLoc = origOp->getLoc();
    _log.trace("[{0}] Got CumSum at '{1}'", getDebugName(), origLoc);

    if (!isSupportedCumSum(origOp)) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto rank = inputType.getRank();
    const auto inputShape = inputType.getShape();
    const auto ctx = rewriter.getContext();

    auto axis = origOp.getAxisValue().value();
    if (axis < 0) {
        axis += rank;
    }

    const auto L = inputShape[Dim(axis)];

    int64_t batchSize = 1;
    for (int64_t i = 0; i < axis; ++i) {
        batchSize *= inputShape[Dim(i)];
    }
    int64_t trailingSize = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        trailingSize *= inputShape[Dim(i)];
    }

    // Step 1: Transpose to move axis dim to last position.
    SmallVector<uint32_t> toLastPerm;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != axis) {
            toLastPerm.push_back(static_cast<uint32_t>(i));
        }
    }
    toLastPerm.push_back(static_cast<uint32_t>(axis));

    const auto toLastPermMap = mlir::AffineMap::getPermutationMap(toLastPerm, ctx);
    mlir::Value transposed = origOp.getInput();
    if (!toLastPermMap.isIdentity()) {
        transposed = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "axis_to_last"), origOp.getInput(), nullptr,
                                                      mlir::AffineMapAttr::get(toLastPermMap))
                             .getOutput();
    }

    // Step 2: Reshape to [1, batch*trailing, L].
    const SmallVector<int64_t> flatShape = {1, batchSize * trailingSize, L};
    auto reshapedInput = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "input_reshape"), transposed,
                                                        getIntArrayAttr(ctx, flatShape));

    // Step 3: Build triangular weight [L, L] and MatMul.
    const bool exclusive = origOp.getExclusive();
    const bool reverse = origOp.getReverse();
    mlir::Value matmulWeight =
            buildTriangularWeights(rewriter, origLoc, L, inputType.getElementType(), exclusive, reverse);

    auto matmulOp = rewriter.create<IE::MatMulOp>(appendLoc(origLoc, "cumsum_as_matmul"), reshapedInput.getOutput(),
                                                  matmulWeight, /*transposeA=*/false, /*transposeB=*/false);

    // Step 4: Reshape back to pre-transpose shape.
    SmallVector<int64_t> preTransposeShape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != axis) {
            preTransposeShape.push_back(inputShape[Dim(i)]);
        }
    }
    preTransposeShape.push_back(L);
    auto outReshape = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "output_reshape"), matmulOp.getOutput(),
                                                     getIntArrayAttr(ctx, preTransposeShape));

    // Step 5: Transpose back to original axis position.
    mlir::Value result = outReshape.getOutput();
    if (!toLastPermMap.isIdentity()) {
        const auto invPermMap = mlir::inversePermutation(toLastPermMap);
        result = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "axis_restore"), result, nullptr,
                                                  mlir::AffineMapAttr::get(invPermMap))
                         .getOutput();
    }

    rewriter.replaceOp(origOp, result);

    _log.trace("[{0}] Successfully converted CumSum to MatMul at '{1}'", getDebugName(), origLoc);
    return mlir::success();
}

//
// FuseOpsToMatMulPass
//

class FuseOpsToMatMulPass final : public IE::impl::FuseOpsToMatMulBase<FuseOpsToMatMulPass> {
public:
    explicit FuseOpsToMatMulPass(const bool enableGroupedMatMul, const bool enableConvertCumSumToMatMul, Logger log)
            : _enableGroupedMatMul(enableGroupedMatMul), _enableConvertCumSumToMatMul(enableConvertCumSumToMatMul) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableGroupedMatMul;
    bool _enableConvertCumSumToMatMul;
};

mlir::LogicalResult FuseOpsToMatMulPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableGroupedMatMul.hasValue()) {
        _enableGroupedMatMul = enableGroupedMatMul;
    }
    if (enableConvertCumSumToMatMul.hasValue()) {
        _enableConvertCumSumToMatMul = enableConvertCumSumToMatMul;
    }
    return mlir::success();
}

void FuseOpsToMatMulPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet adjustPatterns(&ctx);
    adjustPatterns.add<PropagateTransposeRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(adjustPatterns));

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BroadcastMultiplyReduceSumToMatMulRewriter>(&ctx, _log, _enableGroupedMatMul);
    patterns.add<OuterProductMultiplyToMatMulRewriter>(&ctx, _log, _enableGroupedMatMul);
    if (_enableConvertCumSumToMatMul) {
        patterns.add<CumSumToMatMulRewriter>(&ctx, _log);
    }

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseOpsToMatMulPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOpsToMatMulPass(const bool enableGroupedMatMul,
                                                                const bool enableConvertCumSumToMatMul, Logger log) {
    return std::make_unique<FuseOpsToMatMulPass>(enableGroupedMatMul, enableConvertCumSumToMatMul, log);
}
