//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Convert IE::AttentionOp whose K and V inputs are block-based KV-cache (Reshape-wrapped
// Concat or bare Concat) into a cascade of IE::FlashSDPAOp per merged KV-cache block group.
// "Block-based" refers to paged/blocked KV-cache tiles (one Concat input per cached block),
// not to VPU-level computation tiling (which is handled in the VPU dialect).
//
// Background:
//   fuse_attention writes K and V operands of AttentionOp as:
//     Reshape(Concat([tile0, tile1, ...])) — when GQA head-folding is applied
//     Concat([tile0, tile1, ...])          — bare, for non-GQA paths
//   Each tile corresponds to one KV-cache block.
//
// What this pass does:
//   Converts the AttentionOp into a chain of FlashSDPAOp that processes each tile
//   (or merged tile group) individually with online softmax, avoiding materialising
//   the full concatenated KV tensor.  When a Reshape wrapper is detected (GQA
//   head-fold: [1,kvH,...] -> [kvH,1,...]), Q is un-folded to batch=1 so that
//   FlashSDPA's native GQA head-broadcast handles qH > kvH without extra reshaping.

#include <limits>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTBLOCKCACHEATTENTIONTOFLASHSDPA
#define GEN_PASS_DEF_CONVERTBLOCKCACHEATTENTIONTOFLASHSDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Peel an optional Reshape/AffineReshape wrapper and return the underlying ConcatOp.
// FuseAttentionPass writes K/V operands as (AffineReshape|Reshape)(ConcatOp) when
// head-folding reshapes batch/head dimensions, or as a bare ConcatOp otherwise.
// Checking the reshape wrapper first covers both cases in a single branch.
//
// When a Reshape wrapper is present the Concat result (= Reshape input) must have
// batch==1 at dim[0].  This guarantees that the raw K/V tile inputs (Concat operands)
// all carry batch==1 and can therefore be fed directly to FlashSDPA without further
// reshaping.  Reshapes that touch the sequence or embedding dimensions rather than
// the batch/head dimensions are rejected by this check.
//
// When wrapperOut is non-null, *wrapperOut is set to the accepted wrapper op (if any).
static IE::ConcatOp matchConcatInput(mlir::Value val, mlir::Operation** wrapperOut = nullptr) {
    if (wrapperOut != nullptr) {
        *wrapperOut = nullptr;
    }
    mlir::Value inner = val;
    mlir::Operation* candidate = nullptr;
    if (auto reshape = inner.getDefiningOp<IE::ReshapeOp>()) {
        inner = reshape.getInput();
        candidate = reshape.getOperation();
    } else if (auto affineReshape = inner.getDefiningOp<IE::AffineReshapeOp>()) {
        inner = affineReshape.getInput();
        candidate = affineReshape.getOperation();
    }
    auto concatOp = inner.getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return nullptr;
    }
    // When a reshape wrapper is present, verify it only touches batch/head dims:
    // - seq (rank-2) and embed (rank-1) of the Concat result must equal those of the
    //   reshape output, directly confirming the last two dims are unchanged.
    // - Concat result batch==1 ensures all tile inputs (Concat operands) are batch==1.
    if (candidate != nullptr) {
        const auto concatShape = getShape(concatOp.getResult());
        const auto wrapOutShape = getShape(candidate->getResult(0));
        const auto rank = static_cast<int64_t>(concatShape.size());
        if (rank < 2 || static_cast<int64_t>(wrapOutShape.size()) != rank) {
            return nullptr;
        }
        if (concatShape[Dim(rank - 2)] != wrapOutShape[Dim(rank - 2)] ||
            concatShape[Dim(rank - 1)] != wrapOutShape[Dim(rank - 1)]) {
            return nullptr;
        }
        if (concatShape[Dim(0)] != 1) {
            return nullptr;
        }
        // Enforce the exact FuseAttention GQA head-fold pattern for 4D tensors:
        // Concat [1, kvH, S, E] reshaped to [kvH, 1, S, E].
        if (rank != 4 || wrapOutShape[Dim(1)] != 1) {
            return nullptr;
        }
        if (wrapperOut != nullptr) {
            *wrapperOut = candidate;
        }
    }
    return concatOp;
}

// Returns true when the op structurally matches the tiled K/V concat pattern and
// meets the profitability thresholds for a Flash SDPA cascade rewrite.
bool shouldConvertBlockCacheAttentionToFlashSDPA(IE::AttentionOp op, int64_t seqAlignment) {
    if (op.getInputBias() != nullptr || op.getPadSizeSAttr() != nullptr || op.getInputSink() != nullptr) {
        return false;
    }

    // Flash SDPA cascade is only beneficial for long source sequences.
    // Below these thresholds the decomposed MatMul path is faster.
    const auto shapeQ = getShape(op.getInputQ());
    const auto shapeK = getShape(op.getInputK());
    const auto rankQ = static_cast<int64_t>(shapeQ.size());
    const auto rankK = static_cast<int64_t>(shapeK.size());
    if (rankQ < 2 || rankK < 2) {
        return false;
    }
    const auto tSL = shapeQ[Dim(rankQ - 2)];
    const auto sSL = shapeK[Dim(rankK - 2)];
    constexpr int64_t kFlashPrefillMinSSL = 2048;
    constexpr int64_t kFlashDecodeMinSSL = 8192;
    const bool meetsThreshold = (tSL > 1 && sSL >= kFlashPrefillMinSSL) || (tSL == 1 && sSL >= kFlashDecodeMinSSL);
    if (!meetsThreshold) {
        return false;
    }

    auto kConcatOp = matchConcatInput(op.getInputK());
    if (!kConcatOp) {
        return false;
    }
    auto vConcatOp = matchConcatInput(op.getInputV());
    if (!vConcatOp) {
        return false;
    }
    const auto kInputs = kConcatOp.getInputs();
    const auto vInputs = vConcatOp.getInputs();
    // Require at least 3 KV-cache tiles (>= 2 past blocks + 1 present block).
    // Concats with fewer inputs are unlikely to represent a paged/block-attention
    // pattern
    if (kInputs.size() != vInputs.size() || kInputs.size() < 3) {
        return false;
    }
    // Validate that K and V are concatenated along the expected sequence dimensions:
    // K layout is [N, kvH, S, E]  -> seq dim is rank-2.
    // V layout is [N, kvH, E, S]  -> seq dim is rank-1.
    const int64_t kRank = static_cast<int64_t>(getShape(kConcatOp.getResult()).size());
    const int64_t vRank = static_cast<int64_t>(getShape(vConcatOp.getResult()).size());
    if (kRank < 2 || vRank < 2) {
        return false;
    }
    const auto kAxis = IE::getConcatAxis(kConcatOp);
    const auto vAxis = IE::getConcatAxis(vConcatOp);
    if (!kAxis || !vAxis || kAxis->ind() != kRank - 2 || vAxis->ind() != vRank - 1) {
        return false;
    }

    // Validate tile-level structural invariants required by buildMergedTileGroups and
    // the mask-slicing logic:
    //   1. Every K tile has rank == kRank and every V tile has rank == vRank.
    //   2. Each K/V tile pair has the same seq length (K rank-2 == V rank-1).
    //   3. The sum of K tile seq lengths equals sSL.
    // Violations would cause incorrect merged Concat shapes or out-of-bounds mask slices.
    int64_t tileSeqSum = 0;
    for (size_t idx = 0; idx < kInputs.size(); ++idx) {
        const auto kTileShape = getShape(kInputs[idx]);
        const auto vTileShape = getShape(vInputs[idx]);
        if (static_cast<int64_t>(kTileShape.size()) != kRank || static_cast<int64_t>(vTileShape.size()) != vRank) {
            return false;
        }
        const auto kTileSeq = kTileShape[Dim(kRank - 2)];
        const auto vTileSeq = vTileShape[Dim(vRank - 1)];
        if (kTileSeq != vTileSeq) {
            return false;
        }
        tileSeqSum += kTileSeq;
    }
    if (tileSeqSum != sSL) {
        return false;
    }

    // When an attention mask is present, its last dim must equal sSL so that
    // per-tile mask slices [seqOffset : seqOffset+tileS] are always in-bounds.
    if (const auto maskVal = op.getInputMask()) {
        const auto maskShape = getShape(maskVal);
        const auto maskRank = static_cast<int64_t>(maskShape.size());
        if (maskRank < 1 || maskShape[Dim(maskRank - 1)] != sSL) {
            return false;
        }
    }

    // Greedy tile-merge flushes a group only when its cumulative size is a multiple
    // of seqAlignment.  When sSL % seqAlignment == 0 every group, including the last,
    // is guaranteed aligned.  When sSL % seqAlignment != 0 no grouping can fix the tail.
    return sSL % seqAlignment == 0;
}

struct MergedTileGroups {
    SmallVector<SmallVector<mlir::Value>> kTiles;
    SmallVector<SmallVector<mlir::Value>> vTiles;
    SmallVector<int64_t> seqSizes;
    SmallVector<int64_t> seqOffsets;
};

// Merge adjacent K/V tile pairs greedily until cumulative seq_len % seqAlignment == 0.
// shouldConvertBlockCacheAttentionToFlashSDPA guarantees the total sSL is aligned, so every
// group produced here, including the last, is aligned.
// Example: tiles [1024, 1024, ..., 127, 1] -> [1024, 1024, ..., 128]
static MergedTileGroups buildMergedTileGroups(mlir::OperandRange kInputs, mlir::OperandRange vInputs,
                                              int64_t seqAlignment) {
    const auto numTiles = static_cast<int64_t>(kInputs.size());
    // K tile seq dim is rank-2: [N, kvH, tile_S, E]
    SmallVector<int64_t> tileSeqOffsets;
    SmallVector<int64_t> tileSeqSizes;
    {
        int64_t offset = 0;
        for (auto kTile : kInputs) {
            const auto kTileShape = getShape(kTile);
            const auto tileS = kTileShape[Dim(kTileShape.size() - 2)];
            tileSeqOffsets.push_back(offset);
            tileSeqSizes.push_back(tileS);
            offset += tileS;
        }
    }
    MergedTileGroups result;
    SmallVector<mlir::Value> pendingK, pendingV;
    int64_t pendingSize = 0;
    int64_t pendingOffset = 0;
    for (int64_t i = 0; i < numTiles; ++i) {
        if (pendingK.empty()) {
            pendingOffset = tileSeqOffsets[i];
        }
        pendingK.push_back(kInputs[i]);
        pendingV.push_back(vInputs[i]);
        pendingSize += tileSeqSizes[i];
        if (pendingSize % seqAlignment == 0 || i + 1 == numTiles) {
            result.kTiles.push_back(pendingK);
            result.vTiles.push_back(pendingV);
            result.seqSizes.push_back(pendingSize);
            result.seqOffsets.push_back(pendingOffset);
            pendingK.clear();
            pendingV.clear();
            pendingSize = 0;
        }
    }
    return result;
}

class AttentionToFlashSDPARewrite final : public mlir::OpRewritePattern<IE::AttentionOp> {
public:
    AttentionToFlashSDPARewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AttentionOp>(ctx), _log(log) {
        setDebugName("AttentionToFlashSDPARewrite");
    }

    mlir::LogicalResult matchAndRewrite(IE::AttentionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AttentionToFlashSDPARewrite::matchAndRewrite(IE::AttentionOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto seqAlignment =
            VPU::NCEInvariant::getAlignment(mlir::cast<NDTypeInterface>(origOp.getInputQ().getType()).getElementType());
    if (!shouldConvertBlockCacheAttentionToFlashSDPA(origOp, seqAlignment)) {
        return mlir::failure();
    }

    // Re-peel to obtain the tile inputs (shouldConvert already validated the pattern).
    // Capture the Reshape wrapper for K: when present (FuseAttention GQA-folded
    // [1,kvH,...] -> [kvH,1,...]), Q must be un-folded so that FlashSDPA can handle
    // GQA natively (qH > kvH).  When no wrapper is present (bare-Concat / non-GQA
    // path), K/V tiles retain their original batch dimension and Q is used as-is.
    mlir::Operation* kWrapOp = nullptr;
    auto kConcatOp = matchConcatInput(origOp.getInputK(), &kWrapOp);
    auto vConcatOp = matchConcatInput(origOp.getInputV());
    const auto kInputs = kConcatOp.getInputs();
    const auto vInputs = vConcatOp.getInputs();

    auto ctx = getContext();
    const auto loc = origOp->getLoc();

    // K seq dim: rank-2 (NCHW: dim H); V seq dim (before transpose): rank-1 (NCHW: dim W)
    const auto kSeqDim = Dim(static_cast<int64_t>(getShape(kInputs[0]).size()) - 2);
    const auto vSeqDim = Dim(static_cast<int64_t>(getShape(vInputs[0]).size()) - 1);

    // When K has a GQA head-fold wrapper (FuseAttention folded [1,kvH,...] -> [kvH,1,...]),
    // un-fold Q from [kvH, grp, tSL, E] back to [1, kvH*grp, tSL, E] so FlashSDPA
    // receives batch=1 and handles GQA natively (qH > kvH).
    // K/V tiles from the Concat already carry batch=1 ([1, kvH, tileS, ...]).
    mlir::Value queryBase = origOp.getInputQ();
    if (kWrapOp != nullptr) {
        const auto qShape = getShape(queryBase).raw();
        if (qShape.size() != 4) {
            return mlir::failure();
        }
        SmallVector<int64_t> unfoldedShape = {1, qShape[0] * qShape[1], qShape[2], qShape[3]};
        queryBase = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "unfold_q_gqa"), queryBase,
                                                   getIntArrayAttr(ctx, unfoldedShape))
                            .getOutput();
    }

    // Scale Q: use provided scale input only.
    // When the AttentionOp was created from a MatMul→Add→Softmax→MatMul pattern and
    // fuse_attention found no explicit scale multiply, getInputScale() is null, which
    // means the model does not apply scaling in this subgraph (the scale is either
    // baked into Q externally or genuinely absent). Do NOT apply a default 1/sqrt(d)
    // in that case; doing so would double-scale models whose Q is already prescaled.
    // For the SDPAOp path, fuse_attention always fills in the scale before writing the
    // AttentionOp, so null only occurs on the MatMul path.
    mlir::Value queryForFlash = queryBase;
    if (const auto scale = origOp.getInputScale()) {
        queryForFlash = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "query_scaled"), queryBase, scale,
                                                        IE::AutoBroadcastType::NUMPY,
                                                        /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                        /*outputPadding=*/nullptr, /*inputPadding=*/nullptr)
                                .getOutput();
    }

    // V tiles are [N, kvH, E, tile_S] (transposed in AttentionOp convention);
    // vEmbedding is the second-to-last dim (E).
    const auto vTile0Shape = getShape(vInputs[0]);
    const auto vTileRank = static_cast<int64_t>(vTile0Shape.size());
    const auto vEmbedding = vTile0Shape[Dim(vTileRank - 2)];

    // Initialize running state tensors from the (possibly un-folded) Q shape.
    const auto qShapeRaw = getShape(queryBase).raw();
    const auto qRank = static_cast<int64_t>(qShapeRaw.size());

    SmallVector<int64_t> runOutShape(qShapeRaw.begin(), qShapeRaw.end());
    runOutShape[qRank - 1] = vEmbedding;
    auto runOutType = mlir::RankedTensorType::get(runOutShape, mlir::Float16Type::get(ctx));
    auto initRunOut = vpux::Const::createDenseConst(rewriter, appendLoc(loc, "running_output"), runOutType, 0.0f);

    // running_max/sum: drop last dim from Q shape -> [N, H, tSL]
    SmallVector<int64_t> runStatShape(qShapeRaw.begin(), qShapeRaw.end() - 1);
    auto runMaxType = mlir::RankedTensorType::get(runStatShape, mlir::Float16Type::get(ctx));
    auto initRunMax = vpux::Const::createDenseConst(rewriter, appendLoc(loc, "running_max"), runMaxType,
                                                    -std::numeric_limits<float>::infinity());
    auto runSumType = mlir::RankedTensorType::get(runStatShape, mlir::Float32Type::get(ctx));
    auto initRunSum = vpux::Const::createDenseConst(rewriter, appendLoc(loc, "running_sum"), runSumType, 0.0f);

    // V transpose permutation: swap last two dims [..., E, S] -> [..., S, E]
    SmallVector<uint32_t> vTransposeOrder;
    for (int64_t d = 0; d < vTileRank - 2; ++d) {
        vTransposeOrder.push_back(static_cast<uint32_t>(d));
    }
    vTransposeOrder.push_back(static_cast<uint32_t>(vTileRank - 1));
    vTransposeOrder.push_back(static_cast<uint32_t>(vTileRank - 2));
    const auto vTransposeAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(vTransposeOrder, ctx));

    const auto merged = buildMergedTileGroups(kInputs, vInputs, seqAlignment);
    const auto numMergedTiles = static_cast<int64_t>(merged.kTiles.size());

    const auto maskInput = origOp.getInputMask();

    // Concatenate a group of tiles along seqDim into a single value.
    // Returns the sole element unchanged when the group has exactly one tile.
    const auto mergeTilesAlongSeq = [&](ArrayRef<mlir::Value> tiles, Dim seqDim, int64_t mergedSize, StringRef tag,
                                        int64_t tileIdx) -> mlir::Value {
        if (tiles.size() == 1) {
            return tiles[0];
        }
        const auto rank = static_cast<int64_t>(getShape(tiles[0]).size());
        const auto seqIdx = static_cast<int64_t>(seqDim.ind());
        SmallVector<SmallVector<int64_t>> staticOffsets;
        int64_t off = 0;
        for (auto tile : tiles) {
            SmallVector<int64_t> o(rank, 0);
            o[seqIdx] = off;
            off += getShape(tile)[seqDim];
            staticOffsets.push_back(o);
        }
        auto tile0NdType = mlir::cast<vpux::NDTypeInterface>(tiles[0].getType());
        auto mergedShape = Shape(tile0NdType.getShape().raw());
        mergedShape[seqDim] = mergedSize;
        return rewriter
                .create<IE::ConcatOp>(appendLoc(loc, "{0}_{1}", tag, tileIdx), tile0NdType.changeShape(mergedShape),
                                      mlir::ValueRange(tiles), getIntArrayOfArray(ctx, staticOffsets))
                .getOutput();
    };

    // Build cascaded FlashSDPAOp chain (online softmax across merged tiles)
    mlir::Value runOut = initRunOut;
    mlir::Value runMax = initRunMax;
    mlir::Value runSum = initRunSum;

    for (int64_t i = 0; i < numMergedTiles; ++i) {
        const auto& kGroup = merged.kTiles[i];
        const auto& vGroup = merged.vTiles[i];

        // Merge K tiles along seq dim ([N, kvH, tile_S, E]) if needed.
        // K/V tiles are used directly (batch=1 from the Concat); Q was un-folded to
        // [1, qH, tSL, E] above so FlashSDPA handles GQA natively (qH > kvH).
        mlir::Value kTile = mergeTilesAlongSeq(kGroup, kSeqDim, merged.seqSizes[i], "k_merge", i);

        // Merge V tiles along seq dim ([N, kvH, E, tile_S]) if needed, then transpose.
        mlir::Value vMerged = mergeTilesAlongSeq(vGroup, vSeqDim, merged.seqSizes[i], "v_merge", i);
        // Transpose merged V: [..., E, S] -> [..., S, E]
        mlir::Value vTile = rewriter.create<IE::TransposeOp>(appendLoc(loc, "v_tile_transpose_{0}", i), vMerged,
                                                             nullptr, vTransposeAttr)
                                    .getOutput();

        // Slice attention mask for this merged tile group
        mlir::Value maskTile = nullptr;
        if (maskInput != nullptr) {
            const auto maskShape = getShape(maskInput);
            const auto maskRank = static_cast<int64_t>(maskShape.size());
            SmallVector<int64_t> sliceOffsets(maskRank, 0);
            sliceOffsets[maskRank - 1] = merged.seqOffsets[i];
            SmallVector<int64_t> sliceSizes(maskShape.raw().begin(), maskShape.raw().end());
            sliceSizes[maskRank - 1] = merged.seqSizes[i];
            maskTile =
                    rewriter.create<IE::SliceOp>(appendLoc(loc, "mask_slice_{0}", i), maskInput,
                                                 getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, sliceSizes))
                            .getResult();
        }

        const auto isHead = mlir::BoolAttr::get(ctx, i == 0);
        const auto isTail = mlir::BoolAttr::get(ctx, i + 1 == numMergedTiles);
        auto tileOp =
                rewriter.create<IE::FlashSDPAOp>(appendLoc(loc, "flash_sdpa_tile_{0}", i), queryForFlash, kTile, vTile,
                                                 runOut, runMax, runSum, maskTile, isHead, isTail, getIntAttr(ctx, 0));

        runOut = tileOp.getResultRunningOutput();
        runMax = tileOp.getResultRunningMax();
        runSum = tileOp.getResultRunningSum();
    }

    // Fold the FlashSDPA output back to the original AttentionOp output shape when Q was
    // un-folded.  After un-folding Q: [kvH, grp, tSL, E] -> [1, qH, tSL, E], runOut has
    // shape [1, qH, tSL, vE]; fold it back to [kvH, grp, tSL, vE] to match origOp output.
    mlir::Value result = runOut;
    if (kWrapOp != nullptr) {
        const auto origOutShape = getShape(origOp.getOutput()).raw();
        result = rewriter.create<IE::ReshapeOp>(
                                 appendLoc(loc, "fold_output_gqa"), result,
                                 getIntArrayAttr(ctx, SmallVector<int64_t>(origOutShape.begin(), origOutShape.end())))
                         .getOutput();
    }

    // Adjust output element type if needed (running output is fp16; AttentionOp may be fp32)
    const auto origElemType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    if (mlir::cast<mlir::RankedTensorType>(result.getType()).getElementType() != origElemType) {
        result = rewriter.create<IE::ConvertOp>(appendLoc(loc, "convert"), result, origElemType);
    }

    rewriter.replaceOp(origOp, result);
    return mlir::success();
}

//
// ConvertBlockCacheAttentionToFlashSDPA
//

class ConvertBlockCacheAttentionToFlashSDPA final :
        public IE::impl::ConvertBlockCacheAttentionToFlashSDPABase<ConvertBlockCacheAttentionToFlashSDPA> {
public:
    explicit ConvertBlockCacheAttentionToFlashSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertBlockCacheAttentionToFlashSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AttentionToFlashSDPARewrite>(&ctx, _log);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertBlockCacheAttentionToFlashSDPAPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBlockCacheAttentionToFlashSDPAPass(Logger log) {
    return std::make_unique<ConvertBlockCacheAttentionToFlashSDPA>(log);
}
