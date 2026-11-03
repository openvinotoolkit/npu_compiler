//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/AffineMap.h>

#include <numeric>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGQATOATTENTIONDMA
#define GEN_PASS_DEF_CONVERTGQATOATTENTIONDMA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

mlir::Value reshapeTo(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input, ArrayRef<int64_t> shape) {
    return IE::ReshapeOp::create(builder, loc, input, getIntArrayAttr(builder.getContext(), shape)).getOutput();
}

mlir::Value convertToSI32(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) {
    const auto si32Type = getSInt32Type(builder.getContext());
    if (mlir::cast<NDTypeInterface>(input.getType()).getElementType() == si32Type) {
        return input;
    }
    return IE::ConvertOp::create(builder, loc, input, mlir::TypeAttr::get(si32Type)).getOutput();
}

// RoPE positions of the current chunk as a 1D SI32 tensor: position_ids when given, otherwise
// [pastSeqLen, pastSeqLen + seqLen) with pastSeqLen = seqlens_k + 1 - seqLen.
mlir::Value buildPositionIds(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value seqLensK,
                             mlir::Value positionIds, int64_t seqLen) {
    auto* ctx = builder.getContext();

    if (positionIds != nullptr) {
        return reshapeTo(builder, appendLoc(loc, "position_ids"),
                         convertToSI32(builder, appendLoc(loc, "position_ids_to_si32"), positionIds), {seqLen});
    }

    const auto seqLensK32 = convertToSI32(builder, appendLoc(loc, "seqlens_k_to_si32"), seqLensK);
    const auto pastSeqLen = reshapeTo(builder, appendLoc(loc, "seqlens_k_1d"), seqLensK32, {1});

    SmallVector<int32_t> offsets(checked_cast<size_t>(seqLen));
    std::iota(offsets.begin(), offsets.end(), checked_cast<int32_t>(1 - seqLen));
    const auto offsetsType = mlir::RankedTensorType::get({seqLen}, getSInt32Type(ctx));
    const auto offsetsConst =
            Const::createConst<int32_t>(builder, appendLoc(loc, "position_offsets"), offsetsType, ArrayRef(offsets));

    return IE::AddOp::create(builder, appendLoc(loc, "position_ids"), pastSeqLen, offsetsConst,
                             IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY), /*post_op=*/nullptr,
                             /*clamp=*/nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr)
            .getOutput();
}

// Gathers the cos/sin rows of positionIds from [maxSeqLen, rotaryDim / 2] into [1, 1, seqLen, rotaryDim / 2].
mlir::Value buildRotaryTable(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value cache, mlir::Value positionIds,
                             int64_t seqLen, StringRef name) {
    auto* ctx = builder.getContext();
    const auto gathered = IE::GatherOp::create(builder, appendLoc(loc, formatv("{0}_gather", name)), cache, positionIds,
                                               getIntAttr(ctx, 0), getIntAttr(ctx, 0), getIntAttr(ctx, 1))
                                  .getOutput();

    const auto halfRotaryDim = getShape(gathered).back();
    return reshapeTo(builder, appendLoc(loc, formatv("{0}_reshape", name)), gathered, {1, 1, seqLen, halfRotaryDim});
}

// KV cache scale as [1, kvNumHeads, 1, headSize] (per-channel) or [1, 1, 1, 1] (per-tensor).
mlir::Value buildKVCacheScale(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value scale, int64_t kvNumHeads,
                              bool perChannel, StringRef name) {
    const auto scaleSize = getShape(scale).totalSize();
    VPUX_THROW_WHEN(perChannel && scaleSize % kvNumHeads != 0,
                    "GroupQueryAttention per-channel scale of '{0}' elements is not divisible by kv_num_heads '{1}'",
                    scaleSize, kvNumHeads);

    const SmallVector<int64_t> targetShape = perChannel ? SmallVector<int64_t>{1, kvNumHeads, 1, scaleSize / kvNumHeads}
                                                        : SmallVector<int64_t>{1, 1, 1, 1};
    return reshapeTo(builder, appendLoc(loc, name), scale, targetShape);
}

// [batch, numHeads, seqLen, headSize] -> [batch, seqLen, numHeads * headSize]
mlir::Value buildOutput(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value attentionOutput) {
    auto* ctx = builder.getContext();
    const auto shape = getShape(attentionOutput);

    const SmallVector<unsigned> permutation = {0, 2, 1, 3};
    const auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, ctx));
    const auto transposed = IE::TransposeOp::create(builder, appendLoc(loc, "output_transpose"), attentionOutput,
                                                    /*order=*/nullptr, orderAttr)
                                    .getOutput();

    return reshapeTo(builder, appendLoc(loc, "output_reshape"), transposed,
                     {shape[Dim(0)], shape[Dim(2)], shape[Dim(1)] * shape[Dim(3)]});
}

void convertToAttentionDMA(IE::GroupQueryAttentionOp gqaOp) {
    mlir::OpBuilder builder(gqaOp);
    auto* ctx = builder.getContext();
    const auto loc = gqaOp.getLoc();

    // Unsupported configurations are decomposed by OV before import, see isGQAFlashEligible.
    VPUX_THROW_UNLESS(gqaOp.getCausal(), "GroupQueryAttention at '{0}' with causal=false is not supported", loc);

    auto query = gqaOp.getQuery();
    auto key = gqaOp.getKey();
    const auto queryType = mlir::cast<NDTypeInterface>(query.getType());
    const auto queryShape = queryType.getShape();
    const auto seqLen = queryShape[Dim(2)];
    const auto headSize = queryShape[Dim(3)];

    // AttentionDMA has no rotary embedding, so RoPE is applied to Q and K up front.
    if (gqaOp.getDoRotary()) {
        const auto cosCache = gqaOp.getCosCache();
        const auto sinCache = gqaOp.getSinCache();
        VPUX_THROW_WHEN(cosCache == nullptr || sinCache == nullptr,
                        "GroupQueryAttention at '{0}' has do_rotary set without cos/sin caches", loc);
        VPUX_THROW_UNLESS(2 * getShape(cosCache).back() == headSize,
                          "GroupQueryAttention at '{0}' with partial rotary embedding is not supported", loc);

        const auto positionIds = buildPositionIds(builder, loc, gqaOp.getSeqLensK(), gqaOp.getPositionIds(), seqLen);
        const auto cos = buildRotaryTable(builder, loc, cosCache, positionIds, seqLen, "cos");
        const auto sin = buildRotaryTable(builder, loc, sinCache, positionIds, seqLen, "sin");
        const auto modeAttr = IE::RoPEModeAttr::get(
                ctx, gqaOp.getRotaryInterleaved() ? IE::RoPEMode::PAIRWISE_INTERLEAVED : IE::RoPEMode::PAIRWISE);

        query = IE::RoPEOp::create(builder, appendLoc(loc, "rope_q"), query, cos, sin, modeAttr).getOutput();
        key = IE::RoPEOp::create(builder, appendLoc(loc, "rope_k"), key, cos, sin, modeAttr).getOutput();
    }

    // Default softmax scale is 1 / sqrt(head_size).
    const auto attrScale = gqaOp.getScale().convertToFloat();
    const auto scaleValue = attrScale != 0.0f ? attrScale : 1.0f / std::sqrt(checked_cast<float>(headSize));
    const auto scaleType = mlir::RankedTensorType::get({1}, queryType.getElementType());
    const auto scale = Const::createFloatConst(builder, appendLoc(loc, "scale"), scaleType, {scaleValue});

    // A quantized KV cache stays quantized; the kernel dequantizes and requantizes with these scales.
    mlir::Value kScale = nullptr;
    mlir::Value vScale = nullptr;
    const auto kvNumHeads = gqaOp.getKvNumHeads();
    if (gqaOp.getKvCacheBitWidth() != 0 && gqaOp.getKQuantType() != IE::GQAQuantType::NONE) {
        VPUX_THROW_UNLESS(gqaOp.getKvCacheBitWidth() == 8,
                          "GroupQueryAttention at '{0}' supports only an 8-bit quantized KV cache, got '{1}'", loc,
                          gqaOp.getKvCacheBitWidth());
        VPUX_THROW_WHEN(gqaOp.getKScale() == nullptr || gqaOp.getVScale() == nullptr,
                        "GroupQueryAttention at '{0}' has a quantized KV cache without k/v scales", loc);
        kScale = buildKVCacheScale(builder, loc, gqaOp.getKScale(), kvNumHeads,
                                   gqaOp.getKQuantType() == IE::GQAQuantType::PER_CHANNEL, "k_scale");
        vScale = buildKVCacheScale(builder, loc, gqaOp.getVScale(), kvNumHeads,
                                   gqaOp.getVQuantType() == IE::GQAQuantType::PER_CHANNEL, "v_scale");
    }

    // smooth_softmax without head_sink is the all-zero sink case, passed as an attribute.
    const auto headSink = gqaOp.getHeadSink();
    const auto sink = headSink != nullptr ? reshapeTo(builder, appendLoc(loc, "head_sink"), headSink,
                                                      {1, gqaOp.getNumHeads(), 1, 1})
                                          : nullptr;

    // The kernel derives the causal/sliding-window mask from seqlens_k, so no explicit mask is built.
    const auto seqLenK = convertToSI32(builder, appendLoc(loc, "seqlens_k_to_si32"), gqaOp.getSeqLensK());

    const auto localWindowSize = gqaOp.getLocalWindowSize();
    auto attention = IE::AttentionDMAOp::create(
            builder, loc, query, key, gqaOp.getValue(), gqaOp.getPastKey(), gqaOp.getPastValue(),
            /*inputMask=*/nullptr, scale, sink, gqaOp.getAttentionBias(), seqLenK, kScale, vScale,
            /*padSizeS=*/nullptr, localWindowSize >= 1 ? getIntAttr(ctx, localWindowSize) : nullptr,
            (gqaOp.getSmoothSoftmax() && sink == nullptr) ? mlir::UnitAttr::get(ctx) : nullptr,
            gqaOp.getSlidingWindowCacheAttr(), /*vTransposed=*/nullptr);

    gqaOp.getOutput().replaceAllUsesWith(buildOutput(builder, loc, attention.getOutput()));
    gqaOp.getPresentKey().replaceAllUsesWith(attention.getPresentK());
    gqaOp.getPresentValue().replaceAllUsesWith(attention.getPresentV());
    gqaOp.erase();
}

//
// ConvertGQAToAttentionDMAPass
//

class ConvertGQAToAttentionDMAPass final : public IE::impl::ConvertGQAToAttentionDMABase<ConvertGQAToAttentionDMAPass> {
public:
    explicit ConvertGQAToAttentionDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertGQAToAttentionDMAPass::safeRunOnFunc() {
    getOperation()->walk([&](IE::GroupQueryAttentionOp gqaOp) {
        _log.trace("Converting GroupQueryAttention at '{0}'", gqaOp.getLoc());
        convertToAttentionDMA(gqaOp);
    });
}

}  // namespace

//
// createConvertGQAToAttentionDMAPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGQAToAttentionDMAPass(Logger log) {
    return std::make_unique<ConvertGQAToAttentionDMAPass>(log);
}
