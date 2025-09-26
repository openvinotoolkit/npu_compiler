//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEINCREMENTALSDPA
#define GEN_PASS_DEF_DECOMPOSEINCREMENTALSDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class IncrementalSDPARewrite final : public mlir::OpRewritePattern<IE::IncrementalSDPAOp> {
public:
    IncrementalSDPARewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::IncrementalSDPAOp>(ctx), _log(log) {
        setDebugName("IncrementalSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::IncrementalSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

Const::DeclareOp createDenseConstant(mlir::PatternRewriter& rewriter, ShapeRef shape, float value) {
    const auto runningMaxType = mlir::RankedTensorType::get(shape.raw(), rewriter.getF16Type());
    const auto values = SmallVector<type::float16>(shape.totalSize(), type::float16(value));
    auto content = Const::ContentAttr::get(Const::createConstContent(runningMaxType, ArrayRef<type::float16>(values)));

    return rewriter.create<Const::DeclareOp>(mlir::UnknownLoc::get(rewriter.getContext()), content.getType(), content);
}

mlir::LogicalResult IncrementalSDPARewrite::matchAndRewrite(IE::IncrementalSDPAOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ctx = rewriter.getContext();

    VPUX_THROW_UNLESS(origOp.getScale() == nullptr, "Non-default scale tensor for SDPA layer is not supported");

    // Multiply query on scale
    const auto queryShape = getShape(origOp.getQuery());
    const auto qkEmbeddingSize = queryShape.back();
    const auto scalingValue = 1.0f / std::sqrt(checked_cast<float>(qkEmbeddingSize));
    auto scalingConst = createDenseConstant(rewriter, ShapeRef{1}, scalingValue);
    const auto broadcastNumpy = vpux::IE::AutoBroadcastType::NUMPY;

    auto scaledQuery =
            rewriter.create<IE::MultiplyOp>(takeOpLoc(origOp, "scaled_query"), origOp.getQuery(), scalingConst,
                                            broadcastNumpy, nullptr, nullptr, nullptr, nullptr);

    // Multiply Query and Key tensors
    auto qkValue = rewriter.create<IE::MatMulOp>(takeOpLoc(origOp, "qk_value"), scaledQuery, origOp.getKey(),
                                                 /*transposeA*/ false, /*transposeB*/ true, nullptr);

    // Add an optional attention mask
    auto maskedAttention = [&]() -> mlir::Value {
        if (origOp.getAttentionMask() == nullptr) {
            return qkValue;
        }

        auto attention =
                rewriter.create<IE::AddOp>(takeOpLoc(origOp, "masked_attention"), qkValue, origOp.getAttentionMask(),
                                           broadcastNumpy, nullptr, nullptr, nullptr, nullptr);

        return attention.getOutput();
    }();

    // Compute maximum tensor
    const auto attentionRank = static_cast<int64_t>(getShape(maskedAttention).size());
    const auto reduceAxisAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{attentionRank - 1});
    const auto keepDims = false;
    auto attentionMaxValue = rewriter.create<IE::ReduceMaxOp>(takeOpLoc(origOp, "reduced_max"), maskedAttention,
                                                              nullptr, reduceAxisAttr, keepDims);

    // Compute new running maximum
    auto newRunningMax = rewriter.create<IE::MaximumOp>(takeOpLoc(origOp, "new_running_max"),
                                                        origOp.getInputRunningMax(), attentionMaxValue, broadcastNumpy);

    // Compute scaling factor to update running sum and partial output
    auto maxDifference =
            rewriter.create<IE::SubtractOp>(takeOpLoc(origOp, "running_max_difference"), origOp.getInputRunningMax(),
                                            newRunningMax, broadcastNumpy, nullptr, nullptr, nullptr, nullptr);
    auto scalingFactor = rewriter.create<IE::ExpOp>(takeOpLoc(origOp, "scaling_factor"), maxDifference);

    // Main part of the online softmax
    auto scaledSum = rewriter.create<IE::MultiplyOp>(takeOpLoc(origOp, "scaled_sum"), origOp.getInputRunningSum(),
                                                     scalingFactor, broadcastNumpy, nullptr, nullptr, nullptr, nullptr);
    auto newRunningMaxUnsqueezed = rewriter.create<IE::UnsqueezeOp>(
            appendLoc(maskedAttention.getLoc(), "new_running_max_unsqueezed"), newRunningMax, nullptr, reduceAxisAttr);

    auto attentionScoresSub = rewriter.create<IE::SubtractOp>(takeOpLoc(origOp, "subtracted_attention"),
                                                              maskedAttention, newRunningMaxUnsqueezed, broadcastNumpy,
                                                              nullptr, nullptr, nullptr, nullptr);

    auto rawAttention = rewriter.create<IE::ExpOp>(takeOpLoc(origOp, "raw_attention"), attentionScoresSub);
    auto reducedSum = rewriter.create<IE::ReduceSumOp>(takeOpLoc(origOp, "reduced_sum"), rawAttention, nullptr,
                                                       reduceAxisAttr, keepDims, nullptr, nullptr);

    auto newRunningSum = rewriter.create<IE::AddOp>(takeOpLoc(origOp, "new_running_sum"), scaledSum, reducedSum,
                                                    broadcastNumpy, nullptr, nullptr, nullptr, nullptr);

    auto scalingFactorUnsqueezed = rewriter.create<IE::UnsqueezeOp>(
            appendLoc(maskedAttention.getLoc(), "scaling_factor_unsqueezed"), scalingFactor, nullptr, reduceAxisAttr);

    // Last MatMul integrated with online softmax
    auto scaledPartialOutput = rewriter.create<IE::MultiplyOp>(takeOpLoc(origOp, "scaled_partial_output"),
                                                               origOp.getInputPartialOutput(), scalingFactorUnsqueezed,
                                                               broadcastNumpy, nullptr, nullptr, nullptr, nullptr);

    auto attentionValue = rewriter.create<IE::MatMulOp>(takeOpLoc(origOp, "attention_value"), rawAttention,
                                                        origOp.getValue(), false, false, nullptr);

    // Compute new partial output
    auto newPartialOutput =
            rewriter.create<IE::AddOp>(takeOpLoc(origOp, "new_partial_output"), scaledPartialOutput, attentionValue,
                                       broadcastNumpy, nullptr, nullptr, nullptr, nullptr);

    rewriter.replaceOp(origOp, mlir::ValueRange{newPartialOutput, newRunningMax, newRunningSum});

    return mlir::success();
}

//
// DecomposeIncrementalSDPA
//

class DecomposeIncrementalSDPA final : public IE::impl::DecomposeIncrementalSDPABase<DecomposeIncrementalSDPA> {
public:
    explicit DecomposeIncrementalSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void DecomposeIncrementalSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::IncrementalSDPAOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::ExpOp>();
    target.addLegalOp<IE::MatMulOp>();
    target.addLegalOp<IE::MaximumOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addLegalOp<IE::ReduceMaxOp>();
    target.addLegalOp<IE::ReduceSumOp>();
    target.addLegalOp<IE::SubtractOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::UnsqueezeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IncrementalSDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeIncrementalSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeIncrementalSDPAPass(Logger log) {
    return std::make_unique<DecomposeIncrementalSDPA>(log);
}
