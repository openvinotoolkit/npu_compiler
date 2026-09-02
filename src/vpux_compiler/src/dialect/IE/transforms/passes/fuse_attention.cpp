//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/type/float16.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEATTENTION
#define GEN_PASS_DEF_FUSEATTENTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseAttentionPass
//

class FuseAttentionPass final : public IE::impl::FuseAttentionBase<FuseAttentionPass> {
public:
    explicit FuseAttentionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Legality check for converting an AttentionOp into AttentionDMAOp, expressed on raw input
// Values. Forward-declared here so the AttentionOp creators below can skip GQA/batch input folding
// for ops that will become AttentionDMAOp, which must keep raw Q/K/V for the SHAVE kernel.
bool isLegalAttentionDMAInputs(mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV, mlir::Value inputMask,
                               mlir::Value inputSink, mlir::Value inputBias);

mlir::Value findSeqLenKFromMask(mlir::Value mask);

//
// Skip through layout operations (Reshape, AffineReshape, Transpose) that have single use
//
mlir::Operation* skipLayoutAndReshapeOps(mlir::Value value) {
    auto op = value.getDefiningOp();
    while (mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::TransposeOp>(op)) {
        if (!op->hasOneUse()) {
            return op;
        }

        // Check if this operation swaps the last 2 dimensions
        auto inputType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
        auto outputType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
        const auto inputShape = inputType.getShape();
        const auto outputShape = outputType.getShape();
        const auto inputRank = inputShape.size();
        const auto outputRank = outputShape.size();

        if (inputRank >= 2 && outputRank == inputRank) {
            // For Transpose, check if it swaps last 2 dimensions
            if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(op)) {
                auto order = transposeOp.getOrderValue();
                if (order.has_value()) {
                    auto permutation = order.value();
                    if (permutation.getDimPosition(inputRank - 2) == inputRank - 1 &&
                        permutation.getDimPosition(inputRank - 1) == inputRank - 2) {
                        return op;
                    }
                }
            }

            // For Reshape/AffineReshape, check if last 2 dimensions are swapped
            if (mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp>(op)) {
                const auto inputLast2Product = inputShape[Dim(inputRank - 2)] * inputShape[Dim(inputRank - 1)];
                const auto outputLast2Product = outputShape[Dim(outputRank - 2)] * outputShape[Dim(outputRank - 1)];

                if (inputLast2Product != outputLast2Product) {
                    return op;
                }

                if (inputShape[Dim(inputRank - 2)] == outputShape[Dim(outputRank - 1)] &&
                    inputShape[Dim(inputRank - 1)] == outputShape[Dim(outputRank - 2)] &&
                    inputShape[Dim(inputRank - 2)] != inputShape[Dim(inputRank - 1)]) {
                    return op;
                }
            }
        }

        op = op->getOperand(0).getDefiningOp();
    }
    return op;
}

// Peel a `Squeeze/Reshape -> Select(Broadcast(cond), trueVal, falseVal)` chain on the mask and
// rebuild the Select on the un-broadcasted condition so the attention kernel receives the
// minimal broadcast shape (e.g. 1x1x1xsSL) and consumes mask via per-dim broadcast strides.
// Returns the original value when the chain does not match.
mlir::Value extractUnbroadcastedMask(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value mask) {
    if (!mask) {
        return mask;
    }

    mlir::Value currentMask = mask;
    auto maskInputOp = currentMask.getDefiningOp();
    if (mlir::isa_and_present<IE::SqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(maskInputOp)) {
        currentMask = maskInputOp->getOperand(0);
    }

    auto selectOp = currentMask.getDefiningOp<IE::SelectOp>();
    if (selectOp == nullptr) {
        return mask;
    }
    auto broadcastOp = selectOp.getInput1().getDefiningOp<IE::BroadcastOp>();
    if (broadcastOp == nullptr) {
        return mask;
    }

    if (selectOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        return mask;
    }

    auto unbroadcastedCond = broadcastOp.getInput();

    // Restrict to the validated key-padding mask pattern where only the innermost (sSL) dimension
    // varies, i.e. cond is 1x...x1xsSL with the same rank as the broadcast mask. The attention kernel
    // consumes such a mask via per-dim broadcast strides; requiring an equal rank keeps the peeled
    // mask at the rank the kernel indexes (no rank reduction), and other broadcast layouts are not
    // validated against the kernel.
    const auto condShape = getShape(unbroadcastedCond).raw();
    const auto maskShape = getShape(broadcastOp.getOutput()).raw();
    if (condShape.empty() || condShape.size() != maskShape.size() || condShape.back() != maskShape.back() ||
        !std::all_of(condShape.begin(), condShape.end() - 1, [](int64_t dim) {
            return dim == 1;
        })) {
        return mask;
    }

    // Broadcast output must match Select output exactly: guarantees the peeled rewrite
    // is a pure 1->N expansion driven by cond alone.
    if (getShape(broadcastOp.getOutput()) != getShape(selectOp.getOutput())) {
        return mask;
    }

    // Both branches must be 1-element float splat constants holding an attention-mask sentinel value
    // (0 to keep, -inf to mask). This keeps the new Select's NUMPY-inferred output shape equal to
    // cond.shape and restricts the rewrite to genuine masks.
    const auto isMaskSentinelSplat = [](mlir::Value v) {
        auto constOp = v.getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
            return false;
        }
        const auto constType = mlir::cast<NDTypeInterface>(constOp.getOutput().getType());
        if (!mlir::isa<mlir::FloatType>(constType.getElementType()) || constType.getNumElements() != 1) {
            return false;
        }
        const auto splatValue = Const::getSplatValue<double>(constOp);
        if (mlir::failed(splatValue)) {
            return false;
        }
        return splatValue.value() == 0.0 || (std::isinf(splatValue.value()) && splatValue.value() < 0.0);
    };
    if (!isMaskSentinelSplat(selectOp.getInput2()) || !isMaskSentinelSplat(selectOp.getInput3())) {
        return mask;
    }

    auto newSelect =
            builder.create<IE::SelectOp>(appendLoc(loc, "unbroadcast_mask"), unbroadcastedCond, selectOp.getInput2(),
                                         selectOp.getInput3(), selectOp.getAutoBroadcastAttr());
    return newSelect.getOutput();
}

// Optimize Attention inputs by applying two layout transformations:
//
// 1. GQA reshaping: when qHeads > kvHeads and kvHeads > 1, fold the batch and KV-head dimensions
//    together so that each group of Q heads shares exactly one K/V head:
//      Q[N, qC, tSL, E]  -> Q[N*kC, qC/kC, tSL, E]
//      K[N, kC, sSL, E]  -> K[N*kC, 1,     sSL, E]
//      V[N, kC, sSL, E]  -> V[N*kC, 1,     sSL, E]
//    The mask, bias and sink is reshaped to match the new batch/channel dimensions when applicable.
//    The output is reshaped back to the original [N, qC, tSL, vE] layout.
//
// 2. Batch-to-channels concentration: when batch > 1 and all N*C products (Q, K, V, mask, bias,
//    sink) are mutually broadcastable, fold the batch dimension into the channel dimension:
//      X[N, C, S, D] -> X[1, N*C, S, D]
//    This lets the hardware treat multiple batch entries as additional channel groups and improves
//    utilization. The output is reshaped back to the original [N, C, S, D] layout.
//
std::tuple<mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value, bool, SmallVector<int64_t>>
optimizeAttentionInputs(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value inputQ, mlir::Value inputK,
                        mlir::Value inputV, mlir::Value mask, mlir::Value bias, mlir::Value sink) {
    const auto ctx = builder.getContext();
    auto queryType = mlir::cast<NDTypeInterface>(inputQ.getType());
    const auto rank = queryType.getRank();
    const auto queryShape = queryType.getShape();

    if (rank != 4) {
        return {inputQ, inputK, inputV, mask, bias, sink, false, SmallVector<int64_t>{}};
    }

    // Peel mask Squeeze->Select(Broadcast(...)) so the kernel consumes 1x1x1xsSL via stride broadcast.
    mask = extractUnbroadcastedMask(builder, loc, mask);

    mlir::Value optimizedQ = inputQ;
    mlir::Value optimizedK = inputK;
    mlir::Value optimizedV = inputV;
    mlir::Value optimizedMask = mask;
    mlir::Value optimizedBias = bias;
    mlir::Value optimizedSink = sink;
    bool needsReshapeBack = false;
    SmallVector<int64_t> origOutputShape;

    const auto batch = queryShape.raw()[0];
    const auto qChannels = queryShape.raw()[1];

    // Get K and V shapes
    const auto keyType = mlir::cast<NDTypeInterface>(inputK.getType());
    const auto valueType = mlir::cast<NDTypeInterface>(inputV.getType());
    const auto keyShape = keyType.getShape();
    const auto valueShape = valueType.getShape();
    const auto kChannels = keyShape.raw()[1];
    const auto vChannels = valueShape.raw()[1];
    const bool sameBatch = (keyShape.raw()[0] == batch) && (valueShape.raw()[0] == batch);

    // Check for GQA configuration: qHeads > kvHeads and kvHeads > 1
    const bool isGQA = sameBatch && (qChannels > kChannels) && (kChannels > 1) && (kChannels == vChannels) &&
                       (qChannels % kChannels == 0);

    if (isGQA) {
        // GQA reshaping: Q=[N, qC, H, W] K=[N, kC, H, W] -> Q=[N*kC, qC/kC, H, W] K=[N*kC, 1, H, W]
        const auto qSeqLen = queryShape.raw()[2];
        const auto qHeadDim = queryShape.raw()[3];
        const auto kSeqLen = keyShape.raw()[2];
        const auto kHeadDim = keyShape.raw()[3];
        const auto vSeqLen = valueShape.raw()[2];
        const auto vHeadDim = valueShape.raw()[3];

        const auto newBatch = batch * kChannels;
        const auto newQChannels = qChannels / kChannels;

        // Reshape Q: [N, qC, H, W] -> [N*kC, qC/kC, H, W]
        SmallVector<int64_t> newQShape = {newBatch, newQChannels, qSeqLen, qHeadDim};
        optimizedQ =
                builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_q_gqa"), inputQ, getIntArrayAttr(ctx, newQShape))
                        .getOutput();

        // Reshape K: [N, kC, H, W] -> [N*kC, 1, H, W]
        SmallVector<int64_t> newKShape = {newBatch, 1, kSeqLen, kHeadDim};
        optimizedK =
                builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_k_gqa"), inputK, getIntArrayAttr(ctx, newKShape))
                        .getOutput();

        // Reshape V: [N, vC, H, W] -> [N*vC, 1, H, W]
        SmallVector<int64_t> newVShape = {newBatch, 1, vSeqLen, vHeadDim};
        optimizedV =
                builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_v_gqa"), inputV, getIntArrayAttr(ctx, newVShape))
                        .getOutput();

        // Reshape mask/bias/sink to match GQA-reshaped Q/K/V batch and channel dimensions.
        // Per-head operand [operandBatch, qC, S, T] -> [newBatch, qC/kC, S, T];
        // batch-only operand [batch, 1, S, T] -> [newBatch, 1, S, T];
        // broadcast operand [1, 1, S, T] kept as-is.
        auto reshapeGQAOperand = [&](mlir::Value operand, StringRef name) -> mlir::Value {
            if (!operand) {
                return operand;
            }
            const auto operandShape = mlir::cast<NDTypeInterface>(operand.getType()).getShape();
            if (operandShape.size() != 4) {
                return operand;
            }
            const auto operandBatch = operandShape.raw()[0];
            const auto operandChannels = operandShape.raw()[1];
            int64_t newOperandBatch = operandBatch;
            int64_t newOperandChannels = operandChannels;
            if (operandChannels == qChannels && operandChannels > 1) {
                newOperandChannels = newQChannels;
                if (operandBatch == batch) {
                    newOperandBatch = newBatch;
                }
            } else if (operandBatch == batch && operandBatch > 1) {
                newOperandBatch = newBatch;
            }
            if (newOperandBatch != operandBatch || newOperandChannels != operandChannels) {
                SmallVector<int64_t> newOperandShape = {newOperandBatch, newOperandChannels, operandShape.raw()[2],
                                                        operandShape.raw()[3]};
                return builder
                        .create<IE::ReshapeOp>(appendLoc(loc, name), operand, getIntArrayAttr(ctx, newOperandShape))
                        .getOutput();
            }
            return operand;
        };
        optimizedMask = reshapeGQAOperand(mask, "reshape_mask_gqa");
        optimizedBias = reshapeGQAOperand(bias, "reshape_bias_gqa");
        optimizedSink = reshapeGQAOperand(sink, "reshape_sink_gqa");

        needsReshapeBack = true;
        // Match existing convention: use dimension 2 from V for head dimension
        origOutputShape = {batch, qChannels, qSeqLen, vSeqLen};
    } else if (batch != 1) {
        auto getNCProduct = [](mlir::Value input) -> std::optional<int64_t> {
            if (!input) {
                return std::nullopt;
            }
            const auto inputType = mlir::cast<NDTypeInterface>(input.getType());
            const auto inputShape = inputType.getShape();
            const auto inputRank = inputShape.size();
            if (inputRank < 3 || inputRank > 4) {
                return std::nullopt;
            }
            const auto batch = (inputRank == 4) ? inputShape.raw()[0] : 1;
            return batch * inputShape.raw()[1];
        };

        const auto qNC = getNCProduct(inputQ);
        const auto kNC = getNCProduct(inputK);
        const auto vNC = getNCProduct(inputV);

        // Check if all N*C products are broadcastable
        bool compatibleShapes = true;
        if (qNC.has_value() && kNC.has_value() && !vpux::isBroadcastable(qNC.value(), kNC.value())) {
            compatibleShapes = false;
        }
        if (compatibleShapes && qNC.has_value() && vNC.has_value() &&
            !vpux::isBroadcastable(qNC.value(), vNC.value())) {
            compatibleShapes = false;
        }
        if (compatibleShapes && mask) {
            const auto maskNC = getNCProduct(mask);
            if (qNC.has_value() && maskNC.has_value() && !vpux::isBroadcastable(qNC.value(), maskNC.value())) {
                compatibleShapes = false;
            }
        }
        if (compatibleShapes && bias) {
            const auto biasNC = getNCProduct(bias);
            if (qNC.has_value() && biasNC.has_value() && !vpux::isBroadcastable(qNC.value(), biasNC.value())) {
                compatibleShapes = false;
            }
        }
        if (compatibleShapes && sink) {
            const auto sinkNC = getNCProduct(sink);
            if (qNC.has_value() && sinkNC.has_value() && !vpux::isBroadcastable(qNC.value(), sinkNC.value())) {
                compatibleShapes = false;
            }
        }

        if (compatibleShapes) {
            auto reshapeBatchToChannels = [&](mlir::Value input, StringRef name) -> mlir::Value {
                if (!input) {
                    return input;
                }

                const auto inputType = mlir::cast<NDTypeInterface>(input.getType());
                const auto inputShape = inputType.getShape();
                const auto inputRank = inputShape.size();

                if (inputRank != 4) {
                    return input;
                }

                const auto inputBatch = inputShape.raw()[0];
                if (inputBatch != batch) {
                    return input;
                }

                const auto N = inputShape.raw()[0];
                const auto C = inputShape.raw()[1];
                const auto seqLen = inputShape.raw()[2];
                const auto headDim = inputShape.raw()[3];

                SmallVector<int64_t> newShape = {1, N * C, seqLen, headDim};
                auto reshapeOp =
                        builder.create<IE::ReshapeOp>(appendLoc(loc, name), input, getIntArrayAttr(ctx, newShape));
                return reshapeOp.getOutput();
            };

            optimizedQ = reshapeBatchToChannels(inputQ, "reshape_q_batch_to_channels");
            optimizedK = reshapeBatchToChannels(inputK, "reshape_k_batch_to_channels");
            optimizedV = reshapeBatchToChannels(inputV, "reshape_v_batch_to_channels");
            optimizedMask = reshapeBatchToChannels(mask, "reshape_mask_batch_to_channels");
            optimizedBias = reshapeBatchToChannels(bias, "reshape_bias_batch_to_channels");
            optimizedSink = reshapeBatchToChannels(sink, "reshape_sink_batch_to_channels");

            if (optimizedQ != inputQ || optimizedK != inputK || optimizedV != inputV || optimizedMask != mask ||
                optimizedBias != bias || optimizedSink != sink) {
                needsReshapeBack = true;

                // Compute broadcasted output shape from Q, K, V
                const auto maxBatch = std::max({queryShape.raw()[0], keyShape.raw()[0], valueShape.raw()[0]});
                const auto maxChannels = std::max({queryShape.raw()[1], keyShape.raw()[1], valueShape.raw()[1]});
                const auto seqLen = queryShape.raw()[2];
                const auto headDim = valueShape.raw()[2];

                origOutputShape = {maxBatch, maxChannels, seqLen, headDim};
            }
        }
    }

    return {optimizedQ,    optimizedK,    optimizedV,       optimizedMask,
            optimizedBias, optimizedSink, needsReshapeBack, origOutputShape};
}

void createAttentionFromSDPA(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV,
                             mlir::Value mask, mlir::Value scale, mlir::Value sink) {
    auto builder = mlir::OpBuilder(op);
    const auto ctx = builder.getContext();
    const auto loc = op->getLoc();

    if (scale == nullptr) {
        auto inputQType = mlir::cast<NDTypeInterface>(inputQ.getType());
        auto eDim = inputQType.getShape().back();
        float scaleValue = 1.0f / std::sqrt(static_cast<float>(eDim));
        SmallVector<int64_t> scaleShape(inputQType.getRank(), 1);
        const auto scaleTensorType = mlir::RankedTensorType::get(scaleShape, builder.getF32Type());
        const auto scaleLoc = appendLoc(loc, "scale");
        scale = Const::createConst(builder, scaleLoc, scaleTensorType, llvm::ArrayRef<float>{scaleValue});
    }

    mlir::Value finalQ = inputQ, finalK = inputK, finalV = inputV, finalMask = mask, finalBias = nullptr,
                finalSink = sink;
    bool needsReshapeBack = false;
    SmallVector<int64_t> origOutputShape;
    if (!isLegalAttentionDMAInputs(inputQ, inputK, inputV, mask, sink, /*inputBias=*/nullptr)) {
        std::tie(finalQ, finalK, finalV, finalMask, finalBias, finalSink, needsReshapeBack, origOutputShape) =
                optimizeAttentionInputs(builder, loc, inputQ, inputK, inputV, mask, /*bias=*/nullptr, sink);
    }

    auto attentionOp = builder.create<IE::AttentionOp>(appendLoc(loc, "attention_full"), finalQ, finalK, finalV,
                                                       finalMask, scale, finalSink, finalBias, /*padSizeS=*/nullptr);

    if (needsReshapeBack) {
        auto reshapeBackOp =
                builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_output_channels_to_batch"),
                                              attentionOp.getOutput(), getIntArrayAttr(ctx, origOutputShape));
        op->replaceAllUsesWith(reshapeBackOp);
    } else {
        op->replaceAllUsesWith(attentionOp);
    }

    op->erase();
}

void createAttention(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV, mlir::Value mask,
                     mlir::Value scale, mlir::Value sink, mlir::Value bias) {
    auto builder = mlir::OpBuilder(op);
    const auto ctx = builder.getContext();
    const auto loc = op->getLoc();

    mlir::Value finalQ = inputQ, finalK = inputK, finalV = inputV, finalMask = mask, finalBias = bias, finalSink = sink;
    bool needsReshapeBack = false;
    SmallVector<int64_t> origOutputShape;
    if (!isLegalAttentionDMAInputs(inputQ, inputK, inputV, mask, sink, bias)) {
        std::tie(finalQ, finalK, finalV, finalMask, finalBias, finalSink, needsReshapeBack, origOutputShape) =
                optimizeAttentionInputs(builder, loc, inputQ, inputK, inputV, mask, bias, sink);
    }

    auto attentionOp = builder.create<IE::AttentionOp>(appendLoc(loc, "attention_pattern"), finalQ, finalK, finalV,
                                                       finalMask, scale, finalSink, finalBias, /*padSizeS=*/nullptr);

    if (needsReshapeBack) {
        auto reshapeBackOp =
                builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_output_channels_to_batch"),
                                              attentionOp.getOutput(), getIntArrayAttr(ctx, origOutputShape));
        op->replaceAllUsesWith(reshapeBackOp);
    } else {
        op->replaceAllUsesWith(attentionOp);
    }

    op->erase();
}

mlir::Value extractUnbroadcastedInput(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) {
    const auto ctx = builder.getContext();
    mlir::Value current = input;

    auto inputType = mlir::cast<NDTypeInterface>(input.getType());
    const auto inputRank = inputType.getRank();

    if (auto transposeOp = current.getDefiningOp<IE::TransposeOp>()) {
        if (IE::isWHSwappingTranspose(transposeOp)) {
            current = transposeOp.getInput();
        }
    }

    auto currentType = mlir::cast<NDTypeInterface>(current.getType());
    const auto currentShape = currentType.getShape().raw();

    if (auto reshapeOp = current.getDefiningOp<IE::ReshapeOp>()) {
        current = reshapeOp.getInput();
    } else if (auto affineReshapeOp = current.getDefiningOp<IE::AffineReshapeOp>()) {
        current = affineReshapeOp.getInput();
    }

    if (auto broadcastOp = current.getDefiningOp<IE::BroadcastOp>()) {
        current = broadcastOp.getInput();
        if (auto reshapeOp = current.getDefiningOp<IE::ReshapeOp>()) {
            current = reshapeOp.getInput();
        } else if (auto affineReshapeOp = current.getDefiningOp<IE::AffineReshapeOp>()) {
            current = affineReshapeOp.getInput();
        }

        auto unbroadcastedType = mlir::cast<NDTypeInterface>(current.getType());
        const auto unbroadcastedRank = unbroadcastedType.getRank();
        const auto unbroadcastedShape = unbroadcastedType.getShape().raw();

        // Verify last two dimensions (sequence length and embedding dimension) match
        if (unbroadcastedRank < 2 || unbroadcastedShape[unbroadcastedRank - 1] != currentShape[inputRank - 1] ||
            unbroadcastedShape[unbroadcastedRank - 2] != currentShape[inputRank - 2]) {
            return input;
        }

        // Build expected shape preserving the actual KV heads (supports both MQA and GQA)
        SmallVector<int64_t> expectedShape(currentShape.begin(), currentShape.end());
        const auto unbroadcastedHeads = unbroadcastedRank == 2 ? 1 : unbroadcastedShape[unbroadcastedRank - 3];
        expectedShape[inputRank - 3] = unbroadcastedHeads;

        // If the same rank, check if the shape matches expected shape
        bool needsReshape = (unbroadcastedRank != inputRank);
        if (!needsReshape && unbroadcastedRank == inputRank) {
            needsReshape = !std::equal(unbroadcastedShape.begin(), unbroadcastedShape.end(), expectedShape.begin());
        }

        if (needsReshape) {
            current = builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_unbroadcast"), current,
                                                    getIntArrayAttr(ctx, expectedShape))
                              .getOutput();
        }

        return current;
    }

    return input;
}

//
// squeezeSDPALeadingOnesTo4D
//

// Drop leading size==1 dims (everything but the trailing seq_len/head_dim) until the shape is 4D.
// Rightmost eligible dims are dropped first so the outermost batch dim stays in slot 0 and the
// resulting layout remains [batch, heads, seq_len, head_dim]. Returns nullopt when there are not
// enough squeezable dims.
std::optional<SmallVector<int64_t>> squeezeLeadingOnesTo4D(ArrayRef<int64_t> shape) {
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (rank <= 4) {
        return SmallVector<int64_t>(shape.begin(), shape.end());
    }

    int64_t toRemove = rank - 4;
    SmallVector<bool> drop(rank, false);
    for (int64_t i = rank - 3; i >= 0 && toRemove > 0; --i) {
        if (shape[i] == 1) {
            drop[i] = true;
            --toRemove;
        }
    }
    if (toRemove != 0) {
        return std::nullopt;
    }

    SmallVector<int64_t> result;
    for (int64_t i = 0; i < rank; ++i) {
        if (!drop[i]) {
            result.push_back(shape[i]);
        }
    }
    return result;
}

// Some exporters emit SDPA with rank > 4 caused only by leading size==1 dims (e.g. a singleton GQA
// group dim, giving 5D q/k/v like [1, heads, 1, seq, head_dim]). FuseAttention's matchers only
// accept rank in [3, 4], so such SDPAs would be left untouched and later lower onto the `sdpa`
// SHAVE kernel, which is not prebuilt on NPU50XX and fails at blob load.
// Fold the size==1 dims away here, so the matchers below can handle it;
// the output is reshaped back to the original rank.
void squeezeSDPALeadingOnesTo4D(IE::SDPAOp sdpaOp) {
    // All operands that will be reshaped (q/k/v and the optional mask) must be reducible to 4D;
    // bail out otherwise so we never erase the original op and leave a rank-mismatched SDPA.
    const auto isReducible = [](mlir::Value val) {
        return val == nullptr || getShape(val).size() <= 4 || squeezeLeadingOnesTo4D(getShape(val).raw()).has_value();
    };
    if (!isReducible(sdpaOp.getInputQ()) || !isReducible(sdpaOp.getInputK()) || !isReducible(sdpaOp.getInputV()) ||
        !isReducible(sdpaOp.getInputMask())) {
        return;
    }

    mlir::OpBuilder builder(sdpaOp);
    const auto ctx = builder.getContext();
    auto to4D = [&](mlir::Value val, StringRef tag) -> mlir::Value {
        if (val == nullptr || getShape(val).size() <= 4) {
            return val;
        }
        const auto newShape = squeezeLeadingOnesTo4D(getShape(val).raw()).value();
        return builder
                .create<IE::ReshapeOp>(appendLoc(sdpaOp.getLoc(), "squeeze_{0}", tag), val,
                                       getIntArrayAttr(ctx, newShape))
                .getOutput();
    };

    auto newSDPA = builder.create<IE::SDPAOp>(appendLoc(sdpaOp.getLoc(), "to_4d"), to4D(sdpaOp.getInputQ(), "q"),
                                              to4D(sdpaOp.getInputK(), "k"), to4D(sdpaOp.getInputV(), "v"),
                                              to4D(sdpaOp.getInputMask(), "m"), sdpaOp.getInputScale(),
                                              sdpaOp.getInputSink(), sdpaOp.getCausalAttr());

    auto unsqueeze = builder.create<IE::ReshapeOp>(appendLoc(sdpaOp.getLoc(), "unsqueeze_out"), newSDPA.getOutput(),
                                                   getIntArrayAttr(ctx, getShape(sdpaOp.getOutput()).raw()));
    sdpaOp.getOutput().replaceAllUsesWith(unsqueeze.getOutput());
    sdpaOp.erase();
}

// Legal AttentionDMA configurations: {qHeadSize, tSL, sSL}
struct AttentionDMAConfigs {
    int64_t qHeadSize;
    int64_t tSL;
    int64_t sSL;
    int64_t eDim;
    int64_t eVDim;
};

static const SmallVector<AttentionDMAConfigs> LEGAL_ATTENTION_DMA_CONFIGS = {
        {12, 3600, 3600, 16, 16},
};

bool isLegalAttentionDMAInputs(mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV, mlir::Value inputMask,
                               mlir::Value inputSink, mlir::Value inputBias) {
    auto tensorTypeQ = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto tensorTypeK = mlir::cast<NDTypeInterface>(inputK.getType());
    auto tensorTypeV = mlir::cast<NDTypeInterface>(inputV.getType());

    const auto rankQ = tensorTypeQ.getRank();
    const auto rankK = tensorTypeK.getRank();
    const auto rankV = tensorTypeV.getRank();

    if (rankQ < 3 || rankQ > 4) {
        return false;
    }
    if (rankK < 3 || rankK > 4) {
        return false;
    }
    if (rankV < 3 || rankV > 4) {
        return false;
    }
    // Boolean mask (i8 signless) is not yet supported in legal AttentionDMA configs
    if (inputMask) {
        auto tensorTypeMask = mlir::cast<NDTypeInterface>(inputMask.getType());
        auto maskElemType = tensorTypeMask.getElementType();
        if (maskElemType.isSignlessInteger(8)) {
            return false;
        }
    }

    auto shapeQ = tensorTypeQ.getShape().raw();
    auto shapeK = tensorTypeK.getShape().raw();
    auto shapeV = tensorTypeV.getShape().raw();

    // If any Q/K/V dimension is dynamic, always allow conversion to AttentionDynamic
    const auto hasDynamicDim = [](ArrayRef<int64_t> shape) {
        return llvm::any_of(shape, [](int64_t d) {
            return d == mlir::ShapedType::kDynamic;
        });
    };
    if (hasDynamicDim(shapeQ) || hasDynamicDim(shapeK) || hasDynamicDim(shapeV)) {
        vpux::Logger::global().trace("AttentionOp has dynamic shapes, legal for AttentionDma.");
        return true;
    }

    const auto qHeadSize = (rankQ == 3) ? shapeQ[0] : shapeQ[0] * shapeQ[1];
    const auto tSL = shapeQ[rankQ - 2];
    const auto sSL = shapeK[rankK - 2];
    const auto eDim = shapeK[rankK - 1];
    // V may be transposed or not; mirror AttentionOp type inference (inferReturnTypeComponents) to
    // pick the embedding dim, otherwise non-transposed V would be misclassified.
    const auto isTransposedV = shapeK[rankK - 2] != shapeV[rankV - 2];
    const auto eVdim = isTransposedV ? shapeV[rankV - 2] : shapeV[rankV - 1];

    // If the input sink is present, DynamicAttention is not supported
    if (inputSink != nullptr) {
        return false;
    }

    // Bias-only (no mask) is not supported by AttentionDMA DMA kernel yet.
    if (inputBias != nullptr && inputMask == nullptr) {
        return false;
    }

    auto matches = [&](const AttentionDMAConfigs& config) {
        return (config.qHeadSize == qHeadSize) && (config.sSL == sSL) && (config.tSL == tSL) && (config.eDim == eDim) &&
               (config.eVDim == eVdim);
    };

    if (llvm::any_of(LEGAL_ATTENTION_DMA_CONFIGS, matches)) {
        vpux::Logger::global().trace("AttentionOp is legal for AttentionDMA conversion");
        return true;
    }

    // The AttentionDMAFlash kernel handles variable-length KV sequences via an explicit seqLenK argument.
    if (findSeqLenKFromMask(inputMask) != nullptr && sSL >= 8192) {
        return true;
    }

    return false;
}

bool isLegalAttentionDMA(IE::AttentionOp op) {
    auto attDynOpName = mlir::OperationName("IE.AttentionDMA", op->getContext());
    if (!attDynOpName.isRegistered() || !attDynOpName.hasInterface<IE::LayerWithDmaInterface>()) {
        return false;
    }
    // Query the interface to confirm the current workload management mode is FWLM: AttentionDMA relies on
    // the FWLM SHV-submit-DMA flow, so it must not be lowered when FWLM is not active.
    const auto* iface = attDynOpName.getInterface<IE::LayerWithDmaInterface>();
    if (!iface || !iface->isSupported(iface, op.getOperation())) {
        return false;
    }
    return isLegalAttentionDMAInputs(op.getInputQ(), op.getInputK(), op.getInputV(), op.getInputMask(),
                                     op.getInputSink(), op.getInputBias());
}

mlir::Value getNonConstOperand(mlir::Value lhs, mlir::Value rhs) {
    const auto isConst = [](mlir::Value value) {
        return mlir::isa_and_present<Const::DeclareOp>(value.getDefiningOp());
    };
    const bool lhsConst = isConst(lhs);
    const bool rhsConst = isConst(rhs);
    if (lhsConst == rhsConst) {
        return nullptr;
    }
    return lhsConst ? rhs : lhs;
}

// Match the mask tree encoding seq_len_k; return the seq_len_k block argument (nullptr on mismatch).
//   seq_len_k -> Convert -> Add(c) -> AffineReshape
//     -> Add(c)      -> Greater(c)                       (left-aligned:  valid tokens at buffer start, -inf on right)
//     -> Subtract(c) -> Broadcast(c) -> GreaterEqual(c)  (right-aligned: valid tokens at buffer end,   -inf on left)
//     -> Select(cond, c, c) == mask
mlir::Value findSeqLenKFromMask(mlir::Value mask) {
    if (!mask) {
        return nullptr;
    }

    auto selectOp = mask.getDefiningOp<IE::SelectOp>();
    if (selectOp == nullptr) {
        return nullptr;
    }
    mlir::Value current = selectOp.getInput1();

    StringRef alignment;
    if (auto greaterOp = current.getDefiningOp<IE::GreaterOp>()) {
        alignment = "left-aligned";
        current = getNonConstOperand(greaterOp.getInput1(), greaterOp.getInput2());
    } else if (auto greaterEqualOp = current.getDefiningOp<IE::GreaterEqualOp>()) {
        alignment = "right-aligned";
        current = getNonConstOperand(greaterEqualOp.getInput1(), greaterEqualOp.getInput2());
    } else {
        return nullptr;
    }
    if (current == nullptr) {
        return nullptr;
    }

    if (alignment == "right-aligned") {
        auto broadcastOp = current.getDefiningOp<IE::BroadcastOp>();
        if (broadcastOp == nullptr) {
            return nullptr;
        }
        current = broadcastOp.getInput();

        auto subtractOp = current.getDefiningOp<IE::SubtractOp>();
        if (subtractOp == nullptr) {
            return nullptr;
        }
        current = getNonConstOperand(subtractOp.getInput1(), subtractOp.getInput2());
        if (current == nullptr) {
            return nullptr;
        }
    } else {
        // Follow all Add ops until AffineReshape — the count is 1 or 2 depending on
        // whether OV constant-folded the row-offset and scalar-adjustment adds into one.
        while (auto addOp = current.getDefiningOp<IE::AddOp>()) {
            current = getNonConstOperand(addOp.getInput1(), addOp.getInput2());
            if (current == nullptr) {
                return nullptr;
            }
        }
    }

    auto affineReshapeOp = current.getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshapeOp == nullptr) {
        return nullptr;
    }
    current = affineReshapeOp.getInput();

    auto addOp = current.getDefiningOp<IE::AddOp>();
    if (addOp == nullptr) {
        return nullptr;
    }
    current = getNonConstOperand(addOp.getInput1(), addOp.getInput2());
    if (current == nullptr) {
        return nullptr;
    }

    auto convertOp = current.getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr) {
        return nullptr;
    }
    current = convertOp.getInput();

    if (current.getDefiningOp() != nullptr) {
        return nullptr;
    }
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(current.getType());
    if (tensorType == nullptr || !tensorType.getElementType().isInteger(32) || tensorType.getNumElements() != 1) {
        return nullptr;
    }

    // Right-aligned masks are not yet supported by the AttentionDMA kernel.
    if (alignment == "right-aligned") {
        return nullptr;
    }

    return current;
}

//
// safeRunOnFunc
//
void FuseAttentionPass::safeRunOnFunc() {
    auto func = getOperation();

    // Collapse SDPAs with squeezable leading size==1 dims to 4D first so the rank-[3, 4] matchers
    // below can pick them up. Collected up front since squeezeSDPALeadingOnesTo4D erases ops.
    SmallVector<IE::SDPAOp> highRankSDPAs;
    func->walk([&](IE::SDPAOp sdpaOp) {
        if (getShape(sdpaOp.getInputQ()).size() > 4 || getShape(sdpaOp.getInputK()).size() > 4 ||
            getShape(sdpaOp.getInputV()).size() > 4) {
            highRankSDPAs.push_back(sdpaOp);
        }
    });
    for (auto sdpaOp : highRankSDPAs) {
        squeezeSDPALeadingOnesTo4D(sdpaOp);
    }

    func->walk([&](IE::SDPAOp sdpaOp) {
        auto inputKType = mlir::cast<NDTypeInterface>(sdpaOp.getInputK().getType());
        auto inputVType = mlir::cast<NDTypeInterface>(sdpaOp.getInputV().getType());
        const auto rank = inputKType.getRank();

        if (rank < 3 || rank > 4) {
            return;
        }

        const auto inputKShape = inputKType.getShape().raw();
        const auto inputVShape = inputVType.getShape().raw();
        if (inputKShape[rank - 2] != inputVShape[rank - 2]) {
            return;
        }

        auto builder = mlir::OpBuilder(sdpaOp);
        mlir::Value inputK = extractUnbroadcastedInput(builder, sdpaOp->getLoc(), sdpaOp.getInputK());
        mlir::Value inputV = extractUnbroadcastedInput(builder, sdpaOp->getLoc(), sdpaOp.getInputV());

        // Verify K and V have the same number of heads after extraction
        auto extractedKType = mlir::cast<NDTypeInterface>(inputK.getType());
        auto extractedVType = mlir::cast<NDTypeInterface>(inputV.getType());
        const auto extractedKShape = extractedKType.getShape().raw();
        const auto extractedVShape = extractedVType.getShape().raw();

        if (extractedKShape[rank - 3] != extractedVShape[rank - 3]) {
            return;
        }

        SmallVector<uint32_t> permuteNdOrder = {};
        for (int i = 0; i < rank - 2; i++) {
            permuteNdOrder.push_back(i);
        }
        permuteNdOrder.push_back(rank - 1);
        permuteNdOrder.push_back(rank - 2);
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permuteNdOrder, builder.getContext()));
        auto transposedVOp =
                builder.create<IE::TransposeOp>(appendLoc(sdpaOp->getLoc(), "transpose_v"), inputV, nullptr, orderAttr);

        createAttentionFromSDPA(sdpaOp, sdpaOp.getInputQ(), inputK, transposedVOp.getOutput(), sdpaOp.getInputMask(),
                                sdpaOp.getInputScale(), sdpaOp.getInputSink());
    });

    // Custom Attention pattern detection and fusion into Attention
    func->walk([&](IE::MatMulOp matMulVOp) {
        mlir::Operation* currentOp = skipLayoutAndReshapeOps(matMulVOp.getInput1());

        mlir::Value sink = nullptr;
        if (auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(currentOp); sliceOp != nullptr) {
            currentOp = skipLayoutAndReshapeOps(sliceOp.getInput());
        }

        auto softmaxOp = mlir::dyn_cast_or_null<IE::SoftMaxOp>(currentOp);
        if (softmaxOp == nullptr) {
            return;
        }

        currentOp = skipLayoutAndReshapeOps(softmaxOp.getInput());

        if (auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(currentOp); concatOp != nullptr) {
            auto concatInputs = concatOp.getInputs();
            if (concatInputs.size() != 2) {
                return;
            }

            // detect sink from concat
            const auto firstShape = mlir::cast<NDTypeInterface>(concatInputs[0].getType()).getShape().raw();
            const auto secondShape = mlir::cast<NDTypeInterface>(concatInputs[1].getType()).getShape().raw();
            const auto concatShape = mlir::cast<NDTypeInterface>(concatOp.getOutput().getType()).getShape().raw();
            if (firstShape.size() != secondShape.size() || firstShape.size() != concatShape.size()) {
                return;
            }

            const auto lastDim = firstShape.size() - 1;
            if (concatShape[lastDim] != firstShape[lastDim] + secondShape[lastDim]) {
                return;
            }
            if (secondShape[lastDim] != 1) {
                return;
            }

            // not need broadcasted input, internal sdpa handles it, and it may cause extra ops in the graph
            sink = concatInputs[1];
            if (auto broadcastOp = sink.getDefiningOp<IE::BroadcastOp>()) {
                sink = broadcastOp.getInput();
            } else if (auto tileOp = sink.getDefiningOp<IE::TileOp>()) {
                const auto inShape = mlir::cast<NDTypeInterface>(tileOp.getInput().getType()).getShape().raw();
                const auto outShape = mlir::cast<NDTypeInterface>(tileOp.getOutput().getType()).getShape().raw();
                // Replace tile with its input if tile is purely broadcasting (every dim is 1 or equals output dim).
                if (inShape.size() == outShape.size()) {
                    bool isBroadcastOnlyTile = true;
                    for (size_t i = 0; i < inShape.size(); ++i) {
                        if (!vpux::isBroadcastable(inShape[i], outShape[i])) {
                            isBroadcastOnlyTile = false;
                            break;
                        }
                    }
                    if (isBroadcastOnlyTile) {
                        sink = tileOp.getInput();
                    }
                }
            }
            currentOp = skipLayoutAndReshapeOps(concatInputs[0]);
        }

        mlir::Value attentionMask = nullptr;
        mlir::Value bias = nullptr;

        if (auto firstAddOp = mlir::dyn_cast_or_null<IE::AddOp>(currentOp); firstAddOp != nullptr) {
            currentOp = skipLayoutAndReshapeOps(firstAddOp.getInput1());

            if (auto secondAddOp = mlir::dyn_cast_or_null<IE::AddOp>(currentOp)) {
                bias = firstAddOp.getInput2();
                attentionMask = secondAddOp.getInput2();
                currentOp = skipLayoutAndReshapeOps(secondAddOp.getInput1());
            } else {
                attentionMask = firstAddOp.getInput2();
            }
        }

        mlir::Value scale = nullptr;
        if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(currentOp)) {
            scale = multiplyOp.getInput2();
            currentOp = skipLayoutAndReshapeOps(multiplyOp.getInput1());
        }

        auto qkMatMulOp = mlir::dyn_cast_or_null<IE::MatMulOp>(currentOp);
        if (qkMatMulOp == nullptr) {
            return;
        }

        // Skip Reshape ops that concentrate the batch dimension into channels:
        // [N, C, S, D] -> [1, N*C, S, D]. The optimizeAttentionInputs step applies
        // this transformation itself if possible;
        const auto skipBatchToChannelReshape = [](mlir::Value value) -> mlir::Value {
            auto reshapeOp = value.getDefiningOp<IE::ReshapeOp>();
            if (reshapeOp == nullptr) {
                return value;
            }
            const auto inType = mlir::cast<NDTypeInterface>(reshapeOp.getInput().getType());
            const auto outType = mlir::cast<NDTypeInterface>(reshapeOp.getOutput().getType());
            if (inType.getRank() != 4 || outType.getRank() != 4) {
                return value;
            }
            const auto inShape = inType.getShape().raw();
            const auto outShape = outType.getShape().raw();
            if (outShape[0] == 1 && inShape[0] * inShape[1] == outShape[1] && inShape[2] == outShape[2] &&
                inShape[3] == outShape[3]) {
                return reshapeOp.getInput();
            }
            return value;
        };

        mlir::Value inputQ = skipBatchToChannelReshape(qkMatMulOp.getInput1());
        mlir::Value inputK = skipBatchToChannelReshape(qkMatMulOp.getInput2());

        // Check all input ranks before any graph modifications.
        auto hasInvalidRank = [](mlir::Value val) -> bool {
            if (val == nullptr) {
                return false;
            }
            const auto r = mlir::cast<NDTypeInterface>(val.getType()).getRank();
            return r < 2 || r > 4;
        };
        if (hasInvalidRank(inputQ) || hasInvalidRank(inputK) || hasInvalidRank(matMulVOp.getInput2()) ||
            hasInvalidRank(attentionMask) || hasInvalidRank(bias)) {
            return;
        }

        auto builder = mlir::OpBuilder(matMulVOp);
        const auto ctx = builder.getContext();

        inputK = extractUnbroadcastedInput(builder, matMulVOp->getLoc(), inputK);

        auto extractScaleAndReapplyTransforms = [&](mlir::Value input) -> std::pair<mlir::Value, mlir::Value> {
            SmallVector<mlir::Operation*> layoutOps;
            mlir::Value current = input;
            while (auto defOp = current.getDefiningOp()) {
                if (mlir::isa<IE::TransposeOp, IE::ReshapeOp, IE::AffineReshapeOp>(defOp)) {
                    if (!defOp->hasOneUse()) {
                        break;
                    }
                    layoutOps.push_back(defOp);
                    current = defOp->getOperand(0);
                } else {
                    break;
                }
            }

            mlir::Value scale = nullptr;
            mlir::Value baseInput = input;

            if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(current.getDefiningOp())) {
                if (multiplyOp.getOutput().hasOneUse()) {
                    // Single use: extract scale, reroute and delete multiply
                    scale = multiplyOp.getInput2();
                    if (hasInvalidRank(scale)) {
                        return {input, nullptr};
                    }

                    // AttentionOp only supports a scalar scale
                    if (mlir::cast<NDTypeInterface>(scale.getType()).getShape().totalSize() != 1) {
                        return {input, nullptr};
                    }

                    if (!layoutOps.empty()) {
                        // Reconnect the first layout op to bypass the multiply
                        layoutOps.back()->setOperand(0, multiplyOp.getInput1());
                    } else {
                        // No layout ops, update baseInput directly
                        baseInput = multiplyOp.getInput1();
                    }
                }
            }

            return {baseInput, scale};
        };

        // If scale is not found in the direct path, check if it's applied to Q or K with transformations
        if (!scale) {
            auto [qBase, qScale] = extractScaleAndReapplyTransforms(inputQ);
            if (qScale) {
                scale = qScale;
                inputQ = qBase;
            } else {
                auto [kBase, kScale] = extractScaleAndReapplyTransforms(inputK);
                if (kScale) {
                    scale = kScale;
                    inputK = kBase;
                }
            }
        }

        // AttentionOp only supports a scalar scale
        if (scale != nullptr && mlir::cast<NDTypeInterface>(scale.getType()).getShape().totalSize() != 1) {
            return;
        }

        mlir::Value inputV = skipBatchToChannelReshape(matMulVOp.getInput2());

        inputV = extractUnbroadcastedInput(builder, matMulVOp->getLoc(), inputV);

        // Verify K and V have the same number of heads after extraction
        auto extractedKType = mlir::cast<NDTypeInterface>(inputK.getType());
        auto extractedVType = mlir::cast<NDTypeInterface>(inputV.getType());
        const auto kRank = extractedKType.getRank();
        const auto vRank = extractedVType.getRank();

        if (kRank >= 3 && vRank >= 3 && kRank == vRank) {
            const auto extractedKShape = extractedKType.getShape().raw();
            const auto extractedVShape = extractedVType.getShape().raw();

            if (extractedKShape[kRank - 3] != extractedVShape[vRank - 3]) {
                return;
            }
        }

        auto addTransposeIfNeeded = [&](mlir::Value& input, bool needsTranspose, StringRef name) {
            if (!needsTranspose) {
                return;
            }

            auto inputType = mlir::cast<NDTypeInterface>(input.getType());
            const auto rank = inputType.getRank();
            SmallVector<uint32_t> permuteOrder;
            for (int i = 0; i < rank - 2; i++) {
                permuteOrder.push_back(i);
            }
            permuteOrder.push_back(rank - 1);
            permuteOrder.push_back(rank - 2);

            auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permuteOrder, ctx));
            auto transposeOp = builder.create<IE::TransposeOp>(
                    appendLoc(matMulVOp.getLoc(), formatv("transpose_{0}_for_sdpa", name)), input, nullptr, orderAttr);

            input = transposeOp.getOutput();
        };

        auto inputQType = mlir::cast<NDTypeInterface>(inputQ.getType());
        auto inputKType = mlir::cast<NDTypeInterface>(inputK.getType());
        auto inputVType = mlir::cast<NDTypeInterface>(inputV.getType());

        if (inputKType.getRank() >= 2 && inputVType.getRank() >= 2 && inputKType.getRank() == inputVType.getRank() &&
            inputQType.getRank() == inputKType.getRank()) {
            const auto rank = inputKType.getRank();
            const auto inputQShape = inputQType.getShape().raw();
            auto inputKShape = inputKType.getShape().raw();
            const auto inputVShape = inputVType.getShape().raw();

            // Check if K needs transpose: Q[..., tSL, e], K[..., sSL, e]
            bool kNeedsTranspose = inputKShape[rank - 1] != inputQShape[rank - 1];
            addTransposeIfNeeded(inputK, kNeedsTranspose, "k");

            // Update K shape after potential transpose
            if (kNeedsTranspose) {
                inputKShape = mlir::cast<NDTypeInterface>(inputK.getType()).getShape().raw();
            }

            // Check if V needs transpose: K[..., sSL, e], V[..., eV, sSL]
            bool vNeedsTranspose = inputKShape[rank - 2] != inputVShape[rank - 1];
            addTransposeIfNeeded(inputV, vNeedsTranspose, "v");
        }

        createAttention(matMulVOp, inputQ, inputK, inputV, attentionMask, scale, sink, bias);
    });

    // MQA head-folding: for AttentionOp with qHeads > 1 and kvHeads == 1, reshape Q from
    // [N, qHeads, tSL, E] to [N, 1, qHeads*tSL, E] and tile the mask along the tSL dimension.
    // Since all Q heads share the same K and V in MQA, this is mathematically equivalent and
    // merges qHeads separate DPU matmuls into a single larger one, improving hardware utilization.
    func->walk([&](IE::AttentionOp attentionOp) {
        const auto qType = mlir::cast<NDTypeInterface>(attentionOp.getInputQ().getType());
        const auto kType = mlir::cast<NDTypeInterface>(attentionOp.getInputK().getType());

        if (qType.getRank() != 4 || kType.getRank() != 4) {
            return;
        }

        const auto qShape = qType.getShape().raw();
        const auto kShape = kType.getShape().raw();
        const auto N = qShape[0];
        const auto qHeads = qShape[1];
        const auto tSL = qShape[2];
        const auto E = qShape[3];
        const auto kvHeads = kShape[1];
        const auto sSL = kShape[2];

        // Guard: MQA only (qHeads > 1, kvHeads == 1). Bias support requires tiles along the
        // attention-score dimension and is left for a follow-on change.
        if (qHeads <= 1 || kvHeads != 1 || attentionOp.getInputBias() != nullptr) {
            return;
        }

        // Restrict head-folding to shape configurations known to improve performance.
        // Other shapes regress, so they are intentionally excluded until validated.
        struct FoldShape {
            int64_t qHeads;
            int64_t tSL;
            int64_t sSL;
            int64_t E;
        };
        static const SmallVector<FoldShape> ALLOWED_MQA_FOLD_SHAPES = {
                // {qHeads, tSL, sSL, E}
                {8, 10, 826, 256},
        };
        const auto matchesAllowed = [&](const FoldShape& s) {
            return s.qHeads == qHeads && s.tSL == tSL && s.sSL == sSL && s.E == E;
        };
        if (!llvm::any_of(ALLOWED_MQA_FOLD_SHAPES, matchesAllowed)) {
            return;
        }

        auto builder = mlir::OpBuilder(attentionOp);
        const auto ctx = builder.getContext();
        const auto loc = attentionOp->getLoc();

        // Reshape Q: [N, qHeads, tSL, E] -> [N, 1, qHeads*tSL, E]
        const auto newTSL = qHeads * tSL;
        SmallVector<int64_t> newQShape = {N, 1, newTSL, E};
        auto reshapedQ = builder.create<IE::ReshapeOp>(appendLoc(loc, "mqa_fold_q"), attentionOp.getInputQ(),
                                                       getIntArrayAttr(ctx, newQShape))
                                 .getOutput();

        // Tile mask along the tSL dimension (second-to-last) by qHeads.
        // Mask row h*tSL+t maps to original row t, matching the head-folded Q layout.
        mlir::Value tiledMask = attentionOp.getInputMask();
        if (tiledMask) {
            const auto maskRank = mlir::cast<NDTypeInterface>(tiledMask.getType()).getRank();
            SmallVector<int64_t> maskTiles(maskRank, 1);
            maskTiles[maskRank - 2] = qHeads;
            tiledMask = builder.create<IE::TileOp>(appendLoc(loc, "mqa_fold_mask"), tiledMask,
                                                   getIntArrayAttr(ctx, maskTiles))
                                .getOutput();
        }

        auto newAttOp = builder.create<IE::AttentionOp>(appendLoc(loc, "mqa_fold_attention"), reshapedQ,
                                                        attentionOp.getInputK(), attentionOp.getInputV(), tiledMask,
                                                        attentionOp.getInputScale(), attentionOp.getInputSink(),
                                                        /*bias=*/nullptr, attentionOp.getPadSizeSAttr());

        // Reshape output: [N, 1, qHeads*tSL, vE] -> original [N, qHeads, tSL, vE]
        const auto origOutputShape = mlir::cast<NDTypeInterface>(attentionOp.getOutput().getType()).getShape().raw();
        SmallVector<int64_t> reshapeBackShape(origOutputShape.begin(), origOutputShape.end());
        auto reshapedOut = builder.create<IE::ReshapeOp>(appendLoc(loc, "mqa_unfold_output"), newAttOp.getOutput(),
                                                         getIntArrayAttr(ctx, reshapeBackShape));

        attentionOp.getOutput().replaceAllUsesWith(reshapedOut.getOutput());
        attentionOp.erase();
    });

    // Convert AttentionOp to AttentionDMAOp if the configuration is legal for AttentionDMA.
    func->walk([&](IE::AttentionOp attentionOp) {
        if (isLegalAttentionDMA(attentionOp)) {
            auto builder = mlir::OpBuilder(attentionOp);
            mlir::Value seqLenK = findSeqLenKFromMask(attentionOp.getInputMask());
            auto newAttOp = builder.create<IE::AttentionDMAOp>(
                    attentionOp->getLoc(), attentionOp.getInputQ(), attentionOp.getInputK(), attentionOp.getInputV(),
                    seqLenK ? nullptr : attentionOp.getInputMask(), attentionOp.getInputScale(),
                    attentionOp.getInputSink(), attentionOp.getInputBias(), seqLenK, attentionOp.getPadSizeSAttr());

            attentionOp.getOutput().replaceAllUsesWith(newAttOp.getOutput());
            attentionOp.erase();
        }
    });
}

}  // namespace

//
// createFuseAttentionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseAttentionPass(Logger log) {
    return std::make_unique<FuseAttentionPass>(log);
}
