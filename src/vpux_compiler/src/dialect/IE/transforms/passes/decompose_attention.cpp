//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEATTENTION
#define GEN_PASS_DEF_DECOMPOSEATTENTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Legal Attention configurations: {qHeadSize, tSL, sSL}
struct AttentionConfig {
    int64_t qHeadSize;
    int64_t tSL;
    int64_t sSL;
};

static const SmallVector<AttentionConfig> LEGAL_ATTENTION_CONFIGS = {
        // {qHeadSize, tSL, sSL}
        {192, 225, 225}, {12, 3600, 3600}, {8, 300, 300},   {16, 577, 577},  {10, 1024, 1024}, {10, 1024, 77},
        {20, 256, 256},  {20, 256, 77},    {6, 3072, 3072}, {6, 151, 151},   {12, 512, 512},   {16, 256, 256},
        {1, 80, 826},    {6, 2752, 2752},  {6, 1886, 1886}, {20, 1500, 1500}};

bool isLegalAttention(IE::AttentionOp op) {
    auto inputQ = op.getInputQ();
    auto inputK = op.getInputK();

    auto tensorTypeQ = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto tensorTypeK = mlir::cast<NDTypeInterface>(inputK.getType());

    const auto rankQ = tensorTypeQ.getRank();
    const auto rankK = tensorTypeK.getRank();

    if (rankQ < 3 || rankQ > 4) {
        return false;
    }
    if (rankK < 3 || rankK > 4) {
        return false;
    }

    // Boolean mask (i8 signless) is not yet supported in legal SDPA configs
    auto inputMask = op.getInputMask();
    if (inputMask) {
        auto tensorTypeMask = mlir::cast<NDTypeInterface>(inputMask.getType());
        auto maskElemType = tensorTypeMask.getElementType();
        if (maskElemType.isSignlessInteger(8)) {
            return false;
        }
    }

    auto shapeQ = tensorTypeQ.getShape().raw();
    auto shapeK = tensorTypeK.getShape().raw();
    const auto qHeadSize = (rankQ == 3) ? shapeQ[0] : shapeQ[0] * shapeQ[1];
    const auto kvHeadSize = (rankK == 3) ? shapeK[0] : shapeK[0] * shapeK[1];
    const auto tSL = shapeQ[rankQ - 2];
    const auto sSL = shapeK[rankK - 2];

    if (op.getInputSink() != nullptr && tSL == 1024) {
        return true;
    }

    const bool isGQA = (qHeadSize > kvHeadSize) && (kvHeadSize > 1);
    const bool isMQA = (qHeadSize > kvHeadSize) && (kvHeadSize == 1);

    if (isGQA) {  // Disable GQA Legalization
        return false;
    } else if (isMQA) {  // Enable MQA Legalization
        return VPU::AttentionOp::isSupported(op);
    }

    auto matches = [&](const AttentionConfig& config) {
        return (config.qHeadSize == qHeadSize) && (config.sSL == sSL) && (config.tSL == tSL);
    };

    if (llvm::any_of(LEGAL_ATTENTION_CONFIGS, matches)) {
        return true;
    }

    return false;
}

enum class ScalePlacement { NONE, OnQuery, OnKey, OnResult };

ScalePlacement determineScalePlacement(int64_t L, int64_t S, int64_t E) {
    const int64_t costOnQuery = L * E;
    const int64_t costOnKey = S * E;
    const int64_t costOnResult = L * S;

    if (costOnQuery <= costOnKey && costOnQuery <= costOnResult) {
        return ScalePlacement::OnQuery;
    } else if (costOnKey <= costOnResult) {
        return ScalePlacement::OnKey;
    } else {
        return ScalePlacement::OnResult;
    }
}

//
// Prepare attention inputs for MQA/GQA by optionally collapsing batch into channels (Q)
// or broadcasting K/V heads to match the total Q head count.
// Broadcast pattern: Reshape to 5D, broadcast over group dimension, flatten back to 4D
// Example (K/V broadcast): [N, 8, H, W] -> [N, 8, 1, H, W] -> [N, 8, 4, H, W] -> [N, 32, H, W]
// Example (batch collapse): [N, C, H, W] -> [1, N*C, H, W]
//
mlir::Value prepareAttentionInput(mlir::OpBuilder& builder, mlir::Location loc, mlir::MLIRContext* ctx,
                                  mlir::Value input, int64_t batchSize, int64_t currentHeads, int64_t targetHeads,
                                  StringRef suffix, bool isQuery) {
    mlir::Value processedInput = input;

    if (auto transposeOp = input.getDefiningOp<IE::TransposeOp>()) {
        if (IE::isWHSwappingTranspose(transposeOp)) {
            processedInput = transposeOp.getInput();
        }
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(processedInput.getType());
    auto inputShape = inputType.getShape().raw();
    const auto rank = inputShape.size();
    const auto L = inputShape[rank - 2];
    const auto E = inputShape[rank - 1];

    int64_t headsAfterReshape = currentHeads;
    if (batchSize > 1) {
        SmallVector<int64_t> reshapeTarget;
        if (isQuery) {
            reshapeTarget = {1, batchSize * currentHeads, L, E};
            headsAfterReshape = batchSize * currentHeads;
        } else {
            reshapeTarget = {1, batchSize, L, E};
            headsAfterReshape = batchSize;
        }
        // Look through any preceding reshape whose input already has the target shape,
        // avoiding a cancel-out pair (e.g. a FuseAttention batch-merge that goes
        // [1, H, L, E] -> [B, H/B, L, E] immediately undone by the reshape below).
        bool skippedCancelOutReshape = false;
        if (auto defOp = processedInput.getDefiningOp()) {
            if (mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp>(defOp)) {
                const auto srcShape =
                        mlir::cast<vpux::NDTypeInterface>(defOp->getOperand(0).getType()).getShape().raw();
                if (srcShape == ArrayRef<int64_t>(reshapeTarget)) {
                    processedInput = defOp->getOperand(0);
                    skippedCancelOutReshape = true;
                }
            }
        }
        if (!skippedCancelOutReshape) {
            processedInput = builder.create<IE::ReshapeOp>(appendLoc(loc, formatv("{0}_reshape", suffix)),
                                                           processedInput, getIntArrayAttr(ctx, reshapeTarget))
                                     .getOutput();
        }
    }

    // Broadcast K/V to match Q heads using 5D intermediate shape
    if (!isQuery && headsAfterReshape < targetHeads) {
        const int64_t repeatFactor = targetHeads / headsAfterReshape;
        const int64_t N = (batchSize > 1) ? 1 : ((rank == 4) ? inputShape[0] : 1);
        const int64_t heads = headsAfterReshape;

        // Step 1: AffineReshape to 5D [N, heads, 1, L, E]
        // 4D -> 5D: split heads dimension into (heads, 1)
        SmallVector<int64_t> shape5D = {N, heads, 1, L, E};
        SmallVector<SmallVector<int64_t>> dimMapping4Dto5D = {{0}, {1, 2}, {3}, {4}};
        processedInput = builder.create<IE::AffineReshapeOp>(appendLoc(loc, formatv("{0}_reshape_5d", suffix)),
                                                             processedInput, getIntArrayOfArray(ctx, dimMapping4Dto5D),
                                                             getIntArrayAttr(ctx, shape5D))
                                 .getOutput();

        // Step 2: Broadcast [N, heads, 1, L, E] -> [N, heads, repeatFactor, L, E]
        SmallVector<int64_t> broadcastShape = {N, heads, repeatFactor, L, E};
        auto shapeStorageType =
                mlir::RankedTensorType::get({static_cast<int64_t>(broadcastShape.size())}, getSInt64Type(ctx));
        auto shapeConst =
                Const::createConst(builder, appendLoc(loc, formatv("{0}_broadcast_5d", suffix)), shapeStorageType,
                                   ArrayRef(broadcastShape), [&](Const::ContentSetup& setup) {
                                       return setup.castElemType(getSInt32Type(ctx));
                                   });
        processedInput = builder.create<IE::BroadcastOp>(
                                        appendLoc(loc, formatv("{0}_broadcast", suffix)), processedInput, shapeConst,
                                        nullptr, IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::BIDIRECTIONAL))
                                 .getOutput();

        // Step 3: AffineReshape back to 4D [N, targetHeads, L, E]
        // 5D -> 4D: merge (heads, repeatFactor) dimensions into targetHeads
        SmallVector<int64_t> shape4D = {N, targetHeads, L, E};
        SmallVector<SmallVector<int64_t>> dimMapping5Dto4D = {{0}, {1}, {1}, {2}, {3}};
        processedInput = builder.create<IE::AffineReshapeOp>(appendLoc(loc, formatv("{0}_reshape_4d", suffix)),
                                                             processedInput, getIntArrayOfArray(ctx, dimMapping5Dto4D),
                                                             getIntArrayAttr(ctx, shape4D))
                                 .getOutput();
    }

    return processedInput;
}

//
// Collect preprocessing operations that uniquely feed into Value
//
SmallVector<mlir::Operation*> collectUniquePreprocessingOps(mlir::Value value) {
    SmallVector<mlir::Operation*> opsToClone;

    mlir::Value current = value;
    while (auto definingOp = current.getDefiningOp()) {
        // Stop if we hit a constant
        if (mlir::isa<Const::DeclareOp>(definingOp)) {
            break;
        }

        // Check if this operation's result is used only once (by the current chain)
        if (!definingOp->getResult(0).hasOneUse()) {
            break;
        }

        // Add this operation to the list
        opsToClone.push_back(definingOp);
        current = definingOp->getOperand(0);
    }

    // Reverse to get execution order (input to output)
    std::reverse(opsToClone.begin(), opsToClone.end());

    return opsToClone;
}

//
// Clone preprocessing operations
//
mlir::Value clonePreprocessingOps(mlir::OpBuilder& builder, ArrayRef<mlir::Operation*> ops, mlir::Value initialInput) {
    if (ops.empty()) {
        return initialInput;
    }

    mlir::IRMapping mapping;
    // Map the initial input (the source before preprocessing chain)
    if (!ops.empty() && ops[0]->getNumOperands() > 0) {
        mapping.map(ops[0]->getOperand(0), initialInput);
    }

    mlir::Value currentOutput = initialInput;
    for (auto [idx, op] : llvm::enumerate(ops)) {
        auto* clonedOp = builder.clone(*op, mapping);
        extendOpLoc(clonedOp, "cloned_{0}", idx);
        currentOutput = clonedOp->getResult(0);
    }

    return currentOutput;
}

//
// Create a valid MatMul operator with automatic transpose_b handling.
// outWasTransposedB (optional): set to true when a WH-swapping Transpose on B
//   was peeled from the input. Callers should skip that Transpose when collecting
//   V preprocessing ops (use B's pre-transpose value as the collection root).
// outSquareRebuildTranspose (optional): when the matrix is square and a new
//   equivalent WH-swapping Transpose was re-created inside this function, the
//   caller receives that TransposeOp. Step 7 can reuse its orderAttr directly
//   instead of reconstructing the permutation map.
//
mlir::Value createMatMul(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value a, mlir::Value b, StringRef suffix,
                         bool* outWasTransposedB = nullptr, IE::TransposeOp* outSquareRebuildTranspose = nullptr) {
    bool wasTransposedB = false;
    if (auto transposeOp = b.getDefiningOp<IE::TransposeOp>()) {
        if (IE::isWHSwappingTranspose(transposeOp)) {
            b = transposeOp.getInput();
            wasTransposedB = true;
        }
    }

    const auto aType = mlir::cast<vpux::NDTypeInterface>(a.getType());
    const auto bType = mlir::cast<vpux::NDTypeInterface>(b.getType());

    const auto aShape = aType.getShape().raw();
    const auto bShape = bType.getShape().raw();
    VPUX_THROW_UNLESS(aShape.size() == bShape.size(), "MatMul inputs must have the same rank, got {0} and {1}",
                      aShape.size(), bShape.size());

    const auto rank = aShape.size();
    const auto widthA = aShape[rank - 1];
    const auto heightB = bShape[rank - 2];
    const auto widthB = bShape[rank - 1];

    if (widthB == heightB && wasTransposedB) {
        SmallVector<unsigned> transposeOrder;
        for (unsigned i = 0; i < rank - 2; i++) {
            transposeOrder.push_back(i);
        }
        transposeOrder.push_back(rank - 1);
        transposeOrder.push_back(rank - 2);

        auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transposeOrder, builder.getContext()));
        auto rebuildTranspose =
                builder.create<IE::TransposeOp>(appendLoc(loc, "{0}_transpose_b", suffix), b, nullptr, orderAttr);
        b = rebuildTranspose.getOutput();
        if (outSquareRebuildTranspose != nullptr) {
            *outSquareRebuildTranspose = rebuildTranspose;
        }
    }

    if (outWasTransposedB != nullptr) {
        *outWasTransposedB = wasTransposedB;
    }

    bool transposeB = (widthA != heightB) || (widthB == heightB);
    auto matmul = builder.create<IE::MatMulOp>(appendLoc(loc, suffix), a, b,
                                               /*transpose_a=*/false, /*transpose_b=*/transposeB);

    return matmul.getOutput();
}

void decomposeAttention(IE::AttentionOp origOp, Logger log, bool maskAware, bool forceScaleOnResult) {
    log.trace("Got AttentionOp for decomposition - '{0}'", origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    const auto ctx = origOp.getContext();
    const auto loc = origOp.getLoc();

    auto query = origOp.getInputQ();
    auto key = origOp.getInputK();
    auto value = origOp.getInputV();
    auto attentionMask = origOp.getInputMask();
    auto scale = origOp.getInputScale();
    auto bias = origOp.getInputBias();

    const auto queryType = mlir::cast<vpux::NDTypeInterface>(query.getType());
    const auto queryShape = queryType.getShape();
    const auto rank = queryShape.size();

    const auto keyType = mlir::cast<vpux::NDTypeInterface>(key.getType());
    const auto keyShape = keyType.getShape();

    const auto valueType = mlir::cast<vpux::NDTypeInterface>(value.getType());
    const auto valueShape = valueType.getShape();

    const auto L = queryShape.raw()[rank - 2];
    const auto E = queryShape.raw()[rank - 1];
    const auto S = keyShape.raw()[rank - 2];

    const auto qBatch = (rank == 4) ? queryShape.raw()[0] : 1;
    const auto qHeads = (rank == 4) ? queryShape.raw()[1] : queryShape.raw()[0];
    const auto kBatch = (rank == 4) ? keyShape.raw()[0] : 1;
    const auto kHeads = (rank == 4) ? keyShape.raw()[1] : keyShape.raw()[0];
    const auto vBatch = (rank == 4) ? valueShape.raw()[0] : 1;
    const auto vHeads = (rank == 4) ? valueShape.raw()[1] : valueShape.raw()[0];

    // Use mlir::Value for query/key/value since they may be reassigned
    mlir::Value queryVal = query;
    mlir::Value keyVal = key;
    mlir::Value valueVal = value;

    // Handle MQA/GQA: Prepare inputs when Q has more heads than K/V
    if (qHeads > kHeads && qHeads > vHeads) {
        if (qBatch > 1) {
            queryVal = prepareAttentionInput(builder, loc, ctx, queryVal, qBatch, qHeads, qBatch * qHeads, "q",
                                             /*isQuery=*/true);

            if (attentionMask) {
                const auto maskShape = mlir::cast<vpux::NDTypeInterface>(attentionMask.getType()).getShape().raw();
                if (maskShape.size() == 4 && maskShape[0] == qBatch) {
                    SmallVector<int64_t> newMaskShape = {1, qBatch * maskShape[1], maskShape[2], maskShape[3]};
                    attentionMask = builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_mask_gqa"), attentionMask,
                                                                  getIntArrayAttr(ctx, newMaskShape))
                                            .getOutput();
                }
            }
        }
        keyVal = prepareAttentionInput(builder, loc, ctx, keyVal, kBatch, kHeads, qBatch * qHeads, "k",
                                       /*isQuery=*/false);

        valueVal = prepareAttentionInput(builder, loc, ctx, valueVal, vBatch, vHeads, qBatch * qHeads, "v",
                                         /*isQuery=*/false);
    }

    // Cheapest tensor to scale; NPU5+ uses this placement unchanged.
    ScalePlacement scalePlacement = (scale != nullptr) ? determineScalePlacement(L, S, E) : ScalePlacement::NONE;

    if (!forceScaleOnResult) {
        if (scalePlacement == ScalePlacement::OnQuery) {
            queryVal = builder.createOrFold<IE::MultiplyOp>(appendLoc(loc, "scale_query"), queryVal, scale,
                                                            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr,
                                                            nullptr);
            log.trace("Applied scale on Query (L={0}, S={1}, E={2})", L, S, E);
        } else if (scalePlacement == ScalePlacement::OnKey) {
            keyVal = builder.createOrFold<IE::MultiplyOp>(appendLoc(loc, "scale_key"), keyVal, scale,
                                                          IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr,
                                                          nullptr);
            log.trace("Applied scale on Key (L={0}, S={1}, E={2})", L, S, E);
        }
    }

    // Step 1: Compute Q * K^T using MatMul.
    mlir::Value attentionScores = createMatMul(builder, loc, queryVal, keyVal, "qk_matmul");

    // NPU40XX: force OnResult so FuseScale folds the scale into the per-head matmul, but only when the
    // QK MatMul unrolls per-head (!isGroupedMatMulBeneficial) and the shapes are misaligned.
    if (forceScaleOnResult && scale != nullptr) {
        auto qkMatMul = attentionScores.getDefiningOp<IE::MatMulOp>();
        VPUX_THROW_WHEN(qkMatMul == nullptr, "Expected QK MatMul for scale placement");
        const bool seqMisaligned = (S % 16 != 0) || (L % 16 != 0);
        const bool willUnrollPerHead = !IE::isGroupedMatMulBeneficial(qkMatMul, getShape(qkMatMul.getInput1()),
                                                                      getShape(qkMatMul.getInput2()));
        if (willUnrollPerHead && seqMisaligned) {
            scalePlacement = ScalePlacement::OnResult;
        }

        // Scale is a per-tensor scalar, so scaling a MatMul operand equals scaling the result.
        if (scalePlacement == ScalePlacement::OnQuery || scalePlacement == ScalePlacement::OnKey) {
            const bool onQuery = (scalePlacement == ScalePlacement::OnQuery);
            const mlir::Value operandVal = onQuery ? qkMatMul.getInput1() : qkMatMul.getInput2();

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(qkMatMul);
            auto scaledOperand = builder.createOrFold<IE::MultiplyOp>(
                    appendLoc(loc, onQuery ? "scale_query" : "scale_key"), operandVal, scale,
                    IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
            if (onQuery) {
                qkMatMul.getInput1Mutable().assign(scaledOperand);
            } else {
                qkMatMul.getInput2Mutable().assign(scaledOperand);
            }
            log.trace("Applied scale on {0} (L={1}, S={2}, E={3})", onQuery ? "Query" : "Key", L, S, E);
        }
    }

    // Step 2: Apply scale on result if that's the optimal placement
    if (scalePlacement == ScalePlacement::OnResult) {
        attentionScores =
                builder.createOrFold<IE::MultiplyOp>(appendLoc(loc, "scale_multiply"), attentionScores, scale,
                                                     IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied scale on result (L={0}, S={1}, E={2})", L, S, E);
    }

    // Step 3: Add attention mask if provided
    if (attentionMask) {
        auto maskType = mlir::cast<vpux::NDTypeInterface>(attentionMask.getType());
        auto maskElemType = maskType.getElementType();

        if (maskElemType.isSignlessInteger(8)) {
            // Get the target element type from attention scores
            auto scoresType = mlir::cast<vpux::NDTypeInterface>(attentionScores.getType());
            auto targetElemType = mlir::RankedTensorType::get({}, scoresType.getElementType());

            // Create constant for "keep" position
            auto keepValueConst = Const::createConst(builder, appendLoc(loc, "mask_keep_value"), targetElemType,
                                                     llvm::ArrayRef<float>{0.0f});

            // Create constant for "mask" position
            auto maskValueConst = Const::createConst(builder, appendLoc(loc, "mask_mask_value"), targetElemType,
                                                     llvm::ArrayRef<float>{-std::numeric_limits<float>::infinity()});

            // If mask != 0 (true), use keepValue (0.0), else use maskValue (-INF)
            attentionMask = builder.create<IE::SelectOp>(
                                           appendLoc(loc, "mask_select"), attentionMask, keepValueConst, maskValueConst,
                                           IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY))
                                    .getOutput();
        }

        attentionScores =
                builder.createOrFold<IE::AddOp>(appendLoc(loc, "mask_add"), attentionScores, attentionMask,
                                                IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied attention mask");
    }

    // Step 4: Add bias if provided
    if (bias) {
        attentionScores =
                builder.createOrFold<IE::AddOp>(appendLoc(loc, "bias_add"), attentionScores, bias,
                                                IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied bias");
    }

    auto sink = origOp.getInputSink();
    const auto preSinkShape = mlir::cast<vpux::NDTypeInterface>(attentionScores.getType()).getShape();
    const SmallVector<int64_t> preSinkSizes(preSinkShape.begin(), preSinkShape.end());
    if (sink) {
        mlir::Value sinkCol = sink;
        SmallVector<int64_t> sinkColShape(preSinkSizes);
        sinkColShape.back() = 1;
        const auto sinkShape = mlir::cast<vpux::NDTypeInterface>(sink.getType()).getShape();
        if (SmallVector<int64_t>(sinkShape.begin(), sinkShape.end()) != sinkColShape) {
            auto shapeStorageType =
                    mlir::RankedTensorType::get({static_cast<int64_t>(sinkColShape.size())}, getSInt64Type(ctx));
            auto shapeConst = Const::createConst(builder, appendLoc(loc, "sink_shape"), shapeStorageType,
                                                 ArrayRef(sinkColShape), [&](Const::ContentSetup& setup) {
                                                     return setup.castElemType(getSInt32Type(ctx));
                                                 });
            sinkCol = builder.create<IE::BroadcastOp>(appendLoc(loc, "sink_broadcast"), sink, shapeConst, nullptr,
                                                      IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::BIDIRECTIONAL))
                              .getOutput();
        }
        attentionScores = builder.create<IE::ConcatOp>(appendLoc(loc, "sink_concat"),
                                                       SmallVector<mlir::Value>{attentionScores, sinkCol},
                                                       Dim(static_cast<size_t>(rank) - 1))
                                  .getOutput();
        log.trace("Appended attention sink column before softmax");
    }

    // Step 5: Apply Softmax on the last dimension
    const auto softmaxAxisAttr = getIntAttr(ctx, static_cast<int64_t>(rank) - 1);
    const auto maskAwareAttr = maskAware ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr{};
    auto softmaxOp = builder.create<IE::SoftMaxOp>(appendLoc(loc, "softmax"), attentionScores, softmaxAxisAttr,
                                                   mlir::IntegerAttr{}, mlir::TypeAttr{}, maskAwareAttr);
    log.trace("Applied softmax with mask awareness: {0}", maskAware);

    mlir::Value softmaxResult = softmaxOp.getOutput();
    if (sink) {
        const SmallVector<int64_t> sliceOffsets(rank, 0);
        softmaxResult =
                builder.create<IE::SliceOp>(appendLoc(loc, "sink_slice"), softmaxResult,
                                            getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, preSinkSizes))
                        .getOutput();
        log.trace("Dropped attention sink column after softmax");
    }

    // Step 6: Compute attention_output = softmax_output * V using MatMul.
    // Track whether createMatMul consumed a WH-swapping Transpose on V
    // (vTransposePeeled) and whether a square-matrix Transpose was rebuilt
    // internally (vSquareRebuildTranspose). Both guide Step 7.
    bool vTransposePeeled = false;
    IE::TransposeOp vSquareRebuildTranspose;
    mlir::Value outputMatMul = createMatMul(builder, loc, softmaxResult, valueVal, "output_matmul", &vTransposePeeled,
                                            &vSquareRebuildTranspose);

    origOp.getOutput().replaceAllUsesWith(outputMatMul);
    origOp.erase();

    // Step 7: Move V preprocessing operations right before the MatMul that uses V.
    //
    // When vTransposePeeled=true the WH-swapping Transpose on V was consumed by
    // createMatMul. Collect preprocessing ops from the pre-transpose value to
    // avoid re-cloning the Transpose with an incompatible shape.
    //
    // When vSquareRebuildTranspose is set, createMatMul additionally re-created
    // an equivalent WH-swapping Transpose on the square rawV. After cloning the
    // preprocessing chain we re-apply a Transpose — reusing the rebuilt op's
    // orderAttr — so the MatMul (transposeB=true) still computes softmax × rawV.
    mlir::Value vForPreprocessing = valueVal;
    if (vTransposePeeled) {
        if (auto transposeOp = valueVal.getDefiningOp<IE::TransposeOp>()) {
            if (IE::isWHSwappingTranspose(transposeOp)) {
                vForPreprocessing = transposeOp.getInput();
            }
        }
    }
    auto preprocessingOps = collectUniquePreprocessingOps(vForPreprocessing);
    if (!preprocessingOps.empty()) {
        auto matmulOp = outputMatMul.getDefiningOp<IE::MatMulOp>();
        if (matmulOp) {
            builder.setInsertionPoint(matmulOp);
            mlir::Value originalInput = preprocessingOps[0]->getOperand(0);
            mlir::Value newVInput = clonePreprocessingOps(builder, preprocessingOps, originalInput);
            // In the square-transpose-rebuilt case the MatMul relies on
            // transposeB=true. Clone the rebuilt Transpose with its input remapped
            // to newVInput so the computation remains softmax × newVInput.
            if (vSquareRebuildTranspose) {
                mlir::IRMapping mapping;
                mapping.map(vSquareRebuildTranspose.getInput(), newVInput);
                newVInput = mlir::cast<IE::TransposeOp>(builder.clone(*vSquareRebuildTranspose.getOperation(), mapping))
                                    .getOutput();
            }
            matmulOp.getInput2Mutable().assign(newVInput);

            // Delete old preprocessing operations that are no longer used
            for (auto it = preprocessingOps.rbegin(); it != preprocessingOps.rend(); ++it) {
                auto* op = *it;
                if (op->use_empty()) {
                    op->erase();
                }
            }
        }
    }

    log.trace("Successfully decomposed Attention operation");
}

//
// Traverse through reshape ops to reach the underlying constant.
//
static Const::DeclareOp findConstThroughReshapes(mlir::Value val) {
    mlir::Value current = val;
    while (current) {
        if (auto constOp = current.getDefiningOp<Const::DeclareOp>()) {
            return constOp;
        }
        auto defOp = current.getDefiningOp();
        if (mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp>(defOp)) {
            current = defOp->getOperand(0);
        } else {
            break;
        }
    }
    return nullptr;
}

//
// Replace FP16_MIN (-65504) with FP16 -inf on SelectOp branches feeding the attention mask.
//
static bool replaceMaskFp16MinWithInf(IE::AttentionOp op, Logger log) {
    const auto mask = op.getInputMask();
    if (!mask) {
        return false;
    }

    auto selectOp = mask.getDefiningOp<IE::SelectOp>();
    if (!selectOp) {
        return false;
    }

    const auto fp16Min = static_cast<float>(std::numeric_limits<type::float16>::lowest());

    bool anyReplaced = false;

    auto replaceIfFp16Min = [&](mlir::Value branchVal, unsigned operandIdx) {
        auto constOp = findConstThroughReshapes(branchVal);
        if (!constOp) {
            return;
        }

        const auto branchType = mlir::cast<NDTypeInterface>(branchVal.getType());
        if (!mlir::isa<mlir::Float16Type>(branchType.getElementType())) {
            return;
        }

        const auto content = constOp.getContent();
        const auto hasFp16Min = content.isSplat() ? (content.getSplatValue<float>() == fp16Min)
                                                  : llvm::any_of(content.getValues<float>(), [fp16Min](float v) {
                                                        return v == fp16Min;
                                                    });
        if (!hasFp16Min) {
            return;
        }

        const auto allFp16Min = content.isSplat() || llvm::all_of(content.getValues<float>(), [fp16Min](float v) {
                                    return v == fp16Min;
                                });

        mlir::OpBuilder builder(selectOp);
        const auto ctx = builder.getContext();
        const auto shape = branchType.getShape();
        const auto constType = mlir::RankedTensorType::get(SmallVector<int64_t>(shape.begin(), shape.end()),
                                                           mlir::Float16Type::get(ctx));
        const auto minusInf = -std::numeric_limits<float>::infinity();

        if (allFp16Min) {
            const auto minusInfConst = Const::createConst<float>(builder, appendLoc(selectOp->getLoc(), "minus_inf"),
                                                                 constType, llvm::ArrayRef<float>{minusInf});
            selectOp->setOperand(operandIdx, minusInfConst);
            log.trace("  Replaced splat FP16_MIN ({0}) with FP16 -inf in SelectOp mask branch {1}", fp16Min,
                      operandIdx);
            anyReplaced = true;
        } else {
            // Replace only FP16_MIN entries, preserving other values (e.g. 0.0 in causal masks)
            const auto oldVals =
                    std::vector<float>(content.getValues<float>().begin(), content.getValues<float>().end());
            std::vector<float> newVals;
            newVals.reserve(oldVals.size());
            for (float v : oldVals) {
                newVals.push_back(v == fp16Min ? minusInf : v);
            }
            const auto newConst = Const::createConst<float>(builder, appendLoc(selectOp->getLoc(), "minus_inf_mixed"),
                                                            constType, llvm::ArrayRef<float>(newVals));
            selectOp->setOperand(operandIdx, newConst);
            log.trace("  Replaced FP16_MIN values in mixed constant with FP16 -inf in SelectOp mask branch {0}",
                      operandIdx);
            anyReplaced = true;
        }
    };

    replaceIfFp16Min(selectOp.getInput2(), 1);
    replaceIfFp16Min(selectOp.getInput3(), 2);
    log.trace("Replaced FP16_MIN with FP16 -inf in SelectOp mask branches: {0}", anyReplaced);
    return anyReplaced;
}

//
// DecomposeAttentionPass
//

class DecomposeAttentionPass final : public IE::impl::DecomposeAttentionBase<DecomposeAttentionPass> {
public:
    explicit DecomposeAttentionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeAttentionPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](IE::AttentionOp origOp) {
        const bool maskAware = replaceMaskFp16MinWithInf(origOp, _log);

        const auto arch = config::getArch(origOp);
        if (arch != config::ArchKind::NPU40XX && isLegalAttention(origOp)) {
            return;
        }
        const bool forceScaleOnResult = (arch == config::ArchKind::NPU40XX);
        decomposeAttention(origOp, _log, maskAware, forceScaleOnResult);
    });
}

}  // namespace

//
// createDecomposeAttentionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeAttentionPass(Logger log) {
    return std::make_unique<DecomposeAttentionPass>(log);
}
