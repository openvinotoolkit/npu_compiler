//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// ChunkGatedDeltaRuleLoop
//
// Detects Loop ops whose body implements the Gated Delta Rule (GDR) recurrence:
//   h_t = exp(g_t) * h_{t-1} + k_t ⊗ β_t * (v_t − ⟨h_{t-1}, k_t⟩)
//   y_t = ⟨h_t, q_t⟩
// and rewrites them into a chunked form:
//   Loop(N) → [Reshape inputs to chunks] + [Intra-chunk parallel computation] + Loop(N/chunk_size)
//
// The mathematical equivalence relies on the linearity of the recurrence:
//   h_t = M_t * h_{t-1} + u_t,  where M_t = exp(g_t)(I - β_t * k_t * k_t^T)
// which allows decomposition into intra-chunk (parallel) and inter-chunk (sequential) parts.
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CHUNKGATEDDELTARULELOOP
#define GEN_PASS_DEF_CHUNKGATEDDELTARULELOOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//===----------------------------------------------------------------------===//
// GDR Pattern Recognition
//===----------------------------------------------------------------------===//

// Structure to hold recognized GDR Loop components.
// All indices are body block argument indices.
struct GDRLoopDescriptor {
    // Body block argument indices for the 5 sliced inputs
    int64_t kIdx = -1;     // B_bar / key:      [batch, heads, 1, d_k]
    int64_t qIdx = -1;     // C / query:         [batch, heads, 1, d_k]
    int64_t vIdx = -1;     // A_bar / value:     [batch, heads, 1, d_v]
    int64_t gateIdx = -1;  // gate (log domain): [batch, heads, 1]
    int64_t betaIdx = -1;  // beta / dt:         [batch, heads, 1]

    // Body block argument indices for feedback inputs
    int64_t stateIdx = -1;  // h state:         [batch, heads, d_k, d_v]
    int64_t accumIdx = -1;  // output accum:    [batch, heads, seq_len, d_v]

    // Current iteration index
    int64_t iterIdx = -1;

    // Dimensions extracted from shapes
    int64_t batch = -1;
    int64_t heads = -1;
    int64_t seqLen = -1;
    int64_t dK = -1;
    int64_t dV = -1;

    // Slice input descriptors: external_port_id -> internal_body_arg_id for sliced inputs
    SmallVector<std::pair<int64_t, int64_t>> slicePortMap;
};

/// Check whether an operation is a reshape-like op (AffineReshape, Reshape, Squeeze, Unsqueeze).
static bool isReshapeKindOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp, IE::SqueezeOp, IE::UnsqueezeOp>(op);
}

/// Trace a value back through reshape-like ops to find the source BlockArgument.
/// Returns nullptr if the chain leads to a non-reshape op.
static mlir::BlockArgument traceToBlockArg(mlir::Value val) {
    while (val) {
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
            return blockArg;
        }
        auto defOp = val.getDefiningOp();
        if (!defOp) {
            return nullptr;
        }
        if (isReshapeKindOp(defOp)) {
            val = defOp->getOperand(0);
            continue;
        }
        return nullptr;
    }
    return nullptr;
}

/// Trace ScatterUpdate indices back to the source BlockArgument.
/// The index value may be normalized through shape-only ops and type conversion.
static mlir::BlockArgument traceScatterIndicesToBlockArg(mlir::Value val) {
    while (val) {
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
            return blockArg;
        }
        auto defOp = val.getDefiningOp();
        if (auto convertOp = mlir::dyn_cast_if_present<IE::ConvertOp>(defOp)) {
            val = convertOp.getInput();
            continue;
        }
        if (isReshapeKindOp(defOp)) {
            val = defOp->getOperand(0);
            continue;
        }
        return nullptr;
    }
    return nullptr;
}

/// Check whether a value (possibly through reshape-like ops or elementwise unary ops like Negative)
/// eventually feeds into an op of type T.
template <typename T>
static bool feedsIntoOp(mlir::Value val) {
    SmallVector<mlir::Value, 4> worklist;
    worklist.push_back(val);
    while (!worklist.empty()) {
        auto v = worklist.pop_back_val();
        for (auto* user : v.getUsers()) {
            if (mlir::isa<T>(user)) {
                return true;
            }
            if (isReshapeKindOp(user) || mlir::isa<IE::NegativeOp>(user)) {
                worklist.push_back(user->getResult(0));
            }
        }
    }
    return false;
}

/// Check whether the gate signal is negated (or multiplied by a negative constant) on the path
/// from gateArg to the ExpOp. If true, the raw gate values entering the Loop are positive
/// (they require negation inside the body to become decay factors), which means chunking is
/// UNSAFE — the CumSum of positive gate values grows unboundedly, causing exp(cumGate) overflow.
///
/// When this returns false, gate values are already negative at the Loop boundary (standard
/// Mamba2 pattern: gate = -softplus(A)*dt), and chunking is safe since exp(cumGate) <= 1.
///
/// The function traces forward from gateArg toward the ExpOp, tracking whether a negation
/// has been encountered on the path. It only considers the path(s) that actually reach Exp.
static bool isGateNegatedBeforeExp(mlir::Value gateArg) {
    // Each entry: (value to trace, has_been_negated_so_far)
    SmallVector<std::pair<mlir::Value, bool>, 8> worklist;
    worklist.push_back({gateArg, false});
    while (!worklist.empty()) {
        auto [v, negated] = worklist.pop_back_val();
        for (auto* user : v.getUsers()) {
            if (mlir::isa<IE::ExpOp>(user)) {
                // Reached Exp — return whether negation was applied on this path
                if (negated) {
                    return true;
                }
                // Found a path to Exp without negation → not safe
                return false;
            }
            if (mlir::isa<IE::NegativeOp>(user)) {
                // Continue tracing with negation flag set
                worklist.push_back({user->getResult(0), true});
                continue;
            }
            if (auto mulOp = mlir::dyn_cast<IE::MultiplyOp>(user)) {
                auto otherInput = (mulOp.getInput1() == v) ? mulOp.getInput2() : mulOp.getInput1();
                bool mulByNegative = false;
                if (auto constOp = otherInput.getDefiningOp<Const::DeclareOp>()) {
                    auto content = constOp.getContentAttr().fold();
                    if (content.isSplat()) {
                        auto splatVal = content.getSplatValue<float>();
                        if (splatVal < 0.0f) {
                            mulByNegative = true;
                        }
                    }
                }
                worklist.push_back({mulOp->getResult(0), negated || mulByNegative});
                continue;
            }
            if (isReshapeKindOp(user)) {
                worklist.push_back({user->getResult(0), negated});
            }
        }
    }
    // Never reached Exp (shouldn't happen since feedsIntoOp<ExpOp> was checked earlier)
    return false;
}

/// Try to match the GDR body pattern in a Loop op.
/// The pattern consists of:
///   1) exp(gate) → decay factor
///   2) decay * h_state → decayed state [d_k, d_v]
///   3) (h_state * k) → ReduceSum → ⟨h, k⟩ → Subtract(v, ...) → delta correction
///   4) delta * beta → scaled correction
///   5) outer_product(k, correction) → Add(decayed, update) → h_new
///   6) (h_new * q) → ReduceSum → y_t → ScatterUpdate(accum, iter, y_t)
static bool matchGDRPattern(IE::LoopOp loopOp, GDRLoopDescriptor& desc, Logger log) {
    auto& bodyRegion = loopOp.getBodyModule();
    if (bodyRegion.getBlocks().size() != 1) {
        log.trace("GDR match failed: body has multiple blocks");
        return false;
    }

    auto& bodyBlock = bodyRegion.front();
    auto bodyArgs = bodyBlock.getArguments();

    // Parse port map attributes needed by the matcher.
    auto sliceDescs = parseCustomAttrArray<IE::SliceInputPortMapAttr>(loopOp.getSliceInputDescsAttr());
    auto feedbackDescs = parseCustomAttrArray<IE::MergedInputPortMapAttr>(loopOp.getFeedbackInputDescsAttr());

    // Guard diagnostic-only work to avoid unnecessary compile-time overhead
    // when trace logging is disabled, since this pattern runs on every IE.Loop.
    if (log.isActive(LogLevel::Trace)) {
        log.trace("Loop body has {0} args:", bodyArgs.size());
        for (size_t i = 0; i < bodyArgs.size(); ++i) {
            if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(bodyArgs[i].getType())) {
                auto shape = rtt.getShape();
                SmallVector<int64_t> dims(shape.begin(), shape.end());
                log.trace("  arg[{0}]: rank={1} shape=[{2}] elem={3}", i, rtt.getRank(), dims, rtt.getElementType());
            } else {
                log.trace("  arg[{0}]: type={1} (not ranked tensor)", i, bodyArgs[i].getType());
            }
        }

        log.trace("SliceInputDescs: {0}, FeedbackInputDescs: {1}", sliceDescs.size(), feedbackDescs.size());
        for (auto sipm : sliceDescs) {
            log.trace("  Slice: extPort={0} -> bodyArg={1}", sipm.getExternalPortId().getValue().getSExtValue(),
                      sipm.getInternalLayerId().getValue().getSExtValue());
        }
        for (auto fipm : feedbackDescs) {
            log.trace("  Feedback: extPort={0} -> bodyArg={1} (bodyOutputIdx={2})",
                      fipm.getExternalPortId().getValue().getSExtValue(),
                      fipm.getInternalLayerId().getValue().getSExtValue(),
                      fipm.getBodyInputIndex().getValue().getSExtValue());
        }

        DenseMap<StringRef, int> opCounts;
        for (auto& op : bodyBlock.getOperations()) {
            opCounts[op.getName().getStringRef()]++;
        }
        log.trace("Body operations ({0} total):", bodyBlock.getOperations().size());
        for (auto& kv : opCounts) {
            log.trace("  {0}: {1}", kv.first, kv.second);
        }
    }

    // GDR body layout: [iter, 5 sliced inputs, 2 feedback inputs]
    constexpr size_t expectedBodyArgs = 8;
    constexpr size_t expectedSliceInputs = 5;
    constexpr size_t expectedFeedbackInputs = 2;

    if (bodyArgs.size() != expectedBodyArgs) {
        log.trace("GDR match failed: expected {0} body args, got {1}", expectedBodyArgs, bodyArgs.size());
        return false;
    }

    // Validate slice input descriptors
    if (sliceDescs.size() != expectedSliceInputs) {
        log.trace("GDR match failed: expected {0} slice inputs, got {1}", expectedSliceInputs, sliceDescs.size());
        return false;
    }

    // Validate feedback input descriptors
    if (feedbackDescs.size() != expectedFeedbackInputs) {
        log.trace("GDR match failed: expected {0} feedback inputs, got {1}", expectedFeedbackInputs,
                  feedbackDescs.size());
        return false;
    }

    // Identify body argument roles by shape analysis
    int64_t iterArgIdx = loopOp.getCurrentIterIndex();
    SmallVector<int64_t> slicedArgIndices;  // 4D args from sliced inputs
    SmallVector<int64_t> feedbackArgIndices;

    for (auto sipm : sliceDescs) {
        auto inId = sipm.getInternalLayerId().getValue().getSExtValue();
        slicedArgIndices.push_back(inId);
    }
    for (auto fipm : feedbackDescs) {
        auto bodyArgIdx = fipm.getInternalLayerId().getValue().getSExtValue();
        feedbackArgIndices.push_back(bodyArgIdx);
    }

    // Classify sliced inputs by shape:
    // [batch, heads, 1, d_k] or [batch, heads, 1, d_v] → k, q, or v
    // [batch, heads, 1] → gate or beta
    SmallVector<int64_t> rank4Sliced, rank3Sliced;
    for (auto idx : slicedArgIndices) {
        auto type = mlir::cast<mlir::RankedTensorType>(bodyArgs[idx].getType());
        if (type.getRank() == 4) {
            rank4Sliced.push_back(idx);
        } else if (type.getRank() == 3) {
            rank3Sliced.push_back(idx);
        }
    }

    if (rank4Sliced.size() != 3 || rank3Sliced.size() != 2) {
        log.trace("GDR match failed: expected 3 rank-4 and 2 rank-3 sliced inputs, got {0} rank-4 and {1} rank-3",
                  rank4Sliced.size(), rank3Sliced.size());
        return false;
    }

    // Classify feedback inputs by shape:
    // [batch, heads, d_k, d_v] → state
    // [batch, heads, seq_len, d_v] → accumulator
    int64_t stateArgIdx = -1, accumArgIdx = -1;
    for (auto idx : feedbackArgIndices) {
        auto type = mlir::cast<mlir::RankedTensorType>(bodyArgs[idx].getType());
        auto shape = type.getShape();
        if (shape.size() != 4) {
            log.trace("GDR match failed: feedback arg {0} has rank {1}, expected 4", idx, shape.size());
            return false;
        }
        // State: dim[2] == dim[3] (d_k == d_v typically, or dim[2]*dim[3] forms a square-ish matrix)
        // Accumulator: dim[2] == seq_len (much larger than d_k)
        // Simple heuristic: the one with larger dim[2] is the accumulator
        if (stateArgIdx == -1) {
            stateArgIdx = idx;
        } else {
            auto prevShape = mlir::cast<mlir::RankedTensorType>(bodyArgs[stateArgIdx].getType()).getShape();
            if (shape[2] > prevShape[2]) {
                accumArgIdx = idx;
            } else {
                accumArgIdx = stateArgIdx;
                stateArgIdx = idx;
            }
        }
    }

    if (stateArgIdx == -1 || accumArgIdx == -1) {
        log.trace("GDR match failed: could not identify state and accumulator feedback");
        return false;
    }

    auto stateType = mlir::cast<mlir::RankedTensorType>(bodyArgs[stateArgIdx].getType());
    auto accumType = mlir::cast<mlir::RankedTensorType>(bodyArgs[accumArgIdx].getType());
    auto stateShape = stateType.getShape();  // [batch, heads, d_k, d_v]
    auto accumShape = accumType.getShape();  // [batch, heads, seq_len, d_v]

    desc.batch = stateShape[0];
    desc.heads = stateShape[1];
    desc.dK = stateShape[2];
    desc.dV = stateShape[3];
    desc.seqLen = accumShape[2];
    desc.stateIdx = stateArgIdx;
    desc.accumIdx = accumArgIdx;
    desc.iterIdx = iterArgIdx;

    // Now identify which rank-3 arg feeds into Exp (that's the gate)
    // and which rank-3 arg is used as a scaling factor (that's beta)
    // The chain may go through AffineReshape/Reshape before reaching Exp.
    int64_t gateArgIdx = -1, betaArgIdx = -1;
    for (auto idx : rank3Sliced) {
        if (feedsIntoOp<IE::ExpOp>(bodyArgs[idx])) {
            gateArgIdx = idx;
        } else {
            betaArgIdx = idx;
        }
    }

    if (gateArgIdx == -1 || betaArgIdx == -1) {
        log.trace("GDR match failed: could not identify gate (Exp input) and beta among rank-3 args");
        return false;
    }

    // Safety check: if the gate is negated inside the body before Exp, it means the raw
    // gate values at the Loop boundary are POSITIVE. The chunking code uses these raw values
    // directly in CumSum → exp(cumGate), which would overflow for positive values.
    // Standard Mamba2: gate = -softplus(A)*dt (already negative at Loop boundary, no Negative in body) → safe.
    // In QWen3.5 some cases: gate = dt*A (positive at Loop boundary, Negative in body) → unsafe for chunking.
    if (isGateNegatedBeforeExp(bodyArgs[gateArgIdx])) {
        log.trace("GDR match failed: gate is negated inside body (raw gate values are positive). "
                  "Chunked exp(cumGate) would overflow for positive gate values.");
        return false;
    }

    desc.gateIdx = gateArgIdx;
    desc.betaIdx = betaArgIdx;

    // Identify k, q, v among rank-4 sliced args:
    // - k is the one used in Multiply with the state h (Squeeze → Unsqueeze → Multiply(h, k_expanded))
    // - q is the one used in Multiply with h_new (after Add)
    // - v is the one used in Subtract
    //
    // Strategy: trace the state feedback arg's users to find which rank-4 arg multiplies with it.
    // That gives us k. Then trace the Add result's users to find q. The remaining one is v.

    // Find the Multiply that takes the state as input (decay * h or h * k_expanded)
    // There are two Multiplies on state: (1) decay*h, (2) h*k for ReduceSum
    // The one that produces [batch, heads, d_k, d_v] and takes Unsqueeze(Exp(...)) is the decay.
    // The other one that takes an Unsqueeze of a Squeeze of a sliced input is the k·h product.

    int64_t kArgIdx = -1, qArgIdx = -1, vArgIdx = -1;

    // Find the Add op that produces h_new (state update)
    IE::AddOp stateAddOp = nullptr;
    for (auto& op : bodyBlock.getOperations()) {
        if (auto addOp = mlir::dyn_cast<IE::AddOp>(op)) {
            auto outType = mlir::cast<mlir::RankedTensorType>(addOp.getOutput().getType());
            if (outType.getShape() == stateShape) {
                stateAddOp = addOp;
                break;
            }
        }
    }

    if (!stateAddOp) {
        log.trace("GDR match failed: no Add op producing state shape [{0},{1},{2},{3}] found", stateShape[0],
                  stateShape[1], stateShape[2], stateShape[3]);
        return false;
    }

    // h_new is used as: (1) Result (feedback back_edge), (2) Multiply with q → ReduceSum → y_t
    // Find the Multiply on h_new that reduces to output
    for (auto* user : stateAddOp.getOutput().getUsers()) {
        auto mulOp = mlir::dyn_cast<IE::MultiplyOp>(user);
        if (!mulOp) {
            continue;
        }
        // This multiply should produce [batch, heads, d_k, d_v] → then ReduceSum to [batch, heads, 1, d_v]
        // One of its inputs is h_new, the other is derived from q (via reshape chain)
        mlir::Value otherInput = (mulOp.getInput1() == stateAddOp.getOutput()) ? mulOp.getInput2() : mulOp.getInput1();
        // Trace back through reshape-like ops (AffineReshape/Reshape/Squeeze/Unsqueeze) to body arg
        if (auto blockArg = traceToBlockArg(otherInput)) {
            qArgIdx = blockArg.getArgNumber();
        }
    }

    // Find the Subtract op — one of its inputs traces back to v
    for (auto& op : bodyBlock.getOperations()) {
        if (auto subOp = mlir::dyn_cast<IE::SubtractOp>(op)) {
            // v_reshaped - reduceSum_result
            // input1 traces through reshape-like ops to body_arg for v
            if (auto blockArg = traceToBlockArg(subOp.getInput1())) {
                vArgIdx = blockArg.getArgNumber();
            }
            break;
        }
    }

    // k is the remaining rank-4 arg
    for (auto idx : rank4Sliced) {
        if (idx != qArgIdx && idx != vArgIdx) {
            kArgIdx = idx;
            break;
        }
    }

    if (kArgIdx == -1 || qArgIdx == -1 || vArgIdx == -1) {
        log.trace("GDR match failed: could not identify k({0}), q({1}), v({2})", kArgIdx, qArgIdx, vArgIdx);
        return false;
    }

    desc.kIdx = kArgIdx;
    desc.qIdx = qArgIdx;
    desc.vIdx = vArgIdx;

    // Verify ScatterUpdate exists with indices derived from the Loop's current-iteration
    // block argument (possibly via Convert) and writing along the expected sequence axis.
    constexpr int64_t expectedScatterAxis = 2;  // sequence dimension
    bool hasValidScatterUpdate = false;
    for (auto& op : bodyBlock.getOperations()) {
        auto scatterOp = mlir::dyn_cast<IE::ScatterUpdateOp>(op);
        if (!scatterOp) {
            continue;
        }
        // Verify indices trace back to the loop iteration argument.
        auto blockArg = traceScatterIndicesToBlockArg(scatterOp.getIndices());
        if (!blockArg || blockArg.getArgNumber() != static_cast<unsigned>(iterArgIdx)) {
            continue;
        }
        // Verify the scatter axis matches the sequence axis
        auto axisAttr = scatterOp.getAxisValueAttr();
        if (axisAttr) {
            auto axisVal = axisAttr.getValue().getSExtValue();
            if (axisVal != expectedScatterAxis) {
                continue;
            }
        }
        hasValidScatterUpdate = true;
        break;
    }

    if (!hasValidScatterUpdate) {
        log.trace("GDR match failed: no ScatterUpdate with iteration-arg indices on sequence axis");
        return false;
    }

    // Validate that num_iterations matches the accumulator sequence length.
    // The GDR pattern requires the Loop to iterate exactly once per timestep.
    auto numIterations = loopOp.getNumIterations();
    if (numIterations != desc.seqLen) {
        log.trace("GDR match failed: num_iterations ({0}) does not match accumulator seqLen ({1})", numIterations,
                  desc.seqLen);
        return false;
    }

    // Validate the internal execution condition is a constant true (no early termination).
    // GDR chunking assumes a fixed-iteration loop that processes all timesteps.
    auto execCondIndex = loopOp.getExecCondIndex();
    auto loopTerminator = mlir::cast<IE::LoopTerminatorOp>(bodyBlock.getTerminator());
    auto internalExecCond = loopTerminator.getOperands()[execCondIndex];
    auto execCondConst = internalExecCond.getDefiningOp<Const::DeclareOp>();
    if (execCondConst == nullptr) {
        log.trace("GDR match failed: internal execution condition is not a constant (possible early termination)");
        return false;
    }

    // Verify the constant value is true (non-zero). A constant-false exec condition would
    // mean the loop exits immediately, which is incompatible with GDR chunking.
    auto execCondContent = execCondConst.getContentAttr().fold();
    if (execCondContent.isSplat() && execCondContent.getSplatValue<int64_t>() == 0) {
        log.trace("GDR match failed: internal execution condition is constant false");
        return false;
    }

    // Store port mappings
    for (auto sipm : sliceDescs) {
        desc.slicePortMap.push_back({sipm.getExternalPortId().getValue().getSExtValue(),
                                     sipm.getInternalLayerId().getValue().getSExtValue()});
    }

    log.trace("GDR pattern matched: batch={0}, heads={1}, seqLen={2}, dK={3}, dV={4}", desc.batch, desc.heads,
              desc.seqLen, desc.dK, desc.dV);
    log.trace("  k=arg{0}, q=arg{1}, v=arg{2}, gate=arg{3}, beta=arg{4}, state=arg{5}, accum=arg{6}", desc.kIdx,
              desc.qIdx, desc.vIdx, desc.gateIdx, desc.betaIdx, desc.stateIdx, desc.accumIdx);

    return true;
}

//===----------------------------------------------------------------------===//
// GDR Chunked Graph Builder
//===----------------------------------------------------------------------===//

/// Helper: create reshape op.
static mlir::Value createReshape(mlir::OpBuilder& rewriter, mlir::Location loc, mlir::Value input,
                                 ArrayRef<int64_t> newShape) {
    auto shapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    return rewriter.create<IE::ReshapeOp>(loc, input, shapeAttr).getOutput();
}

/// Build the chunked GDR computation that replaces the original Loop.
///
/// Transformation overview:
///   Original: Loop(trip_count=N) over scalar timesteps
///   New:
///     1. Reshape 5 sliced inputs from [B,H,N,D] to [B,H,C,L,D] where C=N/L
///     2. For each chunk (new Loop with trip_count=C):
///        a. Extract chunk slice [B,H,1,L,D] → [B,H,L,D]
///        b. Compute intra-chunk:
///           - decay_mask[L,L] via CumSum of gate values
///           - Q·K^T attention-like matrix [L,L] with decay masking
///           - Triangular solve (forward substitution) for delta correction
///           - corrected_v = attn_matrix @ v_beta → chunk output [B,H,L,D]
///        c. Compute inter-chunk state update:
///           - Apply cumulative decay to h_state from previous chunk
///           - Correct with cross-chunk k·v contributions
///        d. Combine intra and inter outputs
///     3. Reshape output from [B,H,C,L,D] back to [B,H,N,D]
///
static mlir::LogicalResult buildChunkedGDR(IE::LoopOp origLoop, const GDRLoopDescriptor& desc, int64_t chunkSize,
                                           mlir::PatternRewriter& rewriter, Logger log) {
    auto loc = origLoop.getLoc();
    auto* ctx = rewriter.getContext();

    const int64_t N = desc.seqLen;
    const int64_t L = chunkSize;
    const int64_t C = N / L;  // number of chunks
    const int64_t B = desc.batch;
    const int64_t H = desc.heads;
    const int64_t dK = desc.dK;
    const int64_t dV = desc.dV;

    if (N % L != 0) {
        log.warning("Seq length {0} not divisible by chunk size {1}, skipping GDR chunking", N, L);
        return mlir::failure();
    }

    auto opInputs = origLoop.getInputs();
    auto elemType =
            mlir::cast<mlir::RankedTensorType>(origLoop.getBodyModule().front().getArgument(desc.stateIdx).getType())
                    .getElementType();

    // =========================================================================
    // Step 1: Identify external inputs for the 5 sliced sequences
    // Map: body_arg_index → external_input_value
    // =========================================================================
    auto sliceDescs = parseCustomAttrArray<IE::SliceInputPortMapAttr>(origLoop.getSliceInputDescsAttr());
    constexpr int64_t sequenceAxis = 2;
    mlir::DenseMap<int64_t, mlir::Value> bodyArgToExternalInput;
    for (auto sipm : sliceDescs) {
        auto exId = sipm.getExternalPortId().getValue().getSExtValue();
        auto inId = sipm.getInternalLayerId().getValue().getSExtValue();
        bodyArgToExternalInput[inId] = opInputs[exId];
        // Validate that the slice axis matches the expected sequence axis
        auto axis = sipm.getAxis().getValue().getSExtValue();
        if (axis != sequenceAxis) {
            log.warning("Unexpected slice axis {0} for body arg {1}, expected {2}, skipping GDR chunking", axis, inId,
                        sequenceAxis);
            return mlir::failure();
        }
    }

    // External inputs for the 5 sequences (full [B,H,N,...] tensors).
    // Use find() instead of operator[] to detect missing port mappings explicitly.
    auto lookupInput = [&](int64_t bodyArgIdx) -> mlir::Value {
        auto it = bodyArgToExternalInput.find(bodyArgIdx);
        if (it == bodyArgToExternalInput.end()) {
            log.warning("Missing slice port mapping for body arg {0}", bodyArgIdx);
            return nullptr;
        }
        return it->second;
    };

    mlir::Value fullK = lookupInput(desc.kIdx);        // [B,H,N,dK]
    mlir::Value fullQ = lookupInput(desc.qIdx);        // [B,H,N,dK]
    mlir::Value fullV = lookupInput(desc.vIdx);        // [B,H,N,dV]
    mlir::Value fullGate = lookupInput(desc.gateIdx);  // [B,H,N]
    mlir::Value fullBeta = lookupInput(desc.betaIdx);  // [B,H,N]

    if (!fullK || !fullQ || !fullV || !fullGate || !fullBeta) {
        return mlir::failure();
    }

    // Initial state and accumulator from feedback inputs
    auto feedbackDescs = parseCustomAttrArray<IE::MergedInputPortMapAttr>(origLoop.getFeedbackInputDescsAttr());
    mlir::Value initState;  // [B,H,dK,dV]
    mlir::Value initAccum;  // [B,H,N,dV]
    for (auto fipm : feedbackDescs) {
        auto bodyArgIdx = fipm.getInternalLayerId().getValue().getSExtValue();
        auto exId = fipm.getExternalPortId().getValue().getSExtValue();
        if (bodyArgIdx == desc.stateIdx) {
            initState = opInputs[exId];
        } else if (bodyArgIdx == desc.accumIdx) {
            initAccum = opInputs[exId];
        }
    }

    if (!initState) {
        log.trace("buildChunkedGDR: could not find initial state input for stateIdx={0}", desc.stateIdx);
        return mlir::failure();
    }
    if (!initAccum) {
        log.trace("buildChunkedGDR: could not find initial accumulator input for accumIdx={0}", desc.accumIdx);
        return mlir::failure();
    }

    // =========================================================================
    // Step 2: Validate input shapes match expected dimensions, then reshape
    //         inputs into chunks: [B,H,N,D] → [B,H,C,L,D]
    //         For gate and beta: [B,H,N] → [B,H,C,L]
    // =========================================================================
    auto validateShape = [&](mlir::Value val, ArrayRef<int64_t> expectedDims, StringRef name) -> bool {
        auto type = mlir::cast<mlir::RankedTensorType>(val.getType());
        auto shape = type.getShape();
        if (shape.size() != expectedDims.size()) {
            log.warning("{0} rank mismatch: expected {1}, got {2}", name, expectedDims.size(), shape.size());
            return false;
        }
        for (size_t i = 0; i < expectedDims.size(); ++i) {
            if (shape[i] != expectedDims[i]) {
                log.warning("{0} shape mismatch at dim {1}: expected {2}, got {3}", name, i, expectedDims[i], shape[i]);
                return false;
            }
        }
        return true;
    };

    if (!validateShape(fullK, {B, H, N, dK}, "fullK") || !validateShape(fullQ, {B, H, N, dK}, "fullQ") ||
        !validateShape(fullV, {B, H, N, dV}, "fullV") || !validateShape(fullGate, {B, H, N}, "fullGate") ||
        !validateShape(fullBeta, {B, H, N}, "fullBeta")) {
        return mlir::failure();
    }

    auto chunkedK = createReshape(rewriter, appendLoc(loc, "chunk_k"), fullK, {B, H, C, L, dK});
    auto chunkedQ = createReshape(rewriter, appendLoc(loc, "chunk_q"), fullQ, {B, H, C, L, dK});
    auto chunkedV = createReshape(rewriter, appendLoc(loc, "chunk_v"), fullV, {B, H, C, L, dV});
    auto chunkedGate = createReshape(rewriter, appendLoc(loc, "chunk_gate"), fullGate, {B, H, C, L});
    auto chunkedBeta = createReshape(rewriter, appendLoc(loc, "chunk_beta"), fullBeta, {B, H, C, L});

    // =========================================================================
    // Step 3: Build the chunk-level Loop
    //   New Loop: trip_count = C (e.g. 16)
    //   Sliced inputs: chunkedK, chunkedQ, chunkedV, chunkedGate, chunkedBeta (axis=2, part_size=1)
    //   Feedback: h_state [B,H,dK,dV]
    //   Concat output: chunk outputs [B,H,1,L,dV] → concatenated to [B,H,C,L,dV]
    // =========================================================================

    auto outputAccumType = mlir::RankedTensorType::get({B, H, C, L, dV}, elemType);

    // Build new Loop inputs — NO explicit exec_cond input.
    // The UnrollTensorIterator pass expects: inputs.size() == slice + invariant + feedback descriptors.
    // If there's an extra input, it assumes the exec cond is still present and hardcodes opInputs[1].
    // The execution condition is produced internally by the body's LoopTerminator.
    SmallVector<mlir::Value> newLoopInputs = {
            chunkedK,     // ext 0: key chunks [B,H,C,L,dK]
            chunkedQ,     // ext 1: query chunks [B,H,C,L,dK]
            chunkedV,     // ext 2: value chunks [B,H,C,L,dV]
            chunkedGate,  // ext 3: gate chunks [B,H,C,L]
            chunkedBeta,  // ext 4: beta chunks [B,H,C,L]
            initState     // ext 5: initial h state [B,H,dK,dV]
    };

    // Build output types:
    // Result 0: final h_state [B,H,dK,dV] (invariant output, last iteration)
    // Result 1: concatenated chunk outputs [B,H,C,L,dV] (concat output)
    SmallVector<mlir::Type> newLoopOutputTypes = {mlir::RankedTensorType::get({B, H, dK, dV}, elemType),
                                                  outputAccumType};

    // Build descriptor attributes
    // Slice inputs: ext 0..4 → body args 1..5, axis=2, part_size=1, stride=1
    // (body arg 0 is the iteration counter, handled by current_iter_index)
    // Slice inputs: ext 0..4 → body args 1..5, axis=2, part_size=1, stride=1.
    SmallVector<mlir::Attribute> sliceInputDescs;
    for (int64_t i = 0; i < 5; i++) {
        sliceInputDescs.push_back(IE::SliceInputPortMapAttr::get(
                ctx, getIntAttr(ctx, i),  // external_port_id (0..4)
                getIntAttr(ctx, i + 1),   // internal_layer_id (body arg 1..5)
                getIntAttr(ctx, 2),       // axis
                getIntAttr(ctx, 0),       // start
                getIntAttr(ctx, 1),       // stride
                getIntAttr(ctx, 1),       // part_size
                getIntAttr(ctx, C)        // end
                ));
    }

    // Invariant inputs: none
    SmallVector<mlir::Attribute> invariantInputDescs;

    // Feedback inputs: ext 5 → body arg 6 (state)
    // Only h_state needs feedback; output accumulation uses concat_output_descs.
    SmallVector<mlir::Attribute> feedbackInputDescs;
    feedbackInputDescs.push_back(IE::MergedInputPortMapAttr::get(
            ctx, getIntAttr(ctx, 5),  // external_port_id
            getIntAttr(ctx, 6),       // internal_layer_id (body arg 6 = h_state)
            getIntAttr(ctx, 1)        // body_input_index (body output 1 = h_new)
            ));

    // Concat output: each iteration produces [B,H,1,L,dV], concatenated along axis=2
    // to form [B,H,C,L,dV]. This replaces the ScatterUpdate-based accumulation.
    SmallVector<mlir::Attribute> concatOutputDescs;
    concatOutputDescs.push_back(IE::ConcatOutputPortMapAttr::get(
            ctx, getIntAttr(ctx, 1),  // external_port_id (Loop result 1)
            getIntAttr(ctx, 2),       // internal_layer_id (body output 2 = oTotalReshaped)
            getIntAttr(ctx, 2),       // axis = 2 (chunk dimension)
            getIntAttr(ctx, 0),       // start
            getIntAttr(ctx, 1),       // stride
            getIntAttr(ctx, 1),       // part_size
            getIntAttr(ctx, C)        // end
            ));

    // Invariant outputs: only h_state (last iteration)
    // Body terminator outputs: [0]=exec_cond, [1]=h_new, [2]=oTotalReshaped
    SmallVector<mlir::Attribute> invariantOutputDescs;
    invariantOutputDescs.push_back(IE::InvariantOutputPortMapAttr::get(
            ctx, getIntAttr(ctx, 0),  // external_port_id (Loop result 0)
            getIntAttr(ctx, 1),       // internal_layer_id (body output 1 = h_new)
            getIntAttr(ctx, -1)       // iterations: -1 selects the last iteration
            ));

    // Create the new Loop op
    auto newLoop =
            rewriter.create<IE::LoopOp>(appendLoc(loc, "gdr_chunked"), newLoopOutputTypes, newLoopInputs,
                                        /*num_iterations=*/getIntAttr(ctx, C),
                                        /*current_iter_index=*/getIntAttr(ctx, 0),
                                        /*exec_cond_index=*/getIntAttr(ctx, 0),
                                        /*slice_input_descs=*/mlir::ArrayAttr::get(ctx, sliceInputDescs),
                                        /*invariant_input_descs=*/mlir::ArrayAttr::get(ctx, invariantInputDescs),
                                        /*feedback_input_descs=*/mlir::ArrayAttr::get(ctx, feedbackInputDescs),
                                        /*concat_output_descs=*/mlir::ArrayAttr::get(ctx, concatOutputDescs),
                                        /*invariant_output_descs=*/mlir::ArrayAttr::get(ctx, invariantOutputDescs));

    // =========================================================================
    // Step 4: Build the new Loop body
    // =========================================================================
    auto& newBodyRegion = newLoop.getBodyModule();
    auto* newBodyBlock = new mlir::Block();
    newBodyRegion.push_back(newBodyBlock);

    // Body arguments:
    //   [0] current_chunk_iter (scalar)
    //   [1] k_chunk [B,H,1,L,dK]   → Squeeze axis=2 → [B,H,L,dK]
    //   [2] q_chunk [B,H,1,L,dK]
    //   [3] v_chunk [B,H,1,L,dV]
    //   [4] gate_chunk [B,H,1,L]
    //   [5] beta_chunk [B,H,1,L]
    //   [6] h_state [B,H,dK,dV]

    // iteration counter - tensor<1xsi32> to match Loop/UnrollTensorIterator convention
    auto iterType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed));
    newBodyBlock->addArgument(iterType, loc);

    // 5 sliced chunk inputs
    newBodyBlock->addArgument(mlir::RankedTensorType::get({B, H, 1, L, dK}, elemType), loc);  // k_chunk
    newBodyBlock->addArgument(mlir::RankedTensorType::get({B, H, 1, L, dK}, elemType), loc);  // q_chunk
    newBodyBlock->addArgument(mlir::RankedTensorType::get({B, H, 1, L, dV}, elemType), loc);  // v_chunk
    newBodyBlock->addArgument(mlir::RankedTensorType::get({B, H, 1, L}, elemType), loc);      // gate_chunk
    newBodyBlock->addArgument(mlir::RankedTensorType::get({B, H, 1, L}, elemType), loc);      // beta_chunk

    // 1 feedback input (h_state only; output accumulation uses concat_output_descs)
    newBodyBlock->addArgument(mlir::RankedTensorType::get({B, H, dK, dV}, elemType), loc);  // h_state

    auto kChunkSliced = newBodyBlock->getArgument(1);
    auto qChunkSliced = newBodyBlock->getArgument(2);
    auto vChunkSliced = newBodyBlock->getArgument(3);
    auto gateChunkSliced = newBodyBlock->getArgument(4);
    auto betaChunkSliced = newBodyBlock->getArgument(5);
    auto hState = newBodyBlock->getArgument(6);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newBodyBlock);

    // Squeeze the chunk dim (axis=2): [B,H,1,L,D] → [B,H,L,D]
    auto kChunk = createReshape(rewriter, appendLoc(loc, "sq_k"), kChunkSliced, {B, H, L, dK});
    auto qChunk = createReshape(rewriter, appendLoc(loc, "sq_q"), qChunkSliced, {B, H, L, dK});
    auto vChunk = createReshape(rewriter, appendLoc(loc, "sq_v"), vChunkSliced, {B, H, L, dV});
    auto gateChunk = createReshape(rewriter, appendLoc(loc, "sq_gate"), gateChunkSliced, {B, H, L});
    auto betaChunk = createReshape(rewriter, appendLoc(loc, "sq_beta"), betaChunkSliced, {B, H, L});

    // ---- Intra-chunk computation ----

    // 4a. Compute cumulative gate sums for decay mask
    //     cumGateIncl[i] = sum(gate[0..i]) via CumSum on axis=2 (inclusive)
    //     cumGateExcl[i] = sum(gate[0..i-1]) (exclusive: shifted right, padded with 0)
    //
    //     The GDR recurrence h_t = exp(g_t)*h_{t-1} + k_t ⊗ w_t means:
    //       - h_{t-1} decays from chunk start with exp(cumG_{t-1}), NOT exp(cumG_t)
    //       - So the inter-chunk correction k_t^T @ h_{t-1} uses exp(cumG_exclusive[t])
    //       - And the L matrix row index uses cumG_exclusive, NOT cumG_inclusive
    //       - The output y_t = q_t^T @ h_t uses inclusive cumG (h_t already includes g_t)
    auto cumGateType = mlir::RankedTensorType::get({B, H, L}, elemType);
    auto cumGate = rewriter.create<IE::CumSumOp>(appendLoc(loc, "cum_gate"), cumGateType, gateChunk,
                                                 /*axis=*/nullptr, getIntAttr(ctx, 2),
                                                 /*exclusive=*/nullptr, /*reverse=*/nullptr)
                           .getOutput();

    // Exclusive cumsum: [0, cumG[0], cumG[1], ..., cumG[L-2]]
    auto cumGateExcl = rewriter.create<IE::CumSumOp>(appendLoc(loc, "cum_gate_excl"), cumGateType, gateChunk,
                                                     /*axis=*/nullptr, getIntAttr(ctx, 2),
                                                     /*exclusive=*/rewriter.getUnitAttr(), /*reverse=*/nullptr)
                               .getOutput();

    // 4b. Build decay mask D[i,j] = exp(cumGate[i] - cumGate[j]) for i >= j
    //     D = exp(cumGate[:, :, :, None] - cumGate[:, :, None, :])
    //     * masked to lower-triangular

    // Reshape cumGate for broadcasting: [B,H,L,1] and [B,H,1,L]
    auto cumGateRow = createReshape(rewriter, appendLoc(loc, "cg_row"), cumGate, {B, H, L, 1});
    auto cumGateCol = createReshape(rewriter, appendLoc(loc, "cg_col"), cumGate, {B, H, 1, L});

    // diff = cumGateRow - cumGateCol: [B,H,L,L]
    auto diffType = mlir::RankedTensorType::get({B, H, L, L}, elemType);
    auto gateRowBcast = rewriter.create<IE::SubtractOp>(appendLoc(loc, "gate_diff"), cumGateRow, cumGateCol,
                                                        IE::AutoBroadcastType::NUMPY,
                                                        /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                        /*output_padding=*/nullptr, /*input_padding=*/nullptr)
                                .getOutput();

    // decayMask = exp(diff): [B,H,L,L]
    auto decayMask = rewriter.create<IE::ExpOp>(appendLoc(loc, "decay_mask"), diffType, gateRowBcast).getOutput();

    // 4c. Apply lower-triangular mask via Multiply with a constant triangular matrix
    //     tril[i,j] = 1 if i >= j, else 0
    //     Since we need this as a constant, create it
    auto trilType = mlir::RankedTensorType::get({1, 1, L, L}, mlir::Float32Type::get(ctx));
    SmallVector<float> trilValues(L * L, 0.0f);
    for (int64_t i = 0; i < L; i++) {
        for (int64_t j = 0; j <= i; j++) {
            trilValues[i * L + j] = 1.0f;
        }
    }

    auto trilAttr = mlir::DenseElementsAttr::get(trilType, ArrayRef<float>(trilValues));
    auto trilConst =
            rewriter.create<Const::DeclareOp>(appendLoc(loc, "tril_mask"), trilType, Const::ContentAttr::get(trilAttr))
                    .getResult();

    // Convert tril to match element type if needed
    if (elemType != mlir::Float32Type::get(ctx)) {
        auto trilTargetType = mlir::RankedTensorType::get({1, 1, L, L}, elemType);
        trilConst = rewriter.create<IE::ConvertOp>(appendLoc(loc, "tril_convert"), trilTargetType, trilConst, elemType)
                            .getOutput();
    }

    // maskedDecay = decayMask * tril: [B,H,L,L]
    auto maskedDecay = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "masked_decay"), decayMask, trilConst,
                                                       IE::AutoBroadcastType::NUMPY,
                                                       /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                       /*output_padding=*/nullptr, /*input_padding=*/nullptr)
                               .getOutput();

    // 4d. Compute delta correction matrix L
    //     L[i,j] for i > j = -(k_i · k_j) * β_j * exp(cumG_excl[i] - cumG_incl[j])
    //     The row index uses EXCLUSIVE cumsum (h_{t-1} decays with cumG_{t-1}),
    //     while the column index uses INCLUSIVE cumsum.

    // Scale k by beta: k_beta[b,h,l,d] = k[b,h,l,d] * beta[b,h,l]
    auto betaExpanded = createReshape(rewriter, appendLoc(loc, "beta_exp"), betaChunk, {B, H, L, 1});
    auto kBeta = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "k_beta"), kChunk, betaExpanded,
                                                 IE::AutoBroadcastType::NUMPY,
                                                 /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                                 /*input_padding=*/nullptr)
                         .getOutput();

    // kk_T = k @ k_beta^T: [B,H,L,L]
    // IMPORTANT: beta must scale the column (dependency index) in L[j,i], not the row.
    auto kkType = mlir::RankedTensorType::get({B, H, L, L}, elemType);
    auto kkT = rewriter.create<IE::MatMulOp>(appendLoc(loc, "kk_T"), kkType, kChunk, kBeta,
                                             /*transpose_a=*/false, /*transpose_b=*/true)
                       .getOutput();

    // Negate: neg_kkT = -kkT
    auto negKkT = rewriter.create<IE::NegativeOp>(appendLoc(loc, "neg_kkT"), kkType, kkT).getOutput();

    // Build decay for L matrix: exp(cumG_excl[i] - cumG_incl[j]) for strict lower triangular
    // Row index uses EXCLUSIVE cumsum, column index uses INCLUSIVE cumsum
    auto cumGateExclRow = createReshape(rewriter, appendLoc(loc, "cge_row"), cumGateExcl, {B, H, L, 1});
    // cumGateCol (inclusive) already computed above
    auto lDecayDiff = rewriter.create<IE::SubtractOp>(appendLoc(loc, "l_decay_diff"), cumGateExclRow, cumGateCol,
                                                      IE::AutoBroadcastType::NUMPY,
                                                      /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                      /*output_padding=*/nullptr, /*input_padding=*/nullptr)
                              .getOutput();
    auto lDecayMask = rewriter.create<IE::ExpOp>(appendLoc(loc, "l_decay_mask"), diffType, lDecayDiff).getOutput();

    // strict_tril[i,j] = 1 if i > j, else 0
    SmallVector<float> strictTrilValues(L * L, 0.0f);
    for (int64_t i = 1; i < L; i++) {
        for (int64_t j = 0; j < i; j++) {
            strictTrilValues[i * L + j] = 1.0f;
        }
    }

    auto strictTrilAttr = mlir::DenseElementsAttr::get(trilType, ArrayRef<float>(strictTrilValues));
    auto strictTrilConst = rewriter.create<Const::DeclareOp>(appendLoc(loc, "strict_tril"), trilType,
                                                             Const::ContentAttr::get(strictTrilAttr))
                                   .getResult();
    if (elemType != mlir::Float32Type::get(ctx)) {
        auto strictTrilTargetType = mlir::RankedTensorType::get({1, 1, L, L}, elemType);
        strictTrilConst = rewriter.create<IE::ConvertOp>(appendLoc(loc, "strict_tril_convert"), strictTrilTargetType,
                                                         strictTrilConst, elemType)
                                  .getOutput();
    }

    // L_matrix = neg_kkT * lDecayMask * strict_tril
    auto lMatrix =
            rewriter.create<IE::MultiplyOp>(appendLoc(loc, "l_decay"), negKkT, lDecayMask, IE::AutoBroadcastType::NUMPY,
                                            /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                            /*input_padding=*/nullptr)
                    .getOutput();
    lMatrix = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "l_matrix"), lMatrix, strictTrilConst,
                                              IE::AutoBroadcastType::NUMPY,
                                              /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                              /*input_padding=*/nullptr)
                      .getOutput();

    // 4e. Approximate (I - L)^{-1} using Neumann series: I + L + L^2 + L^3
    //     For chunk_size=64, a few iterations suffice due to L being strictly lower triangular
    //     (L^64 = 0 exactly for strict lower triangular L of size 64)
    //     In practice, L + L@L + L@L@L converges fast.

    // Identity matrix [1,1,L,L]
    SmallVector<float> eyeValues(L * L, 0.0f);
    for (int64_t i = 0; i < L; i++) {
        eyeValues[i * L + i] = 1.0f;
    }
    auto eyeAttr = mlir::DenseElementsAttr::get(trilType, ArrayRef<float>(eyeValues));
    auto eyeConst = rewriter.create<Const::DeclareOp>(appendLoc(loc, "eye"), trilType, Const::ContentAttr::get(eyeAttr))
                            .getResult();
    if (elemType != mlir::Float32Type::get(ctx)) {
        auto eyeTargetType = mlir::RankedTensorType::get({1, 1, L, L}, elemType);
        eyeConst = rewriter.create<IE::ConvertOp>(appendLoc(loc, "eye_convert"), eyeTargetType, eyeConst, elemType)
                           .getOutput();
    }

    // Neumann series: (I-L)^{-1} = I + L + L^2 + ... + L^{L-1}
    // Since L is strictly lower triangular of size L, L^L = 0 exactly.
    // Compute iteratively: acc = I, power = L, acc += power, power = power @ L
    // For efficiency, do log2(L) squarings: M = I + L, M = M + M@L^2, etc.
    // But since this is a compile-time graph construction, just do a few MatMul accumulations.
    // A practical approach: repeated squaring of (I + L)
    //   Step 1: A = I + L
    //   Step 2: A = A + A @ L^2  ... this gets complicated
    //
    // Simpler: since L is strictly lower-triangular, the forward substitution loop from
    // HuggingFace is equivalent to computing (I - L)^{-1} exactly.
    // We emit it as: attn = L, then for i in 1..L-1: attn[i,:i] += attn[i,:i] @ attn[:i,:i]
    // But that's hard to express as ops. Instead, use the direct Neumann expansion:
    //   invIminusL = I + L + L@L + L@L@L + ...
    // Since L^L = 0, we need at most L-1 terms to be exact.
    // For L=64, this means 63 MatMuls — too many ops.
    //
    // Better approach: express as (I - L)^{-1} @ v_beta directly via a sequential scan.
    // But that defeats the purpose of chunking.
    //
    // Practical solution: Use the block recurrence.
    //   Split each chunk of L=64 into sub-blocks of size S=8.
    //   Within each sub-block (8x8), the Neumann series converges in 7 terms.
    //   Between sub-blocks, propagate corrections.
    //
    // For v1.0: Use a fixed number of Neumann iterations (since L^L=0 for strict lower tri,
    // the sum I + L + L^2 + ... + L^{L-1} is EXACT, not an approximation).
    // But generating L-1 MatMuls is expensive. Use doubling:
    //   Let A = I + L. Then (I-L)^{-1} = I + L + L^2 + ...
    //   Note: if P_k = sum_{i=0}^{2^k - 1} L^i, then P_{k+1} = P_k + L^{2^k} * P_k = P_k (I + L^{2^k})
    //   And L^{2^{k+1}} = (L^{2^k})^2
    //   So we need only log2(L) = 6 squarings and multiplications for L=64.

    // Compute via doubling: P = I, Lpower = L
    // for k = 0 to ceil(log2(L))-1:
    //   P = P + Lpower @ P    (P = P * (I + Lpower) doesn't work because of non-commutativity)
    //   Lpower = Lpower @ Lpower
    auto P = eyeConst;
    auto Lpower = lMatrix;

    int numDoubling = 0;
    int64_t temp = L;
    while (temp > 1) {
        temp = (temp + 1) / 2;
        numDoubling++;
    }

    for (int k = 0; k < numDoubling; k++) {
        // LP = Lpower @ P
        auto LP = rewriter.create<IE::MatMulOp>(appendLoc(loc, StringRef("neumann_lp_" + std::to_string(k))), kkType,
                                                Lpower, P, false, false)
                          .getOutput();
        // P = P + LP
        P = rewriter.create<IE::AddOp>(appendLoc(loc, StringRef("neumann_p_" + std::to_string(k))), P, LP,
                                       IE::AutoBroadcastType::NUMPY,
                                       /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                       /*input_padding=*/nullptr)
                    .getOutput();
        // Lpower = Lpower @ Lpower
        Lpower = rewriter.create<IE::MatMulOp>(appendLoc(loc, StringRef("neumann_l2_" + std::to_string(k))), kkType,
                                               Lpower, Lpower, false, false)
                         .getOutput();
    }
    // P is now (I - L)^{-1} [B,H,L,L], EXACT for strictly lower triangular L

    // 4f. Compute corrected values w by solving (I - L)w = RHS
    //     RHS_t = v_t - exp(cumG_exclusive[t]) * (k_t^T @ h_state)
    //     Uses EXCLUSIVE cumsum because RHS involves h_{t-1} which decays with cumG_{t-1}.
    //     L has β_j factor, so we solve for w (not β*w). After solving,
    //     compute c = β * w for use in output and state update.
    auto vBetaType = mlir::RankedTensorType::get({B, H, L, dV}, elemType);

    // Inter-chunk correction: hk = K_chunk @ h_state: [B,H,L,dK] @ [B,H,dK,dV] → [B,H,L,dV]
    auto hk = rewriter.create<IE::MatMulOp>(appendLoc(loc, "hk_inter"), vBetaType, kChunk, hState,
                                            /*transpose_a=*/false, /*transpose_b=*/false)
                      .getOutput();

    // Scale by exp(cumG_exclusive) (EXCLUSIVE, not inclusive!): [B,H,L,1]
    // h_{t-1} = exp(cumG_{t-1}) * h_init + ..., so inter-chunk uses cumG_exclusive[t]
    auto cumGateExclExpanded = createReshape(rewriter, appendLoc(loc, "cge_exp"), cumGateExcl, {B, H, L, 1});
    auto exclDecayScalar =
            rewriter.create<IE::ExpOp>(appendLoc(loc, "excl_decay_scalar"),
                                       mlir::RankedTensorType::get({B, H, L, 1}, elemType), cumGateExclExpanded)
                    .getOutput();
    auto interCorrection = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "inter_correction"), exclDecayScalar, hk,
                                                           IE::AutoBroadcastType::NUMPY,
                                                           /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                           /*output_padding=*/nullptr, /*input_padding=*/nullptr)
                                   .getOutput();

    // RHS = v - exp(cumG_exclusive) * K@h  (no beta!)
    auto rhs = rewriter.create<IE::SubtractOp>(appendLoc(loc, "rhs"), vChunk, interCorrection,
                                               IE::AutoBroadcastType::NUMPY,
                                               /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                               /*input_padding=*/nullptr)
                       .getOutput();

    // w = P @ rhs: [B,H,L,L] @ [B,H,L,dV] → [B,H,L,dV]  (solving for w)
    auto w = rewriter.create<IE::MatMulOp>(appendLoc(loc, "w_solved"), vBetaType, P, rhs, false, false).getOutput();

    // corrected_v = β * w: the final c values used for output and state update
    auto correctedV = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "corrected_v"), w, betaExpanded,
                                                      IE::AutoBroadcastType::NUMPY,
                                                      /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                      /*output_padding=*/nullptr, /*input_padding=*/nullptr)
                              .getOutput();

    // 4g. Intra-chunk output: y_intra = (maskedDecay @ corrected_v ⊗ k) · q
    //     More precisely, the intra-chunk h contributions give:
    //     y_intra[i] = sum_j decay[i,j] * corrected_v[j] * k[j] · q[i]
    //     = q[i]^T @ (sum_j decay[i,j] * k[j] ⊗ corrected_v[j])
    //
    //     This can be computed as:
    //     A = maskedDecay @ (k_beta_corrected ⊗ ... ) -- too complex for outer product
    //
    //     Simpler path following HuggingFace:
    //     intra_output = (maskedDecay @ (k * corrected_v^T)) · q  -- but these are tensors
    //
    //     Actually from HuggingFace chunk code:
    //       attn = (q @ k^T) * decay_mask   → [B,H,L,L]
    //       o_intra = attn @ corrected_v     → [B,H,L,dV]
    //
    //     Let's follow that path:

    // QK^T = q @ k^T: [B,H,L,dK] @ [B,H,dK,L] → [B,H,L,L]
    auto qkT = rewriter.create<IE::MatMulOp>(appendLoc(loc, "qk_T"), kkType, qChunk, kChunk, false, true).getOutput();

    // masked_qkT = qkT * maskedDecay: [B,H,L,L]
    auto maskedQkT = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "masked_qkT"), qkT, maskedDecay,
                                                     IE::AutoBroadcastType::NUMPY,
                                                     /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                                     /*input_padding=*/nullptr)
                             .getOutput();

    // o_intra = maskedQkT @ corrected_v: [B,H,L,L] @ [B,H,L,dV] → [B,H,L,dV]
    auto oIntraType = mlir::RankedTensorType::get({B, H, L, dV}, elemType);
    auto oIntra =
            rewriter.create<IE::MatMulOp>(appendLoc(loc, "o_intra"), oIntraType, maskedQkT, correctedV, false, false)
                    .getOutput();

    // 4h. Inter-chunk: contribution from h_state of previous chunk
    //     For each position i in the chunk:
    //       y_inter[i] = q[i]^T @ (decay_from_start[i] * h_state) @ I
    //     decay_from_start[i] = exp(cumGate[i]) (cumulative from chunk start)
    //
    //     h_state_decayed[b,h,dk,dv] = decay[i] * h_state[b,h,dk,dv]
    //     y_inter[b,h,i,dv] = sum_dk q[b,h,i,dk] * h_state_decayed[b,h,dk,dv] * decay_i[b,h,i]

    // decay_from_start = exp(cumGate): [B,H,L]
    auto decayFromStart =
            rewriter.create<IE::ExpOp>(appendLoc(loc, "decay_from_start"), cumGateType, cumGate).getOutput();

    // q_scaled = q * decay_from_start: [B,H,L,dK] * [B,H,L,1] → [B,H,L,dK]
    auto decayStartExpanded = createReshape(rewriter, appendLoc(loc, "dfs_exp"), decayFromStart, {B, H, L, 1});
    auto qScaled = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "q_scaled"), qChunk, decayStartExpanded,
                                                   IE::AutoBroadcastType::NUMPY,
                                                   /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                                   /*input_padding=*/nullptr)
                           .getOutput();

    // o_inter = q_scaled @ h_state: [B,H,L,dK] @ [B,H,dK,dV] → [B,H,L,dV]
    auto oInter = rewriter.create<IE::MatMulOp>(appendLoc(loc, "o_inter"), oIntraType, qScaled, hState, false, false)
                          .getOutput();

    // 4i. Total chunk output: o = o_intra + o_inter
    auto oTotal = rewriter.create<IE::AddOp>(appendLoc(loc, "o_total"), oIntra, oInter, IE::AutoBroadcastType::NUMPY,
                                             /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                             /*input_padding=*/nullptr)
                          .getOutput();

    // 4j. Update h_state for next chunk
    //     h_new = decay_total * h_state + sum_i (k_i ⊗ corrected_v_i * decay_from_end_i)
    //     decay_total = exp(sum of all gate values in chunk) = exp(cumGate[L-1])
    //
    //     The second term: sum of k_i ⊗ corrected_v_i * decay(from i to end of chunk)
    //     = k^T @ diag(decay_to_end) @ corrected_v
    //     where decay_to_end[i] = exp(cumGate[L-1] - cumGate[i])

    // cumGate_last = cumGate[:,:,L-1:L] → scalar-like
    auto cumGateLastType = mlir::RankedTensorType::get({B, H, 1}, elemType);
    auto cumGateLast = rewriter.create<IE::SliceOp>(appendLoc(loc, "cg_last"), cumGateLastType, cumGate,
                                                    getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, L - 1}),
                                                    getIntArrayAttr(ctx, SmallVector<int64_t>{B, H, 1}))
                               .getResult();

    // decay_total = exp(cumGate_last): [B,H,1]
    auto decayTotal =
            rewriter.create<IE::ExpOp>(appendLoc(loc, "decay_total"), cumGateLastType, cumGateLast).getOutput();

    // Expand for broadcasting with h_state [B,H,dK,dV]
    auto decayTotalExp = createReshape(rewriter, appendLoc(loc, "dt_exp"), decayTotal, {B, H, 1, 1});

    // h_decayed = decay_total * h_state
    auto stateType = mlir::RankedTensorType::get({B, H, dK, dV}, elemType);
    auto hDecayed = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "h_decayed"), hState, decayTotalExp,
                                                    IE::AutoBroadcastType::NUMPY,
                                                    /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                                    /*input_padding=*/nullptr)
                            .getOutput();

    // decay_to_end[i] = exp(cumGate[L-1] - cumGate[i])
    // = decay_total / decay_from_start[i] = exp(cumGateLast - cumGate)
    auto cgLastExpanded = createReshape(rewriter, appendLoc(loc, "cgl_exp"), cumGateLast, {B, H, 1});
    auto decayToEndDiff = rewriter.create<IE::SubtractOp>(appendLoc(loc, "dte_diff"), cgLastExpanded, cumGate,
                                                          IE::AutoBroadcastType::NUMPY,
                                                          /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                          /*output_padding=*/nullptr, /*input_padding=*/nullptr)
                                  .getOutput();
    auto decayToEnd =
            rewriter.create<IE::ExpOp>(appendLoc(loc, "decay_to_end"), cumGateType, decayToEndDiff).getOutput();

    // Scale k by decay_to_end: k_decayed[b,h,l,dk] = k[b,h,l,dk] * decay_to_end[b,h,l]
    auto dteExpanded = createReshape(rewriter, appendLoc(loc, "dte_exp4"), decayToEnd, {B, H, L, 1});
    auto kDecayed = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "k_decayed"), kChunk, dteExpanded,
                                                    IE::AutoBroadcastType::NUMPY,
                                                    /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                                    /*input_padding=*/nullptr)
                            .getOutput();

    // state_update = k_decayed^T @ corrected_v: [B,H,dK,L] @ [B,H,L,dV] → [B,H,dK,dV]
    auto stateUpdate = rewriter.create<IE::MatMulOp>(appendLoc(loc, "state_update"), stateType, kDecayed, correctedV,
                                                     /*transpose_a=*/true, /*transpose_b=*/false)
                               .getOutput();

    // h_new = h_decayed + state_update
    auto hNew = rewriter.create<IE::AddOp>(appendLoc(loc, "h_new"), hDecayed, stateUpdate, IE::AutoBroadcastType::NUMPY,
                                           /*post_op=*/nullptr, /*clamp=*/nullptr, /*output_padding=*/nullptr,
                                           /*input_padding=*/nullptr)
                        .getOutput();

    // 4k. Reshape chunk output for concat_output_descs: [B,H,L,dV] → [B,H,1,L,dV]
    //     The Loop's concat_output_descs mechanism concatenates each iteration's
    //     [B,H,1,L,dV] output along axis=2 to form the final [B,H,C,L,dV].
    //     This eliminates the expensive ScatterUpdate that previously copied
    //     the entire 4MB accumulator on every iteration.
    auto oTotalReshaped = createReshape(rewriter, appendLoc(loc, "o_5d"), oTotal, {B, H, 1, L, dV});

    // 4l. Create LoopTerminator with results: [exec_cond, h_new, oTotalReshaped]
    auto bodyExecCond = Const::createConst<int8_t>(rewriter, appendLoc(loc, "body_cond"),
                                                   mlir::RankedTensorType::get({1}, mlir::IntegerType::get(ctx, 8)),
                                                   ArrayRef<int8_t>{1});

    rewriter.create<IE::LoopTerminatorOp>(appendLoc(loc, "chunk_term"),
                                          SmallVector<mlir::Value>{bodyExecCond, hNew, oTotalReshaped});

    // =========================================================================
    // Step 5: Reshape output and replace original Loop
    // =========================================================================

    // Restore insertion point to after the new Loop (outside the body block)
    rewriter.setInsertionPointAfter(newLoop);

    // newLoop result 0: final h_state [B,H,dK,dV]
    // newLoop result 1: output_accum [B,H,C,L,dV]

    auto finalState = newLoop.getResult(0);
    auto chunkedOutput = newLoop.getResult(1);

    // Reshape output: [B,H,C,L,dV] → [B,H,N,dV]
    auto finalOutput = createReshape(rewriter, appendLoc(loc, "unchunk_output"), chunkedOutput, {B, H, N, dV});

    // Map original Loop results to new results using invariant output descriptors.
    // Each invariant output descriptor maps: external_port_id (Loop result index) →
    // internal_layer_id (body terminator output index).
    // In the original body, the feedback descriptors tell which body output feeds back
    // to which body arg. The state feedback body output maps to finalState, and
    // the accumulator feedback body output maps to finalOutput.
    auto origInvariantDescs =
            parseCustomAttrArray<IE::InvariantOutputPortMapAttr>(origLoop.getInvariantOutputDescsAttr());
    auto origFeedbackDescs = parseCustomAttrArray<IE::MergedInputPortMapAttr>(origLoop.getFeedbackInputDescsAttr());
    auto origConcatDescs = parseCustomAttrArray<IE::ConcatOutputPortMapAttr>(origLoop.getConcatOutputDescsAttr());

    // The GDR pattern uses only invariant outputs. If concat outputs exist,
    // those results cannot be mapped and would leave null entries in replacements.
    // Erase the newly created ops to avoid leaving partially built IR on failure.
    if (!origConcatDescs.empty()) {
        log.warning("Original Loop has {0} concat output descriptors, which are not supported by GDR chunking",
                    origConcatDescs.size());
        rewriter.eraseOp(newLoop);
        return mlir::failure();
    }

    // Build mapping: body_output_index → which descriptor role (state or accum)
    mlir::DenseMap<int64_t, mlir::Value> bodyOutputToReplacement;
    for (auto fipm : origFeedbackDescs) {
        auto bodyArgIdx = fipm.getInternalLayerId().getValue().getSExtValue();
        auto bodyOutputIdx = fipm.getBodyInputIndex().getValue().getSExtValue();
        if (bodyArgIdx == desc.stateIdx) {
            bodyOutputToReplacement[bodyOutputIdx] = finalState;
        } else if (bodyArgIdx == desc.accumIdx) {
            bodyOutputToReplacement[bodyOutputIdx] = finalOutput;
        }
    }

    SmallVector<mlir::Value> replacements;
    auto origResults = origLoop.getResults();
    replacements.resize(origResults.size());

    for (auto invDesc : origInvariantDescs) {
        auto extPortId = invDesc.getExternalPortId().getValue().getSExtValue();
        auto intLayerId = invDesc.getInternalLayerId().getValue().getSExtValue();
        auto it = bodyOutputToReplacement.find(intLayerId);
        if (it != bodyOutputToReplacement.end()) {
            replacements[extPortId] = it->second;
        } else {
            log.warning("Unmapped invariant output: external_port={0}, internal_layer={1}", extPortId, intLayerId);
            rewriter.eraseOp(newLoop);
            return mlir::failure();
        }
    }

    rewriter.replaceOp(origLoop, replacements);

    log.trace("GDR chunking SUCCESS: Loop(trip_count={0}) -> Loop(trip_count={1}), chunk_size={2}", N, C, L);
    log.trace("  Dimensions: batch={0}, heads={1}, d_k={2}, d_v={3}", B, H, dK, dV);

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// GDR Chunked Loop Rewriter Pattern
//===----------------------------------------------------------------------===//

class GDRChunkedLoopRewriter final : public mlir::OpRewritePattern<IE::LoopOp> {
public:
    GDRChunkedLoopRewriter(mlir::MLIRContext* ctx, int64_t chunkSize, Logger log)
            : mlir::OpRewritePattern<IE::LoopOp>(ctx), _chunkSize(chunkSize), _log(log) {
        setDebugName("GDRChunkedLoopRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LoopOp loopOp, mlir::PatternRewriter& rewriter) const final {
        if (_chunkSize <= 0) {
            _log.trace("Invalid chunk_size={0}, must be > 0", _chunkSize);
            return mlir::failure();
        }

        GDRLoopDescriptor desc;
        if (!matchGDRPattern(loopOp, desc, _log)) {
            return mlir::failure();
        }

        // Verify chunk size divides sequence length
        if (desc.seqLen % _chunkSize != 0) {
            _log.trace("GDR Loop seq_len={0} not divisible by chunk_size={1}", desc.seqLen, _chunkSize);
            return mlir::failure();
        }

        _log.trace("Applying GDR chunking: Loop({0}) → Loop({1}) with chunk_size={2}", desc.seqLen,
                   desc.seqLen / _chunkSize, _chunkSize);

        return buildChunkedGDR(loopOp, desc, _chunkSize, rewriter, _log);
    }

private:
    int64_t _chunkSize;
    Logger _log;
};

//===----------------------------------------------------------------------===//
// ChunkGatedDeltaRuleLoop Pass
//===----------------------------------------------------------------------===//

class ChunkGatedDeltaRuleLoop final : public IE::impl::ChunkGatedDeltaRuleLoopBase<ChunkGatedDeltaRuleLoop> {
public:
    explicit ChunkGatedDeltaRuleLoop(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ChunkGatedDeltaRuleLoop::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    _log.trace("ChunkGatedDeltaRuleLoop pass started with chunk_size={0}", chunkSize);

    if (chunkSize <= 0) {
        _log.warning("Invalid chunk_size={0}, must be > 0. Skipping pass.", chunkSize);
        return;
    }

    if (_log.isActive(LogLevel::Trace)) {
        int64_t loopCount = 0;
        func->walk([&](IE::LoopOp) {
            ++loopCount;
        });
        _log.trace("Found {0} Loop op(s) in function", loopCount);
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GDRChunkedLoopRewriter>(&ctx, chunkSize, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createChunkGatedDeltaRuleLoopPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createChunkGatedDeltaRuleLoopPass(Logger log) {
    return std::make_unique<ChunkGatedDeltaRuleLoop>(log);
}
