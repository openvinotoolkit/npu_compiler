//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Fuses an IE::TileOp that broadcasts the innermost memory dimension into the
// consuming IE::AddOp.  The explicit Tile is replaced by NUMPY auto-broadcast
// on the AddOp itself, which is then lowered to a SHAVE broadcast-add kernel.
// This avoids a TileDMA pass over every element of the innermost dimension,
// which is the worst-case access pattern for DMA.
//
// Matched pattern:
//
//   %tile = IE.Tile(%in) {repeats_values = [..., N]}   (N > 1 on innermost mem dim)
//   %out  = IE.Add(%tile, %other) {auto_broadcast = NUMPY}
//
// After fusion:
//
//   %out  = IE.Add(%in, %other) {auto_broadcast = NUMPY}

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEINEFFICIENTTILEFORADD
#define GEN_PASS_DEF_FUSEINEFFICIENTTILEFORADD
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseInefficientTileRewriter
//
// Matches: TileOp -> AddOp (NUMPY)
// Condition: the TileOp repeats along the innermost memory dimension of its output.
// Action: replace the Tile-derived AddOp operand with the un-tiled input, letting
//         NUMPY broadcast on AddOp handle the broadcast implicitly via SHAVE.
//

class FuseInefficientTileRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FuseInefficientTileRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("FuseInefficientTileRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    // Returns true when the TileOp repeats along the innermost memory dimension of
    // its output (i.e. the broadcast is on the fastest-varying memory axis).
    bool isInnermostMemDimBroadcast(IE::TileOp tileOp) const;

    // Returns true when each dimension of inShape is either equal to the
    // corresponding outShape dimension or is 1 (pure broadcast shape).
    bool isBroadcastShape(ShapeRef inShape, ShapeRef outShape) const;

    Logger _log;
};

bool FuseInefficientTileRewriter::isBroadcastShape(ShapeRef inShape, ShapeRef outShape) const {
    if (inShape.size() != outShape.size()) {
        return false;
    }
    for (size_t i = 0; i < inShape.size(); ++i) {
        const auto inDim = inShape[Dim(i)];
        const auto outDim = outShape[Dim(i)];
        if (inDim != 1 && inDim != outDim) {
            return false;
        }
    }
    return true;
}

bool FuseInefficientTileRewriter::isInnermostMemDimBroadcast(IE::TileOp tileOp) const {
    if (!tileOp.getRepeatsValues().has_value()) {
        // Dynamic repeats — cannot statically determine the broadcast dimension.
        return false;
    }

    // Dynamic shape dims make totalSize() and getInnermostNonTrivialDim() unsafe.
    const auto inputType = mlir::cast<mlir::ShapedType>(tileOp.getInput().getType());
    const auto outputType = mlir::cast<mlir::ShapedType>(tileOp.getOutput().getType());
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
        return false;
    }

    const auto inShape = getShape(tileOp.getInput());
    const auto outShape = getShape(tileOp.getOutput());

    if (!isBroadcastShape(inShape, outShape)) {
        return false;
    }

    const auto outputOrder = mlir::cast<NDTypeInterface>(tileOp.getOutput().getType()).getDimsOrder();

    // Splat input (all dims == 1): scalar-to-many broadcast is handled efficiently
    // by DMA and does not need to be fused.
    if (inShape.totalSize() == 1) {
        return false;
    }

    // The Tile is inefficient when the innermost non-trivial output dim is a repeated
    // axis (size 1 in the input): TileDMA must write every output element along the
    // fastest-varying relevant memory axis from a single source element.
    const auto innermostNonTrivialOut = getInnermostNonTrivialDim(outShape, outputOrder);
    return innermostNonTrivialOut.has_value() && inShape[innermostNonTrivialOut.value()] == 1;
}

mlir::LogicalResult FuseInefficientTileRewriter::matchAndRewrite(IE::AddOp addOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), addOp->getName(), addOp->getLoc());

    if (addOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        _log.trace("auto_broadcast is not NUMPY — skipping");
        return mlir::failure();
    }

    if (addOp.getOutputPaddingAttr() || addOp.getInputPaddingAttr()) {
        return mlir::failure();
    }

    // Locate the broadcast Tile directly on either input: TileOp -> AddOp.
    // A PermuteCast between Tile and Add is not handled: layout propagation
    // would change which dimension is innermost, making the optimisation
    // rationale inconsistent.
    IE::TileOp tileOp;

    for (auto* defOp : {addOp.getInput1().getDefiningOp(), addOp.getInput2().getDefiningOp()}) {
        auto tileCandidate = mlir::dyn_cast_if_present<IE::TileOp>(defOp);
        if (!tileCandidate || !tileCandidate->hasOneUse()) {
            continue;
        }
        if (!isInnermostMemDimBroadcast(tileCandidate)) {
            continue;
        }
        tileOp = tileCandidate;
        break;
    }

    if (!tileOp) {
        _log.trace("No inefficient broadcast Tile found");
        return mlir::failure();
    }

    // The un-tiled input must form a valid NUMPY broadcast pair with the other AddOp
    // input: for every dim where the fused (small) input has size 1, the other input
    // must fully cover that dim.
    const auto tileInput = tileOp.getInput();
    const auto tileInShape = getShape(tileInput);

    const auto isTileOnInput1 = addOp.getInput1() == tileOp.getOutput();
    const auto otherInput = isTileOnInput1 ? addOp.getInput2() : addOp.getInput1();
    const auto otherInputShape = getShape(otherInput);
    const auto outputShape = getShape(addOp.getOutput());
    const auto rank = outputShape.size();

    if (tileInShape.size() != rank || otherInputShape.size() != rank) {
        return mlir::failure();
    }

    // Safety check: NUMPY broadcast is only correct when, for each dim where the
    // fused input has size 1, the other input covers that dim entirely.
    for (size_t i = 0; i < rank; ++i) {
        if (tileInShape[Dim(i)] == 1 && otherInputShape[Dim(i)] != outputShape[Dim(i)]) {
            _log.trace("NUMPY broadcast safety check failed at dim {0}", i);
            return mlir::failure();
        }
    }

    const auto addOutputType = mlir::cast<vpux::NDTypeInterface>(addOp.getOutput().getType());
    const auto newInput1 = isTileOnInput1 ? tileInput : addOp.getInput1();
    const auto newInput2 = isTileOnInput1 ? addOp.getInput2() : tileInput;

    const auto addLoc = addOp->getLoc();

    auto newAddOp = rewriter.create<IE::AddOp>(
            takeOpLoc(addOp, "fuse_broadcast"), addOutputType, newInput1, newInput2, addOp.getAutoBroadcastAttr(),
            addOp.getPostOpAttr(), addOp.getClampAttr(), addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());
    rewriter.replaceOp(addOp, newAddOp.getOutput());
    rewriter.eraseOp(tileOp);

    _log.trace("Fused inefficient Tile broadcast into Add at '{0}'", addLoc);
    return mlir::success();
}

//
// FuseInefficientTileForAddPass
//

class FuseInefficientTileForAddPass final :
        public IE::impl::FuseInefficientTileForAddBase<FuseInefficientTileForAddPass> {
public:
    explicit FuseInefficientTileForAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseInefficientTileForAddPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseInefficientTileRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseInefficientTileForAddPass(Logger log) {
    return std::make_unique<FuseInefficientTileForAddPass>(log);
}
