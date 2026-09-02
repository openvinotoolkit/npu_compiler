//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGATHER
#define GEN_PASS_DEF_CONVERTGATHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertGatherPass
//

class ConvertGatherPass final : public IE::impl::ConvertGatherBase<ConvertGatherPass> {
public:
    explicit ConvertGatherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherToSlice;
    class GatherToReverse;
    class GatherRepeatInterleaveToBroadcast;
    class FuseBroadcastGatherND;

private:
    void safeRunOnFunc() final;
};

bool checkAttrsForGatherOp(IE::GatherOp gatherOp) {
    const auto batchDims = gatherOp.getBatchDims();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getInput().getType());
    if (auto boundedInputTensor = mlir::dyn_cast<Core::BoundedTensorType>(inType)) {
        return false;
    }

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    if (indices == nullptr) {
        return false;
    }

    return batchDims == 0;
}

//
// GatherToSlice
//

class ConvertGatherPass::GatherToSlice final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherToSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherPass::GatherToSlice");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherPass::GatherToSlice::matchAndRewrite(IE::GatherOp gatherOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Gather Op: {0}", gatherOp);
    auto* ctx = rewriter.getContext();

    if (!checkAttrsForGatherOp(gatherOp)) {
        return mlir::failure();
    }

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.getContent();
    if (indicesContent.getType().getNumElements() != 1) {
        return mlir::failure();
    }

    const auto indicesVal = indicesContent.getSplatValue<int64_t>();

    const auto axisVal = gatherOp.getAxisValue();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(gatherOp.getInput().getType());
    const auto inputShape = inType.getShape();
    auto staticOffsets = SmallVector<int64_t>(inputShape.size(), 0);
    staticOffsets[axisVal] = indicesVal;

    SmallVector<int64_t> staticSizes(inputShape.begin(), inputShape.end());
    staticSizes[axisVal] = 1;

    const auto sliceOpLoc = appendLoc(gatherOp.getLoc(), "slice");
    auto sliceOp = rewriter.create<IE::SliceOp>(sliceOpLoc, gatherOp.getInput(), getIntArrayAttr(ctx, staticOffsets),
                                                getIntArrayAttr(ctx, staticSizes));

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(gatherOp, sliceOp.getResult(),
                                               getIntArrayAttr(ctx, getShape(gatherOp.getOutput())));

    return mlir::success();
}

//
// GatherToReverse
//

class ConvertGatherPass::GatherToReverse final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherToReverse(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherPass::GatherToReverse");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherPass::GatherToReverse::matchAndRewrite(IE::GatherOp gatherOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), gatherOp->getName(), gatherOp->getLoc());
    if (!checkAttrsForGatherOp(gatherOp)) {
        return mlir::failure();
    }

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.getContent();
    const auto indicesNums = indicesContent.getType().getNumElements();

    if (indicesNums == 1) {
        _log.trace("[{0}] Only one index value, converting to Slice", this->getDebugName());
        return mlir::failure();
    }

    const auto axisVal = gatherOp.getAxisValue();
    const auto inputShape = getShape(gatherOp.getInput());

    // For GatherDMA all dimensions before axis dimension must be 1
    if (std::all_of(inputShape.begin(), inputShape.begin() + axisVal, [](int64_t dim) {
            return dim == 1;
        })) {
        _log.trace("[{0}] All dimensions before axis dimension are 1, converting to GatherDMA", this->getDebugName());
        return mlir::failure();
    }

    if (inputShape[Dim(axisVal)] != indicesNums || inputShape != getShape(gatherOp.getOutput())) {
        _log.trace("[{0}] Input shape does not match indices number or output shape, cannot convert to Reverse",
                   this->getDebugName());
        return mlir::failure();
    }

    auto areIndicesReverseContiguous = [](const SmallVector<int64_t>& indicesValues) -> bool {
        auto it = std::adjacent_find(indicesValues.begin(), indicesValues.end(), [](int64_t prev, int64_t curr) {
            return curr != prev - 1;
        });
        return it == indicesValues.end();
    };

    const auto vals = to_small_vector(indicesContent.getValues<int64_t>());
    if (vals.empty() || !areIndicesReverseContiguous(vals)) {
        _log.trace("[{0}] Indices are not reverse contiguous, cannot convert to Reverse", this->getDebugName());
        return mlir::failure();
    }

    const auto ctx = gatherOp.getContext();
    const auto axisAttr = getIntArrayAttr(ctx, ArrayRef(axisVal));
    const auto modeAttr = IE::ReverseModeAttr::get(ctx, IE::ReverseMode::INDEX);
    auto reverseOp =
            rewriter.create<IE::ReverseOp>(gatherOp.getLoc(), gatherOp.getInput(), nullptr, axisAttr, modeAttr);

    _log.trace("[{0}] Replaced with Reverse '{1}' at '{2}'", this->getDebugName(), gatherOp->getName(),
               gatherOp->getLoc());
    rewriter.replaceOp(gatherOp, reverseOp.getResult());
    return mlir::success();
}

//
// GatherRepeatInterleaveToBroadcast
//
// Converts a Gather that implements repeat_interleave(n) into Reshape + Broadcast + Reshape.
// Pattern: indices = [0, 0, 1, 1, 2, 2, ...] (each value repeated n times).
//
// Before:
//   Gather(input[..., D, ...], indices=[D*n], axis=A) -> [..., D*n, ...]
//
// After:
//   Reshape(input, [..., D, 1, ...]) -> Broadcast([..., D, n, ...]) -> Reshape([..., D*n, ...])
//

class ConvertGatherPass::GatherRepeatInterleaveToBroadcast final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherRepeatInterleaveToBroadcast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherPass::GatherRepeatInterleaveToBroadcast");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherPass::GatherRepeatInterleaveToBroadcast::matchAndRewrite(
        IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), gatherOp->getName(), gatherOp->getLoc());

    if (!checkAttrsForGatherOp(gatherOp)) {
        return mlir::failure();
    }

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.getContent();
    const auto indicesNums = indicesContent.getType().getNumElements();

    if (indicesNums <= 1) {
        return mlir::failure();
    }

    const auto axisVal = gatherOp.getAxisValue();
    const auto inputShape = getShape(gatherOp.getInput());
    const auto axisDim = inputShape[Dim(axisVal)];

    if (axisDim == 0 || indicesNums % axisDim != 0) {
        return mlir::failure();
    }

    const auto repeatFactor = indicesNums / axisDim;
    if (repeatFactor <= 1) {
        return mlir::failure();
    }

    // Verify indices follow repeat_interleave(n) pattern: indices[i] = i / repeatFactor
    const auto vals = to_small_vector(indicesContent.getValues<int64_t>());
    for (int64_t i = 0; i < indicesNums; ++i) {
        if (vals[i] != i / repeatFactor) {
            return mlir::failure();
        }
    }

    _log.trace("[{0}] Detected repeat_interleave({1}) on axis {2}, dim {3} -> {4}", this->getDebugName(), repeatFactor,
               axisVal, axisDim, indicesNums);

    auto* ctx = rewriter.getContext();
    const auto loc = gatherOp.getLoc();

    SmallVector<int64_t> reshapeShape1(inputShape.begin(), inputShape.end());
    reshapeShape1.insert(reshapeShape1.begin() + axisVal + 1, 1);

    auto reshape1 = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "unsqueeze"), gatherOp.getInput(),
                                                   getIntArrayAttr(ctx, reshapeShape1));

    SmallVector<int64_t> broadcastShape(std::move(reshapeShape1));
    broadcastShape[axisVal + 1] = repeatFactor;

    auto broadcastResult =
            IE::createBroadcast(rewriter, appendLoc(loc, "repeat"), reshape1.getOutput(), ShapeRef(broadcastShape));

    const auto outputShape = to_small_vector(getShape(gatherOp.getOutput()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(gatherOp, broadcastResult, getIntArrayAttr(ctx, outputShape));

    return mlir::success();
}

//
// FuseBroadcastGatherND
//
// Folds away a GatherND whose input is a Broadcast (NUMPY or BIDIRECTIONAL mode) when all of the
// first `lastDim` left-padded broadcast input dimensions are size 1 and at least one of them is a
// broadcasting axis (i.e. expands to > 1 in the output). Under these conditions every index tuple
// selects identical data, so the GatherND is redundant.
//
// Before:
//   Broadcast(input[...] -> [1 x N x ...rest...]) -> GatherND(indices shape=[..., lastDim]) -> out
//
// After:
//   Reshape(input, [...rest...]) -> Broadcast(-> out_shape)
//

class ConvertGatherPass::FuseBroadcastGatherND final : public mlir::OpRewritePattern<IE::GatherNDOp> {
public:
    FuseBroadcastGatherND(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherNDOp>(ctx), _log(log) {
        setDebugName("ConvertGatherPass::FuseBroadcastGatherND");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherNDOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherPass::FuseBroadcastGatherND::matchAndRewrite(IE::GatherNDOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto broadcastOp = origOp.getInput().getDefiningOp<IE::BroadcastOp>();
    if (broadcastOp == nullptr) {
        return mlir::failure();
    }

    const auto outType = mlir::cast<mlir::ShapedType>(origOp.getOutput().getType());
    if (!outType.hasStaticShape()) {
        return mlir::failure();
    }

    // EXPLICIT mode uses an axis mapping that may reorder dimensions.
    const auto broadcastMode = broadcastOp.getMode().value_or(IE::BroadcastType::NUMPY);
    if (broadcastMode == IE::BroadcastType::EXPLICIT) {
        return mlir::failure();
    }

    const int64_t batchDims = origOp.getBatchDims();
    if (batchDims != 0) {
        return mlir::failure();
    }

    const auto indicesType = mlir::cast<mlir::ShapedType>(origOp.getIndices().getType());
    const auto indicesShape = indicesType.getShape();
    if (indicesShape.empty()) {
        return mlir::failure();
    }

    const int64_t lastDim = indicesShape.back();
    const auto broadcastInputShape = to_small_vector(getShape(broadcastOp.getInput()));
    const auto broadcastOutputShape = to_small_vector(getShape(broadcastOp.getOutput()));

    if (broadcastInputShape.size() > broadcastOutputShape.size()) {
        return mlir::failure();
    }

    if (static_cast<int64_t>(broadcastOutputShape.size()) < lastDim) {
        return mlir::failure();
    }

    const int64_t rankDiff =
            static_cast<int64_t>(broadcastOutputShape.size()) - static_cast<int64_t>(broadcastInputShape.size());
    SmallVector<int64_t> alignedInputShape(rankDiff, 1);
    alignedInputShape.append(broadcastInputShape.begin(), broadcastInputShape.end());

    bool hasBroadcastAxis = false;
    for (int64_t d = 0; d < lastDim; ++d) {
        if (alignedInputShape[d] != 1) {
            return mlir::failure();
        }
        if (broadcastOutputShape[d] > 1) {
            hasBroadcastAxis = true;
        }
    }
    if (!hasBroadcastAxis) {
        return mlir::failure();
    }

    SmallVector<int64_t> reshapeShape(alignedInputShape.begin() + lastDim, alignedInputShape.end());

    const auto ctx = rewriter.getContext();
    const auto loc = origOp.getLoc();

    auto reshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "reshape"), broadcastOp.getInput(),
                                                    getIntArrayAttr(ctx, reshapeShape));

    const auto outShape = to_small_vector(outType.getShape());
    auto newBroadcast =
            IE::createBroadcast(rewriter, appendLoc(loc, "broadcast"), reshapeOp.getOutput(), ShapeRef(outShape));

    rewriter.replaceOp(origOp, newBroadcast);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GatherToSlice>(&ctx, _log);
    patterns.add<GatherToReverse>(&ctx, _log);
    patterns.add<GatherRepeatInterleaveToBroadcast>(&ctx, _log);
    patterns.add<FuseBroadcastGatherND>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGatherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGatherPass(Logger log) {
    return std::make_unique<ConvertGatherPass>(log);
}
