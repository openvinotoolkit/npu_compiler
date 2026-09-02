//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPELTWISEANDREDUCE
#define GEN_PASS_DEF_SWAPELTWISEANDREDUCE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <typename ReduceOp>
ReduceOp buildReduce(mlir::PatternRewriter& rewriter, ReduceOp origOp, mlir::Value input) {
    mlir::IRMapping mapper;
    mapper.map(origOp.getInput(), input);
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->setLoc(takeOpLoc(origOp, "reduce_swap"));
    mlir::cast<ReduceOp>(newOp).setKeepDims(true);
    inferReturnTypes(newOp, InferShapedTypeMode::ALL);
    return mlir::cast<ReduceOp>(newOp);
}

template <typename EltwiseOp>
EltwiseOp buildEltwise(mlir::PatternRewriter& rewriter, EltwiseOp origOp, mlir::Value lhs, mlir::Value rhs) {
    const auto origEltwiseType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector{lhs, rhs});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->setLoc(takeOpLoc(origOp, "eltwise_swap"));
    inferReturnTypes(newOp, InferShapedTypeMode::ALL);
    auto newResult = newOp->getResult(0);
    newResult.setType(mlir::cast<vpux::NDTypeInterface>(newResult.getType()).changeElemType(origEltwiseType));
    return mlir::cast<EltwiseOp>(newOp);
}

//
// SwapEltwiseAndReduce<EltwiseOp, ReduceOp>
//
// Applies the distributive law to hoist a ReduceOp over an EltwiseOp when one
// EltwiseOp operand is broadcast (size==1) on every reduce axis:
//
//   keep_dims=true:
//     EltwiseOp(A, B) -> ReduceOp(axis, keep_dims=true)
//     =>
//     ReduceOp(A, axis, keep_dims=true) -> EltwiseOp(result, B)
//
//   keep_dims=false (and no output_padding on the ReduceOp):
//     EltwiseOp(A, B) -> ReduceOp(axis, keep_dims=false)
//     =>
//     ReduceOp(A, axis, keep_dims=true) -> EltwiseOp(result, B) -> Reshape
//     (output_padding present is always skipped: the padding changes output shape
//      and is incompatible with the rank-preserved intermediate ReduceOp output)
//
// The intermediate ReduceOp always uses keep_dims=true so that rank is
// preserved and the broadcast input can be applied correctly. A final Reshape
// squeezes the size-1 reduce axes when the original keep_dims was false.
//
// This eliminates the large intermediate tensor produced by the full EltwiseOp
// and allows subsequent passes to operate on the smaller post-reduce tensors.
//

template <typename EltwiseOp, typename ReduceOp>
class SwapEltwiseAndReduce final : public mlir::OpRewritePattern<ReduceOp> {
public:
    SwapEltwiseAndReduce(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ReduceOp>(ctx), _log(log) {
        this->setDebugName("SwapEltwiseAndReduce");
    }

    mlir::LogicalResult matchAndRewrite(ReduceOp reduceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename EltwiseOp, typename ReduceOp>
mlir::LogicalResult SwapEltwiseAndReduce<EltwiseOp, ReduceOp>::matchAndRewrite(ReduceOp reduceOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), reduceOp->getName(), reduceOp->getLoc());

    auto eltwiseOp = reduceOp.getInput().template getDefiningOp<EltwiseOp>();
    if (eltwiseOp == nullptr) {
        return matchFailed(_log, rewriter, reduceOp, "Input is not the expected EltwiseOp");
    }

    if (!eltwiseOp->hasOneUse()) {
        return matchFailed(_log, rewriter, reduceOp, "EltwiseOp has more than one use");
    }

    if (eltwiseOp.getPostOpAttr() != nullptr || eltwiseOp.getClampAttr() != nullptr ||
        eltwiseOp.getInputPaddingAttr() != nullptr || eltwiseOp.getOutputPaddingAttr() != nullptr) {
        return matchFailed(_log, rewriter, reduceOp, "EltwiseOp has post_op, clamp, or padding attrs");
    }

    if (reduceOp.getInputPaddingAttr() != nullptr || reduceOp.getOutputPaddingAttr() != nullptr) {
        return matchFailed(_log, rewriter, reduceOp, "ReduceOp has padding attrs; cannot safely hoist");
    }

    auto axes = parseIntArrayAttr<int64_t>(reduceOp.getAxesValue());
    auto input1 = eltwiseOp.getInput1();
    auto input2 = eltwiseOp.getInput2();
    auto shape1 = getShape(input1);
    auto shape2 = getShape(input2);

    if (shape1.size() != shape2.size()) {
        return matchFailed(_log, rewriter, reduceOp, "Inputs have different ranks");
    }

    auto isBroadcastOnAxes = [&](ShapeRef shape) {
        return llvm::all_of(axes, [&](int64_t axis) {
            return shape[Dim(axis)] == 1;
        });
    };

    const bool isInput2Broadcast = isBroadcastOnAxes(shape2);
    const bool isInput1Broadcast = isBroadcastOnAxes(shape1);

    if (!isInput2Broadcast && !isInput1Broadcast) {
        return matchFailed(_log, rewriter, reduceOp, "Neither input is broadcast on all reduce axes");
    }

    if (isInput2Broadcast && isInput1Broadcast) {
        // Both inputs are size==1 on every reduce axis, so the EltwiseOp output is also
        // size==1 on those axes. The ReduceOp compresses nothing meaningful; swapping
        // produces no memory savings and no benefit.
        return matchFailed(_log, rewriter, reduceOp,
                           "Both inputs are broadcast on all reduce axes; no benefit to swap");
    }

    // The intermediate ReduceOp always uses keep_dims=true so that rank is preserved and the subsequent EltwiseOp
    // can broadcast correctly against the broadcast input (which has size==1 on every reduce axis).
    auto newReduce = buildReduce(rewriter, reduceOp, isInput2Broadcast ? input1 : input2);
    auto newEltwise = isInput2Broadcast ? buildEltwise(rewriter, eltwiseOp, newReduce.getOutput(), input2)
                                        : buildEltwise(rewriter, eltwiseOp, input1, newReduce.getOutput());

    if (reduceOp.getKeepDims()) {
        rewriter.replaceOp(reduceOp, newEltwise.getOutput());
    } else {
        // keep_dims=false: the reduce axes have size==1 in newEltwise's output.
        // Use Reshape to squeeze them out instead of a second reduce.
        const auto origOutShape = getShape(reduceOp.getOutput());
        auto reshapeOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(
                reduceOp, newEltwise.getOutput(), getIntArrayAttr(rewriter.getContext(), origOutShape.raw()));
        extendOpLoc(reshapeOp, "squeeze");
    }

    _log.trace("[{0}] Swapped EltwiseOp+ReduceOp to ReduceOp+EltwiseOp done", this->getDebugName());
    return mlir::success();
}

//
// SwapEltwiseAndReducePass
//

class SwapEltwiseAndReducePass final : public IE::impl::SwapEltwiseAndReduceBase<SwapEltwiseAndReducePass> {
public:
    explicit SwapEltwiseAndReducePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapEltwiseAndReducePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapEltwiseAndReduce<IE::MultiplyOp, IE::ReduceSumOp>>(&ctx, _log);
    patterns.add<SwapEltwiseAndReduce<IE::MultiplyOp, IE::ReduceMeanOp>>(&ctx, _log);
    patterns.add<SwapEltwiseAndReduce<IE::AddOp, IE::ReduceMeanOp>>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createSwapEltwiseAndReducePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapEltwiseAndReducePass(Logger log) {
    return std::make_unique<SwapEltwiseAndReducePass>(log);
}
