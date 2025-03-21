//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEOPTHROUGHBATCHCONCAT
#define GEN_PASS_DEF_PROPAGATEOPTHROUGHBATCHCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

std::optional<Dim> getBatchDim(ShapeRef shape) {
    std::optional<Dim> batchDim = std::nullopt;
    switch (shape.size()) {
    case 4:
        // batch dim is at position 1 for 4d shape when dim 0 is 1
        if (shape[Dim(0)] == 1) {
            batchDim = Dim(1);
        }
        break;
    case 3:
    case 2:
        // batch dim is at position 0 for 3d/2d shape
        batchDim = Dim(0);
        break;
    default:
        batchDim = std::nullopt;
        break;
    }
    return batchDim;
}

bool isBatchConcat(IE::ConcatOp concatOp) {
    const auto concatAttrs = concatOp.getPerAxisAttr();
    if (concatAttrs == nullptr) {
        return false;
    }

    const auto outputType = concatOp.getOutput().getType().dyn_cast<NDTypeInterface>();
    const auto rank = outputType.getRank();
    const auto concatAxis = getPositiveAxisInd(concatAttrs.getAxis(), rank);
    const auto batchDim = getBatchDim(outputType.getShape());
    if (!batchDim.has_value()) {
        return false;
    }
    if (concatAxis != batchDim.value().ind()) {
        return false;
    }

    const auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() == 0) {
        return false;
    }
    const auto firstShape = getShape(concatInputs.front());
    return llvm::all_of(concatInputs, [&](const mlir::Value v) {
        return getShape(v) == firstShape;
    });
}

class PropagateSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    PropagateSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (origOp.getInput().isa<mlir::BlockArgument>()) {
        return matchFailed(_log, rewriter, origOp, "Input of SoftmaxOp is block argument");
    }

    const auto isEnabledInput = [](mlir::Value input) {
        auto inputOp = input.getDefiningOp();
        while (mlir::isa_and_nonnull<IE::ReshapeOp>(inputOp)) {
            if (!inputOp->hasOneUse()) {
                return false;
            }
            inputOp = inputOp->getOperand(0).getDefiningOp();
        }
        return mlir::isa_and_nonnull<IE::MatMulOp>(inputOp) && inputOp->hasOneUse();
    };

    auto maybeAddOp = origOp.getInput().getDefiningOp<IE::AddOp>();
    auto concatOp = maybeAddOp == nullptr ? origOp.getInput().getDefiningOp<IE::ConcatOp>()
                                          : maybeAddOp.getInput1().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !isBatchConcat(concatOp) || !llvm::all_of(concatOp.getInputs(), isEnabledInput)) {
        return matchFailed(_log, rewriter, origOp, "No valid ConcatOp found");
    }

    const auto isValidAddOp = [&](IE::AddOp addOp, IE::ConcatOp concatOp) {
        auto input2 = addOp.getInput2();
        if ((input2.getDefiningOp<Const::DeclareOp>() != nullptr && getShape(input2).totalSize() == 1)) {
            return true;
        }

        return llvm::all_of(concatOp.getInputs(), [&](mlir::Value input) {
            return getShape(input) == getShape(input2);
        });
    };
    if (maybeAddOp != nullptr && !isValidAddOp(maybeAddOp, concatOp)) {
        return matchFailed(_log, rewriter, origOp, "Found invalid AddOp before SoftmaxOp");
    }

    // Concat axis must be different from softmax axis
    const auto rank = origOp.getInput().getType().dyn_cast<NDTypeInterface>().getRank();
    const auto concatAttrs = concatOp.getPerAxisAttr();
    const auto concatAxis = getPositiveAxisInd(concatAttrs.getAxis(), rank);
    const auto softmaxAxis = getPositiveAxisInd(origOp.getAxisIndAttr(), rank);
    if (concatAxis == softmaxAxis) {
        return matchFailed(_log, rewriter, origOp, "Concat axis conflicts with softmax axis");
    }

    auto concatInputShape = getShape(concatOp.getInputs()[0]);
    auto oneAndOnlySoftmaxAxisNotOne = [&]() {
        SmallVector<Dim> nonOneDims = getNonOneDim(concatInputShape);
        if (nonOneDims.size() == 1) {
            auto nonOneDim = nonOneDims.front();
            if (nonOneDim.ind() == softmaxAxis) {
                return true;
            }
        }

        return false;
    };
    if (oneAndOnlySoftmaxAxisNotOne()) {
        return matchFailed(_log, rewriter, origOp,
                           "No dim left for Multi-Cluster and Multi-SHAVEs tiling after propagation");
    }

    SmallVector<mlir::Value> newConcatInputs;
    for (auto concatInput : concatOp.getInputs() | indexed) {
        mlir::Value sliceSoftmaxInput = concatInput.value();
        if (maybeAddOp != nullptr) {
            auto newAddOp = rewriter.create<IE::AddOp>(
                    takeOpLoc(maybeAddOp, llvm::StringLiteral("slice_{0}"), concatInput.index()), concatInput.value(),
                    maybeAddOp.getInput2(), maybeAddOp.getAutoBroadcastAttr(), maybeAddOp.getPostOpAttr(),
                    maybeAddOp.getClampAttr(), maybeAddOp.getOutputChannelsAttr(), maybeAddOp.getInputChannelsAttr());
            sliceSoftmaxInput = newAddOp.getOutput();
        }

        auto sliceSoftmaxOp =
                rewriter.create<IE::SoftMaxOp>(takeOpLoc(origOp, StringLiteral("slice_{0}"), concatInput.index()),
                                               sliceSoftmaxInput, origOp.getAxisIndAttr(), origOp.getPadSizeAttr());
        newConcatInputs.push_back(sliceSoftmaxOp.getOutput());
    }

    auto newConcatOp = rewriter.create<IE::ConcatOp>(concatOp->getLoc(), newConcatInputs, Dim(concatAxis));
    rewriter.replaceOp(origOp, newConcatOp.getOutput());

    return mlir::success();
}

//
// PropagateReshape
//
template <class ReshapeT>
class PropagateReshape final : public mlir::OpRewritePattern<ReshapeT> {
public:
    PropagateReshape(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ReshapeT>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateReshape");
    }

private:
    mlir::LogicalResult matchAndRewrite(ReshapeT origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ReshapeT>
mlir::LogicalResult PropagateReshape<ReshapeT>::matchAndRewrite(ReshapeT origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (origOp.getInput().template isa<mlir::BlockArgument>()) {
        return matchFailed(_log, rewriter, origOp, "Input of ReshapeOp is block argument");
    }

    auto concatOp = origOp.getInput().template getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !concatOp->hasOneUse() || !isBatchConcat(concatOp)) {
        return matchFailed(_log, rewriter, origOp, "ConcatOp not found or invalid");
    }

    const auto inputShape = getShape(origOp.getInput());
    if (inputShape.size() != 2) {
        return matchFailed(_log, rewriter, origOp, "Unsupported input shape: {0}", inputShape);
    }

    const auto outputShape = getShape(origOp.getOutput());
    const auto batchDim = getBatchDim(outputShape);
    if (!batchDim.has_value()) {
        return matchFailed(_log, rewriter, origOp, "Unsupported output shape: {0}", outputShape);
    }

    auto sliceOutShape4D = outputShape.toValues();
    sliceOutShape4D[batchDim.value()] = 1;

    const auto concatInputs = concatOp.getInputs();
    const auto concatInputShape = getShape(concatInputs.front());

    if (concatInputShape.totalSize() != sliceOutShape4D.totalSize()) {
        return matchFailed(_log, rewriter, origOp,
                           "Size of inferred 4D shape of concat input ({0}) does not match with original shape ({1})",
                           concatInputShape, sliceOutShape4D);
    }

    _log.nest().trace("Propagating ReshapeOp before batch ConcatOp");

    const auto sliceOutShape4DAttr = getIntArrayAttr(rewriter.getContext(), sliceOutShape4D.raw());

    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& concatInput : concatInputs) {
        auto sliceReshape4D = rewriter.create<IE::ReshapeOp>(
                takeOpLoc(origOp, StringLiteral("slice_{0}_reshape"), newConcatInputs.size()), concatInput, nullptr,
                false, sliceOutShape4DAttr);
        _log.nest(2).trace("Inserted ReshapeOp: {0}", sliceReshape4D);
        newConcatInputs.push_back(sliceReshape4D.getOutput());
    }

    auto newConcatOp = rewriter.create<IE::ConcatOp>(takeOpLoc(concatOp, StringLiteral("out_concat")), newConcatInputs,
                                                     batchDim.value());
    rewriter.replaceOp(origOp, newConcatOp.getOutput());

    return mlir::success();
}

class PropagateFakeQuantize final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    PropagateFakeQuantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateFakeQuantize");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateFakeQuantize::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (origOp.getInput().isa<mlir::BlockArgument>()) {
        return matchFailed(_log, rewriter, origOp, "Input of SoftmaxOp is block argument");
    }

    const auto isEnabledInput = [&](mlir::Value input) {
        auto inputOp = input.getDefiningOp();
        if (mlir::isa_and_nonnull<IE::SoftMaxOp>(inputOp) && inputOp->hasOneUse()) {
            inputOp = inputOp->getOperand(0).getDefiningOp();
            if (mlir::isa_and_nonnull<IE::ReshapeOp>(inputOp) && inputOp->hasOneUse()) {
                inputOp = inputOp->getOperand(0).getDefiningOp();
            }
        }
        return mlir::isa_and_nonnull<IE::MatMulOp>(inputOp) && inputOp->hasOneUse();
    };

    auto concatOp = origOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !concatOp->hasOneUse() || !isBatchConcat(concatOp) ||
        !llvm::all_of(concatOp.getInputs(), isEnabledInput)) {
        return matchFailed(_log, rewriter, origOp, "ConcatOp not found or invalid");
    }

    const auto rank = origOp.getInput().getType().dyn_cast<NDTypeInterface>().getRank();
    const auto concatAxis = getPositiveAxisInd(concatOp.getPerAxisAttr().getAxis(), rank);
    if (!IE::isPerTensorFQ({origOp}) && concatAxis == Dims4D::Act::C.ind()) {
        return matchFailed(_log, rewriter, origOp, "Concat axis conflicts with per channel FakeQuantize axis");
    }

    rewriter.startOpModification(concatOp);
    rewriter.setInsertionPoint(concatOp);

    for (auto concatInput : concatOp.getInputs() | indexed) {
        auto sliceFQInput = concatInput.value();

        auto sliceFQOp = rewriter.create<IE::FakeQuantizeOp>(
                takeOpLoc(origOp, StringLiteral("slice_{0}"), concatInput.index()), sliceFQInput, origOp.getInputLow(),
                origOp.getInputHigh(), origOp.getOutputLow(), origOp.getOutputHigh(), origOp.getLevelsAttr(),
                origOp.getLowFpTypeAttr(), origOp.getAutoBroadcastAttr());
        concatOp.setOperand(checked_cast<uint32_t>(concatInput.index()), sliceFQOp.getOutput());
    }

    rewriter.replaceOp(origOp, concatOp->getResults());
    rewriter.finalizeOpModification(concatOp);

    return mlir::success();
}

//
// PropagateOpThroughBatchConcat
//

class PropagateOpThroughBatchConcat final :
        public IE::impl::PropagateOpThroughBatchConcatBase<PropagateOpThroughBatchConcat> {
public:
    explicit PropagateOpThroughBatchConcat(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

// Propagate SoftmaxOp(FakeQuantizeOp) with Unrolled MatmulOp for easier subgraph match when
// applying vertical fusion later for vertical graph "matmul->softmax->(fakeQuantize)->matmul"
// Need to generalize this method: E#80881
void PropagateOpThroughBatchConcat::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateReshape<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<PropagateReshape<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<PropagateSoftmax>(&ctx, _log);
    patterns.add<PropagateFakeQuantize>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateOpThroughBatchConcatPass(Logger log) {
    return std::make_unique<PropagateOpThroughBatchConcat>(log);
}
