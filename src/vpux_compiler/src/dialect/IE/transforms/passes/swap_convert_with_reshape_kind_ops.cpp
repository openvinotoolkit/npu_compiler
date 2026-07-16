//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPCONVERTWITHRESHAPEKINDOPS
#define GEN_PASS_DEF_SWAPCONVERTWITHRESHAPEKINDOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapConvertWithReshapeKindOps
//

class SwapConvertWithReshapeKindOps final :
        public IE::impl::SwapConvertWithReshapeKindOpsBase<SwapConvertWithReshapeKindOps> {
public:
    explicit SwapConvertWithReshapeKindOps(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class OpSwapConverter;
    class PropagateConvertToFuseConverter;
    class EliminateConvertRoundTripThroughReshapeAndSlice;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

bool isReshapeKindOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return mlir::isa<IE::AffineReshapeOp, IE::DepthToSpaceOp, IE::ReshapeOp, IE::SqueezeOp, IE::TransposeOp,
                     IE::UnsqueezeOp>(op);
}

enum class PropagationDecision { NoPropagate, PropagateForward, PropagateBackward };

bool canPropagateThroughOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return isReshapeKindOp(op) || mlir::isa<IE::GatherOp>(op);
}

// For OV 2.0 API U8 we can have:
// NetworkInput (NCHW) -> Convert -> Transpose-> FQ . Because of this lately after propagate quantize
// dequantize pass and fuse convert with quantize pass, will be needed to propagate the quantizeCast
// quantParams to Transpose. We want to avoid this. Also in the end this Transpose will be done as
// PermuteCast.

// Output Case:
// Convert -> N reshapeKindOps -> return => N reshapeKindOps -> Convert -> return

bool canBeSwapped(IE::ConvertOp origOp, mlir::Operation* swapOp) {
    return (origOp.getDstElemType().isUnsignedInteger(8) &&
            mlir::isa<mlir::func::ReturnOp>(*swapOp->getResult(0).getUsers().begin())) ||
           mlir::isa<mlir::BlockArgument>(origOp.getInput());
}

void swapOps(IE::ConvertOp origOp, mlir::Operation* swapOp, mlir::PatternRewriter& rewriter) {
    const auto origDataType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto swapDataType = mlir::cast<vpux::NDTypeInterface>(swapOp->getResult(0).getType());
    const auto newDataType = swapDataType.changeElemType(origDataType.getElementType());

    rewriter.setInsertionPointAfter(swapOp);
    auto newConvert = rewriter.create<IE::ConvertOp>(origOp->getLoc(), swapOp->getResult(0), origOp.getDstElemType());
    swapOp->getResult(0).replaceAllUsesExcept(newConvert.getOutput(),
                                              llvm::SmallPtrSet<mlir::Operation*, 1>{newConvert});
    origOp->replaceAllUsesWith(mlir::ValueRange(origOp.getInput()));
    swapOp->getResult(0).setType(newDataType);
    rewriter.eraseOp(origOp);
}

mlir::Operation* findLastReshapeKindOp(mlir::Operation* swapOp) {
    while (isReshapeKindOp(*swapOp->getResult(0).getUsers().begin())) {
        swapOp = *swapOp->getResult(0).getUsers().begin();
    }
    return swapOp;
}

//
// OpSwapConverter
//

class SwapConvertWithReshapeKindOps::OpSwapConverter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    OpSwapConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapConvertWithReshapeKindOps::OpSwapConverter::matchAndRewrite(
        IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!origOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto swapOp = *origOp.getOutput().getUsers().begin();
    auto swapOpLoop = swapOp;

    if (isReshapeKindOp(swapOp)) {
        if (canBeSwapped(origOp, swapOp)) {
            swapOps(origOp, swapOp, rewriter);
            return mlir::success();
        }

        // Handle intermediate reshape kind ops
        swapOp = findLastReshapeKindOp(swapOp);

        if (!mlir::isa<mlir::func::ReturnOp>(*swapOp->getResult(0).getUsers().begin())) {
            return mlir::failure();
        }

        // Process swap Convert loop for reshape kind ops
        while (isReshapeKindOp(*swapOpLoop->getResult(0).getUsers().begin())) {
            swapOps(origOp, swapOpLoop, rewriter);
            return mlir::success();
        }
    }

    return mlir::failure();
}

//
// PropagateConvertToFuseConverter
//

class SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter final :
        public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    PropagateConvertToFuseConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origConvertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    // Find the Convert pair can be propagated to fuse
    mlir::Operation* findPropagationTarget(mlir::Operation* startOp) const;

    // Decide propagation direction based on data types
    PropagationDecision decidePropagationDirection(IE::ConvertOp firstConvert, IE::ConvertOp lastConvert) const;

    // Propagate Convert operation through the chain (forward direction)
    void propagateConvertForward(IE::ConvertOp firstConvert, IE::ConvertOp lastConvert,
                                 mlir::PatternRewriter& rewriter) const;

    // Propagate Convert operation through the chain (backward direction)
    void propagateConvertBackward(IE::ConvertOp firstConvert, IE::ConvertOp lastConvert,
                                  mlir::PatternRewriter& rewriter) const;
};

mlir::Operation* SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter::findPropagationTarget(
        mlir::Operation* startOp) const {
    mlir::Operation* currentOp = startOp;
    while (currentOp && currentOp->hasOneUse()) {
        auto nextOp = *currentOp->getUsers().begin();
        if (canPropagateThroughOp(nextOp)) {
            if (auto gatherOp = mlir::dyn_cast<IE::GatherOp>(nextOp)) {
                if (gatherOp.getInput() != currentOp->getResult(0)) {
                    // No propagation if the current Op is not Gather's data input
                    break;
                }
            }
            currentOp = nextOp;
        } else if (mlir::isa<IE::ConvertOp>(nextOp)) {
            // Found target ConvertOp
            return nextOp;
        } else {
            break;
        }
    }

    return nullptr;
}

PropagationDecision SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter::decidePropagationDirection(
        IE::ConvertOp firstConvert, IE::ConvertOp lastConvert) const {
    const auto firstInputType = mlir::cast<vpux::NDTypeInterface>(firstConvert.getInput().getType()).getElementType();
    const auto firstOutputType = mlir::cast<vpux::NDTypeInterface>(firstConvert.getOutput().getType()).getElementType();
    const auto lastOutputType = mlir::cast<vpux::NDTypeInterface>(lastConvert.getOutput().getType()).getElementType();
    const auto firstInputSize = getElemTypeSize(firstInputType).to<Bit>().count();
    const auto middleTypeSize = getElemTypeSize(firstOutputType).to<Bit>().count();
    const auto lastOutputSize = getElemTypeSize(lastOutputType).to<Bit>().count();
    if (!firstInputSize || !middleTypeSize || !lastOutputSize) {
        return PropagationDecision::NoPropagate;
    }

    int64_t minTypeSize = std::min({firstInputSize, middleTypeSize, lastOutputSize});
    if (minTypeSize == lastOutputSize) {
        return PropagationDecision::PropagateForward;
    } else if (minTypeSize == firstInputSize) {
        return PropagationDecision::PropagateBackward;
    }

    return PropagationDecision::NoPropagate;
}

void SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter::propagateConvertForward(
        IE::ConvertOp firstConvert, IE::ConvertOp lastConvert, mlir::PatternRewriter& rewriter) const {
    // Get the chain of operations between first and last convert
    SmallVector<mlir::Operation*> propagationChain;
    mlir::Operation* currentOp = *firstConvert.getOutput().getUsers().begin();
    while (currentOp && currentOp != lastConvert) {
        propagationChain.push_back(currentOp);
        currentOp = *currentOp->getUsers().begin();
    }

    auto lastConvertOutputType = mlir::cast<vpux::NDTypeInterface>(lastConvert.getOutput().getType());
    auto targetElementType = lastConvertOutputType.getElementType();

    // Create new Convert using firstConvert's input, skipping firstConvert entirely
    rewriter.setInsertionPointAfter(firstConvert);
    auto newConvertOp = rewriter.create<IE::ConvertOp>(appendLoc(firstConvert.getLoc(), "forward"),
                                                       firstConvert.getInput(), targetElementType);

    // Replace firstConvert with the new convert op's output
    rewriter.replaceOp(firstConvert, newConvertOp.getOutput());

    // Update all intermediate operations
    mlir::Value currentValue = newConvertOp.getOutput();
    for (auto* op : propagationChain) {
        auto currentOutputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
        auto newOutputType = currentOutputType.changeElemType(targetElementType);

        rewriter.startOpModification(op);
        op->getResult(0).setType(newOutputType);
        op->setOperand(0, currentValue);
        rewriter.finalizeOpModification(op);

        currentValue = op->getResult(0);
    }

    rewriter.replaceOp(lastConvert, currentValue);
}

void SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter::propagateConvertBackward(
        IE::ConvertOp firstConvert, IE::ConvertOp lastConvert, mlir::PatternRewriter& rewriter) const {
    // Get the chain of operations between first and last convert
    SmallVector<mlir::Operation*> propagationChain;
    mlir::Operation* currentOp = *firstConvert.getOutput().getUsers().begin();
    while (currentOp && currentOp != lastConvert) {
        propagationChain.push_back(currentOp);
        currentOp = *currentOp->getUsers().begin();
    }

    auto originalInputType = mlir::cast<vpux::NDTypeInterface>(firstConvert.getInput().getType());
    auto originalElementType = originalInputType.getElementType();

    // Update all intermediate operations
    for (auto* op : propagationChain) {
        auto currentOutputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
        auto newOutputType = currentOutputType.changeElemType(originalElementType);

        rewriter.startOpModification(op);
        op->getResult(0).setType(newOutputType);
        rewriter.finalizeOpModification(op);
    }

    // Create new Convert before the last Convert
    auto lastOp = propagationChain.back();
    rewriter.setInsertionPointAfter(lastOp);
    auto newConvertOp = rewriter.create<IE::ConvertOp>(appendLoc(firstConvert.getLoc(), "backward"),
                                                       lastOp->getResult(0), firstConvert.getDstElemType());

    rewriter.replaceOp(firstConvert, firstConvert.getInput());

    rewriter.startOpModification(lastConvert);
    lastConvert->setOperand(0, newConvertOp.getOutput());
    rewriter.finalizeOpModification(lastConvert);
}

mlir::LogicalResult SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter::matchAndRewrite(
        IE::ConvertOp origConvertOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Propagate Convert operation '{0}' at '{1}'", origConvertOp->getName(), origConvertOp->getLoc());

    if (!origConvertOp.getOutput().hasOneUse()) {
        return matchFailed(rewriter, origConvertOp, "The Convert has more than one consumer");
    }

    auto firstUser = *origConvertOp.getOutput().getUsers().begin();
    if (!canPropagateThroughOp(firstUser)) {
        return matchFailed(rewriter, origConvertOp, "The Convert consumer is not a reshape kind operation or gather");
    }

    if (auto gatherOp = mlir::dyn_cast<IE::GatherOp>(firstUser)) {
        if (gatherOp.getInput() != origConvertOp.getOutput()) {
            return matchFailed(rewriter, origConvertOp, "The Convert is not Gather's data input");
        }
    }

    auto targetOp = findPropagationTarget(firstUser);
    if (!targetOp || !mlir::isa<IE::ConvertOp>(targetOp)) {
        return matchFailed(rewriter, origConvertOp, "Could not find target Convert to fuse");
    }

    auto lastConvertOp = mlir::cast<IE::ConvertOp>(targetOp);

    // Decide propagation direction and finish propagation
    auto decision = decidePropagationDirection(origConvertOp, lastConvertOp);
    switch (decision) {
    case PropagationDecision::PropagateForward:
        propagateConvertForward(origConvertOp, lastConvertOp, rewriter);
        break;
    case PropagationDecision::PropagateBackward:
        propagateConvertBackward(origConvertOp, lastConvertOp, rewriter);
        break;
    default:
        return matchFailed(rewriter, origConvertOp, "Not propagate as no benefit");
    }

    _log.trace("Propagate Convert operation to fuse successfully");
    return mlir::success();
}

//
// EliminateConvertRoundTripThroughReshapeAndSlice
//
// Pattern:
//   Convert(A->B) -> ReshapeKindOp -> [Slice, Slice, ...] -> [Convert(B->A), Convert(B->A), ...]
//
// Optimization: eliminate the leading Convert and all trailing Converts by updating intermediate
// op types from B back to A -- the round-trip conversion is a no-op.
//
// Requirements:
//   - The leading Convert has exactly one user (the reshape-kind op).
//   - ALL users of the reshape-kind op are IE::SliceOps.
//   - Each SliceOp has exactly one user, which is a Convert back to the original type A.
//

class SwapConvertWithReshapeKindOps::EliminateConvertRoundTripThroughReshapeAndSlice final :
        public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    EliminateConvertRoundTripThroughReshapeAndSlice(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx, /*benefit=*/2), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp leadConvert, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapConvertWithReshapeKindOps::EliminateConvertRoundTripThroughReshapeAndSlice::matchAndRewrite(
        IE::ConvertOp leadConvert, mlir::PatternRewriter& rewriter) const {
    _log.trace("EliminateConvertRoundTrip: checking Convert '{0}' at '{1}'", leadConvert->getName(),
               leadConvert->getLoc());

    // Leading Convert must have exactly one user.
    if (!leadConvert.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto* reshapeOp = *leadConvert.getOutput().getUsers().begin();
    if (!isReshapeKindOp(reshapeOp)) {
        return mlir::failure();
    }

    // All users of the reshape-kind op must be SliceOps.
    SmallVector<IE::SliceOp> sliceOps;
    for (auto* user : reshapeOp->getResult(0).getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (!sliceOp) {
            return mlir::failure();
        }
        sliceOps.push_back(sliceOp);
    }
    if (sliceOps.empty()) {
        return mlir::failure();
    }

    // Each Slice must have exactly one user: a Convert back to the original type.
    const auto originalElemType = mlir::cast<vpux::NDTypeInterface>(leadConvert.getInput().getType()).getElementType();
    const auto middleElemType = mlir::cast<vpux::NDTypeInterface>(leadConvert.getOutput().getType()).getElementType();

    SmallVector<IE::ConvertOp> trailingConverts;
    for (auto sliceOp : sliceOps) {
        if (!sliceOp.getResult().hasOneUse()) {
            return mlir::failure();
        }
        auto* sliceUser = *sliceOp.getResult().getUsers().begin();
        auto tailConvert = mlir::dyn_cast<IE::ConvertOp>(sliceUser);
        if (!tailConvert) {
            return mlir::failure();
        }
        // Must convert back to the original type.
        if (tailConvert.getDstElemType() != originalElemType) {
            return mlir::failure();
        }
        // Sanity: input to tail convert should be middleElemType.
        const auto tailInputElemType =
                mlir::cast<vpux::NDTypeInterface>(tailConvert.getInput().getType()).getElementType();
        if (tailInputElemType != middleElemType) {
            return mlir::failure();
        }
        trailingConverts.push_back(tailConvert);
    }

    _log.trace("EliminateConvertRoundTrip: matched -- eliminating round-trip Convert '{0}' at '{1}' through reshape "
               "'{2}' at '{3}' + {4} slices",
               leadConvert->getName(), leadConvert->getLoc(), reshapeOp->getName(), reshapeOp->getLoc(),
               sliceOps.size());

    // Step 1 & 2: bypass the leading Convert -- wire original input into the reshape-kind op
    // and update its result type to originalElemType.
    auto reshapeOutType = mlir::cast<vpux::NDTypeInterface>(reshapeOp->getResult(0).getType());
    rewriter.startOpModification(reshapeOp);
    reshapeOp->setOperand(0, leadConvert.getInput());
    reshapeOp->getResult(0).setType(mlir::cast<mlir::Type>(reshapeOutType.changeElemType(originalElemType)));
    rewriter.finalizeOpModification(reshapeOp);

    // Step 3 & 4: update each Slice result type and replace the trailing Convert with the Slice.
    for (size_t i = 0; i < sliceOps.size(); ++i) {
        auto sliceOp = sliceOps[i];
        auto sliceOutType = mlir::cast<vpux::NDTypeInterface>(sliceOp.getResult().getType());
        rewriter.startOpModification(sliceOp);
        sliceOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(
                mlir::cast<mlir::Type>(sliceOutType.changeElemType(originalElemType))));
        rewriter.finalizeOpModification(sliceOp);

        rewriter.replaceOp(trailingConverts[i], sliceOp.getResult());
    }

    // Step 5: erase the now-unused leading Convert.
    rewriter.eraseOp(leadConvert);

    return mlir::success();
}

void SwapConvertWithReshapeKindOps::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapConvertWithReshapeKindOps::OpSwapConverter>(&ctx, _log);
    patterns.add<SwapConvertWithReshapeKindOps::PropagateConvertToFuseConverter>(&ctx, _log);
    patterns.add<SwapConvertWithReshapeKindOps::EliminateConvertRoundTripThroughReshapeAndSlice>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapConvertWithReshapeKindOpsPass(Logger log) {
    return std::make_unique<SwapConvertWithReshapeKindOps>(log);
}
