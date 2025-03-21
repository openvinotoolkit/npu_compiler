//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPFAKEQUANTWITHRESHAPEANDSTRIDEDSLICE
#define GEN_PASS_DEF_SWAPFAKEQUANTWITHRESHAPEANDSTRIDEDSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isReshapeKindOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(op);
}

mlir::Operation* getTargetParent(IE::FakeQuantizeOp fqOp) {
    auto parentOp = fqOp.getInput().getDefiningOp();
    while (parentOp != nullptr && isReshapeKindOp(parentOp)) {
        parentOp = parentOp->getOperand(0).getDefiningOp();
    }
    return parentOp;
}

mlir::Operation* getTargetChild(IE::FakeQuantizeOp fqOp) {
    auto childOp = *fqOp.getOutput().getUsers().begin();
    while (childOp != nullptr && isReshapeKindOp(childOp)) {
        if (!childOp->getResult(0).hasOneUse()) {
            return nullptr;
        }
        childOp = *childOp->getResult(0).getUsers().begin();
    }
    return childOp;
}

mlir::Operation* getLastReshape(IE::FakeQuantizeOp fqOp) {
    auto lastReshapeOp = *fqOp.getOutput().getUsers().begin();
    while (isReshapeKindOp(lastReshapeOp)) {
        auto nextOp = *lastReshapeOp->getResult(0).getUsers().begin();
        if (!isReshapeKindOp(nextOp)) {
            break;
        }
        lastReshapeOp = nextOp;
    }
    return lastReshapeOp;
}

bool matchFakeQuantReshapePattern(IE::FakeQuantizeOp fqOp) {
    // match [non-channel-aligned op] -> [optional Reshapes] -> [FQ] -> [Reshapes] -> [channel-aligned op]
    // swap to avoid redundant expand and permute ops
    auto outType = fqOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return false;
    }

    if (!fqOp.getOutput().hasOneUse()) {
        return false;
    }

    auto nextOp = *fqOp.getOutput().getUsers().begin();
    if (!isReshapeKindOp(nextOp)) {
        return false;
    }

    auto parentOp = getTargetParent(fqOp);
    auto childOp = getTargetChild(fqOp);
    auto lastReshapeOp = getLastReshape(fqOp);
    if (parentOp == nullptr || childOp == nullptr || mlir::isa<Const::DeclareOp>(parentOp)) {
        return false;
    }

    if (lastReshapeOp == nullptr ||
        lastReshapeOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::N] != 1) {
        // The batch size would be unrolled if moving the FQ after this Reshape
        return false;
    }

    if (!(!mlir::isa<IE::AlignedChannelsOpInterface>(parentOp) && mlir::isa<IE::AlignedChannelsOpInterface>(childOp))) {
        return false;
    }

    return true;
}

//
// FakeQuantReshapeSwapper
//

class FakeQuantReshapeSwapper final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantReshapeSwapper(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("FakeQuantReshapeSwapper");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantReshapeSwapper::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    if (!matchFakeQuantReshapePattern(origOp)) {
        return mlir::failure();
    }

    auto parentOp = origOp.getInput().getDefiningOp();
    auto targetChildOp = getTargetChild(origOp);
    auto lastReshapeOp = getLastReshape(origOp);

    VPUX_THROW_WHEN(targetChildOp == nullptr, "Target child op is nullptr!");
    rewriter.setInsertionPoint(targetChildOp);
    vpux::IE::FakeQuantizeOp newFQ;

    const auto notEqualToOne = [](const auto dim) {
        return dim != 1;
    };

    const auto inputLowShape = getShape(origOp.getInputLow());
    const auto bigDim = llvm::find_if(inputLowShape, notEqualToOne);
    const auto allOnes = (bigDim == inputLowShape.end());
    const auto bigDimNotLast = (bigDim != std::prev(inputLowShape.end()));
    if (allOnes || bigDimNotLast) {  // Per-tensor quantization
        newFQ = rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), lastReshapeOp->getResult(0), origOp.getInputLow(),
                                                    origOp.getInputHigh(), origOp.getOutputLow(),
                                                    origOp.getOutputHigh(), origOp.getLevelsAttr(),
                                                    origOp.getLowFpTypeAttr(), origOp.getAutoBroadcastAttr());
    } else {  // Reshape the quantization arrays accordingly when the quantization is per-channel
        auto inputLow = origOp.getInputLow();
        auto inputHigh = origOp.getInputHigh();

        auto fqOperationinputShape = to_small_vector(getShape(lastReshapeOp->getResult(0)));

        const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), fqOperationinputShape);

        auto inputLowReshaped =
                rewriter.create<IE::ReshapeOp>(origOp->getLoc(), inputLow, nullptr, false, newInputShapeAttr);
        auto inputHighReshaped =
                rewriter.create<IE::ReshapeOp>(origOp->getLoc(), inputHigh, nullptr, false, newInputShapeAttr);
        newFQ = rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), lastReshapeOp->getResult(0), inputLowReshaped,
                                                    inputHighReshaped, origOp.getOutputLow(), origOp.getOutputHigh(),
                                                    origOp.getLevelsAttr(), origOp.getLowFpTypeAttr(),
                                                    origOp.getAutoBroadcast());
    }

    lastReshapeOp->getResult(0).replaceAllUsesExcept(newFQ.getOutput(), llvm::SmallPtrSet<mlir::Operation*, 1>{newFQ});
    origOp->replaceAllUsesWith(parentOp);
    origOp->erase();

    return mlir::success();
}

//
// FakeQuantStridedSliceSwapper
//

class FakeQuantStridedSliceSwapper final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantStridedSliceSwapper(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("FakeQuantStridedSliceSwapper");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantStridedSliceSwapper::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());

    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        _log.nest().trace("Per-Axis FakeQuantize is not supported '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    // All users are StridedSliceOps
    const auto isStridedSlice = [&](auto user) {
        return mlir::isa<IE::StridedSliceOp>(user);
    };

    if (!llvm::all_of(origOp.getOutput().getUsers(), isStridedSlice)) {
        return mlir::failure();
    }

    // Rewrite the sub-graph.
    for (auto user : origOp.getOutput().getUsers()) {
        rewriter.setInsertionPointAfter(user);
        auto newFQ = rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), user->getResult(0), origOp.getInputLow(),
                                                         origOp.getInputHigh(), origOp.getOutputLow(),
                                                         origOp.getOutputHigh(), origOp.getLevelsAttr(),
                                                         origOp.getLowFpTypeAttr(), origOp.getAutoBroadcastAttr());
        user->getResult(0).replaceAllUsesExcept(newFQ.getOutput(), llvm::SmallPtrSet<mlir::Operation*, 1>{newFQ});
    }

    origOp.replaceAllUsesWith(origOp.getInput());
    _log.trace("[{0}] Rewrite successfuly '{1}'", getDebugName(), origOp->getLoc());
    origOp->erase();

    return mlir::success();
}

//
// SwapFakeQuantWithReshapeAndStridedSlicePass
//

class SwapFakeQuantWithReshapeAndStridedSlicePass final :
        public IE::impl::SwapFakeQuantWithReshapeAndStridedSliceBase<SwapFakeQuantWithReshapeAndStridedSlicePass> {
public:
    explicit SwapFakeQuantWithReshapeAndStridedSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};
void SwapFakeQuantWithReshapeAndStridedSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FakeQuantReshapeSwapper>(&ctx, _log);
    patterns.add<FakeQuantStridedSliceSwapper>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapFakeQuantWithReshapeAndStridedSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapFakeQuantWithReshapeAndStridedSlicePass(Logger log) {
    return std::make_unique<SwapFakeQuantWithReshapeAndStridedSlicePass>(log);
}
