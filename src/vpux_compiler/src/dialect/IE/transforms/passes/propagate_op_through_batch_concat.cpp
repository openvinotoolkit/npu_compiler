//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
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

    const auto outputType = mlir::dyn_cast<vpux::NDTypeInterface>(concatOp.getOutput().getType());
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

    if (mlir::isa<mlir::BlockArgument>(origOp.getInput())) {
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
            // The experiment indicates that performance is highly influenced by the size of Concat input. For small
            // sizes of the Concat input this rewrite will fragment to much the tensors resulting into a large number of
            // small tasks of which scheduling cost will overcome the vertical fusion benefits. The empirical threshold
            // for Concat input size indicated by experiments is 1 MB
            constexpr Byte MIN_CONCAT_INPUT_SIZE = 1_MB;
            const Byte inputSize = getTotalSize(input);
            if (inputSize < MIN_CONCAT_INPUT_SIZE) {
                return false;
            }
            auto inputShape = getShape(input);
            auto addInput2Shape = getShape(input2);
            int64_t inputShapeRank = inputShape.size();
            int64_t addInput2ShapeRank = addInput2Shape.size();
            // Check if Add input operands has equal shapes or broadcastable shapes
            if (inputShapeRank != addInput2ShapeRank) {
                int64_t maxRank = std::max(inputShapeRank, addInput2ShapeRank);
                // Iterate through the dimensions in reverse order
                for (int64_t i = 0; i < maxRank; ++i) {
                    int64_t inputDimIdx = inputShapeRank - 1 - i;
                    int64_t addInput2DimIdx = addInput2ShapeRank - 1 - i;
                    // Get dimension value and if the index is outside the range consider the dimension as equal to 1
                    int64_t inputDim = (inputDimIdx < 0) ? 1 : inputShape[Dim(inputDimIdx)];
                    int64_t addInput2Dim = (addInput2DimIdx < 0) ? 1 : addInput2Shape[Dim(addInput2DimIdx)];
                    // The dimensions must be equal
                    if (inputDim != addInput2Dim) {
                        return false;
                    }
                }
                return true;
            }
            // Check that the shapes are equal
            return inputShape == addInput2Shape;
        });
    };
    if (maybeAddOp != nullptr && !isValidAddOp(maybeAddOp, concatOp)) {
        return matchFailed(_log, rewriter, origOp, "Found invalid AddOp before SoftmaxOp");
    }

    // Concat axis must be different from softmax axis
    const auto rank = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank();
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
                    maybeAddOp.getClampAttr(), maybeAddOp.getOutputPaddingAttr(), maybeAddOp.getInputPaddingAttr());
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

    if (mlir::isa<mlir::BlockArgument>(origOp.getInput())) {
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

    if (mlir::isa<mlir::BlockArgument>(origOp.getInput())) {
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

    const auto rank = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank();
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
// PropagateMultiply
//
// (Multiply(Const(splat), Concat(RMS...)) -> Concat(Multiply(Const, RMS) ...))
class PropagateMultiply final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    PropagateMultiply(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateMultiply");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateMultiply::matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto input1 = origOp.getInput1();
    auto input2 = origOp.getInput2();
    // Identify concat and const(splat) sides
    auto concatOp = input1.getDefiningOp<IE::ConcatOp>();
    mlir::Value constVal = input2;
    if (concatOp == nullptr) {
        concatOp = input2.getDefiningOp<IE::ConcatOp>();
        constVal = input1;
    }
    if (concatOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "No ConcatOp operand");
    }
    if (!concatOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "ConcatOp has multiple uses");
    }

    auto constOp = constVal.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Other operand is not Const::DeclareOp");
    }
    if (!constOp.getContent().isSplat()) {
        return matchFailed(_log, rewriter, origOp, "Const is not splat");
    }

    // All concat inputs must be RMS outputs and each RMS must have single user (the concat)
    const auto inputs = concatOp.getInputs();
    if (!llvm::all_of(inputs, [&](mlir::Value v) {
            auto rmsOp = v.getDefiningOp<IE::RMSOp>();
            if (rmsOp == nullptr || !rmsOp->hasOneUse()) {
                return false;
            }
            auto gammaConstOp = rmsOp.getGamma().getDefiningOp<Const::DeclareOp>();
            return gammaConstOp != nullptr;
        })) {
        return matchFailed(_log, rewriter, origOp, "Not all Concat inputs are single-use RMS ops");
    }

    _log.nest().trace("Propagating MultiplyOp with splat const before ConcatOp");

    // Clone multiply per input (re-use same const splat)
    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(inputs.size());
    for (auto inVal : inputs) {
        auto newMul = rewriter.create<IE::MultiplyOp>(
                takeOpLoc(origOp, StringLiteral("slice_{0}"), newConcatInputs.size()), inVal, constVal,
                origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
        newConcatInputs.push_back(newMul.getOutput());
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(takeOpLoc(concatOp, StringLiteral("prop_mul_out")), newConcatInputs,
                                                   concatOp.getPerAxisAttr(), concatOp.getStaticOffsetsAttr());
    rewriter.replaceOp(origOp, newConcat.getOutput());

    if (concatOp->use_empty()) {
        rewriter.eraseOp(concatOp);
    }

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
    patterns.add<PropagateMultiply>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateOpThroughBatchConcatPass(Logger log) {
    return std::make_unique<PropagateOpThroughBatchConcat>(log);
}
