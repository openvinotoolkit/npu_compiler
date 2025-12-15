//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATETRANSPOSE
#define GEN_PASS_DEF_PROPAGATETRANSPOSE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MoveThroughSoftmax
//

class MoveThroughSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    MoveThroughSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto transposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto softmaxInputRank = mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getRank();
    const auto softmaxAxisInd = getPositiveAxisInd(origOp.getAxisIndAttr(), softmaxInputRank);

    const auto transposePerm = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    const auto newSoftmaxAxisInd = transposePerm.dimAt(softmaxAxisInd).ind();

    auto newSoftmaxOp =
            rewriter.create<IE::SoftMaxOp>(origOp.getLoc(), transposeOp.getInput().getType(), transposeOp.getInput(),
                                           getIntAttr(getContext(), newSoftmaxAxisInd), origOp.getPadSizeAttr());
    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newSoftmaxOp.getOutput(),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    origOp.replaceAllUsesWith(newTransposeOp.getOutput());

    return mlir::success();
}

//
// MoveThroughSlice
//

void updateSliceAttributes(mlir::ArrayAttr& staticSizes, mlir::ArrayAttr& staticOffsets, mlir::AffineMap permutation,
                           DimsOrder inOrder) {
    VPUX_THROW_UNLESS(permutation.isPermutation(), "Incorrect permutation");
    const auto order = DimsOrder::fromAffineMap(permutation);
    const auto dimsPermutation = order.toPermutation();

    VPUX_THROW_WHEN((staticSizes == nullptr) || (staticOffsets == nullptr), "Incorrect Slice parameters");
    const auto oldOffsets = parseIntArrayAttr<int64_t>(staticOffsets);
    const auto oldSizes = parseIntArrayAttr<int64_t>(staticSizes);
    SmallVector<int64_t> newOffsets;
    SmallVector<int64_t> newSizes;
    newOffsets.resize(oldOffsets.size(), 0);
    newSizes.resize(oldSizes.size(), 0);

    for (auto ind : irange(oldOffsets.size())) {
        const auto inDim = Dim(inOrder.dimAt(ind).ind());
        const auto outDim = dimsPermutation[inDim.ind()];

        newOffsets[outDim.ind()] = oldOffsets[inDim.ind()];
        newSizes[outDim.ind()] = oldSizes[inDim.ind()];
    }

    staticOffsets = getIntArrayAttr(staticOffsets.getContext(), newOffsets);
    staticSizes = getIntArrayAttr(staticSizes.getContext(), newSizes);
}

class MoveThroughSlice final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveThroughSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughSlice::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto isLegalSliceOp = [&](IE::SliceOp op) -> bool {
        auto prevTranspose = op.getSource().getDefiningOp<IE::TransposeOp>();
        if (prevTranspose == nullptr) {
            return true;
        }

        auto orderAttr = prevTranspose.getOrderValueAttr();
        auto order = DimsOrder::fromAffineMap(orderAttr.getValue());
        for (auto user : prevTranspose.getOutput().getUsers()) {
            if (!mlir::isa<IE::SliceOp>(user)) {
                return true;
            }

            auto slice = mlir::cast<IE::SliceOp>(user);
            auto inShape = getShape(slice.getSource());
            auto outShape = getShape(slice.getResult());
            if (inShape.size() != 4 || outShape.size() != 4) {
                return true;
            }

            auto sliceIdx = -1;
            for (auto i : irange(4)) {
                if (inShape[vpux::Dim(i)] != outShape[vpux::Dim(i)]) {
                    sliceIdx = i;
                }
            }
            if (sliceIdx == -1) {
                return true;
            }
            // Only slice on conv channel can be fused into conv
            if (order.dimAt(sliceIdx) != Dims4D::Act::C) {
                return true;
            }
        }

        auto prevConv = prevTranspose.getInput().getDefiningOp<IE::ConvolutionOp>();
        if (prevConv == nullptr || !prevConv->hasOneUse()) {
            return true;
        }

        _log.trace("Find illegal SliceOp {0}", op);

        return false;
    };

    if (isLegalSliceOp(origOp)) {
        return mlir::failure();
    }

    auto origTransposeOp = origOp.getSource().getDefiningOp<IE::TransposeOp>();
    auto orderAttr = origTransposeOp.getOrderValueAttr();
    const auto origPermuteInputOrder = DimsOrder::fromValue(origTransposeOp.getInput());

    SmallVector<mlir::Operation*> users;
    for (auto* user : llvm::make_early_inc_range(origTransposeOp.getOutput().getUsers())) {
        users.push_back(user);
    }

    llvm::sort(users, [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return !lhs->isBeforeInBlock(rhs);
    });

    for (auto op : users) {
        if (auto slice = mlir::dyn_cast_or_null<IE::SliceOp>(op)) {
            auto staticSizes = slice.getStaticSizesAttr();
            auto staticOffsets = slice.getStaticOffsetsAttr();

            updateSliceAttributes(staticSizes, staticOffsets, orderAttr.getValue(), origPermuteInputOrder);

            auto newSliceOp = rewriter.create<IE::SliceOp>(slice->getLoc(), origTransposeOp.getInput(), staticOffsets,
                                                           staticSizes);

            auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(slice, newSliceOp.getResult(), nullptr,
                                                                             origTransposeOp.getOrderValueAttr());

            extendOpLoc(newTranspose, "_transpose");
        }
    }

    rewriter.eraseOp(origTransposeOp);

    _log.trace("Swap slice success");
    return mlir::success();
}

//
// MoveThroughEltwiseGeneric
//

using VerifyCb = FuncRef<bool(mlir::Operation*)>;

template <class ConcreteOp>
class MoveThroughEltwiseGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughEltwiseGeneric(mlir::MLIRContext* ctx, Logger log, VerifyCb verifyFunc = nullptr)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _verifyFunc(verifyFunc) {
        this->setDebugName("MoveThroughEltwiseGeneric");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VerifyCb _verifyFunc;
};

template <class ConcreteOp>
mlir::LogicalResult MoveThroughEltwiseGeneric<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(origOp->getNumResults() == 1 && origOp->getNumOperands() == 1,
                      "Not a single input & output operation");

    auto transposeOp = origOp.getInput().template getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto transposeOrder = transposeOp.getOrderValue();
    if (!transposeOrder.has_value()) {
        return matchFailed(_log, rewriter, origOp, "Found invalid TransposeOp");
    }

    if ((_verifyFunc) && !_verifyFunc(origOp.getOperation())) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperand(0), transposeOp.getInput());
    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newOp->getResult(0),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    rewriter.replaceOp(origOp, newTransposeOp.getOutput());

    return mlir::success();
}

class MoveTransposeThroughMultiply final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveTransposeThroughMultiply(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("MoveTransposeThroughMultiply");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult processMultiplyOpWithBroadCastNonTransposeInput(IE::MultiplyOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult MoveTransposeThroughMultiply::processMultiplyOpWithBroadCastNonTransposeInput(
        IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const {
    /* Convert pattern
                     Input2
                      |
         Input1    Transpose
            \        /
             Multiply
                |
              Output

       to:

       New Input1   Input2
           |           |
    [inv Transpose]    |
           \           /
             Multiply
                |
           Transpose
                |
              Output
   */

    const auto& nestedLog = _log.nest();
    auto tranposeInput = origOp.getInput1();
    mlir::Value anotherInput = origOp.getInput2();
    auto transposeOp = tranposeInput.getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr) {
        tranposeInput = origOp.getInput2();
        anotherInput = origOp.getInput1();
        transposeOp = tranposeInput.getDefiningOp<IE::TransposeOp>();
    }

    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        nestedLog.trace("No TransposeOp or TransposeOp has more than one use");
        return mlir::failure();
    }

    auto transposeOutShape = getShape(transposeOp.getOutput());
    auto origOpOutShape = getShape(origOp.getOutput());
    if (transposeOutShape != origOpOutShape) {
        nestedLog.trace("TransposeOp output shape is different from Multiply output shape");
        return mlir::failure();
    }

    auto anotherInputShape = getShape(anotherInput);
    const auto isScalar = llvm::all_of(anotherInputShape, [](auto dim) {
        return dim == 1;
    });
    const auto isVector = llvm::count(anotherInputShape, 1) == static_cast<int64_t>(anotherInputShape.size() - 1);
    if (!isVector && !isScalar) {
        nestedLog.trace("Multiply's other input is not a vector or a scalar");
        return mlir::failure();
    }
    const auto orderVal = transposeOp.getOrderValue();
    if (!orderVal.has_value()) {
        nestedLog.trace("TranposeOp does not have order_value");
    }

    if (isVector) {
        auto inversePermutation = mlir::inversePermutation(orderVal.value());
        anotherInput = rewriter.createOrFold<IE::TransposeOp>(anotherInput.getLoc(), anotherInput, nullptr,
                                                              mlir::AffineMapAttr::get(inversePermutation));
    }

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            origOp.getLoc(), transposeOp.getInput(), anotherInput, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newMultiply.getOutput(), nullptr,
                                                 transposeOp.getOrderValueAttr());
    _log.debug("Successfully moved Transpose through Multiply.");
    return mlir::success();
}

mlir::LogicalResult MoveTransposeThroughMultiply::matchAndRewrite(IE::MultiplyOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.debug("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto& nestedLog = _log.nest();

    auto transpose1Op = origOp.getInput1().getDefiningOp<IE::TransposeOp>();
    auto transpose2Op = origOp.getInput2().getDefiningOp<IE::TransposeOp>();
    auto singleTranspose = (transpose1Op != nullptr && transpose2Op == nullptr) ||
                           (transpose1Op == nullptr && transpose2Op != nullptr);

    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    bool channelBiggerThanAlignment = origOutputType.getShape()[Dims4D::Act::C] >=
                                      VPU::NCEInvariant::getAlignment(origOutputType.getElementType());
    if (singleTranspose && channelBiggerThanAlignment) {
        nestedLog.trace("Processing Multiply with single Transpose and broadcastable input.");
        return processMultiplyOpWithBroadCastNonTransposeInput(origOp, rewriter);
    }

    if (getShape(origOp.getInput1()) != getShape(origOp.getInput2())) {
        nestedLog.trace("MultiplyOp does not have inputs with same shape");
        return mlir::failure();
    }

    if (transpose1Op == nullptr || !transpose1Op->hasOneUse() || transpose2Op == nullptr ||
        !transpose2Op->hasOneUse() || transpose1Op.getOrder() != nullptr || transpose2Op.getOrder() != nullptr) {
        // Only constant input is supported when order is set. In this case transpose can be fused into constant
        nestedLog.trace("Unsupported Transpose pattern.");
        return mlir::failure();
    }

    if (transpose1Op.getOrderValue() != transpose2Op.getOrderValue()) {
        nestedLog.trace("Transpose ops have different order values.");
        return mlir::failure();
    }

    auto input1 = transpose1Op.getInput();
    auto input2 = transpose2Op.getInput();
    auto newOutputType = origOutputType.changeShape(getShape(input1));
    auto newMultiplyOp = rewriter.create<IE::MultiplyOp>(
            origOp.getLoc(), newOutputType, input1, input2, origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(
            origOp, newMultiplyOp.getOutput(), transpose1Op.getOrder(), transpose1Op.getOrderValueAttr());
    extendOpLoc(newTranspose, "_transpose");

    rewriter.eraseOp(transpose1Op);
    rewriter.eraseOp(transpose2Op);
    _log.debug("Successfully moved Transpose through Multiply.");
    return mlir::success();
}

//
// MoveThroughOneInputEltwise
//

class MoveThroughOneInputEltwise final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    MoveThroughOneInputEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughOneInputEltwise");
    }

private:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughOneInputEltwise::matchAndRewrite(mlir::Operation* eltwiseOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), eltwiseOp->getName(), eltwiseOp->getLoc());

    if (eltwiseOp->getNumOperands() != 1 || eltwiseOp->getNumResults() != 1) {
        return matchFailed(_log, rewriter, eltwiseOp, "EltwiseOp is not a single input & output operation");
    }

    auto transposeOp = eltwiseOp->getOperand(0).getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, eltwiseOp, "TransposeOp not found or has multiple uses");
    }

    // No benefit to propagate TransposeOp through , e.g., ConvertOp {f16 -> f32}
    const auto srcType = eltwiseOp->getOperand(0).getType();
    const auto dstElemType = eltwiseOp->getResult(0).getType();
    if (getElemTypeSize(srcType) < getElemTypeSize(dstElemType)) {
        return matchFailed(_log, rewriter, eltwiseOp,
                           "Input element type size is smaller than output element type size");
    }

    _log.trace("[{0}] Propagate '{1}' at '{2}' through  '{3}' at '{4}'", this->getDebugName(), transposeOp->getName(),
               transposeOp->getLoc(), eltwiseOp->getName(), eltwiseOp->getLoc());

    mlir::IRMapping eltwiseMapper;
    eltwiseMapper.map(eltwiseOp->getOperand(0), transposeOp.getInput());
    auto newEltwiseOp = rewriter.clone(*eltwiseOp, eltwiseMapper);
    vpux::inferReturnTypes(newEltwiseOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(eltwiseOp, newEltwiseOp->getResult(0), transposeOp.getOrder(),
                                                 transposeOp.getOrderValueAttr());

    return mlir::success();
}

//
// MoveTransposeThroughRMS
//

class MoveTransposeThroughRMS final : public mlir::OpRewritePattern<IE::RMSOp> {
public:
    MoveTransposeThroughRMS(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::RMSOp>(ctx), _log(log) {
        this->setDebugName("MoveTransposeThroughRMS");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::RMSOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveTransposeThroughRMS::matchAndRewrite(IE::RMSOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto transposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto orderAttr = transposeOp.getOrderValueAttr();
    if (orderAttr == nullptr) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp missing order value");
    }

    const auto orderMap = orderAttr.getValue();
    if (!orderMap.isPermutation()) {
        return matchFailed(_log, rewriter, origOp, "Transpose order is not a permutation");
    }

    const auto rank = mlir::cast<vpux::NDTypeInterface>(transposeOp.getInput().getType()).getRank();
    const auto lastExpr = llvm::dyn_cast<mlir::AffineDimExpr>(orderMap.getResult(rank - 1));
    if (!lastExpr || lastExpr.getPosition() != static_cast<int64_t>(rank - 1)) {
        return matchFailed(_log, rewriter, origOp, "Transpose changes the last dimension");
    }

    auto newRmsOp = rewriter.create<IE::RMSOp>(origOp->getLoc(), transposeOp.getInput(), origOp.getGamma(),
                                               origOp.getEpsAttr());

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newRmsOp.getOutput(), transposeOp.getOrder(), orderAttr);
    _log.trace("Successfully moved Transpose through RMS.");

    return mlir::success();
}

//
// PropagateTransposePass
//

class PropagateTransposePass final : public IE::impl::PropagateTransposeBase<PropagateTransposePass> {
public:
    explicit PropagateTransposePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagateTransposePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto verifyAvgPool = [](mlir::Operation* op) {
        auto avgPoolOp = mlir::dyn_cast<IE::AvgPoolOp>(op);
        if ((avgPoolOp == nullptr) || (!IE::isEltwisePooling<IE::AvgPoolOp>(avgPoolOp))) {
            return false;
        }

        auto transposeOp = avgPoolOp.getInput().getDefiningOp<IE::TransposeOp>();
        if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
            return false;
        }

        auto convertOp = transposeOp.getInput().getDefiningOp<IE::ConvertOp>();
        // Not to swap for model input or model input -> convert to transpose case, to avoid introduce nce.permute op
        // before avgPool
        if (transposeOp.getInput().getDefiningOp() == nullptr ||
            (convertOp != nullptr && convertOp.getInput().getDefiningOp() == nullptr)) {
            return false;
        }

        return true;
    };

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughSoftmax>(&ctx, _log);
    patterns.add<MoveThroughSlice>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::AvgPoolOp>>(&ctx, _log, verifyAvgPool);
    patterns.add<IE::MoveTransposeAffineReshapeThroughAdd>(&ctx, vpux::benefitHigh, _log);
    patterns.add<MoveTransposeThroughMultiply>(&ctx, _log);
    patterns.add<MoveThroughOneInputEltwise>(&ctx, _log);
    patterns.add<MoveTransposeThroughRMS>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateTransposePass(Logger log) {
    return std::make_unique<PropagateTransposePass>(log);
}
