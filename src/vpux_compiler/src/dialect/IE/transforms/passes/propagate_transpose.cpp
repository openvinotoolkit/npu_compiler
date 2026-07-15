//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
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

    auto newSoftmaxOp = rewriter.create<IE::SoftMaxOp>(
            takeOpLoc(origOp, "as_softmax"), transposeOp.getInput().getType(), transposeOp.getInput(),
            getIntAttr(getContext(), newSoftmaxAxisInd), origOp.getPadSizeAttr(), origOp.getDstElemTypeAttr(),
            origOp.getMaskAwareAttr());
    auto newTransposeOp =
            rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_softmax_out"), newSoftmaxOp.getOutput(),
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

            auto newSliceOp = rewriter.create<IE::SliceOp>(takeOpLoc(slice, "as_slice"), origTransposeOp.getInput(),
                                                           staticOffsets, staticSizes);

            auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(slice, newSliceOp.getResult(), nullptr,
                                                                             origTransposeOp.getOrderValueAttr());

            extendOpLoc(newTranspose, "transpose");
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

    auto newTransposeOp =
            rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_eltwise_out"), newOp->getResult(0),
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
        anotherInput = rewriter.createOrFold<IE::TransposeOp>(takeOpLoc(origOp, "inv_transpose_in"), anotherInput,
                                                              nullptr, mlir::AffineMapAttr::get(inversePermutation));
    }

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            takeOpLoc(origOp, "as_multiply"), transposeOp.getInput(), anotherInput, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newMultiply.getOutput(), nullptr,
                                                                     transposeOp.getOrderValueAttr());
    extendOpLoc(newTranspose, "transpose_mul1_out");
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
            takeOpLoc(origOp, "as_multiply"), newOutputType, input1, input2, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(
            origOp, newMultiplyOp.getOutput(), transpose1Op.getOrder(), transpose1Op.getOrderValueAttr());
    extendOpLoc(newTranspose, "transpose_mul2_out");

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

    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(
            eltwiseOp, newEltwiseOp->getResult(0), transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    extendOpLoc(newTranspose, "transpose_one_input_out");

    return mlir::success();
}

//
// MoveConcatThroughTranspose
//

class MoveConcatThroughTranspose final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveConcatThroughTranspose(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        this->setDebugName("MoveConcatThroughTranspose");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

    // Helper methods
    IE::TransposeOp findSingleTransposeInput(mlir::OperandRange inputs,
                                             const mlir::DenseSet<int64_t>& modifiedAxes) const;
    IE::TransposeOp findPostConcatTranspose(mlir::Operation* userOp) const;
    bool doTransposesCancel(mlir::AffineMap preTranspose, mlir::AffineMap postTranspose) const;
    mlir::LogicalResult validateNonTransposeInputs(mlir::OperandRange inputs, mlir::AffineMap inversePermutation) const;

    Logger _log;
};

// Find single Transpose input among concat inputs - must be from the largest input on concat axis
IE::TransposeOp MoveConcatThroughTranspose::findSingleTransposeInput(
        mlir::OperandRange inputs, const mlir::DenseSet<int64_t>& modifiedAxes) const {
    return IE::findSingleOpFromLargestInput<IE::TransposeOp>(inputs, modifiedAxes, _log);
}

// Find Transpose after Concat (directly or after SoftMax)
IE::TransposeOp MoveConcatThroughTranspose::findPostConcatTranspose(mlir::Operation* userOp) const {
    if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(userOp)) {
        return transposeOp;
    }

    if (auto softmaxOp = mlir::dyn_cast<IE::SoftMaxOp>(userOp)) {
        if (softmaxOp->hasOneUse()) {
            return mlir::dyn_cast<IE::TransposeOp>(*softmaxOp->getUsers().begin());
        }
    }

    _log.trace("No Transpose found after Concat (directly or after SoftMax)");
    return nullptr;
}

// Check if two transpose operations can cancel each other
bool MoveConcatThroughTranspose::doTransposesCancel(mlir::AffineMap preTranspose, mlir::AffineMap postTranspose) const {
    if (!preTranspose.isPermutation() || !postTranspose.isPermutation()) {
        _log.trace("One or both transpose orders are not permutations");
        return false;
    }

    // They cancel if: post(pre(x)) = x, i.e., composed map is identity
    const auto composedMap = postTranspose.compose(preTranspose);
    return composedMap.isIdentity();
}

// Validate that non-Transpose inputs are either constant or can handle inverse transpose
mlir::LogicalResult MoveConcatThroughTranspose::validateNonTransposeInputs(mlir::OperandRange inputs,
                                                                           mlir::AffineMap inversePermutation) const {
    for (auto input : inputs) {
        if (mlir::isa_and_present<IE::TransposeOp>(input.getDefiningOp())) {
            continue;
        }

        // Non-Transpose input: must be constant or inverse transpose must be trivial
        auto constOp = input.getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr) {
            // Check if inverse transpose is trivial based on this input's MemShape
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
            const auto inputMemShape = inputType.getMemShape();
            if (!isTrivialPermute(inputMemShape, inversePermutation)) {
                _log.trace("Non-Transpose input is not constant and inverse transpose is non-trivial");
                return mlir::failure();
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult MoveConcatThroughTranspose::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origConcatOp->getName(), origConcatOp->getLoc());

    auto inputs = origConcatOp.getInputs();
    if (inputs.size() < 2) {
        return mlir::failure();
    }

    // Check concat has only single axis
    const auto modifiedAxes = IE::getConcatAxes(origConcatOp);
    if (modifiedAxes.size() != 1) {
        _log.trace("Only single concat axis is supported, but got {0}", modifiedAxes.size());
        return mlir::failure();
    }

    // Find single Transpose input from the largest input on concat axis
    auto singleTransposeOp = findSingleTransposeInput(inputs, modifiedAxes);
    if (singleTransposeOp == nullptr) {
        return mlir::failure();
    }

    // Validate compute op pattern: ComputeOp -> Transpose
    auto computeOp = singleTransposeOp.getInput().getDefiningOp();
    if (!IE::isValidComputeOp(computeOp)) {
        return mlir::failure();
    }

    // Get and validate pre-concat transpose order
    const auto transposeOrder = singleTransposeOp.getOrderValue();
    if (!transposeOrder.has_value() || !transposeOrder->isPermutation()) {
        _log.trace("Invalid transpose order");
        return mlir::failure();
    }
    const auto orderMap = transposeOrder.value();

    // Find and validate post-concat transpose - check all users
    IE::TransposeOp postConcatTranspose = nullptr;
    mlir::AffineMap postTransposeOrder;

    for (auto userOp : origConcatOp->getUsers()) {
        auto currentTranspose = findPostConcatTranspose(userOp);
        if (currentTranspose == nullptr) {
            _log.trace("Not all users have Transpose pattern");
            return mlir::failure();
        }

        const auto currentOrder = currentTranspose.getOrderValue();
        if (!currentOrder.has_value()) {
            return mlir::failure();
        }

        // First user: save the transpose order
        if (postConcatTranspose == nullptr) {
            postConcatTranspose = currentTranspose;
            postTransposeOrder = currentOrder.value();
        } else {
            // Subsequent users: verify they have the same transpose order
            if (currentOrder.value() != postTransposeOrder) {
                _log.trace("Users have different Transpose orders");
                return mlir::failure();
            }
        }
    }

    // Check if transposes can cancel each other
    if (!doTransposesCancel(orderMap, postTransposeOrder)) {
        return mlir::failure();
    }

    // Validate non-Transpose inputs
    const auto inversePermutation = mlir::inversePermutation(orderMap);
    if (mlir::failed(validateNonTransposeInputs(inputs, inversePermutation))) {
        return mlir::failure();
    }

    // Calculate new concat axis after transpose
    const auto transposePermutation = DimsOrder::fromAffineMap(orderMap);
    const auto dimsPermutation = transposePermutation.toPermutation();
    const auto origConcatAxis = *modifiedAxes.begin();
    const auto transposedConcatAxis = dimsPermutation[origConcatAxis];
    const auto newConcatAxis = transposedConcatAxis.ind();

    // Build new inputs with inverse transpose for non-Transpose inputs
    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(inputs.size());

    for (auto inputsIdx : irange(inputs.size())) {
        auto input = inputs[inputsIdx];
        if (auto transposeOp = mlir::dyn_cast_if_present<IE::TransposeOp>(input.getDefiningOp())) {
            newInputs.push_back(transposeOp.getInput());
        } else {
            auto newTranspose =
                    rewriter.create<IE::TransposeOp>(takeOpLoc(origConcatOp, "inv_transpose_in{0}", inputsIdx), input,
                                                     nullptr, mlir::AffineMapAttr::get(inversePermutation));
            newInputs.push_back(newTranspose.getOutput());
        }
    }

    // Create new Concat and forward Transpose
    auto newConcat = rewriter.create<IE::ConcatOp>(takeOpLoc(origConcatOp, "as_concat"), newInputs, Dim(newConcatAxis));
    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origConcatOp, newConcat.getOutput(), nullptr,
                                                                     singleTransposeOp.getOrderValueAttr());
    extendOpLoc(newTranspose, "transpose_out");

    _log.trace("Successfully moved Concat through Transpose");
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

    auto newRmsOp = rewriter.create<IE::RMSOp>(takeOpLoc(origOp, "as_rms"), transposeOp.getInput(), origOp.getGamma(),
                                               origOp.getEpsAttr(), origOp.getConditionalEpsAttr());

    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newRmsOp.getOutput(),
                                                                     transposeOp.getOrder(), orderAttr);
    extendOpLoc(newTranspose, "transpose_out");
    _log.trace("Successfully moved Transpose through RMS.");

    return mlir::success();
}

//
// MoveThroughScatterUpdate
//

class MoveThroughScatterUpdate final : public mlir::OpRewritePattern<IE::ScatterUpdateOp> {
public:
    MoveThroughScatterUpdate(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterUpdateOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughScatterUpdate");
    }

private:
    struct MatchResult {
        IE::TransposeOp preTransposeOp;
        IE::TransposeOp postTransposeOp;
        int64_t axisValue = 0;
        DimsOrder invPreOrder;
    };

    mlir::FailureOr<MatchResult> matchPattern(IE::ScatterUpdateOp origOp) const {
        auto preTransposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
        if (preTransposeOp == nullptr || !preTransposeOp->hasOneUse()) {
            _log.trace("[{0}] Input TransposeOp not found or has multiple uses", getDebugName());
            return mlir::failure();
        }

        if (!origOp.getAxisValue().has_value()) {
            _log.trace("[{0}] axis_value attribute not set", getDebugName());
            return mlir::failure();
        }
        int64_t axisValue = origOp.getAxisValue().value();

        const auto dataRank =
                static_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank());
        if (axisValue < 0) {
            axisValue += dataRank;
        }
        if (axisValue < 0 || axisValue >= dataRank) {
            _log.trace("[{0}] axis_value out of range for data rank {1}", getDebugName(), dataRank);
            return mlir::failure();
        }

        const auto updatesRank =
                static_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(origOp.getUpdates().getType()).getRank());
        if (updatesRank < dataRank) {
            _log.trace("[{0}] updates rank {1} inconsistent with data rank {2}", getDebugName(), updatesRank, dataRank);
            return mlir::failure();
        }

        if (!origOp->hasOneUse()) {
            _log.trace("[{0}] ScatterUpdateOp has multiple uses", getDebugName());
            return mlir::failure();
        }
        auto postTransposeOp = mlir::dyn_cast<IE::TransposeOp>(*origOp->user_begin());
        if (postTransposeOp == nullptr) {
            _log.trace("[{0}] ScatterUpdateOp single user is not a TransposeOp", getDebugName());
            return mlir::failure();
        }

        const auto preOrderVal = preTransposeOp.getOrderValue();
        const auto postOrderVal = postTransposeOp.getOrderValue();
        if (!preOrderVal.has_value() || !postOrderVal.has_value()) {
            _log.trace("[{0}] Transpose order_value attribute not set", getDebugName());
            return mlir::failure();
        }
        if (!preOrderVal.value().isPermutation() || !postOrderVal.value().isPermutation()) {
            _log.trace("[{0}] Transpose order_value is not a permutation map", getDebugName());
            return mlir::failure();
        }

        const auto invPreOrder = DimsOrder::fromAffineMap(mlir::inversePermutation(preOrderVal.value()));
        if (invPreOrder != DimsOrder::fromAffineMap(postOrderVal.value())) {
            _log.trace("[{0}] Post-transpose is not the inverse of pre-transpose", getDebugName());
            return mlir::failure();
        }

        return MatchResult{preTransposeOp, postTransposeOp, axisValue, invPreOrder};
    }

    mlir::AffineMapAttr computeUpdatesPermutation(IE::ScatterUpdateOp origOp, int64_t axisValue, int64_t newAxisValue,
                                                  const DimsOrder& invPreOrder) const {
        const auto dataRank =
                static_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank());
        const auto updatesRank =
                static_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(origOp.getUpdates().getType()).getRank());
        const int64_t indicesRank = updatesRank - dataRank + 1;

        const auto invPermArr = to_small_vector(invPreOrder.toPermutation() | transformed([](Dim dim) {
                                                    return checked_cast<unsigned>(dim.ind());
                                                }));

        const auto toTransUpdatesPos = [&](int64_t origDim) -> unsigned {
            const auto transposedDim = static_cast<int64_t>(invPermArr[origDim]);
            return transposedDim < axisValue ? static_cast<unsigned>(transposedDim)
                                             : static_cast<unsigned>(transposedDim + indicesRank - 1);
        };

        SmallVector<unsigned> updatesPermArr(updatesRank);
        for (int64_t k = 0; k < updatesRank; k++) {
            if (k < newAxisValue) {
                updatesPermArr[k] = toTransUpdatesPos(k);
            } else if (k < newAxisValue + indicesRank) {
                updatesPermArr[k] = static_cast<unsigned>(axisValue + (k - newAxisValue));
            } else {
                updatesPermArr[k] = toTransUpdatesPos(k - indicesRank + 1);
            }
        }

        return mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(updatesPermArr, origOp.getContext()));
    }

    mlir::LogicalResult matchAndRewrite(IE::ScatterUpdateOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

        auto matchResult = matchPattern(origOp);
        if (mlir::failed(matchResult)) {
            return mlir::failure();
        }
        auto [preTransposeOp, postTransposeOp, axisValue, invPreOrder] = matchResult.value();

        const int64_t newAxisValue = checked_cast<int64_t>(
                DimsOrder::fromAffineMap(preTransposeOp.getOrderValue().value()).toPermutation()[axisValue].ind());

        // The SW kernel only supports axis=0. If the rewrite would produce a non-zero axis,
        // only proceed when the DMA path (which handles any axis) is available.
        if (newAxisValue != 0) {
            auto opWithDma = mlir::dyn_cast<IE::LayerWithDmaInterface>(origOp.getOperation());
            if (!opWithDma || !opWithDma.isSupported()) {
                _log.trace("[{0}] newAxisValue={1} but DMA path unavailable, skip to preserve kernel form",
                           getDebugName(), newAxisValue);
                return mlir::failure();
            }
        }

        const auto updatesPermAttr = computeUpdatesPermutation(origOp, axisValue, newAxisValue, invPreOrder);
        auto newUpdates = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "updates_transpose"), origOp.getUpdates(),
                                                           nullptr, updatesPermAttr);

        const auto newAxisAttr = getIntAttr(rewriter.getContext(), newAxisValue);
        auto newScatterOp =
                rewriter.create<IE::ScatterUpdateOp>(origOp->getLoc(), preTransposeOp.getInput(), origOp.getIndices(),
                                                     newUpdates.getOutput(), nullptr, newAxisAttr);

        rewriter.replaceOp(postTransposeOp, newScatterOp.getOutput());
        rewriter.eraseOp(origOp);
        rewriter.eraseOp(preTransposeOp);
        _log.trace("Successfully moved Transpose through ScatterUpdate");
        return mlir::success();
    }

private:
    Logger _log;
};

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
    patterns.add<MoveConcatThroughTranspose>(&ctx, _log);
    patterns.add<MoveTransposeThroughRMS>(&ctx, _log);
    patterns.add<MoveThroughScatterUpdate>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateTransposePass(Logger log) {
    return std::make_unique<PropagateTransposePass>(log);
}
