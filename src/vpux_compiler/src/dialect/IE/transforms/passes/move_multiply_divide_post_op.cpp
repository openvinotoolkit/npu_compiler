//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MOVEMULTIPLYDIVIDEPOSTOP
#define GEN_PASS_DEF_MOVEMULTIPLYDIVIDEPOSTOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MoveMultiplyDividePostMatmul
//

//                  (1x32x1024x80)           (1x1x1x1)
//                              \               /
//      (1x32x1x80)   IE.Multiply/IE.Divide (1x32x1024x80)
//               \            /
//            IE.Matmul(1x32x1x1024)

// To

//        (1x32x1x80)    1x32x1024x80)
//              \            /
//             IE.Matmul(1x32x1x1024)       (1x1x1x1)
//                         \                   /
//               IE.Multiply/IE.Divide (1x32x1x1024)

template <typename ConcreteOp>
class MoveMultiplyDividePostLayerGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveMultiplyDividePostLayerGeneric(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalTransformation(mlir::Operation* op) const;
    mlir::LogicalResult genericConditionChecker(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const;
    Logger _log;
};

template <typename ConcreteOp>
bool MoveMultiplyDividePostLayerGeneric<ConcreteOp>::isLegalTransformation(mlir::Operation* op) const {
    // IE::MatMulOp should not have post op
    if (auto matMulOp = mlir::dyn_cast<IE::MatMulOp>(op)) {
        return matMulOp.getPostOpAttr() == nullptr;
    }
    // IE::FullyConnectedOp should not have bias
    else if (auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(op)) {
        return fcOp.getBias() == nullptr;
    }

    return false;
}

template <typename ConcreteOp>
mlir::LogicalResult MoveMultiplyDividePostLayerGeneric<ConcreteOp>::genericConditionChecker(
        ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!origOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "op has more than one user");
    }

    if constexpr (std::is_same_v<ConcreteOp, IE::MultiplyOp>) {
        if (origOp.getPostOpAttr() != nullptr) {
            return matchFailed(_log, rewriter, origOp, "op has post op attr");
        }

        if (origOp.getClampAttr() != nullptr) {
            return matchFailed(_log, rewriter, origOp, "op has clamp attr");
        }

        return mlir::success();
    } else if constexpr (std::is_same_v<ConcreteOp, IE::DivideOp>) {
        return mlir::success();
    }

    return mlir::failure();
}

bool isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) {
    return inShape.totalSize() > outShape.totalSize();
}

mlir::Value getSingleDataInput(mlir::Operation* op) {
    for (auto operand : op->getOperands()) {
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(operand.getDefiningOp())) {
            if (IE::isBaseContentSplat(constOp)) {
                return operand;
            }
            return nullptr;
        }
        auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        const auto numElements = operandType.getNumElements();
        if (numElements == 1) {
            return operand;
        }
    }
    return nullptr;
}

template <typename ConcreteOp>
mlir::LogicalResult MoveMultiplyDividePostLayerGeneric<ConcreteOp>::matchAndRewrite(
        ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got multiply layer at '{1}'", origOp->getName(), origOp->getLoc());

    if (mlir::failed(genericConditionChecker(origOp, rewriter))) {
        return mlir::failure();
    }

    auto singleDataInput = getSingleDataInput(origOp);
    if (singleDataInput == nullptr) {
        return matchFailed(_log, rewriter, origOp, "op doesn't have single data input");
    }

    const mlir::Value nonSingleDataOperand =
            origOp.getInput1() == singleDataInput ? origOp.getInput2() : origOp.getInput1();

    auto layerOp = *origOp.getOutput().getUsers().begin();
    if (!mlir::isa<IE::MatMulOp, IE::FullyConnectedOp>(layerOp)) {
        return matchFailed(_log, rewriter, origOp, "invalid op user");
    }

    if (!isLegalTransformation(layerOp)) {
        return matchFailed(_log, rewriter, origOp, "illegal to swap op with layerOp");
    }

    if (!isBeneficialToConvert(getShape(origOp.getOutput()), getShape(layerOp->getResult(0)))) {
        return matchFailed(_log, rewriter, origOp, "not benefical to swap op with layerOp");
    }

    rewriter.setInsertionPoint(layerOp);
    auto origLhs = layerOp->getOperand(0);
    auto origRhs = layerOp->getOperand(1);
    mlir::Value newLhs = origLhs.getDefiningOp() == origOp ? nonSingleDataOperand : origLhs;
    mlir::Value newRhs = origRhs.getDefiningOp() == origOp ? nonSingleDataOperand : origRhs;
    SmallVector<mlir::Value> newLayerOpOperands = {newLhs, newRhs};
    mlir::IRMapping layerOpMapper;
    layerOpMapper.map(layerOp->getOperands(), newLayerOpOperands);
    auto newLayerOp = rewriter.clone(*layerOp, layerOpMapper);
    mlir::Value newLayerOpOutput = newLayerOp->getResult(0);

    auto newInput1 = origOp.getInput1() == singleDataInput ? origOp.getInput1() : newLayerOpOutput;
    auto newInput2 = origOp.getInput2() == singleDataInput ? origOp.getInput2() : newLayerOpOutput;

    rewriter.setInsertionPointAfter(newLayerOp);
    SmallVector<mlir::Value> newOrigOpOperands = {newInput1, newInput2};
    mlir::IRMapping origOpMapper;
    origOpMapper.map(origOp->getOperands(), newOrigOpOperands);
    auto newOrigOp = rewriter.clone(*origOp, origOpMapper);
    vpux::inferReturnTypes(newOrigOp, vpux::InferShapedTypeMode::ALL);
    rewriter.replaceAllUsesWith(layerOp->getResults(), newOrigOp->getResult(0));

    _log.trace("Successfully swap op with layerOp");
    return mlir::success();
}

class MoveMultiplyDividePostConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveMultiplyDividePostConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyDividePostConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Reshape from 32x64 to 1x32x64
bool isUnsqueezeLikeReshape(IE::ReshapeOp reshapeOp) {
    SmallVector<int64_t> inShape(getShape(reshapeOp.getInput()).raw());
    SmallVector<int64_t> outShape(getShape(reshapeOp.getOutput()).raw());
    if (outShape.size() <= inShape.size()) {
        return false;
    }
    outShape.erase(outShape.begin(), outShape.begin() + outShape.size() - inShape.size());

    return inShape == outShape;
}

bool isOptimizableMultiplyOp(IE::MultiplyOp multiplyOp) {
    auto leftInShape = getShape(multiplyOp.getInput1());
    auto rightInShape = getShape(multiplyOp.getInput2());
    auto outShape = getShape(multiplyOp.getOutput());
    if (leftInShape != outShape || rightInShape != outShape) {
        return false;
    }

    if (multiplyOp.getPostOpAttr() != nullptr || multiplyOp.getClampAttr() != nullptr ||
        multiplyOp.getOutputPaddingAttr() != nullptr || multiplyOp.getInputPaddingAttr() != nullptr) {
        return false;
    }

    // In LLM, the optimization is only for KVcache, still need to keep multiply after FC/Matmul
    // for prefill to get better performance.
    return outShape.raw().front() == 1;
}

mlir::LogicalResult MoveMultiplyDividePostConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = origOp.getContext();
    // if concat doesn't have static offst attr, then it is single axis concat
    if (origOp.getStaticOffsetsAttr() != nullptr) {
        auto axis = IE::getConcatAxes(origOp);
        if (axis.size() > 1) {
            return matchFailed(rewriter, origOp, "concat has multi axis");
        }
    }

    SmallVector<IE::MultiplyOp> multiplyOps;
    SmallVector<IE::ReshapeOp> reshapeOps;
    auto inputNums = origOp.getOperands().size();
    for (auto input : origOp.getOperands()) {
        if (auto reshapeOp = mlir::dyn_cast_or_null<IE::ReshapeOp>(input.getDefiningOp())) {
            if (reshapeOp->hasOneUse() && isUnsqueezeLikeReshape(reshapeOp)) {
                if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(reshapeOp.getInput().getDefiningOp())) {
                    reshapeOps.push_back(reshapeOp);
                    multiplyOps.push_back(multiplyOp);
                }
            }
        }
        if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(input.getDefiningOp())) {
            multiplyOps.push_back(multiplyOp);
        }
    }

    if (multiplyOps.size() != inputNums || (reshapeOps.size() != 0 && reshapeOps.size() != inputNums)) {
        return matchFailed(rewriter, origOp, "not all of concat parent is multiply");
    }

    SmallVector<mlir::Value> multiplyLeftInputs;
    SmallVector<mlir::Value> multiplyRightInputs;
    for (auto multiplyOp : multiplyOps) {
        if (!isOptimizableMultiplyOp(multiplyOp)) {
            return matchFailed(rewriter, origOp, "not optimizable multiply");
        }
        multiplyLeftInputs.push_back(multiplyOp.getInput1());
        multiplyRightInputs.push_back(multiplyOp.getInput2());
    }

    if (reshapeOps.size() == inputNums) {
        multiplyLeftInputs.clear();
        multiplyRightInputs.clear();
        for (auto p : multiplyOps | indexed) {
            auto multiplyOp = p.value();
            auto reshapeOp = reshapeOps[p.index()];
            auto leftReshape =
                    rewriter.create<IE::ReshapeOp>(appendLoc(reshapeOp->getLoc(), "_left_reshape"),
                                                   multiplyOp.getInput1(), nullptr, false,
                                                   getIntArrayAttr(ctx, getShape(reshapeOp.getOutput()).raw()))
                            .getOutput();
            multiplyLeftInputs.push_back(leftReshape);

            auto rightReshape =
                    rewriter.create<IE::ReshapeOp>(appendLoc(reshapeOp->getLoc(), "_right_reshape"),
                                                   multiplyOp.getInput2(), nullptr, false,
                                                   getIntArrayAttr(ctx, getShape(reshapeOp.getOutput()).raw()))
                            .getOutput();
            multiplyRightInputs.push_back(rightReshape);
        }
    }

    auto multiplyRightConcat = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_right_concat"),
                                                             mlir::ValueRange(multiplyRightInputs),
                                                             origOp.getPerAxisAttr(), origOp.getStaticOffsetsAttr());
    auto multiplyLeftConcat = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_left_concat"),
                                                            mlir::ValueRange(multiplyLeftInputs),
                                                            origOp.getPerAxisAttr(), origOp.getStaticOffsetsAttr());
    auto multiply = rewriter.create<IE::MultiplyOp>(appendLoc(multiplyOps.front()->getLoc(), "_multiply_after_concat"),
                                                    multiplyLeftConcat.getOutput(), multiplyRightConcat.getOutput(),
                                                    multiplyOps.front().getAutoBroadcastAttr(), nullptr, nullptr,
                                                    nullptr, nullptr);
    rewriter.replaceOp(origOp, multiply.getOutput());

    _log.trace("Successfully move multiply post Concat");
    return mlir::success();
}

//
// MoveMultiplyDividePostOpPass
//

class MoveMultiplyDividePostOpPass final : public IE::impl::MoveMultiplyDividePostOpBase<MoveMultiplyDividePostOpPass> {
public:
    explicit MoveMultiplyDividePostOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveMultiplyDividePostOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveMultiplyDividePostLayerGeneric<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveMultiplyDividePostLayerGeneric<IE::DivideOp>>(&ctx, _log);
    patterns.add<MoveMultiplyDividePostConcat>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveMultiplyDividePostOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMoveMultiplyDividePostOpPass(Logger log) {
    return std::make_unique<MoveMultiplyDividePostOpPass>(log);
}
