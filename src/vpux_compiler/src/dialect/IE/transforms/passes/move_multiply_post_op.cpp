//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MOVEMULTIPLYPOSTOP
#define GEN_PASS_DEF_MOVEMULTIPLYPOSTOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MoveMultiplyPostMatmul
//

//                  (1x32x1024x80)           (1x1x1x1)
//                              \               /
//      (1x32x1x80)         IE.Multiply (1x32x1024x80)
//               \            /
//            IE.Matmul(1x32x1x1024)

// To

//        (1x32x1x80)    1x32x1024x80)
//              \            /
//             IE.Matmul(1x32x1x1024)       (1x1x1x1)
//                         \                   /
//                        IE.Multiply (1x32x1x1024)

class MoveMultiplyPostMatmul final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveMultiplyPostMatmul(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyPostMatmul");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) {
    return inShape.totalSize() > outShape.totalSize();
}

mlir::Value getSingleDataInput(IE::MultiplyOp multiplyOp) {
    for (auto operand : multiplyOp->getOperands()) {
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(operand.getDefiningOp())) {
            if (IE::isBaseContentSplat(constOp)) {
                return operand;
            }
            return nullptr;
        }
        auto shape = getShape(operand);
        if (vpux::details::calcTotalShapeSize(shape.raw()) == 1) {
            return operand;
        }
    }
    return nullptr;
}

mlir::LogicalResult MoveMultiplyPostMatmul::matchAndRewrite(IE::MultiplyOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got multiply layer at '{1}'", origOp->getName(), origOp->getLoc());
    if (!origOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "multiply has more than one user");
    }

    if (origOp.getPostOpAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "multiply has post op attr");
    }

    if (origOp.getClampAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "multiply has clamp attr");
    }

    auto singleDataInput = getSingleDataInput(origOp);
    if (singleDataInput == nullptr) {
        return matchFailed(rewriter, origOp, "multiply doesn't have single data input");
    }

    const auto nonSingleDataOperand = origOp.getInput1() == singleDataInput ? origOp.getInput2() : origOp.getInput1();

    auto matmulOp = mlir::dyn_cast<IE::MatMulOp>(*origOp.getOutput().getUsers().begin());
    if (matmulOp == nullptr) {
        return matchFailed(rewriter, origOp, "multiply user is not a matmul");
    }

    if (!isBeneficialToConvert(getShape(origOp.getOutput()), getShape(matmulOp.getOutput()))) {
        return matchFailed(rewriter, origOp, "not benefical to swap multiply with matmul");
    }

    rewriter.setInsertionPoint(matmulOp);
    auto matmulInput1 = matmulOp.getInput1().getDefiningOp() == origOp ? nonSingleDataOperand : matmulOp.getInput1();
    auto matmulInput2 = matmulOp.getInput2().getDefiningOp() == origOp ? nonSingleDataOperand : matmulOp.getInput2();
    auto newMatMul =
            rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), matmulInput1, matmulInput2, matmulOp.getTransposeA(),
                                          matmulOp.getTransposeB(), matmulOp.getPostOpAttr());

    auto multiplyInput1 = origOp.getInput1() == singleDataInput ? origOp.getInput1() : newMatMul.getOutput();
    auto multiplyInput2 = origOp.getInput2() == singleDataInput ? origOp.getInput2() : newMatMul.getOutput();

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            origOp->getLoc(), multiplyInput1, multiplyInput2, origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    rewriter.replaceOp(matmulOp, newMultiply.getOutput());
    _log.trace("Successfully swap multiply with matmul");

    return mlir::success();
}

class MoveMultiplyPostConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveMultiplyPostConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyPostConcat");
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
        multiplyOp.getOutputChannelsAttr() != nullptr || multiplyOp.getInputChannelsAttr() != nullptr) {
        return false;
    }

    // In LLM, the optimization is only for KVcache, still need to keep multiply after FC/Matmul
    // for prefill to get better performance.
    return outShape.raw().front() == 1;
}

mlir::LogicalResult MoveMultiplyPostConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = origOp.getContext();
    // if concat doesn't have static offst attr, then it is single axis concat
    if (origOp.getStaticOffsetsAttr() != nullptr) {
        auto axis = IE::getConcatModifiedAxis(origOp);
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
// MoveMultiplyPostOpPass
//

class MoveMultiplyPostOpPass final : public IE::impl::MoveMultiplyPostOpBase<MoveMultiplyPostOpPass> {
public:
    explicit MoveMultiplyPostOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveMultiplyPostOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveMultiplyPostMatmul>(&ctx, _log);
    patterns.add<MoveMultiplyPostConcat>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveMultiplyPostOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMoveMultiplyPostOpPass(Logger log) {
    return std::make_unique<MoveMultiplyPostOpPass>(log);
}
