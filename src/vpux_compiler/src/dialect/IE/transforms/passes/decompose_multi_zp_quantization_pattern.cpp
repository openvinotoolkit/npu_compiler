//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEMULTIZPQUANTIZATIONPATTERN
#define GEN_PASS_DEF_DECOMPOSEMULTIZPQUANTIZATIONPATTERN
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

namespace vpux {

struct PatternOps {
    mlir::Value activation = nullptr;
    mlir::Operation* weights = nullptr;
    mlir::Value scale = nullptr;
    mlir::Value zeroPoint = nullptr;
    IE::SubtractOp subtractOp = nullptr;
    IE::MultiplyOp multiplyOp = nullptr;
    mlir::Operation* matMulOp = nullptr;
};

mlir::Operation* createMatMulOrFullyConnected(mlir::Operation* origOp, mlir::Value lhs, mlir::Value rhs,
                                              mlir::PatternRewriter& rewriter) {
    if (mlir::isa<IE::MatMulOp>(origOp)) {
        return rewriter.create<IE::MatMulOp>(appendLoc(origOp->getLoc(), "matmul_decomposed"), lhs, rhs, false, true,
                                             nullptr);
    }

    if (mlir::isa<IE::FullyConnectedOp>(origOp)) {
        return rewriter.create<IE::FullyConnectedOp>(appendLoc(origOp->getLoc(), "fc_decomposed"), lhs, rhs, nullptr);
    }

    VPUX_THROW("Unsupported operation type: {0}", origOp->getName());
}

template <typename ConcreteOp>
class GroupWisePatternRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GroupWisePatternRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log.nest()) {
        this->setDebugName("GroupWisePatternRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::FailureOr<PatternOps> initializePatternOps(const IE::WeightsDequantizeStructureInfo& wdInfo) const;
    Logger _log;
};

template <typename ConcreteOp>
mlir::FailureOr<PatternOps> GroupWisePatternRewriter<ConcreteOp>::initializePatternOps(
        const IE::WeightsDequantizeStructureInfo& wdInfo) const {
    PatternOps patternOps;
    auto opChain = wdInfo.getOpChain();
    for (auto op : opChain) {
        if (op == opChain.front()) {
            if (!mlir::isa<ConcreteOp>(op)) {
                return mlir::failure();
            }

            patternOps.weights = op;
            continue;
        }

        if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(op)) {
            if (patternOps.subtractOp != nullptr) {
                return mlir::failure();
            }

            patternOps.subtractOp = subtractOp;
            continue;
        }

        if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(op)) {
            if (patternOps.multiplyOp != nullptr) {
                return mlir::failure();
            }

            patternOps.multiplyOp = multiplyOp;
            continue;
        }
    }

    auto lastOp = wdInfo.getLastOp();
    if (!lastOp->getResult(0).hasOneUse()) {
        return mlir::failure();
    }

    auto userOp = *lastOp->user_begin();
    while (mlir::isa<IE::ConvertOp, IE::AffineReshapeOp, IE::ReshapeOp>(userOp)) {
        if (!userOp->getResult(0).hasOneUse()) {
            return mlir::failure();
        }

        userOp = *userOp->user_begin();
    }
    if (!mlir::isa<IE::MatMulOp, IE::FullyConnectedOp>(userOp)) {
        return mlir::failure();
    }

    if (patternOps.matMulOp != nullptr || patternOps.activation != nullptr) {
        return mlir::failure();
    }

    patternOps.matMulOp = userOp;
    patternOps.activation =
            userOp->getOperand(0) == lastOp->getResult(0) ? userOp->getOperand(1) : userOp->getOperand(0);

    if (patternOps.weights == nullptr || patternOps.matMulOp == nullptr || patternOps.activation == nullptr ||
        patternOps.multiplyOp == nullptr || patternOps.subtractOp == nullptr) {
        return mlir::failure();
    }

    patternOps.zeroPoint = patternOps.subtractOp.getInput1() == patternOps.weights->getResult(0)
                                   ? patternOps.subtractOp.getInput2()
                                   : patternOps.subtractOp.getInput1();
    patternOps.scale = patternOps.multiplyOp.getInput1() == patternOps.subtractOp.getOutput()
                               ? patternOps.multiplyOp.getInput2()
                               : patternOps.multiplyOp.getInput1();

    return patternOps;
}

/*
         Weights        Zero-points       weights                  Parameter   Zero-points
       (Cx128x64)       (Cx128x1)        (Cx128x64)                (Lx8192)    (Cx128x1)
              |          |                     |                   /   |            |
         Convert        Convert           Convert     Scales      / Reshape     Convert   Scales
       (Cx128x64)       (Cx128x1)        (Cx128x64)  (Cx128x1)   / (Lx128x64)  (Cx128x1) (Cx128x1)
               \        /                      |      /         /      |            |     /
                Subtract    Scales    =>       Multiply        /   ReduceSum     Multiply
                (Cx128x64) (Cx128x1)          (Cx128x64)      /     (Lx128)      (Cx128x1)
                   |       /                       |         /         |            |
                   Multiply                     Reshape     /          |          Reshape
                  (Cx128x64)                    (Cx8192)   /           |          (Cx128)
                      |                             |     /            |           /
    Parameter      Reshape                      FullyConnected        FullyConnected
    (Lx8192)       (Cx8192)                            (LxC)            (LxC)
            \        /                                      \          /
          FullyConnected                                      Subtract
              (LxC)                                             (LxC)
*/
template <typename ConcreteOp>
mlir::LogicalResult GroupWisePatternRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", origOp->getName(), origOp->getLoc());

    // Match the weights dequantize structure once...
    const auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
    if (mlir::failed(maybeWdInfo)) {
        return matchFailed(rewriter, origOp, "Failed to match WeightsDequantize structure.");
    }
    const auto& wdInfo = maybeWdInfo.value();

    if (!wdInfo.isKVcachedPattern() && IE::getTrueElemType(origOp).isInteger(2)) {
        return matchFailed(rewriter, origOp, "Skipping decomposing for u2 groupwise prefill pattern.");
    }

    auto maybePatternOps = initializePatternOps(wdInfo);
    if (mlir::failed(maybePatternOps)) {
        return matchFailed(rewriter, origOp, "Failed to initialize pattern ops.");
    }
    const auto& patternOps = maybePatternOps.value();

    auto origWeights = patternOps.weights;
    auto origMultiplyOp = patternOps.multiplyOp;
    auto origMatMulOp = patternOps.matMulOp;
    auto scale = patternOps.scale;
    auto zeroPoint = patternOps.zeroPoint;
    auto activation = patternOps.activation;

    auto is3DShape = [](mlir::Value val) {
        if (val == nullptr) {
            return false;
        }
        return getShape(val).size() == 3;
    };
    if (!is3DShape(origOp.getOutput()) || !is3DShape(zeroPoint) || !is3DShape(scale)) {
        return matchFailed(rewriter, origOp, "Expect 3D shape weights, zero-point and scale for group-wise pattern");
    }

    if (getShape(zeroPoint)[Dims3D::Act::B] == 1) {
        return matchFailed(rewriter, origOp, "ZP is not per-channel. Skipping decomposing.");
    }

    rewriter.setInsertionPointAfter(patternOps.weights);
    // Remove ZP from original pattern
    rewriter.replaceOpWithNewOp<IE::MultiplyOp>(origMultiplyOp, origWeights->getResult(0), scale,
                                                origMultiplyOp.getAutoBroadcastAttr(), origMultiplyOp.getPostOpAttr(),
                                                origMultiplyOp.getClampAttr(), origMultiplyOp.getOutputPaddingAttr(),
                                                origMultiplyOp.getInputPaddingAttr());

    // Create ReduceSum branch
    rewriter.setInsertionPointAfter(patternOps.matMulOp);
    auto actShape = getShape(activation);
    auto wtShape = getShape(origWeights->getResult(0));
    VPUX_THROW_UNLESS(actShape.totalSize() % (wtShape[Dims3D::Filter::IC] * wtShape[Dims3D::Filter::OC]) == 0,
                      "Got illegal group-wise pattern!");
    auto seqLen = actShape.totalSize() / (wtShape[Dims3D::Filter::IC] * wtShape[Dims3D::Filter::OC]);
    auto newActShape = Shape({seqLen, wtShape[Dims3D::Filter::IC], wtShape[Dims3D::Filter::OC]});
    auto actReshapeOp =
            rewriter.create<IE::ReshapeOp>(appendLoc(origMatMulOp->getLoc(), "reshape_act"), activation, nullptr, false,
                                           getIntArrayAttr(rewriter.getContext(), newActShape));

    auto axis = actReshapeOp.getOutput().getType().getRank() - 1;
    auto axesAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{axis});
    auto reduceSumOp =
            rewriter.create<IE::ReduceSumOp>(appendLoc(actReshapeOp.getLoc(), "reduce_sum_act"),
                                             actReshapeOp.getOutput(), nullptr, axesAttr, false, nullptr, nullptr);

    // Create Scale*ZP Multiply branch
    auto multiplyScaleZP =
            rewriter.create<IE::MultiplyOp>(appendLoc(origMatMulOp->getLoc(), "multiply_scale_zp"), scale, zeroPoint,
                                            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    auto multiplyScaleZPOutShape = getShape(multiplyScaleZP.getOutput());
    auto newMultiplyScaleZPOutShape =
            Shape({multiplyScaleZPOutShape[Dims3D::Act::B],
                   multiplyScaleZPOutShape.totalSize() / multiplyScaleZPOutShape[Dims3D::Act::B]});
    auto multiplyScaleZPReshapeOp = rewriter.create<IE::ReshapeOp>(
            appendLoc(multiplyScaleZP.getLoc(), "reshape_multiply_scale_zp"), multiplyScaleZP.getOutput(), nullptr,
            false, getIntArrayAttr(rewriter.getContext(), newMultiplyScaleZPOutShape));

    auto newMatMulRhs = multiplyScaleZPReshapeOp.getOutput();
    auto reduceSumOutElemType = mlir::cast<vpux::NDTypeInterface>(reduceSumOp.getOutput().getType()).getElementType();
    auto multiplyScaleZPReshapeOutElemType =
            mlir::cast<vpux::NDTypeInterface>(multiplyScaleZPReshapeOp.getOutput().getType()).getElementType();
    if (reduceSumOutElemType != multiplyScaleZPReshapeOutElemType) {
        auto convertOp =
                rewriter.create<IE::ConvertOp>(appendLoc(multiplyScaleZP.getLoc(), "convert_multiply_scale_zp"),
                                               multiplyScaleZPReshapeOp.getOutput(), reduceSumOutElemType);
        newMatMulRhs = convertOp.getOutput();
    }

    auto newMatMulOp = createMatMulOrFullyConnected(origMatMulOp, reduceSumOp.getOutput(), newMatMulRhs, rewriter);

    // Create final SubtractOp
    auto newSubtractOp = rewriter.create<IE::SubtractOp>(
            appendLoc(origMatMulOp->getLoc(), "subtract_decomposed"), origMatMulOp->getResult(0),
            newMatMulOp->getResult(0), IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    rewriter.replaceAllUsesExcept(origMatMulOp->getResult(0), newSubtractOp.getOutput(), {newSubtractOp});

    return mlir::success();
}

class DecomposeMultiZPQuantizationPatternPass final :
        public IE::impl::DecomposeMultiZPQuantizationPatternBase<DecomposeMultiZPQuantizationPatternPass> {
public:
    explicit DecomposeMultiZPQuantizationPatternPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DecomposeMultiZPQuantizationPatternPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    IE::ConvertOp::getCanonicalizationPatterns(patterns, &ctx);  // Ensures Convert chains are folded
    patterns.add<GroupWisePatternRewriter<IE::ConvertOp>>(&ctx, _log);
    patterns.add<GroupWisePatternRewriter<Const::DeclareOp>>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace vpux

//
// createDecomposeMultiZPQuantizationPatternPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeMultiZPQuantizationPatternPass(Logger log) {
    return std::make_unique<DecomposeMultiZPQuantizationPatternPass>(log);
}
