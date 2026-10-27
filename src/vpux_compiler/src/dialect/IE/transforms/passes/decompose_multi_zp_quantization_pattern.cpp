//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {

struct PatternOps {
    mlir::Value activation = nullptr;
    mlir::Operation* weights = nullptr;
    mlir::Value scale = nullptr;
    mlir::Value zeroPoint = nullptr;
    IE::SubtractOp subtractOp = nullptr;
    IE::MultiplyOp multiplyOp = nullptr;
    mlir::Operation* matMulOp = nullptr;
    IE::TransposeOp transposeOp = nullptr;
};

mlir::Operation* createMatMulOrFullyConnected(mlir::Operation* origOp, mlir::Value lhs, mlir::Value rhs,
                                              mlir::PatternRewriter& rewriter) {
    if (mlir::isa<IE::MatMulOp>(origOp)) {
        return IE::MatMulOp::create(rewriter, appendLoc(origOp->getLoc(), "matmul_decomposed"), lhs, rhs, false, true);
    }

    if (mlir::isa<IE::FullyConnectedOp>(origOp)) {
        return IE::FullyConnectedOp::create(rewriter, appendLoc(origOp->getLoc(), "fc_decomposed"), lhs, rhs, nullptr);
    }

    VPUX_THROW("Unsupported operation type: {0}", origOp->getName());
}

mlir::Value skipConverts(mlir::Value val) {
    while (auto convertOp = val.getDefiningOp<IE::ConvertOp>()) {
        val = convertOp.getInput();
    }
    return val;
}

mlir::LogicalResult decomposePerChannelPattern(mlir::PatternRewriter& rewriter, mlir::Operation* origWeights,
                                               IE::MultiplyOp origMultiplyOp, mlir::Operation* origMatMulOp,
                                               mlir::Value scale, mlir::Value zeroPoint, mlir::Value activation) {
    const auto wtShape = getShape(origWeights->getResult(0));

    rewriter.setInsertionPoint(origMultiplyOp);
    // Remove ZP from original pattern
    rewriter.replaceOpWithNewOp<IE::MultiplyOp>(origMultiplyOp, origWeights->getResult(0), scale,
                                                origMultiplyOp.getAutoBroadcastAttr(), origMultiplyOp.getPostOpAttr(),
                                                origMultiplyOp.getClampAttr(), origMultiplyOp.getOutputPaddingAttr(),
                                                origMultiplyOp.getInputPaddingAttr());

    rewriter.setInsertionPointAfter(origMatMulOp);
    const auto actAxis = static_cast<int64_t>(getShape(activation).size() - 1);
    const auto actAxesAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{actAxis});
    auto reduceSumOp = IE::ReduceSumOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "reduce_sum_act"),
                                               activation, actAxesAttr, /*keep_dims=*/true, nullptr, nullptr);

    auto multiplyScaleZP =
            IE::MultiplyOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "multiply_scale_zp"), scale, zeroPoint,
                                   IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    const Shape zpScaleOutShape({1, wtShape.raw()[0]});
    auto zpScaleReshapeOp =
            IE::ReshapeOp::create(rewriter, appendLoc(multiplyScaleZP.getLoc(), "reshape_scale_zp"),
                                  multiplyScaleZP.getOutput(), getIntArrayAttr(rewriter.getContext(), zpScaleOutShape));

    mlir::Value zpScaleVal = zpScaleReshapeOp.getOutput();
    auto reduceSumOutElemType = mlir::cast<vpux::NDTypeInterface>(reduceSumOp.getOutput().getType()).getElementType();
    auto zpScaleElemType = mlir::cast<vpux::NDTypeInterface>(zpScaleVal.getType()).getElementType();
    if (reduceSumOutElemType != zpScaleElemType) {
        auto convertOp = IE::ConvertOp::create(rewriter, appendLoc(multiplyScaleZP.getLoc(), "convert_scale_zp"),
                                               zpScaleVal, reduceSumOutElemType);
        zpScaleVal = convertOp.getOutput();
    }

    auto correction = IE::MultiplyOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "multiply_zp_correction"),
                                             reduceSumOp.getOutput(), zpScaleVal, IE::AutoBroadcastType::NUMPY, nullptr,
                                             nullptr, nullptr, nullptr);

    auto newSubtractOp = IE::SubtractOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "subtract_decomposed"),
                                                origMatMulOp->getResult(0), correction.getOutput(),
                                                IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    rewriter.replaceAllUsesExcept(origMatMulOp->getResult(0), newSubtractOp.getOutput(), {newSubtractOp});

    return mlir::success();
}

template <typename ConcreteOp>
mlir::FailureOr<PatternOps> initializePatternOps(const IE::WeightsDequantizeStructureInfo& wdInfo) {
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
    if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(userOp)) {
        if (!transposeOp->getResult(0).hasOneUse()) {
            return mlir::failure();
        }
        patternOps.transposeOp = transposeOp;
        userOp = *transposeOp->user_begin();
    }
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

template <typename ConcreteOp>
mlir::FailureOr<PatternOps> matchWeightsDequantizePattern(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                                          Logger log) {
    const auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, log.nest());
    if (mlir::failed(maybeWdInfo)) {
        return matchFailed(log, rewriter, origOp, "Failed to match WeightsDequantize structure.");
    }
    const auto& wdInfo = maybeWdInfo.value();

    if (config::asymmetricPerChannelZeroPointSupported(getModuleOp(origOp)) &&
        (wdInfo.hasConstWeights() || !IE::getTrueElemType(origOp).isInteger(2))) {
        return matchFailed(log, rewriter, origOp,
                           "Skip decomposition: asymmetricPerChannelZeroPoint is supported and weights are either "
                           "constant or not int2.");
    }

    if (!wdInfo.isKVcachedPattern() && IE::getTrueElemType(origOp).isInteger(2)) {
        return matchFailed(log, rewriter, origOp, "Skipping decomposing for u2 groupwise prefill pattern.");
    }

    auto maybePatternOps = initializePatternOps<ConcreteOp>(wdInfo);
    if (mlir::failed(maybePatternOps)) {
        return matchFailed(log, rewriter, origOp, "Failed to initialize pattern ops.");
    }
    return maybePatternOps;
}

class PerChannelPatternRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    PerChannelPatternRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx, benefit), _log(log.nest()) {
        this->setDebugName("PerChannelPatternRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    // Per-output-channel case: weights/zero-point/scale are 2D ([OC,IC]/[OC,1]/[OC,1]).
    static bool isPerChannelPattern(mlir::Operation* origWeights, mlir::Value scale, mlir::Value zeroPoint,
                                    mlir::Value activation, IE::TransposeOp transposeOp) {
        if (transposeOp != nullptr) {
            return false;
        }
        const auto wtShape = getShape(origWeights->getResult(0));
        if (wtShape.size() != 2 || getShape(activation).size() != 2) {
            return false;
        }
        const SmallVector<int64_t> perChannelShape{wtShape.raw()[0], 1};
        return getShape(zeroPoint).raw() == ArrayRef<int64_t>(perChannelShape) &&
               getShape(scale).raw() == ArrayRef<int64_t>(perChannelShape);
    }

    // The cases with constant zp and scale will be handled later in the compilation.
    static bool areOperandsModelInputs(mlir::Operation* weightsOp, mlir::Value scale, mlir::Value zeroPoint) {
        if (weightsOp->getNumOperands() != 1) {
            return false;
        }
        return mlir::isa<mlir::BlockArgument>(skipConverts(weightsOp->getOperand(0))) &&
               mlir::isa<mlir::BlockArgument>(skipConverts(scale)) &&
               mlir::isa<mlir::BlockArgument>(skipConverts(zeroPoint));
    }

    Logger _log;
};

/*
      Weights  Zero-points                   Weights                 Activation       Scale     Zero-points
      (OCxIC)     (OCx1)                     (OCxIC)                     |            (OCx1)     (OCx1)
         |          |                           |                        |               |         |
      Convert    Convert                     Convert     Scales       Reshape            |      Convert
      (OCxIC)     (OCx1)                     (OCxIC)     (OCx1)        (LxIC)            |       (OCx1)
           \      /                                \      /           /      \            \      /
           Subtract      Scales        =>          Multiply           |     ReduceSum     Multiply
           (OCxIC)       (OCx1)                     (OCxIC)           |       (Lx1)        (OCx1)
               \        /                              |              |         |             |
                Multiply                            Convert           |         |          Reshape
                (OCxIC)                             (OCxIC)           |         |          (1xOC)
                   |                                    \            /          |           /
                Convert                                 FullyConnected          |      Convert
Activation      (OCxIC)                                      (LxOC)             |      (1xOC)
    |             /                                              \              |    /
Reshape          /                                                \            Multiply
  (LxIC)        /                                                  \           (LxOC)
       \       /                                                    \         /
      FullyConnected                                                 Subtract
          (LxOC)                                                      (LxOC)
*/
mlir::LogicalResult PerChannelPatternRewriter::matchAndRewrite(IE::ConvertOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", origOp->getName(), origOp->getLoc());

    if (!config::hasEnableWeightsDynamicDequantization(getModuleOp(origOp))) {
        return matchFailed(_log, rewriter, origOp,
                           "Weights dynamic dequantization is disabled, the logic is not supported for constant "
                           "weights/zero-point cases.");
    }

    auto maybePatternOps = matchWeightsDequantizePattern(origOp, rewriter, _log);
    if (mlir::failed(maybePatternOps)) {
        return mlir::failure();
    }
    const auto& patternOps = maybePatternOps.value();

    if (!isPerChannelPattern(patternOps.weights, patternOps.scale, patternOps.zeroPoint, patternOps.activation,
                             patternOps.transposeOp)) {
        return matchFailed(_log, rewriter, origOp, "Not a 2D per-output-channel pattern.");
    }

    if (!areOperandsModelInputs(patternOps.weights, patternOps.scale, patternOps.zeroPoint)) {
        return matchFailed(_log, rewriter, origOp, "Weights, scale and zero-point must be model inputs.");
    }

    return decomposePerChannelPattern(rewriter, patternOps.weights, patternOps.multiplyOp, patternOps.matMulOp,
                                      patternOps.scale, patternOps.zeroPoint, patternOps.activation);
}

template <typename ConcreteOp>
class GroupWisePatternRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GroupWisePatternRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefit), _log(log.nest()) {
        this->setDebugName("GroupWisePatternRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

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

    auto maybePatternOps = matchWeightsDequantizePattern(origOp, rewriter, _log);
    if (mlir::failed(maybePatternOps)) {
        return matchFailed(_log, rewriter, origOp, "Failed to initialize pattern ops.");
    }
    const auto& patternOps = maybePatternOps.value();

    auto origWeights = patternOps.weights;
    auto origMultiplyOp = patternOps.multiplyOp;
    auto origMatMulOp = patternOps.matMulOp;
    auto scale = patternOps.scale;
    auto zeroPoint = patternOps.zeroPoint;
    auto activation = patternOps.activation;
    auto transposeOp = patternOps.transposeOp;

    auto is3DShape = [](mlir::Value val) {
        if (val == nullptr) {
            return false;
        }
        return getShape(val).size() == 3;
    };

    if (!is3DShape(origOp.getOutput()) || !is3DShape(zeroPoint) || !is3DShape(scale)) {
        return matchFailed(_log, rewriter, origOp,
                           "Expect 3D shape weights, zero-point and scale for group-wise pattern");
    }

    if (getShape(zeroPoint)[Dims3D::Act::B] == 1) {
        return matchFailed(_log, rewriter, origOp, "ZP is not per-channel. Skipping decomposing.");
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

    auto numGroups = transposeOp != nullptr ? wtShape[Dims3D::Filter::B] : wtShape[Dims3D::Filter::IC];
    const auto innerDimSize = numGroups * wtShape[Dims3D::Filter::OC];
    VPUX_THROW_UNLESS(actShape.totalSize() % innerDimSize == 0, "Got illegal group-wise pattern!");
    auto seqLen = actShape.totalSize() / innerDimSize;
    auto newActShape = Shape({seqLen, numGroups, wtShape[Dims3D::Filter::OC]});
    auto actReshapeOp = IE::ReshapeOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "reshape_act"), activation,
                                              getIntArrayAttr(rewriter.getContext(), newActShape));

    auto axis = actReshapeOp.getOutput().getType().getRank() - 1;
    auto axesAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{axis});
    auto reduceSumOp = IE::ReduceSumOp::create(rewriter, appendLoc(actReshapeOp.getLoc(), "reduce_sum_act"),
                                               actReshapeOp.getOutput(), axesAttr, false, nullptr, nullptr);

    // Create Scale*ZP Multiply branch
    auto multiplyScaleZP =
            IE::MultiplyOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "multiply_scale_zp"), scale, zeroPoint,
                                   IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);

    mlir::Value zpScaleVal = multiplyScaleZP.getOutput();
    if (transposeOp != nullptr) {
        zpScaleVal = IE::TransposeOp::create(rewriter, appendLoc(multiplyScaleZP.getLoc(), "transpose_zp"), zpScaleVal,
                                             nullptr, transposeOp.getOrderValueAttr())
                             .getOutput();
    }

    const auto zpScaleShape = getShape(zpScaleVal);
    const Shape newMultiplyScaleZPOutShape(
            {zpScaleShape[Dims3D::Act::B], zpScaleShape.totalSize() / zpScaleShape[Dims3D::Act::B]});
    auto multiplyScaleZPReshapeOp =
            IE::ReshapeOp::create(rewriter, appendLoc(multiplyScaleZP.getLoc(), "reshape_multiply_scale_zp"),
                                  zpScaleVal, getIntArrayAttr(rewriter.getContext(), newMultiplyScaleZPOutShape));

    auto newMatMulRhs = multiplyScaleZPReshapeOp.getOutput();
    auto reduceSumOutElemType = mlir::cast<vpux::NDTypeInterface>(reduceSumOp.getOutput().getType()).getElementType();
    auto multiplyScaleZPReshapeOutElemType =
            mlir::cast<vpux::NDTypeInterface>(multiplyScaleZPReshapeOp.getOutput().getType()).getElementType();
    if (reduceSumOutElemType != multiplyScaleZPReshapeOutElemType) {
        auto convertOp =
                IE::ConvertOp::create(rewriter, appendLoc(multiplyScaleZP.getLoc(), "convert_multiply_scale_zp"),
                                      multiplyScaleZPReshapeOp.getOutput(), reduceSumOutElemType);
        newMatMulRhs = convertOp.getOutput();
    }

    auto newMatMulOp = createMatMulOrFullyConnected(origMatMulOp, reduceSumOp.getOutput(), newMatMulRhs, rewriter);

    // Create final SubtractOp
    auto newSubtractOp = IE::SubtractOp::create(rewriter, appendLoc(origMatMulOp->getLoc(), "subtract_decomposed"),
                                                origMatMulOp->getResult(0), newMatMulOp->getResult(0),
                                                IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    rewriter.replaceAllUsesExcept(origMatMulOp->getResult(0), newSubtractOp.getOutput(), {newSubtractOp});

    return mlir::success();
}

}  // namespace vpux

void vpux::IE::registerDecomposeMultiZPQuantizationRewriters(RewriterRegistry& registry,
                                                             ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                             Logger log) {
    registry.registerRewriterSet("decompose-multi-zp-quantization", [&]() {
        vpux::IE::registerConvertOpRewriters(registry);
        registry.registerRewriter<GroupWisePatternRewriter<IE::ConvertOp>>("group-wise-pattern-rewriter-convert-op",
                                                                           log, benefitLevels[index]);
        registry.registerRewriter<GroupWisePatternRewriter<Const::DeclareOp>>("group-wise-pattern-rewriter-declare-op",
                                                                              log, benefitLevels[index]);
        registry.registerRewriter<PerChannelPatternRewriter>("per-channel-pattern-rewriter-convert-op", log,
                                                             benefitLevels[index]);
    });
}
