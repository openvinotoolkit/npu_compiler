//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/IRMapping.h>

namespace vpux {
namespace IE {

class FloatOutConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FloatOutConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convolutionOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FloatOutGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    FloatOutGroupConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FloatOutTransposedConvRewriter final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    FloatOutTransposedConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp transposedConvOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FloatOutAvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    FloatOutAvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp avgPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FloatOutAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FloatOutAddRewriter(mlir::MLIRContext* ctx, const bool allowDifferentScales, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _allowDifferentScales(allowDifferentScales), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const bool _allowDifferentScales;
    Logger _log;
};

class QuantizeWithNCERewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeWithNCERewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    // Walk through leading view-only ops to recover the semantic producer value.
    mlir::Value peelLeadingPureViewOps(mlir::Value value) const;

    // Walk through trailing single-use view-only users to recover the semantic consumer value.
    mlir::Value peelTrailingPureViewOps(mlir::Value value) const;

    // Checks whether a producer operand originates from the conv-lowered SDPA chain:
    // Convolution -> Softmax -> (view ops) -> Convolution consumer operand.
    // This intentionally excludes MatMul-form SDPA currently seen in decoder paths.
    bool isSdpaSoftmaxConsumerOperand(mlir::Value operand) const;

    // Reject Quantize+NCE fusion only for conv-lowered SDPA outputs with non-zero zero-point.
    // Keeping these outputs unfused preserves eligibility for downstream softmax decomposition.
    // The MatMul -> Softmax -> MatMul variant is intentionally not matched here.
    bool shouldRejectQuantizeFusionForSdpaSoftmax(mlir::Operation* producerOp,
                                                  mlir::quant::QuantizedType qElemType) const;

private:
    Logger _log;
};

// Fuses a trailing IE.QuantizeOp into a float-input IE.MultiplyOp, producing a
// quantized-output Multiply. This handles the pattern IE.Multiply(fp16, fp16) -> fp16 -> IE.Quantize -> quant,
// which is intended to remain a MultiplyOp with quantized output, that will be executed on SHAVE.
class QuantizeWithMultiplyRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeWithMultiplyRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
class MixedFloatInQuantWeightsRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MixedFloatInQuantWeightsRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MixedFloatInQuantWeightsRewriter<ConcreteOp>::matchAndRewrite(
        ConcreteOp convOp, mlir::PatternRewriter& rewriter) const {
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(true)) {
        return mlir::failure();
    }

    auto op = convOp.getOperation();

    const auto dequantizeType = IE::findQuantizedInput(op->getOperand(0), {});
    const auto filterDequantizeType = IE::findQuantizedInput(op->getOperand(1), IE::getLegalWeightsQuantAxes(convOp));

    // Not fit for input weights mixed precision, other rewriters will apply
    if (dequantizeType != nullptr || filterDequantizeType == nullptr) {
        return mlir::failure();
    }

    const auto quantFilterDequantizeType = mlir::dyn_cast<mlir::quant::QuantizedType>(
            mlir::cast<vpux::NDTypeInterface>(filterDequantizeType.getType()).getElementType());
    if (quantFilterDequantizeType == nullptr) {
        return mlir::failure();
    }

    const auto isSignedQuantizedType = [](mlir::quant::QuantizedType quantType) {
        if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
            const auto quantized = mlir::cast<mlir::quant::QuantizedType>(quantType);
            if (const auto quantileStorage = mlir::dyn_cast<vpux::type::QuantileType>(quantized.getStorageType())) {
                mlir::Type quantileType = quantileStorage.getQuantileType();
                if (auto intType = mlir::dyn_cast<mlir::IntegerType>(quantileType)) {
                    return intType.isSigned();
                } else {
                    // quantile type is a float type
                    return true;
                }
            }
        }
        return quantType.isSigned();
    };

    const auto perChannelQuantType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(quantFilterDequantizeType);
    const auto perTensorQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantFilterDequantizeType);
    const auto isSymmetricQuant = IE::isSymmetricQuantType(quantFilterDequantizeType);
    auto moduleOp = getModuleOp(convOp.getOperation());
    const auto isAsymmetricPerChannelSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
    const auto isAsymmetricPerTensorSupported = config::asymmetricPerTensorZeroPointSupported(moduleOp);

    // Only signed quant is supported for input + wt mixed precision
    if (!isSignedQuantizedType(quantFilterDequantizeType) ||
        (perChannelQuantType && !isAsymmetricPerChannelSupported && !isSymmetricQuant) ||
        (perTensorQuantType && !isAsymmetricPerTensorSupported && !isSymmetricQuant)) {
        return mlir::failure();
    }

    const auto hasLeakyReLUConsumer = llvm::any_of(convOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::LeakyReluOp>(op);
    });

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(quantFilterDequantizeType) &&
        (hasLeakyReLUConsumer || IE::hasLeakyReLUPostOp(convOp))) {
        return mlir::failure();
    }

    const auto hasReLUConsumer = llvm::any_of(convOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::ReLUOp>(op);
    });

    // Check for problematic combination: per-axis quantization + ReLU postOp + negative quant scales on MTL and LNL
    const auto arch = config::getArch(convOp);
    const bool isPerAxisQuantized = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(quantFilterDequantizeType);
    const bool hasReLUConsumerOrPostOp = hasReLUConsumer || IE::hasReLUPostOp(convOp);
    const bool hasNegativeQuantScales = IE::hasNegativeScales(quantFilterDequantizeType);
    const bool isProblematicPlatform = (arch == config::ArchKind::NPU37XX || arch == config::ArchKind::NPU40XX);

    if (isPerAxisQuantized && hasReLUConsumerOrPostOp && hasNegativeQuantScales && isProblematicPlatform) {
        // ReLU post-op with negative quant scales introduces inaccuracy for NPU3720 (MTL) and NPU4000 (LNL)
        // Tracking number [E#174751]
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(op->getOperand(1), filterDequantizeType);
    auto newOp = rewriter.clone(*convOp, mapper);
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newOp)) {
        rewriter.eraseOp(newOp);
        return mlir::failure();
    }
    rewriter.replaceOp(convOp, newOp->getResults());

    return mlir::success();
}

class FloatOutMatMulRewriter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    FloatOutMatMulRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
class MixedFloatInQuantWeightsWithDynamicDequantizeRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MixedFloatInQuantWeightsWithDynamicDequantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
