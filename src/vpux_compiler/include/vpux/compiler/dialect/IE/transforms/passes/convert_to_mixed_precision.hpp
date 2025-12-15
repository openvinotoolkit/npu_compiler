//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/IRMapping.h>

namespace vpux {
namespace IE {
using CheckPostOpFunctor = llvm::function_ref<bool(IE::LayerWithPostOpInterface layerWithPostOp,
                                                   bool isPerAxisQuantizedOutput, bool isFloatInput)>;

using SupportedMixedPrecisionFunctor = std::function<bool(mlir::Operation*, const bool isPReLUSupported, Logger log)>;

class FloatOutConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FloatOutConvRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                         Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convolutionOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

class FloatOutGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    FloatOutGroupConvRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                              Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

class FloatOutTransposedConvRewriter final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    FloatOutTransposedConvRewriter(mlir::MLIRContext* ctx,
                                   const SupportedMixedPrecisionFunctor& isMixPrecisionSupported, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp transposedConvOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
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
    FloatOutAddRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                        const bool allowDifferentScales, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _allowDifferentScales(allowDifferentScales),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    const bool _allowDifferentScales;
    Logger _log;
};

class QuantizeWithNCERewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeWithNCERewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                            CheckPostOpFunctor checkPostOp, bool isPerAxesSupported, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _checkPostOp(checkPostOp),
              _isPerAxesSupported(isPerAxesSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    CheckPostOpFunctor _checkPostOp;
    bool _isPerAxesSupported;
    Logger _log;
};

template <typename ConcreteOp>
class MixedFloatInQuantWeightsRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MixedFloatInQuantWeightsRewriter(mlir::MLIRContext* ctx,
                                     const SupportedMixedPrecisionFunctor& isMixPrecisionSupported, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _isMixPrecisionSupported(isMixPrecisionSupported), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MixedFloatInQuantWeightsRewriter<ConcreteOp>::matchAndRewrite(
        ConcreteOp convOp, mlir::PatternRewriter& rewriter) const {
    if (!_isMixPrecisionSupported(convOp, true, _log)) {
        return mlir::failure();
    }

    auto op = convOp.getOperation();

    const auto dequantizeType = IE::findQuantizedInput(op->getOperand(0), false);
    const auto filterDequantizeType = IE::findQuantizedInput(op->getOperand(1), true);

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
        if (mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(quantType)) {
            mlir::Type quantileType =
                    mlir::isa<mlir::quant::QuantileQuantizedType>(quantType)
                            ? mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(quantType).getQuantileType()
                            : mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(quantType).getQuantileType();

            if (auto intType = mlir::dyn_cast<mlir::IntegerType>(quantileType)) {
                return intType.isSigned();
            } else {
                // quantileType is a float type
                return true;
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
    FloatOutMatMulRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                           Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx), _isMixPrecisionSupported(isMixPrecisionSupported), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
