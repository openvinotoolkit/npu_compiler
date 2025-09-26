//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDYNAMICDEQUANTIZETODEQUANTIZE
#define GEN_PASS_DEF_CONVERTDYNAMICDEQUANTIZETODEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//                Input2(512x128xi4:f16)    Input3(512x1xf16)
//                              \           /
//        Input1(1x128xf16)     IE.DynamicDequantize(512x128xf16)
//                        \       /
//                    IE.FullyConnected(1x512xf16)

// To:

//                Input2(512x128xi4:f16)
//                               |
//        Input1(1x128xf16)    IE.Dequantize(512x128xf16)
//                        \       /                      Input3(512x1xf16)
//                    IE.FullyConnected(1x512xf16)              |
//                                    \                IE.Transpose(1x512xf16)
//                                     \               /
//                                 IE.Multiply(1x512xf16)

class ConvertDynamicDequantizeToDequantize final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    ConvertDynamicDequantizeToDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
        setDebugName("ConvertDynamicDequantizeToDequantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isOptimizableDynamicDequantizeOp(IE::DynamicDequantizeOp origOp) {
    if (origOp.getZp() != nullptr) {
        return false;
    }

    if (!origOp->hasOneUse()) {
        return false;
    }

    if (IE::findAxes(origOp).size() > 1) {
        return false;
    }

    auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputType.getElementType());
    if (uniformType == nullptr) {
        return false;
    }
    const auto scale = uniformType.getScale();
    if (!isDoubleEqual(scale, 1.0f)) {
        return false;
    }

    auto quantStorageType = uniformType.getStorageType();
    auto isNF4Quantized = [](mlir::Type type) -> bool {
        const auto qType = mlir::cast<mlir::quant::QuantizedType>(type);
        const auto bitWidth = qType.getStorageTypeIntegralWidth();
        return (bitWidth == 4) && mlir::isa<mlir::quant::QuantileQuantizedType>(qType);
    }(inputType.getElementType());

    if (isNF4Quantized || quantStorageType.isInteger(CHAR_BIT) || vpux::isFloat8(quantStorageType)) {
        auto quantCastOp = origOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
        if (quantCastOp == nullptr) {
            return false;
        }
    }
    return true;
}

// Reshape from 1x32x64 to 32x64 or from 32x1x64 to 32x64
bool isSqueezeLikeReshape(IE::ReshapeOp reshapeOp) {
    SmallVector<int64_t> inShape(getShape(reshapeOp.getInput()).raw());
    SmallVector<int64_t> outShape(getShape(reshapeOp.getOutput()).raw());
    if (outShape.size() >= inShape.size()) {
        return false;
    }

    // remove all the '1' in the shape
    inShape.erase(std::remove(inShape.begin(), inShape.end(), 1), inShape.end());
    outShape.erase(std::remove(outShape.begin(), outShape.end(), 1), outShape.end());

    return inShape == outShape;
}

std::optional<IE::FullyConnectedOp> isDirectConnected(IE::DynamicDequantizeOp origOp) {
    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*origOp->user_begin());
    if (fcOp && fcOp.getWeights() == origOp.getOutput()) {
        const auto fcOutShape = getShape(fcOp.getOutput()).raw();
        const auto scaleShape = getShape(origOp.getScale()).raw();
        if (getShape(origOp.getScale()).totalSize() == 1) {
            return fcOp;
        }

        if (fcOutShape.back() == scaleShape.front()) {
            return fcOp;
        }
    }

    return std::nullopt;
}

std::optional<IE::FullyConnectedOp> isReshapeTranspose(IE::DynamicDequantizeOp origOp) {
    auto reshapeOp = mlir::dyn_cast<IE::ReshapeOp>(*origOp->user_begin());
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse() || !isSqueezeLikeReshape(reshapeOp)) {
        return std::nullopt;
    }

    auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*reshapeOp->user_begin());
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return std::nullopt;
    }

    const auto maybeMap = transposeOp.getOrderValue();
    if (!maybeMap.has_value()) {
        return std::nullopt;
    }
    const SmallVector<unsigned> expectedMap = {1, 0};
    if (maybeMap.value() != mlir::AffineMap::getPermutationMap(expectedMap, origOp.getContext())) {
        return std::nullopt;
    }

    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*transposeOp->user_begin());
    if (fcOp && fcOp.getWeights() == transposeOp.getOutput()) {
        const auto fcOutShape = getShape(fcOp.getOutput()).raw();
        const auto scaleShape = getShape(origOp.getScale()).raw();
        const auto dequantOutShape = getShape(origOp.getOutput()).raw();
        // no need to check if scale is one single data
        if (getShape(origOp.getScale()).totalSize() == 1) {
            return fcOp;
        }

        // check from the end due to the transpose permutation is {1, 0}
        for (auto i : llvm::reverse(irange(dequantOutShape.size()))) {
            if (dequantOutShape[i] == 1) {
                continue;
            }
            if (dequantOutShape[i] == fcOutShape.back() && scaleShape[i] == fcOutShape.back()) {
                return fcOp;
            }
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<IE::FullyConnectedOp> isTransposeReshape(IE::DynamicDequantizeOp origOp) {
    auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*origOp->user_begin());
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return std::nullopt;
    }

    const auto maybeMap = transposeOp.getOrderValue();
    if (!maybeMap.has_value()) {
        return std::nullopt;
    }

    auto reshapeOp = mlir::dyn_cast<IE::ReshapeOp>(*transposeOp->user_begin());
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse() || !isSqueezeLikeReshape(reshapeOp)) {
        return std::nullopt;
    }

    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*reshapeOp->user_begin());
    if (fcOp && fcOp.getWeights() == reshapeOp.getOutput()) {
        const auto fcOutShape = getShape(fcOp.getOutput()).raw();
        const auto scaleShape = getShape(origOp.getScale()).raw();
        const auto dequantOutShape = getShape(origOp.getOutput()).raw();
        const auto transposeOutShape = getShape(transposeOp.getOutput()).raw();
        if (getShape(origOp.getScale()).totalSize() == 1) {
            return fcOp;
        }

        // check from the front
        for (auto i : irange(transposeOutShape.size())) {
            if (transposeOutShape[i] == 1) {
                continue;
            }
            if (transposeOutShape[i] == fcOutShape.back()) {
                // get the correct dim before transpose
                const auto permutation = DimsOrder::fromAffineMap(maybeMap.value()).toPermutation();
                const auto dim = permutation[i];
                if (dequantOutShape[dim.ind()] == fcOutShape.back() && scaleShape[dim.ind()] == fcOutShape.back()) {
                    return fcOp;
                }
            }
            return std::nullopt;
        }
    }

    return std::nullopt;
}

std::optional<IE::FullyConnectedOp> isReshapeOnly(IE::DynamicDequantizeOp origOp) {
    auto reshapeOp = mlir::dyn_cast<IE::ReshapeOp>(*origOp->user_begin());
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse() || !isSqueezeLikeReshape(reshapeOp)) {
        return std::nullopt;
    }

    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*reshapeOp->user_begin());
    if (fcOp && fcOp.getWeights() == reshapeOp.getOutput()) {
        const auto fcOutShape = getShape(fcOp.getOutput()).raw();
        const auto scaleShape = getShape(origOp.getScale()).raw();
        const auto dequantOutShape = getShape(origOp.getOutput()).raw();
        if (getShape(origOp.getScale()).totalSize() == 1) {
            return fcOp;
        }

        // check from the front
        for (auto i : irange(dequantOutShape.size())) {
            if (dequantOutShape[i] == 1) {
                continue;
            }

            if (dequantOutShape[i] == fcOutShape.back() && scaleShape[i] == fcOutShape.back()) {
                return fcOp;
            }

            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<IE::FullyConnectedOp> isTransposeOnly(IE::DynamicDequantizeOp origOp) {
    auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*origOp->user_begin());
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return std::nullopt;
    }

    const auto maybeMap = transposeOp.getOrderValue();
    if (!maybeMap.has_value()) {
        return std::nullopt;
    }
    const SmallVector<unsigned> expectedMap = {1, 0};
    if (maybeMap.value() != mlir::AffineMap::getPermutationMap(expectedMap, origOp.getContext())) {
        return std::nullopt;
    }

    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*transposeOp->user_begin());
    if (fcOp && fcOp.getWeights() == transposeOp.getOutput()) {
        const auto fcOutShape = getShape(fcOp.getOutput()).raw();
        const auto scaleShape = getShape(origOp.getScale()).raw();

        if (fcOutShape.back() == scaleShape.back()) {
            return fcOp;
        }
    }
    return std::nullopt;
}

// We need to check the scale carefully or we will do a invalid convert.
// Like for below case is legal:
// %0 = IE.DynamicDequantize(%arg0, %arg1) : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
// %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
// below case is illegal:
// %0 = IE.DynamicDequantize(%arg0, %arg1) : tensor<4096x4096x!qElemType>, tensor<1x4096xf16> -> tensor<4096x4096xf16>
// %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
// the only difference is dynamicDequant scale is tensor<4096x1xf16> Vs tensor<1x4096xf16>
// the reason is we only handle the case that dynamicDequant is the weights of FC, FC weights shape is always <AxB>.
// B will multiply and merge with first input. A is remaining dimension, so the scale can only happend on A. The same
// logic is needed for tranpose, reshape or mixed case.

std::optional<IE::FullyConnectedOp> tryToFindValidFCPattern(IE::DynamicDequantizeOp origOp) {
    auto isValidPattern = isDirectConnected(origOp);
    if (isValidPattern.has_value()) {
        return isValidPattern.value();
    }

    isValidPattern = isReshapeTranspose(origOp);
    if (isValidPattern.has_value()) {
        return isValidPattern.value();
    }

    isValidPattern = isTransposeReshape(origOp);
    if (isValidPattern.has_value()) {
        return isValidPattern.value();
    }

    isValidPattern = isReshapeOnly(origOp);
    if (isValidPattern.has_value()) {
        return isValidPattern.value();
    }

    isValidPattern = isTransposeOnly(origOp);
    if (isValidPattern.has_value()) {
        return isValidPattern.value();
    }

    return std::nullopt;
}

mlir::LogicalResult ConvertDynamicDequantizeToDequantize::matchAndRewrite(IE::DynamicDequantizeOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    if (!isOptimizableDynamicDequantizeOp(origOp)) {
        return matchFailed(rewriter, origOp, "not a valid dynamic dequantize op");
    }

    auto isValidPattern = tryToFindValidFCPattern(origOp);
    if (!isValidPattern.has_value()) {
        return matchFailed(rewriter, origOp, "not a valid FC pattern");
    }
    auto fcOp = isValidPattern.value();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto uniformType = mlir::cast<mlir::quant::UniformQuantizedType>(inputType.getElementType());
    auto quantStorageType = uniformType.getStorageType();
    // reshape the scale
    const auto fcOutShape = getShape(fcOp.getOutput()).raw();
    const auto scaleSize = getShape(origOp.getScale()).totalSize();
    const SmallVector<int64_t> outShape{1, scaleSize == 1 ? 1 : fcOutShape.back()};
    const auto outShapeAttr = getIntArrayAttr(origOp->getContext(), outShape);
    auto scale = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "_reshape_scale"), origOp.getScale(), nullptr, false,
                                                outShapeAttr)
                         .getOutput();
    auto isNF4Quantized = [](mlir::Type type) -> bool {
        const auto qType = mlir::cast<mlir::quant::QuantizedType>(type);
        const auto bitWidth = qType.getStorageTypeIntegralWidth();
        return (bitWidth == 4) && mlir::isa<mlir::quant::QuantileQuantizedType>(qType);
    }(inputType.getElementType());

    if (isNF4Quantized || quantStorageType.isInteger(CHAR_BIT) || vpux::isFloat8(quantStorageType)) {
        // Rescale scales by descaler and weights by rescaler to scale down the I8/FP8/NF4 weights into I4 range to
        // avoid overflow in the following FullyConnected. See details in #E161479

        _log.trace("Found DynamicDequantizeOp for I8/FP8/NF4, try to rescale it into I4 range");
        const float minI4 = -8.f;
        const float maxI4 = 7.f;

        float minComputationValue = uniformType.getStorageTypeMin();
        float maxComputationValue = uniformType.getStorageTypeMax();
        if (isNF4Quantized) {
            const auto quantLUT = mlir::cast<mlir::quant::QuantileQuantizedType>(uniformType).getQuantiles();
            minComputationValue = *std::min_element(quantLUT.begin(), quantLUT.end());
            maxComputationValue = *std::max_element(quantLUT.begin(), quantLUT.end());
        }
        auto rescaleForMin = minI4 / minComputationValue;
        auto rescaleForMax = maxI4 / maxComputationValue;
        _log.trace("rescaleForMin = minI4/minComputationValue = {0}/{1} = {2};", minI4, minComputationValue,
                   rescaleForMin);
        _log.trace("rescaleForMax = maxI4/maxComputationValue = {0}/{1} = {2};", maxI4, maxComputationValue,
                   rescaleForMax);
        auto rescaler = [&]() -> float {
            VPUX_THROW_WHEN((rescaleForMin <= 0.f) && (rescaleForMax <= 0.f),
                            "Both rescaleForMin and rescaleForMax are negative, which is not expected. "
                            "rescaleForMin: {0}, rescaleForMax: {1}",
                            rescaleForMin, rescaleForMax);
            if (rescaleForMin <= 0.f) {
                return rescaleForMax;
            } else if (rescaleForMax <= 0.f) {
                return rescaleForMin;
            } else {
                return std::min(rescaleForMin, rescaleForMax);
            }
        }();
        if (rescaler >= 1.f) {
            _log.trace("rescaler = {0} >= 1.f, no need to rescale", rescaler);
        } else {
            const auto descaler = 1.f / rescaler;
            _log.trace("rescaler = {0} < 1.f, rescaling with descaler: {1}", rescaler, descaler);
            auto quantCastOp = origOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
            mlir::quant::QuantizedType quantTypeRescaled = nullptr;
            if (isNF4Quantized) {
                const auto quantizedType = mlir::cast<mlir::quant::QuantileQuantizedType>(uniformType);
                const auto quantLUT = quantizedType.getQuantiles();
                quantTypeRescaled = mlir::quant::QuantileQuantizedType::get(
                        quantizedType.getFlags(), quantizedType.getStorageType(), quantizedType.getQuantileType(),
                        quantizedType.getExpressedType(), quantLUT, rescaler, /*zp=*/0,
                        /*min=*/quantizedType.getStorageTypeMin(), /*max=*/quantizedType.getStorageTypeMax());
            } else {
                quantTypeRescaled = mlir::quant::UniformQuantizedType::get(
                        uniformType.getFlags(), quantStorageType, uniformType.getExpressedType(), rescaler,
                        /*zp=*/0, /*min=*/uniformType.getStorageTypeMin(),
                        /*max=*/uniformType.getStorageTypeMax());
            }
            auto quantCastOpRescaled =
                    rewriter.create<IE::QuantizeCastOp>(appendLoc(origOp->getLoc(), "_quant_cast_i8_with_rescaler"),
                                                        quantCastOp.getInput(), quantTypeRescaled);
            rewriter.replaceOp(quantCastOp, quantCastOpRescaled.getOutput());

            const auto descalerBaseType =
                    mlir::RankedTensorType::get({1}, mlir::Float16Type::get(uniformType.getContext()));
            const auto descalerBaseAttr =
                    Const::createConstContent(descalerBaseType, ArrayRef(vpux::type::float16(descaler)));
            const auto descalerContentAttr = Const::ContentAttr::get(descalerBaseAttr);
            auto descalerConstAttr = descalerContentAttr.transform().get();
            const auto descalerType = mlir::cast<vpux::NDTypeInterface>(descalerContentAttr.getType());
            auto descalerConstOp = rewriter.create<Const::DeclareOp>(appendLoc(origOp->getLoc(), "_descaler"),
                                                                     descalerType, std::move(descalerConstAttr));

            auto multiplyWithRescaler = rewriter.create<IE::MultiplyOp>(
                    appendLoc(origOp->getLoc(), "_descale_scaler"), scale, descalerConstOp,
                    IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
            scale = multiplyWithRescaler;
        }
    }

    // insert a multiply post FC
    rewriter.setInsertionPointAfter(fcOp);
    auto multiply =
            rewriter.create<IE::MultiplyOp>(appendLoc(origOp->getLoc(), "_post_multiply"), fcOp.getOutput(), scale,
                                            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    fcOp.getOutput().replaceUsesWithIf(multiply.getOutput(), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() != multiply;
    });

    // replace dynamic dequantize with dequantize
    rewriter.setInsertionPointAfter(origOp);
    auto dequantizeOp = rewriter.create<IE::DequantizeOp>(origOp->getLoc(), origOp.getInput(), origOp.getDstElemType());
    rewriter.replaceOp(origOp, dequantizeOp.getOutput());

    _log.trace("Convert dynamic dequantize to dequantize successful");
    return mlir::success();
}

//
// ConvertDynamicDequantizeToDequantizePass
//

class ConvertDynamicDequantizeToDequantizePass final :
        public IE::impl::ConvertDynamicDequantizeToDequantizeBase<ConvertDynamicDequantizeToDequantizePass> {
public:
    explicit ConvertDynamicDequantizeToDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertDynamicDequantizeToDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertDynamicDequantizeToDequantize>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDynamicDequantizeToDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDynamicDequantizeToDequantizePass(Logger log) {
    return std::make_unique<ConvertDynamicDequantizeToDequantizePass>(log);
}
