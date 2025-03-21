//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONSOLIDATEWEIGHTSDEQUANTIZATION
#define GEN_PASS_DEF_CONSOLIDATEWEIGHTSDEQUANTIZATION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

namespace vpux {

mlir::quant::QuantizedType createWeightsQuantizedType(mlir::Type weightsElemType, mlir::Type expressedType,
                                                      double scale, int64_t zeroPoint) {
    const auto [storageMin, storageMax, storageType] = getStorageParams(weightsElemType);
    if (const auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(weightsElemType)) {
        const auto ctx = weightsElemType.getContext();

        // The quantile type represents how the quantiles are stored by HW after the mapping, so it only makes sense to
        // be FP16 or lower. The expressed type maintains the normal precision type of the network (FP16/FP32).
        return mlir::quant::QuantileQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, storageType, /*quantileType=*/mlir::Float16Type::get(ctx),
                expressedType, quantileFloatType.getQuantiles(), scale, zeroPoint, storageMin, storageMax);

    } else {
        return mlir::quant::UniformQuantizedType::get(
                weightsElemType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
                expressedType, scale, zeroPoint, storageMin, storageMax);
    }
}

mlir::quant::QuantizedType createWeightsQuantizedPerAxisType(mlir::Type weightsElemType, mlir::Type expressedType,
                                                             ArrayRef<double> scales, int64_t zeroPoint,
                                                             Dim quantizedDimension) {
    const auto [storageMin, storageMax, storageType] = getStorageParams(weightsElemType);
    if (const auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(weightsElemType)) {
        const auto ctx = weightsElemType.getContext();

        // The quantile type represents how the quantiles are stored by HW after the mapping, so it only makes sense to
        // be FP16 or lower. The expressed type maintains the normal precision type of the network (FP16/FP32).
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                mlir::quant::QuantizationFlags::Signed, storageType, /*quantileType=*/mlir::Float16Type::get(ctx),
                expressedType, quantileFloatType.getQuantiles(), scales, {zeroPoint}, quantizedDimension.ind(),
                storageMin, storageMax);

    } else {
        return mlir::quant::UniformQuantizedPerAxisType::get(
                weightsElemType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
                expressedType, scales, SmallVector(scales.size(), zeroPoint), quantizedDimension.ind(), storageMin,
                storageMax);
    }
}

mlir::FailureOr<Dim> getSingleDim(ArrayRef<int64_t> shape) {
    const auto dimIt = std::find_if(shape.begin(), shape.end(), [](const auto d) {
        return d != 1;
    });
    if (dimIt == shape.end()) {
        return mlir::failure();
    }

    const auto hasSecondDim = std::any_of(dimIt + 1, shape.end(), [](const auto d) {
        return d != 1;
    });
    if (hasSecondDim) {
        return mlir::failure();
    }
    return Dim(std::distance(shape.begin(), dimIt));
}

class WeightsDequantizeRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    WeightsDequantizeRewriter(mlir::MLIRContext* ctx, bool enableWeightsDynamicDequantization, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx),
              _enableWeightsDynamicDequantization(enableWeightsDynamicDequantization),
              _log(log.nest()) {
        setDebugName("WeightsDequantizeRewriter");
    }

private:
    mlir::LogicalResult staticMatchAndRewrite(const IE::WeightsDequantizeStructureInfo& wdInfo, IE::ConvertOp origOp,
                                              mlir::PatternRewriter& rewriter) const;
    mlir::LogicalResult dynamicMatchAndRewrite(const IE::WeightsDequantizeStructureInfo& wdInfo, IE::ConvertOp origOp,
                                               mlir::PatternRewriter& rewriter) const;

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableWeightsDynamicDequantization;
    Logger _log;
};

mlir::LogicalResult WeightsDequantizeRewriter::staticMatchAndRewrite(const IE::WeightsDequantizeStructureInfo& wdInfo,
                                                                     IE::ConvertOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    if (wdInfo.getScale() == nullptr && wdInfo.getShift() == nullptr) {
        // For now we don't want to rewrite single Convert's with no scale or shift. A later pass,
        // FuseConvertWithQuantize, may handle some of them more efficiently while the remaining ones get converted to
        // QuantizeCast->Dequantize afterwards.
        _log.trace("Match failed: Missing both scale and shift.");
        return mlir::failure();
    }

    const auto inputElemType = IE::getTrueElemTypeOfWeights(origOp);

    // The only supported weights data type are I8, U8, I4, U4, FP8 and NF4
    if (!inputElemType.isInteger(4) && !inputElemType.isInteger(8) && !isFloat8(inputElemType) &&
        !mlir::isa<vpux::type::QuantileFloatType>(inputElemType)) {
        _log.trace("Match failed: Input data type {0} is not supported.", inputElemType);
        return mlir::failure();
    }

    int64_t shiftValue = 0;
    if (const auto shiftAttr = wdInfo.getShift()) {
        if (!shiftAttr.isSplat()) {
            _log.trace("Match failed: Shift is not scalar.");
            return mlir::failure();
        }
        shiftValue = shiftAttr.fold().getSplatValue<int64_t>();
    }

    const auto dstType = origOp.getDstElemType();  // Usually F16/F32

    mlir::quant::QuantizedType quantElemType = nullptr;
    if (const auto scaleAttr = wdInfo.getScale()) {
        if (scaleAttr.isSplat()) {
            const auto scaleValue = scaleAttr.fold().getSplatValue<double>();
            quantElemType = createWeightsQuantizedType(inputElemType, dstType, scaleValue, shiftValue);

        } else {
            const auto scaleContent = scaleAttr.fold();
            const auto scaleShape = scaleContent.getType().getShape();
            const auto singleDimOrFail = getSingleDim(scaleShape.raw());
            if (mlir::failed(singleDimOrFail)) {
                // Support will be added in the future.
                _log.trace("Match failed: Got group quantization scale.");
                return mlir::failure();
            }
            const auto quantDim = singleDimOrFail.value();

            const auto inputShape = mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getShape();
            if (inputShape[quantDim] != scaleShape[quantDim]) {
                _log.trace("Match failed: Scale shape: {0} doesn't match the input shape: {1} on dim: {2}.", scaleShape,
                           inputShape, quantDim);
                return mlir::failure();
            }

            const auto scaleValues = to_small_vector(scaleContent.getValues<double>());
            quantElemType =
                    createWeightsQuantizedPerAxisType(inputElemType, dstType, scaleValues, shiftValue, quantDim);
        }

    } else {
        quantElemType = createWeightsQuantizedType(inputElemType, dstType, /*scale=*/1.0, shiftValue);
    }

    const auto loc = wdInfo.getLastOp()->getLoc();
    rewriter.setInsertionPointAfter(origOp);

    auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, origOp.getInput(), quantElemType).getOutput();
    if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue =
                rewriter.create<IE::TransposeOp>(loc, inputValue, nullptr, transposeOp.getOrderValueAttr()).getOutput();
    }

    auto dequantizeOp = rewriter.create<IE::DequantizeOp>(appendLoc(loc, "artificial_dequant"), inputValue,
                                                          origOp.getDstElemType());

    wdInfo.getLastOp()->replaceAllUsesWith(dequantizeOp);
    wdInfo.cleanUpCurrentWdChain(rewriter);
    return mlir::success();
}

mlir::LogicalResult WeightsDequantizeRewriter::dynamicMatchAndRewrite(const IE::WeightsDequantizeStructureInfo& wdInfo,
                                                                      IE::ConvertOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    // The only supported weights data type for dynamic quantize is I4, U4 and I8
    const auto inputElemType = IE::getTrueElemTypeOfWeights(origOp);
    if (!inputElemType.isInteger(4) && !inputElemType.isSignedInteger(8) &&
        !mlir::isa_and_nonnull<vpux::type::QuantileFloatType>(inputElemType)) {
        _log.trace("Match failed: Input data type {0} is not supported.", inputElemType);
        return mlir::failure();
    }

    int64_t shiftValue = 0;
    const auto shift = wdInfo.getShift();
    if (shift != nullptr) {
        if (!shift.isSplat()) {
            _log.trace("Match failed: Shift is not scalar.");
            return mlir::failure();
        }
        shiftValue = shift.fold().getSplatValue<int64_t>();
    }

    const auto quantElemType =
            createWeightsQuantizedType(inputElemType, origOp.getDstElemType(), /*scale=*/1.0, shiftValue);

    const auto loc = wdInfo.getLastOp()->getLoc();
    rewriter.setInsertionPointAfter(origOp);

    auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, origOp.getInput(), quantElemType).getOutput();
    if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue =
                rewriter.create<IE::TransposeOp>(loc, inputValue, nullptr, transposeOp.getOrderValueAttr()).getOutput();
    }
    auto dynamicDequantizeOp =
            rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "artificial_dyn_dequant"), inputValue,
                                                     wdInfo.getDynamicScale(), nullptr, origOp.getDstElemType());

    wdInfo.getLastOp()->replaceAllUsesWith(dynamicDequantizeOp);
    wdInfo.cleanUpCurrentWdChain(rewriter);
    return mlir::success();
}

mlir::LogicalResult WeightsDequantizeRewriter::matchAndRewrite(IE::ConvertOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

    // Match the weights dequantize structure once...
    const auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
    if (mlir::failed(maybeWdInfo)) {
        _log.trace("Failed to match WeightsDequantize structure.");
        return mlir::failure();
    }
    const auto& wdInfo = maybeWdInfo.value();

    // ...then split depending on dynamic/static quantization.
    if (wdInfo.getDynamicScale() != nullptr) {
        if (!_enableWeightsDynamicDequantization) {
            _log.trace("Match failed: Got dynamic scale but dynamic dequantization is disabled.");
            return mlir::failure();
        }

        return dynamicMatchAndRewrite(wdInfo, origOp, rewriter);
    } else {
        return staticMatchAndRewrite(wdInfo, origOp, rewriter);
    }
}

class ConsolidateWeightsDequantizationPass final :
        public IE::impl::ConsolidateWeightsDequantizationBase<ConsolidateWeightsDequantizationPass> {
public:
    ConsolidateWeightsDequantizationPass() = default;
    explicit ConsolidateWeightsDequantizationPass(const IE::LowPrecisionTransformOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());

        Base::copyOptionValuesFrom(options);
        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void initializeFromOptions();

    void safeRunOnFunc() final;

private:
    bool _enableWeightsDynamicDequantization = false;
};

mlir::LogicalResult ConsolidateWeightsDequantizationPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void ConsolidateWeightsDequantizationPass::initializeFromOptions() {
    if (enableWeightsDynamicDequantization.hasValue()) {
        _enableWeightsDynamicDequantization = enableWeightsDynamicDequantization.getValue();
    }
}

void ConsolidateWeightsDequantizationPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    IE::ConvertOp::getCanonicalizationPatterns(patterns, &ctx);  // Ensures Convert chains are folded
    patterns.add<WeightsDequantizeRewriter>(&ctx, _enableWeightsDynamicDequantization, _log);

    auto config = getDefaultGreedyRewriteConfig();
    config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace vpux

//
// createConsolidateWeightsDequantizationPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConsolidateWeightsDequantizationPass() {
    return std::make_unique<ConsolidateWeightsDequantizationPass>();
}

std::unique_ptr<mlir::Pass> vpux::IE::createConsolidateWeightsDequantizationPass(
        const IE::LowPrecisionTransformOptions& options, Logger log) {
    return std::make_unique<ConsolidateWeightsDequantizationPass>(options, log);
}
