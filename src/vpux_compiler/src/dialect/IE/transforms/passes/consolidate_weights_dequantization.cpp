//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
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
        // The quantile type represents how the quantiles are stored by HW after the mapping, so it only makes sense to
        // be FP16 or lower. The expressed type maintains the normal precision type of the network (FP16/FP32).
        return mlir::quant::QuantileQuantizedType::get(
                storageType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
                quantileFloatType.getQuantileType(), expressedType, quantileFloatType.getQuantiles(), scale, zeroPoint,
                storageMin, storageMax);

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
        // The quantile type represents how the quantiles are stored by HW after the mapping, so it only makes sense to
        // be FP16 or lower. The expressed type maintains the normal precision type of the network (FP16/FP32).
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                storageType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
                quantileFloatType.getQuantileType(), expressedType, quantileFloatType.getQuantiles(), scales,
                SmallVector<int64_t>(scales.size(), zeroPoint), quantizedDimension.ind(), storageMin, storageMax);
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

template <typename ConcreteOp>
class WeightsDequantizeRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    WeightsDequantizeRewriter(mlir::MLIRContext* ctx, bool enableWeightsDynamicDequantization, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx),
              _enableWeightsDynamicDequantization(enableWeightsDynamicDequantization),
              _log(log.nest()) {
        this->setDebugName("WeightsDequantizeRewriter");
    }

private:
    bool isSupportedInputElemType(mlir::Type elemType) const;
    bool isSupportedShiftElemType(mlir::Type elemType) const;
    mlir::LogicalResult staticMatchAndRewrite(const IE::WeightsDequantizeStructureInfo& wdInfo, ConcreteOp origOp,
                                              mlir::PatternRewriter& rewriter) const;
    mlir::LogicalResult dynamicMatchAndRewrite(const IE::WeightsDequantizeStructureInfo& wdInfo, ConcreteOp origOp,
                                               mlir::PatternRewriter& rewriter) const;

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableWeightsDynamicDequantization;
    Logger _log;
};

template <typename ConcreteOp>
bool WeightsDequantizeRewriter<ConcreteOp>::isSupportedInputElemType(mlir::Type elemType) const {
    // The only supported weights data types are I8, U8, I4, U4, I2, U2, FP8 and NF4
    return elemType.isSignedInteger(2) || elemType.isUnsignedInteger(2) || elemType.isSignedInteger(4) ||
           elemType.isUnsignedInteger(4) || elemType.isSignedInteger(8) || elemType.isUnsignedInteger(8) ||
           isFloat8(elemType) || mlir::isa_and_nonnull<vpux::type::QuantileFloatType>(elemType);
}

template <typename ConcreteOp>
bool WeightsDequantizeRewriter<ConcreteOp>::isSupportedShiftElemType(mlir::Type elemType) const {
    // The only supported shift data type is U2
    return elemType.isUnsignedInteger(2);
}

template <typename ConcreteOp>
mlir::LogicalResult WeightsDequantizeRewriter<ConcreteOp>::staticMatchAndRewrite(
        const IE::WeightsDequantizeStructureInfo& wdInfo, ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputElemType = IE::getTrueElemType(origOp);
    if (!isSupportedInputElemType(inputElemType)) {
        _log.trace("Match failed: Input data type {0} is not supported.", inputElemType);
        return mlir::failure();
    }

    const auto dstType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();  // Usually F16/F32
    mlir::quant::QuantizedType quantElemType = createWeightsQuantizedType(inputElemType, dstType,
                                                                          /*scale*/ 1.0, /*shift*/ 0);

    // Due to limited support for multi-axis quantization and per-channel/per-group zero points, we have to use
    // DynamicDequantize. But in specific cases where shift is splat and scale is per-axis, Dequantize can be used
    // instead of DynamicDequantize
    auto useDequantize = [&](Const::ContentAttr scaleAttr, Const::ContentAttr shiftAttr) {
        auto shiftValue = 0;
        if (shiftAttr != nullptr) {
            auto shiftContent = shiftAttr.fold();
            if (!shiftContent.isSplat()) {
                return false;
            }
            shiftValue = shiftContent.getSplatValue<int64_t>();
            quantElemType = createWeightsQuantizedType(inputElemType, dstType,
                                                       /*scale*/ 1.0, shiftValue);
        }

        if (scaleAttr != nullptr) {
            if (scaleAttr.isSplat()) {
                quantElemType = createWeightsQuantizedType(inputElemType, dstType,
                                                           scaleAttr.fold().getSplatValue<double>(), shiftValue);
            } else {
                const auto scaleContent = scaleAttr.fold();
                const auto scaleShape = scaleContent.getType().getShape();
                const auto singleDimOrFail = getSingleDim(scaleShape.raw());
                if (mlir::failed(singleDimOrFail)) {
                    return false;
                }
                const auto quantDim = singleDimOrFail.value();
                const auto scaleValues = to_small_vector(scaleContent.getValues<double>());
                quantElemType =
                        createWeightsQuantizedPerAxisType(inputElemType, dstType, scaleValues, shiftValue, quantDim);
            }
        }

        return true;
    };

    auto scaleAttr = wdInfo.getStaticScaleAttr();
    auto shiftAttr = wdInfo.getStaticShiftAttr();
    auto shiftValue = wdInfo.getStaticShift();
    bool canUseDequantize = useDequantize(scaleAttr, shiftAttr);
    if (!canUseDequantize && shiftValue != nullptr) {
        auto shiftElemType = IE::getTrueElemType(shiftValue.getDefiningOp<Const::DeclareOp>());
        auto expectedShiftElemType = inputElemType;
        if (auto quantileFloatType = mlir::dyn_cast_or_null<vpux::type::QuantileFloatType>(inputElemType)) {
            expectedShiftElemType = quantileFloatType.getQuantileType();
        }
        if (!isSupportedShiftElemType(shiftElemType) || shiftElemType != expectedShiftElemType) {
            _log.trace("Match failed: The supported shift data type is U2, and must be consistent with the weights "
                       "type.");
            return mlir::failure();
        }
    }

    const auto loc = wdInfo.getLastOp()->getLoc();
    rewriter.setInsertionPointAfter(origOp);

    auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, IE::getTrueInputValue(origOp, rewriter), quantElemType)
                              .getOutput();
    if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue =
                rewriter.create<IE::TransposeOp>(loc, inputValue, nullptr, transposeOp.getOrderValueAttr()).getOutput();
    }

    mlir::Operation* dequantizeOp = nullptr;
    if (canUseDequantize) {
        dequantizeOp = rewriter.create<IE::DequantizeOp>(appendLoc(loc, "artificial_dequant"), inputValue, dstType)
                               .getOperation();
    } else {
        // Shift has been considered as zero-point if it is splat or nullptr
        auto realShiftValue = shiftAttr == nullptr || shiftAttr.isSplat()
                                      ? nullptr
                                      : IE::getTrueInputValue(shiftValue.getDefiningOp<Const::DeclareOp>(), rewriter);
        dequantizeOp = rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "artificial_dequant"), inputValue,
                                                                wdInfo.getStaticScale(), realShiftValue, dstType)
                               .getOperation();
    }

    wdInfo.getLastOp()->replaceAllUsesWith(dequantizeOp);
    wdInfo.cleanUpCurrentWdChain(rewriter);
    return mlir::success();
}

template <typename ConcreteOp>
mlir::LogicalResult WeightsDequantizeRewriter<ConcreteOp>::dynamicMatchAndRewrite(
        const IE::WeightsDequantizeStructureInfo& wdInfo, ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputElemType = IE::getTrueElemType(origOp);
    if (!isSupportedInputElemType(inputElemType)) {
        _log.trace("Match failed: Input data type {0} is not supported.", inputElemType);
        return mlir::failure();
    }

    // After DecomposeMultiZPQuantization pass, we might get the pattern
    //    OCxNGx1xui2  OCxNGx1xf16
    //    zero-point    scale
    //           \      /
    //           Multiply
    // In theory, it can be converted into a DynamicDQ, but for now the only supported modes are those requested in
    // #E-175589, so temporarily disable DynamicDQ for this case
    if (inputElemType.isUnsignedInteger(2)) {
        auto wtShape = getShape(origOp.getOutput());
        if (wtShape.back() == 1) {
            _log.trace("Match failed: Got unsupported u2 case.");
            return mlir::failure();
        }
    }

    if (auto dynamicShift = wdInfo.getDynamicShift()) {
        auto shiftElemType = IE::getTrueElemType(*dynamicShift.user_begin());
        auto expectedShiftElemType = inputElemType;
        if (auto quantileFloatType = mlir::dyn_cast_or_null<vpux::type::QuantileFloatType>(inputElemType)) {
            expectedShiftElemType = quantileFloatType.getQuantileType();
        }
        if (!isSupportedShiftElemType(shiftElemType) || shiftElemType != expectedShiftElemType) {
            _log.trace("Match failed: The supported shift data type is U2, and must be consistent with the weights "
                       "type.");
            return mlir::failure();
        }
    }

    const auto dstType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();  // Usually F16/F32
    const auto quantElemType = createWeightsQuantizedType(inputElemType, dstType, /*scale=*/1.0, /*shift*/ 0);

    const auto loc = wdInfo.getLastOp()->getLoc();
    rewriter.setInsertionPointAfter(origOp);

    mlir::Value scale = wdInfo.getDynamicScale();
    if (scale != nullptr) {
        if (auto convertOp = mlir::dyn_cast_or_null<ConcreteOp>(*scale.user_begin())) {
            scale = convertOp.getOutput();
            rewriter.setInsertionPointAfter(convertOp);
        }
    }

    auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, IE::getTrueInputValue(origOp, rewriter), quantElemType)
                              .getOutput();
    if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue =
                rewriter.create<IE::TransposeOp>(loc, inputValue, nullptr, transposeOp.getOrderValueAttr()).getOutput();
    }

    auto shift = wdInfo.hasShift() ? wdInfo.getDynamicShift() : nullptr;
    auto dynamicDequantizeOp = rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "artificial_dyn_dequant"),
                                                                        inputValue, scale, shift, dstType);
    wdInfo.getLastOp()->replaceAllUsesWith(dynamicDequantizeOp);
    wdInfo.cleanUpCurrentWdChain(rewriter);
    return mlir::success();
}

template <typename ConcreteOp>
mlir::LogicalResult WeightsDequantizeRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

    // Match the weights dequantize structure once...
    const auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
    if (mlir::failed(maybeWdInfo)) {
        _log.trace("Failed to match WeightsDequantize structure.");
        return mlir::failure();
    }
    const auto& wdInfo = maybeWdInfo.value();
    if (!wdInfo.hasScale() && !wdInfo.hasShift()) {
        // For now we don't want to rewrite single Convert's with no scale or shift. A later pass,
        // FuseConvertWithQuantize, may handle some of them more efficiently while the remaining ones get converted
        // to QuantizeCast->Dequantize afterwards.
        _log.trace("Match failed: Missing both scale and shift.");
        return mlir::failure();
    }
    if (wdInfo.hasScale() && wdInfo.hasShift()) {
        if (!((wdInfo.getDynamicScale() != nullptr && wdInfo.getDynamicShift() != nullptr) ||
              (wdInfo.getStaticScale() != nullptr && wdInfo.getStaticShift() != nullptr))) {
            _log.trace("Match failed: The forms of scale and shift need to be consistent.");
            return mlir::failure();
        }
    }

    auto quantParamsAsInput = wdInfo.getDynamicScale() != nullptr || wdInfo.getDynamicShift() != nullptr;
    // ...then split depending on dynamic/static quantization.
    if (quantParamsAsInput) {
        if (mlir::isa<Const::DeclareOp>(origOp)) {
            _log.trace("Match failed: Got dynamic scale but weights is a constant.");
            return mlir::failure();
        }

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
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void initializeFromOptions();

    void safeRunOnFunc() final;

private:
    bool _enableWeightsDynamicDequantization = false;
};

mlir::LogicalResult ConsolidateWeightsDequantizationPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
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
    patterns.add<WeightsDequantizeRewriter<IE::ConvertOp>>(&ctx, _enableWeightsDynamicDequantization, _log);
    patterns.add<WeightsDequantizeRewriter<Const::DeclareOp>>(&ctx, _enableWeightsDynamicDequantization, _log);

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
