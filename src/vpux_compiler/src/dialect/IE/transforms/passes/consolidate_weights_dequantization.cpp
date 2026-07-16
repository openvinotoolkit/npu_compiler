//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONSOLIDATEWEIGHTSDEQUANTIZATION
#define GEN_PASS_DEF_CONSOLIDATEWEIGHTSDEQUANTIZATION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

namespace vpux {

mlir::quant::QuantizedType createWeightsQuantizedType(mlir::Type weightsElemType, mlir::Type expressedType,
                                                      double scale, int64_t zeroPoint) {
    const auto storageParams = getStorageParams(weightsElemType);
    VPUX_THROW_WHEN(mlir::failed(storageParams), "Unsupported quantized element type '{0}'", weightsElemType);

    const auto [storageMin, storageMax, storageType] = *storageParams;
    if (const auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(weightsElemType)) {
        // Use QuantileType directly as storage type
        // This represents quantile-backed quantization as quant.uniform with QuantileType storage

        const auto flags = quantileType.shouldDefaultToSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
        return mlir::quant::UniformQuantizedType::get(flags, quantileType, expressedType, scale, zeroPoint, storageMin,
                                                      storageMax);

    } else {
        return mlir::quant::UniformQuantizedType::get(
                storageType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
                expressedType, scale, zeroPoint, storageMin, storageMax);
    }
}

mlir::quant::QuantizedType createWeightsQuantizedPerAxisType(mlir::Type weightsElemType, mlir::Type expressedType,
                                                             ArrayRef<double> scales, int64_t zeroPoint,
                                                             Dim quantizedDimension) {
    const auto storageParams = getStorageParams(weightsElemType);
    VPUX_THROW_WHEN(mlir::failed(storageParams), "Unsupported quantized element type '{0}'", weightsElemType);
    const auto [storageMin, storageMax, storageType] = *storageParams;
    if (const auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(weightsElemType)) {
        // Use QuantileType directly as storage type.
        // This represents quantile-backed quantization as quant.uniform with QuantileType storage
        const auto flags = quantileType.shouldDefaultToSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
        return mlir::quant::UniformQuantizedPerAxisType::get(flags, quantileType, expressedType, scales,
                                                             SmallVector<int64_t>(scales.size(), zeroPoint),
                                                             quantizedDimension.ind(), storageMin, storageMax);
    } else {
        return mlir::quant::UniformQuantizedPerAxisType::get(
                storageType.isUnsignedInteger() ? 0 : mlir::quant::QuantizationFlags::Signed, storageType,
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
class WeightsDequantizeRewriter final : public mlir::OpRewritePattern<ConcreteOp>, public IInitializableRewriter {
public:
    WeightsDequantizeRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1,
                              bool enableWeightsDynamicDequantization = false)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefit),
              _log(log.nest()),
              _enableWeightsDynamicDequantization(enableWeightsDynamicDequantization) {
        this->setDebugName("WeightsDequantizeRewriter");
    }

    void initialize(mlir::func::FuncOp funcOp) override;

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
    Logger _log;
    bool _enableWeightsDynamicDequantization;
};

template <typename ConcreteOp>
void WeightsDequantizeRewriter<ConcreteOp>::initialize(mlir::func::FuncOp funcOp) {
    auto module = getModuleOp(funcOp);
    _enableWeightsDynamicDequantization = config::hasEnableWeightsDynamicDequantization(module);
}

// Match signed or unsigned integer of the given width, but NOT signless `iN`.
// Signless integer weights cannot be lowered by createWeightsQuantizedType:
// getStorageParams() rejects signless for all widths (see utils/quantization.cpp).
// Contract is covered by test @NotConvertToDequantizeForSignlessType.
static bool isSignedOrUnsignedIntOfWidth(mlir::Type elemType, unsigned width) {
    return elemType.isSignedInteger(width) || elemType.isUnsignedInteger(width);
}

template <typename ConcreteOp>
bool WeightsDequantizeRewriter<ConcreteOp>::isSupportedInputElemType(mlir::Type elemType) const {
    return isSignedOrUnsignedIntOfWidth(elemType, 2) || isSignedOrUnsignedIntOfWidth(elemType, 4) ||
           isSignedOrUnsignedIntOfWidth(elemType, 8) || isSignedOrUnsignedIntOfWidth(elemType, 16) ||
           isLowFpType(elemType) || mlir::isa_and_nonnull<vpux::type::QuantileType>(elemType);
}

template <typename ConcreteOp>
bool WeightsDequantizeRewriter<ConcreteOp>::isSupportedShiftElemType(mlir::Type elemType) const {
    // Supported shift data types: U2/I2, U8/I8.
    return elemType.isUnsignedInteger(2) || isSignedOrUnsignedIntOfWidth(elemType, 8);
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
        if (auto quantileType = mlir::dyn_cast_or_null<vpux::type::QuantileType>(inputElemType)) {
            expectedShiftElemType = quantileType.getQuantileType();
        }
        if (!isSupportedShiftElemType(shiftElemType) || shiftElemType != expectedShiftElemType) {
            _log.trace("Match failed: Shift data type {0} is not supported or does not match weights type {1}.",
                       shiftElemType, expectedShiftElemType);
            return mlir::failure();
        }
    }

    const auto loc = wdInfo.getLastOp()->getLoc();
    rewriter.setInsertionPointAfter(origOp);

    auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, IE::getTrueInputValue(origOp, rewriter), quantElemType)
                              .getOutput();
    if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue = rewriter.create<IE::TransposeOp>(appendLoc(loc, "transpose_in"), inputValue, nullptr,
                                                      transposeOp.getOrderValueAttr())
                             .getOutput();
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
        if (auto quantileType = mlir::dyn_cast_or_null<vpux::type::QuantileType>(inputElemType)) {
            expectedShiftElemType = quantileType.getQuantileType();
        }
        if (!isSupportedShiftElemType(shiftElemType) || shiftElemType != expectedShiftElemType) {
            _log.trace("Match failed: Shift data type {0} is not supported or does not match weights type {1}.",
                       shiftElemType, expectedShiftElemType);
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
        } else if (auto stridedSliceOp = mlir::dyn_cast_or_null<IE::StridedSliceOp>(scale.getDefiningOp())) {
            rewriter.setInsertionPointAfter(stridedSliceOp);
        } else if (auto gatherOp = mlir::dyn_cast_or_null<IE::GatherOp>(scale.getDefiningOp())) {
            rewriter.setInsertionPointAfter(gatherOp);
        }
    } else {
        // Static embedding table pattern: scale is a constant; use it as a value input so that
        // a unit-scale QuantizeCastOp is produced. This allows swap-operation-with-gather to
        // hoist Gather before QuantizeCast without breaking the per-axis type dimension constraint.
        scale = wdInfo.getStaticScale();
    }

    auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, IE::getTrueInputValue(origOp, rewriter), quantElemType)
                              .getOutput();
    if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue = rewriter.create<IE::TransposeOp>(appendLoc(loc, "transpose_in"), inputValue, nullptr,
                                                      transposeOp.getOrderValueAttr())
                             .getOutput();
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

    const auto users = to_small_vector(origOp->getUsers());

    const auto processUser = [&](mlir::Operation* user) -> mlir::LogicalResult {
        const auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest(), user);
        if (mlir::failed(maybeWdInfo)) {
            _log.trace("Failed to match WeightsDequantize structure.");
            return mlir::failure();
        }
        const auto& wdInfo = maybeWdInfo.value();
        if (!wdInfo.hasScale() && !wdInfo.hasShift()) {
            // For now we don't want to rewrite single Convert's with no scale or shift. A later pass,
            // FuseConvertWithQuantize, may handle some of them more efficiently while the remaining ones get
            // converted to QuantizeCast->Dequantize afterwards.
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

        const auto quantParamsAsInput = wdInfo.getDynamicScale() != nullptr || wdInfo.getDynamicShift() != nullptr;

        // Per-axis quantized constant weights feeding a Gather must go through dynamicMatchAndRewrite
        // so that scale is passed as a value input to DynamicDequantize rather than baked into the
        // QuantizeCast per-axis type, which would block later swap-operation-with-gather optimizations.
        const bool feedsGather =
                !quantParamsAsInput && mlir::isa<Const::DeclareOp>(origOp) && wdInfo.isQuantizedConsumedByGather();
        if (feedsGather) {
            return dynamicMatchAndRewrite(wdInfo, origOp, rewriter);
        }

        if (!quantParamsAsInput) {
            return staticMatchAndRewrite(wdInfo, origOp, rewriter);
        }

        // Dynamic scale/shift path.
        if (mlir::isa<Const::DeclareOp>(origOp)) {
            _log.trace("Match failed: Got dynamic scale but weights is a constant.");
            return mlir::failure();
        }
        if (!_enableWeightsDynamicDequantization) {
            _log.trace("Match failed: Got dynamic scale but dynamic dequantization is disabled.");
            return mlir::failure();
        }
        return dynamicMatchAndRewrite(wdInfo, origOp, rewriter);
    };

    bool anySucceeded = false;
    for (auto* user : users) {
        const auto result = processUser(user);
        _log.trace("Consolidation for user {0} {1}.", user, mlir::succeeded(result) ? "succeeded" : "failed");
        if (mlir::succeeded(result)) {
            anySucceeded = true;
        }
    }
    return mlir::success(anySucceeded);
}

}  // namespace vpux

void vpux::IE ::registerConsolidateWeightsDequantizationRewriters(RewriterRegistry& registry,
                                                                  ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                  size_t index, Logger log) {
    registry.registerRewriterSet("consolidate-weights-dequantization", [&registry, log, benefitLevels, index]() {
        registry.registerRewriter<WeightsDequantizeRewriter<IE::ConvertOp>>("weights-dequantize-convert", log,
                                                                            benefitLevels[index]);
        registry.registerRewriter<WeightsDequantizeRewriter<Const::DeclareOp>>("weights-dequantize-declare-op", log,
                                                                               benefitLevels[index]);
        vpux::IE::registerConvertOpRewriters(registry);
    });
}
