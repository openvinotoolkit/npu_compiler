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
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
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

    auto shiftAttr = wdInfo.getStaticShiftAttr();
    auto shiftValue = wdInfo.getStaticShift();

    // ZP is always externalized as a DynamicDequantize tensor operand.
    const auto storageParamsResult = getStorageParams(inputElemType);
    if (mlir::failed(storageParamsResult)) {
        _log.trace("Match failed: Could not get storage params for type {0}.", inputElemType);
        return mlir::failure();
    }
    mlir::Type storageType;
    std::tie(std::ignore, std::ignore, storageType) = *storageParamsResult;
    if (auto quantileStorageType = mlir::dyn_cast_if_present<vpux::type::QuantileType>(storageType)) {
        // getStorageParams() returns QuantileType itself as the storage type.
        // For ZP tensor construction and bit-width calculations, use the underlying integer storage type.
        storageType = quantileStorageType.getStorageType();
    }
    const auto bitWidth = storageType.getIntOrFloatBitWidth();

    // f4E2M1FN is a symmetric floating-point format; zero-point shifts have no meaningful
    // quantization semantics and are not supported for this storage type.
    // A splat zero shift (equivalent to no shift) is allowed through; any non-zero or
    // per-element shift is rejected.
    if (mlir::isa<mlir::Float4E2M1FNType>(storageType) && shiftAttr != nullptr) {
        const auto shiftContent = shiftAttr.fold();
        if (!shiftContent.isSplat() || shiftContent.getSplatValue<double>() != 0.0) {
            _log.trace("Match failed: Non-zero or per-element shift is not supported for f4E2M1FN storage type.");
            return mlir::failure();
        }
    }

    // Validate that non-splat shift has a supported element type (U2) matching the weights type
    if (shiftAttr != nullptr && !shiftAttr.fold().isSplat() && shiftValue != nullptr) {
        auto shiftElemType = IE::getTrueElemType(shiftValue.getDefiningOp<Const::DeclareOp>());
        auto expectedShiftElemType = inputElemType;
        if (auto quantileType = mlir::dyn_cast_if_present<vpux::type::QuantileType>(inputElemType)) {
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

    auto inputValue = IE::getTrueInputValue(origOp, rewriter);
    if (auto transposeOp = mlir::dyn_cast_if_present<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue = rewriter.create<IE::TransposeOp>(appendLoc(loc, "transpose_in"), inputValue, nullptr,
                                                      transposeOp.getOrderValueAttr())
                             .getOutput();
    }

    // DynamicDequantizeOp requires a non-null scale operand. When the WD chain has no
    // Multiply (shift-only patterns), getStaticScale() returns nullptr; create a unit
    // scale constant so the dequantization identity (scale=1.0) is explicit.
    mlir::Value scaleValue = wdInfo.getStaticScale();
    if (scaleValue == nullptr) {
        const auto inputRank = mlir::cast<vpux::NDTypeInterface>(inputValue.getType()).getRank();
        SmallVector<int64_t> scaleShape(inputRank, 1);
        const auto scaleTensorType = mlir::RankedTensorType::get(scaleShape, dstType);
        const auto unitScaleValueAttr = mlir::FloatAttr::get(mlir::cast<mlir::FloatType>(dstType), 1.0);
        const auto unitScaleAttr = mlir::DenseElementsAttr::get(scaleTensorType, unitScaleValueAttr);
        scaleValue = rewriter.create<Const::DeclareOp>(appendLoc(loc, "unit_scale"), scaleTensorType,
                                                       Const::ContentAttr::get(unitScaleAttr))
                             .getOutput();
    }

    // Build the ZP tensor operand. Both splat and non-splat shifts are externalized as
    // explicit tensor inputs to DynamicDequantizeOp instead of being embedded in the quant type.
    mlir::Value zpValue = nullptr;
    if (shiftAttr != nullptr) {
        auto shiftContent = shiftAttr.fold();
        if (storageType.isInteger()) {
            if (shiftContent.isSplat()) {
                const auto inputRank = mlir::cast<vpux::NDTypeInterface>(inputValue.getType()).getRank();
                SmallVector<int64_t> zpShape(inputRank, 1);
                const auto zpVal = shiftContent.getSplatValue<int64_t>();
                if (zpVal != 0) {
                    const auto signedness =
                            storageType.isSignedInteger() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
                    const auto zpType = mlir::IntegerType::get(rewriter.getContext(), bitWidth, signedness);
                    const auto zpTensorType = mlir::RankedTensorType::get(zpShape, zpType);
                    const auto zpDenseAttr =
                            mlir::DenseElementsAttr::get(zpTensorType, mlir::IntegerAttr::get(zpType, zpVal));
                    zpValue = rewriter.create<Const::DeclareOp>(appendLoc(loc, "zp"), zpTensorType,
                                                                Const::ContentAttr::get(zpDenseAttr))
                                      .getOutput();
                }
            } else {
                // Non-splat integer shift: use getTrueInputValue to get the integer-typed shift tensor
                zpValue = IE::getTrueInputValue(shiftValue.getDefiningOp<Const::DeclareOp>(), rewriter);
            }
        } else {
            if (shiftContent.isSplat()) {
                const auto zpVal = shiftContent.getSplatValue<double>();
                if (zpVal != 0.0) {
                    const auto inputRank = mlir::cast<vpux::NDTypeInterface>(inputValue.getType()).getRank();
                    SmallVector<int64_t> zpShape(inputRank, 1);
                    const auto zpTensorType = mlir::RankedTensorType::get(zpShape, storageType);
                    const auto zpDenseAttr =
                            mlir::DenseElementsAttr::get(zpTensorType, mlir::FloatAttr::get(storageType, zpVal));
                    zpValue = rewriter.create<Const::DeclareOp>(appendLoc(loc, "zp"), zpTensorType,
                                                                Const::ContentAttr::get(zpDenseAttr))
                                      .getOutput();
                }
            } else {
                zpValue = IE::getTrueInputValue(shiftValue.getDefiningOp<Const::DeclareOp>(), rewriter);
            }
        }
    }

    auto dequantizeOp = rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "artificial_dequant"), inputValue,
                                                                 scaleValue, zpValue, dstType);

    // Only mark with the synthetic attribute when the original develop-branch logic would have
    // used a static IE.Dequantize (i.e., canUseDequantize was true). The bridge pass
    // (ConvertConstantDynamicDequantizeToDequantize) converts only marked ops back to Dequantize.
    // Conditions for canUseDequantize on develop:
    //   - shift is null or splat
    //   - scale is null, splat, or has a single non-unit dimension (per-axis)
    auto scaleAttr = wdInfo.getStaticScaleAttr();

    bool canUseDequantize = !(shiftAttr != nullptr && !shiftAttr.fold().isSplat());

    if (canUseDequantize && scaleAttr != nullptr && !scaleAttr.isSplat()) {
        const auto scaleShape = scaleAttr.fold().getType().getShape();
        if (mlir::failed(getSingleDim(scaleShape.raw()))) {
            canUseDequantize = false;
        }
    }
    if (canUseDequantize) {
        dequantizeOp->setAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR, mlir::UnitAttr::get(rewriter.getContext()));
    }

    wdInfo.getLastOp()->replaceAllUsesWith(dequantizeOp.getOperation());
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

    mlir::Type shiftElemType = nullptr;
    if (auto dynamicShift = wdInfo.getDynamicShift()) {
        shiftElemType = IE::getTrueElemType(*dynamicShift.user_begin());
    } else if (auto staticShift = wdInfo.getStaticShift()) {
        shiftElemType = IE::getTrueElemType(staticShift.getDefiningOp<Const::DeclareOp>());
    }
    if (shiftElemType != nullptr) {
        auto expectedShiftElemType = inputElemType;
        if (auto quantileType = mlir::dyn_cast_if_present<vpux::type::QuantileType>(inputElemType)) {
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

    const auto loc = wdInfo.getLastOp()->getLoc();
    rewriter.setInsertionPointAfter(origOp);

    mlir::Value scale = wdInfo.getDynamicScale();
    if (scale != nullptr) {
        if (auto convertOp = mlir::dyn_cast_if_present<ConcreteOp>(*scale.user_begin())) {
            scale = convertOp.getOutput();
            rewriter.setInsertionPointAfter(convertOp);
        } else if (auto stridedSliceOp = mlir::dyn_cast_if_present<IE::StridedSliceOp>(scale.getDefiningOp())) {
            rewriter.setInsertionPointAfter(stridedSliceOp);
        } else if (auto gatherOp = mlir::dyn_cast_if_present<IE::GatherOp>(scale.getDefiningOp())) {
            rewriter.setInsertionPointAfter(gatherOp);
        } else if (auto affineReshapeOp = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(scale.getDefiningOp())) {
            rewriter.setInsertionPointAfter(affineReshapeOp);
        }
    } else {
        // Static embedding table pattern: scale is a constant; use it as a value input so that
        // swap-operations-with-gather-and-slice can hoist Gather before the dequantize without breaking
        // the per-axis type dimension constraint.
        scale = wdInfo.getStaticScale();
    }

    auto inputValue = IE::getTrueInputValue(origOp, rewriter);
    if (auto transposeOp = mlir::dyn_cast_if_present<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
        inputValue = rewriter.create<IE::TransposeOp>(appendLoc(loc, "transpose_in"), inputValue, nullptr,
                                                      transposeOp.getOrderValueAttr())
                             .getOutput();
    }

    mlir::Value shift = nullptr;
    if (wdInfo.hasShift()) {
        if (auto dynamicShift = wdInfo.getDynamicShift()) {
            shift = dynamicShift;
        } else if (auto staticShift = wdInfo.getStaticShift()) {
            shift = IE::getTrueInputValue(staticShift.getDefiningOp<Const::DeclareOp>(), rewriter);
        }
    }
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
                  (wdInfo.getStaticScale() != nullptr && wdInfo.getStaticShift() != nullptr) ||
                  (wdInfo.getDynamicScale() != nullptr && wdInfo.getStaticShift() != nullptr))) {
                _log.trace("Match failed: unsupported combination of static and dynamic scale/shift.");
                return mlir::failure();
            }
        }

        const auto quantParamsAsInput = wdInfo.getDynamicScale() != nullptr || wdInfo.getDynamicShift() != nullptr;

        // Per-axis quantized constant weights feeding a Gather must go through dynamicMatchAndRewrite
        // so that scale is passed as a value input to DynamicDequantize rather than baked into the
        // QuantizeCast per-axis type, which would block later swap-operations-with-gather-and-slice optimizations.
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
