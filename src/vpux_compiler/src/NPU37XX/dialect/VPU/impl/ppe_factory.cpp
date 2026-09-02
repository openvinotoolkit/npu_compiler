//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <cassert>
#include <limits>

using namespace vpux;
using namespace vpux::VPU;
using namespace vpux::VPU::arch37xx;

PpeFactory::AttrBuilder::AttrBuilder(mlir::MLIRContext* ctx): _ctx(ctx) {
}

PPEIntAttr PpeFactory::AttrBuilder::getAttr() const {
    const auto quantScaleAttr = quantScale.has_value() ? vpux::getFPArrayAttr(_ctx, *quantScale) : nullptr;
    const auto quantMultAttr = quantMult.has_value() ? vpux::getIntArrayAttr(_ctx, *quantMult) : nullptr;
    const auto quantShiftAttr = quantShift.has_value() ? vpux::getIntArrayAttr(_ctx, *quantShift) : nullptr;
    const auto quantPostShiftAttr = quantPostShift.has_value() ? vpux::getIntAttr(_ctx, *quantPostShift) : nullptr;
    const auto in1QuantMultAttr = in1QuantMult.has_value() ? vpux::getIntArrayAttr(_ctx, *in1QuantMult) : nullptr;
    const auto in2QuantMultAttr = in2QuantMult.has_value() ? vpux::getIntArrayAttr(_ctx, *in2QuantMult) : nullptr;

    return PPEIntAttr::get(_ctx, PPEModeAttr::get(_ctx, mode), vpux::getIntAttr(_ctx, clampLow),
                           vpux::getIntAttr(_ctx, clampHigh), vpux::getIntAttr(_ctx, lReluMult),
                           vpux::getIntAttr(_ctx, lReluShift), quantScaleAttr, quantMultAttr, quantShiftAttr,
                           quantPostShiftAttr, in1QuantMultAttr, in2QuantMultAttr, vpux::getFPAttr(_ctx, fpPReluAlpha));
}

void PpeFactory::applyStaticScale(mlir::Operation* op, AttrBuilder& builder) const {
    auto staticScale = 1.0;
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        staticScale = convOp.getStaticScaleAttr() != nullptr ? convOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    } else if (auto addOp = mlir::dyn_cast<IE::AddOp>(op)) {
        staticScale = addOp.getStaticScaleAttr() != nullptr ? addOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    }

    if (isDoubleEqual(staticScale, 1.0)) {
        return;
    }

    if (!builder.quantScale.has_value()) {
        builder.quantScale = SmallVector<double>{staticScale};
        return;
    }

    auto newScale = SmallVector<double>();
    llvm::transform(*builder.quantScale, std::back_inserter(newScale), [&staticScale](const auto s) {
        return s * staticScale;
    });
    builder.quantScale = std::move(newScale);
}

void PpeFactory::configureAttrForAvgPool(mlir::Operation* op, AttrBuilder& builder) const {
    auto avgPoolOp = mlir::dyn_cast<vpux::IE::AvgPoolOp>(op);
    if (avgPoolOp == nullptr) {
        return;
    }

    auto kernelSize = vpux::parseIntArrayAttr<int64_t>(avgPoolOp.getKernelSizeAttr());
    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
    auto staticScale =
            avgPoolOp.getStaticScaleAttr() != nullptr ? avgPoolOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    if (!mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
        builder.quantScale = mlir::SmallVector<double>{
                (computeAvgPoolQuantScale(nullptr, outputElemType, kernelSize) * staticScale)};
        return;
    }

    const auto scaleApproximation = vpux::QuantizationApproximation(
            computeAvgPoolQuantScale(inputElemType, outputElemType, kernelSize) * staticScale);

    builder.quantMult = SmallVector<int64_t>{scaleApproximation.mult()};
    builder.quantShift = SmallVector<int64_t>{scaleApproximation.shift()};
    builder.quantPostShift = scaleApproximation.postShift();
}

void PpeFactory::calculateFpPReluAlpha(mlir::Operation* operation, PpeFactory::AttrBuilder& builder) const {
    const auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    if (!mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
        // Mixed mode with float input and quant output requires negative slope rescaling.
        if (mlir::isa<mlir::quant::UniformQuantizedType>(outputElemType)) {
            const auto perTensor = mlir::cast<mlir::quant::UniformQuantizedType>(outputElemType);
            builder.fpPReluAlpha /= static_cast<float>(perTensor.getScale());
        }

        // Mixed mode with float input and quant weights requires negative slope rescaling.
        const auto weightsType =
                llvm::TypeSwitch<mlir::Operation*, mlir::Type>(operation)
                        .Case<NCEOpInterface>([](auto op) {
                            const auto weights = op.getWeightsOperand();
                            return weights != nullptr
                                           ? mlir::dyn_cast<vpux::NDTypeInterface>(weights.getType()).getElementType()
                                           : nullptr;
                        })
                        // This method may also be called before NCEOpInterface is attached, so case-by-case checks are
                        // needed (until a WeightedOpInterface is devised):
                        .Case<IE::ConvolutionOp>([](auto op) {
                            return mlir::dyn_cast<vpux::NDTypeInterface>(op.getFilter().getType()).getElementType();
                        })
                        .Case<IE::GroupConvolutionOp>([](auto op) {
                            return mlir::dyn_cast<vpux::NDTypeInterface>(op.getFilter().getType()).getElementType();
                        })
                        .Case<IE::TransposedConvolutionOp>([](auto op) {
                            return mlir::dyn_cast<vpux::NDTypeInterface>(op.getFilter().getType()).getElementType();
                        })
                        .Default([](auto) {
                            return mlir::Type(nullptr);
                        });

        if (const auto uqType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(weightsType)) {
            builder.fpPReluAlpha *= static_cast<float>(uqType.getScale());
        }
    }
}

struct ClampIntersectionResult {
    int32_t low;
    int32_t high;
    PPEMode mode;
};

static ClampIntersectionResult calcClampIntersection(const int32_t currentLow, const int32_t currentHigh,
                                                     const double newLow, const double newHigh,
                                                     mlir::Type outputElemType) {
    VPUX_THROW_WHEN(outputElemType == nullptr, "Expected a valid output element type but got NULL.");

    if (auto quantizedType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        const auto scale = quantizedType.getScale();
        const auto zp = quantizedType.getZeroPoint();
        const auto storageMin = quantizedType.getStorageTypeMin();
        const auto storageMax = quantizedType.getStorageTypeMax();

        // Adapt the new interval to include scale and zp, since clamping occurs after scaling and zp-shifting on HW
        const auto qMin = checked_cast<int32_t>(std::round(newLow / scale) + zp);
        const auto qMax = checked_cast<int32_t>(std::round(newHigh / scale) + zp);

        const auto targetLow = std::max<int64_t>(storageMin, std::max(qMin, currentLow));
        const auto targetHigh = std::min<int64_t>(storageMax, std::min(qMax, currentHigh));
        const auto mode =
                targetLow - zp == 0 && targetHigh - zp < storageMax ? VPU::PPEMode::LRELUX : VPU::PPEMode::NOOP;

        return {checked_cast<int32_t>(targetLow), checked_cast<int32_t>(targetHigh), mode};

    } else if (outputElemType.isF16()) {
        auto targetLow = static_cast<type::float16>(newLow);
        auto targetHigh = static_cast<type::float16>(newHigh);

        if (currentHigh < std::numeric_limits<int32_t>::max()) {
            const auto [fLow, fHigh] = unpackClamp<type::float16>(currentHigh);
            targetLow = std::max(fLow, targetLow);
            targetHigh = std::min(fHigh, targetHigh);
        }

        const auto floatMax = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
        const auto mode = targetHigh < floatMax ? VPU::PPEMode::LRELUX : VPU::PPEMode::LRELU;

        return {std::numeric_limits<int32_t>::min(), packClamp(targetLow, targetHigh), mode};

    } else if (outputElemType.isBF16()) {
        auto targetLow = static_cast<type::bfloat16>(newLow);
        auto targetHigh = static_cast<type::bfloat16>(newHigh);

        if (currentHigh < std::numeric_limits<int32_t>::max()) {
            const auto [fLow, fHigh] = unpackClamp<type::bfloat16>(currentHigh);
            targetLow = std::max(fLow, targetLow);
            targetHigh = std::min(fHigh, targetHigh);
        }

        const auto floatMax = checked_cast<double>(std::numeric_limits<vpux::type::bfloat16>::max());
        const auto mode = targetHigh < floatMax ? VPU::PPEMode::LRELUX : VPU::PPEMode::LRELU;

        return {std::numeric_limits<int32_t>::min(), packClamp(targetLow, targetHigh), mode};

    } else {
        VPUX_THROW("Got invalid PPE output element type: {0}", outputElemType);
    }
}

template <>
PpeFactory::AttrBuilder PpeFactory::callback<IE::ReluAttr>(vpux::IE::LayerWithPostOpInterface operation,
                                                           IE::ReluAttr) const {
    PpeFactory::AttrBuilder builder(operation.getContext());

    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    if (auto outElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
        VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                        "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");
        builder.clampLow = vpux::extractScalesAndZeroPoints(outputElemType).second.front();
        builder.clampHigh = outElemQType.getStorageTypeMax();

    } else {
        builder.mode = PPEMode::LRELU;
    }

    configureAttrForAvgPool(operation, builder);
    calculateFpPReluAlpha(operation, builder);
    return builder;
}

template <>
PpeFactory::AttrBuilder PpeFactory::callback<IE::LeakyReluAttr>(vpux::IE::LayerWithPostOpInterface operation,
                                                                IE::LeakyReluAttr leakyRelu) const {
    PpeFactory::AttrBuilder builder(operation.getContext());
    builder.mode = PPEMode::LPRELU;

    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();

    if (auto outElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
        VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                        "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");
        builder.clampLow = outElemQType.getStorageTypeMin();
        builder.clampHigh = outElemQType.getStorageTypeMax();
    }

    builder.fpPReluAlpha = leakyRelu.getNegativeSlope().getValueAsDouble();
    if (isFloatEqual(builder.fpPReluAlpha, 0.0f)) {
        builder.lReluMult = 0;
    } else if (!isFloatEqual(builder.fpPReluAlpha, 1.0f)) {
        const auto alphaApproximation = PReLUApproximation(builder.fpPReluAlpha);
        builder.lReluMult = alphaApproximation.mult();
        builder.lReluShift = alphaApproximation.shift();
    }

    configureAttrForAvgPool(operation, builder);
    calculateFpPReluAlpha(operation, builder);
    return builder;
}

PpeFactory::AttrBuilder PpeFactory::retrieveNonEltwisePPEAttribute(mlir::Operation* operation) const {
    PpeFactory::AttrBuilder builder(operation->getContext());

    auto layerWithPostOpIfc = mlir::dyn_cast<vpux::IE::LayerWithPostOpInterface>(operation);
    if (layerWithPostOpIfc == nullptr || layerWithPostOpIfc.getPostOp() == nullptr) {
        auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();
        if (auto outElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
            VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                            "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");

            builder.clampLow = outElemQType.getStorageTypeMin();
            builder.clampHigh = outElemQType.getStorageTypeMax();
        }
        configureAttrForAvgPool(operation, builder);
        calculateFpPReluAlpha(operation, builder);
    } else {
        llvm::TypeSwitch<IE::PostOpAttr, void>(layerWithPostOpIfc.getPostOp())
                .Case<IE::ReluAttr, IE::LeakyReluAttr>([&](const auto postOp) {
                    builder = this->callback(layerWithPostOpIfc, postOp);
                })
                .Default([](const auto postOp) {
                    VPUX_THROW("Received unknown PPE post-op: {0}", postOp.getName());
                });
    }

    if (layerWithPostOpIfc != nullptr && layerWithPostOpIfc.getClampAttr() != nullptr) {
        const auto clamp = layerWithPostOpIfc.getClampAttr();
        const auto clampMin = clamp.getAs<mlir::FloatAttr>("min").getValueAsDouble();
        const auto clampMax = clamp.getAs<mlir::FloatAttr>("max").getValueAsDouble();

        auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();
        const auto isQuantized = mlir::isa<mlir::quant::UniformQuantizedType>(outputElemType);

        const auto intersection =
                calcClampIntersection(builder.clampLow, builder.clampHigh, clampMin, clampMax, outputElemType);

        builder.clampLow = intersection.low;
        builder.clampHigh = intersection.high;
        builder.mode = isQuantized ? builder.mode : intersection.mode;
    }

    return builder;
}

PpeFactory::AttrBuilder PpeFactory::retrievePermuteQuantizePPEAttribute(mlir::Operation* operation) const {
    VPUX_THROW_WHEN(!mlir::isa<IE::PermuteQuantizeOp>(operation), "Expected PermuteQuantizeOp but got: {0}",
                    operation->getName());

    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();

    auto builder = PpeFactory::AttrBuilder(operation->getContext());
    if (auto outElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
        builder.clampLow = outElemQType.getStorageTypeMin();
        builder.clampHigh = outElemQType.getStorageTypeMax();
    }

    // In this case NCEPermuteOp only perform permutation and will get converted to an AddOp,
    // so it's scale should be halved
    if (!mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
        const auto scaleVal = computeQuantScale(nullptr, outputElemType) / 2.0;
        builder.quantScale = mlir::SmallVector<double>{scaleVal};
        // alpha and scale shared a single multiplier in PPEFp and
        // there was no post-op for PermuteQuantize, so their values remained consistent.
        builder.fpPReluAlpha = scaleVal;
        return builder;
    }

    VPUX_THROW_WHEN(inputElemType != outputElemType,
                    "Input and output quantized types must be the same for PermuteQuantizeOp");

    const auto input1QuantScale = 1.0;
    const auto input2QuantScale = 1.0;
    const auto outputQuantScale = 2.0;

    const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(operation);
    const auto allScaleApproximation =
            VPU::EltwiseQuantizationApproximation(input1QuantScale, input2QuantScale, outputQuantScale, eltwiseType);

    builder.quantMult = mlir::SmallVector<int64_t>{allScaleApproximation.output().mult()};
    builder.quantShift = mlir::SmallVector<int64_t>{allScaleApproximation.output().shift()};
    builder.quantPostShift = allScaleApproximation.output().postShift();
    builder.in1QuantMult = mlir::SmallVector<int64_t>{allScaleApproximation.input1().mult()};
    builder.in2QuantMult = mlir::SmallVector<int64_t>{allScaleApproximation.input2().mult()};
    return builder;
}

PpeFactory::AttrBuilder PpeFactory::retrieveEltwisePPEAttribute(mlir::Operation* operation) const {
    VPUX_THROW_WHEN(!mlir::isa<IE::AddOp>(operation), "Unsupported PPE eltwise operation: {0}", operation->getName());

    auto inputVal1 = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    auto inputVal2 = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(1).getType()).getElementType();

    VPUX_THROW_UNLESS(
            mlir::isa<mlir::quant::QuantizedType>(inputVal1) == mlir::isa<mlir::quant::QuantizedType>(inputVal2),
            "Not supporting mixed precision on the inputs of eltwise!");
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();

    auto layerWithPostOp = mlir::dyn_cast<vpux::IE::LayerWithPostOpInterface>(operation);
    auto builder = layerWithPostOp != nullptr ? PpeFactory::AttrBuilder(retrieveNonEltwisePPEAttribute(layerWithPostOp))
                                              : PpeFactory::AttrBuilder(operation->getContext());

    const auto outElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType);
    if (layerWithPostOp == nullptr && outElemQType != nullptr) {
        VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                        "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");

        builder.clampLow = outElemQType.getStorageTypeMin();
        builder.clampHigh = outElemQType.getStorageTypeMax();
    }

    if (!mlir::isa<mlir::quant::QuantizedType>(inputVal1)) {
        VPUX_THROW_WHEN(mlir::isa<mlir::quant::QuantizedType>(inputVal2),
                        "Currently not supporting both quantized and non-quantized inputs on the same op");

        builder.quantScale = mlir::SmallVector<double>{(computeQuantScale(nullptr, outputElemType))};
        return builder;
    }

    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputVal1) ||
                            mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputVal2),
                    "Currently not supporting quantized per-axis types as PPE input");

    auto input1QuantScale = vpux::extractScalesAndZeroPoints(inputVal1).first.front();
    auto input2QuantScale = vpux::extractScalesAndZeroPoints(inputVal2).first.front();
    auto outputQuantScale = mlir::isa<mlir::quant::QuantizedType>(outputElemType)
                                    ? vpux::extractScalesAndZeroPoints(outputElemType).first.front()
                                    : 1.0;

    const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(operation);
    const auto allScaleApproximation =
            VPU::EltwiseQuantizationApproximation(input1QuantScale, input2QuantScale, outputQuantScale, eltwiseType);

    builder.quantMult = mlir::SmallVector<int64_t>{allScaleApproximation.output().mult()};
    builder.quantShift = mlir::SmallVector<int64_t>{allScaleApproximation.output().shift()};
    builder.quantPostShift = allScaleApproximation.output().postShift();
    builder.in1QuantMult = mlir::SmallVector<int64_t>{allScaleApproximation.input1().mult()};
    builder.in2QuantMult = mlir::SmallVector<int64_t>{allScaleApproximation.input2().mult()};
    return builder;
}

PPEAttr PpeFactory::retrievePPEAttribute(mlir::Operation* operation) const {
    AttrBuilder builder = operation->getContext();
    if (mlir::isa<IE::PermuteQuantizeOp>(operation)) {
        builder = retrievePermuteQuantizePPEAttribute(operation);
    } else if (operation->hasTrait<IE::EltwiseOp>()) {
        builder = retrieveEltwisePPEAttribute(operation);
    } else {
        builder = retrieveNonEltwisePPEAttribute(operation);
    }
    applyStaticScale(operation, builder);
    return builder.getAttr();
}

vpux::VPU::PPEIntAttr PpeFactory::castToConcreteAttr(PPEAttr ppeAttr) const {
    const auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    VPUX_THROW_WHEN(intPpeAttr == nullptr,
                    "Expected PPEIntAttr type but got {0}, make sure to use the right factory version", ppeAttr);
    return intPpeAttr;
}

std::pair<double, double> PpeFactory::getClamps(vpux::VPU::PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    return std::make_pair(intPpeAttr.getClampLow().getValue().getSExtValue(),
                          intPpeAttr.getClampHigh().getValue().getSExtValue());
}

vpux::VPU::PPEAttr PpeFactory::updateClamps(vpux::VPU::PPEAttr orig, PPEAttr newClamps) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    const auto newClampsAttr = castToConcreteAttr(newClamps);

    const auto newLow = static_cast<int32_t>(newClampsAttr.getClampLow().getValue().getSExtValue());
    const auto newHigh = static_cast<int32_t>(newClampsAttr.getClampHigh().getValue().getSExtValue());

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), vpux::getIntAttr(ctx, newLow), vpux::getIntAttr(ctx, newHigh),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), intPpeAttr.getQuantScale(),
                           intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(),
                           intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}

vpux::VPU::PPEAttr PpeFactory::intersectClamps(vpux::VPU::PPEAttr orig, double newLow, double newHigh,
                                               mlir::Type outputElemType) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    VPUX_THROW_WHEN(outputElemType == nullptr, "Expected a valid output element type but got NULL.");

    const auto currentLow = static_cast<int32_t>(intPpeAttr.getClampLow().getValue().getSExtValue());
    const auto currentHigh = static_cast<int32_t>(intPpeAttr.getClampHigh().getValue().getSExtValue());
    auto targetLow = currentLow;
    auto targetHigh = currentHigh;

    if (const auto quantizedType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        const auto scale = quantizedType.getScale();
        const auto zp = quantizedType.getZeroPoint();

        // Adapt the new interval to include scale and zp, since clamping occurs after scaling and zp-shifting on HW
        const auto quantizedNewLow = static_cast<int32_t>(std::round(newLow / scale) + zp);
        const auto quantizedNewHigh = static_cast<int32_t>(std::round(newHigh / scale) + zp);

        targetLow = std::max(currentLow, quantizedNewLow);
        targetHigh = std::min(currentHigh, quantizedNewHigh);

    } else if (outputElemType.isF16()) {
        // compare intervals as F16's
        const auto currentF16LowHigh = unpackClamp<type::float16>(currentHigh);
        const auto targetF16Low = std::max(currentF16LowHigh.first, static_cast<type::float16>(newLow));
        const auto targetF16High = std::min(currentF16LowHigh.second, static_cast<type::float16>(newHigh));

        targetLow = std::numeric_limits<int32_t>::min();
        targetHigh = packClamp(targetF16Low, targetF16High);

    } else if (outputElemType.isBF16()) {
        // compare intervals as F16's
        const auto currentF16LowHigh = unpackClamp<type::bfloat16>(currentHigh);
        const auto targetF16Low = std::max(currentF16LowHigh.first, static_cast<type::bfloat16>(newLow));
        const auto targetF16High = std::min(currentF16LowHigh.second, static_cast<type::bfloat16>(newHigh));

        targetLow = std::numeric_limits<int32_t>::min();
        targetHigh = packClamp(targetF16Low, targetF16High);

    } else {
        VPUX_THROW("Got invalid PPE output element type: {0}", outputElemType);
    }

    if (targetLow == currentLow && targetHigh == currentHigh) {
        return orig;  // same clamps, don't recreate attribute
    }

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), vpux::getIntAttr(ctx, targetLow),
                           vpux::getIntAttr(ctx, targetHigh), intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(),
                           intPpeAttr.getQuantScale(), intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(),
                           intPpeAttr.getQuantPostShift(), intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(),
                           intPpeAttr.getFpPreluAlpha());
}

PPEAttr PpeFactory::discardClamp(vpux::VPU::PPEAttr orig, mlir::Type outputElemType) const {
    const auto intPpeAttr = castToConcreteAttr(orig);

    // Default for UniformQuantizedType
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    auto ctx = orig.getContext();

    if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        clampLow = static_cast<int32_t>(quantType.getStorageTypeMin());
        clampHigh = static_cast<int32_t>(quantType.getStorageTypeMax());
    }

    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), vpux::getIntAttr(ctx, clampLow), vpux::getIntAttr(ctx, clampHigh),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), intPpeAttr.getQuantScale(),
                           intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(),
                           intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}

std::optional<SmallVector<double>> PpeFactory::getScale(PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    if (const auto scaleAttr = intPpeAttr.getQuantScale()) {
        return parseFPArrayAttr<double>(scaleAttr);
    }
    return std::nullopt;
}

std::optional<double> PpeFactory::getBias(PPEAttr) const {
    // Return nullopt, PPEIntAttr has no bias field.
    return std::nullopt;
}

PPEAttr PpeFactory::updateScale(PPEAttr orig, ArrayRef<double> scale) const {
    const auto intPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), intPpeAttr.getClampLow(), intPpeAttr.getClampHigh(),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), vpux::getFPArrayAttr(ctx, scale),
                           intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(),
                           intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}

PPEAttr PpeFactory::updateBias(PPEAttr orig, double) const {
    // Do nothing, PPEIntAttr has no bias field.
    return orig;
}

PPEAttr PpeFactory::discardScaleBias(PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), intPpeAttr.getClampLow(), intPpeAttr.getClampHigh(),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), nullptr, intPpeAttr.getQuantMult(),
                           intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(), intPpeAttr.getIn1QuantMult(),
                           intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}

SmallVector<double> PpeFactory::getFpPreluAlpha(PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    if (const auto fpPreluAlphaAttr = intPpeAttr.getFpPreluAlpha()) {
        return {fpPreluAlphaAttr.getValueAsDouble()};
    }
    return {1.0};
}

PPEAttr PpeFactory::updateFpPreluAlpha(PPEAttr orig, ArrayRef<double> fpPreluAlpha) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    VPUX_THROW_WHEN(fpPreluAlpha.size() != 1, "IntPPE only supports scalar pRelu alpha's");

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), intPpeAttr.getClampLow(), intPpeAttr.getClampHigh(),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), intPpeAttr.getQuantScale(),
                           intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(),
                           intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(),
                           vpux::getFPAttr(ctx, fpPreluAlpha.front()));
}

bool PpeFactory::hasQuantScalingThroughPreluAlpha(PPEAttr) const {
    // Quantization scale is never applied through pReluAlpha on this arch.
    return false;
}

PPEAttr PpeFactory::recomputeQuantParams(PPEAttr orig, mlir::Type inputElemType, mlir::Type outputElemType,
                                         ArrayRef<int64_t> kernelShape) const {
    const auto intPpeAttr = castToConcreteAttr(orig);

    const auto scaleApproximation = vpux::QuantizationApproximation(
            vpux::VPU::computeAvgPoolQuantScale(inputElemType, outputElemType, kernelShape));

    const auto quantMult = SmallVector<int64_t>{scaleApproximation.mult()};
    const auto quantShift = SmallVector<int64_t>{scaleApproximation.shift()};
    const auto quantPostShift = scaleApproximation.postShift();

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), intPpeAttr.getClampLow(), intPpeAttr.getClampHigh(),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), intPpeAttr.getQuantScale(),
                           vpux::getIntArrayAttr(ctx, quantMult), vpux::getIntArrayAttr(ctx, quantShift),
                           vpux::getIntAttr(ctx, quantPostShift), intPpeAttr.getIn1QuantMult(),
                           intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}

vpux::VPU::PPEMode PpeFactory::getMode(vpux::VPU::PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    return intPpeAttr.getMode().getValue();
}

PPEAttr PpeFactory::updateMode(PPEAttr orig, PPEMode mode) const {
    const auto intPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, PPEModeAttr::get(ctx, mode), intPpeAttr.getClampLow(), intPpeAttr.getClampHigh(),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), intPpeAttr.getQuantScale(),
                           intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(),
                           intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}
