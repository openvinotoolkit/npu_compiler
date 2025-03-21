//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <llvm/ADT/TypeSwitch.h>

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
    auto inputElemType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    auto outputElemType = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    auto staticScale =
            avgPoolOp.getStaticScaleAttr() != nullptr ? avgPoolOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    if (!inputElemType.isa<mlir::quant::QuantizedType>()) {
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
    const auto outputElemType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto inputElemType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!inputElemType.isa<mlir::quant::QuantizedType>()) {
        // Mixed mode with float input and quant output requires negative slope rescaling.
        if (outputElemType.isa<mlir::quant::UniformQuantizedType>()) {
            const auto perTensor = outputElemType.cast<mlir::quant::UniformQuantizedType>();
            builder.fpPReluAlpha /= static_cast<float>(perTensor.getScale());
        }

        // Mixed mode with float input and quant weights requires negative slope rescaling.
        const auto weightsType =
                llvm::TypeSwitch<mlir::Operation*, mlir::Type>(operation)
                        .Case<NCEOpInterface>([](auto op) {
                            const auto weights = op.getWeightsOperand();
                            return weights != nullptr ? weights.getType()
                                                                .template dyn_cast<vpux::NDTypeInterface>()
                                                                .getElementType()
                                                      : nullptr;
                        })
                        // This method may also be called before NCEOpInterface is attached, so case-by-case checks are
                        // needed (until a WeightedOpInterface is devised):
                        .Case<IE::ConvolutionOp>([](auto op) {
                            return op.getFilter().getType().template dyn_cast<vpux::NDTypeInterface>().getElementType();
                        })
                        .Case<IE::GroupConvolutionOp>([](auto op) {
                            return op.getFilter().getType().template dyn_cast<vpux::NDTypeInterface>().getElementType();
                        })
                        .Case<IE::TransposedConvolutionOp>([](auto op) {
                            return op.getFilter().getType().template dyn_cast<vpux::NDTypeInterface>().getElementType();
                        })
                        .Default([](auto) {
                            return mlir::Type(nullptr);
                        });

        if (const auto uqType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(weightsType)) {
            builder.fpPReluAlpha *= static_cast<float>(uqType.getScale());
        }
    }
}

PpeFactory::AttrBuilder PpeFactory::callbackReluOp(vpux::IE::LayerWithPostOpInterface operation) const {
    VPUX_THROW_UNLESS(operation.getPostOpAttrs().empty(), "'{0}' PostOp should not have any attributes",
                      operation.getPostOp());

    PpeFactory::AttrBuilder builder(operation.getContext());

    auto outputElemType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
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

PpeFactory::AttrBuilder PpeFactory::callbackClampOp(vpux::IE::LayerWithPostOpInterface operation) const {
    auto clamp = getPostOpAdaptor<vpux::IE::ClampOp>(operation);

    PpeFactory::AttrBuilder builder(operation.getContext());

    auto outputElemType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto scalesAndZp = extractScalesAndZeroPoints(outElemQType);
        const auto scale = scalesAndZp.first.front();
        const auto zp = scalesAndZp.second.front();

        auto clampLowStorageMin = outElemQType.getStorageTypeMin();
        auto clampHighStorageMax = outElemQType.getStorageTypeMax();

        auto qMin = checked_cast<int64_t>(std::round(clamp.getMin().convertToDouble() / scale)) + zp;
        auto qMax = checked_cast<int64_t>(std::round(clamp.getMax().convertToDouble() / scale)) + zp;

        auto clampLow = std::max(clampLowStorageMin, qMin);
        auto clampHigh = std::min(clampHighStorageMax, qMax);

        builder.clampLow = clampLow;
        builder.clampHigh = clampHigh;
        if (std::max(clampLowStorageMin, qMin) - zp == 0 &&
            std::min(clampHighStorageMax, qMax) - zp < outElemQType.getStorageTypeMax()) {
            builder.mode = VPU::PPEMode::LRELUX;
        }

    } else if (outputElemType.isF16()) {
        auto clampMax = clamp.getMax().convertToDouble();
        if (clampMax < checked_cast<double>(std::numeric_limits<vpux::type::float16>::max())) {
            builder.clampHigh = packClamp<type::float16>(clamp.getMin().convertToDouble(), clampMax);
            builder.mode = VPU::PPEMode::LRELUX;
        } else {
            builder.mode = VPU::PPEMode::LRELU;
        }
    } else if (outputElemType.isBF16()) {  // bf16 is supported by the FuseClampPass
        auto clampMax = clamp.getMax().convertToDouble();
        if (clampMax < checked_cast<double>(std::numeric_limits<vpux::type::float16>::max())) {
            builder.clampHigh = packClamp<type::bfloat16>(clamp.getMin().convertToDouble(), clampMax);
            builder.mode = VPU::PPEMode::LRELUX;
        } else {
            builder.mode = VPU::PPEMode::LRELU;
        }
    } else {
        VPUX_THROW("Got invalid PPE output element type: {0}", outputElemType);
    }

    configureAttrForAvgPool(operation, builder);
    calculateFpPReluAlpha(operation, builder);
    return builder;
}

PpeFactory::AttrBuilder PpeFactory::callbackLeakyReluOp(vpux::IE::LayerWithPostOpInterface operation) const {
    auto leakyRelu = getPostOpAdaptor<vpux::IE::LeakyReluOp>(operation);

    PpeFactory::AttrBuilder builder(operation.getContext());
    builder.mode = PPEMode::LPRELU;

    auto outputElemType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();

    if (auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                        "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");
        builder.clampLow = outElemQType.getStorageTypeMin();
        builder.clampHigh = outElemQType.getStorageTypeMax();
    }

    builder.fpPReluAlpha = leakyRelu.getNegativeSlope().convertToDouble();
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
    if (layerWithPostOpIfc == nullptr || layerWithPostOpIfc.getPostOpAttrs() == nullptr) {
        auto outputElemType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        if (auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                            "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");

            builder.clampLow = outElemQType.getStorageTypeMin();
            builder.clampHigh = outElemQType.getStorageTypeMax();
        }
        configureAttrForAvgPool(operation, builder);
        calculateFpPReluAlpha(operation, builder);
        return builder;
    }

    const auto postOp = layerWithPostOpIfc.getPostOp();
    VPUX_THROW_UNLESS(postOp.has_value(), "Missing PostOp on LayerWithPostOp");

    if (postOp->getStringRef() == IE::ReLUOp::getOperationName()) {
        return callbackReluOp(layerWithPostOpIfc);
    }
    if (postOp->getStringRef() == IE::ClampOp::getOperationName()) {
        return callbackClampOp(layerWithPostOpIfc);
    }
    if (postOp->getStringRef() == IE::LeakyReluOp::getOperationName()) {
        return callbackLeakyReluOp(layerWithPostOpIfc);
    }
    VPUX_THROW("Received unknown PPE PostOp: {0}", postOp->getStringRef());
}

PpeFactory::AttrBuilder PpeFactory::retrieveEltwisePPEAttribute(mlir::Operation* operation) const {
    VPUX_THROW_WHEN(!mlir::isa_and_nonnull<IE::AddOp>(operation), "Unsupported PPE eltwise operation: {0}",
                    operation->getName());

    auto inputVal1 = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    auto inputVal2 = operation->getOperand(1).getType().cast<vpux::NDTypeInterface>().getElementType();

    VPUX_THROW_UNLESS(inputVal1.isa<mlir::quant::QuantizedType>() == inputVal2.isa<mlir::quant::QuantizedType>(),
                      "Not supporting mixed precision on the inputs of eltwise!");
    auto outputElemType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();

    auto layerWithPostOp = mlir::dyn_cast<vpux::IE::LayerWithPostOpInterface>(operation);
    auto builder = layerWithPostOp != nullptr ? PpeFactory::AttrBuilder(retrieveNonEltwisePPEAttribute(layerWithPostOp))
                                              : PpeFactory::AttrBuilder(operation->getContext());

    const auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>();
    if (layerWithPostOp == nullptr && outElemQType != nullptr) {
        VPUX_THROW_WHEN(mlir::failed(vpux::extractScalarOrUniformZP(outElemQType)),
                        "Currently not supporting non-symmetric quantized per-axis types for PPE clamping");

        builder.clampLow = outElemQType.getStorageTypeMin();
        builder.clampHigh = outElemQType.getStorageTypeMax();
    }

    if (!inputVal1.isa<mlir::quant::QuantizedType>()) {
        VPUX_THROW_WHEN(inputVal2.isa<mlir::quant::QuantizedType>(),
                        "Currently not supporting both quantized and non-quantized inputs on the same op");

        builder.quantScale = mlir::SmallVector<double>{(computeQuantScale(nullptr, outputElemType))};
        return builder;
    }

    VPUX_THROW_WHEN(inputVal1.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            inputVal2.isa<mlir::quant::UniformQuantizedPerAxisType>(),
                    "Currently not supporting quantized per-axis types as PPE input");

    auto input1QuantScale = vpux::extractScalesAndZeroPoints(inputVal1).first.front();
    auto input2QuantScale = vpux::extractScalesAndZeroPoints(inputVal2).first.front();
    auto outputQuantScale = outputElemType.isa<mlir::quant::QuantizedType>()
                                    ? vpux::extractScalesAndZeroPoints(outputElemType).first.front()
                                    : 1.0;

    const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(operation);
    const auto allScaleApproximation =
            vpux::EltwiseQuantizationApproximation(input1QuantScale, input2QuantScale, outputQuantScale, eltwiseType);

    builder.quantMult = mlir::SmallVector<int64_t>{allScaleApproximation.output().mult()};
    builder.quantShift = mlir::SmallVector<int64_t>{allScaleApproximation.output().shift()};
    builder.quantPostShift = allScaleApproximation.output().postShift();
    builder.in1QuantMult = mlir::SmallVector<int64_t>{allScaleApproximation.input1().mult()};
    builder.in2QuantMult = mlir::SmallVector<int64_t>{allScaleApproximation.input2().mult()};
    return builder;
}

PPEAttr PpeFactory::retrievePPEAttribute(mlir::Operation* operation) const {
    auto builder = operation->hasTrait<IE::EltwiseOp>() ? retrieveEltwisePPEAttribute(operation)
                                                        : retrieveNonEltwisePPEAttribute(operation);
    applyStaticScale(operation, builder);
    return builder.getAttr();
}

vpux::VPU::PPEIntAttr PpeFactory::castToConcreteAttr(PPEAttr ppeAttr) const {
    const auto intPpeAttr = ppeAttr.dyn_cast<PPEIntAttr>();
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

    if (const auto quantizedType = outputElemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
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

SmallVector<double> PpeFactory::getScale(PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    if (const auto scaleAttr = intPpeAttr.getQuantScale())
        return parseFPArrayAttr<double>(scaleAttr);
    return {1.0};
}

PPEAttr PpeFactory::updateScale(PPEAttr orig, ArrayRef<double> scale) const {
    const auto intPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEIntAttr::get(ctx, intPpeAttr.getMode(), intPpeAttr.getClampLow(), intPpeAttr.getClampHigh(),
                           intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(), vpux::getFPArrayAttr(ctx, scale),
                           intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(), intPpeAttr.getQuantPostShift(),
                           intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(), intPpeAttr.getFpPreluAlpha());
}

SmallVector<double> PpeFactory::getFpPreluAlpha(PPEAttr orig) const {
    const auto intPpeAttr = castToConcreteAttr(orig);
    if (const auto fpPreluAlphaAttr = intPpeAttr.getFpPreluAlpha())
        return {fpPreluAlphaAttr.getValueAsDouble()};
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
