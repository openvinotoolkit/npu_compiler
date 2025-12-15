//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_generator.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;
using namespace VPU::arch50xx;

// TODO: E#150106, some of the methods here should be moved back to the common ppe_utils.hpp, once IntPPE is also
// updated.

std::optional<double> getPerTensorScaleOrNull(mlir::Type elemType) {
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        return std::nullopt;  // Per-channel
    }
    if (const auto elemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
        return elemQType.getScale();
    }

    return 1.0;  // No quantization
}

std::optional<double> getPerTensorScaleOrNull(mlir::Value value) {
    return getPerTensorScaleOrNull(mlir::cast<NDTypeInterface>(value.getType()).getElementType());
}

template <typename T>
std::optional<double> getPerTensorFilterScale(T convLikeOp) {
    return getPerTensorScaleOrNull(convLikeOp.getFilter());
}

template <typename T>
std::optional<double> getPerTensorWeightsScale(T weightedOp) {
    return getPerTensorScaleOrNull(weightedOp.getWeights());
}

template <typename T>
std::optional<double> getPerTensorInput2Scale(T matmulOp) {
    return getPerTensorScaleOrNull(matmulOp.getInput2());
}

template <typename T, typename = std::enable_if_t<llvm::is_one_of<T, IE::ReduceMeanOp, VPU::NCEReduceOp>::value>>
double getReduceMeanScale(T reduceMeanOp, ArrayRef<int64_t> axes) {
    // HW only supports reduction on a single axis, the channels.
    VPUX_THROW_WHEN(axes.size() != 1 || axes.front() != Dims4D::Act::C.ind(), "Unsupported {0} axes: {1}",
                    reduceMeanOp->getName(), axes);

    const auto inputShape = getShape(reduceMeanOp.getInput());
    VPUX_THROW_UNLESS(static_cast<size_t>(Dims4D::Act::C.ind()) < inputShape.size(), "Invalid {0} input shape: {1}",
                      reduceMeanOp->getName(), inputShape);
    const auto channels = inputShape[Dims4D::Act::C];

    // The mean is computed by applying a PPE scale of 1 / size but the input channels may be padded to a multiple
    // of 16. The scale must remain relative to the original channel size.
    int64_t channelPadding = 0LL;
    if (const auto inputPaddingAttr = reduceMeanOp.getInputPaddingAttr()) {
        const auto inputPadding = parseIntArrayAttr<int64_t>(inputPaddingAttr);
        channelPadding = inputPadding[Dims4D::Act::C.ind()];
    }

    return 1.0 / (channels - channelPadding);
}

template <typename T>
std::optional<double> getPerTensorBias(T convLikeOp) {
    const auto bias = convLikeOp.getBias();
    if (bias == nullptr) {
        return 0.0;  // No bias
    }

    const auto biasConst = bias.template getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(biasConst == nullptr, "Cannot apply non-constant bias");

    const auto biasContent = biasConst.getContentAttr();
    if (!biasContent.isSplat()) {
        return std::nullopt;  // Multiple values => WT
    }

    // Since bias is applied before scale it must be adapted by the inputs scale (but not by output scale).
    const auto inputElemType = mlir::cast<NDTypeInterface>(convLikeOp->getOperand(0).getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
        return std::nullopt;  // Per-channel input scale => per-channel bias => WT
    }
    auto inputScale = 1.0;
    if (const auto inputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputElemType)) {
        inputScale = inputElemQType.getScale();
    }

    const auto filterScale = getPerTensorFilterScale(convLikeOp);
    if (!filterScale.has_value()) {
        return std::nullopt;  // Per-channel filter scale => per-channel bias => WT
    }

    return biasContent.fold().template getSplatValue<double>() / (inputScale * (*filterScale));
}

int64_t computeZeroPoint(mlir::Operation* operation) {
    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        return outputElemQType.getZeroPoint();
    }

    // Sometimes per-channel quantized types have multiple equal zero-points. These can still be applied
    // using PPE as a per-tensor shift.
    if (const auto outputElemQPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
        const auto zps = outputElemQPerAxisType.getZeroPoints();
        const auto firstZp = zps.front();
        const auto hasNonUniformZp = llvm::any_of(zps, [&firstZp](const auto zp) {
            return firstZp != zp;
        });

        VPUX_THROW_WHEN(hasNonUniformZp, "PPE can only apply per-tensor zero-points.");
        return firstZp;
    }

    return 1.0;
}

std::optional<double> computeBias(mlir::Operation* operation) {
    // Similar to scale, PPE attributes can store per-tensor biases. In case of per-channel biases the WeightsTable (WT)
    // will be used, this is signaled by setting bias to null inside PPE attributes.
    const auto bias = llvm::TypeSwitch<mlir::Operation*, std::optional<double>>(operation)
                              .Case<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp,
                                    VPU::ConvolutionOp, VPU::GroupConvolutionOp, VPU::TransposedConvolutionOp>(
                                      [](auto convLikeOp) -> std::optional<double> {
                                          return getPerTensorBias(convLikeOp);
                                      })
                              .Default([](auto) {
                                  return 0.0;  // Neutral value for addition
                              });
    return bias;
}

std::optional<double> computeScale(mlir::Operation* operation) {
    const auto inputElemType = mlir::cast<NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();

    // MaxPool is considered quantization-agnostic, PropagateFQ nGraph pass will propagate input and output FQ's which
    // should always result in neutral quantization scale (1.0). Non-quantized conversions (f16 <-> f32) are allowed.
    if (mlir::isa<IE::MaxPoolOp>(operation)) {
        VPUX_THROW_WHEN(
                inputElemType != outputElemType && (mlir::isa<mlir::quant::QuantizedType>(inputElemType) ||
                                                    mlir::isa<mlir::quant::QuantizedType>(outputElemType)),
                "Quantization-agnostic operation (MaxPool) has different quantized input ({0}) and output ({1}) types.",
                inputElemType, outputElemType);
        return 1.0;
    }

    // PPE attributes can store per-tensor scales. In case of per-channel scales the WeightsTable (WT) will be used,
    // this is signaled by setting scale to null inside PPE attributes.
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType) ||
        mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
        return std::nullopt;  // WT
    }

    // Scale Formula: (input.scale * weights.scale * static_scale) / (output.scale * avg_pool_kernel_size)
    auto scale = 1.0;

    // Apply input scale
    if (const auto inputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputElemType)) {
        scale *= inputElemQType.getScale();
    }

    // For weighted/filtered operations the weights/filter scale must be taken into account similar to input scale
    const auto weightsScale =
            llvm::TypeSwitch<mlir::Operation*, std::optional<double>>(operation)
                    .Case<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp, VPU::ConvolutionOp,
                          VPU::GroupConvolutionOp, VPU::TransposedConvolutionOp, VPU::NCEDepthConvolutionOp>(
                            [](auto convLikeOp) -> std::optional<double> {
                                return getPerTensorFilterScale(convLikeOp);
                            })
                    .Case<VPU::FullyConnectedOp, VPU::GRUSequenceFirstPartOp, VPU::GRUSequenceOp, VPU::LSTMCellOp,
                          VPU::NCEInterpolateOp, VPU::NCEMatMulOp, VPU::NormalizeIEOp, VPU::ScaleShiftOp>(
                            [](auto weightedOp) -> std::optional<double> {
                                return getPerTensorWeightsScale(weightedOp);
                            })
                    .Case<IE::MatMulOp>([](auto matmulOp) -> std::optional<double> {
                        return getPerTensorInput2Scale(matmulOp);
                    })
                    .Case<IE::ReduceMeanOp>([](auto reduceMeanOp) -> std::optional<double> {
                        return getReduceMeanScale(reduceMeanOp, IE::extractAxes(reduceMeanOp->getLoc(), reduceMeanOp));
                    })
                    .Case<VPU::NCEReduceOp>([](auto reduceOp) -> std::optional<double> {
                        return reduceOp.getOpType() == VPU::ReduceType::MEAN
                                       ? getReduceMeanScale(reduceOp, parseIntArrayAttr<int64_t>(reduceOp.getAxes()))
                                       : 1.0;
                    })
                    .Default([](auto) {
                        return 1.0;  // Neutral value for multiplication
                    });
    if (!weightsScale.has_value()) {
        return std::nullopt;
    }
    scale *= *weightsScale;

    // Convolution's may have static scale
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(operation)) {
        if (const auto staticScale = convOp.getStaticScale()) {
            scale *= staticScale->convertToDouble();
        }
    }

    // AvgPool averaging is done through PPE scale, may also have static scale
    if (auto avgPoolOp = mlir::dyn_cast<IE::AvgPoolOp>(operation)) {
        if (const auto staticScale = avgPoolOp.getStaticScale()) {
            scale *= staticScale->convertToDouble();
        }

        // Divide by D1 *...* Dn, where <D1 x...x Dn> is the kernel shape
        const auto kernelShape = parseIntArrayAttr<int64_t>(avgPoolOp.getKernelSizeAttr());
        const auto kernelSize = vpux::details::calcTotalShapeSize(kernelShape);
        scale /= kernelSize;
    }

    // Apply output scale
    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        scale /= outputElemQType.getScale();
    }

    return scale;
}

PpeFactory::AttrBuilder::AttrBuilder(mlir::MLIRContext* ctx): _ctx(ctx) {
}

PPEFpAttr PpeFactory::AttrBuilder::getAttr() const {
    const auto scaleAttr = scale.has_value() ? getFPAttr(_ctx, *scale) : nullptr;
    const auto biasAttr = bias.has_value() ? getFPAttr(_ctx, *bias) : nullptr;
    const auto in1MultAttr = in1Mult.has_value() ? getFPArrayAttr(_ctx, *in1Mult) : nullptr;
    const auto in2MultAttr = in2Mult.has_value() ? getFPArrayAttr(_ctx, *in2Mult) : nullptr;
    const auto sprLUTAttr = [this]() -> mlir::DenseElementsAttr {
        if (!sprLUT.has_value()) {
            return nullptr;
        }
        auto uint16Type = mlir::IntegerType::get(_ctx, 16, mlir::IntegerType::SignednessSemantics::Unsigned);
        auto sprLUTType = mlir::RankedTensorType::get({checked_cast<int64_t>(sprLUT->size())}, uint16Type);
        return mlir::DenseElementsAttr::get(sprLUTType, ArrayRef(*sprLUT));
    }();

    return PPEFpAttr::get(_ctx, PPEModeAttr::get(_ctx, mode), getFPAttr(_ctx, clampLow), getFPAttr(_ctx, clampHigh),
                          scaleAttr, getFPArrayAttr(_ctx, pReluAlpha), biasAttr, getFPAttr(_ctx, adder), in1MultAttr,
                          in2MultAttr, sprLUTAttr);
}

void PpeFactory::callbackDefault(mlir::Operation* operation, AttrBuilder& builder) const {
    // By default only quantization needs to be applied.
    // Since zero-point is applied after clamping, the interval must be shifted by the zero-point.
    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType);
        outputElemQType != nullptr && !mlir::isa<IE::MaxPoolOp>(operation)) {
        builder.adder = ::computeZeroPoint(operation);
        builder.clampLow = checked_cast<double>(outputElemQType.getStorageTypeMin() - builder.adder);
        builder.clampHigh = checked_cast<double>(outputElemQType.getStorageTypeMax() - builder.adder);
    }
}

template <>
void PpeFactory::callback<IE::ReluAttr>(IE::LayerWithPostOpInterface operation, IE::ReluAttr,
                                        AttrBuilder& builder) const {
    // Similar to the default case, but the lower clamp is configured to clamp-off negative values.
    builder.mode = PPEMode::LRELU;

    // note: -0.0, to ensure zero-gained data uses positive zero in FP32
    // (0x00000000), not negative zero (0x80000000)
    builder.pReluAlpha = {-0.0f};

    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
        VPUX_THROW_WHEN(mlir::isa<IE::MaxPoolOp>(operation),
                        "Relu post-op is not implemented for MaxPool's with quantized output");

        builder.adder = ::computeZeroPoint(operation);
        builder.clampLow = std::max(checked_cast<double>(outputElemQType.getStorageTypeMin()), 0.0) - builder.adder;
        builder.clampHigh = checked_cast<double>(outputElemQType.getStorageTypeMax() - builder.adder);

    } else {
        builder.clampLow = 0.0;
    }
}

template <>
void PpeFactory::callback<IE::ClampAttr>(IE::LayerWithPostOpInterface operation, IE::ClampAttr clamp,
                                         AttrBuilder& builder) const {
    // Similar to the ReLU case, but with a given clamp low-high interval.
    // The clamping interval must be adapted to the scale (since scaling occurs before clamping), intersected with the
    // quantization min-max interval and then shifted by the zero-point (since zero-point addition occurs before
    // clamping).
    const auto clampLow = clamp.getMin().getValueAsDouble();
    const auto clampHigh = clamp.getMax().getValueAsDouble();

    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType),
                    "PPE clamping for per-axis quantized outputs is not supported");

    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        const auto scale = outputElemQType.getScale();
        const auto zp = outputElemQType.getZeroPoint();

        if (mlir::isa<IE::MaxPoolOp>(operation)) {
            // Special case for MaxPool since it is quantization-agnostic.
            const auto inputElemType = mlir::cast<NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
            // TODO: E#148432 FP8 FakeConvert should be propagated in the same way I8 FakeQuantize is.
            VPUX_THROW_WHEN(isFloat8Quantized(outputElemType) || isFloat8Quantized(inputElemType),
                            "FP8 MaxPool->Clamp is not fully supported yet.");

            builder.clampLow =
                    std::max(checked_cast<double>(outputElemQType.getStorageTypeMin()), clampLow / scale) + zp;
            builder.clampHigh =
                    std::min(checked_cast<double>(outputElemQType.getStorageTypeMax()), clampHigh / scale) + zp;

        } else {
            builder.adder = zp;
            builder.clampLow =
                    std::max(checked_cast<double>(outputElemQType.getStorageTypeMin() - zp), clampLow / scale);
            builder.clampHigh =
                    std::min(checked_cast<double>(outputElemQType.getStorageTypeMax() - zp), clampHigh / scale);
        }

    } else {
        builder.clampLow = clampLow;
        builder.clampHigh = clampHigh;
        builder.mode = VPU::PPEMode::LRELUX;
    }
}

template <>
void PpeFactory::callback<IE::LeakyReluAttr>(IE::LayerWithPostOpInterface operation, IE::LeakyReluAttr leakyRelu,
                                             AttrBuilder& builder) const {
    // Similar to the default case, but PPE is configured to apply the "leaky" alpha to negative values.
    builder.pReluAlpha = {leakyRelu.getNegativeSlope().getValueAsDouble()};
    builder.mode = PPEMode::LPRELU;

    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
        VPUX_THROW_WHEN(mlir::isa<IE::MaxPoolOp>(operation),
                        "LeakyRelu post-op is not implemented for MaxPool's with quantized output");

        builder.adder = ::computeZeroPoint(operation);
        builder.clampLow = checked_cast<double>(outputElemQType.getStorageTypeMin() - builder.adder);
        builder.clampHigh = checked_cast<double>(outputElemQType.getStorageTypeMax() - builder.adder);
    }
}

template <>
void PpeFactory::callback<IE::TanhAttr>(vpux::IE::LayerWithPostOpInterface operation, IE::TanhAttr,
                                        AttrBuilder& builder) const {
    builder.mode = PPEMode::TANH;
    builder.sprLUT = SprLUTGenerator(tanhf, AbsoluteError(TANH_ERROR))
                             .setIsSymmetric()
                             .addBypassRange(TANH_BYPASS_LOW, TANH_BYPASS_HIGH)
                             .addSaturationRange(TANH_SAT_LOW, TANH_SAT_HIGH, TANH_SAT_VALUE)
                             .generate();

    callbackDefault(operation, builder);
}

template <>
void PpeFactory::callback<IE::SigmoidAttr>(vpux::IE::LayerWithPostOpInterface operation, IE::SigmoidAttr,
                                           AttrBuilder& builder) const {
    builder.mode = PPEMode::SIGMOID;
    builder.sprLUT = SprLUTGenerator(
                             [eulerConstant = std::exp(1.0)](float x) {
                                 return 1 / (1 + pow(eulerConstant, -x));
                             },
                             AbsoluteError(SIGMOID_ERROR))
                             .addSaturationRange(SIGMOID_NEG_SAT_LOW, SIGMOID_NEG_SAT_HIGH, SIGMOID_NEG_SAT_VALUE)
                             .addSaturationRange(SIGMOID_POS_SAT_LOW, SIGMOID_POS_SAT_HIGH, SIGMOID_POS_SAT_VALUE)
                             .generate();

    callbackDefault(operation, builder);
}

template <>
void PpeFactory::callback<IE::ExpAttr>(vpux::IE::LayerWithPostOpInterface operation, IE::ExpAttr,
                                       AttrBuilder& builder) const {
    builder.mode = PPEMode::EXP;
    builder.sprLUT = SprLUTGenerator(
                             [](float x) {
                                 return std::exp(x);
                             },
                             RelativeError(EXP_ERROR))
                             .addSaturationRange(EXP_NEG_SAT_LOW, EXP_NEG_SAT_HIGH, EXP_NEG_SAT_VALUE)
                             .addSaturationRange(EXP_POS_SAT_LOW, EXP_POS_SAT_HIGH, EXP_POS_SAT_VALUE)
                             .generate();
    callbackDefault(operation, builder);
}

template <>
void PpeFactory::callback<IE::SwishAttr>(vpux::IE::LayerWithPostOpInterface operation, IE::SwishAttr swish,
                                         AttrBuilder& builder) const {
    const auto beta = swish.getBeta().getValueAsDouble();
    VPUX_THROW_UNLESS(beta >= 1.0, "sprLUT support Swish only with beta >= 1.0, but got {0} in {1}", beta, operation);

    builder.mode = PPEMode::SWISH;
    builder.sprLUT = SprLUTGenerator(
                             [beta, eulerConstant = std::exp(1.0)](float x) {
                                 return x / (1 + pow(eulerConstant, -x * beta));
                             },
                             AbsoluteError(SWISH_ERROR))
                             .addSaturationRange(SWISH_SAT_LOW, SWISH_SAT_HIGH, SWISH_SAT_VALUE)
                             .addBypassRange(SWISH_BYPASS_LOW, SWISH_BYPASS_HIGH)
                             .generate();

    callbackDefault(operation, builder);
}

template <>
void PpeFactory::callback<IE::GeluAttr>(vpux::IE::LayerWithPostOpInterface operation, IE::GeluAttr,
                                        AttrBuilder& builder) const {
    builder.mode = PPEMode::GELU;
    builder.sprLUT = SprLUTGenerator(
                             [](float x) {
                                 return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
                             },
                             AbsoluteError(GELU_ERROR))
                             .addSaturationRange(GELU_SAT_LOW, GELU_SAT_HIGH, GELU_SAT_VALUE)
                             .addBypassRange(GELU_BYPASS_LOW, GELU_BYPASS_HIGH)
                             .generate();

    callbackDefault(operation, builder);
}

template <>
void PpeFactory::callback<IE::HSwishAttr>(vpux::IE::LayerWithPostOpInterface operation, IE::HSwishAttr,
                                          AttrBuilder& builder) const {
    builder.mode = PPEMode::HSWISH;
    builder.sprLUT = SprLUTGenerator(
                             [](float x) {
                                 return x * std::min(std::max(x + 3.0f, 0.f), 6.0f) / 6.0f;
                             },
                             AbsoluteError(HSWISH_ERROR))
                             .addSaturationRange(HSWISH_NEG_SAT_LOW, HSWISH_NEG_SAT_HIGH, HSWISH_NEG_SAT_VALUE)
                             .addBypassRange(HSWISH_POS_BYPASS_LOW, HSWISH_POS_BYPASS_HIGH)
                             .generate();

    callbackDefault(operation, builder);
}

PpeFactory::AttrBuilder PpeFactory::retrieveNonEltwisePPEAttribute(mlir::Operation* operation) const {
    PpeFactory::AttrBuilder builder(operation->getContext());

    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(operation);
    if (layerWithPostOp == nullptr || layerWithPostOp.getPostOp() == nullptr) {
        callbackDefault(operation, builder);

    } else {
        llvm::TypeSwitch<IE::PostOpAttr, void>(layerWithPostOp.getPostOp())
                .Case<IE::ReluAttr, IE::ClampAttr, IE::LeakyReluAttr, IE::TanhAttr, IE::SigmoidAttr, IE::SwishAttr,
                      IE::GeluAttr, IE::ExpAttr, IE::HSwishAttr>([&](const auto postOp) {
                    this->callback(layerWithPostOp, postOp, builder);
                })
                .Default([](const auto postOp) {
                    VPUX_THROW("Received unknown PPE post-op: {0}", postOp.getName());
                });
    }

    // Sometimes (i.e. EnsureNCEOpsSizeRequirements) PPE Factory is used to regenerate an attribute for an NCE
    // operation. Initial WT info must be preserved even if the current NCE operation has no bias/quantization, thus
    // per-tensor scale and bias must remain null.
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);
        nceOp != nullptr && !mlir::isa<VPU::PPEStubAttr>(nceOp.getPPE()) && hasWeightsTable(nceOp.getPPE())) {
        return builder;
    }

    // It's not possible to pick only the bias or the scale from WT, so WT is either used for both or not used at all.
    builder.bias = ::computeBias(operation);
    builder.scale = builder.bias ? ::computeScale(operation) : std::nullopt;
    if (!builder.scale.has_value()) {
        builder.bias = std::nullopt;
    }

    if (!builder.sprLUT.has_value() || mlir::isa<IE::MaxPoolOp>(operation)) {
        return builder;
    }

    // When sprLUT is enabled output_scale must be applied after the activation function, preserving the natural order
    // of the operations. This is achieved through the pReluAlpha multiplier, which is configured to apply alpha to both
    // positive and negative values.
    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType),
                    "Cannot apply per-channel output scale when sprLUT is enabled.");

    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        const auto outputScale = outputElemQType.getScale();

        // "Move" output_scale from scale to pReluAlpha
        builder.pReluAlpha = SmallVector{1.0 / outputScale};
        if (builder.scale.has_value()) {  // Otherwise this must be applied in WT
            *builder.scale *= outputScale;
        }
    }

    return builder;
}

PpeFactory::AttrBuilder PpeFactory::retrieveEltwisePPEAttribute(mlir::Operation* operation) const {
    if (!mlir::isa<IE::AddOp, IE::SubtractOp, IE::MultiplyOp>(operation)) {
        // Supported ops: AddOp, SubtractOp, MultiplyOp
        VPUX_THROW("Unsupported PPE eltwise operation: {0}", operation->getName());
    }

    const auto in1ElemType = mlir::cast<NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    const auto in2ElemType = mlir::cast<NDTypeInterface>(operation->getOperand(1).getType()).getElementType();
    const auto outputElemType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType()).getElementType();

    VPUX_THROW_UNLESS(
            mlir::isa<mlir::quant::QuantizedType>(in1ElemType) == mlir::isa<mlir::quant::QuantizedType>(in2ElemType),
            "Mixed precision on the inputs of eltwise operations is not supported by PPE");
    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(in1ElemType) ||
                            mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(in2ElemType) ||
                            mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType),
                    "Per-axis quantized eltiwise operations are not supported by PPE");

    // Similar to the non-eltwise with except to per-channel quantization contraints and the possibility to scale inputs
    // individually through IDU.
    auto builder = PpeFactory::AttrBuilder(retrieveNonEltwisePPEAttribute(operation));
    if (!mlir::isa<mlir::quant::QuantizedType>(in1ElemType)) {
        return builder;
    }

    const auto input1Scale = mlir::cast<mlir::quant::UniformQuantizedType>(in1ElemType).getScale();
    const auto input2Scale = mlir::cast<mlir::quant::UniformQuantizedType>(in2ElemType).getScale();
    auto outputScale = 1.0;
    if (const auto outputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        outputScale = outputElemQType.getScale();
    }

    const auto elemType = mlir::dyn_cast<mlir::quant::QuantizedType>(in1ElemType).getStorageType();
    if (mlir::isa<mlir::FloatType>(elemType)) {
        // Float Quantization (fp8)
        builder.scale = outputScale;
        builder.in1Mult = SmallVector<double>{input1Scale};
        builder.in2Mult = SmallVector<double>{input2Scale};

    } else {
        // Integer Quantization (i8, i4 etc.)
        const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(operation);
        const auto allScaleApproximation =
                vpux::EltwiseQuantizationApproximation(input1Scale, input2Scale, outputScale, eltwiseType);
        builder.scale = static_cast<double>(allScaleApproximation.output().mult()) /
                        pow(2, allScaleApproximation.output().shift());
        builder.in1Mult = SmallVector{static_cast<double>(allScaleApproximation.input1().mult())};
        builder.in2Mult = SmallVector{static_cast<double>(allScaleApproximation.input2().mult())};
    }

    return builder;
}

PPEAttr PpeFactory::retrievePPEAttribute(mlir::Operation* operation) const {
    const auto builder = operation->hasTrait<IE::EltwiseOp>() ? retrieveEltwisePPEAttribute(operation)
                                                              : retrieveNonEltwisePPEAttribute(operation);
    return builder.getAttr();
}

vpux::VPU::PPEFpAttr PpeFactory::castToConcreteAttr(PPEAttr ppeAttr) const {
    const auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    VPUX_THROW_WHEN(fpPpeAttr == nullptr,
                    "Expected PPEFpAttr type but got {0}, make sure to use the right factory version", ppeAttr);
    return fpPpeAttr;
}

std::pair<double, double> PpeFactory::getClamps(VPU::PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    return std::make_pair(fpPpeAttr.getClampLow().getValueAsDouble(), fpPpeAttr.getClampHigh().getValueAsDouble());
}

VPU::PPEAttr PpeFactory::updateClamps(VPU::PPEAttr orig, PPEAttr newClamps) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    const auto newClampsAttr = castToConcreteAttr(newClamps);

    const auto newLow = newClampsAttr.getClampLow().getValueAsDouble();
    const auto newHigh = newClampsAttr.getClampHigh().getValueAsDouble();

    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), getFPAttr(ctx, newLow), getFPAttr(ctx, newHigh),
                          fpPpeAttr.getScale(), fpPpeAttr.getPreluAlpha(), fpPpeAttr.getBias(), fpPpeAttr.getAdder(),
                          fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

VPU::PPEAttr PpeFactory::intersectClamps(VPU::PPEAttr orig, double newLow, double newHigh,
                                         mlir::Type outputElemType) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);

    const auto currentLow = fpPpeAttr.getClampLow().getValueAsDouble();
    const auto currentHigh = fpPpeAttr.getClampHigh().getValueAsDouble();
    auto targetLow = currentLow;
    auto targetHigh = currentLow;

    if (const auto quantizedType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outputElemType)) {
        const auto scale = quantizedType.getScale();

        // Adapt the new interval to include scale, since clamping occurs after scaling (but before zp-shifting) on HW
        const auto quantizedNewLow = newLow / scale;
        const auto quantizedNewHigh = newHigh / scale;

        targetLow = std::max(currentLow, quantizedNewLow);
        targetHigh = std::min(currentHigh, quantizedNewHigh);

    } else {
        targetLow = std::max(currentLow, newLow);
        targetHigh = std::min(currentHigh, newHigh);
    }

    if (targetLow == currentLow && targetHigh == currentHigh) {
        return orig;  // same clamps, don't recreate attribute
    }

    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), getFPAttr(ctx, targetLow), getFPAttr(ctx, targetHigh),
                          fpPpeAttr.getScale(), fpPpeAttr.getPreluAlpha(), fpPpeAttr.getBias(), fpPpeAttr.getAdder(),
                          fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

std::optional<SmallVector<double>> PpeFactory::getScale(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    const auto staticScale = fpPpeAttr.getScale();
    if (staticScale == nullptr) {
        return std::nullopt;
    }
    return SmallVector<double>{staticScale.getValueAsDouble()};
}

std::optional<double> PpeFactory::getBias(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    const auto staticBias = fpPpeAttr.getBias();
    if (staticBias == nullptr) {
        return std::nullopt;
    }
    return staticBias.getValueAsDouble();
}

PPEAttr PpeFactory::updateScale(PPEAttr orig, ArrayRef<double> scale) const {
    VPUX_THROW_WHEN(scale.size() != 1,
                    "PPEFp Attribute can only store per-tensor scales, WT must be used for per-channel scales");
    const auto fpPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(),
                          getFPAttr(ctx, scale.front()), fpPpeAttr.getPreluAlpha(), fpPpeAttr.getBias(),
                          fpPpeAttr.getAdder(), fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

PPEAttr PpeFactory::updateBias(PPEAttr orig, double perTensorBias) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(),
                          fpPpeAttr.getScale(), fpPpeAttr.getPreluAlpha(), getFPAttr(ctx, perTensorBias),
                          fpPpeAttr.getAdder(), fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

PPEAttr PpeFactory::discardScaleBias(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(), nullptr,
                          fpPpeAttr.getPreluAlpha(), nullptr, fpPpeAttr.getAdder(), fpPpeAttr.getIn1Mult(),
                          fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

SmallVector<double> PpeFactory::getFpPreluAlpha(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    return parseFPArrayAttr<double>(fpPpeAttr.getPreluAlpha());
}

PPEAttr PpeFactory::updateFpPreluAlpha(PPEAttr orig, ArrayRef<double> fpPreluAlpha) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);

    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(),
                          fpPpeAttr.getScale(), vpux::getFPArrayAttr(ctx, fpPreluAlpha), fpPpeAttr.getBias(),
                          fpPpeAttr.getAdder(), fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

bool PpeFactory::hasQuantScalingThroughPreluAlpha(PPEAttr orig) const {
    // Output quantization scale must be applied through pReluAlpha when SprLUT is enabled.
    const auto fpPpeAttr = castToConcreteAttr(orig);
    return fpPpeAttr.getSprlut() != nullptr;
}

VPU::PPEMode PpeFactory::getMode(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    return fpPpeAttr.getMode().getValue();
}

PPEAttr PpeFactory::updateMode(PPEAttr orig, PPEMode mode) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, PPEModeAttr::get(ctx, mode), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(),
                          fpPpeAttr.getScale(), fpPpeAttr.getPreluAlpha(), fpPpeAttr.getBias(), fpPpeAttr.getAdder(),
                          fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

bool PpeFactory::hasWeightsTable(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    return fpPpeAttr.getScale() == nullptr || fpPpeAttr.getBias() == nullptr;
}

PPEAttr PpeFactory::discardWeightsTableIfPresent(PPEAttr orig, double perTensorScale, double perTensorBias) const {
    if (!hasWeightsTable(orig)) {
        return orig;
    }
    const auto fpPpeAttr = castToConcreteAttr(orig);
    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(),
                          getFPAttr(ctx, perTensorScale), fpPpeAttr.getPreluAlpha(), getFPAttr(ctx, perTensorBias),
                          fpPpeAttr.getAdder(), fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

PPEAttr PpeFactory::discardClamp(vpux::VPU::PPEAttr orig, mlir::Type) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);

    const float clampLow = std::numeric_limits<float>::lowest();
    const float clampHigh = std::numeric_limits<float>::max();
    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), getFPAttr(ctx, clampLow), getFPAttr(ctx, clampHigh),
                          fpPpeAttr.getScale(), fpPpeAttr.getPreluAlpha(), fpPpeAttr.getBias(), fpPpeAttr.getAdder(),
                          fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}

PPEAttr PpeFactory::useWeightsTable(PPEAttr orig) const {
    const auto fpPpeAttr = castToConcreteAttr(orig);
    auto ctx = orig.getContext();
    return PPEFpAttr::get(ctx, fpPpeAttr.getMode(), fpPpeAttr.getClampLow(), fpPpeAttr.getClampHigh(),
                          /* scale= */ nullptr, fpPpeAttr.getPreluAlpha(), /* bias= */ nullptr, fpPpeAttr.getAdder(),
                          fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());
}
