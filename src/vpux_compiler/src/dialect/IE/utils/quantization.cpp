//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

std::optional<int64_t> IE::getFQAxisIndex(IE::FakeQuantizeOp fq, Logger log) {
    const auto extractAxis = [log](mlir::Value input) -> std::optional<int64_t> {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };

        const auto shape = getShape(input);

        const auto axisCount = llvm::count_if(shape, greaterThanOne);
        if (axisCount > 1) {
            log.trace("FakeQuantize constant input with unsupported shape.");
            return std::nullopt;
        }

        auto axis = llvm::find_if(shape, greaterThanOne);
        if (axis != shape.end()) {
            return std::distance(shape.begin(), axis);
        }

        return std::nullopt;
    };

    const auto inputLowAxis = extractAxis(fq.getInputLow());
    const auto outputLowAxis = extractAxis(fq.getOutputLow());

    if (!inputLowAxis && !outputLowAxis) {
        return std::nullopt;
    }

    if (inputLowAxis && outputLowAxis) {
        VPUX_THROW_UNLESS(*inputLowAxis == *outputLowAxis, "FakeQuantize constant inputs use different axis");
    }

    return inputLowAxis ? *inputLowAxis : *outputLowAxis;
}

std::optional<int64_t> IE::getQuantAxisIndex(mlir::Operation* op, Logger log) {
    std::optional<int64_t> axis = std::nullopt;
    const auto getPerAxisQType = [](mlir::Value tensor) {
        return mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                mlir::cast<vpux::NDTypeInterface>(tensor.getType()).getElementType());
    };

    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        axis = getFQAxisIndex(fqOp, log);
    } else if (mlir::isa<IE::DequantizeOp, IE::QuantizeOp>(op)) {
        if (const auto perAxisQType = getPerAxisQType(op->getOperand(0))) {
            axis = perAxisQType.getQuantizedDimension();
        }
        if (const auto perAxisQType = getPerAxisQType(op->getResult(0))) {
            axis = perAxisQType.getQuantizedDimension();
        }
    }

    return axis;
}

bool IE::hasLeakyReLUPostOp(mlir::Operation* op) {
    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
    if (layerWithPostOp == nullptr) {
        return false;
    }

    return mlir::isa_and_nonnull<IE::LeakyReluAttr>(layerWithPostOp.getPostOp());
}

bool IE::hasReLUPostOp(mlir::Operation* op) {
    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
    if (layerWithPostOp == nullptr) {
        return false;
    }

    return mlir::isa_and_nonnull<IE::ReluAttr>(layerWithPostOp.getPostOp());
}

bool IE::hasNegativeScales(mlir::quant::QuantizedType quantType) {
    if (auto perAxisQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
        auto scales = perAxisQuantType.getScales();
        return std::any_of(scales.begin(), scales.end(), [](double scale) {
            return scale < 0.0;
        });
    } else if (auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantType)) {
        return uniformQuantType.getScale() < 0.0;
    }
    return false;
}

bool IE::areAnyUserQuantizeOps(mlir::Operation* op) {
    return llvm::any_of(op->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::QuantizeOp>(op);
    });
}

bool IE::areAllUsersQuantized(mlir::Operation* op) {
    for (auto user : op->getUsers()) {
        if (mlir::dyn_cast<IE::QuantizeOp>(user) == nullptr) {
            return false;
        }
    }
    return true;
}

bool IE::isPerAxisQuant(mlir::Value val) {
    auto elemType = mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType();
    return mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);
}

bool IE::checkQuantApproximation(mlir::Operation* op) {
    SmallVector<double> scales;
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outElemType)) {
        const auto perAxis = mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(outElemType);
        std::copy(perAxis.getScales().begin(), perAxis.getScales().end(), std::back_inserter(scales));
    } else if (mlir::isa<mlir::quant::UniformQuantizedType>(outElemType)) {
        const auto perTensor = mlir::cast<mlir::quant::UniformQuantizedType>(outElemType);
        scales = {perTensor.getScale()};
    } else {
        return false;
    }

    // Check that all scales can be approximated without post-shift (i.e. exponent must fit 15 bits).
    // Negative power is used here because rescaling is computed as scale_in * scale_w / scale_out
    // In case of float input and float weights, scale_in = 1, scale_w = 1, thus we get 1 / scale_out.
    const double scaleLimit = std::pow(2, -15);
    for (const auto& scale : scales) {
        if (std::fabs(scale) < scaleLimit) {
            return false;
        }
    }

    return true;
}

mlir::Value IE::findQuantizedInput(mlir::Value opInput, bool allowPerAxisQuantize) {
    if (opInput == nullptr) {
        return nullptr;
    }

    // When the input is not a DequantizeOp, the pass is not applicable
    auto maybeDequant = opInput.getDefiningOp<IE::DequantizeOp>();
    if (maybeDequant == nullptr) {
        return nullptr;
    }

    const auto dequantType = mlir::cast<vpux::NDTypeInterface>(maybeDequant.getInput().getType());
    if (!allowPerAxisQuantize && !mlir::isa<mlir::quant::UniformQuantizedType>(dequantType.getElementType())) {
        return nullptr;
    }

    return maybeDequant.getInput();
}

bool IE::isSymmetricQuantType(mlir::quant::QuantizedType type) {
    // Check that zero points are all 0s
    if (const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
        return uniformQuantType.getZeroPoint() == 0;
    } else if (const auto uniformPerAxisQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type)) {
        const auto zeroPoints = uniformPerAxisQuantType.getZeroPoints();
        return std::all_of(zeroPoints.begin(), zeroPoints.end(), [](const int64_t zp) {
            return zp == 0;
        });
    }

    return false;
}

mlir::quant::UniformQuantizedType IE::getQuantizedTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp) {
    if (fqOp == nullptr) {
        return nullptr;
    }
    const auto iLoShape = getShape(fqOp.getInputLow());
    const auto iHiShape = getShape(fqOp.getInputHigh());
    const auto oLoShape = getShape(fqOp.getOutputLow());
    const auto oHiShape = getShape(fqOp.getOutputHigh());
    const auto expectedShape = Shape{1, 1, 1, 1};
    if (iLoShape != expectedShape || iHiShape != expectedShape || oLoShape != expectedShape ||
        oHiShape != expectedShape) {
        return nullptr;
    }
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return nullptr;
    }
    const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
    const auto realElemType = mlir::dyn_cast<mlir::FloatType>(realType.getElementType());
    const auto outQuantizeElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(),
                             fqOp.getLowFpType(), realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast());
    if (outQuantizeElemType == nullptr) {
        return nullptr;
    }

    return mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outQuantizeElemType);
}

bool vpux::IE::isPerTensorFQ(ArrayRef<IE::FakeQuantizeOp> fqOps) {
    const auto checkFQAxis = [](IE::FakeQuantizeOp fq) -> bool {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };
        const auto inputLowShape = getShape(fq.getInputLow());
        const auto outputLowShape = getShape(fq.getOutputLow());
        const auto inputAxisCount = llvm::count_if(inputLowShape, greaterThanOne);
        const auto outputAxisCount = llvm::count_if(outputLowShape, greaterThanOne);
        // In case of per axis FQ, make sure that the quantization axis is the same between input and output
        if (inputAxisCount > 0 && outputAxisCount > 0) {
            VPUX_THROW_WHEN(inputLowShape.size() != outputLowShape.size(),
                            "Unaligned tensor rank for FakeQuantize constant inputs.");
            for (size_t i = 0; i < inputLowShape.size(); ++i) {
                VPUX_THROW_WHEN((inputLowShape[Dim(i)] > 1) ^ (outputLowShape[Dim(i)] > 1),
                                "FakeQuantize constant inputs use different axis");
            }
        }
        return (inputAxisCount > 0 || outputAxisCount > 0);
    };

    for (const auto& fqOp : fqOps) {
        if (checkFQAxis(fqOp)) {
            return false;
        }
    }
    return true;
}

bool vpux::IE::hasStaticLowAndHighValues(IE::FakeQuantizeOp fakeQuantizeOp) {
    auto inLow = fakeQuantizeOp.getInputLow();
    auto inHigh = fakeQuantizeOp.getInputHigh();
    auto outLow = fakeQuantizeOp.getOutputLow();
    auto outHigh = fakeQuantizeOp.getOutputHigh();

    auto isDeclareOp = [](mlir::Value value) {
        return value.getDefiningOp<Const::DeclareOp>() != nullptr;
    };

    return isDeclareOp(inLow) && isDeclareOp(inHigh) && isDeclareOp(outLow) && isDeclareOp(outHigh);
}

IE::FakeQuantizeOp vpux::IE::createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq,
                                      mlir::Location loc) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(fq.getOutput().getType());
    const auto newOutputType = outputType.changeShape(getShape(input));
    return rewriter.create<IE::FakeQuantizeOp>(loc, newOutputType, input, fq.getInputLow(), fq.getInputHigh(),
                                               fq.getOutputLow(), fq.getOutputHigh(), fq.getLevelsAttr(),
                                               fq.getLowFpTypeAttr(), fq.getAutoBroadcastAttr());
}

Const::DeclareOp vpux::IE::createFQConst(mlir::MLIRContext* ctx, mlir::Location loc, float val,
                                         mlir::RankedTensorType argType, mlir::PatternRewriter& rewriter) {
    const auto denseElementVal = Const::createConstContent(
            mlir::RankedTensorType::get({1, 1, 1, 1}, mlir::Float32Type::get(ctx)), ArrayRef(val));
    VPUX_THROW_UNLESS(denseElementVal != nullptr, "Failed to generate the denseElementVal.");
    auto cstAttr = Const::ContentAttr::get(
            denseElementVal, Const::ContentSetup(denseElementVal.getType())
                                     .castElemType(mlir::cast<vpux::NDTypeInterface>(argType).getElementType()));
    return rewriter.create<Const::DeclareOp>(loc, argType, std::move(cstAttr));
}

mlir::Value vpux::IE::createFQScaling(mlir::Location loc, mlir::Value input, float scaleFactor, mlir::Type elemType,
                                      std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                                      vpux::IE::AutoBroadcastTypeAttr autoBroadcast, mlir::PatternRewriter& rewriter) {
    // Creates and inserts an FQ which scales the given input by a factor.
    VPUX_THROW_WHEN(scaleFactor > 1.0f, "Superunitary scaling factor causes FQ to overflow: {0} > 1.0", scaleFactor);

    const auto fqArgType = mlir::RankedTensorType::get({}, elemType);
    if (levels.has_value()) {
        // Integer case
        const auto levels_value = *levels;
        VPUX_THROW_WHEN(levels_value != 256 && levels_value != 255, "Got (currently) unsupported levels: {0}",
                        levels_value);

        auto fqLevelsVal = getIntAttr(rewriter, levels_value);
        auto fqLowVal = Const::createFloatConst(rewriter, loc, fqArgType, 0.0f);
        auto fqInHighVal = Const::createFloatConst(rewriter, loc, fqArgType, levels_value - 1);
        auto fqOutHighVal = Const::createFloatConst(rewriter, loc, fqArgType, (levels_value - 1) * scaleFactor);

        auto fq = rewriter.create<IE::FakeQuantizeOp>(loc, input.getType(), input, fqLowVal, fqInHighVal, fqLowVal,
                                                      fqOutHighVal, fqLevelsVal,
                                                      /*lowFpType=*/nullptr, autoBroadcast);
        return fq.getOutput();
    }

    if (lowFpType.has_value()) {
        // Low precision floating-point case
        const auto rangeOrFail = vpux::getFp8Range(*lowFpType);
        VPUX_THROW_WHEN(mlir::failed(rangeOrFail), "Unsupported FQ lowFpType: {0}", *lowFpType);
        const auto lowVal = std::get<0>(*rangeOrFail), highVal = std::get<1>(*rangeOrFail);

        auto fqInLowVal = Const::createFloatConst(rewriter, loc, fqArgType, lowVal);
        auto fqInHighVal = Const::createFloatConst(rewriter, loc, fqArgType, highVal);
        auto fqOutLowVal = Const::createFloatConst(rewriter, loc, fqArgType, lowVal * scaleFactor);
        auto fqOutHighVal = Const::createFloatConst(rewriter, loc, fqArgType, highVal * scaleFactor);

        auto fq = rewriter.create<IE::FakeQuantizeOp>(
                loc, input.getType(), input, fqInLowVal, fqInHighVal, fqOutLowVal, fqOutHighVal,
                /*levels=*/nullptr, mlir::TypeAttr::get(*lowFpType), autoBroadcast);
        return fq.getOutput();
    }

    VPUX_THROW("Neither levels nor lowFpType were provided.");
}

SmallVector<float> vpux::IE::getConst(Const::DeclareOp declOp) {
    const auto content = declOp.getContentAttr().fold();
    return to_small_vector(content.getValues<float>());
}

bool vpux::IE::checkRescaledQuantApproximationForConvBasedOp(mlir::Operation* op) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp>(op)) {
        return true;
    }

    auto inElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto outElemType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType()).getElementType();

    auto inQuantScales = extractScalesOrDefault(inElemType, 1.0);
    auto outQuantScales = extractScalesOrDefault(outElemType, 1.0);
    auto weightsQuantScales = extractScalesOrDefault(weightsType, 1.0);

    const auto OC = getShape(op->getOperand(1))[Dims4D::Filter::OC];
    broadcast(inQuantScales, OC);
    broadcast(outQuantScales, OC);
    broadcast(weightsQuantScales, OC);

    for (int64_t i = 0; i < OC; i++) {
        int16_t mult = 0;
        uint8_t shift = 0;
        int8_t postShift = 0;
        double rescale = (weightsQuantScales[i] * inQuantScales[i]) / outQuantScales[i];
        std::tie(mult, shift, postShift) = approximate<decltype(mult)>(15, rescale);
        if (postShift != 0) {
            return false;
        }
    }

    return true;
}

bool vpux::IE::hasFQSameZeroPoint(IE::FakeQuantizeOp fqOp) {
    auto inLowConstantOp = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConstantOp = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConstantOp = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConstantOp = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConstantOp == nullptr || inHighConstantOp == nullptr || outLowConstantOp == nullptr ||
        outHighConstantOp == nullptr) {
        return false;
    }

    auto allElementsEqual = [](const auto& vec) {
        return std::all_of(vec.begin(), vec.end(), [&](const auto& val) {
            return val == vec.front();
        });
    };

    auto inputScalesAndZeroPoints = getScalesAndZeroPointsFromContentAttr(
            inLowConstantOp.getContentAttr(), inHighConstantOp.getContentAttr(), fqOp.getAutoBroadcast(),
            fqOp.getLevels(), fqOp.getLowFpType(), /*isSigned=*/false);
    if (mlir::failed(inputScalesAndZeroPoints)) {
        return false;
    }
    const auto& inZeroPoints = std::get<1>(inputScalesAndZeroPoints.value());

    if (!allElementsEqual(inZeroPoints)) {
        return false;
    }

    auto outputScalesAndZeroPoints = getScalesAndZeroPointsFromContentAttr(
            outLowConstantOp.getContentAttr(), outHighConstantOp.getContentAttr(), fqOp.getAutoBroadcast(),
            fqOp.getLevels(), fqOp.getLowFpType(), /*isSigned=*/false);
    if (mlir::failed(outputScalesAndZeroPoints)) {
        return false;
    }
    const auto& outZeroPoints = std::get<1>(outputScalesAndZeroPoints.value());
    return allElementsEqual(outZeroPoints);
}

mlir::Type vpux::IE::composeWeightsExpressedType(const mlir::Type convolutionInputType) {
    // Compose quantized weight type for convolution with quantized input.
    // It must share the int/float trait of the input, have scale=1 and shift=0 and sufficient [min, max] interval
    // for storing 1's and 0's.
    // Let's keep it obvious: quantized 0 means 0, quantized 1 means 1.
    // For non-quantized cases just use the provided element type.
    if (const auto inputQuantType = mlir::dyn_cast<mlir::quant::QuantizedType>(convolutionInputType)) {
        const auto ctx = convolutionInputType.getContext();

        // Note: IEEE float types can precisely represent 0.0 and 1.0, this may not hold for all types.
        if (vpux::isFloat8Quantized(inputQuantType)) {
            const auto quantType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/inputQuantType.getStorageType(),
                    /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/inputQuantType.getStorageTypeMin(),
                    /*storageTypeMax=*/inputQuantType.getStorageTypeMax());
            return quantType;
        }

        const auto quantType = mlir::quant::UniformQuantizedType::get(
                /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
        return quantType;
    }
    return convolutionInputType;
}

mlir::FailureOr<double> IE::getQuantizedSplatConstant(mlir::Value input) {
    auto inputOp = input.getDefiningOp();
    if (inputOp == nullptr) {
        return mlir::failure();
    }

    // Possible paths:
    //  - Const.Declare
    //  - IE.FakeQuantize <- Const.Declare
    //  - IE.Dequantize <- Const.Declare
    return llvm::TypeSwitch<mlir::Operation*, mlir::FailureOr<double>>(inputOp)
            .Case<Const::DeclareOp>([&](auto declareOp) -> mlir::FailureOr<double> {
                if (!declareOp.getContentAttr().isSplat()) {
                    return mlir::failure();
                }
                return declareOp.getContent().template getSplatValue<double>();
            })
            .Case<IE::FakeQuantizeOp>([&](auto fqOp) -> mlir::FailureOr<double> {
                auto scalarInputConst = fqOp.getInput().template getDefiningOp<Const::DeclareOp>();
                if (scalarInputConst == nullptr || !scalarInputConst.getContentAttr().isSplat()) {
                    return mlir::failure();
                }

                auto inLowConst = fqOp.getInputLow().template getDefiningOp<Const::DeclareOp>();
                auto inHighConst = fqOp.getInputHigh().template getDefiningOp<Const::DeclareOp>();
                auto outLowConst = fqOp.getOutputLow().template getDefiningOp<Const::DeclareOp>();
                auto outHighConst = fqOp.getOutputHigh().template getDefiningOp<Const::DeclareOp>();

                if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr ||
                    outHighConst == nullptr) {
                    return mlir::failure();
                }

                if (!inLowConst.getContentAttr().isSplat() || !inHighConst.getContentAttr().isSplat() ||
                    !outLowConst.getContentAttr().isSplat() || !outHighConst.getContentAttr().isSplat()) {
                    return mlir::failure();
                }

                const auto inputVal = scalarInputConst.getContent().template getSplatValue<double>();
                const auto inLowConstContentVal = inLowConst.getContent().template getSplatValue<double>();
                const auto inHighConstContentVal = inHighConst.getContent().template getSplatValue<double>();
                const auto outLowConstContentVal = outLowConst.getContent().template getSplatValue<double>();
                const auto outHighConstContentVal = outHighConst.getContent().template getSplatValue<double>();

                if (const auto levels = fqOp.getLevels()) {
                    return fakeQuantize(inputVal, inLowConstContentVal, inHighConstContentVal, outLowConstContentVal,
                                        outHighConstContentVal, levels.value());
                } else {
                    return mlir::failure();
                }
            })
            .Case<IE::DequantizeOp>([&](auto dqOp) -> mlir::FailureOr<double> {
                auto scalarInputConst = dqOp.getInput().template getDefiningOp<Const::DeclareOp>();
                if (scalarInputConst == nullptr || !scalarInputConst.getContentAttr().isSplat()) {
                    return mlir::failure();
                }

                const auto inputElemType = mlir::cast<NDTypeInterface>(dqOp.getInput().getType()).getElementType();
                const auto inputElemQType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputElemType);
                if (inputElemQType == nullptr) {
                    return mlir::failure();
                }

                const auto inputVal = scalarInputConst.getContent().template getSplatValue<double>();
                return dequantizeDouble(inputVal, inputElemQType.getScale(), inputElemQType.getZeroPoint());
            })
            .Default([&](auto) {
                return mlir::failure();
            });
}

bool vpux::IE::isNCEOpCandidatesWithWeights(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MatMulOp>(op);
}

bool nceOpCandidateHasSIWeightsAsInput(mlir::Operation* op) {
    if (!vpux::IE::isNCEOpCandidatesWithWeights(op)) {
        return false;
    }

    auto findNCEOpWeightsAsInput = [](mlir::Operation* op) -> mlir::FailureOr<mlir::Value> {
        mlir::Value filterOperand = op->getOperand(1);

        while (true) {
            if (mlir::isa<mlir::BlockArgument>(filterOperand)) {
                return filterOperand;
            } else if ((IE::isPureViewOp(filterOperand.getDefiningOp()) ||
                        mlir::isa<IE::QuantizeCastOp, IE::DequantizeOp, IE::ConvertOp, IE::SliceOp>(
                                filterOperand.getDefiningOp())) &&
                       filterOperand.hasOneUse()) {
                filterOperand = filterOperand.getDefiningOp()->getOperand(0);
                continue;
            } else {
                break;
            }
        }

        // Return failure if not WAI (no BlockArgument is found)
        return mlir::failure();
    };

    auto weights = findNCEOpWeightsAsInput(op);
    if (mlir::failed(weights)) {
        return false;
    }

    // Verify SI data type
    auto inputElemType = mlir::cast<NDTypeInterface>(weights.value().getType()).getElementType();
    return inputElemType.isSignedInteger();
}

mlir::FailureOr<SmallVector<mlir::Operation*>> findNCEOpCandidatesWithWeights(mlir::Operation* origOp) {
    if (origOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> nceOpCandidatesWithWeights;
    mlir::Operation* currentOp = origOp;

    auto allUsersAreNCEOpCandidates = [](mlir::Operation* op) {
        return llvm::all_of(op->getUsers(), vpux::IE::isNCEOpCandidatesWithWeights);
    };

    while (currentOp) {
        if (vpux::IE::isNCEOpCandidatesWithWeights(currentOp)) {
            // Single NCEOp candidate is found
            return SmallVector<mlir::Operation*>{currentOp};
        }

        if (allUsersAreNCEOpCandidates(currentOp)) {
            // If layer has multiple users, all users should be NCEOp candidate
            for (auto user : currentOp->getUsers()) {
                nceOpCandidatesWithWeights.push_back(user);
            }
            return nceOpCandidatesWithWeights;
        } else {
            // Propagate pure view and quantization layers with single user
            if ((IE::isPureViewOp(currentOp) ||
                 mlir::isa<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::QuantizeOp, IE::DequantizeOp>(
                         currentOp)) &&
                currentOp->hasOneUse()) {
                currentOp = *(currentOp->getUsers().begin());
                continue;
            }
            break;
        }
    }

    // Return failure if no NCEOpCandidate are found
    return mlir::failure();
}

bool vpux::IE::keepIntTypeForSIWeightsAsInput(mlir::Operation* op) {
    const auto moduleOp = getModuleOp(op);
    const auto isAsymmetricPerChannelZeroPointSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
    const auto isAsymmetricPerTensorZeroPointSupported = config::asymmetricPerTensorZeroPointSupported(moduleOp);

    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        if (IE::hasStaticLowAndHighValues(fqOp)) {
            auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
            auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();

            const auto lowAttr = inLowConst.getContentAttr().fold();
            const auto highAttr = inHighConst.getContentAttr().fold();
            const auto isPerAxisQuant = (!lowAttr.isSplat() || !highAttr.isSplat());

            auto isChannelTheOnlyNonOneDim = [](mlir::Value value) {
                auto shape = mlir::cast<NDTypeInterface>(value.getType()).getShape();
                SmallVector<size_t> nonOneDims;
                for (auto index : irange(shape.size())) {
                    if (shape[Dim(index)] != 1) {
                        nonOneDims.push_back(index);
                    }
                }

                if (nonOneDims.size() != 1) {
                    return false;
                }

                if (shape.size() == 3) {
                    return nonOneDims.front() == 0;
                } else if (shape.size() == 4) {
                    return nonOneDims.front() == 1;
                }

                return false;
            };

            const auto isPerChannelQuant =
                    isChannelTheOnlyNonOneDim(fqOp.getInputLow()) || isChannelTheOnlyNonOneDim(fqOp.getInputHigh());

            if (isPerChannelQuant && isAsymmetricPerChannelZeroPointSupported) {
                return false;
            }
            if (!isPerAxisQuant && isAsymmetricPerTensorZeroPointSupported) {
                return false;
            }
        }
    }

    auto isAsymmetricZPSupported = [isAsymmetricPerTensorZeroPointSupported,
                                    isAsymmetricPerChannelZeroPointSupported](NDTypeInterface weightsType) {
        auto elementType = weightsType.getElementType();
        if (mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(elementType) != nullptr &&
            isAsymmetricPerTensorZeroPointSupported) {
            return true;
        } else if (mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(elementType) != nullptr &&
                   isAsymmetricPerChannelZeroPointSupported) {
            return true;
        }

        return false;
    };

    // If current or child NCEOp candidate has SI weights as input, keep Integer data type
    if (isNCEOpCandidatesWithWeights(op)) {
        auto weightsType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
        if (isAsymmetricZPSupported(weightsType)) {
            return false;
        }

        if (nceOpCandidateHasSIWeightsAsInput(op)) {
            return true;
        }
    }

    auto isSIRequiredByAllUsers = llvm::all_of(op->getUsers(), [isAsymmetricZPSupported](const auto& user) {
        auto childNCEOps = findNCEOpCandidatesWithWeights(user);
        if (mlir::succeeded(childNCEOps)) {
            if (llvm::all_of(childNCEOps.value(), [&](mlir::Operation* childNCEOp) {
                    auto weightsType = mlir::cast<NDTypeInterface>(childNCEOp->getOperand(0).getType());
                    return isAsymmetricZPSupported(weightsType);
                })) {
                return false;
            }

            if (llvm::all_of(childNCEOps.value(), [&](mlir::Operation* childNCEOp) {
                    return nceOpCandidateHasSIWeightsAsInput(childNCEOp);
                })) {
                return true;
            }
        }
        return false;
    });

    return isSIRequiredByAllUsers;
}
