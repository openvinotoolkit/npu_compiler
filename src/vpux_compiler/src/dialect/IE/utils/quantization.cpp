//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

std::optional<int64_t> getFQAxisIndex(IE::FakeQuantizeOp fq, Logger log) {
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
        return tensor.getType()
                .cast<NDTypeInterface>()
                .getElementType()
                .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
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

    const auto postOpName = layerWithPostOp.getPostOp();
    return postOpName.has_value() && postOpName.value().getStringRef() == IE::LeakyReluOp::getOperationName();
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
    return elemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
}

bool IE::checkQuantApproximation(mlir::Operation* op) {
    SmallVector<double> scales;
    const auto outElemType = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto perAxis = outElemType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        std::copy(perAxis.getScales().begin(), perAxis.getScales().end(), std::back_inserter(scales));
    } else if (outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        const auto perTensor = outElemType.cast<mlir::quant::UniformQuantizedType>();
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

    const auto dequantType = maybeDequant.getInput().getType().cast<vpux::NDTypeInterface>();
    if (!allowPerAxisQuantize && !dequantType.getElementType().isa<mlir::quant::UniformQuantizedType>()) {
        return nullptr;
    }

    return maybeDequant.getInput();
}

bool IE::isSymmetricQuantType(mlir::quant::QuantizedType type) {
    // Check that zero points are all 0s
    if (const auto uniformQuantType = type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        return uniformQuantType.getZeroPoint() == 0;
    } else if (const auto uniformPerAxisQuantType = type.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
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
    const auto realType = fqOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto realElemType = realType.getElementType().dyn_cast<mlir::FloatType>();
    const auto outQuantizeElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(),
                             fqOp.getLowFpType(), realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast());
    if (outQuantizeElemType == nullptr) {
        return nullptr;
    }

    return outQuantizeElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
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

IE::FakeQuantizeOp vpux::IE::createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq,
                                      mlir::Location loc) {
    const auto outputType = fq.getOutput().getType().cast<vpux::NDTypeInterface>();
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
                                     .castElemType(argType.cast<vpux::NDTypeInterface>().getElementType()));
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

    auto inElemType = op->getOperand(0).getType().cast<NDTypeInterface>().getElementType();
    auto outElemType = op->getResult(0).getType().cast<NDTypeInterface>().getElementType();
    auto weightsType = op->getOperand(1).getType().cast<NDTypeInterface>().getElementType();

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
