//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

mlir::Type IE::rescaleUniformQuantizedType(const mlir::Type tensorType, const double factor) {
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(tensorType);
    VPUX_THROW_UNLESS(ndType != nullptr, "Type {0} does not implement NDTypeInterface", tensorType);
    auto elemType = ndType.getElementType();
    auto uniformQElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    VPUX_THROW_UNLESS(uniformQElemType != nullptr, "Type {0} is not a UniformQuantizedType", elemType);
    const auto scale = uniformQElemType.getScale();
    const auto newScale = static_cast<double>(scale * factor);
    const auto zeroPoint = uniformQElemType.getZeroPoint();

    auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType);
    auto quantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());
    auto resultType = ndType.changeElemType(quantizeElemType);

    return resultType;
}

mlir::quant::UniformQuantizedPerAxisType IE::rescaleUniformQuantizedPerAxisType(
        const mlir::quant::UniformQuantizedPerAxisType perAxisQType, ArrayRef<float> factors) {
    const auto originalScales = perAxisQType.getScales();
    VPUX_THROW_UNLESS(factors.size() == 1 || factors.size() == originalScales.size(),
                      "Factors size {0} must be 1 (broadcast) or match scales size {1}", factors.size(),
                      originalScales.size());
    SmallVector<double> newScales;
    newScales.reserve(originalScales.size());
    const auto factor = factors.front();
    const bool isBroadcast = factors.size() == 1;

    for (size_t i = 0; i < originalScales.size(); ++i) {
        newScales.push_back(originalScales[i] * (isBroadcast ? factor : factors[i]));
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(), newScales,
            perAxisQType.getZeroPoints(), perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

void IE::getFakeQuantParams(vpux::NDTypeInterface qType, int64_t& levels, mlir::RankedTensorType& attrType,
                            mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr) {
    const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(qType.getElementType());
    VPUX_THROW_WHEN(qElemType == nullptr, "Unsupported Quantized Type '{0}'", qType.getElementType());

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qElemType)) {
        float rMin, rMax;
        vpux::getFakeQuantParams(uniformType, levels, rMin, rMax);

        Shape attrShape(qType.getRank(), 1);
        attrType = mlir::RankedTensorType::get(attrShape.raw(), mlir::Float32Type::get(qType.getContext()));
        rMinAttr = Const::createConstContent(attrType, ArrayRef(rMin));
        rMaxAttr = Const::createConstContent(attrType, ArrayRef(rMax));
    } else if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qElemType)) {
        SmallVector<float> rMinVals, rMaxVals;
        vpux::getFakeQuantParams(perAxisQType, levels, rMinVals, rMaxVals);

        const auto axis = Dim(perAxisQType.getQuantizedDimension());

        Shape attrShape(qType.getRank(), 1);
        attrShape[axis] = rMinVals.size();

        attrType = mlir::RankedTensorType::get(attrShape.raw(), mlir::Float32Type::get(qType.getContext()));
        rMinAttr = Const::createConstContent(attrType, ArrayRef(rMinVals));
        rMaxAttr = Const::createConstContent(attrType, ArrayRef(rMaxVals));
    } else {
        VPUX_THROW("Unsupported Quantized Type '{0}'", qElemType);
    }
}

mlir::quant::QuantizedType IE::getQuantizedType(const Const::ContentAttr& lowConst, const Const::ContentAttr& highConst,
                                                std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                                                mlir::FloatType expressedType, bool isSigned, mlir::Location loc,
                                                IE::AutoBroadcastType broadcast, bool ignoreZPCheck,
                                                const Logger& log) {
    const auto innerLog = log.nest("getQuantizedType");

    if (levels.has_value() == lowFpType.has_value()) {
        innerLog.warning("Exactly one of 'levels' or 'lowFpType' must have a value");
        return nullptr;
    }

    auto scalesAndZeroPoints =
            getScalesAndZeroPointsFromContentAttr(lowConst, highConst, broadcast, levels, lowFpType, isSigned, log);
    if (mlir::failed(scalesAndZeroPoints)) {
        innerLog.warning("Unable to retrieve zero points and scales");
        return nullptr;
    }
    const auto [scales, zeroPoints] = *scalesAndZeroPoints;

    const auto lowAttr = lowConst.fold();
    const auto highAttr = highConst.fold();
    const auto isPerAxisQuant = (!lowAttr.isSplat() || !highAttr.isSplat());

    int32_t quantizedDim = 0;
    if (isPerAxisQuant) {
        if (!ignoreZPCheck && !std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin())) {
            innerLog.warning("Zero points are not the same");
            return nullptr;
        }

        auto quantizedDimRef = getQuantizedDimension(lowAttr.getType().getShape(), highAttr.getType().getShape(),
                                                     broadcast, loc, innerLog);
        if (mlir::failed(quantizedDimRef)) {
            innerLog.warning("Failed to get quantized dimension");
            return nullptr;
        }
        quantizedDim = quantizedDimRef.value();
    }

    const auto ctx = lowConst.getContext();

    if (levels.has_value()) {
        const auto storageParams = getStorageParams(ctx, *levels, isSigned);
        VPUX_THROW_WHEN(mlir::failed(storageParams), "Unsupported quantization levels '{0}'", levels);
        const auto [storageMin, storageMax, storageType] = *storageParams;

        if (isPerAxisQuant) {
            return mlir::quant::UniformQuantizedPerAxisType::get(
                    isSigned ? mlir::quant::QuantizationFlags::Signed : 0, storageType, expressedType,
                    std::move(scales), std::move(zeroPoints), quantizedDim, storageMin, storageMax);
        }
        return mlir::quant::UniformQuantizedType::get(isSigned ? mlir::quant::QuantizationFlags::Signed : 0,
                                                      storageType, expressedType, scales[0], zeroPoints[0], storageMin,
                                                      storageMax);
    }

    if (lowFpType.has_value()) {
        const auto lowFpTypeVal = lowFpType.value();
        const auto storageParams = getStorageParams(lowFpTypeVal);
        VPUX_THROW_WHEN(mlir::failed(storageParams), "Unsupported lowFpType '{0}'", lowFpTypeVal);
        const auto [storageMin, storageMax, storageType] = *storageParams;

        if (isLowFpType(lowFpTypeVal)) {
            const auto hasUnsupportedZP = llvm::any_of(zeroPoints, [](int64_t zp) {
                return zp != 0;
            });

            if (hasUnsupportedZP) {
                innerLog.warning("HW unsupported zero point (!= 0) for storage type '{0}'", storageType);
                return nullptr;
            }

            if (isPerAxisQuant) {
                return mlir::quant::UniformQuantizedPerAxisType::get(
                        isSigned ? mlir::quant::QuantizationFlags::Signed : 0, storageType, expressedType,
                        std::move(scales), std::move(zeroPoints), quantizedDim, storageMin, storageMax);
            }
            return mlir::quant::UniformQuantizedType::get(isSigned ? mlir::quant::QuantizationFlags::Signed : 0,
                                                          storageType, expressedType, scales[0], zeroPoints[0],
                                                          storageMin, storageMax);
        }

        if (const auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(lowFpTypeVal)) {
            const auto flags = quantileType.shouldDefaultToSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
            if (isPerAxisQuant) {
                return mlir::quant::UniformQuantizedPerAxisType::getChecked(loc, flags, quantileType, expressedType,
                                                                            std::move(scales), std::move(zeroPoints),
                                                                            quantizedDim, storageMin, storageMax);
            }
            return mlir::quant::UniformQuantizedType::getChecked(loc, flags, quantileType, expressedType, scales[0],
                                                                 zeroPoints[0], storageMin, storageMax);
        }
    }

    VPUX_THROW("Got neither levels (for integer types) nor lowFpType");
}

bool IE::canConvertToSignedQuantizedType(const Const::ContentAttr& inLowConst, const Const::ContentAttr& inHighConst,
                                         const Const::ContentAttr& outLowConst, const Const::ContentAttr& outHighConst,
                                         std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                                         IE::AutoBroadcastType broadcast, const Logger& log) {
    if (levels.has_value() == lowFpType.has_value()) {
        return false;
    }

    auto isAllZeroPoints = [](const auto& zeroPoints) {
        return llvm::all_of(zeroPoints, [](int64_t zp) {
            return zp == 0;
        });
    };

    auto inScalesAndZeroPoints =
            getScalesAndZeroPointsFromContentAttr(inLowConst, inHighConst, broadcast, levels, lowFpType, true, log);
    if (mlir::failed(inScalesAndZeroPoints)) {
        return false;
    }
    if (!isAllZeroPoints(std::get<1>(*inScalesAndZeroPoints))) {
        return false;
    }

    auto outScalesAndZeroPoints =
            getScalesAndZeroPointsFromContentAttr(outLowConst, outHighConst, broadcast, levels, lowFpType, true, log);
    if (mlir::failed(outScalesAndZeroPoints)) {
        return false;
    }
    return isAllZeroPoints(std::get<1>(*outScalesAndZeroPoints));
}

mlir::FailureOr<int32_t> IE::getQuantizedDimension(ShapeRef lowShape, ShapeRef highShape,
                                                   IE::AutoBroadcastType broadcast, mlir::Location loc,
                                                   const Logger& log) {
    const auto innerLog = log.nest("getQuantizedDimension");

    const auto broadcastShapeRes = IE::broadcastEltwiseShape(lowShape, highShape, broadcast, loc);
    if (mlir::failed(broadcastShapeRes)) {
        innerLog.warning("Low values shape '{0}' doesn't match with high values shape '{1}' and cannot be broadcast",
                         lowShape, highShape);
        return mlir::failure();
    }
    const auto broadcastShape = broadcastShapeRes.value();

    auto axisIt = std::find_if(broadcastShape.begin(), broadcastShape.end(), [](int dim) {
        return dim != 1;
    });

    if (axisIt == broadcastShape.end() || std::find_if(axisIt + 1, broadcastShape.end(), [](int dim) {
                                              return dim != 1;
                                          }) != broadcastShape.end()) {
        innerLog.warning("Can't get quantized dimension from shape '{0}'", broadcastShape);
        return mlir::failure();
    }

    return std::distance(broadcastShape.begin(), axisIt);
}

mlir::FailureOr<std::tuple<SmallVector<double>, SmallVector<int64_t>>> IE::getScalesAndZeroPointsFromContentAttr(
        const Const::ContentAttr& lowContentAttr, const Const::ContentAttr& highContentAttr,
        IE::AutoBroadcastType broadcast, const std::optional<int64_t> levels, const std::optional<mlir::Type> lowFpType,
        bool isSigned, const Logger& log) {
    const auto innerLog = log.nest("getScalesAndZeroPointsFromContentAttr");

    if (lowContentAttr == nullptr || highContentAttr == nullptr) {
        innerLog.warning("Failed to obtain the quantization ContentAttr");
        return mlir::failure();
    }

    auto ctx = lowContentAttr.getContext();
    const auto lowContent = lowContentAttr.fold();
    const auto highContent = highContentAttr.fold();

    auto lowVals = to_small_vector(lowContent.getValues<double>());
    auto highVals = to_small_vector(highContent.getValues<double>());
    broadcastRange(lowVals, highVals, broadcast);
    if (lowVals.size() != highVals.size()) {
        innerLog.warning("Low values size '{0}' should equal high values size '{1}' after broadcasting", lowVals.size(),
                         highVals.size());
        return mlir::failure();
    }

    double qMin = 0.;
    double qMax = 0.;
    if (levels.has_value()) {
        const auto storageParams = getStorageParams(ctx, *levels, isSigned);
        if (mlir::failed(storageParams)) {
            return mlir::failure();
        }
        std::tie(qMin, qMax, std::ignore) = *storageParams;
    } else if (lowFpType.has_value()) {
        const auto representableRange = getRepresentableRange(*lowFpType);
        if (mlir::failed(representableRange)) {
            return mlir::failure();
        }
        std::tie(qMin, qMax) = *representableRange;
    } else {
        VPUX_THROW("Got neither levels (for integer types) nor lowFpType");
    }

    const auto dataSize = lowVals.size();
    SmallVector<double> scales(dataSize);
    SmallVector<int64_t> zeroPoints(dataSize);
    bool zeroPointRetrievalFailed = false;

    auto processElement = [&](size_t i) {
        auto scaleAndZeroPoint = calcScaleAndZeroPoint(qMin, qMax, lowVals[i], highVals[i], log);
        if (mlir::failed(scaleAndZeroPoint)) {
            zeroPointRetrievalFailed = true;
            return;
        }
        std::tie(scales[i], zeroPoints[i]) = *scaleAndZeroPoint;
    };

    if (dataSize <= PARALLEL_EXECUTION_THRESHOLD) {
        for (size_t i = 0; i < dataSize; i++) {
            processElement(i);
        }
    } else {
        loop_1d(LoopExecPolicy::Parallel, ctx, dataSize, [&](size_t i) {
            processElement(i);
        });
    }

    if (zeroPointRetrievalFailed) {
        log.warning("Unable to retrieve zero points and scales");
        return mlir::failure();
    }

    return std::make_tuple(std::move(scales), std::move(zeroPoints));
}

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

    if (auto fqOp = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(op)) {
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
    // The boundary is inclusive: scale == 2^-15 gives rescale == 2^15, whose exponent (16) still exceeds
    // the 15 mantissa bits and therefore requires a post-shift.
    const double scaleLimit = std::pow(2, -15);
    for (const auto& scale : scales) {
        if (std::fabs(scale) <= scaleLimit) {
            return false;
        }
    }

    return true;
}

DimArr IE::getLegalActivationQuantAxes(mlir::Operation* op) {
    if (mlir::isa_and_present<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp, IE::MaxPoolOp,
                              IE::AvgPoolOp, IE::AddOp, IE::MultiplyOp>(op)) {
        return {Dims4D::Act::C};
    } else if (mlir::isa_and_present<IE::MatMulOp>(op)) {
        return {Dims4D::Act::C, Dims4D::Act::H};
    }

    return {};
}

DimArr IE::getLegalWeightsQuantAxes(mlir::Operation* op) {
    if (mlir::isa_and_present<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp>(op)) {
        return {Dims4D::Filter::OC};
    } else if (mlir::isa_and_present<IE::MaxPoolOp, IE::AvgPoolOp, IE::AddOp, IE::MultiplyOp>(op)) {
        return {Dims4D::Filter::IC};
    } else if (mlir::isa_and_present<IE::MatMulOp>(op)) {
        return {Dims4D::Filter::IC, Dims4D::Filter::KY};
    }

    return {};
}

mlir::Value IE::findQuantizedInput(mlir::Value opInput, ArrayRef<Dim> allowedQuantAxes) {
    if (opInput == nullptr) {
        return nullptr;
    }

    // When the input is not a DequantizeOp, the pass is not applicable
    auto maybeDequant = opInput.getDefiningOp<IE::DequantizeOp>();
    if (maybeDequant == nullptr) {
        return nullptr;
    }

    const auto dequantType = mlir::cast<vpux::NDTypeInterface>(maybeDequant.getInput().getType());
    if (!mlir::isa_and_present<mlir::quant::UniformQuantizedType>(dequantType.getElementType())) {
        if (allowedQuantAxes.empty()) {
            return nullptr;
        }

        const auto quantType = dequantType.getElementType();

        if (mlir::isa_and_present<mlir::quant::UniformQuantizedSubChannelType>(quantType)) {
            return nullptr;
        }

        if (auto perAxisType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
            auto quantDim = perAxisType.getQuantizedDimension();
            if (std::find(allowedQuantAxes.begin(), allowedQuantAxes.end(), Dim(quantDim)) == allowedQuantAxes.end()) {
                return nullptr;
            }
        }
    }

    return maybeDequant.getInput();
}

mlir::Operation* IE::findDynDequantized(mlir::Value opInput, bool allowPerAxisQuantize) {
    if (opInput == nullptr) {
        return nullptr;
    }

    // When the input is not a DequantizeOp, the pass is not applicable
    auto maybeDequant = opInput.getDefiningOp<IE::DynamicDequantizeOp>();
    if (maybeDequant == nullptr) {
        return nullptr;
    }

    const auto dequantType = mlir::cast<vpux::NDTypeInterface>(maybeDequant.getInput().getType());
    if (!allowPerAxisQuantize && !mlir::isa<mlir::quant::UniformQuantizedType>(dequantType.getElementType())) {
        return nullptr;
    }

    return maybeDequant;
}

bool IE::isSymmetricQuantType(mlir::quant::QuantizedType type) {
    int64_t targetZeroPoint = 0;
    if (auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
        if (auto quantileStorageType = mlir::dyn_cast<vpux::type::QuantileType>(uniformType.getStorageType())) {
            const auto quantileType = quantileStorageType.getQuantileType();
            if (auto intType = mlir::dyn_cast<mlir::IntegerType>(quantileType)) {
                bool isSignedInteger = intType.isSigned();
                const int64_t integerMin =
                        mlir::quant::QuantizedType::getDefaultMinimumForInteger(isSignedInteger, intType.getWidth());
                const int64_t integerMax =
                        mlir::quant::QuantizedType::getDefaultMaximumForInteger(isSignedInteger, intType.getWidth());
                targetZeroPoint = (integerMax + integerMin + 1) / 2;
            }
        }
        return uniformType.getZeroPoint() == targetZeroPoint;
    } else if (const auto uniformPerAxisQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type)) {
        const auto zeroPoints = uniformPerAxisQuantType.getZeroPoints();
        return std::all_of(zeroPoints.begin(), zeroPoints.end(), [targetZeroPoint](const int64_t zp) {
            return zp == targetZeroPoint;
        });
    } else if (mlir::isa<mlir::IntegerType>(type.getStorageType())) {
        const auto qMin = type.getStorageTypeMin();
        const auto qMax = type.getStorageTypeMax();
        targetZeroPoint = (qMax + qMin + 1) / 2;
    } else if (!mlir::isa_and_nonnull<mlir::FloatType>(type.getStorageType())) {
        return false;
    }
    return false;
}

bool IE::areAllQuantTypeZeroPointsEqualToZero(mlir::quant::QuantizedType type) {
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
            denseElementVal, Const::ContentSetup(denseElementVal, denseElementVal.getType())
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
        const auto storageParams = getStorageParams(*lowFpType);
        VPUX_THROW_WHEN(mlir::failed(storageParams), "Unsupported lowFpType '{0}'", *lowFpType);
        const auto [lowVal, highVal, _] = *storageParams;

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
        if (vpux::isLowFpTypeQuantized(inputQuantType)) {
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

int64_t vpux::IE::getMaximumQuantizationLevels(int64_t currentLevels, mlir::Operation* op) {
    if (currentLevels != QuantizationLevels::QUANT_LEVELS_16BIT) {
        return QuantizationLevels::QUANT_LEVELS_8BIT;
    }
    // Output FQ case: allow 16-bit only when this FQ feeds directly into the function return,
    // meaning there are no downstream consumers that could be unable to handle 16-bit input.
    if (auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(op)) {
        const auto hasOnlyReturnUsers = llvm::all_of(fqOp->getResults(), [](mlir::Value result) {
            return result.use_empty() || llvm::all_of(result.getUsers(), [](mlir::Operation* userOp) {
                       return mlir::isa<mlir::func::ReturnOp>(userOp);
                   });
        });
        if (hasOnlyReturnUsers) {
            if (auto quantizedLayerOp =
                        mlir::dyn_cast_if_present<IE::QuantizedLayerOpInterface>(fqOp.getInput().getDefiningOp())) {
                if (quantizedLayerOp.getMaximumQuantizationLevels() == QuantizationLevels::QUANT_LEVELS_16BIT) {
                    return QuantizationLevels::QUANT_LEVELS_16BIT;
                }
            }
        }
    }
    return QuantizationLevels::QUANT_LEVELS_8BIT;
}

bool vpux::IE::isNCEOpCandidatesWithWeights(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MatMulOp>(op);
}

// Walk up an NCE op's filter operand through view-like/quant ops to the originating
// BlockArgument (weights-as-input) or Const::DeclareOp (weights-as-const). Returns failure
// when neither is found (no WAI nor WAC), or when the op has no filter operand.
//
// When firstQuantType is non-null, it is populated with the first quantized element
// type seen while walking the chain. For weights-as-input the terminal BlockArgument may carry a
// bare (non-quantized) type, while the quantized type with its zero-point can be applied by a
// QuantizeCast earlier in the chain, so it must be tracked separately to correctly detect SI
// weights.
static mlir::FailureOr<mlir::Value> findNCEOpWeightsAsInputOrConst(
        mlir::Operation* op, mlir::quant::QuantizedType* firstQuantType = nullptr) {
    if (op == nullptr || op->getNumOperands() < 2) {
        return mlir::failure();
    }

    mlir::Value filterOperand = op->getOperand(1);

    auto trackFirstQuantType = [&](mlir::Value value) {
        if (firstQuantType == nullptr || *firstQuantType != nullptr) {
            return;
        }
        const auto elemType = mlir::cast<NDTypeInterface>(value.getType()).getElementType();
        const auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType);
        if (quantType != nullptr) {
            *firstQuantType = quantType;
        }
    };

    while (true) {
        trackFirstQuantType(filterOperand);
        if (mlir::isa<mlir::BlockArgument>(filterOperand) ||
            mlir::isa<Const::DeclareOp>(filterOperand.getDefiningOp())) {
            return filterOperand;
        } else if (auto concatOp = mlir::dyn_cast_if_present<IE::ConcatOp>(filterOperand.getDefiningOp())) {
            for (auto input : concatOp.getInputs()) {
                if (mlir::isa<mlir::BlockArgument>(input)) {
                    trackFirstQuantType(input);
                    return input;
                }
            }
            break;
        } else if (IE::isPureViewOp(filterOperand.getDefiningOp()) ||
                   mlir::isa<IE::QuantizeCastOp, IE::DequantizeOp, IE::ConvertOp, IE::SliceOp, IE::TransposeOp>(
                           filterOperand.getDefiningOp())) {
            filterOperand = filterOperand.getDefiningOp()->getOperand(0);
            continue;
        } else {
            break;
        }
    }

    // Return failure if no BlockArgument or Const::DeclareOp is found (no WAI nor WAC)
    return mlir::failure();
}

// Predicts whether an NCE op's signed weights will be converted to unsigned by the platform's
// ConvertWeightsToUnsigned strategy. Block-argument (weights-as-input) weights keep their
// external storage type and are never converted, so they are reported as staying signed. Used
// to keep the producer/activation signedness aligned with the final weights signedness.
// Strategy parameter allows amortizing allocations across multiple calls.
static bool willNCEWeightsConvertToUnsigned(mlir::Operation* op,
                                            const IE::IConvertWeightsToUnsignedStrategy* strategy = nullptr) {
    const auto weights = findNCEOpWeightsAsInputOrConst(op);
    if (mlir::failed(weights)) {
        return false;
    }
    if (mlir::isa<mlir::BlockArgument>(weights.value())) {
        return false;
    }
    const auto weightsElemType = mlir::cast<NDTypeInterface>(weights.value().getType()).getElementType();
    const auto weightsQType = mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElemType);
    if (weightsQType == nullptr) {
        return false;
    }

    // Use caller-provided strategy or retrieve one-off if not provided (for backward compatibility).
    std::unique_ptr<IE::IConvertWeightsToUnsignedStrategy> ownedStrategy;
    const IE::IConvertWeightsToUnsignedStrategy* strategyPtr = strategy;
    if (strategyPtr == nullptr) {
        const auto& strategyFactory = IE::getIEStrategyFactory(op->getContext());
        if (strategyFactory == nullptr) {
            return false;
        }
        ownedStrategy = strategyFactory->getConvertWeightsToUnsignedStrategy();
        VPUX_THROW_UNLESS(ownedStrategy != nullptr, "ConvertWeightsToUnsigned strategy is not available");
        strategyPtr = ownedStrategy.get();
    }

    return strategyPtr->tryChangeStorageTypeToUnsigned(weightsQType) != weightsQType;
}

static bool nceOpCandidateHasSIWeightsAsInputOrConst(mlir::Operation* op, bool isAsymmetricPerTensorZeroPointSupported,
                                                     bool isAsymmetricPerChannelZeroPointSupported) {
    if (!vpux::IE::isNCEOpCandidatesWithWeights(op)) {
        return false;
    }

    mlir::quant::QuantizedType firstQuantType = nullptr;
    auto weights = findNCEOpWeightsAsInputOrConst(op, &firstQuantType);
    if (mlir::failed(weights)) {
        return false;
    }

    if (firstQuantType != nullptr) {
        const auto elemTypeSize = vpux::getElemTypeSize(firstQuantType).count();
        if (elemTypeSize >= CHAR_BIT) {
            if (firstQuantType.isSigned() && !mlir::isa<mlir::BlockArgument>(weights.value())) {
                // Verify that Asymmetric Zero Point is supported for the current platform.
                // Keep signed when there is a non-zero zero point and the platform supports it.
                if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(firstQuantType)) {
                    if (llvm::any_of(perAxisType.getZeroPoints(), [](int64_t zp) {
                            return zp != 0;
                        })) {
                        return isAsymmetricPerChannelZeroPointSupported;
                    }
                }
                if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(firstQuantType)) {
                    if (uniformType.getZeroPoint() != 0) {
                        return isAsymmetricPerTensorZeroPointSupported;
                    }
                }
            }
            // For WAI, or for WAC with zero-point=0, the signedness is preserved
            return firstQuantType.isSigned();
        } else {
            // Verify SI data type. The first quantized type seen while walking the chain is used
            // because for weights-as-input the terminal BlockArgument can carry only a bare (non-quantized)
            // type, while the quantized type with its zero-point is applied by a
            // QuantizeCast earlier in the chain.
            auto storageType = firstQuantType.getStorageType();
            if (auto quantileStorageType = mlir::dyn_cast<vpux::type::QuantileType>(storageType)) {
                mlir::Type quantileType = quantileStorageType.getQuantileType();
                if (auto intType = mlir::dyn_cast<mlir::IntegerType>(quantileType)) {
                    return intType.isSigned();
                }
            }

            if (firstQuantType.isSigned()) {
                return true;
            } else {
                if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(firstQuantType)) {
                    return llvm::any_of(perAxisType.getZeroPoints(), [](int64_t zp) {
                        return zp != 0;
                    });
                }
                if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(firstQuantType)) {
                    return uniformType.getZeroPoint() != 0;
                }
            }
        }
    }

    return false;
}

mlir::FailureOr<SmallVector<mlir::Operation*>> findNCEOpCandidatesWithWeights(mlir::Operation* origOp) {
    if (origOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> nceOpCandidatesWithWeights;

    // Recursive function to find all NCE candidates through view ops and quantization layers
    // Return true if all end users are NCE candidates
    std::function<bool(mlir::Operation*, SmallVector<mlir::Operation*>&)> collectNCECandidates;
    collectNCECandidates = [&](mlir::Operation* currentOp, SmallVector<mlir::Operation*>& candidates) {
        if (vpux::IE::isNCEOpCandidatesWithWeights(currentOp)) {
            candidates.push_back(currentOp);
            return true;
        }

        if (IE::isPureViewOp(currentOp) ||
            mlir::isa<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::QuantizeOp, IE::DequantizeOp,
                      IE::QuantizeCastOp, IE::ConcatOp, IE::SliceOp>(currentOp)) {
            return llvm::all_of(currentOp->getUsers(), [&](mlir::Operation* user) {
                return collectNCECandidates(user, candidates);
            });
        }

        return false;
    };

    bool allUsersAreNCEOpCandidates = collectNCECandidates(origOp, nceOpCandidatesWithWeights);
    if (!allUsersAreNCEOpCandidates) {
        return mlir::failure();
    }

    return nceOpCandidatesWithWeights;
}

bool vpux::IE::keepIntTypeForSIWeightsAsInputOrConst(mlir::Operation* op,
                                                     const IE::IConvertWeightsToUnsignedStrategy* strategy) {
    const auto moduleOp = getModuleOp(op);
    const auto isAsymmetricPerChannelZeroPointSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
    const auto isAsymmetricPerTensorZeroPointSupported = config::asymmetricPerTensorZeroPointSupported(moduleOp);

    if (auto fqOp = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(op)) {
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

            auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
            auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

            const auto canConvertToSigned = canConvertToSignedQuantizedType(
                    inLowConst.getContentAttr(), inHighConst.getContentAttr(), outLowConst.getContentAttr(),
                    outHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(), fqOp.getAutoBroadcast());

            if (isPerChannelQuant && !isAsymmetricPerChannelZeroPointSupported && !canConvertToSigned) {
                return false;
            }
            if (!isPerAxisQuant && !isAsymmetricPerTensorZeroPointSupported && !canConvertToSigned) {
                return false;
            }
        }
    }

    // Helper lambda to validate whether SI is required for a given operation
    // Checks if the operation traces to NCE candidates and validates their SI/asymmetric ZP requirements
    auto checkNCECandidatesRequireSI = [isAsymmetricPerTensorZeroPointSupported,
                                        isAsymmetricPerChannelZeroPointSupported,
                                        strategy](mlir::Operation* currentOp) {
        auto nceOps = findNCEOpCandidatesWithWeights(currentOp);
        if (mlir::succeeded(nceOps)) {
            if (llvm::all_of(nceOps.value(), [&](mlir::Operation* nceOp) {
                    return nceOpCandidateHasSIWeightsAsInputOrConst(nceOp, isAsymmetricPerTensorZeroPointSupported,
                                                                    isAsymmetricPerChannelZeroPointSupported);
                })) {
                const bool anyNCECandidateRequireSI = !llvm::all_of(nceOps.value(), [strategy](mlir::Operation* nceOp) {
                    return willNCEWeightsConvertToUnsigned(nceOp, strategy);
                });
                return anyNCECandidateRequireSI;
            }
        }
        return false;
    };

    if (vpux::IE::isNCEOpCandidatesWithWeights(op) && checkNCECandidatesRequireSI(op)) {
        return true;
    }

    return llvm::all_of(op->getUsers(), checkNCECandidatesRequireSI);
}

bool vpux::IE::isQuantizationSupported(IE::QuantizeOp quantizeOp, mlir::Operation* mainOp,
                                       IE::TypeComparisonMode elemComparisonMode) {
    auto quantizeOutputType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
    auto quantizeOutputQType = mlir::cast<mlir::quant::QuantizedType>(quantizeOutputType.getElementType());
    bool isFirstDequantizeOperand = true;
    bool isFirstSignedInteger = false;
    for (auto operand : mainOp->getOperands()) {
        // Only inputs that come from Dequantize should be taken into consideration
        auto dequantizeOp = operand.getDefiningOp<IE::DequantizeOp>();
        if (dequantizeOp != nullptr) {
            auto dequantizeInputType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
            auto dequantizeQInputType = mlir::cast<mlir::quant::QuantizedType>(dequantizeInputType.getElementType());
            if ((quantizeOutputQType.getExpressedType() != dequantizeQInputType.getExpressedType()) ||
                (quantizeOutputQType.getStorageType() != dequantizeQInputType.getStorageType())) {
                if (!IE::bitEnumContainsAny(elemComparisonMode, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
                    return false;
                }
            }

            // In case of integer quantization, the operands of origOp that are produced by Dequantize must have the
            // same signedness
            auto dequantizeStorageType = dequantizeQInputType.getStorageType();
            if (mlir::isa<mlir::IntegerType>(dequantizeStorageType)) {
                // First operand specifies the signedness of the storage data type and it should match with the storage
                // type of the rest of the storage types
                if (isFirstDequantizeOperand) {
                    isFirstSignedInteger = dequantizeQInputType.isSigned();
                    isFirstDequantizeOperand = false;
                } else {
                    bool isCurrentSignedInteger = dequantizeQInputType.isSigned();
                    if (isCurrentSignedInteger != isFirstSignedInteger) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool vpux::IE::isInputQuantizationSupported(mlir::Value activationInput, mlir::Value filterInput) {
    if (activationInput == nullptr || filterInput == nullptr) {
        return false;
    }
    auto activationInputType = mlir::cast<vpux::NDTypeInterface>(activationInput.getType());
    auto filterInputType = mlir::cast<vpux::NDTypeInterface>(filterInput.getType());

    auto activationQInputType = mlir::dyn_cast<mlir::quant::QuantizedType>(activationInputType.getElementType());
    auto filterQInputType = mlir::dyn_cast<mlir::quant::QuantizedType>(filterInputType.getElementType());
    if (activationQInputType == nullptr || filterQInputType == nullptr) {
        return false;
    }
    auto activationIntStorageType = mlir::dyn_cast<mlir::IntegerType>(activationQInputType.getStorageType());
    auto filterIntStorageType = mlir::dyn_cast<mlir::IntegerType>(filterQInputType.getStorageType());
    if (activationIntStorageType != nullptr && filterIntStorageType != nullptr) {
        const bool isActivationSignedInteger = activationQInputType.isSigned();
        const bool isFilterSignedInteger = filterQInputType.isSigned();
        if (isActivationSignedInteger != isFilterSignedInteger) {
            return false;
        }
    }
    return true;
}

bool vpux::IE::checkQuantizePattern(mlir::Type activationStorageType, mlir::Type filterStorageType,
                                    llvm::ArrayRef<std::pair<unsigned, unsigned>> patterns) {
    return llvm::any_of(patterns, [&](auto p) {
        const auto [actBits, wtBits] = p;
        return (activationStorageType.isSignedInteger(actBits) && filterStorageType.isSignedInteger(wtBits)) ||
               (activationStorageType.isUnsignedInteger(actBits) && filterStorageType.isUnsignedInteger(wtBits)) ||
               (activationStorageType.isSignlessInteger(actBits) && filterStorageType.isSignlessInteger(wtBits));
    });
}

bool vpux::IE::hasIdentityQuantizationParams(mlir::Type quantizedElemType) {
    if (auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantizedElemType)) {
        return isDoubleEqual(uniformType.getScale(), 1.0) && uniformType.getZeroPoint() == 0;
    }

    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(quantizedElemType)) {
        if (!llvm::all_of(perAxisType.getScales(), [](double scale) {
                return isDoubleEqual(scale, 1.0);
            })) {
            return false;
        }

        return llvm::all_of(perAxisType.getZeroPoints(), [](int64_t zp) {
            return zp == 0;
        });
    }

    return false;
}

mlir::quant::QuantizedType IE::keepZeroScaleForConstDequant(mlir::quant::QuantizedType qElemType,
                                                            const Const::ContentAttr& lowConst,
                                                            const Const::ContentAttr& highConst,
                                                            IE::AutoBroadcastType broadcast) {
    const auto isZeroRange = [](double low, double high) {
        return isDoubleEqual(low, high) && isDoubleEqual(low, 0.0);
    };

    const auto lowAttr = lowConst.fold();
    const auto highAttr = highConst.fold();

    if (auto perAxisType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(qElemType)) {
        auto lowVals = to_small_vector(lowAttr.getValues<double>());
        auto highVals = to_small_vector(highAttr.getValues<double>());
        broadcastRange(lowVals, highVals, broadcast);

        SmallVector<double> scales(perAxisType.getScales());
        if (scales.size() != lowVals.size() || scales.size() != highVals.size()) {
            return qElemType;
        }

        bool changed = false;
        for (size_t i = 0; i < scales.size(); ++i) {
            if (isZeroRange(lowVals[i], highVals[i]) && !isDoubleEqual(scales[i], 0.0)) {
                scales[i] = 0.0;
                changed = true;
            }
        }
        if (!changed) {
            return qElemType;
        }

        return mlir::quant::UniformQuantizedPerAxisType::get(
                perAxisType.getFlags(), perAxisType.getStorageType(), perAxisType.getExpressedType(), scales,
                SmallVector<int64_t>(perAxisType.getZeroPoints()), perAxisType.getQuantizedDimension(),
                perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
    }

    if (auto perTensorType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedType>(qElemType)) {
        if (lowAttr.isSplat() && highAttr.isSplat() &&
            isZeroRange(lowAttr.getSplatValue<double>(), highAttr.getSplatValue<double>()) &&
            !isDoubleEqual(perTensorType.getScale(), 0.0)) {
            return mlir::quant::UniformQuantizedType::get(
                    perTensorType.getFlags(), perTensorType.getStorageType(), perTensorType.getExpressedType(),
                    /*scale=*/0.0, perTensorType.getZeroPoint(), perTensorType.getStorageTypeMin(),
                    perTensorType.getStorageTypeMax());
        }
    }

    return qElemType;
}
