//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/QuantStorageTypeInterface.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstdint>

using namespace vpux;

namespace {

mlir::FailureOr<std::tuple<double, double>> getStorageRange(mlir::Type type, bool isSigned) {
    if (auto quantStorageType = mlir::dyn_cast<mlir::QuantStorageTypeInterface>(type)) {
        return std::make_tuple(static_cast<double>(quantStorageType.getDefaultMinimum(isSigned)),
                               static_cast<double>(quantStorageType.getDefaultMaximum(isSigned)));
    }
    return mlir::failure();
}

}  // namespace

//
// Utilities for quantized types
//

bool vpux::isSupportedEltwiseQuantization(mlir::Type lhsElemType, mlir::Type rhsElemType, bool allowDifferentScales,
                                          bool allowDifferentZp, VPU::EltwiseType eltwiseType, LogCb logCb) {
    auto lhsQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(lhsElemType);
    auto rhsQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(rhsElemType);

    if (lhsQuantType == nullptr || rhsQuantType == nullptr) {
        return false;
    }

    auto lhsQuantStorageType = vpux::normalizeQuantStorageType(lhsQuantType);
    auto rhsQuantStorageType = vpux::normalizeQuantStorageType(rhsQuantType);

    // Check that the input Dequantize operands have compatible types
    if (lhsQuantType.getExpressedType() != rhsQuantType.getExpressedType() ||
        lhsQuantStorageType != rhsQuantStorageType || lhsQuantType.isSigned() != rhsQuantType.isSigned()) {
        logCb(formatv("Mismatch in inputs quantization parameters"));
        return false;
    }

    if (!allowDifferentZp && (lhsQuantType.getZeroPoint() != rhsQuantType.getZeroPoint())) {
        logCb(formatv("Mismatch in inputs zero points"));
        return false;
    }

    auto lhsQuantScale = lhsQuantType.getScale();
    auto rhsQuantScale = rhsQuantType.getScale();
    // If target architecture does not support different scales, check that they are the same
    if (!allowDifferentScales) {
        // In this case, we'll just program the PPE scale which can support negative values.
        if (!isDoubleEqual(lhsQuantScale, rhsQuantScale)) {
            logCb(formatv("Mismatch in inputs quantization scales"));
            return false;
        }
    } else {
        // Although we support different scale per input tensor, for integer quantized types
        // the HW scale uses 2 U16 register fields meaning we don't support negative scales
        // in that case.
        // For low precision FP types it is not the case because there we use FP16/BF16 register fields.
        // Also, if the scales are identical, then there's no need to use the U16 per input
        // tensor scale and we can use the I16/FP32 scale in the PPE which can support negative scales.
        // For now we support just cases of negative scales which be handled purely by adjusting
        // internal signed PPE scale.
        if (!isDoubleEqual(lhsQuantScale, rhsQuantScale)) {
            if (eltwiseType == VPU::EltwiseType::ADD || eltwiseType == VPU::EltwiseType::SUBTRACT) {
                if ((!isLowFpType(lhsQuantStorageType) && lhsQuantScale < 0) ^
                    (!isLowFpType(rhsQuantStorageType) && rhsQuantScale < 0)) {
                    logCb(formatv("Unsupported negative scales per eltwise input tensors"));
                    return false;
                }
            }
        }
    }

    return true;
}

bool vpux::isSupportedEltwisePerAxisQuantization(mlir::Type lhsElemType, mlir::Type rhsElemType, LogCb logCb) {
    const auto lhsPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(lhsElemType);
    const auto rhsPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(rhsElemType);

    if (lhsPerAxisType == nullptr || rhsPerAxisType == nullptr) {
        logCb(formatv("Per-axis eltwise quantization requires both inputs to be per-axis quantized"));
        return false;
    }

    // Both inputs must have identical quantization: same scales, zero-points, expressed type,
    // storage type, signedness, and quantized dimension.
    if (lhsElemType != rhsElemType) {
        logCb(formatv("Per-axis eltwise inputs must have identical quantization types, got '{0}' and '{1}'",
                      lhsElemType, rhsElemType));
        return false;
    }

    return isSupportedEltwisePerAxisQuantization(lhsElemType, logCb);
}

bool vpux::isSupportedEltwisePerAxisQuantization(mlir::Type perAxisElemType, LogCb logCb) {
    const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(perAxisElemType);

    if (perAxisType == nullptr) {
        logCb(formatv("Expected a per-axis quantized type, got '{0}'", perAxisElemType));
        return false;
    }

    // The WT scale table is indexed by output channel (OC), so quantization must be along the channel axis.
    if (perAxisType.getQuantizedDimension() != Dims4D::Act::C.ind()) {
        logCb(formatv("Per-axis eltwise quantization must be along the channel axis (dim {0}), got dim {1}",
                      Dims4D::Act::C.ind(), perAxisType.getQuantizedDimension()));
        return false;
    }

    return true;
}

bool vpux::isSupportedElemTypeQuantization(mlir::Type elemType, ShapeRef shape, LogCb logCb) {
    if (elemType == nullptr) {
        logCb(formatv("Element type is null"));
        return false;
    }

    const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    if (perAxisQType == nullptr) {
        return true;
    }

    const auto rank = checked_cast<int64_t>(shape.size());
    const auto quantizedDimension = perAxisQType.getQuantizedDimension();
    if (quantizedDimension < 0 || quantizedDimension >= rank) {
        logCb(formatv("Quantized axis '{0}' is out of main type rank '{1}'", quantizedDimension, rank));
        return false;
    }

    const auto dimSize = shape[Dim(static_cast<uint32_t>(quantizedDimension))];
    const auto numScales = perAxisQType.getScales().size();
    if (dimSize != mlir::ShapedType::kDynamic && checked_cast<size_t>(dimSize) != numScales) {
        logCb(formatv("Number of scales '{0}' in per-axis quantized type do not match the quantized "
                      "dimension size '{1}'",
                      numScales, dimSize));
        return false;
    }

    return true;
}

mlir::LogicalResult vpux::validateQuantElemType(mlir::Location loc, vpux::NDTypeInterface mainType) {
    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };
    return mlir::success(isSupportedElemTypeQuantization(mainType.getElementType(), mainType.getShape(), logCb));
}

mlir::Type vpux::normalizeQuantStorageType(mlir::quant::QuantizedType qType) {
    auto elemType = qType.getStorageType();
    auto ctx = qType.getContext();
    if (const auto intType = mlir::dyn_cast_or_null<mlir::IntegerType>(elemType)) {
        return mlir::IntegerType::get(ctx, intType.getWidth(),
                                      qType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
    }
    if (const auto lowFpType = mlir::dyn_cast_or_null<mlir::Float8E4M3FNType>(elemType)) {
        return mlir::Float8E4M3FNType::get(ctx);
    }
    if (const auto lowFpType = mlir::dyn_cast_or_null<mlir::Float8E5M2Type>(elemType)) {
        return mlir::Float8E5M2Type::get(ctx);
    }
    if (const auto lowFpType = mlir::dyn_cast_or_null<mlir::Float4E2M1FNType>(elemType)) {
        return mlir::Float4E2M1FNType::get(ctx);
    }
    if (const auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(elemType)) {
        auto innerStorageType = quantileType.getStorageType();
        if (const auto intType = mlir::dyn_cast<mlir::IntegerType>(innerStorageType)) {
            return mlir::IntegerType::get(ctx, intType.getWidth(),
                                          intType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
        }
        VPUX_THROW("QuantileType has unsupported inner storage type {0}", innerStorageType);
    }
    VPUX_THROW("Unsupported storage element type {0}", elemType);
}

static mlir::quant::UniformQuantizedPerAxisType getPerAxisTypeElem(
        const mlir::quant::UniformQuantizedPerAxisType perAxisQType, llvm::ArrayRef<double> newScales,
        llvm::ArrayRef<int64_t> newZeroPoints) {
    if (const auto quantileStorageType =
                mlir::dyn_cast_if_present<vpux::type::QuantileType>(perAxisQType.getStorageType())) {
        return mlir::quant::UniformQuantizedPerAxisType::get(
                perAxisQType.getFlags(), quantileStorageType, perAxisQType.getExpressedType(), newScales, newZeroPoints,
                perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(),
                perAxisQType.getStorageTypeMax());
    }
    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(), newScales,
            newZeroPoints, perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

static mlir::quant::UniformQuantizedPerAxisType getPerAxisTypeElem(
        const mlir::quant::UniformQuantizedPerAxisType perAxisQType, const int32_t newAxis) {
    if (const auto quantileStorageType =
                mlir::dyn_cast_if_present<vpux::type::QuantileType>(perAxisQType.getStorageType())) {
        return mlir::quant::UniformQuantizedPerAxisType::get(
                perAxisQType.getFlags(), quantileStorageType, perAxisQType.getExpressedType(), perAxisQType.getScales(),
                perAxisQType.getZeroPoints(), newAxis, perAxisQType.getStorageTypeMin(),
                perAxisQType.getStorageTypeMax());
    }
    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), newAxis, perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

// Supported Float type is Float8E5M2, Float8E4M3FN, Float8E4M3, F16, F32, F64, F128 by APFloat
static void extractSubChannelFloatToSmallVector(mlir::DenseElementsAttr scales, const int64_t blockIndex,
                                                const int64_t scalesWidth, mlir::SmallVector<double>& slicedScales) {
    slicedScales.reserve(scalesWidth);
    auto iteratorValues = scales.getValues<llvm::APFloat>();
    auto startIterator = iteratorValues.begin() + (blockIndex * scalesWidth);
    auto stopIterator = iteratorValues.begin() + ((blockIndex + 1) * scalesWidth);

    for (auto beginIt = startIterator; beginIt != stopIterator; beginIt++) {
        double doubleScale = (*beginIt).convertToDouble();
        slicedScales.push_back(doubleScale);
    }
}

// APInt supports any width bit type, so i1, i2, i3, i4, i8, i16, i32 are supported
// If it is either signed or unsigned we can extend with the sign bit or with zero
static void extractSubChannelIntToSmallVector(mlir::DenseElementsAttr zeroPoints, const int64_t blockIndex,
                                              const int64_t zeroPointsWidth, bool isUnsigned,
                                              mlir::SmallVector<int64_t>& slicedZeroPoints) {
    slicedZeroPoints.reserve(zeroPointsWidth);
    auto iteratorValues = zeroPoints.getValues<llvm::APInt>();
    auto startIterator = iteratorValues.begin() + (blockIndex * zeroPointsWidth);
    auto stopIterator = iteratorValues.begin() + ((blockIndex + 1) * zeroPointsWidth);

    for (auto beginIt = startIterator; beginIt != stopIterator; beginIt++) {
        isUnsigned ? slicedZeroPoints.push_back(static_cast<int64_t>((*beginIt).getZExtValue()))
                   : slicedZeroPoints.push_back((*beginIt).getSExtValue());
    }
}

mlir::quant::UniformQuantizedPerAxisType vpux::getPerAxisTypeForBlock(
        mlir::quant::UniformQuantizedSubChannelType subchannelQType, const int64_t blockIndex) {
    const auto scales = subchannelQType.getScales();
    const auto zeroPoints = subchannelQType.getZeroPoints();
    auto scalesShape = scales.getType().getShape();

    // The shape of the tensor is not known so the shape will be [no. of groups, blocksize]
    // For each line in the scales tensor, there will be the scales for that particular group

    VPUX_THROW_UNLESS(blockIndex < scalesShape[0],
                      "Block Index is greater than number of blocks, BlockIndex = {0} , Number of Groups = {1}",
                      blockIndex, scalesShape[0]);

    const auto scalesElementType = mlir::cast<vpux::NDTypeInterface>(scales.getType()).getElementType();
    mlir::SmallVector<double> slicedScales;
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(scalesElementType)) {
        extractSubChannelFloatToSmallVector(scales, blockIndex, scalesShape[1], slicedScales);
    } else {
        VPUX_THROW("Unsupported element type for scales.");
    }

    const auto zeroPointsElementType = mlir::cast<vpux::NDTypeInterface>(zeroPoints.getType()).getElementType();
    mlir::SmallVector<int64_t> slicedZeroPoints;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(zeroPointsElementType)) {
        extractSubChannelIntToSmallVector(zeroPoints, blockIndex, scalesShape[1], intType.isUnsignedInteger(),
                                          slicedZeroPoints);
    } else {
        VPUX_THROW("Unsupported element type for zero points.");
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            subchannelQType.getFlags(), subchannelQType.getStorageType(), subchannelQType.getExpressedType(),
            vpux::ArrayRef<double>(slicedScales), vpux::ArrayRef<int64_t>(slicedZeroPoints),
            subchannelQType.getQuantizedDimensions()[1], subchannelQType.getStorageTypeMin(),
            subchannelQType.getStorageTypeMax());
}

mlir::Type vpux::expandScalesAndZP(mlir::Type perAxisQType, ShapeRef padBefore, ShapeRef padAfter) {
    const auto perAxisUniformQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(perAxisQType);
    VPUX_THROW_UNLESS(perAxisUniformQType != nullptr, "perAxisQType should be a UniformQuantizedPerAxisType!");

    VPUX_THROW_UNLESS(padBefore.size() >= static_cast<size_t>(perAxisUniformQType.getQuantizedDimension()),
                      "Unsupported shape size {0}. Quantized dimension index {1}", padBefore.size(),
                      perAxisUniformQType.getQuantizedDimension());
    VPUX_THROW_UNLESS(padAfter.size() >= static_cast<size_t>(perAxisUniformQType.getQuantizedDimension()),
                      "Unsupported shape size {0}. Quantized dimension index {1}", padAfter.size(),
                      perAxisUniformQType.getQuantizedDimension());

    const auto quantizedDim = Dim(perAxisUniformQType.getQuantizedDimension());

    const auto padBeforeOC = padBefore[quantizedDim];
    const auto padAfterOC = padAfter[quantizedDim];

    if (padBeforeOC == 0 && padAfterOC == 0) {
        return perAxisUniformQType;
    }

    const auto scales = perAxisUniformQType.getScales();
    VPUX_THROW_UNLESS(!scales.empty(), "Can't get value for expand scales.");

    const auto zeroPoints = perAxisUniformQType.getZeroPoints();
    VPUX_THROW_UNLESS(!zeroPoints.empty(), "Can't get value for expand zero points.");

    // Here we need to expand scales & zero points with some values which will allow correct execution of expanded
    // convolution. Some default values (e.g. 1) does not fit here since it may lead to unsupported quantization
    // parameters (e.g. big scale value which approximation does not fit into mult & shift registers of target HW)
    // Heuristic that scales are not that different between each other is used here
    // Technically we need some way to detect if output channels we are processing are expanded ones (fake)
    // And do validation of them accordingly
    std::vector<double> newScales(padBeforeOC, scales.front());
    newScales.insert(newScales.end(), scales.begin(), scales.end());
    newScales.insert(newScales.end(), padAfterOC, scales.back());

    std::vector<int64_t> newZeroPoints(padBeforeOC, zeroPoints.front());
    newZeroPoints.insert(newZeroPoints.end(), zeroPoints.begin(), zeroPoints.end());
    newZeroPoints.insert(newZeroPoints.end(), padAfterOC, zeroPoints.back());

    VPUX_THROW_UNLESS(newScales.size() == newZeroPoints.size(),
                      "Scales & Zero Points must be of the same size, got {0} vs {1} correspondingly", newScales.size(),
                      newZeroPoints.size());
    return getPerAxisTypeElem(perAxisUniformQType, newScales, newZeroPoints);
}

mlir::Type vpux::tileScalesAndZP(mlir::Type perAxisQType, ShapeRef shape, ShapeRef offsets, ShapeRef strides) {
    const auto perAxisUniformQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(perAxisQType);
    VPUX_THROW_UNLESS(perAxisUniformQType != nullptr, "perAxisQType should be a UniformQuantizedPerAxisType!");

    VPUX_THROW_UNLESS(offsets.size() == shape.size(), "Offsets '{0}' doesn't match shape '{1}'", offsets, shape);
    VPUX_THROW_UNLESS(strides.empty() || strides.size() == shape.size(),
                      "Strides '{0}' are not empty and do not match shape '{1}'", strides, shape);

    VPUX_THROW_UNLESS(shape.size() >= static_cast<size_t>(perAxisUniformQType.getQuantizedDimension()),
                      "Unsupported shape size {0}. Quantized dimension index {1}", shape.size(),
                      perAxisUniformQType.getQuantizedDimension());

    const auto qDim = Dim(perAxisUniformQType.getQuantizedDimension());
    const auto qSliceSize = checked_cast<size_t>(shape[qDim]);
    const auto qSliceOffset = checked_cast<size_t>(offsets[qDim]);

    const auto scales = perAxisUniformQType.getScales();
    const auto zeroPoints = perAxisUniformQType.getZeroPoints();

    if (qSliceOffset == 0 && qSliceSize == scales.size()) {
        return perAxisUniformQType;
    }

    auto getTiledElemType = [&](ArrayRef<double> tiledScale, ArrayRef<int64_t> tiledZp) -> mlir::Type {
        return getPerAxisTypeElem(perAxisUniformQType, tiledScale, tiledZp);
    };

    if (strides.empty() || strides[qDim] <= 1) {
        return getTiledElemType(scales.slice(qSliceOffset, qSliceSize), zeroPoints.slice(qSliceOffset, qSliceSize));
    }

    SmallVector<double> newScales;
    SmallVector<int64_t> newZeroPoints;

    for (auto offset = qSliceOffset; newScales.size() < qSliceSize; offset += strides[qDim]) {
        newScales.push_back(scales[offset]);
        newZeroPoints.push_back(zeroPoints[offset]);
    }

    return getTiledElemType(newScales, newZeroPoints);
}

mlir::Type vpux::tileScalesAndZP(mlir::Type perAxisQType, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
    const auto perAxisUniformQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(perAxisQType);
    VPUX_THROW_UNLESS(perAxisUniformQType != nullptr, "perAxisQType should be a UniformQuantizedPerAxisType!");

    const auto inScales = perAxisUniformQType.getScales();
    const auto inZeroes = perAxisUniformQType.getZeroPoints();

    std::vector<double> newScales;
    std::vector<int64_t> newZeroes;
    auto numTiles = offsets.size();
    int64_t length = inScales.size();

    for (size_t k = 0; k < numTiles; k++) {
        VPUX_THROW_UNLESS(offsets[k] + sizes[k] <= length, "Slice exceeds full type length: {0} + {1} > {2}",
                          offsets[k], sizes[k], length);
        newScales.insert(newScales.end(), inScales.begin() + offsets[k], inScales.begin() + offsets[k] + sizes[k]);
        newZeroes.insert(newZeroes.end(), inZeroes.begin() + offsets[k], inZeroes.begin() + offsets[k] + sizes[k]);
    }

    return getPerAxisTypeElem(perAxisUniformQType, newScales, newZeroes);
}

mlir::Type vpux::changeAxis(mlir::Type perAxisQType, int32_t newAxis) {
    const auto perAxisUniformQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(perAxisQType);
    VPUX_THROW_UNLESS(perAxisUniformQType != nullptr, "perAxisQType should be a UniformQuantizedPerAxisType!");

    VPUX_THROW_UNLESS(newAxis >= 0, "Invalid axis {0} was passed", newAxis);

    if (newAxis == perAxisUniformQType.getQuantizedDimension()) {
        return perAxisUniformQType;
    }

    return getPerAxisTypeElem(perAxisUniformQType, newAxis);
}

mlir::quant::QuantizedType vpux::changeStorageType(mlir::quant::QuantizedType qType, mlir::Type storageType) {
    VPUX_THROW_UNLESS(mlir::isa<mlir::IntegerType>(storageType), "Cannot change storage type to non-integer type");

    if (qType.getStorageType() == storageType) {
        return qType;
    }
    if (auto perTensor = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qType)) {
        if (const auto quantileStorageType =
                    mlir::dyn_cast_if_present<vpux::type::QuantileType>(perTensor.getStorageType())) {
            const auto newQuantileTypeStorageType = vpux::type::QuantileType::get(
                    quantileStorageType.getContext(), storageType, quantileStorageType.getQuantileType(),
                    quantileStorageType.getQuantiles());
            return mlir::quant::UniformQuantizedType::get(perTensor.getFlags(), newQuantileTypeStorageType,
                                                          perTensor.getExpressedType(), perTensor.getScale(),
                                                          perTensor.getZeroPoint(), perTensor.getStorageTypeMin(),
                                                          perTensor.getStorageTypeMax());
        }
        return mlir::quant::UniformQuantizedType::get(perTensor.getFlags(), storageType, perTensor.getExpressedType(),
                                                      perTensor.getScale(), perTensor.getZeroPoint(),
                                                      perTensor.getStorageTypeMin(), perTensor.getStorageTypeMax());
    } else if (auto perAxis = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
        if (const auto quantileStorageType =
                    mlir::dyn_cast_if_present<vpux::type::QuantileType>(perAxis.getStorageType())) {
            const auto newQuantileTypeStorageType = vpux::type::QuantileType::get(
                    quantileStorageType.getContext(), storageType, quantileStorageType.getQuantileType(),
                    quantileStorageType.getQuantiles());
            return mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxis.getFlags(), newQuantileTypeStorageType, perAxis.getExpressedType(), perAxis.getScales(),
                    perAxis.getZeroPoints(), perAxis.getQuantizedDimension(), perAxis.getStorageTypeMin(),
                    perAxis.getStorageTypeMax());
        }
        return mlir::quant::UniformQuantizedPerAxisType::get(perAxis.getFlags(), storageType,
                                                             perAxis.getExpressedType(), perAxis.getScales(),
                                                             perAxis.getZeroPoints(), perAxis.getQuantizedDimension(),
                                                             perAxis.getStorageTypeMin(), perAxis.getStorageTypeMax());
    }

    VPUX_THROW("Unsupported original type: {0}", qType);
}

mlir::quant::QuantizedType vpux::changeExpressedType(mlir::quant::QuantizedType quantType, mlir::Type expressedType) {
    return mlir::TypeSwitch<mlir::quant::QuantizedType, mlir::quant::QuantizedType>(quantType)
            .Case<mlir::quant::UniformQuantizedType>([&](const auto qType) {
                if (const auto quantileStorageType =
                            mlir::dyn_cast_if_present<vpux::type::QuantileType>(qType.getStorageType())) {
                    return mlir::quant::UniformQuantizedType::get(qType.getFlags(), quantileStorageType, expressedType,
                                                                  qType.getScale(), qType.getZeroPoint(),
                                                                  qType.getStorageTypeMin(), qType.getStorageTypeMax());
                }
                return mlir::quant::UniformQuantizedType::get(qType.getFlags(), qType.getStorageType(), expressedType,
                                                              qType.getScale(), qType.getZeroPoint(),
                                                              qType.getStorageTypeMin(), qType.getStorageTypeMax());
            })
            .Case<mlir::quant::UniformQuantizedPerAxisType>([&](const auto qType) {
                if (const auto quantileStorageType =
                            mlir::dyn_cast_if_present<vpux::type::QuantileType>(qType.getStorageType())) {
                    return mlir::quant::UniformQuantizedPerAxisType::get(
                            qType.getFlags(), quantileStorageType, expressedType, qType.getScales(),
                            qType.getZeroPoints(), qType.getQuantizedDimension(), qType.getStorageTypeMin(),
                            qType.getStorageTypeMax());
                }
                return mlir::quant::UniformQuantizedPerAxisType::get(
                        qType.getFlags(), qType.getStorageType(), expressedType, qType.getScales(),
                        qType.getZeroPoints(), qType.getQuantizedDimension(), qType.getStorageTypeMin(),
                        qType.getStorageTypeMax());
            })
            .Default([](const auto qType) {
                VPUX_THROW("Unsupported original type: {0}", qType);
                return nullptr;
            });
}

bool vpux::canBeMerged(mlir::Type type1, mlir::Type type2) {
    auto uniformPerAxisType1 = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(type1);
    auto uniformPerAxisType2 = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(type2);

    if (!(uniformPerAxisType1 && uniformPerAxisType2)) {
        return false;
    }

    const auto flags1 = uniformPerAxisType1.getFlags();
    const auto storageType1 = uniformPerAxisType1.getStorageType();
    const auto realType1 = uniformPerAxisType1.getExpressedType();
    const auto qDim1 = uniformPerAxisType1.getQuantizedDimension();
    const auto qMin1 = uniformPerAxisType1.getStorageTypeMin();
    const auto qMax1 = uniformPerAxisType1.getStorageTypeMax();

    const auto flags2 = uniformPerAxisType2.getFlags();
    const auto storageType2 = uniformPerAxisType2.getStorageType();
    const auto realType2 = uniformPerAxisType2.getExpressedType();
    const auto qDim2 = uniformPerAxisType2.getQuantizedDimension();
    const auto qMin2 = uniformPerAxisType2.getStorageTypeMin();
    const auto qMax2 = uniformPerAxisType2.getStorageTypeMax();

    if (!(flags1 == flags2 && storageType1 == storageType2 && realType1 == realType2 && qDim1 == qDim2 &&
          qMin1 == qMin2 && qMax1 == qMax2)) {
        return false;
    }

    auto quantileStorage1 = mlir::dyn_cast<vpux::type::QuantileType>(uniformPerAxisType1.getStorageType());
    auto quantileStorage2 = mlir::dyn_cast<vpux::type::QuantileType>(uniformPerAxisType2.getStorageType());

    if (quantileStorage1 == nullptr && quantileStorage2 == nullptr) {
        return true;
    }

    if (quantileStorage1 && quantileStorage2) {
        return (quantileStorage1.getQuantileType() == quantileStorage2.getQuantileType()) &&
               (quantileStorage1.getQuantiles() == quantileStorage2.getQuantiles());
    }

    return false;
}

mlir::Type vpux::concatScalesAndZP(ArrayRef<mlir::quant::UniformQuantizedPerAxisType> types) {
    VPUX_THROW_WHEN(types.empty(), "Got empty types list in concatScalesAndZP");

    size_t newAxisSize = 0;
    for (const auto type : types) {
        VPUX_THROW_UNLESS(canBeMerged(type, types.front()), "Types '{0}' and '{1}' can't be merged", type,
                          types.front());

        newAxisSize += type.getScales().size();
    }

    SmallVector<double> newScales;
    SmallVector<int64_t> newZeroPoints;

    newScales.reserve(newAxisSize);
    newZeroPoints.reserve(newAxisSize);

    for (const auto type : types) {
        const auto scales = type.getScales();
        const auto zeroPoints = type.getZeroPoints();

        newScales.append(scales.begin(), scales.end());
        newZeroPoints.append(zeroPoints.begin(), zeroPoints.end());
    }

    if (auto uniformPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(types.front())) {
        return getPerAxisTypeElem(uniformPerAxisType, newScales, newZeroPoints);
    }

    VPUX_THROW("Unexpected element type {0} (not a quant per axis type)", types.front());
    return getPerAxisTypeElem(types.front(), newScales, newZeroPoints);
}

std::pair<Scales, ZeroPoints> vpux::extractScalesAndZeroPoints(mlir::Type tensorElemType) {
    const auto qType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(tensorElemType);
    if (const auto uniformParams = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(qType)) {
        SmallVector<double> scales{uniformParams.getScale()};
        SmallVector<int64_t> zeroPoints{uniformParams.getZeroPoint()};

        return {scales, zeroPoints};
    } else if (const auto perAxisParams = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
        SmallVector<double> scales{perAxisParams.getScales().begin(), perAxisParams.getScales().end()};
        SmallVector<int64_t> zeroPoints{perAxisParams.getZeroPoints().begin(), perAxisParams.getZeroPoints().end()};

        return {scales, zeroPoints};
    }

    VPUX_THROW("Unsupported Quantized Type {0}", qType);
}

Scales vpux::extractScalesOrDefault(mlir::Type elemType, double defaultScale) {
    if (elemType == nullptr || !mlir::isa<mlir::quant::QuantizedType>(elemType)) {
        return SmallVector{defaultScale};
    }

    return extractScalesAndZeroPoints(elemType).first;
}

std::optional<int64_t> vpux::extractSingleZeroPoint(mlir::quant::QuantizedType type) {
    auto zeroPoints = extractScalesAndZeroPoints(type).second;
    VPUX_THROW_WHEN(zeroPoints.empty(), "Extracted no zero points");
    const auto canonical = zeroPoints.front();
    const bool zeroPointsEqual = std::all_of(std::next(zeroPoints.begin()), zeroPoints.end(), [&](int64_t zp) {
        return zp == canonical;
    });
    return zeroPointsEqual ? std::make_optional(canonical) : std::nullopt;
}

bool vpux::areAllZeroPointsEqual(mlir::quant::UniformQuantizedPerAxisType type) {
    const auto zeroPoints = type.getZeroPoints();
    if (zeroPoints.empty()) {
        return true;
    }
    const auto firstZeroPoint = zeroPoints[0];
    return std::all_of(zeroPoints.begin(), zeroPoints.end(), [&firstZeroPoint](const int64_t zp) {
        return zp == firstZeroPoint;
    });
}

mlir::IntegerAttr vpux::getPerTensorZeroPointAttr(mlir::Value value) {
    const auto elementType = mlir::cast<vpux::NDTypeInterface>(value.getType()).getElementType();

    const auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(elementType);
    if (quantizedType == nullptr) {
        return nullptr;
    }

    if (const auto perTensorType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantizedType)) {
        return mlir::Builder(value.getContext()).getI64IntegerAttr(perTensorType.getZeroPoint());
    }

    if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(quantizedType)) {
        if (areAllZeroPointsEqual(perAxisType)) {
            return mlir::Builder(value.getContext()).getI64IntegerAttr(perAxisType.getZeroPoints().front());
        }
    }

    return nullptr;
}

vpux::QuantizationApproximation::QuantizationApproximation(double target): _mult(0), _shift(0), _postShift(0) {
    std::tie(_mult, _shift, _postShift) = approximate<decltype(_mult)>(15, target);

    VPUX_THROW_WHEN(_postShift != 0,
                    "Encountered an attempt to approximate {0} as mult = {1}, shift = {2}, postShift = {3}, "
                    "but postShift is not supported",
                    target, mult(), shift(), postShift());
}

int64_t vpux::QuantizationApproximation::mult() const {
    return _mult;
}

int64_t vpux::QuantizationApproximation::shift() const {
    return _shift;
}

int64_t vpux::QuantizationApproximation::postShift() const {
    return _postShift;
}

void vpux::QuantizationApproximation::setMult(int32_t mult) {
    _mult = mult;
}

void vpux::QuantizationApproximation::setShift(uint8_t shift) {
    _shift = shift;
}

vpux::PReLUApproximation::PReLUApproximation(double target): _mult(0), _shift(0) {
    // TODO return logic for 11 bits for quantized case VPUX37XX back as soon as it works.
    const auto bits = 11;
    int8_t postShift = 0;
    std::tie(_mult, _shift, postShift) = approximate<decltype(_mult)>(bits, target);

    VPUX_THROW_UNLESS(postShift == 0,
                      "Encountered an attempt to approximate {0} as mult = {1}, shift = {2}, postShift = {3}, "
                      "but postShift is not supported",
                      target, mult(), shift(), int64_t(postShift));
}

int64_t vpux::PReLUApproximation::mult() const {
    return _mult;
}

int64_t vpux::PReLUApproximation::shift() const {
    return _shift;
}

mlir::FailureOr<int64_t> vpux::extractScalarOrUniformZP(mlir::quant::QuantizedType quantizedType) {
    // Returns the single ZP of a quantized type. If the type has more than one distinct ZP, the function fails.
    // Useful for signaling ignored ZP's in areas which only support a single ZP.
    const auto zps = extractScalesAndZeroPoints(quantizedType).second;
    const auto firstZP = zps.front();

    const auto hasNonUniformZp = llvm::any_of(zps, [&firstZP](const auto zp) {
        return zp != firstZP;
    });
    if (hasNonUniformZp == false) {
        return firstZP;
    }

    return mlir::failure();
}

bool vpux::hasScalarOrUniformZP(mlir::quant::QuantizedType quantizedType) {
    return mlir::succeeded(extractScalarOrUniformZP(quantizedType));
}

//
// FakeQuantize support
//

mlir::FailureOr<std::tuple<double, int64_t>> vpux::calcScaleAndZeroPoint(double qMinFP, double qMaxFP, double rMin,
                                                                         double rMax, const Logger& log) {
    const auto innerLog = log.nest("calcScaleAndZeroPoint");

    // Is the given range actually a range or a single scalar like [-0.00, 0.00] or [3, 3]?
    if (std::fabs(rMax - rMin) < std::numeric_limits<double>::epsilon()) {
        const double scale = rMin;
        // (-inf, -eps] => scale = rMin, zp = 2
        // (-eps, eps) => scale = 1.0, zp = 0
        // [eps, inf) => scale = rMin, zp = 0

        if (std::fabs(scale) < std::numeric_limits<double>::epsilon()) {
            // "-epsilon < scale < epsilon" means that scale should be zero  ===>  scale = 1.0
            // to avoid division by zero in formula Q = R/scale + zp
            return std::make_tuple(1.0, static_cast<int64_t>(0));
        }
        if (scale >= std::numeric_limits<double>::epsilon()) {
            return std::make_tuple(scale, static_cast<int64_t>(0));
        }
        if (scale <= -std::numeric_limits<double>::epsilon()) {
            // Due to LLVM limitation scale must be >=0
            // thirdparty/llvm-project/mlir/lib/Dialect/Quant/IR/QuantTypes.cpp lines 278-280
            // quantized_value = real_value / scale + zero_point
            // real_value = (quantized_value - zero_point) * scale
            // As a workaround for a negative real value scalar -R
            // 1. apply positive scale as usual: -R/scale = -1
            // 2. set zero point to 2, which gives us Q = (-R/scale) + 2 = -1 + 2 = 1
            return std::make_tuple(scale * (-1), static_cast<int64_t>(2));
        }

        innerLog.warning("Unhandled scale value.");
        return mlir::failure();
    }

    // Ranges that do not contain zero will generate negative zero-point which is not supported in DPU PPE pipeline.
    // Also rMin > rMax ranges are valid; they are used to signal the presence of negative scales.
    // Currently there is no other information in FakeQuantize operation, to deduce back the negative scale
    // based on just the range information.
    const auto doesRangeContainZero = (rMin <= 0 && rMax >= 0) || (rMin >= 0 && rMax <= 0);
    if (!doesRangeContainZero) {
        innerLog.warning("Real values range does not contain value zero ['{0}', '{1}']", rMin, rMax);
        return mlir::failure();
    }

    //
    // Determine the scale.
    //

    const double scale = (rMax - rMin) / (qMaxFP - qMinFP);

    // In case of rMin > rMax, rMin > 0, rMax < 0, (rMax - rMin) > epsilon, scale < 0, |scale| < epsilon
    const double minScale = std::numeric_limits<double>::epsilon();
    if (std::fabs(scale) < minScale) {
        // very low negative scale
        if (scale < 0) {
            return std::make_tuple(minScale * (-1), static_cast<int64_t>(0));
        } else {  // very low positive scale
            return std::make_tuple(minScale, static_cast<int64_t>(0));
        }
    }

    if (std::fabs(scale) <= std::numeric_limits<double>::epsilon()) {
        innerLog.warning("Quantization scale is too small : '{0}'", scale);
        return mlir::failure();
    }

    //
    // Zero point computation.
    //

    double x = qMinFP - rMin / scale;
    int64_t zp = static_cast<int64_t>(std::round(x));

    return std::make_tuple(scale, zp);
}

int64_t vpux::calculateZeroPoint(double low, double high, int levels, mlir::IntegerType type) {
    VPUX_THROW_UNLESS((low <= 0.f) && (high >= 0.f) && (low != high), "Wrong low and high values.");
    VPUX_THROW_UNLESS(levels <= 256, "Levels must be less then 256.");

    int64_t zeroPoint = 0;

    if (type.isUnsignedInteger()) {
        auto x = -static_cast<double>(levels - 1) * low / (high - low);
        zeroPoint = static_cast<int64_t>(std::round(x));
    } else if (type.isSignedInteger()) {
        auto x = -static_cast<double>(levels - 1) * ((high + low) * 0.5f) / (high - low);
        zeroPoint = static_cast<int64_t>(std::round(x));
    } else {
        VPUX_THROW("Unsupported element type {0}.", type);
    }

    return zeroPoint;
}

mlir::FailureOr<int64_t> vpux::getSingleZeroPointOrFail(mlir::quant::QuantizedType quantType) {
    if (auto perAxis = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
        auto zeroPoints = perAxis.getZeroPoints();
        VPUX_THROW_WHEN(zeroPoints.empty(), "Missing zero points from UniformQuantizedPerAxisType object.");

        const auto zpFront = zeroPoints[0];
        const bool allZpEqual = std::all_of(zeroPoints.begin(), zeroPoints.end(), [zpFront](auto zp) {
            return zp == zpFront;
        });
        if (allZpEqual) {
            return zpFront;
        }

        return mlir::failure();
    } else if (auto perTensor = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(quantType)) {
        return perTensor.getZeroPoint();
    }

    VPUX_THROW("Unsupported quantized type '{0}': only UniformQuantizedPerAxisType and UniformQuantizedType and "
               "derived types are supported.",
               quantType);
    return mlir::failure();
}

int64_t vpux::getDefaultQuantizedZeroPoint(mlir::quant::QuantizedType quantType) {
    const auto getDefaultIntegerZp = [](mlir::IntegerType intType, bool isSigned) -> int64_t {
        const auto bitwidth = intType.getWidth();
        auto qMin = mlir::quant::QuantizedType::getDefaultMinimumForInteger(isSigned, bitwidth);
        auto qMax = mlir::quant::QuantizedType::getDefaultMaximumForInteger(isSigned, bitwidth);
        // u8: min = 0, max = 255
        // zp = (255 + 0 + 1) / 2 = 128
        // i8: min = -128, max = 127
        // zp = (127 + -128 + 1) / 2 = 0
        // u4: min = 0, max = 15
        // zp = (15 + 0 + 1) / 2 = 8
        // i4: min = -8, max = 7
        // zp = (7 + -8 + 1) / 2 = 0
        return (qMax + qMin + 1) / 2;
    };

    // TODO #E-164790: Add handling of UniformQuantized types to check the quantileType field for the signedness
    // instead of the storageType as part of [#E-164790] fix
    if (mlir::isa_and_nonnull<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(quantType.getStorageType())) {
            return getDefaultIntegerZp(intType, quantType.isSigned());
        }
    } else {
        VPUX_THROW("Unsupported quantized type '{0}'", quantType);
    }

    // return 0 as a default mid range zp value in case of floating point storage type or quantile type
    return 0;
}

SmallVector<int64_t> vpux::getQuantizedTypeZeroPoints(mlir::quant::QuantizedType quantType) {
    if (auto perAxis = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
        auto zeroPoints = perAxis.getZeroPoints();
        if (!zeroPoints.empty()) {
            return to_small_vector(zeroPoints);
        }

        VPUX_THROW("Missing zero points from UniformQuantizedPerAxisType object.");
    } else if (auto perTensor = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(quantType)) {
        return SmallVector<int64_t>{perTensor.getZeroPoint()};
    }

    VPUX_THROW("Unsupported quantized type '{0}': only UniformQuantizedPerAxisType and UniformQuantizedType and "
               "derived types are supported.",
               quantType);
}

bool vpux::isSymmetricZeroPoint(mlir::quant::QuantizedType quantType) {
    const auto targetZeroPoint = getDefaultQuantizedZeroPoint(quantType);
    const auto isSymmetricZP = [targetZeroPoint](const int64_t zp) -> bool {
        return zp == targetZeroPoint;
    };

    auto zeroPoints = getQuantizedTypeZeroPoints(quantType);
    return std::all_of(zeroPoints.begin(), zeroPoints.end(), isSymmetricZP);
}

mlir::FailureOr<std::tuple<double, double, mlir::Type>> vpux::getStorageParams(mlir::MLIRContext* ctx, int64_t levels,
                                                                               bool isSigned) {
    auto min = 0.;
    auto max = static_cast<double>(levels - 1);
    mlir::Type storageType;

    switch (levels) {
    case 256:
        if (isSigned) {
            min = -128.;
            max = 127.;
            storageType = getSInt8Type(ctx);
        } else {
            storageType = getUInt8Type(ctx);
        }
        break;

    case 255:
        if (isSigned) {
            min = -127.;
            max = 127.;
            storageType = getSInt8Type(ctx);
        } else {
            storageType = getUInt8Type(ctx);
        }
        break;

    case 65536:
        if (isSigned) {
            min = -32768.;
            max = 32767.;
            storageType = getSInt16Type(ctx);
        } else {
            storageType = getUInt16Type(ctx);
        }
        break;

    case 16:
        if (isSigned) {
            min = -8.;
            max = 7.;
            storageType = getSInt4Type(ctx);
        } else {
            storageType = getUInt4Type(ctx);
        }
        break;

    case 15:
        if (isSigned) {
            min = -7.;
            max = 7.;
            storageType = getSInt4Type(ctx);
        } else {
            storageType = getUInt4Type(ctx);
        }
        break;

    case 4:
        if (isSigned) {
            min = -2.;
            max = 1.;
            storageType = getSInt2Type(ctx);
        } else {
            storageType = getUInt2Type(ctx);
        }
        break;

    // Because in the absence of I1 support, we must use U8 datatype.
    // [Track number: E#24341].
    case 2:
        if (isSigned) {
            min = 0.;
            max = 1.;
            storageType = getSInt8Type(ctx);
        } else {
            storageType = getUInt8Type(ctx);
        }
        break;

    default:
        return mlir::failure();
    }

    return std::tuple(min, max, storageType);
}

mlir::FailureOr<std::tuple<double, double, mlir::Type>> vpux::getStorageParams(mlir::Type type) {
    double min = 0.;
    double max = 0.;
    mlir::Type storageType;

    if (const auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(type)) {
        storageType = quantileType;

        const auto range = getStorageRange(quantileType, quantileType.shouldDefaultToSigned());
        if (mlir::failed(range)) {
            return mlir::failure();
        }

        min = std::get<0>(*range);
        max = std::get<1>(*range);
    } else if (const auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        // SI8 / UI8 / SI4 / UI4
        if (intType.isSignless()) {
            return mlir::failure();
        }
        const auto range = getStorageRange(intType, intType.isSigned());
        if (mlir::failed(range)) {
            return mlir::failure();
        }
        min = std::get<0>(*range);
        max = std::get<1>(*range);
        storageType = type;
    } else if (isLowFpType(type)) {
        // Low FP types (Float8, Float4, etc.)
        // All low-FP data types are signed
        const auto range = getStorageRange(type, /*isSigned=*/true);
        if (mlir::failed(range)) {
            return mlir::failure();
        }
        min = std::get<0>(*range);
        max = std::get<1>(*range);
        storageType = type;
    } else {
        return mlir::failure();
    }

    return std::tuple(min, max, storageType);
}

mlir::FailureOr<std::tuple<double, double>> vpux::getRepresentableRange(mlir::Type lowPrecisionType) {
    // Quantile float types (NF4)
    if (const auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(lowPrecisionType)) {
        const auto quantileTable = quantileType.getQuantiles();
        return std::tuple(quantileTable.front(), quantileTable.back());
    }

    // For the other types, the storage range is also the representable range.
    const auto storageParams = getStorageParams(lowPrecisionType);
    if (mlir::failed(storageParams)) {
        return mlir::failure();
    }

    const auto [min, max, _] = *storageParams;
    return std::tuple(min, max);
}

bool vpux::isFloat8(mlir::Type type) {
    return mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(type);
}

bool vpux::isFloat8Quantized(mlir::Type type) {
    const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type);
    if (qType == nullptr) {
        return false;
    }

    const auto storageType = qType.getStorageType();
    return isFloat8(storageType);
}

bool vpux::isInt8(mlir::Type type) {
    return mlir::isa<mlir::IntegerType>(type) && type.getIntOrFloatBitWidth() == CHAR_BIT;
}

bool vpux::isInt8Quantized(mlir::Type type) {
    const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type);
    if (qType == nullptr) {
        return false;
    }

    const auto storageType = qType.getStorageType();
    return isInt8(storageType) && qType.isSigned();
}

bool vpux::isFloat4(mlir::Type type) {
    return mlir::isa<mlir::Float4E2M1FNType>(type);
}

bool vpux::isFloat4Quantized(mlir::Type type) {
    const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type);
    if (qType == nullptr) {
        return false;
    }

    const auto storageType = qType.getStorageType();
    return isFloat4(storageType);
}

bool vpux::isLowFpType(mlir::Type type) {
    if (isFloat8(type)) {
        return true;
    }
    if (isFloat4(type)) {
        return true;
    }
    return false;
}

bool vpux::isLowFpTypeQuantized(mlir::Type type) {
    if (isFloat8Quantized(type)) {
        return true;
    }
    if (isFloat4Quantized(type)) {
        return true;
    }
    return false;
}

bool vpux::isNF4SpecQuantized(mlir::Type type) {
    const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type);
    if (qType == nullptr) {
        return false;
    }
    const auto bitWidth = qType.getStorageTypeIntegralWidth();
    if (bitWidth != 4) {
        return false;
    }

    if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(qType)) {
        if (const auto quantileStorageType = mlir::dyn_cast<vpux::type::QuantileType>(qType.getStorageType())) {
            return quantileStorageType.getQuantiles() == vpux::type::NF4Type::getSpecQuantiles();
        }
    }
    return false;
}

void vpux::getFakeQuantParams(mlir::quant::UniformQuantizedType qElemType, int64_t& levels, float& rMin, float& rMax) {
    const auto qMin = qElemType.getStorageTypeMin();
    const auto qMax = qElemType.getStorageTypeMax();

    levels = qMax - qMin + 1;

    const auto scale = qElemType.getScale();
    const auto zeroPoint = qElemType.getZeroPoint();

    rMin = dequantize(qMin, scale, zeroPoint);
    rMax = dequantize(qMax, scale, zeroPoint);
}

void vpux::getFakeQuantParams(mlir::quant::UniformQuantizedPerAxisType qElemType, int64_t& levels,
                              SmallVectorImpl<float>& rMinVals, SmallVectorImpl<float>& rMaxVals) {
    const auto qMin = qElemType.getStorageTypeMin();
    const auto qMax = qElemType.getStorageTypeMax();

    levels = qMax - qMin + 1;

    const auto scales = qElemType.getScales();
    const auto zeroPoints = qElemType.getZeroPoints();

    rMinVals.resize(scales.size());
    rMaxVals.resize(scales.size());

    for (size_t i = 0; i < scales.size(); ++i) {
        rMinVals[i] = dequantize(qMin, scales[i], zeroPoints[i]);
        rMaxVals[i] = dequantize(qMax, scales[i], zeroPoints[i]);
    }
}

double vpux::fakeQuantize(double inVal, double inLow, double inHigh, double qLow, double qHigh, int64_t levels) {
    if (inVal <= inLow) {
        return qLow;
    } else if (inVal > inHigh) {
        return qHigh;
    } else {
        return std::round((inVal - inLow) / (inHigh - inLow) * (levels - 1)) / (levels - 1) * (qHigh - qLow) + qLow;
    }
}

/**
 * @brief Infers UniformQuantizedPerAxisType after shape cast for NHWC layout.
 *
 * Used mainly for adjustConvolutionShape pass. Supports changes only in channel (C) and width (W) dimensions;
 * other dimensions must remain unchanged. Adjusts quantization parameters (scales/zero points) accordingly.
 */
std::optional<mlir::quant::UniformQuantizedPerAxisType> vpux::inferPerAxisQuantizedTypeAfterShapeCastOrNull(
        const mlir::Type inType, ArrayRef<int64_t> outShape, LogCb logCb) {
    auto inNDType = mlir::dyn_cast<vpux::NDTypeInterface>(inType);
    if (inNDType == nullptr) {
        logCb(formatv("Type {0} does not implement NDTypeInterface", inType));
        return std::nullopt;
    }

    auto perAxisUniformQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inNDType.getElementType());
    if (perAxisUniformQType == nullptr) {
        logCb(formatv("Only UniformQuantizedPerAxisType is supported"));
        return std::nullopt;
    }

    if (inNDType.getDimsOrder() != DimsOrder::NHWC) {
        logCb(formatv("Only NHWC layout is supported"));
        return std::nullopt;
    }

    const auto inShape = inNDType.getShape().raw();
    if (inShape.size() != outShape.size()) {
        logCb(formatv("Input and output shapes must have the same rank. Input shape: {0}, Output shape: {1}", inShape,
                      outShape));
        return std::nullopt;
    }

    const auto quantizedAxis = perAxisUniformQType.getQuantizedDimension();
    if (quantizedAxis < 0 || checked_cast<size_t>(quantizedAxis) >= inShape.size()) {
        logCb(formatv("Quantized axis {0} is out of range for input shape {1}", quantizedAxis, inShape));
        return std::nullopt;
    }

    // If the quantized axis dimension hasn't changed, we can safely adjust the shape
    if (inShape[quantizedAxis] == outShape[quantizedAxis]) {
        return perAxisUniformQType;
    }

    if (quantizedAxis != Dims4D::Act::C.ind()) {
        logCb(formatv("Only adjustment of quantized axis == 1 (channel) is supported. Got quantizedAxis={0} for "
                      "inShape={1}, outShape={2}",
                      quantizedAxis, inShape, outShape));
        return std::nullopt;
    }

    const auto wDimIdx = Dims4D::Act::W.ind();
    if (checked_cast<size_t>(wDimIdx) >= inShape.size()) {
        logCb(formatv("W axis {0} is out of range for input shape {1}", wDimIdx, inShape));
        return std::nullopt;
    }

    // Only C and W dimension changes are supported at the moment
    for (size_t i = 0; i < inShape.size(); ++i) {
        if (i != static_cast<size_t>(quantizedAxis) && i != static_cast<size_t>(wDimIdx)) {
            if (inShape[i] != outShape[i]) {
                logCb(formatv("Only quantized axis and W dimension changes are supported. "
                              "Input shape: {0}, Output shape: {1}, Quantized axis: {2}, W axis: {3}, Mismatched dim: "
                              "{4}",
                              inShape, outShape, quantizedAxis, wDimIdx, i));
                return std::nullopt;
            }
        }
    }

    if (inShape[quantizedAxis] <= 0 || inShape[wDimIdx] <= 0) {
        logCb(formatv("Input quantized axis and W dimensions must be positive. Input shape: {0}", inShape));
        return std::nullopt;
    }

    const double dimQRescaleFactor = static_cast<double>(outShape[quantizedAxis]) / inShape[quantizedAxis];
    const double dimWRescaleFactor = static_cast<double>(outShape[wDimIdx]) / inShape[wDimIdx];

    if (!std::isfinite(dimQRescaleFactor) || !std::isfinite(dimWRescaleFactor) || dimQRescaleFactor <= 0.0 ||
        dimWRescaleFactor <= 0.0) {
        logCb(formatv("Invalid quantized axis or W rescale factors: dimQRescaleFactor={0}, dimWRescaleFactor={1}. "
                      "Shape change: {2} -> {3}",
                      dimQRescaleFactor, dimWRescaleFactor, inShape, outShape));
        return std::nullopt;
    }

    if (std::abs(dimQRescaleFactor * dimWRescaleFactor - 1.0) >= std::numeric_limits<double>::epsilon()) {
        logCb(formatv("Only shape-cast operations like reshape are supported for quantized tensors. "
                      "Expected equilibrium between quantized axis and W axis rescale factors, but got: "
                      "dimQRescaleFactor={0}, dimWRescaleFactor={1}. "
                      "Shape change: {2} -> {3}",
                      dimQRescaleFactor, dimWRescaleFactor, inShape, outShape));
        return std::nullopt;
    }

    const auto& scales = perAxisUniformQType.getScales();
    const auto& zeroPoints = perAxisUniformQType.getZeroPoints();
    const auto hasZeroPoints = !zeroPoints.empty();
    if (hasZeroPoints && zeroPoints.size() != scales.size()) {
        logCb(formatv("Zero-point size ({0}) does not match scale size ({1})", zeroPoints.size(), scales.size()));
        return std::nullopt;
    }
    llvm::SmallVector<double> newScales;
    llvm::SmallVector<int64_t> newZeroPoints;

    if (dimQRescaleFactor > 1.0) {
        const int repeat = static_cast<int>(std::round(dimQRescaleFactor));
        for (int i = 0; i < repeat; ++i) {
            newScales.append(scales.begin(), scales.end());
            if (hasZeroPoints) {
                newZeroPoints.append(zeroPoints.begin(), zeroPoints.end());
            }
        }
    } else {
        const int group = static_cast<int>(std::round(1.0 / dimQRescaleFactor));
        const size_t patternSize = scales.size() / group;
        if (patternSize * group != scales.size()) {
            logCb(formatv("Scale size ({0}) is not divisible by group size ({1})", scales.size(), group));
            return std::nullopt;
        }

        for (size_t i = 0; i < patternSize; ++i) {
            for (int j = 1; j < group; ++j) {
                const auto hasSameScale = scales[i] == scales[i + j * patternSize];
                const auto hasSameZeroPoint = !hasZeroPoints || zeroPoints[i] == zeroPoints[i + j * patternSize];
                if (!hasSameScale || !hasSameZeroPoint) {
                    logCb(formatv("Inconsistent repeating pattern in scales/zero points at position {0}", i));
                    return std::nullopt;
                }
            }
        }
        newScales.append(scales.begin(), scales.begin() + patternSize);
        if (hasZeroPoints) {
            newZeroPoints.append(zeroPoints.begin(), zeroPoints.begin() + patternSize);
        }

        if (patternSize != static_cast<size_t>(outShape[quantizedAxis])) {
            logCb(formatv("Pattern size ({0}) does not match output dimension size ({1})", patternSize,
                          outShape[quantizedAxis]));
            return std::nullopt;
        }
    }

    if (newScales.size() != static_cast<size_t>(outShape[quantizedAxis])) {
        logCb(formatv("New scales size ({0}) does not match output shape dimension ({1})", newScales.size(),
                      outShape[quantizedAxis]));
        return std::nullopt;
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisUniformQType.getFlags(), perAxisUniformQType.getStorageType(),
            perAxisUniformQType.getExpressedType(), newScales, newZeroPoints,
            perAxisUniformQType.getQuantizedDimension(), perAxisUniformQType.getStorageTypeMin(),
            perAxisUniformQType.getStorageTypeMax());
}

mlir::quant::UniformQuantizedPerAxisType vpux::inferPerAxisQuantizedTypeAfterShapeCast(const mlir::Type inType,
                                                                                       ArrayRef<int64_t> outShape) {
    const auto logCb = [](const formatv_object_base& msg) {
        VPUX_THROW("{0}", msg.str());
    };
    auto perAxisType = inferPerAxisQuantizedTypeAfterShapeCastOrNull(inType, outShape, logCb);
    VPUX_THROW_UNLESS(perAxisType.has_value(), "Failed to infer UniformQuantizedPerAxisType after ShapeCast");
    return perAxisType.value();
}
