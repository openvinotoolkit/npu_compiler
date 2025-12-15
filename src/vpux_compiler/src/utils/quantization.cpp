//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cmath>

using namespace vpux;

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
        const auto isSupportedFP8Type = [](mlir::Type type) {
            return mlir::isa<mlir::Float8E4M3FNType>(type) || mlir::isa<mlir::Float8E5M2Type>(type);
        };
        // Although we support different scale per input tensor, for integer quantized types
        // the HW scale uses 2 U16 register fields meaning we don't support negative scales
        // in that case.
        // For FP8 and BF8 is not the case because there we use FP16/BF16 register fields.
        // Also, if the scales are identical, then there's no need to use the U16 per input
        // tensor scale and we can use the I16/FP32 scale in the PPE which can support negative scales.
        // For now we support just cases of negative scales which be handled purely by adjusting
        // internal signed PPE scale.
        if (!isDoubleEqual(lhsQuantScale, rhsQuantScale)) {
            if (eltwiseType == VPU::EltwiseType::ADD || eltwiseType == VPU::EltwiseType::SUBTRACT) {
                if ((!isSupportedFP8Type(lhsQuantStorageType) && lhsQuantScale < 0) ^
                    (!isSupportedFP8Type(rhsQuantStorageType) && rhsQuantScale < 0)) {
                    logCb(formatv("Unsupported negative scales per eltwise input tensors"));
                    return false;
                }
            }
        }
    }

    return true;
}

mlir::LogicalResult vpux::validateQuantElemType(mlir::Location loc, vpux::NDTypeInterface mainType) {
    auto validateQuantizedPerAxisType = [](auto perAxisQType, mlir::Location loc,
                                           vpux::NDTypeInterface mainType) -> mlir::LogicalResult {
        const auto qDim = perAxisQType.getQuantizedDimension();

        if (qDim < 0 || static_cast<int64_t>(qDim) >= mainType.getRank()) {
            return errorAt(loc, "Quantized axis '{0}' is out of main type rank '{1}'", qDim, mainType.getRank());
        }

        const auto qDimSize = mainType.getShape()[Dim(static_cast<uint32_t>(qDim))];
        const auto numScales = perAxisQType.getScales().size();

        if (qDimSize != mlir::ShapedType::kDynamic) {
            if (checked_cast<size_t>(qDimSize) != numScales) {
                return errorAt(loc,
                               "Number of scales '{0}' in per-axis quantized type do not match the quantized "
                               "dimension "
                               "size '{1}'",
                               numScales, qDimSize);
            }
        }

        return mlir::success();
    };

    if (auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(mainType.getElementType())) {
        return validateQuantizedPerAxisType(perAxisQType, loc, mainType);
    }

    return mlir::success();
}

mlir::Type vpux::normalizeQuantStorageType(mlir::quant::QuantizedType qType) {
    auto elemType = qType.getStorageType();
    if (const auto intType = mlir::dyn_cast_or_null<mlir::IntegerType>(elemType)) {
        return mlir::IntegerType::get(intType.getContext(), intType.getWidth(),
                                      qType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
    }
    if (const auto lowFpType = mlir::dyn_cast_or_null<mlir::Float8E4M3FNType>(elemType)) {
        return mlir::Float8E4M3FNType::get(lowFpType.getContext());
    }
    if (const auto lowFpType = mlir::dyn_cast_or_null<mlir::Float8E5M2Type>(elemType)) {
        return mlir::Float8E5M2Type::get(lowFpType.getContext());
    }

    VPUX_THROW("Unsupported storage element type {0}", elemType);
}

static mlir::quant::UniformQuantizedPerAxisType getPerAxisTypeElem(
        const mlir::quant::UniformQuantizedPerAxisType perAxisQType, llvm::ArrayRef<double> newScales,
        llvm::ArrayRef<int64_t> newZeroPoints) {
    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(), newScales,
            newZeroPoints, perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

static mlir::quant::QuantileQuantizedPerAxisType getPerAxisTypeElem(
        const mlir::quant::QuantileQuantizedPerAxisType perAxisQType, llvm::ArrayRef<double> newScales,
        llvm::ArrayRef<int64_t> newZeroPoints) {
    return mlir::quant::QuantileQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getQuantileType(),
            perAxisQType.getExpressedType(), perAxisQType.getQuantiles(), newScales, newZeroPoints,
            perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
}

static mlir::quant::UniformQuantizedPerAxisType getPerAxisTypeElem(
        const mlir::quant::UniformQuantizedPerAxisType perAxisQType, const int32_t newAxis) {
    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), newAxis, perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

static mlir::quant::QuantileQuantizedPerAxisType getPerAxisTypeElem(
        const mlir::quant::QuantileQuantizedPerAxisType perAxisQType, const int32_t newAxis) {
    return mlir::quant::QuantileQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getQuantileType(),
            perAxisQType.getExpressedType(), perAxisQType.getQuantiles(), perAxisQType.getScales(),
            perAxisQType.getZeroPoints(), newAxis, perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
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
        if (const auto perAxisQuantileQType =
                    mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(perAxisUniformQType)) {
            return perAxisQuantileQType;
        }
        return perAxisUniformQType;
    }

    const auto scales = perAxisUniformQType.getScales();
    VPUX_THROW_UNLESS(!scales.empty(), "Can't get value for expand scales.");

    const auto zeroPoints = perAxisUniformQType.getZeroPoints();
    VPUX_THROW_UNLESS(!zeroPoints.empty(), "Can't get value for expand zero points.");
    VPUX_THROW_UNLESS(std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin()),
                      "All zero points should be equal");

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

    if (const auto perAxisQuantileQType =
                mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(perAxisUniformQType)) {
        return getPerAxisTypeElem(perAxisQuantileQType, newScales, newZeroPoints);
    }

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
        if (const auto perAxisQuantileQType =
                    mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(perAxisUniformQType)) {
            return perAxisQuantileQType;
        }
        return perAxisUniformQType;
    }

    auto getTiledElemType = [&](ArrayRef<double> tiledScale, ArrayRef<int64_t> tiledZp) -> mlir::Type {
        if (const auto perAxisQuantileQType =
                    mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(perAxisUniformQType)) {
            return getPerAxisTypeElem(perAxisQuantileQType, tiledScale, tiledZp);
        }

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
        if (const auto perAxisQuantileQType =
                    mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(perAxisUniformQType)) {
            return perAxisQuantileQType;
        }
        return perAxisUniformQType;
    }

    if (const auto perAxisQuantileQType =
                mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(perAxisUniformQType)) {
        return getPerAxisTypeElem(perAxisQuantileQType, newAxis);
    }

    return getPerAxisTypeElem(perAxisUniformQType, newAxis);
}

mlir::quant::QuantizedType vpux::changeStorageType(mlir::quant::QuantizedType qType, mlir::Type storageType) {
    VPUX_THROW_UNLESS(mlir::isa<mlir::IntegerType>(storageType), "Cannot change storage type to non-integer type");

    if (qType.getStorageType() == storageType) {
        return qType;
    }

    if (auto perTensor = mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(qType)) {
        return mlir::quant::QuantileQuantizedType::get(perTensor.getFlags(), storageType, perTensor.getQuantileType(),
                                                       perTensor.getExpressedType(), perTensor.getQuantiles(),
                                                       perTensor.getScale(), perTensor.getZeroPoint(),
                                                       perTensor.getStorageTypeMin(), perTensor.getStorageTypeMax());
    } else if (auto perAxis = mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(qType)) {
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxis.getFlags(), storageType, perAxis.getQuantileType(), perAxis.getExpressedType(),
                perAxis.getQuantiles(), perAxis.getScales(), perAxis.getZeroPoints(), perAxis.getQuantizedDimension(),
                perAxis.getStorageTypeMin(), perAxis.getStorageTypeMax());
    } else if (auto perTensor = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qType)) {
        return mlir::quant::UniformQuantizedType::get(perTensor.getFlags(), storageType, perTensor.getExpressedType(),
                                                      perTensor.getScale(), perTensor.getZeroPoint(),
                                                      perTensor.getStorageTypeMin(), perTensor.getStorageTypeMax());
    } else if (auto perAxis = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
        return mlir::quant::UniformQuantizedPerAxisType::get(perAxis.getFlags(), storageType,
                                                             perAxis.getExpressedType(), perAxis.getScales(),
                                                             perAxis.getZeroPoints(), perAxis.getQuantizedDimension(),
                                                             perAxis.getStorageTypeMin(), perAxis.getStorageTypeMax());
    }

    VPUX_THROW("Unsupported original type: {0}", qType);
}

mlir::quant::QuantizedType vpux::changeExpressedType(mlir::quant::QuantizedType quantType, mlir::Type expressedType) {
    return mlir::TypeSwitch<mlir::quant::QuantizedType, mlir::quant::QuantizedType>(quantType)
            .Case<mlir::quant::QuantileQuantizedType>([&](const auto qType) {
                return mlir::quant::QuantileQuantizedType::get(
                        qType.getFlags(), qType.getStorageType(), qType.getQuantileType(), expressedType,
                        qType.getQuantiles(), qType.getScale(), qType.getZeroPoint(), qType.getStorageTypeMin(),
                        qType.getStorageTypeMax());
            })
            .Case<mlir::quant::UniformQuantizedType>([&](const auto qType) {
                return mlir::quant::UniformQuantizedType::get(qType.getFlags(), qType.getStorageType(), expressedType,
                                                              qType.getScale(), qType.getZeroPoint(),
                                                              qType.getStorageTypeMin(), qType.getStorageTypeMax());
            })
            .Case<mlir::quant::QuantileQuantizedPerAxisType>([&](const auto qType) {
                return mlir::quant::QuantileQuantizedPerAxisType::get(
                        qType.getFlags(), qType.getStorageType(), qType.getQuantileType(), expressedType,
                        qType.getQuantiles(), qType.getScales(), qType.getZeroPoints(), qType.getQuantizedDimension(),
                        qType.getStorageTypeMin(), qType.getStorageTypeMax());
            })
            .Case<mlir::quant::UniformQuantizedPerAxisType>([&](const auto qType) {
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

    auto quantilePerAxisType1 = mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(type1);
    auto quantilePerAxisType2 = mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(type2);

    if (quantilePerAxisType1 == nullptr && quantilePerAxisType2 == nullptr) {
        return true;
    }

    if (quantilePerAxisType1 && quantilePerAxisType2) {
        return (quantilePerAxisType1.getQuantileType() == quantilePerAxisType2.getQuantileType()) &&
               (quantilePerAxisType1.getQuantiles() == quantilePerAxisType2.getQuantiles());
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

    if (auto quantilePerAxisType = mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(types.front())) {
        return getPerAxisTypeElem(quantilePerAxisType, newScales, newZeroPoints);
    } else if (auto uniformPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(types.front())) {
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

vpux::EltwiseQuantizationApproximation::EltwiseQuantizationApproximation(double input1Target, double input2Target,
                                                                         double outputTarget,
                                                                         VPU::EltwiseType eltwiseType)
        : _input1(input1Target), _input2(input2Target), _output(1 / outputTarget) {
    // We align shifts to the smaller one by dividing input MULT with 2^diff, inputs shift will be set to 0 in
    // nce_cluster_task.cpp and added to the output shift .
    //
    // what we actually do is input1 * MULT1 i32 --> + --> * MULT_OUT >> (SHIFT_OUT + SHIFT_IN) --> u8
    //                       input2 * MULT2 i32 ----^

    const auto minShift = std::min(_input1.shift(), _input2.shift());
    const auto maxShift = std::max(_input1.shift(), _input2.shift());
    // shift register is using 6 bits, so the maximum shift value is 2^6 - 1
    const int64_t maxRegisterShift = pow(2, 6) - 1;
    // the multiply register for each individual IDU unit is unsigned 16 bit;
    // so unlike the common PPE logic that uses signed 16 bit register and a maximum
    // multiply of pow(2, 15) - 1, here we can safely scale up to pow(2, 16) - 1
    const int64_t maxRegisterMult = pow(2, 16) - 1;

    const auto supportsShiftToMaximum = [&]() -> bool {
        if (maxShift + _output.shift() > maxRegisterShift) {
            return false;
        }
        if (_input1.mult() > maxRegisterMult >> (maxShift - _input1.shift())) {
            return false;
        }
        if (_input2.mult() > maxRegisterMult >> (maxShift - _input2.shift())) {
            return false;
        }

        return true;
    };

    // Currently handle just the case when both input scales are negative
    if (_input1.mult() < 0 || _input2.mult() < 0) {
        if (eltwiseType == VPU::EltwiseType::ADD || eltwiseType == VPU::EltwiseType::SUBTRACT) {
            // Can't handle cases when just one scale is negative
            if ((_input1.mult() < 0) ^ (_input2.mult() < 0)) {
                VPUX_THROW("Unsupported case for ADD/SUB eltwise, just one negative scale {0} {1}.", _input1.mult(),
                           _input2.mult());
            }
            _output.setMult(-1 * _output.mult());
        }
        if (eltwiseType == VPU::EltwiseType::MULTIPLY && ((_input1.mult() < 0) ^ (_input2.mult() < 0))) {
            _output.setMult(-1 * _output.mult());
        }
        _input1.setMult(static_cast<uint16_t>(std::abs(_input1.mult())));
        _input2.setMult(static_cast<uint16_t>(std::abs(_input2.mult())));
    }

    if (supportsShiftToMaximum()) {
        _input1.setMult(static_cast<uint16_t>(_input1.mult() << (maxShift - _input1.shift())));
        _input2.setMult(static_cast<uint16_t>(_input2.mult() << (maxShift - _input2.shift())));
        _output.setShift(_output.shift() + maxShift);
    } else if (minShift + _output.shift() < maxRegisterShift) {
        _input1.setMult(static_cast<uint16_t>(_input1.mult() >> (_input1.shift() - minShift)));
        _input2.setMult(static_cast<uint16_t>(_input2.mult() >> (_input2.shift() - minShift)));
        _output.setShift(_output.shift() + minShift);
    } else {
        VPUX_THROW("Elwise add input1_MULT/input2_MULT/output_SHIFT out of register range");
    }
}

QuantizationApproximation vpux::EltwiseQuantizationApproximation::input1() const {
    return _input1;
}

QuantizationApproximation vpux::EltwiseQuantizationApproximation::input2() const {
    return _input2;
}

QuantizationApproximation vpux::EltwiseQuantizationApproximation::output() const {
    return _output;
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

mlir::FailureOr<std::tuple<SmallVector<double>, SmallVector<int64_t>>> vpux::getScalesAndZeroPointsFromContentAttr(
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
        std::tie(qMin, qMax, std::ignore) = getStorageParams(ctx, *levels, isSigned);
    } else if (lowFpType.has_value()) {
        std::tie(qMin, qMax) = getRepresentableRange(*lowFpType);
    } else {
        VPUX_THROW("Got neither levels (for integer types) nor lowFpType (for float8 versions)");
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

    // TODO #E-164790: Add handling of QuantileQuantized types to check the quantileType field for the signedness
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

mlir::FailureOr<std::tuple<double, double>> vpux::getFp8Range(mlir::Type lowFpType) {
    if (mlir::isa<mlir::Float8E4M3FNType>(lowFpType)) {
        return std::make_tuple(static_cast<double>(mlir::quant::QuantizedType::getDefaultMinimumForF8E4M3FN()),
                               static_cast<double>(mlir::quant::QuantizedType::getDefaultMaximumForF8E4M3FN()));
    }
    if (mlir::isa<mlir::Float8E5M2Type>(lowFpType)) {
        return std::make_tuple(static_cast<double>(mlir::quant::QuantizedType::getDefaultMinimumForF8E5M2()),
                               static_cast<double>(mlir::quant::QuantizedType::getDefaultMaximumForF8E5M2()));
    }
    return mlir::failure();
}

std::tuple<double, double, mlir::Type> vpux::getStorageParams(mlir::MLIRContext* ctx, int64_t levels, bool isSigned) {
    switch (levels) {
    case 256:
        if (isSigned) {
            return {-128., 127., getSInt8Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt8Type(ctx)};
    case 255:
        if (isSigned) {
            return {-127., 127., getSInt8Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt8Type(ctx)};
    case 65536:
        if (isSigned) {
            return {-32768., 32767., getSInt16Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt16Type(ctx)};
    case 16:
        if (isSigned) {
            return {-8., 7., getSInt4Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt4Type(ctx)};
    case 15:
        if (isSigned) {
            return {-7., 7., getSInt4Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt4Type(ctx)};
    case 4:
        if (isSigned) {
            return {-2., 1., getSInt2Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt2Type(ctx)};
    // Because in the absence of I1 support, we must use U8 datatype.
    // [Track number: E#24341].
    case 2:
        if (isSigned) {
            return {0., 1., getSInt8Type(ctx)};
        }

        return {0., static_cast<double>(levels - 1), getUInt8Type(ctx)};

    default:
        VPUX_THROW("Got unsupported levels '{0}'", levels);
    }
}

std::tuple<double, double, mlir::Type> vpux::getStorageParams(mlir::Type lowPrecisionType) {
    // SI8 / UI8 / SI4 / UI4
    if (const auto intType = mlir::dyn_cast<mlir::IntegerType>(lowPrecisionType)) {
        VPUX_THROW_WHEN(intType.isSignless(), "Signless types do not have a range.");
        const auto isSigned = intType.isSigned();
        const auto bitWidth = intType.getWidth();
        const auto storageMin = mlir::quant::QuantizedType::getDefaultMinimumForInteger(isSigned, bitWidth);
        const auto storageMax = mlir::quant::QuantizedType::getDefaultMaximumForInteger(isSigned, bitWidth);

        return {storageMin, storageMax, lowPrecisionType};
    }

    // FP8
    if (const auto minMax = getFp8Range(lowPrecisionType); mlir::succeeded(minMax)) {
        return {std::get<0>(*minMax), std::get<1>(*minMax), lowPrecisionType};
    }

    // Quantile float types (NF4)
    if (const auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(lowPrecisionType)) {
        const auto bitWidth = quantileFloatType.getStorageTypeIntegralWidth();

        // Although the quantile float representable range is [first_quantile, last_quantile], its storage type is a
        // unsigned integer of the same bit-width. The storage range is set according to the storage type.
        const auto storageType =
                mlir::IntegerType::get(lowPrecisionType.getContext(), bitWidth, mlir::IntegerType::Unsigned);
        const auto storageMin = mlir::quant::QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/false, bitWidth);
        const auto storageMax = mlir::quant::QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/false, bitWidth);

        return {storageMin, storageMax, storageType};
    }

    VPUX_THROW("Got unsupported low precision type '{0}'", lowPrecisionType);
}

std::tuple<double, double> vpux::getRepresentableRange(mlir::Type lowPrecisionType) {
    // Quantile float types (NF4)
    if (const auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(lowPrecisionType)) {
        const auto quantileTable = quantileFloatType.getQuantiles();
        return {quantileTable.front(), quantileTable.back()};
    }

    // For the other types, the storage range is also the representable range.
    const auto [min, max, _] = getStorageParams(lowPrecisionType);
    return {min, max};
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

bool vpux::isNF4SpecQuantized(mlir::Type type) {
    const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type);
    if (qType == nullptr) {
        return false;
    }
    const auto bitWidth = qType.getStorageTypeIntegralWidth();
    if (bitWidth != 4) {
        return false;
    }

    if (const auto quantileUniformType = mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(qType)) {
        return quantileUniformType.getQuantiles() == vpux::type::NF4Type::getSpecQuantiles();
    }

    if (const auto quantilePerAxisType = mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(qType)) {
        return quantilePerAxisType.getQuantiles() == vpux::type::NF4Type::getSpecQuantiles();
    }

    return false;
}

mlir::FailureOr<int32_t> vpux::getQuantizedDimension(ShapeRef lowShape, ShapeRef highShape,
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

mlir::quant::QuantizedType vpux::getQuantizedType(const Const::ContentAttr& lowConst,
                                                  const Const::ContentAttr& highConst, std::optional<int64_t> levels,
                                                  std::optional<mlir::Type> lowFpType, mlir::FloatType expressedType,
                                                  bool isSigned, mlir::Location loc, IE::AutoBroadcastType broadcast,
                                                  bool ignoreZPCheck, const Logger& log) {
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
        const auto [storageMin, storageMax, storageType] = getStorageParams(ctx, *levels, isSigned);

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
        const auto [storageMin, storageMax, storageType] = getStorageParams(lowFpTypeVal);

        if (isFloat8(lowFpTypeVal)) {
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

        if (const auto quantileFloatType = mlir::dyn_cast<vpux::type::QuantileFloatType>(lowFpTypeVal)) {
            const auto quantileType = mlir::Float16Type::get(ctx);

            if (isPerAxisQuant) {
                return mlir::quant::QuantileQuantizedPerAxisType::getChecked(
                        loc, storageType.isSignedInteger(), storageType, quantileType, expressedType,
                        quantileFloatType.getQuantiles(), std::move(scales), std::move(zeroPoints), quantizedDim,
                        storageMin, storageMax);
            }
            return mlir::quant::QuantileQuantizedType::getChecked(
                    loc, storageType.isSignedInteger(), storageType, quantileType, expressedType,
                    quantileFloatType.getQuantiles(), scales[0], zeroPoints[0], storageMin, storageMax);
        }
    }

    VPUX_THROW("Got neither levels (for integer types) nor lowFpType (for float8 versions)");
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

void vpux::getFakeQuantParams(vpux::NDTypeInterface qType, int64_t& levels, mlir::RankedTensorType& attrType,
                              mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr) {
    const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(qType.getElementType());
    VPUX_THROW_WHEN(qElemType == nullptr, "Unsupported Quantized Type '{0}'", qType.getElementType());

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qElemType)) {
        float rMin, rMax;
        getFakeQuantParams(uniformType, levels, rMin, rMax);

        Shape attrShape(qType.getRank(), 1);
        attrType = mlir::RankedTensorType::get(attrShape.raw(), mlir::Float32Type::get(qType.getContext()));
        rMinAttr = Const::createConstContent(attrType, ArrayRef(rMin));
        rMaxAttr = Const::createConstContent(attrType, ArrayRef(rMax));
    } else if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qElemType)) {
        SmallVector<float> rMinVals, rMaxVals;
        getFakeQuantParams(perAxisQType, levels, rMinVals, rMaxVals);

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

double vpux::fakeQuantize(double inVal, double inLow, double inHigh, double qLow, double qHigh, int64_t levels) {
    if (inVal <= inLow) {
        return qLow;
    } else if (inVal > inHigh) {
        return qHigh;
    } else {
        return std::round((inVal - inLow) / (inHigh - inLow) * (levels - 1)) / (levels - 1) * (qHigh - qLow) + qLow;
    }
}

mlir::Type vpux::rescaleUniformQuantizedType(const mlir::Type tensorType, const double factor) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(tensorType);
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
