//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/bit.h>

#include <cstdint>
#include <tuple>

namespace vpux::VPU {
enum class EltwiseType : uint64_t;
}  // namespace vpux::VPU

namespace vpux {
class NDTypeInterface;

struct QuantizationLevels final {
    static constexpr int64_t QUANT_LEVELS_2BIT = 4;
    static constexpr int64_t QUANT_LEVELS_4BIT = 16;
    static constexpr int64_t QUANT_LEVELS_8BIT = 256;
    static constexpr int64_t QUANT_LEVELS_16BIT = 65536;
};
static constexpr int64_t MAX_QUANT_LEVELS = QuantizationLevels::QUANT_LEVELS_16BIT;
static constexpr int64_t PARALLEL_EXECUTION_THRESHOLD = 4096;

//
// Utilities for quantized types
//

mlir::quant::UniformQuantizedPerAxisType getPerAxisTypeForBlock(
        mlir::quant::UniformQuantizedSubChannelType subchannelQType, const int64_t blockIndex);

bool isSupportedEltwiseQuantization(mlir::Type lhsElemType, mlir::Type rhsElemType, bool allowDifferentScales,
                                    bool allowDifferentZp, VPU::EltwiseType eltwiseType, LogCb logCb = emptyLogCb);

// Checks compatibility of per-axis quantized eltwise inputs (dequantize direction only).
// Both inputs must have identical per-axis quantization types and be quantized along the channel axis.
bool isSupportedEltwisePerAxisQuantization(mlir::Type lhsElemType, mlir::Type rhsElemType, LogCb logCb = emptyLogCb);

// Checks that a single per-axis quantized type is valid for eltwise use:
// must be quantized along the channel axis.
bool isSupportedEltwisePerAxisQuantization(mlir::Type perAxisElemType, LogCb logCb = emptyLogCb);

mlir::LogicalResult validateQuantElemType(mlir::Location loc, vpux::NDTypeInterface mainType);

mlir::Type normalizeQuantStorageType(mlir::quant::QuantizedType qType);

mlir::Type expandScalesAndZP(mlir::Type perAxisQType, ShapeRef padBefore, ShapeRef padAfter);

mlir::Type tileScalesAndZP(mlir::Type perAxisQType, ShapeRef shape, ShapeRef offsets, ShapeRef strides = {});

mlir::Type tileScalesAndZP(mlir::Type perAxisQType, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes);

mlir::Type changeAxis(mlir::Type perAxisQType, int32_t axis);

mlir::quant::QuantizedType changeExpressedType(mlir::quant::QuantizedType quantType, mlir::Type expressedType);

mlir::quant::QuantizedType changeStorageType(mlir::quant::QuantizedType qType, mlir::Type storageType);

bool canBeMerged(mlir::Type type1, mlir::Type type2);

mlir::Type concatScalesAndZP(ArrayRef<mlir::quant::UniformQuantizedPerAxisType> types);

using Scales = SmallVector<double>;
using ZeroPoints = SmallVector<int64_t>;

std::pair<Scales, ZeroPoints> extractScalesAndZeroPoints(mlir::Type tensorElemType);
Scales extractScalesOrDefault(mlir::Type elemType, double defaultScale);

/// @brief Returns a single zero-point for currently supported quantized types.
/// In case of uniform per-axis type, this means that all zero points are the
/// same and a single value could be returned successfully.
std::optional<int64_t> extractSingleZeroPoint(mlir::quant::QuantizedType type);

/// @brief Returns true if all zero-points are equal
bool areAllZeroPointsEqual(mlir::quant::UniformQuantizedPerAxisType type);

template <typename MultType>
std::tuple<MultType, uint8_t, int8_t> approximate(uint8_t bits, double target) {
    int exponent = 0;
    const auto mantissa = std::frexp(target, &exponent);

    // Doing floor because frexp gives values which when rounded are 32768
    // which is not representable on max(int16_t)
    const auto mult = checked_cast<MultType>(std::floor(mantissa * std::pow(2, bits)));
    const auto shift = exponent > bits ? 0 : checked_cast<uint8_t>(bits - exponent);
    const auto postShift = exponent > bits ? checked_cast<int8_t>(bits - exponent) : 0;

    return std::make_tuple(mult, shift, postShift);
}

class QuantizationApproximation {
public:
    QuantizationApproximation(double target);

    int64_t mult() const;
    int64_t shift() const;
    int64_t postShift() const;
    void setMult(int32_t mult);
    void setShift(uint8_t shift);

private:
    // PPE mult is I16 while IDU MULT for eltwise is U16
    // using int32_t as common storage
    int32_t _mult;
    uint8_t _shift;
    int8_t _postShift;
};

class PReLUApproximation {
public:
    PReLUApproximation(double alpha);

    int64_t mult() const;
    int64_t shift() const;

private:
    // PRELU mult is U11 - using int32_t as common storage
    int32_t _mult;
    uint8_t _shift;
};

mlir::FailureOr<int64_t> extractScalarOrUniformZP(mlir::quant::QuantizedType quantizedType);
bool hasScalarOrUniformZP(mlir::quant::QuantizedType quantizedType);

//
// FakeQuantize support
//

void getFakeQuantParams(mlir::quant::UniformQuantizedType qElemType, int64_t& levels, float& rMin, float& rMax);

void getFakeQuantParams(mlir::quant::UniformQuantizedPerAxisType qElemType, int64_t& levels,
                        SmallVectorImpl<float>& rMinVals, SmallVectorImpl<float>& rMaxVals);

mlir::FailureOr<std::tuple<double, int64_t>> calcScaleAndZeroPoint(double qMinFP, double qMaxFP, double rMin,
                                                                   double rMax, const Logger& log = Logger::global());

int64_t calculateZeroPoint(double low, double high, int levels, mlir::IntegerType type);

mlir::FailureOr<int64_t> getSingleZeroPointOrFail(mlir::quant::QuantizedType quantType);

int64_t getDefaultQuantizedZeroPoint(mlir::quant::QuantizedType quantType);

SmallVector<int64_t> getQuantizedTypeZeroPoints(mlir::quant::QuantizedType quantType);

bool isSymmetricZeroPoint(mlir::quant::QuantizedType quantType);

// Returns the integral storage type and the storage type's representable range for the given quantization levels.
// Returns failure if levels is unsupported
mlir::FailureOr<std::tuple<double, double, mlir::Type>> getStorageParams(mlir::MLIRContext* ctx, int64_t levels,
                                                                         bool isSigned);
// Returns the storage type and the storage type's representable range for a given low-precision type.
// Returns failure if type is unsupported
mlir::FailureOr<std::tuple<double, double, mlir::Type>> getStorageParams(mlir::Type type);
// Returns the representable range of the given type (may differ from the range of the corresponding storage type).
// Returns failure if type is unsupported
mlir::FailureOr<std::tuple<double, double>> getRepresentableRange(mlir::Type lowPrecisionType);

// Returns true if the given type is an 8-bit float type.
bool isFloat8(mlir::Type type);
// Returns true if the given type is a quantized type with 8-bit float storage.
bool isFloat8Quantized(mlir::Type type);

// Returns true if the given type is a signed 8-bit integer type.
bool isInt8(mlir::Type type);
// Returns true if the given type is a signed 8-bit integer quantized type.
bool isInt8Quantized(mlir::Type type);

// Returns true if the given type is an 4-bit float type.
bool isFloat4(mlir::Type type);
// Returns true if the given type is a quantized type with 4-bit float storage.
bool isFloat4Quantized(mlir::Type type);
// Returns true if the given type is a low float type.
bool isLowFpType(mlir::Type type);
// Returns true if the given type is a quantized type with low float type storage.
bool isLowFpTypeQuantized(mlir::Type type);
// Returns true if the given type is a quantized NF4 using Spec quantiles.
bool isNF4SpecQuantized(mlir::Type type);

/// Returns whether lowVals and highVals represet correct quantization range
/// specified by quantization levels and sign.
template <typename Range>
bool isLowPrecisionTypeRange(mlir::MLIRContext* ctx, Range lowVals, Range highVals, int64_t levels, bool isSigned) {
    VPUX_THROW_UNLESS(lowVals.size() == highVals.size(), "Sizes of valLow and valHigh arrays are not equal: {0} != {1}",
                      lowVals.size(), highVals.size());

    double qLow = 0.;
    double qHigh = 0.;

    const auto storageParams = getStorageParams(ctx, levels, isSigned);
    VPUX_THROW_WHEN(mlir::failed(storageParams), "Unsupported quantization levels '{0}'", levels);
    std::tie(qLow, qHigh, std::ignore) = *storageParams;
    const auto fLow = checked_cast<float>(qLow);
    const auto fHigh = checked_cast<float>(qHigh);
    // In order to decide if FakeQuantize input constant need to be requantized it is needed to check the FakeQuantize
    // input range.
    // Quantized weights have the content in the low precision data type - I/U 16,8,4...,1. For example, U8 quantized
    // weights are stored in U8 constants. Because NPU compiler relies on legacy weights de-quantization representation
    // which is: Const(FP16/32)->FakeQuantize(inLow = 0, inHigh = 255, ...), the WeightsDequantizeToFakeQuantize pass is
    // required to be applied. After this pass weights constant is FP16/32 data type and its content is floating point
    // values from 0.0 to 255.0 - just a cast of the U8 values - work in progress to keep the values in low precision
    // storage type: E#107322. There are passes that alter the weights content or create artificial FakeQuantize
    // (example: ConvertSubtractToAdd) and modify also FakeQuantize input range, which will no longer match the low
    // precision storage type as it initially was. It is needed to treat also the case when FakeQuantize inLow ==
    // inHigh, this is a special use case when re-quantization is not needed.
    const auto isEqualToLowPrecisionTypeRange = [&](float lowVal, float highVal) -> bool {
        return (isFloatEqual(lowVal, fLow) && isFloatEqual(highVal, fHigh)) || isFloatEqual(lowVal, highVal);
    };

    for (size_t i = 0; i < lowVals.size(); ++i) {
        if (!isEqualToLowPrecisionTypeRange(lowVals[i], highVals[i])) {
            return false;
        }
    }
    return true;
}

//
// Dequantize support
//

inline float dequantize(int64_t qVal, double scale, int64_t zeroPoint) {
    return static_cast<float>((qVal - zeroPoint) * scale);
}

inline double dequantizeDouble(double qVal, double scale, int64_t zeroPoint) {
    return (qVal - zeroPoint) * scale;
}

//
// FakeQuantize support
//

double fakeQuantize(double inVal, double inLow, double inHigh, double qLow, double qHigh, int64_t levels);

// Dequantize -> Operation -> Quantize fusing

inline bool areQuantizationRangesSimilar(float fqOutputHighVal, float parentFqOutputHighVal) {
    const auto quotient = fqOutputHighVal / parentFqOutputHighVal;
    // If values are similar, and within range, return true to indicate no need to align scales
    if (quotient >= 0.99 && quotient <= 1.01) {
        return true;
    }
    return false;
}

inline bool areQuantizationScalesSimilar(double quantizeScale, double dequantizeScale) {
    const auto quotient = quantizeScale / dequantizeScale;
    // If values are not similar, return false to indicate failure
    if (quotient < 0.99 || quotient > 1.01) {
        return false;
    }
    return true;
}

}  // namespace vpux
