//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <openvino/core/type/float16.hpp>

namespace vpux {
namespace VPU {

//
// Saturation/bypass
//

// These entries are used to either output some constant (saturation) or the input itself (bypass)
// for some particular range. The same entry is used both for saturation/bypass but in case of bypass
// a special magic value is used instead of saturation value

constexpr auto NUM_SATURATION_BYPASS_ENTRIES = 8;
constexpr auto BYPASS_MAGIC = ov::float16::from_bits(0xFFFF);

struct SaturationBypassRange {
    ov::float16 lowerThreshold = std::numeric_limits<ov::float16>::quiet_NaN();
    ov::float16 upperThreshold = std::numeric_limits<ov::float16>::quiet_NaN();
    ov::float16 saturationValue = std::numeric_limits<ov::float16>::quiet_NaN();

    void serialize(std::vector<uint16_t>& buffer) const;
};

// Right now saturation/bypass values are set statically based on function plot analysis and adjusting
// the values to satisfy the required error

//
// TanH
//

// Can be approximated as f(x) = x when its close to zero
constexpr auto TANH_BYPASS_LOW = 0.f;
constexpr auto TANH_BYPASS_HIGH = 0.03125f;

// Converges to 1 if x -> +inf
constexpr auto TANH_SAT_LOW = 4.156f;
constexpr auto TANH_SAT_HIGH = std::numeric_limits<float>::infinity();
constexpr auto TANH_SAT_VALUE = 1.f;

constexpr auto TANH_ERROR = 0.0001f;

//
// Sigmoid
//

// Converges to 0 if x -> -inf
constexpr auto SIGMOID_NEG_SAT_LOW = -1 * std::numeric_limits<float>::infinity();
constexpr auto SIGMOID_NEG_SAT_HIGH = -16.f;
constexpr auto SIGMOID_NEG_SAT_VALUE = 0.f;

// Converges to 1 if x -> +inf
constexpr auto SIGMOID_POS_SAT_LOW = 16.f;
constexpr auto SIGMOID_POS_SAT_HIGH = std::numeric_limits<float>::infinity();
constexpr auto SIGMOID_POS_SAT_VALUE = 1.f;

constexpr auto SIGMOID_ERROR = 0.0001f;

//
// Swish
//

// Converges to 0 if x -> -inf
constexpr auto SWISH_SAT_LOW = -1 * std::numeric_limits<float>::infinity();
constexpr auto SWISH_SAT_HIGH = -16.f;
constexpr auto SWISH_SAT_VALUE = 0.f;

// Can be approximated as f(x) = x when x -> -inf
constexpr auto SWISH_BYPASS_LOW = 16.f;
constexpr auto SWISH_BYPASS_HIGH = std::numeric_limits<float>::infinity();

constexpr auto SWISH_ERROR = 0.00014f;

//
// GELU
//

// Converges to 0 if x -> -inf
constexpr auto GELU_SAT_LOW = -1 * std::numeric_limits<float>::infinity();
constexpr auto GELU_SAT_HIGH = -4.f;
constexpr auto GELU_SAT_VALUE = 0.f;

// Can be approximated as f(x) = x when x -> +inf
constexpr auto GELU_BYPASS_LOW = 4.f;
constexpr auto GELU_BYPASS_HIGH = std::numeric_limits<float>::infinity();

constexpr auto GELU_ERROR = 0.00013f;

//
// EXP
//

// Converges to 0 if x -> -inf
constexpr auto EXP_NEG_SAT_LOW = -1 * std::numeric_limits<float>::infinity();
constexpr auto EXP_NEG_SAT_HIGH = -7.22f;
constexpr auto EXP_NEG_SAT_VALUE = 0.f;

// Diverges to +inf if x -> +inf, we return the max representable value instead
constexpr auto EXP_POS_SAT_LOW = 11.08986f;
constexpr auto EXP_POS_SAT_HIGH = std::numeric_limits<float>::infinity();
constexpr auto EXP_POS_SAT_VALUE = std::numeric_limits<ov::float16>::max();

constexpr auto EXP_ERROR = 0.003f;

//
// HSwish
//

// Converges to 0 if x -> -inf
constexpr auto HSWISH_NEG_SAT_LOW = -1 * std::numeric_limits<float>::infinity();
constexpr auto HSWISH_NEG_SAT_HIGH = -3;
constexpr auto HSWISH_NEG_SAT_VALUE = 0.f;

// Completely linear past value 3
constexpr auto HSWISH_POS_BYPASS_LOW = 3;
constexpr auto HSWISH_POS_BYPASS_HIGH = std::numeric_limits<float>::infinity();

constexpr auto HSWISH_ERROR = 0.0002f;

//
// Lut config
//

// These entries are used to map each exponent into a table of lines.
// Sign & exponent of the number represented in FP16 format are used as an index into this table
// Thus we have 2^(1 + 5) = 64 possible entries (1 bit - sign, 5 bits - exponent)
// Each entry inside this table consists of two parts with the following layout:
// |<- n->|<- base address ->|
// |6 bits|<-   10 bits     >|
// Base address is the address into line table, it addresses lines, not bytes
// n - the number of MSBs of mantissa that are used in addition to address segments of the exponent
// It allows us to have multiple lines for the same exponent, i.e. 2^n lines

constexpr auto NUM_LUT_CFG_ENTRIES = 64;
constexpr auto BASE_ADDRESS_SIZE = 10;

struct LutConfig {
    uint16_t baseAddress = 0;
    uint16_t numOfMantissaMSBs = 0;

    void serialize(std::vector<uint16_t>& buffer) const;
};

//
// Slope/intercept
//

// These entries represent a line in the form f(x) = ax + b, where
//   a - slope
//   b - intercept
// Slope is calculated as (y1 - y0) / (x1 - x0) where:
//   x0 and x1 - the beginning and the end of the segment (see lut config above)
//   y0 and y1 - reference function values at x0 and x1 respectively
// Intercept is calculated as reference function value at x0 if range is positive, or at x1 if negative
// Because we are calculating slope as ref(x0) and not as ref(0), HW uses the following formula:
// f(x) = a(x - x0) + b

constexpr auto NUM_LUT_LINE_ENTRIES = 256;

struct LineDesc {
    float slope;
    float intercept;

    void serialize(std::vector<uint16_t>& buffer) const;
};

//
// Reserved region
//

constexpr auto NUM_RESERVED_SIZE = 7;

namespace ReservedRegion {

void serialize(std::vector<uint16_t>& buffer);

}  // namespace ReservedRegion

//
// SpecialConfig
//

// 16-bit value consisting of different special purpose bits
// xxxx xxxx xxxx xxxx
//                3210
// 0 - isSymmetric: function is symmetric around zero (no need to calculate negative part, e.g. tanh)
// 1 - reciprocalMode: special mode for reciprocal function
// 2 - reverseSquareRootMode: special mode for reverse square root function
// 3 - sinusMode: special mode for sinus function

struct SpecialConfig {
    bool isSymmetric = false;
    bool reciprocalMode = false;
    bool reverseSquareRootMode = false;
    bool sinusMode = false;

    void serialize(std::vector<uint16_t>& buffer) const;
};

//
// Bit manipulation utils and constants
//

constexpr uint16_t FP16_EXPONENT_SIZE = 5;
constexpr uint16_t FP16_EXPONENT_COUNT = 32;

constexpr uint16_t FP16_MANTISSA_SIZE = 10;
constexpr uint16_t FP16_MANTISSA_COUNT = 1024;

uint16_t extractMantissaMSBs(uint16_t mantissa, uint16_t numOfMantissaMSBs);
std::pair<float, float> getSegmentBeginEnd(uint16_t sign, uint16_t exponent, uint16_t numOfMantissaMSBs = 0,
                                           uint16_t mantissaMSBs = 0);
float getValue(uint16_t sign, uint16_t exponent, uint16_t mantissa);

struct Error {
    enum class Type : uint8_t { ABSOLUTE, RELATIVE };

    Error(float v, Type t): value(v), type(t) {
    }
    bool isRelative() const {
        return type == Type::RELATIVE;
    }

    float getValue() const {
        return value;
    }
    operator float() const {
        return value;
    }

    float value;
    Type type;
};
struct AbsoluteError : Error {
    explicit AbsoluteError(float v): Error(v, Type::ABSOLUTE) {
    }
};

struct RelativeError : Error {
    explicit RelativeError(float v): Error(v, Type::RELATIVE) {
    }
};

//
// SprLUTGenerator
//

// Generation algorithm:
//   1) Start with 1 linear segment per input exponent
//   2) Error within desired accuracy (absolute value)?
//     Yes: save slope and intercept values
//     No: double the number of linear segments
//   3) Repeat the iterative process, until the error requirement is met

class SprLUTGenerator {
public:
    SprLUTGenerator(std::function<float(float)> refFunction, AbsoluteError maxAbsoluteError,
                    Logger log = Logger::global().nest("sprlut-generator"));

    SprLUTGenerator(std::function<float(float)> refFunction, RelativeError maxRelativeError,
                    Logger log = Logger::global().nest("sprlut-generator"));

    SprLUTGenerator& setIsSymmetric();

    SprLUTGenerator& addBypassRange(float lowerValue, float upperValue);
    SprLUTGenerator& addSaturationRange(float lowerValue, float upperValue, float saturationValue);

    std::vector<uint16_t> generate();

private:
    void generateSprLUTContent();

    bool isApproximationRequired(uint16_t sign, uint16_t exponent) const;
    bool isSegmentCoveredBySaturationBypass(float segmentBegin, float segmentEnd) const;

    uint16_t calculateNumOfMantissaMSBs(uint16_t sign, uint16_t exponent) const;
    float getError(uint16_t sign, uint16_t exponent, uint16_t mantissa, int numOfMantissaMSBs) const;
    LineDesc generateLine(float x0, float x1) const;
    void addLinesToTable(uint16_t sign, uint16_t signAndExponent, uint16_t numOfMantissaMSBs);

    std::vector<uint16_t> serializeSprLUT();

private:
    std::function<float(float)> _refFunction;
    Error _error;

    llvm::SmallVector<SaturationBypassRange> _saturationBypassRanges;
    std::vector<LutConfig> _lutConfig;
    std::vector<LineDesc> _lines;
    SpecialConfig _specialConfig;

    Logger _log;
};

}  // namespace VPU
}  // namespace vpux
