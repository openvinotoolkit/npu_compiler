//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_generator.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

using namespace vpux;
using namespace VPU;

//
// Serializers
//

void SaturationBypassRange::serialize(std::vector<uint16_t>& buffer) const {
    buffer.insert(buffer.end(), saturationValue.to_bits());
    if (lowerThreshold < 0) {
        // In case of negative range HW expects reversed low & upper
        buffer.insert(buffer.end(), {upperThreshold.to_bits(), lowerThreshold.to_bits()});
    } else {
        buffer.insert(buffer.end(), {lowerThreshold.to_bits(), upperThreshold.to_bits()});
    }
}

void ReservedRegion::serialize(std::vector<uint16_t>& buffer) {
    for (int index = 0; index < NUM_RESERVED_SIZE; ++index) {
        buffer.push_back(0x0000);
    }
}

void LutConfig::serialize(std::vector<uint16_t>& buffer) const {
    return buffer.push_back((numOfMantissaMSBs << BASE_ADDRESS_SIZE) | baseAddress);
}

void LineDesc::serialize(std::vector<uint16_t>& buffer) const {
    buffer.insert(buffer.end(), {ov::float16(slope).to_bits(), ov::float16(intercept).to_bits()});
}

void SpecialConfig::serialize(std::vector<uint16_t>& buffer) const {
    uint16_t serializedValue = isSymmetric;
    serializedValue |= reciprocalMode << 1;
    serializedValue |= reverseSquareRootMode << 2;
    serializedValue |= sinusMode << 3;
    buffer.push_back(serializedValue);
}

//
// Bit manipulation
//

uint16_t vpux::VPU::extractMantissaMSBs(uint16_t mantissa, uint16_t numOfMantissaMSBs) {
    const auto numOfMantissaLSBs = FP16_MANTISSA_SIZE - numOfMantissaMSBs;
    return mantissa >> numOfMantissaLSBs;
}

std::pair<float, float> vpux::VPU::getSegmentBeginEnd(uint16_t sign, uint16_t exponent, uint16_t numOfMantissaMSBs,
                                                      uint16_t mantissaMSBs) {
    const uint16_t numOfMantissaLSBs = FP16_MANTISSA_SIZE - numOfMantissaMSBs;
    const uint16_t mantissaLSBsMask = (1 << numOfMantissaLSBs) - 1;

    const uint16_t mantissaZeroes = mantissaMSBs << numOfMantissaLSBs;
    const uint16_t mantissaOnes = mantissaZeroes | mantissaLSBsMask;

    const auto segmentBegin = getValue(sign, exponent, mantissaZeroes);
    const auto segmentEnd = getValue(sign, exponent, mantissaOnes);

    // Negative number with big mantissa is smaller than with small mantissa, so min/max ensures that begin < end
    return std::make_pair(std::min(segmentBegin, segmentEnd), std::max(segmentBegin, segmentEnd));
}

float vpux::VPU::getValue(uint16_t sign, uint16_t exponent, uint16_t mantissa) {
    uint16_t value = mantissa;
    value |= exponent << FP16_MANTISSA_SIZE;
    value |= sign << (FP16_MANTISSA_SIZE + FP16_EXPONENT_SIZE);
    return static_cast<float>(ov::float16::from_bits(value));
}

//
// SprLUTGenerator
//

SprLUTGenerator::SprLUTGenerator(std::function<float(float)> refFunction, AbsoluteError error, Logger log)
        : _refFunction(std::move(refFunction)), _error(error), _log(std::move(log)) {
}

SprLUTGenerator::SprLUTGenerator(std::function<float(float)> refFunction, RelativeError error, Logger log)
        : _refFunction(std::move(refFunction)), _error(error), _log(std::move(log)) {
}

SprLUTGenerator& SprLUTGenerator::setIsSymmetric() {
    _specialConfig.isSymmetric = true;
    return *this;
}

SprLUTGenerator& SprLUTGenerator::addSaturationRange(float lowerValue, float upperValue, float saturationValue) {
    VPUX_THROW_WHEN(_saturationBypassRanges.size() == NUM_SATURATION_BYPASS_ENTRIES,
                    "Maximum number of saturation/bypass ranges ({0}) is exceeded", NUM_SATURATION_BYPASS_ENTRIES);
    _saturationBypassRanges.push_back({lowerValue, upperValue, saturationValue});
    return *this;
}

SprLUTGenerator& SprLUTGenerator::addBypassRange(float lowerValue, float upperValue) {
    VPUX_THROW_WHEN(_saturationBypassRanges.size() == NUM_SATURATION_BYPASS_ENTRIES,
                    "Maximum number of saturation/bypass ranges ({0}) is exceeded", NUM_SATURATION_BYPASS_ENTRIES);
    _saturationBypassRanges.push_back({lowerValue, upperValue, BYPASS_MAGIC});
    return *this;
}

std::vector<uint16_t> SprLUTGenerator::generate() {
    generateSprLUTContent();
    return serializeSprLUT();
}

void SprLUTGenerator::generateSprLUTContent() {
    for (uint16_t sign = 0; sign <= 1; ++sign) {
        for (uint16_t exponent = 0; exponent < FP16_EXPONENT_COUNT; ++exponent) {
            _log.debug("Generating sprLUT for values with sign: {0}, exponent: {1}", sign, exponent);
            LutConfig lutCfg{};
            if (isApproximationRequired(sign, exponent)) {
                lutCfg.numOfMantissaMSBs = calculateNumOfMantissaMSBs(sign, exponent);
                lutCfg.baseAddress = _lines.size();
                addLinesToTable(sign, exponent, lutCfg.numOfMantissaMSBs);
            }
            _lutConfig.push_back(lutCfg);
        }
    }
    VPUX_THROW_UNLESS(_lutConfig.size() == NUM_LUT_CFG_ENTRIES,
                      "Total number of lut config entries should be {0}, but got {1}", _lutConfig.size(),
                      NUM_LUT_CFG_ENTRIES);
    _log.debug("Overall number of lines generated: {0}", _lines.size());
}

bool SprLUTGenerator::isApproximationRequired(uint16_t sign, uint16_t exponent) const {
    // No need to calculate negative values if the function is symmetric around zero
    if (_specialConfig.isSymmetric && sign) {
        _log.nest().debug("Skipping generation: function is symmetric and sign is negative");
        return false;
    }

    // Skip if it's the last exponent as it's used to represent only infinity/NaNs
    if (exponent == FP16_EXPONENT_COUNT - 1) {
        _log.nest().debug("Skipping generation: last exponent represents only infinity/NaNs");
        return false;
    }

    auto [segmentBegin, segmentEnd] = getSegmentBeginEnd(sign, exponent);
    return !isSegmentCoveredBySaturationBypass(segmentBegin, segmentEnd);
}

bool SprLUTGenerator::isSegmentCoveredBySaturationBypass(float segmentBegin, float segmentEnd) const {
    for (const auto& saturationBypassRange : _saturationBypassRanges) {
        if (segmentBegin >= saturationBypassRange.lowerThreshold &&
            segmentEnd <= saturationBypassRange.upperThreshold) {
            _log.nest().debug("Segment [{0:f6}, {1:f6}] is covered by saturation/bypass [{2}, {3}]", segmentBegin,
                              segmentEnd, saturationBypassRange.lowerThreshold, saturationBypassRange.upperThreshold);
            return true;
        }
    }
    return false;
}

uint16_t SprLUTGenerator::calculateNumOfMantissaMSBs(uint16_t sign, uint16_t exponent) const {
    uint16_t numOfMantissaMSBs = 0;
    for (uint16_t mantissa = 0; mantissa < FP16_MANTISSA_COUNT; ++mantissa) {
        float curError = 0;
        while ((curError = getError(sign, exponent, mantissa, numOfMantissaMSBs)) > _error) {
            _log.nest().debug("Got error {0:f6} that is higher than maximum absolute error {1:f6}, retrying with "
                              "higher number of "
                              "segments",
                              curError, _error);
            const auto value = getValue(sign, exponent, mantissa);
            const auto [segmentBegin, segmentEnd] = getSegmentBeginEnd(
                    sign, exponent, numOfMantissaMSBs, extractMantissaMSBs(mantissa, numOfMantissaMSBs));
            _log.nest(2).debug("Value: {0:f6}, segment: [{1:f6}, {2:f6}]", value, segmentBegin, segmentEnd);
            numOfMantissaMSBs++;
            VPUX_THROW_WHEN(numOfMantissaMSBs > FP16_MANTISSA_SIZE,
                            "Number of mantissa MSBs exceeded maximum size of FP16 mantissa");
        }
    }
    return numOfMantissaMSBs;
}

void SprLUTGenerator::addLinesToTable(uint16_t sign, uint16_t exponent, uint16_t numOfMantissaMSBs) {
    const auto numSegments = static_cast<uint16_t>(pow(2, numOfMantissaMSBs));
    _log.nest().debug("Generating lines for sign: {0}, exponent: {1}, number of segments: {2}", sign, exponent,
                      numSegments);

    for (uint16_t segment = 0; segment < numSegments; ++segment) {
        auto [segmentBegin, segmentEnd] = getSegmentBeginEnd(sign, exponent, numOfMantissaMSBs, segment);
        const auto line = generateLine(segmentBegin, segmentEnd);
        _log.nest(1).debug("Generated line for segment {0}, slope={1:f6}, intercept={2:f6}", segment, line.slope,
                           line.intercept);
        _lines.push_back(line);
    }
    VPUX_THROW_WHEN(_lines.size() > NUM_LUT_LINE_ENTRIES, "Actual number of lines ({0}) exceeds the maximum ({1})",
                    _lines.size(), NUM_LUT_LINE_ENTRIES);
}

LineDesc SprLUTGenerator::generateLine(float x0, float x1) const {
    const auto y0 = _refFunction(x0);
    const auto y1 = _refFunction(x1);

    const auto slope = (y1 - y0) / (x1 - x0);
    return {slope, x1 >= 0 ? y0 : y1};
}

float SprLUTGenerator::getError(uint16_t sign, uint16_t exponent, uint16_t mantissa, int numOfMantissaMSBs) const {
    const auto mantissaMSBs = extractMantissaMSBs(mantissa, numOfMantissaMSBs);
    const auto [segmentBegin, segmentEnd] = getSegmentBeginEnd(sign, exponent, numOfMantissaMSBs, mantissaMSBs);
    const auto value = getValue(sign, exponent, mantissa);

    const auto line = generateLine(segmentBegin, segmentEnd);
    const auto approx = line.slope * (value - (value >= 0 ? segmentBegin : segmentEnd)) + line.intercept;
    const auto ref = _refFunction(value);
    const auto error = std::abs(approx - ref);
    if (_error.isRelative()) {
        constexpr float epsilon = 1e-7f;  // in case ref == 0.0f, we divide by a small number to avoid NaN
        return error / std::max(std::abs(ref), epsilon);
    }
    return error;
}

std::vector<uint16_t> SprLUTGenerator::serializeSprLUT() {
    std::vector<uint16_t> sprLUTBuffer;

    for (const auto& saturationBypassRange : _saturationBypassRanges) {
        saturationBypassRange.serialize(sprLUTBuffer);
    }
    // Fill unused Saturation/bypass entries
    for (auto i = _saturationBypassRanges.size(); i < NUM_SATURATION_BYPASS_ENTRIES; ++i) {
        SaturationBypassRange().serialize(sprLUTBuffer);
    }

    _specialConfig.serialize(sprLUTBuffer);

    ReservedRegion::serialize(sprLUTBuffer);

    for (const auto& lutConfig : _lutConfig) {
        lutConfig.serialize(sprLUTBuffer);
    }

    for (const auto& line : _lines) {
        line.serialize(sprLUTBuffer);
    }

    return sprLUTBuffer;
}
