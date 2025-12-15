//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/attributes_properties_conversion.hpp"
#include "vpux/utils/core/type/bfloat16.hpp"
#include "vpux/utils/core/type/float16.hpp"

namespace vpux::VPU {

double computeQuantScale(mlir::Type inputType, mlir::Type outputType);
double computeQuantScaleWithWeightedOps(mlir::Type inputType, mlir::Type outputType, mlir::Type weightsType);
int64_t computeQuantZPForEltwise(mlir::Type type);

double computeAvgPoolQuantScale(mlir::Type inputType, mlir::Type outputType, mlir::ArrayRef<int64_t> filterShape);

double computeScale(mlir::Operation* operation);

template <typename F16Type>
static int32_t packClamp(double value) {
    // The IntPPE HW pipeline branches into 2 cases depending on what is the output type of the operations it handles:
    // Quantized / Non-quantized (F16)

    // Non-quantized takes advantage of half bit-size by merging both low and high clamping values into a single int32
    // value: 0xllll_hhhh. By convention this value is stored in the clamp_high attribute, while clamp_low is set to the
    // most negative int32 value, and ignored.

    // Although the attributes have integer types, the actual bits represent fixed point real values.

    static_assert(std::is_same_v<F16Type, vpux::type::float16> || std::is_same_v<F16Type, vpux::type::bfloat16>,
                  "Invalid packing type, expected a 16-bit float type");
    // Convert to f16 using the type ctor:
    const F16Type valueF16(value);
    // Reinterpret back to integer:
    const auto valueI16 = reinterpret_cast<const int16_t*>(&valueF16);
    // Cast into the attr type, int32:
    return static_cast<int32_t>(*valueI16);
}

template <typename F16Type>
static int32_t packClamp(F16Type low, F16Type high) {
    // Arranges the clamp [low, high] interval into the 0xllll_hhhh bit format by concatening two int32 values
    // having only the least significant 16 bits used.
    const auto lowI16 = reinterpret_cast<const int16_t*>(&low);
    const auto lowI32 = static_cast<int32_t>(*lowI16);

    const auto highI16 = reinterpret_cast<const int16_t*>(&high);
    const auto highI32 = static_cast<int32_t>(*highI16);

    return (lowI32 << 16) + highI32;
}

template <typename F16Type>
static int32_t packClamp(double low, double high) {
    return (packClamp<F16Type>(low) << 16) + packClamp<F16Type>(high);
}

template <typename F16Type>
std::pair<F16Type, F16Type> unpackClamp(int32_t clampHigh) {
    static_assert(std::is_same_v<F16Type, vpux::type::float16> || std::is_same_v<F16Type, vpux::type::bfloat16>,
                  "Invalid packing type, expected a 16-bit float type");

    // Unpack 0xllll_hhhh non-quantized clamp_high format into individual low and high values
    const int16_t highI16 = clampHigh & 0xFFFF;         // high = last 16 bits
    const int16_t lowI16 = (clampHigh >> 16) & 0xFFFF;  // low  = first 16 bits

    // The bits are expected to be in the f16 format
    const auto lowF16 = *reinterpret_cast<const F16Type*>(&lowI16);
    const auto highF16 = *reinterpret_cast<const F16Type*>(&highI16);
    return std::make_pair(lowF16, highF16);
}

}  // namespace vpux::VPU
