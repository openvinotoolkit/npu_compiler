//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>

#define ROUND_MODE_TO_NEAREST_EVEN

namespace vpux {
namespace type {
class float16 {
public:
    float16() = default;

    static uint32_t constexpr frac_size = 10;
    static uint32_t constexpr exp_size = 5;
    static uint32_t constexpr exp_bias = 15;

    float16(uint32_t sign, uint32_t biased_exponent, uint32_t fraction)
            : m_value((sign & 0x01) << 15 | (biased_exponent & 0x1F) << 10 | (fraction & 0x03FF)) {
    }

    float16(float value);

    template <typename I>
    explicit float16(I value): float16(static_cast<float>(value)) {
    }

    std::string to_string() const;
    size_t size() const;
    template <typename T>
    bool operator==(const T& other) const;
    bool operator==(const float16& other) const;
    template <typename T>
    bool operator!=(const T& other) const {
        return !(*this == other);
    }
    template <typename T>
    bool operator<(const T& other) const;
    template <typename T>
    bool operator<=(const T& other) const;
    template <typename T>
    bool operator>(const T& other) const;
    template <typename T>
    bool operator>=(const T& other) const;
    template <typename T>
    float16 operator+(const T& other) const;
    template <typename T>
    float16 operator+=(const T& other);
    template <typename T>
    float16 operator-(const T& other) const;
    template <typename T>
    float16 operator-=(const T& other);
    template <typename T>
    float16 operator*(const T& other) const;
    template <typename T>
    float16 operator*=(const T& other);
    template <typename T>
    float16 operator/(const T& other) const;
    template <typename T>
    float16 operator/=(const T& other);
    operator float() const;

    static constexpr float16 from_bits(uint16_t bits) {
        return float16(bits, true);
    }
    uint16_t to_bits() const;
    friend std::ostream& operator<<(std::ostream& out, const float16& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    constexpr float16(uint16_t x, bool): m_value{x} {
    }

    uint16_t m_value = 0;
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4756)
#endif
template <typename T>
bool float16::operator==(const T& other) const {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

template <typename T>
bool float16::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool float16::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool float16::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool float16::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
float16 float16::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
float16 float16::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
float16 float16::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
float16 float16::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
float16 float16::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
float16 float16::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
float16 float16::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
float16 float16::operator/=(const T& other) {
    return *this = *this / other;
}

bool iszero(float16 x);

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}  // namespace type
}  // namespace vpux

namespace std {
bool isnan(vpux::type::float16 x);
bool isinf(vpux::type::float16 x);

template <>
class numeric_limits<vpux::type::float16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr vpux::type::float16 min() noexcept {
        return vpux::type::float16::from_bits(0x0200);
    }
    static constexpr vpux::type::float16 max() noexcept {
        return vpux::type::float16::from_bits(0x7BFF);
    }
    static constexpr vpux::type::float16 lowest() noexcept {
        return vpux::type::float16::from_bits(0xFBFF);
    }
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr vpux::type::float16 epsilon() noexcept {
        return vpux::type::float16::from_bits(0x1200);
    }
    static constexpr vpux::type::float16 round_error() noexcept {
        return vpux::type::float16::from_bits(0x3C00);
    }
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr vpux::type::float16 infinity() noexcept {
        return vpux::type::float16::from_bits(0x7C00);
    }
    static constexpr vpux::type::float16 quiet_NaN() noexcept {
        return vpux::type::float16::from_bits(0x7FFF);
    }
    static constexpr vpux::type::float16 signaling_NaN() noexcept {
        return vpux::type::float16::from_bits(0x7DFF);
    }
    static constexpr vpux::type::float16 denorm_min() noexcept {
        return vpux::type::float16::from_bits(0);
    }
    static vpux::type::float16 clamp(vpux::type::float16 old_value, vpux::type::float16 low = lowest(),
                                     vpux::type::float16 high = max()) noexcept {
        if (old_value < low) {
            return low;
        }

        if (high < old_value) {
            return high;
        }

        return old_value;
    }

    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;

    // This is the minimum safe epsilon value to use for fused low-precision normalization
    // layers that utilize float32 internal computation but have float16 inputs and outputs.
    // In order to prevent divide-by-zero and NaNs in the output of such layers when the
    // input is all or mostly zeros, epsilon must be > (1/FLOAT16_MAX)^2. A safety factor
    // has been added due to varying implementations of Sqrt and Reciprocal functions.
    static constexpr float smallest_mixed_precision_eps = 0.000000001f;

    // This is the minimum safe epsilon value to use for unfused low-precision normalization
    // subgraphs. Here all intermediate steps in the normalization process are conducted in
    // float16 and care must be taken to ensure the epsilon value successfully prevents the
    // injection of Inf and/or NaNs in the full subgraph. A safety factor has been added.
    // (Empirical testing shows 0.8E-4 works while 0.6E-4 does not).
    static constexpr float nearest_non_zero_positive_value = 0.0001f;
};
}  // namespace std
