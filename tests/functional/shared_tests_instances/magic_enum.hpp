//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include "vpux/utils/core/helper_macros.hpp"

namespace PrettyEnum {

// Use template trick E V to get the Value type (enum field name)
// with the __PRETTY_FUNCTION__ macro we can get current function signature
// it will contain the field name somewere in the string.
// We can capture the signature with a string_view and look for the field name
template <typename E, E V>
constexpr std::string_view parseEnumFieldNameFromSignature() {
#if defined(__clang__)
    // ... [E = MyEnum, V = MyEnum::FieldName]
    constexpr std::string_view sig = __PRETTY_FUNCTION__;
    const auto right = sig.rfind(']');
    const auto left = sig.rfind(':');
    return sig.substr(left + 1, right - left - 1);
#elif defined(__GNUC__)
    // ... [with E = MyEnum; E V = MyEnum::FieldName; std::string_view = std::basic_string_view<char>]
    constexpr std::string_view sig = __PRETTY_FUNCTION__;
    const auto right = sig.rfind(';');
    const auto left = sig.rfind(':', right);
    return sig.substr(left + 1, right - left - 1);
#elif defined(_MSC_VER)
    // ...<enum MyEnum,MyEnum::FieldName>(void)"
    constexpr std::string_view sig = __FUNCSIG__;
    const auto right = sig.rfind('>');
    const auto left = sig.rfind(':');
    return sig.substr(left + 1, right - left - 1);
#else
    return {};
#endif
}

template <typename Enum, int I>
bool matchSet(Enum e, std::string_view& out) noexcept {
    using U = std::underlying_type_t<Enum>;

    // If the enum value matches with the provided integer,
    // we can look for the enum field name equal to that integer
    if (static_cast<U>(static_cast<Enum>(I)) == static_cast<U>(e)) {
        out = parseEnumFieldNameFromSignature<Enum, static_cast<Enum>(I)>();
        return true;
    }

    return false;
}

// Unpack integer sequence to try out all the values in the sequence
// until we find an integer equal to the enum value
template <typename Enum, int... I>
std::string_view pickName(Enum e, std::integer_sequence<int, I...>) {
    std::string_view out{};
    VPUX_UNUSED((matchSet<Enum, I>(e, out)) || ...);
    return out;
}

// Create an integer sequence from 0 to maxEnumValue
// to try and match underlying enum value (by default starts from 0)
template <typename Enum>
std::string_view enumName(Enum e) {
    constexpr int maxEnumValue = 64;
    return pickName<Enum>(e, std::make_integer_sequence<int, maxEnumValue>{});
}

// Print field name of the enum value
template <typename Enum>
void printFieldName(Enum e, std::ostream& os) {
    if (auto sv = enumName(e); !sv.empty()) {
        os << sv;
    } else {
        os << static_cast<std::underlying_type_t<Enum>>(e);
    }
}

}  // namespace PrettyEnum
