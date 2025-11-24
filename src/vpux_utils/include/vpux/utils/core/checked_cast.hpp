//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

//
// Safe version of `static_cast` with run-time checks.
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <llvm/Support/TypeName.h>

#include <cmath>
#include <limits>

namespace vpux {

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<OutT, InT>> checked_cast(InT value) {
    return value;
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_integral<OutT>, std::is_signed<OutT>,
         not_<std::is_same<OutT, InT>>>
checked_cast(InT value) {
    if constexpr (std::numeric_limits<InT>::lowest() < std::numeric_limits<OutT>::lowest()) {
        VPUX_THROW_UNLESS(value >= std::numeric_limits<OutT>::lowest(), "Can not safely cast {0} from {1} to {2}",
                          static_cast<int64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    }

    if constexpr (std::numeric_limits<InT>::max() > std::numeric_limits<OutT>::max()) {
        VPUX_THROW_UNLESS(value <= std::numeric_limits<OutT>::max(), "Can not safely cast {0} from {1} to {2}",
                          static_cast<int64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    }

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_integral<OutT>, std::is_unsigned<OutT>,
         not_<std::is_same<OutT, InT>>>
checked_cast(InT value) {
    if constexpr (std::numeric_limits<InT>::max() > std::numeric_limits<OutT>::max()) {
        VPUX_THROW_UNLESS(value <= std::numeric_limits<OutT>::max(), "Can not safely cast {0} from {1} to {2}",
                          static_cast<uint64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    }

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_integral<OutT>, std::is_signed<OutT>> checked_cast(
        InT value) {
    if constexpr (std::numeric_limits<InT>::max() >
                  static_cast<std::make_unsigned_t<OutT>>(std::numeric_limits<OutT>::max())) {
        VPUX_THROW_UNLESS(value <= static_cast<std::make_unsigned_t<OutT>>(std::numeric_limits<OutT>::max()),
                          "Can not safely cast {0} from {1} to {2}", static_cast<uint64_t>(value),
                          llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    }

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_integral<OutT>, std::is_unsigned<OutT>> checked_cast(
        InT value) {
    VPUX_THROW_UNLESS(value >= 0, "Can not safely cast {0} from {1} to {2}", static_cast<int64_t>(value),
                      llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    if constexpr (static_cast<std::make_unsigned_t<InT>>(std::numeric_limits<InT>::max()) >
                  std::numeric_limits<OutT>::max()) {
        VPUX_THROW_UNLESS(static_cast<std::make_unsigned_t<InT>>(value) <= std::numeric_limits<OutT>::max(),
                          "Can not safely cast {0} from {1} to {2}", static_cast<int64_t>(value),
                          llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    }

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_floating_point<InT>, std::is_integral<OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(value <= static_cast<InT>(std::numeric_limits<OutT>::max()),
                      "Can not safely cast {0} from {1} to {2}", value, llvm::getTypeName<InT>(),
                      llvm::getTypeName<OutT>());

    VPUX_THROW_UNLESS(value >= static_cast<InT>(std::numeric_limits<OutT>::lowest()),
                      "Can not safely cast {0} from {1} to {2}", value, llvm::getTypeName<InT>(),
                      llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_floating_point<OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast {0} from {1} to {2}",
                      static_cast<int64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_floating_point<OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast {0} from {1} to {2}",
                      static_cast<uint64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<double, InT>, std::is_same<float, OutT>> checked_cast(InT value) {
    VPUX_THROW_WHEN(
            value > static_cast<InT>(std::numeric_limits<OutT>::max()) && !std::isnan(value) && !std::isinf(value),
            "Can not safely cast {0} from {1} to {2}", value, llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    VPUX_THROW_WHEN(
            value < static_cast<InT>(std::numeric_limits<OutT>::lowest()) && !std::isnan(value) && !std::isinf(value),
            "Can not safely cast {0} from {1} to {2}", value, llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<float, InT>, std::is_same<double, OutT>> checked_cast(InT value) {
    return static_cast<OutT>(value);
}

}  // namespace vpux
