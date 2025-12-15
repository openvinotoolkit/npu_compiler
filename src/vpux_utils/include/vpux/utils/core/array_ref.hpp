//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//
// Non-owning wrapper for array range.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/hash.hpp"

#include <llvm/ADT/ArrayRef.h>

namespace vpux {

using llvm::ArrayRef;
using llvm::MutableArrayRef;

template <typename Type>
bool checkAllElementsIfEqualTo(llvm::ArrayRef<Type> array, const Type& value) {
    return llvm::all_of(array, [&](const auto& elem) {
        return elem == value;
    });
}

}  // namespace vpux

//
// std::hash specialization
//

namespace std {

template <typename T>
struct hash<vpux::ArrayRef<T>> final {
    size_t operator()(vpux::ArrayRef<T> arr) const {
        return vpux::getRangeHash(arr);
    }
};

template <typename T>
struct hash<vpux::MutableArrayRef<T>> final {
    size_t operator()(vpux::MutableArrayRef<T> arr) const {
        return vpux::getRangeHash(arr);
    }
};

}  // namespace std

//
// llvm::format_provider specialization
//

namespace llvm {

template <typename T>
struct format_provider<ArrayRef<T>> final : vpux::ListFormatProvider {};

template <typename T>
struct format_provider<MutableArrayRef<T>> final : vpux::ListFormatProvider {};

}  // namespace llvm
