//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Tensor Dimension representation.
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace vpux {

constexpr size_t MAX_NUM_DIMS = 15;

//
// DimBase
//

namespace details {

[[noreturn]] void throwNegativeDim(StringRef className, int64_t ind);
[[noreturn]] void throwDimOutOfRange(StringRef className, uint64_t ind);

template <class ConcreteDim, typename IndexType>
constexpr int32_t validateDimInd(IndexType ind) {
    if constexpr (std::is_signed<IndexType>::value) {
        if (ind < 0) {
            throwNegativeDim(ConcreteDim::getClassName(), static_cast<int64_t>(ind));
        }
        if (static_cast<uint64_t>(ind) >= MAX_NUM_DIMS) {
            throwDimOutOfRange(ConcreteDim::getClassName(), static_cast<uint64_t>(ind));
        }
    } else {
        if (ind >= MAX_NUM_DIMS) {
            throwDimOutOfRange(ConcreteDim::getClassName(), static_cast<uint64_t>(ind));
        }
    }

    return static_cast<int32_t>(ind);
}

template <class ConcreteDim>
class DimBase {
public:
    constexpr DimBase() = default;

    template <typename IndexType, typename = require_t<std::is_integral<IndexType>>>
    constexpr explicit DimBase(IndexType ind): _ind(validateDimInd<ConcreteDim>(ind)) {
    }

public:
    constexpr int32_t ind() const {
        return _ind;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        stream << "d" << ind();
    }

private:
    int32_t _ind = 0;
};

template <class ConcreteDim>
constexpr bool operator==(const DimBase<ConcreteDim>& d1, const DimBase<ConcreteDim>& d2) {
    return d1.ind() == d2.ind();
}
template <class ConcreteDim>
constexpr bool operator!=(const DimBase<ConcreteDim>& d1, const DimBase<ConcreteDim>& d2) {
    return d1.ind() != d2.ind();
}

template <class ConcreteDim>
constexpr bool operator<(const DimBase<ConcreteDim>& d1, const DimBase<ConcreteDim>& d2) {
    return d1.ind() < d2.ind();
}

}  // namespace details

//
// Dim
//

// Represents logical dimension index.

class Dim final : public details::DimBase<Dim> {
public:
    static StringRef getClassName();

public:
    using details::DimBase<Dim>::DimBase;
};

using DimArr = SmallVector<Dim>;
using DimArrRef = ArrayRef<Dim>;

//
// MemDim
//

// Represents memory dimension index (inner dimension has lower index).

class MemDim final : public details::DimBase<MemDim> {
public:
    static StringRef getClassName();

public:
    using details::DimBase<MemDim>::DimBase;
};

using MemDimArr = SmallVector<MemDim>;
using MemDimArrRef = ArrayRef<MemDim>;

}  // namespace vpux

//
// Hash
//

namespace std {

template <>
struct hash<vpux::Dim> final {
    size_t operator()(vpux::Dim dim) const {
        return static_cast<size_t>(dim.ind());
    }
};

template <>
struct hash<vpux::MemDim> final {
    size_t operator()(vpux::MemDim dim) const {
        return static_cast<size_t>(dim.ind());
    }
};

}  // namespace std
