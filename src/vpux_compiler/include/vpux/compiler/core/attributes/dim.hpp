//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Tensor Dimension representation.
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <functional>

#include <cassert>

namespace vpux {

constexpr size_t MAX_NUM_DIMS = 15;

//
// DimBase
//

namespace details {

void validateDimAttrs(StringRef className, int32_t ind);

template <class ConcreteDim>
class DimBase {
public:
    DimBase() = default;

    template <typename IndexType, typename = require_t<std::is_integral<IndexType>>>
    explicit DimBase(IndexType ind): _ind(checked_cast<int32_t>(ind)) {
        validateDimAttrs(ConcreteDim::getClassName(), _ind);
    }

public:
    int32_t ind() const {
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
bool operator==(const DimBase<ConcreteDim>& d1, const DimBase<ConcreteDim>& d2) {
    return d1.ind() == d2.ind();
}
template <class ConcreteDim>
bool operator!=(const DimBase<ConcreteDim>& d1, const DimBase<ConcreteDim>& d2) {
    return d1.ind() != d2.ind();
}

template <class ConcreteDim>
bool operator<(const DimBase<ConcreteDim>& d1, const DimBase<ConcreteDim>& d2) {
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
