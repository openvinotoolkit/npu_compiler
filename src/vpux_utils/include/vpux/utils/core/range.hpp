//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//
// Range concept from C++20.
//

#pragma once

#include "vpux/utils/core/containers.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator_range.h>

#include <numeric>
#include <vector>

namespace vpux {

//
// IteratorRange - range wrapper for STL iterators and containers.
//

using llvm::make_range;

template <class Range>
auto make_range(Range&& range) {
    return make_range(std::begin(std::forward<Range>(range)), std::end(std::forward<Range>(range)));
}

namespace details {

struct AsRangeTag final {};

template <class Range>
auto operator|(Range&& range, AsRangeTag) {
    return make_range(std::forward<Range>(range));
}

}  // namespace details

constexpr details::AsRangeTag as_range;

//
// Drop first N elements from the Range.
//

using llvm::drop_begin;

//
// Transformed range.
//

using llvm::map_range;

namespace details {

template <class FuncTy>
struct MapRangeTag final {
    FuncTy func;
};

template <class Range, class FuncTy>
auto operator|(Range&& range, MapRangeTag<FuncTy>&& tag) {
    return map_range(std::forward<Range>(range), std::move(tag.func));
}

}  // namespace details

template <class FuncTy>
auto transformed(FuncTy&& func) {
    return details::MapRangeTag<std::remove_reference_t<FuncTy>>{std::forward<FuncTy>(func)};
}

//
// Reversed range.
//

using llvm::reverse;

namespace details {

struct ReverseTag final {};

template <class Range>
auto operator|(Range&& range, ReverseTag) {
    return reverse(std::forward<Range>(range));
}

}  // namespace details

constexpr details::ReverseTag reversed;

//
// Filtered range.
//

using llvm::make_filter_range;

namespace details {

template <class PredicateT>
struct FilterRangeTag final {
    PredicateT pred;
};

template <class Range, class PredicateT>
auto operator|(Range&& range, FilterRangeTag<PredicateT>&& tag) {
    return make_filter_range(std::forward<Range>(range), std::move(tag.pred));
}

}  // namespace details

template <class PredicateT>
auto filtered(PredicateT&& pred) {
    return details::FilterRangeTag<std::remove_reference_t<PredicateT>>{std::forward<PredicateT>(pred)};
}

//
// Zipped range.
//

using llvm::zip;

//
// Concatenated range.
//

using llvm::concat;

//
// Ranges for map keys/values.
//

using llvm::make_first_range;
using llvm::make_second_range;

namespace details {

struct MapKeysTag final {};
struct MapValuesTag final {};

template <class Range>
auto operator|(Range&& range, MapKeysTag) {
    return make_first_range(std::forward<Range>(range));
}

template <class Range>
auto operator|(Range&& range, MapValuesTag) {
    return make_second_range(std::forward<Range>(range));
}

}  // namespace details

constexpr details::MapKeysTag map_keys;
constexpr details::MapValuesTag map_values;

//
// Indexed range.
//

using llvm::enumerate;

namespace details {

// Since 16 version of llvm enum was replaced with std::nullopt_t for NoneType
// Restore here deprecated enum, since there is no operator==
// for std::nullopt_t that required by indexed_accessor_range
enum class NoneType { None = 1 };
const NoneType None = NoneType::None;

struct EnumerateTag final {};

template <class Range>
auto operator|(Range&& range, EnumerateTag) {
    return enumerate(std::forward<Range>(range));
}

}  // namespace details

constexpr details::EnumerateTag indexed;

//
// Integer values range
//

namespace details {

template <class T, typename = require_t<std::is_integral<T>>>
class IntegerValuesRange final :
        public llvm::indexed_accessor_range<IntegerValuesRange<T>, details::NoneType, T, T*, T> {
public:
    using llvm::indexed_accessor_range<IntegerValuesRange<T>, details::NoneType, T, T*, T>::indexed_accessor_range;

public:
    static T dereference(const details::NoneType&, ptrdiff_t index) {
        return static_cast<T>(index);
    }
};

}  // namespace details

template <class T, typename = require_t<std::is_integral<T>>>
auto irange(T startIndex, T endIndex) {
    assert(endIndex >= startIndex);
    return details::IntegerValuesRange<T>(details::None, startIndex, endIndex - startIndex);
}

template <class T, typename = require_t<std::is_integral<T>>>
auto irange(T size) {
    return irange(static_cast<T>(0), size);
}

//
// Copy range to container.
//

using llvm::to_vector;

template <class Range>
auto to_small_vector(Range&& range) {
    using ValueType = std::decay_t<decltype(*std::begin(range))>;
    return SmallVector<ValueType>(std::begin(range), std::end(range));
}

template <class Range>
auto to_std_vector(Range&& range) {
    using ValueType = std::decay_t<decltype(*std::begin(range))>;
    return std::vector<ValueType>(std::begin(range), std::end(range));
}

template <class Container, class Range>
auto to_container(Range&& range) {
    Container out;
    for (auto&& val : range) {
        addToContainer(out, std::forward<decltype(val)>(val));
    }
    return out;
}

// Range accumulate
template <class Range, class T, class Op = std::plus<>>
T accumulate(Range&& range, T init, Op op = {}) {
    return std::accumulate(llvm::adl_begin(range), llvm::adl_end(range), std::move(init), op);
}

}  // namespace vpux

//
// llvm::format_provider specialization
//

namespace llvm {

template <typename T>
struct format_provider<vpux::details::IntegerValuesRange<T>> final : vpux::ListFormatProvider {};

}  // namespace llvm
