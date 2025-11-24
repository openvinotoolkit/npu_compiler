//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <openvino/core/dimension.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <vpux/utils/core/checked_cast.hpp>

#include <type_traits>
#include <vector>

// Helpers to detect if the type has ::value_type attribute
template <typename, typename = std::void_t<>>
struct HasValueType : std::false_type {};

template <typename T>
struct HasValueType<T, std::void_t<typename T::value_type>> : std::true_type {};

#define PRETTY_PARAM(name, type)                                                                                   \
    class name { /*NOLINT move argument should be enclosed in ()*/                                                 \
    public:                                                                                                        \
        using paramType = type;                                                                                    \
                                                                                                                   \
        name() = default;                                                                                          \
        name(const paramType& v): val_(v) {                                                                        \
        }                                                                                                          \
        name(paramType&& v): val_(std::move(v)) {                                                                  \
        }                                                                                                          \
                                                                                                                   \
        /* generic: enable when paramType can be built directly from U */                                          \
        template <typename U, std::enable_if_t<std::is_constructible_v<paramType, U&&>, int> = 0>                  \
        name(U&& u): val_(std::forward<U>(u)) {                                                                    \
        }                                                                                                          \
                                                                                                                   \
        /* container: initializer_list of exact value_type */                                                      \
        template <typename T_ = paramType, std::enable_if_t<HasValueType<T_>::value, int> = 0>                     \
        name(std::initializer_list<typename T_::value_type> init): val_(init) {                                    \
        }                                                                                                          \
                                                                                                                   \
        /* container: initializer_list of *convertible* types */                                                   \
        template <typename T_ = paramType, typename U,                                                             \
                  std::enable_if_t<HasValueType<T_>::value && std::is_constructible_v<typename T_::value_type, U>, \
                                   int> = 0>                                                                       \
        name(std::initializer_list<U> init): val_(init.begin(), init.end()) {                                      \
        }                                                                                                          \
                                                                                                                   \
        operator const paramType&() const {                                                                        \
            return val_;                                                                                           \
        }                                                                                                          \
                                                                                                                   \
    private:                                                                                                       \
        paramType val_{};                                                                                          \
    };                                                                                                             \
    static inline void PrintTo(const name& param, ::std::ostream* os) { /*NOLINT use anonymous namespace*/         \
        *os << #name ": " << ::testing::PrintToString((name::paramType)(param));                                   \
    }

//
// BoundedDim
//

struct BoundedDim {
    int dim;
    int bound;

    BoundedDim(int value = 1): dim(value), bound(value) {
        VPUX_THROW_UNLESS(value > 0, "Static dimension must have positive value, got: {0}", value);
    }

    BoundedDim(int dim, int bound): dim(dim), bound(bound) {
        VPUX_THROW_UNLESS(dim == -1, "Dynamic dimension must have value -1, got: {0}", dim);
        VPUX_THROW_UNLESS(bound > 0, "Upper bound must have positive value, got: {0}", bound);
    }
};

inline BoundedDim operator""_Dyn(unsigned long long value) {
    return BoundedDim{-1, vpux::checked_cast<int>(value)};
}

inline void PrintTo(const BoundedDim& boundedDim, std::ostream* os) {  // NOLINT function case style
    if (boundedDim.dim == -1) {
        *os << "1..";
    }

    *os << boundedDim.bound;
}

//
// Input test shapes creation
//

template <typename... Dims>
ov::Shape makeShape(Dims... dims) {
    using signed_dims_t = std::make_signed_t<ov::Shape::value_type>;
    return ov::Shape{vpux::checked_cast<ov::Shape::value_type, signed_dims_t>(dims)...};
}

// Creates test::InputShape(PartialShape, RuntimeShapes)
// For each BoundedDim dimension, that could be created with _Dyn literal, acts as an upper bound value.
// Create 3 static dimensions = (1, bound / 2, bound)
// Each BoundedDim multiplies the number of total RuntimeShapes by 3.
ov::test::InputShape generateTestShape(const std::vector<BoundedDim>& dims);

// Generate simple static input shape
ov::test::InputShape generateTestShape(const ov::Shape& shape);

// Examples of what the resulting InputShape would look like
// generateTestShape(5, 20)     -> ov::test::InputShape(PartialShape{5, 20}, std::vector<ov::Shape>{{5, 20}})
// generateTestShape(5, 20_Dyn) -> ov::test::InputShape(PartialShape{5, 1..20},
//                                                      std::vector<ov::Shape>{{5, 1}, {5, 10}, {5, 20}})
template <typename... Dims>
ov::test::InputShape generateTestShape(Dims... dims) {
    auto boundedDims = std::vector<BoundedDim>{BoundedDim(dims)...};
    return generateTestShape(boundedDims);
}

//
// Combine utility
//

template <typename T, typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
void appendTo(std::vector<T>& result, const U& value) {
    result.push_back(static_cast<T>(value));
}

template <typename T>
void appendTo(std::vector<T>& result, const std::vector<T>& vec) {
    result.insert(result.end(), vec.begin(), vec.end());
}

// Combines a list of single elements and std::vector of elements into one std::vector
// combine<int>(1, std::vector<int>{2, 3}, 4) => std::vector<int>{1, 2, 3, 4}
template <typename T, typename... Us>
std::vector<T> combine(const Us&... us) {
    auto result = std::vector<T>();
    (appendTo<T>(result, us), ...);
    return result;
}
