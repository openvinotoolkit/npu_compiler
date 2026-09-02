//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"

#include <numeric>

//
// Representations Impl
//

namespace vpux {
namespace details {

template <class Base>
class BoundsTag : public Base {
public:
    using Base::Base;
};

template <class Base>
class DimsMaskTag : public Base {
public:
    using Base::Base;
};

}  // namespace details

//
// Representations
//

// Dynamic tensor can have two representations:
//  - Upper bounded representation, dynamic shape with static bounds attribute:
using Bounds = details::DimValues<Dim, int64_t, details::BoundsTag>;
using BoundsRef = details::DimValuesRef<Dim, int64_t, details::BoundsTag>;
//  - Dynamic dims mask representation, static shape equal to the upper bounds with an attribute
//    where each value represents static dimension with (0) or dynamic dimension with (1)
using DynamicDimsMask = details::DimValues<Dim, int64_t, details::DimsMaskTag>;
using DynamicDimsMaskRef = details::DimValuesRef<Dim, int64_t, details::DimsMaskTag>;

BoundsRef getBounds(mlir::Type type);
DynamicDimsMaskRef getDynamicDimsMask(mlir::Type type);

//
// Element Size-Representation Pairs
//

class MaskedDim;

// Represents a possibly dynamic dimension size together with its bound. The bound is always set as the maximum size the
// dimension can have, for static sizes the bound is equal to the size.
class BoundedDim {
public:
    using ReprType = Bounds;

private:
    int64_t _dimValue;
    int64_t _bound;

public:
    BoundedDim(int64_t dimValue, int64_t bound);
    BoundedDim(int64_t dimValue);
    BoundedDim(const MaskedDim& maskedDim);

public:
    int64_t dimValue() const;
    int64_t representation() const;
    bool isDynamic() const;
    int64_t reifiedSize() const;

private:
    template <typename OperatorT>
    static BoundedDim apply(const BoundedDim& lhs, const BoundedDim& rhs, OperatorT op) {
        // static_dim  x static_dim -> static_dim
        // dynamic_dim x any_dim    -> dynamic_dim
        const auto bound = op(lhs._bound, rhs._bound);
        return (lhs.isDynamic() || rhs.isDynamic()) ? BoundedDim(mlir::ShapedType::kDynamic, bound)
                                                    : BoundedDim(op(lhs._dimValue, rhs._dimValue), bound);
    }

public:
    BoundedDim operator+(const BoundedDim& other) const;
    BoundedDim operator-(const BoundedDim& other) const;
    BoundedDim operator*(const BoundedDim& other) const;
    BoundedDim operator/(const BoundedDim& other) const;

    BoundedDim& operator+=(const BoundedDim& other);
    BoundedDim& operator-=(const BoundedDim& other);
    BoundedDim& operator*=(const BoundedDim& other);
    BoundedDim& operator/=(const BoundedDim& other);

    bool operator==(const BoundedDim& other) const;
    bool operator!=(const BoundedDim& other) const;
    bool operator<(const BoundedDim& other) const;
    bool operator>(const BoundedDim& other) const;
    bool operator<=(const BoundedDim& other) const;
    bool operator>=(const BoundedDim& other) const;

    friend BoundedDim operator+(int64_t x, const BoundedDim& y);
    friend BoundedDim operator-(int64_t x, const BoundedDim& y);
    friend BoundedDim operator*(int64_t x, const BoundedDim& y);
    friend BoundedDim operator/(int64_t x, const BoundedDim& y);

    friend bool operator==(int64_t x, const BoundedDim& y);
    friend bool operator!=(int64_t x, const BoundedDim& y);
    friend bool operator<(int64_t x, const BoundedDim& y);
    friend bool operator>(int64_t x, const BoundedDim& y);
    friend bool operator<=(int64_t x, const BoundedDim& y);
    friend bool operator>=(int64_t x, const BoundedDim& y);
};

BoundedDim operator+(int64_t x, const BoundedDim& y);
BoundedDim operator-(int64_t x, const BoundedDim& y);
BoundedDim operator*(int64_t x, const BoundedDim& y);
BoundedDim operator/(int64_t x, const BoundedDim& y);

bool operator==(int64_t x, const BoundedDim& y);
bool operator!=(int64_t x, const BoundedDim& y);
bool operator>(int64_t x, const BoundedDim& y);
bool operator<(int64_t x, const BoundedDim& y);
bool operator>=(int64_t x, const BoundedDim& y);
bool operator<=(int64_t x, const BoundedDim& y);

// Represents a possibly dynamic dimension size. For dynamic sizes the bound is stored as the size and isDynamic is set
// to true.
class MaskedDim {
public:
    using ReprType = DynamicDimsMask;

private:
    int64_t _dimValue;
    int64_t _isDynamic;

public:
    MaskedDim(int64_t dimValue, int64_t isDynamic);
    MaskedDim(int64_t dimValue);
    MaskedDim(const BoundedDim& boundedDim);

public:
    int64_t dimValue() const;
    int64_t representation() const;
    bool isDynamic() const;
    int64_t reifiedSize() const;

private:
    template <typename OperatorT>
    static MaskedDim apply(const MaskedDim& lhs, const MaskedDim& rhs, OperatorT op) {
        // static_shape  x static_shape -> static_shape
        // dynamic_shape x any_shape    -> dynamic_shape
        return MaskedDim(op(lhs._dimValue, rhs._dimValue), lhs.isDynamic() || rhs.isDynamic());
    }

public:
    MaskedDim operator+(const MaskedDim& other) const;
    MaskedDim operator-(const MaskedDim& other) const;
    MaskedDim operator*(const MaskedDim& other) const;
    MaskedDim operator/(const MaskedDim& other) const;

    MaskedDim& operator+=(const MaskedDim& other);
    MaskedDim& operator-=(const MaskedDim& other);
    MaskedDim& operator*=(const MaskedDim& other);
    MaskedDim& operator/=(const MaskedDim& other);

    bool operator==(const MaskedDim& other) const;
    bool operator!=(const MaskedDim& other) const;
    bool operator<(const MaskedDim& other) const;
    bool operator>(const MaskedDim& other) const;
    bool operator<=(const MaskedDim& other) const;
    bool operator>=(const MaskedDim& other) const;

    friend MaskedDim operator+(int64_t x, const MaskedDim& y);
    friend MaskedDim operator-(int64_t x, const MaskedDim& y);
    friend MaskedDim operator*(int64_t x, const MaskedDim& y);
    friend MaskedDim operator/(int64_t x, const MaskedDim& y);

    friend bool operator==(int64_t x, const MaskedDim& y);
    friend bool operator!=(int64_t x, const MaskedDim& y);
    friend bool operator<(int64_t x, const MaskedDim& y);
    friend bool operator>(int64_t x, const MaskedDim& y);
    friend bool operator<=(int64_t x, const MaskedDim& y);
    friend bool operator>=(int64_t x, const MaskedDim& y);
};

MaskedDim operator+(int64_t x, const MaskedDim& y);
MaskedDim operator-(int64_t x, const MaskedDim& y);
MaskedDim operator*(int64_t x, const MaskedDim& y);
MaskedDim operator/(int64_t x, const MaskedDim& y);

bool operator==(int64_t x, const MaskedDim& y);
bool operator!=(int64_t x, const MaskedDim& y);
bool operator>(int64_t x, const MaskedDim& y);
bool operator<(int64_t x, const MaskedDim& y);
bool operator>=(int64_t x, const MaskedDim& y);
bool operator<=(int64_t x, const MaskedDim& y);

namespace details {

//
// Tag Impl
//

template <class B>
class DynamicShapeTag : public B {
public:
    using Base = B;
    using Base::Base;

    using ValueType = typename Base::ValueType;

    using StaticShape = DimValues<Dim, ShapeRef::ValueType, ShapeTag>;
    using StaticShapeRef = DimValuesRef<Dim, ShapeRef::ValueType, ShapeTag>;
    using ReprType = typename ValueType::ReprType;  // Bounds / DynamicDimsMask
    using ReprRefType = RefType<ReprType>;

private:
    template <typename Out, typename Func>
    Out unzip(Func getter) const {
        Out result;
        result.reserve(this->size());
        llvm::transform(this->raw(), std::back_inserter(result), getter);
        return result;
    }

public:
    // Converts this dynamic shape to a regular shape, discarding representation data.
    StaticShape toShape() const {
        return unzip<StaticShape>([](const auto& val) {
            return val.dimValue();
        });
    }

    // Converts this dynamic shape to its reified static form (dynamic dimension sizes are replaced by their bounds).
    StaticShape toReifiedShape() const {
        return unzip<StaticShape>([](const auto& val) {
            return val.reifiedSize();
        });
    }

    // Extracts the representation data of this dynamic shape.
    ReprType toRepresentation() const {
        return unzip<ReprType>([](const auto& val) {
            return val.representation();
        });
    }

    // Extracts the representation data of this dynamic shape, converting it (if needed) to another representation.
    template <typename DynamicShapeType>
    typename DynamicShapeType::ReprType toRepresentationOf() const {
        using OutRepr = typename DynamicShapeType::ReprType;
        if constexpr (std::is_same_v<OutRepr, ReprType>) {
            return toRepresentation();
        } else {
            return unzip<OutRepr>([](const auto& val) {
                return typename DynamicShapeType::ValueType(val).representation();
            });
        }
    }

    // Returns the maximum (reified) size of this shape.
    int64_t totalSize() const {
        return std::accumulate(this->raw().begin(), this->raw().end(), int64_t{1}, [](int64_t acc, ValueType value) {
            const auto mult = acc * value.reifiedSize();
            return mult;
        });
    }
};

}  // namespace details

//
// Dynamic Shapes (and their references)
//

using BoundedShape = details::DimValues<Dim, BoundedDim, details::DynamicShapeTag>;
using BoundedShapeRef = details::DimValuesRef<Dim, BoundedDim, details::DynamicShapeTag>;

using DimsMaskedShape = details::DimValues<Dim, MaskedDim, details::DynamicShapeTag>;
using DimsMaskedShapeRef = details::DimValuesRef<Dim, MaskedDim, details::DynamicShapeTag>;

//
// Utils
//

// Creates a dynamic shape by "zipping" a regular shape with the corresponding dynamic representation.
template <typename ShapeType>
ShapeType makeShape(ShapeRef shape, typename ShapeType::ReprRefType repr) {
    VPUX_THROW_UNLESS(shape.size() == repr.size(),
                      "Got shape: {0} and dynamic representation: {1} with different ranks.", shape, repr);

    SmallVector<typename ShapeType::ValueType> result;
    result.reserve(shape.size());
    for (const auto& [shapeDim, reprDim] : llvm::zip(shape, repr)) {
        result.emplace_back(shapeDim, reprDim);
    }

    return ShapeType(std::move(result));
}

// Creates a shape of the same type as the first argument, initialized with the given value expanded to the given size.
template <typename D, typename T, template <class> class Tag, typename V = T>
details::DimValues<D, T, Tag> makeShape(const details::DimValues<D, T, Tag>&, size_t size, V value) {
    return details::DimValues<D, T, Tag>(size, value);
}

// Creates a shape of the same type as the first argument, initialized with the given value expanded to the given size.
template <typename D, typename T, template <class> class Tag, typename V = T>
details::DimValues<D, T, Tag> makeShape(details::DimValuesRef<D, T, Tag>, size_t size, V value) {
    return details::DimValues<D, T, Tag>(size, value);
}

// Creates a deep copy of the given shape ref.
template <typename D, typename T, template <class> class Tag>
details::DimValues<D, T, Tag> copyShape(details::DimValuesRef<D, T, Tag> shape) {
    return details::DimValues<D, T, Tag>(shape);
}

// Creates a copy of the given shape.
template <typename D, typename T, template <class> class Tag>
details::DimValues<D, T, Tag> copyShape(const details::DimValues<D, T, Tag>& shape) {
    return details::DimValues<D, T, Tag>(shape);
}

// Allows the following (safe) conversions between shape types:
// Shape         -> BoundedShape
// Shape         -> DimsMaskedShape
// BoundedShape <-> DimsMaskedShape
template <typename DstShapeType, typename SrcShapeType>
DstShapeType shapeCast(const SrcShapeType& src) {
    DstShapeType dst;
    dst.reserve(src.size());
    llvm::transform(src, std::back_inserter(dst), [](const auto& srcDim) {
        return typename DstShapeType::ValueType(srcDim);
    });
    return dst;
}

}  // namespace vpux

//
// Formatter
//

namespace llvm {

template <typename T>
struct format_provider<T, std::enable_if_t<llvm::is_one_of<T, vpux::BoundedDim, vpux::MaskedDim>::value>> {
    static void format(const T& dimValue, raw_ostream& stream, StringRef) {
        if (dimValue.isDynamic()) {
            stream << "1..";
        }
        stream << dimValue.reifiedSize();
    }
};

}  // namespace llvm
