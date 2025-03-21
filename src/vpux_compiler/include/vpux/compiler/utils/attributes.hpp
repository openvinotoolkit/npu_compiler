//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

//
// get<Scalar>Attr
//

template <typename T>
mlir::IntegerAttr getIntAttr(mlir::MLIRContext* ctx, T val) {
    return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), checked_cast<int64_t>(val));
}
template <typename T>
mlir::IntegerAttr getIntAttr(mlir::Builder& b, T val) {
    return getIntAttr(b.getContext(), val);
}

template <typename T>
mlir::FloatAttr getFPAttr(mlir::MLIRContext* ctx, T val) {
    return mlir::FloatAttr::get(mlir::FloatType::getF64(ctx), checked_cast<double>(val));
}
template <typename T>
mlir::FloatAttr getFPAttr(mlir::Builder& b, T val) {
    return getFPAttr(b.getContext(), val);
}

//
// get<Scalar>ArrayAttr
//

template <class Range>
mlir::ArrayAttr getIntArrayAttr(mlir::MLIRContext* ctx, Range&& range) {
    SmallVector<mlir::Attribute> attrs;

    for (auto&& val : range) {
        attrs.push_back(getIntAttr(ctx, val));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}
template <class Range>
mlir::ArrayAttr getIntArrayAttr(mlir::Builder& b, Range&& range) {
    return getIntArrayAttr(b.getContext(), std::forward<Range>(range));
}

template <class Range>
mlir::ArrayAttr getFPArrayAttr(mlir::MLIRContext* ctx, Range&& range) {
    SmallVector<mlir::Attribute> attrs;

    for (auto&& val : range) {
        attrs.push_back(getFPAttr(ctx, val));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}
template <class Range>
mlir::ArrayAttr getFPArrayAttr(mlir::Builder& b, Range&& range) {
    return getFPArrayAttr(b.getContext(), std::forward<Range>(range));
}

//
// getArrayOfArrayAttr
//

template <class Range>
mlir::ArrayAttr getIntArrayOfArray(mlir::MLIRContext* ctx, Range&& arrayOfArray) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto& val : arrayOfArray) {
        attrs.push_back(getIntArrayAttr(ctx, val));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getFPArrayOfArray(mlir::MLIRContext* ctx, Range&& arrayOfArray) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto& val : arrayOfArray) {
        attrs.push_back(getFPArrayAttr(ctx, val));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

//
// parse<Scalar>Attr
//

template <typename T>
T parseIntAttr(mlir::Attribute attr) {
    const auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>();
    VPUX_THROW_UNLESS(intAttr != nullptr, "Got non Integer Attribute '{0}'", attr);
    if (intAttr.getType().isUnsignedInteger()) {
        return checked_cast<T>(intAttr.getUInt());
    }

    return checked_cast<T>(intAttr.getValue().getSExtValue());
}

//
// parse<Scalar>ArrayAttr
//

template <typename T>
SmallVector<T> parseIntArrayAttr(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>();
                               VPUX_THROW_UNLESS(intAttr != nullptr, "Got non Integer Attribute '{0}' in Array", attr);

                               if (intAttr.getType().isUnsignedInteger()) {
                                   return checked_cast<T>(intAttr.getUInt());
                               }

                               return checked_cast<T>(intAttr.getValue().getSExtValue());
                           }));
}

template <typename T>
SmallVector<T> parseFPArrayAttr(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto fpAttr = attr.dyn_cast_or_null<mlir::FloatAttr>();
                               VPUX_THROW_UNLESS(fpAttr != nullptr, "Got non fpAttr Attribute '{0}' in Array", attr);

                               return checked_cast<T>(fpAttr.getValueAsDouble());
                           }));
}

//
// parse<CustomizedAttr>ArrayAttr
//

template <typename T>
SmallVector<T> parseCustomAttrArray(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto customAttr = attr.dyn_cast_or_null<T>();
                               VPUX_THROW_UNLESS(customAttr != nullptr, "Got non-required Attribute '{0}' in Array",
                                                 attr);
                               return customAttr;
                           }));
}

//
// parseArrayOfArrayAttr
//

template <typename T>
SmallVector<SmallVector<T>> parseIntArrayOfArrayAttr(mlir::ArrayAttr arr) {
    SmallVector<SmallVector<T>> arrayOfArray;
    arrayOfArray.reserve(arr.size());

    for (const auto attr : arr) {
        const auto arrAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>();
        VPUX_THROW_UNLESS(arrAttr != nullptr, "Got non Array Attribute '{0}' in Array", attr);

        arrayOfArray.push_back(parseIntArrayAttr<T>(arrAttr));
    }

    return arrayOfArray;
}

template <typename T>
SmallVector<SmallVector<T>> parseFPArrayOfArrayAttr(mlir::ArrayAttr arr) {
    SmallVector<SmallVector<T>> arrayOfArray;
    arrayOfArray.reserve(arr.size());

    for (const auto attr : arr) {
        const auto arrAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>();
        VPUX_THROW_UNLESS(arrAttr != nullptr, "Got non Array Attribute '{0}' in Array", attr);

        arrayOfArray.push_back(parseFPArrayAttr<T>(arrAttr));
    }

    return arrayOfArray;
}

/// Returns a dense_resource<> key.
mlir::StringRef getResourceName(mlir::DenseResourceElementsAttr attr);
/// Returns a dense_resource<> key. If the attribute is not a resource, an empty
/// string is returned.
mlir::StringRef getResourceName(mlir::ElementsAttr attr);

}  // namespace vpux
