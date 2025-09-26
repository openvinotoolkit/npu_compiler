//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dim_values.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace vpux {

//
// Shape utils
//

namespace details {

bool isDynamicDimValues(ArrayRef<int64_t> shape);
int64_t calcTotalShapeSize(ArrayRef<int64_t> shape);

}  // namespace details

//
// ShapeImpl
//

namespace details {

template <class Base>
class ShapeTag : public Base {
public:
    using Base::Base;

public:
    bool isDynamic() const {
        return details::isDynamicDimValues(this->raw());
    }

    bool isStatic() const {
        return !isDynamic();
    }

public:
    int64_t totalSize() const {
        return details::calcTotalShapeSize(this->raw());
    }
};

}  // namespace details

//
// Shape
//

using Shape = details::DimValues<Dim, int64_t, details::ShapeTag>;
using ShapeRef = details::DimValuesRef<Dim, int64_t, details::ShapeTag>;

ShapeRef getShape(mlir::Value val);

// Returns the upper bounds of the shape if it is bounded, otherwise returns the shape itself.
ShapeRef getBoundedShape(mlir::Value val);
ShapeRef getBoundedShape(mlir::Type type);

//
// MemShape
//

using MemShape = details::DimValues<MemDim, int64_t, details::ShapeTag>;
using MemShapeRef = details::DimValuesRef<MemDim, int64_t, details::ShapeTag>;

MemShape getMemShape(mlir::Value val);

// Returns the upper bounds of the shape if it is bounded, otherwise returns the shape itself.
MemShape getBoundedMemShape(mlir::Value val);
MemShape getBoundedMemShape(mlir::Type type);

MemShape getMemIndexND(int64_t memIndex1D, MemShapeRef memShape);
int64_t getMemIndex1D(MemShapeRef memIndexND, MemShapeRef memShape);

}  // namespace vpux
