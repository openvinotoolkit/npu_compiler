//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux::IE {

bool hasDynamicShape(const mlir::Value value);
bool hasDynamicTensors(mlir::Operation* op);
bool needsStaticShape(mlir::Operation* op);
bool isDynamicDataContiguous(vpux::ShapeRef shape, vpux::DimsOrder order);

template <typename T>
SmallVector<T> replaceDynamicDimsWithValue(const SmallVector<int64_t>& original, T value) {
    const auto originalRank = static_cast<int64_t>(original.size());
    const auto transformDim = [value](auto dim) -> T {
        return dim != mlir::ShapedType::kDynamic ? static_cast<T>(dim) : value;
    };

    SmallVector<T> transformed(originalRank);
    transform(original, std::begin(transformed), transformDim);

    return transformed;
}

}  // namespace vpux::IE
