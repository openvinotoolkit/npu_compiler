//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/strides.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPURT/IR/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>

using namespace vpux;

//
// Strides utils
//

bool vpux::details::isDynamicDimValues(ArrayRef<Bit> strides) {
    return std::any_of(strides.begin(), strides.end(), [](Bit val) {
        return val.count() < 0;
    });
}

//
// Strides
//

Strides vpux::getStrides(mlir::Value val) {
    const auto type = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
    VPUX_THROW_WHEN(type == nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getStrides();
}

//
// MemStrides
//

MemStrides vpux::getMemStrides(mlir::Value val) {
    const auto type = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
    VPUX_THROW_WHEN(type == nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getMemStrides();
}
