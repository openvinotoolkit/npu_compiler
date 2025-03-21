//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {

// Get sparsify value and update element type to storage type
int64_t getSparsifyValue(mlir::Type& inputElementType);
int64_t getValuesPerSparsityBit(mlir::Type& elementType);
SmallVector<int64_t> countNonSparseElementsPerOC(const Const::Content& content, mlir::Type elementType);

}  // namespace vpux
