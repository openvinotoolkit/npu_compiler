//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include <cstddef>

namespace vpux {
class DimsOrder;

DimsOrder inferNewDimsOrder(const DimsOrder& origOrder, size_t numShapeDims);

/// @brief Checks if the given type is mlir::MemRefType with mlir::StridedLayoutAttr
bool hasStridedLayout(mlir::Type type);

}  // namespace vpux
