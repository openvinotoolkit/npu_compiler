//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

namespace vpux {

mlir::IntegerAttr getAllocSizeAttr(mlir::Type type);

vpux::NDTypeInterface setAllocSizeAttr(vpux::NDTypeInterface type, int64_t allocSize);

DimsOrder inferNewDimsOrder(DimsOrder origOrder, size_t numShapeDims);

}  // namespace vpux
