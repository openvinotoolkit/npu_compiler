//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux::VPU {

bool checkStrategyCompatibilityReduce(VPU::MultiClusterStrategy strategy, size_t numTiles, ShapeRef inShape,
                                      ArrayRef<int64_t> axesVec);

bool fitIntoCMXReduce(mlir::Operation* operation, llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

bool fitIntoCMXReduce(mlir::Operation* operation, llvm::ArrayRef<vpux::NDTypeInterface> buffers);

}  // namespace vpux::VPU
