//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux::VPUIP {

struct BoundedBufferComponents {
    mlir::MemRefType dataType;
    mlir::MemRefType dynamicShapeType;
};

BoundedBufferComponents unpackBoundedBufferType(VPUIP::BoundedBufferType type);
// Same as unpackBoundedBufferType() but uses a memref with bounds as an input.
BoundedBufferComponents unpackDynamicMemRefType(mlir::MemRefType type, BoundsRef bounds = {});

void addShapeTensorDataInfo(mlir::func::FuncOp funcOp, mlir::MemRefType dynamicShapeMemRef, mlir::Block& infoBlock,
                            mlir::StringRef dataInfoName, size_t dataBufferCount);

}  // namespace vpux::VPUIP
