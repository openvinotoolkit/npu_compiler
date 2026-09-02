//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <optional>

namespace vpux {
namespace VPUIP {

// Builds a DistributedBufferType only if the layout rank and distribution are
// valid for the requested logical shape; returns nullopt otherwise. This is used
// by speculative back-inference paths where an invalid distribution should make
// the candidate fail quietly instead of emitting verifier diagnostics.
std::optional<VPUIP::DistributedBufferType> createDistributedBufferTypeOrNull(
        mlir::MLIRContext* ctx, ShapeRef shape, mlir::Type elementType, mlir::MemRefLayoutAttrInterface order,
        IndexedSymbolAttr memSpace, VPU::DistributionInfoAttr distribution,
        VPUIP::SparsityCompressionAttr sparsityCompression = nullptr);

// Non-throwing variant of VPU::getPerClusterMemoryShapes. For explicit
// distributions it reads the stored memory view; for implicit distributions it
// derives it from shape, distribution and element type. Returns nullopt instead
// of failing when the per-cluster view cannot be computed.
std::optional<SmallVector<Shape>> getPerClusterMemoryShapesOrNull(VPUIP::DistributedBufferType distributedType);

// Returns whether per-cluster memory shapes and offsets can be computed for this
// distributed type.
bool isSupportedPerClusterMemoryShapesAndOffsets(VPUIP::DistributedBufferType distributedType);

}  // namespace VPUIP
}  // namespace vpux
