//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"

namespace vpux::VPUIP {

// A DDR buffer type with a dimension without an upper bound gets a VPURT.Alloc placeholder with no size.
// FuseLastCopy may replace it with a network output; otherwise the DDR StaticAllocation pass rejects it.
SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::Type bufferType, bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::RewriterBase& rewriter,
                                               mlir::Value value, vpux::IndexedSymbolAttr memSpace,
                                               bool individualBuffers = false);

//
// allocateBuffers & allocateBuffersForValue using bufferizable interface
//

SmallVector<mlir::Value> allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                 mlir::Value value, bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::ValueRange values, bool individualBuffers = false);

mlir::Value allocateBuffer(const Logger& log, mlir::Location loc, mlir::RewriterBase& rewriter, mlir::Value value,
                           vpux::IndexedSymbolAttr memSpace);

}  // namespace vpux::VPUIP
