//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/builders.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

#include <utility>

namespace vpux::bytecode {

SmallVector<int64_t> getStridesWithStaticZeroOffset(mlir::MemRefType memrefType, StringRef diagnosticContext) {
    auto [strides, offset] = memrefType.getStridesAndOffset();
    VPUX_THROW_WHEN(mlir::ShapedType::isDynamic(offset), "{0} cannot encode dynamic offset for memref {1}",
                    diagnosticContext, memrefType);
    VPUX_THROW_WHEN(offset != 0, "{0} cannot encode non-zero offset {1} for memref {2}", diagnosticContext, offset,
                    memrefType);
    return std::move(strides);
}

}  // namespace vpux::bytecode
