//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

#include <functional>

using namespace vpux;
using namespace vpux::VPUIPDPU;
using namespace mlir;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.cpp.inc>

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult ODULocalRegionOp::verify() {
    const auto beginX = getBeginCoordX();
    const auto endX = getEndCoordX();
    const auto beginY = getBeginCoordY();
    const auto endY = getEndCoordY();
    const auto suppressOutput = getSuppressOutput();
    if (!suppressOutput && (beginX > endX || beginY > endY)) {
        return errorAt(getLoc(), "Operation {0} has invalid coordinates: beginX={1}, endX={2}, beginY={3}, endY={4}.",
                       getOperationName(), beginX, endX, beginY, endY);
    }
    if (beginX < 0 || beginY < 0 || endX < 0 || endY < 0) {
        return errorAt(getLoc(),
                       "Operation {0} has invalid coordinates: beginX={1}, endX={2}, beginY={3}, endY={4}; Negative "
                       "values are not allowed.",
                       getOperationName(), beginX, endX, beginY, endY);
    }

    return ::mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
