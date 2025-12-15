//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
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
#include <vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.cpp.inc>

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult PPEFpScaleMultOp::verify() {
    auto scaleTableExists = (getScaleTable() != nullptr);
    auto scaleStaticExists = getScaleStatic().has_value();

    // scale_table only
    if (scaleTableExists && !scaleStaticExists) {
        return ::mlir::success();
    }

    // scale_static only
    if (scaleStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(getLoc(), "Operation {0} needs either scale_table or scale_static as parameter", getOperationName());
}

}  // namespace VPUIPDPU
}  // namespace vpux
