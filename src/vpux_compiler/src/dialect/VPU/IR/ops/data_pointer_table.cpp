//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/utils/verifier_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DataPointerTableOp::verify() {
    if (const auto zp = getZeroPoints()) {
        if (mlir::failed(isZeroPointsTypeValidForQuantization(*this, zp))) {
            return mlir::failure();
        }
    }
    return mlir::success();
}
