//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUVariantPPE(VPUASM::DPUVariantOp origVarOp,
                                                                 mlir::OpBuilder& builder) {
    if (origVarOp.getSprLutRead()) {
        PPEsprLUTReadOp::create(builder, origVarOp.getLoc());
    }

    return mlir::success();
}
