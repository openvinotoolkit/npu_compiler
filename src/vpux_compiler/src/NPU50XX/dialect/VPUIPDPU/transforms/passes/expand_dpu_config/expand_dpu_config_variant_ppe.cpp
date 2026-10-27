//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

mlir::LogicalResult vpux::VPUIPDPU::arch50xx::buildDPUVariantPPE(
        VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder,
        vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode) {
    if (npu5PPEBackwardsCompatibilityMode == vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode::ENABLED) {
        return mlir::success();
    }

    if (origVarOp.getSprLutRead()) {
        PPEsprLUTReadOp::create(builder, origVarOp.getLoc());
    }

    return mlir::success();
}
