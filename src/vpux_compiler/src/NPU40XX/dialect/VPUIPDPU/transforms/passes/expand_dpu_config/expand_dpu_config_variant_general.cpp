//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_general.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"

namespace vpux::VPUIPDPU::arch40xx::General {
mlir::LogicalResult buildGeneralForceInvRead(mlir::OpBuilder& builder, const mlir::Location& loc,
                                             std::optional<bool> forceInvRead) {
    if (forceInvRead.value_or(false)) {
        VPUIPDPU::ForceInvReadOp::create(builder, loc);
    }

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch40xx::General

using namespace vpux::VPUIPDPU::arch40xx::General;

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUVariantGeneral(VPUASM::DPUVariantOp origVarOp,
                                                                     mlir::OpBuilder& builder,
                                                                     const Logger& /*logger*/) {
    if (buildGeneralForceInvRead(builder, origVarOp.getLoc(), origVarOp.getForceInvRead()).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
