//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIPDPU::arch60xx::General {
mlir::LogicalResult buildGeneralDynPrimitiveIdVar(mlir::OpBuilder& builder, const mlir::Location& loc,
                                                  const SmallVector<int64_t>&& primitiveIds) {
    mlir::IntegerAttr primitiveId1Attr = nullptr;

    if (primitiveIds.size() > 1) {
        primitiveId1Attr = builder.getI64IntegerAttr(primitiveIds[1]);
    }
    VPUIPDPU::DynPrimitiveIdVarOp::create(builder, loc, primitiveIds[0], primitiveId1Attr);

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch60xx::General

using namespace vpux::VPUIPDPU::arch60xx::General;

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUVariantGeneral(VPUASM::DPUVariantOp origVarOp,
                                                                     mlir::OpBuilder& builder,
                                                                     const Logger& /*logger*/) {
    if (arch40xx::General::buildGeneralForceInvRead(builder, origVarOp.getLoc(), origVarOp.getForceInvRead())
                .failed()) {
        return mlir::failure();
    }

    if (origVarOp.getVariantPrimitiveIdAttr()) {
        if (buildGeneralDynPrimitiveIdVar(builder, origVarOp.getLoc(),
                                          parseIntArrayAttr<int64_t>(origVarOp.getVariantPrimitiveIdAttr()))
                    .failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}
