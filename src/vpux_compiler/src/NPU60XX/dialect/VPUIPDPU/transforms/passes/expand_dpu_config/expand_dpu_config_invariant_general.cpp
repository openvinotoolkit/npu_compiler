//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIPDPU::arch60xx::General {

mlir::LogicalResult buildGeneralDynScaleCfgInv(mlir::OpBuilder& builder, const mlir::Location& loc,
                                               VPUIP::DynamicScaleConfigAttr dynamicScaleConfigAttr, bool isContinued) {
    mlir::IntegerAttr primitiveId0Attr = nullptr;
    mlir::IntegerAttr primitiveId1Attr = nullptr;
    if (dynamicScaleConfigAttr.getInvariantPrimitiveId()) {
        auto primitiveIds = parseIntArrayAttr<int64_t>(dynamicScaleConfigAttr.getInvariantPrimitiveId());
        primitiveId0Attr = builder.getI64IntegerAttr(primitiveIds[0]);
        if (primitiveIds.size() > 1) {
            primitiveId1Attr = builder.getI64IntegerAttr(primitiveIds[1]);
        }
    }

    VPUIPDPU::DynScaleCfgInvOp::create(
            builder, loc, primitiveId0Attr, primitiveId1Attr, dynamicScaleConfigAttr.getPpeOverwriteData(), isContinued,
            dynamicScaleConfigAttr.getDynamicSkipMode(), dynamicScaleConfigAttr.getDynamicAdjustMode());

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch60xx::General

using namespace vpux::VPUIPDPU::arch60xx::General;

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUInvariantGeneral(VPUASM::DPUInvariantOp origInvOp,
                                                                       mlir::OpBuilder& builder) {
    if (auto dynamicScaleConfigAttr = origInvOp.getDynamicScaleConfigAttr()) {
        if (buildGeneralDynScaleCfgInv(builder, origInvOp.getLoc(), dynamicScaleConfigAttr, origInvOp.getIsContinued())
                    .failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}
