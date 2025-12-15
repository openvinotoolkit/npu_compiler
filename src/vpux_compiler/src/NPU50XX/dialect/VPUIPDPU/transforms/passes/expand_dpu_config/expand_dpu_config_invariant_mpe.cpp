//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

mlir::LogicalResult vpux::VPUIPDPU::arch50xx::buildDPUInvariantMPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    if (arch40xx::buildDPUInvariantMPE(origInvOp, builder, invBlock, invBlockArgsPos).failed()) {
        return mlir::failure();
    }

    auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos);
    auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos);

    if (inAct && weights) {
        auto inActType = getBaseType(mlir::cast<mlir::MemRefType>(inAct.getType()).getElementType());
        auto wtType = getBaseType(mlir::cast<mlir::MemRefType>(weights.getType()).getElementType());

        if ((mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(inActType) &&
             mlir::isa<mlir::BFloat16Type>(wtType)) ||
            (mlir::isa<mlir::BFloat16Type>(inActType) &&
             mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(wtType))) {
            builder.create<VPUIPDPU::MPEBf16ModeOp>(origInvOp.getLoc());
        }
    }

    // mpe_daz not set/used in graph_file nce_lib so then
    // MPEDenormalOperandsFTZOp will not be instantiated here.

    return mlir::success();
}
