//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUInvariantMPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    auto mpeEngineAttr = mlir::dyn_cast_or_null<VPU::MPEEngineAttr>(origInvOp.getMpeEngineAttr());

    if (auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos)) {
        auto inActType = getBaseType(mlir::cast<mlir::MemRefType>(inAct.getType()).getElementType());
        if (inActType.isInteger(8)) {
            const int64_t activationZp =
                    mpeEngineAttr != nullptr && mpeEngineAttr.getActivationZp()
                            ? mlir::cast<mlir::IntegerAttr>(mpeEngineAttr.getActivationZp()).getInt()
                            : 0;
            builder.create<MPEActivationBiasOp>(origInvOp.getLoc(), static_cast<uint8_t>(activationZp));
        }
    }

    if (auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
        auto wtType = getBaseType(mlir::cast<mlir::MemRefType>(weights.getType()).getElementType());
        if (wtType.isUnsignedInteger(8)) {
            const int64_t weightZp = mpeEngineAttr != nullptr && mpeEngineAttr.getWeightZp()
                                             ? mlir::cast<mlir::IntegerAttr>(mpeEngineAttr.getWeightZp()).getInt()
                                             : 0;
            builder.create<MPEWeightsBiasOp>(origInvOp.getLoc(), static_cast<uint8_t>(weightZp));
        }
    }

    // mpe_daz not set/used in graph_file nce_lib so then
    // MPEDenormalOperandsFTZOp will not be instantiated here.

    return mlir::success();
}
