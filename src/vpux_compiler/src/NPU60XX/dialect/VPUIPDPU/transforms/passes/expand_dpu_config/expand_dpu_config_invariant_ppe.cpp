//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_ppe.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

using namespace vpux;
using namespace VPUIPDPU;

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::PPE::configurePPE(arch50xx::PPE::PPEConfig& config,
                                                                mlir::Type outDataType, VPUIP::NCETaskType dpuTaskType,
                                                                const arch50xx::PPE::PPETask& ppeTask,
                                                                bool isWeightTableProvided,
                                                                bool isSprLookUpTableProvided) {
    if (arch50xx::PPE::configurePPE(config, outDataType, dpuTaskType, ppeTask, isWeightTableProvided,
                                    isSprLookUpTableProvided)
                .failed()) {
        return mlir::failure();
    }

    // spr lookup table
    if (isSprLookUpTableProvided) {
        if (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::SIN) {
            config.scaleMult.scaleStatic = ppeTask.fpScaleData;
        }
        if (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::COS) {
            config.scaleMult.scaleStatic = ppeTask.fpScaleData;
            config.biasAdd.biasStatic = ppeTask.fpBias;
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUInvariantPPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos, ELF::SymbolReferenceMap& symRefMap) {
    if (!origInvOp.getPpe().hasOneBlock()) {
        log.error("VPUASM::DPUInvariant->PPE is not a single block region");
        return mlir::failure();
    }

    auto ppeTask = arch50xx::PPE::evalPPETasks(log, origInvOp.getPpe());
    if (mlir::failed(ppeTask)) {
        return mlir::failure();
    }

    mlir::MemRefType outDataType;
    if (auto outAct = getInvBlockArg(BlockArg::ACT_OUT, invBlock, invBlockArgsPos)) {
        outDataType = mlir::cast<mlir::MemRefType>(outAct.getType());
    } else {
        outDataType = VPUASM::getMinMaxDataType(origInvOp, symRefMap);
    }

    if (!outDataType) {
        log.error("Both activation and min/max outputs are missing");
        return mlir::failure();
    }

    auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos);
    if (!inAct) {
        // If there is no input activation, it means that the PPE is not used in this DPU task.
        // This is the case for ACCU_STORE, which does not use PPE.

        return mlir::success();
    } else {
        auto inActElementType = mlir::cast<mlir::MemRefType>(inAct.getType()).getElementType();
        auto inBaseType = getBaseType(inActElementType);
        if (inBaseType.isF32() || inBaseType.isInteger(CHAR_BIT * sizeof(int32_t))) {
            // If the input activation is I32/F32, it means that the PPE is not used in this DPU task.
            // This is the case for ACCU_LOAD, which does not use PPE.

            return mlir::success();
        }
    }

    arch50xx::PPE::PPEConfig config;
    auto dpuTaskType = origInvOp.getNceTaskType();
    if (arch60xx::PPE::configurePPE(
                config, outDataType.getElementType(), dpuTaskType, ppeTask.value(),
                getInvBlockArg(BlockArg::WEIGHTS_TABLE, invBlock, invBlockArgsPos) != nullptr ||
                        getInvBlockArg(BlockArg::WEIGHTS_TABLE_SCALE, invBlock, invBlockArgsPos) != nullptr,
                getInvBlockArg(BlockArg::SPR_LOOKUP_TABLE, invBlock, invBlockArgsPos) != nullptr)
                .failed()) {
        return mlir::failure();
    }

    auto weightsTableScale = getInvBlockArg(BlockArg::WEIGHTS_TABLE, invBlock, invBlockArgsPos);
    if (!weightsTableScale) {
        weightsTableScale = getInvBlockArg(BlockArg::WEIGHTS_TABLE_SCALE, invBlock, invBlockArgsPos);
    }
    if (arch50xx::PPE::buildPPEConfig(builder, origInvOp.getLoc(), log, config, weightsTableScale,
                                      /*hasBypassOp*/ false)
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
