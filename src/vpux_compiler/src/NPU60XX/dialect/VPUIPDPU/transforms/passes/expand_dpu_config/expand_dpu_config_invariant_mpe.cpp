//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace {

bool isValidWeightType(mlir::Type weightType, mlir::Type inActType, mlir::Type elemType) {
    if ((weightType.isSignedInteger(8) && inActType.isF16()) ||
        (weightType.isInteger() && !weightType.isUnsignedInteger(8))) {
        return true;
    }

    if (auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        if (auto quantileStorage = mlir::dyn_cast<mlir::quant::QuantileType>(quantizedType.getStorageType())) {
            auto storageType = quantileStorage.getStorageType();
            auto lutType = quantileStorage.getQuantileType();
            bool isValidWidth = storageType.getIntOrFloatBitWidth() == 4;
            bool isValidLut = (lutType.isSignedInteger(8) || lutType.isUnsignedInteger(8));
            return isValidWidth && isValidLut && inActType.isF16();
        }
    }

    return false;
}

}  // namespace

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUInvariantMPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    if (arch50xx::buildDPUInvariantMPE(origInvOp, builder, invBlock, invBlockArgsPos).failed()) {
        return mlir::failure();
    }

    auto mpeEngine60xx = mlir::dyn_cast_or_null<VPU::MPEEngine60XXAttr>(origInvOp.getMpeEngineAttr());

    if (auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos)) {
        auto inActType = getBaseType(mlir::cast<mlir::MemRefType>(inAct.getType()).getElementType());
        if (inActType.isInteger(16)) {
            const int64_t activationZp =
                    mpeEngine60xx != nullptr && mpeEngine60xx.getActivationZp()
                            ? mlir::cast<mlir::IntegerAttr>(mpeEngine60xx.getActivationZp()).getInt()
                            : 0;
            MPEActivationBiasOp::create(builder, origInvOp.getLoc(), static_cast<uint16_t>(activationZp));
        }
        if (mpeEngine60xx != nullptr) {
            const auto arch = config::getArch(origInvOp);
            // For enabling code reuse in NPU60XX+, verify only for NPU60XX the DCIM restrictions specific to NPU60XX
            if (arch == config::ArchKind::NPU60XX) {
                auto inActNDType = mlir::cast<NDTypeInterface>(inAct.getType());
                if (mpeEngine60xx.getMode().getValue() == VPU::MPEEngine60XXMode::DCIM &&
                    inActNDType.getShape()[Dims4D::Act::C] % 16 != 0) {
                    log.error("DCIM MPE Engine option is available only for input channels which are a multiple of "
                              "16");
                    return mlir::failure();
                }
            }
            MPEEngineModeOp::create(builder, origInvOp.getLoc(), mpeEngine60xx.getMode());
        }

        if (auto inWeight = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
            auto elemType = mlir::cast<NDTypeInterface>(inWeight.getType()).getElementType();
            auto weightType = getBaseType(mlir::cast<mlir::MemRefType>(inWeight.getType()).getElementType());

            // mpe_wtbias register field stores the per-tensor zero-point for int8/uint8/int4/uint4 weights
            if (isValidWeightType(weightType, inActType, elemType) && mpeEngine60xx != nullptr) {
                // mpe_wtbias register field stores the per-tensor zero-point.
                // Avoid double configuration when zero-point table is created.
                if (getInvBlockArg(BlockArg::WEIGHTS_ZERO_POINTS, invBlock, invBlockArgsPos)) {
                    return mlir::success();
                }

                // With data-pointer table: set only when weight ZP attribute is present.
                // Without data-pointer table: set if weight ZP attribute is present.
                if (const auto weightZpAttr = mpeEngine60xx.getWeightZp()) {
                    MPEWeightsBiasOp::create(
                            builder, origInvOp.getLoc(),
                            static_cast<uint8_t>(mlir::cast<mlir::IntegerAttr>(weightZpAttr).getInt()));
                }
            }
        }
    }

    return mlir::success();
}
