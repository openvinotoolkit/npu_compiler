//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_odu.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

using namespace VPUIPDPU;
using namespace VPUIPDPU::arch40xx::ODU;

namespace vpux::VPUIPDPU::arch40xx::ODU {

mlir::LogicalResult configureOutTensorSize(const Logger& log, ODUConfig::OutTensorSize& config,
                                           ODUPermuteDataMode permuteMode, const Strides& outStrides) {
    auto outStridesVec = getFloatStrides(outStrides);
    if (std::count(outStridesVec.begin(), outStridesVec.end(), 0.0f)) {
        log.error("One or more activation output strides are zero, invalid configuration");
        return mlir::failure();
    }

    auto strideB = outStridesVec[Dims4D::Act::N.ind()];
    auto strideX = outStridesVec[Dims4D::Act::W.ind()];
    auto strideY = outStridesVec[Dims4D::Act::H.ind()];
    auto strideZ = outStridesVec[Dims4D::Act::C.ind()];

    switch (permuteMode) {
    case ODUPermuteDataMode::PERMUTE_ZXY:
        // NHWC
        config.dimY = strideB / strideY;
        config.dimX = strideY / strideX;
        config.dimZ = strideX / strideZ;
        break;
    case ODUPermuteDataMode::PERMUTE_YZX:
        // NWCH
        config.dimX = strideB / strideX;
        config.dimZ = strideX / strideZ;
        config.dimY = strideZ / strideY;
        break;
    case ODUPermuteDataMode::PERMUTE_ZYX:
        // NWHC
        config.dimX = strideB / strideX;
        config.dimY = strideX / strideY;
        config.dimZ = strideY / strideZ;
        break;
    case ODUPermuteDataMode::PERMUTE_XZY:
        // NHCW
        config.dimY = strideB / strideY;
        config.dimZ = strideY / strideZ;
        config.dimX = strideZ / strideX;
        break;
    case ODUPermuteDataMode::PERMUTE_YXZ:
        // NCWH
        config.dimZ = strideB / strideZ;
        config.dimX = strideZ / strideX;
        config.dimY = strideX / strideY;
        break;
    case ODUPermuteDataMode::PERMUTE_XYZ:
        // NCHW
        config.dimZ = strideB / strideZ;
        config.dimY = strideZ / strideY;
        config.dimX = strideY / strideX;
        break;
    default:
        log.error("Wrong permutation {0}, invalid configuration", stringifyODUPermuteDataMode(permuteMode));
        return mlir::failure();
    }

    if (!(config.dimX && config.dimY && config.dimZ)) {
        log.error("All output dimensions must be >= 1: dimX={0}, dimY={1}, dimZ={2}", config.dimX, config.dimY,
                  config.dimZ);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureDataReuse(const Logger& log, ODUConfig::DataReuse& config, VPU::MPEMode mpeFrequentMode,
                                       VPUIP::NCETaskType dpuTaskType) {
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (dpuTaskType == VPUIP::NCETaskType::ELTWISE) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_8x16;
    }

    switch (mpeFrequentMode) {
    case VPU::MPEMode::CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        config.activationReuse = ODUActivationReuseMode::NTHW_4;
        break;
    case VPU::MPEMode::CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    case VPU::MPEMode::VECTOR:       // dpu_runtime::MPE_GRID_16x1 not valid for NPU37XX
        config.activationReuse = ODUActivationReuseMode::NTHW_8;
        break;
    case VPU::MPEMode::CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        config.activationReuse = ODUActivationReuseMode::NTHW_16;
        break;
    default:
        log.error("ODU NTHW mode not supported for MPE mode {0}", mpeFrequentMode);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configurePermuteMode(const Logger& log, ODUConfig::PermuteData& config,
                                         const DimsOrder& outDimsOrder) {
    if (outDimsOrder == DimsOrder::NHWC) {
        config.permuteMode = ODUPermuteDataMode::PERMUTE_ZXY;
    } else if (outDimsOrder == DimsOrder::NWHC) {
        config.permuteMode = ODUPermuteDataMode::PERMUTE_ZYX;
    } else if (outDimsOrder == DimsOrder::NWCH) {
        config.permuteMode = ODUPermuteDataMode::PERMUTE_YZX;
    } else if (outDimsOrder == DimsOrder::NCWH) {
        config.permuteMode = ODUPermuteDataMode::PERMUTE_YXZ;
    } else if (outDimsOrder == DimsOrder::NHCW) {
        config.permuteMode = ODUPermuteDataMode::PERMUTE_XZY;
    } else if (outDimsOrder == DimsOrder::NCHW) {
        config.permuteMode = ODUPermuteDataMode::PERMUTE_XYZ;
    } else {
        log.error("Can't get ODU permutation by output dimsOrder: '{0}'", outDimsOrder);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureSwizzleData(const Logger& log, ODUConfig::SwizzleData& config,
                                         std::optional<int64_t> outSwizzling) {
    if (outSwizzling.has_value()) {
        auto swizzleKey = symbolizeDPUSwizzleKey(outSwizzling.value());
        if (!swizzleKey.has_value()) {
            log.error("Invalid output swizzle key '{0}'", outSwizzling.value());
            return mlir::failure();
        }
        config.swizzleKey = swizzleKey.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureOutActivations(const Logger& log, ODUConfig::OutActivations& config,
                                            mlir::Type outDataType) {
    auto outDataWidth = VPUIPDPU::getOutDataWidth(outDataType);

    if (!outDataWidth.has_value()) {
        log.error("Invalid output data type '{0}'", outDataType);
        return mlir::failure();
    }

    if (mlir::isa<mlir::quant::QuantizedType>(outDataType)) {
        config.dataWidth = outDataWidth.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureSparsity(const Logger&, ODUConfig::Sparsity& config, bool outSparsityEnabled,
                                      int64_t sparseValue) {
    if (outSparsityEnabled) {
        config.compressionEnabled = true;
        config.sparseValue = sparseValue;
    }
    return mlir::success();
}

mlir::LogicalResult configureMemoryMode(const Logger& log, ODUConfig::MemoryMode& config,
                                        std::optional<bool> isSuperdense) {
    if (isSuperdense.has_value()) {
        auto memMode = symbolizeODUMemoryMode(isSuperdense.value());
        if (!memMode.has_value()) {
            log.error("Invalid output mode '{0}'", isSuperdense.value());
            return mlir::failure();
        }
        config.memMode = memMode.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureODU(const Logger& log, ODUConfig& config, const NDTypeInterface& outActType,
                                 VPU::MPEMode mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                 std::optional<int64_t> outSwizzling, std::optional<bool> isSuperdense,
                                 bool outSparsityEnabled) {
    if (configurePermuteMode(log, config.permuteData, outActType.getDimsOrder()).failed()) {
        return mlir::failure();
    }

    if (configureOutTensorSize(log, config.outTensorSize, config.permuteData.permuteMode, outActType.getStrides())
                .failed()) {
        return mlir::failure();
    }

    if (configureDataReuse(log, config.dataReuse, mpeFrequentMode, dpuTaskType).failed()) {
        return mlir::failure();
    }
    if (VPUIPDPU::arch40xx::ODU::configureSparsity(log, config.sparsity, outSparsityEnabled,
                                                   VPUIPDPU::getZeroPoint(outActType))
                .failed()) {
        return mlir::failure();
    }

    if (configureSwizzleData(log, config.swizzleData, outSwizzling).failed()) {
        return mlir::failure();
    }

    if (configureOutActivations(log, config.outActivations, outActType.getElementType()).failed()) {
        return mlir::failure();
    }

    if (configureMemoryMode(log, config.memoryMode, isSuperdense).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult buildODUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const ODUConfig& config, mlir::Value outAct, mlir::Value outSparsityMap) {
    // ODUOutTensorSize
    builder.create<ODUOutTensorSizeOp>(loc, config.outTensorSize.dimX, config.outTensorSize.dimY,
                                       config.outTensorSize.dimZ);

    // ODUDataReuse
    if (config.dataReuse.activationReuse != ODUActivationReuseMode::NTHW_1) {
        builder.create<ODUDataReuseOp>(loc, config.dataReuse.activationReuse);
    }

    // ODUPermuteData
    if (config.permuteData.permuteMode != ODUPermuteDataMode::PERMUTE_ZXY) {
        builder.create<ODUPermuteDataOp>(loc, config.permuteData.permuteMode);
    }

    // ODUSparsity
    if (config.sparsity.compressionEnabled.has_value()) {
        if (!outSparsityMap) {
            log.error("Expected output_sparsity_map operand in ODU invariant");
            return mlir::failure();
        }
        mlir::IntegerAttr sparseValue = nullptr;
        if (config.sparsity.sparseValue.has_value() && config.sparsity.sparseValue.value()) {
            sparseValue = builder.getI64IntegerAttr(config.sparsity.sparseValue.value());
        }
        builder.create<ODUSparsityOp>(loc, outSparsityMap, nullptr, sparseValue);
    }

    // ODUSwizzleData
    if (config.swizzleData.swizzleKey != DPUSwizzleKey::SWIZZLE_OFF) {
        builder.create<ODUSwizzleDataOp>(loc, config.swizzleData.swizzleKey);
    }

    // ODUOutActivations
    if (config.outActivations.dataWidth.has_value()) {
        auto dataWidthAttr = ODUDataBitWidthAttr::get(builder.getContext(), config.outActivations.dataWidth.value());
        builder.create<ODUOutActivationsOp>(loc, outAct, nullptr, dataWidthAttr);
    } else {
        builder.create<ODUOutActivationsOp>(loc, outAct, nullptr, nullptr);
    }

    // ODUMemoryMode
    if (config.memoryMode.memMode != ODUMemoryMode::MODE_DENSE) {
        builder.create<ODUMemoryModeOp>(loc, config.memoryMode.memMode);
    }

    // ODUCmxPorts
    // invariant.odu_cfg.odu_cfg_bf.cmx_port_muxing_disable not set in nce_lib
    // leaving field value to reset initialization in lowering VPUASM2NPUReg40XX

    // ODUWriteCombineBuffer
    // invariant.odu_cfg.odu_cfg_bf.wcb_ac_mode/wcb_sp_mode not set in nce_lib
    // leaving field values to reset initialization in lowering VPUASM2NPUReg40XX

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch40xx::ODU

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUInvariantODU(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos, ELF::SymbolReferenceMap& symRefMap) {
    ODUConfig config;
    mlir::MemRefType outType;
    uint64_t outputSwizzling = 0;
    if (!origInvOp.getIsContinued() && origInvOp.getOutput()) {
        auto outBuffer = symRefMap.lookupSymbol(origInvOp.getOutput().value());
        outType = getBufferType(outBuffer);
        outputSwizzling = getSwizzlingKey(outBuffer);
    } else if (origInvOp.getIsContinued() && origInvOp.getOutputTypeContinued()) {
        auto outBufferType = origInvOp.getOutputTypeContinued().value();
        outType = outBufferType.getMemref();
        outputSwizzling = outBufferType.getTraits().getSwizzlingKey();
    } else {
        log.error("Expected either output buffer or output type for continued mode");
        return mlir::failure();
    }

    if (configureODU(log, config, outType, origInvOp.getMpeFrequentMode(), origInvOp.getNceTaskType(), outputSwizzling,
                     origInvOp.getIsSuperdense(), origInvOp.getOutputSparsityMap().has_value())
                .failed()) {
        return mlir::failure();
    }

    if (buildODUConfig(builder, origInvOp.getLoc(), log, config,
                       getInvBlockArg(BlockArg::ACT_OUT, invBlock, invBlockArgsPos),
                       getInvBlockArg(BlockArg::ACT_SPARSE_MAP_OUT, invBlock, invBlockArgsPos))
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
