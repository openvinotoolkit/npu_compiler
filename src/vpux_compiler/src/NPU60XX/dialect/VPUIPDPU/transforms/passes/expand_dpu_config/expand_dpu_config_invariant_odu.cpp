//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_odu.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;
using namespace VPUIPDPU;

namespace vpux::VPUIPDPU::arch60xx::ODU {
int64_t getTileSelectMaskForReduceTotal(ELF::SymbolReferenceMap& symRefMap,
                                        std::optional<mlir::ArrayAttr> minMaxArrayAttr,
                                        const NDTypeInterface& outType) {
    int64_t minMaxTileSelect = 0;
    if (minMaxArrayAttr.has_value()) {
        // TODO: E#-95417
        // refactor symbolRefAttr search and cast after solution is implemented: either symRef contains proper
        // information or buffer type is replaced; due to slowness of symmbolRef search
        for (auto minMaxAttr : minMaxArrayAttr.value()) {
            auto minMaxAttrSymRef = mlir::cast<mlir::SymbolRefAttr>(minMaxAttr);
            auto minMaxBufferRef = symRefMap.lookupSymbol(minMaxAttrSymRef);
            if (auto minMaxDeclareBufffer = mlir::dyn_cast_if_present<VPUASM::DeclareBufferOp>(minMaxBufferRef)) {
                auto tileIndex = minMaxDeclareBufffer.getBufferType().getLocation().getSectionIndex();
                minMaxTileSelect |= 1LL << tileIndex;
            }
        }

        auto outputTileIdx = outType.getMemSpace().getIndex();
        VPUX_THROW_UNLESS(outputTileIdx.has_value(), "Missing output buffer tile for ODU reduce total operation.");

        // Mask out the tile the reduce op is done on, as the tile can't replicate ITI traffic to itself, only to other
        // tiles.
        auto outputTileMask = 1LL << outputTileIdx.value();
        minMaxTileSelect = minMaxTileSelect & (~(outputTileMask));
    }
    return minMaxTileSelect;
}

mlir::LogicalResult configureReduceOutputLayer(const Logger&, VPUIPDPU::arch60xx::ODU::ODUConfig::ReduceLayer& config,
                                               bool doReduceMaxPerXy, bool doReduceMinPerXy,
                                               bool doReduceMinMaxPerTensor, int64_t tileSelect,
                                               const NDTypeInterface& reduceDataType) {
    if (doReduceMaxPerXy || doReduceMinPerXy || doReduceMinMaxPerTensor) {
        config.reduceMax = doReduceMaxPerXy;
        config.reduceMin = doReduceMinPerXy;
        config.reduceTotal = doReduceMinMaxPerTensor;

        config.reduceDataType = getOutReduceType(reduceDataType.getElementType());
        if (tileSelect) {
            config.tileSelect = tileSelect;
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureDataReuse(const Logger& log, ODUConfig::DataReuse& config, VPU::MPEMode& mpeFrequentMode,
                                       VPUIP::NCETaskType dpuTaskType, bool enableStencilSelection,
                                       bool allowExtension) {
    if (!enableStencilSelection) {
        return mlir::success();
    }
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (dpuTaskType == VPUIP::NCETaskType::ELTWISE) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_8x16;
    }
    // E#170316 - Hardware only supports MPE_Mode_CUBOID_16x16 for depthwise conv, max_pool and avg_pool
    else if (dpuTaskType == VPUIP::NCETaskType::DWCONV || dpuTaskType == VPUIP::NCETaskType::MAXPOOL ||
             dpuTaskType == VPUIP::NCETaskType::AVEPOOL) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_16x16;
    }

    switch (mpeFrequentMode) {
    case VPU::MPEMode::CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        config.activationReuse = ODUActivationReuseMode::NTHW_4;
        break;
    case VPU::MPEMode::CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    case VPU::MPEMode::VECTOR:
        config.activationReuse = ODUActivationReuseMode::NTHW_8;
        break;
    case VPU::MPEMode::CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        config.activationReuse = ODUActivationReuseMode::NTHW_16;
        break;
    case VPU::MPEMode::DCIM_IC128:
        config.activationReuse = ODUActivationReuseMode::NTHW_CIM;
        break;
    default:
        // For enabling code reuse from future arches, check the parameter allowExtension
        if (!allowExtension) {
            log.error("ODU NTHW mode not supported for MPE mode {0}", mpeFrequentMode);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureLocalRegion(const Logger&, ODUConfig::LocalRegion& config,
                                         std::optional<vpux::VPUIP::LocalRegionAttr> localRegionAttr) {
    if (localRegionAttr.has_value()) {
        const auto& localRegion = localRegionAttr.value();
        config.localRegionEnable = true;

        // no need to set begin/end fields if full output suppression is enabled
        if (auto suppress = localRegion.getSuppressOutput()) {
            config.suppressOutput = true;
            return mlir::success();
        }
        config.beginX = localRegion.getXStart().getInt();
        config.beginY = localRegion.getYStart().getInt();
        config.endX = localRegion.getXEnd().getInt();
        config.endY = localRegion.getYEnd().getInt();
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

    config.dataWidth = outDataWidth.value();

    return mlir::success();
}

mlir::LogicalResult buildODUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const ODUConfig& config, mlir::Value outAct, mlir::Value outSparsityMap) {
    if (arch40xx::ODU::buildODUConfig(builder, loc, log, config, outAct, outSparsityMap).failed()) {
        return mlir::failure();
    }

    if (config.reduceLayer.reduceMax || config.reduceLayer.reduceMin || config.reduceLayer.reduceTotal) {
        mlir::IntegerAttr tileSelectAttr = nullptr;

        if (config.reduceLayer.tileSelect.has_value()) {
            tileSelectAttr = builder.getI64IntegerAttr(config.reduceLayer.tileSelect.value());
        }

        ODUReduceLayerOp::create(builder, loc, config.reduceLayer.reduceMax, config.reduceLayer.reduceMin,
                                 config.reduceLayer.reduceTotal, config.reduceLayer.reduceDataType, tileSelectAttr);
    }

    if (config.localRegion.localRegionEnable) {
        ODULocalRegionOp::create(builder, loc, config.localRegion.beginX, config.localRegion.beginY,
                                 config.localRegion.endX, config.localRegion.endY, config.localRegion.suppressOutput);
    }

    return mlir::success();
}

mlir::LogicalResult configureODU(const Logger& log, ODUConfig& config, const NDTypeInterface& outActType,
                                 VPU::MPEMode& mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                 std::optional<int64_t> outSwizzling, std::optional<bool> isSuperdense,
                                 bool outSparsityEnabled, int64_t minMaxTileSelect, bool doReduceMaxPerXy,
                                 bool doReduceMinPerXy, bool doReduceMinMaxPerTensor,
                                 const NDTypeInterface& minMaxDataType,
                                 std::optional<vpux::VPUIP::LocalRegionAttr> localRegionAttr,
                                 bool enableStencilSelection, bool allowExtension) {
    // If output activation type is not present, use min/max data type for configuration
    auto outType = outActType ? outActType : minMaxDataType;

    // arch40xx functions
    if (arch40xx::ODU::configurePermuteMode(log, config.permuteData, outType.getDimsOrder()).failed()) {
        return mlir::failure();
    }

    if (arch40xx::ODU::configureOutTensorSize(log, config.outTensorSize, config.permuteData.permuteMode,
                                              outType.getStrides())
                .failed()) {
        return mlir::failure();
    }

    int64_t zeroPointVal = 0;
    if (outSparsityEnabled) {
        zeroPointVal = VPUIPDPU::getZeroPoint(outType);
    }
    if (arch40xx::ODU::configureSparsity(log, config.sparsity, outSparsityEnabled, zeroPointVal).failed()) {
        return mlir::failure();
    }

    if (arch40xx::ODU::configureSwizzleData(log, config.swizzleData, outSwizzling).failed()) {
        return mlir::failure();
    }

    if (arch40xx::ODU::configureMemoryMode(log, config.memoryMode, isSuperdense).failed()) {
        return mlir::failure();
    }

    // arch40xx function - rewritten for arch60xx to add DCIM case
    if (arch60xx::ODU::configureDataReuse(log, config.dataReuse, mpeFrequentMode, dpuTaskType, enableStencilSelection,
                                          allowExtension)
                .failed()) {
        return mlir::failure();
    }

    // arch60xx functions
    if (arch60xx::ODU::configureReduceOutputLayer(log, config.reduceLayer, doReduceMaxPerXy, doReduceMinPerXy,
                                                  doReduceMinMaxPerTensor, minMaxTileSelect, minMaxDataType)
                .failed()) {
        return mlir::failure();
    }

    if (arch60xx::ODU::configureLocalRegion(log, config.localRegion, localRegionAttr).failed()) {
        return mlir::failure();
    }

    if (arch60xx::ODU::configureOutActivations(log, config.outActivations, outType.getElementType()).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch60xx::ODU

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUInvariantODU(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos, ELF::SymbolReferenceMap& symRefMap) {
    vpux::VPUIPDPU::arch60xx::ODU::ODUConfig config;

    mlir::MemRefType outActType;
    uint64_t outputSwizzling = 0;
    if (!origInvOp.getIsContinued() && origInvOp.getOutput()) {
        auto outBuffer = symRefMap.lookupSymbol(origInvOp.getOutput().value());
        outActType = getBufferType(outBuffer);
        outputSwizzling = getSwizzlingKey(outBuffer);
    } else if (origInvOp.getIsContinued() && origInvOp.getOutputTypeContinued()) {
        auto outBufferType = origInvOp.getOutputTypeContinued().value();
        outActType = outBufferType.getMemref();
        outputSwizzling = outBufferType.getTraits().getSwizzlingKey();
    }

    mlir::MemRefType minMaxDataType = VPUASM::getMinMaxDataType(origInvOp, symRefMap);

    if (!outActType && !minMaxDataType) {
        log.error("Expected either output buffer, output type for continued mode, or min/max output for reduce ops");
        return mlir::failure();
    }

    auto minMaxTileSelect = vpux::VPUIPDPU::arch60xx::ODU::getTileSelectMaskForReduceTotal(
            symRefMap, origInvOp.getMinMaxPerTensor(), minMaxDataType);

    auto mpeFreqMode = origInvOp.getMpeFrequentMode();
    if (configureODU(log, config, outActType, mpeFreqMode, origInvOp.getNceTaskType(), outputSwizzling,
                     origInvOp.getIsSuperdense(), origInvOp.getOutputSparsityMap().has_value(), minMaxTileSelect,
                     origInvOp.getMaxPerXy().has_value(), origInvOp.getMinPerXy().has_value(),
                     origInvOp.getMinMaxPerTensor().has_value(), minMaxDataType, origInvOp.getLocalRegion())
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
