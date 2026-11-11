//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_idu.hpp"
#include <cstdint>
#include <optional>
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPUIPDPU::arch60xx::IDU {

mlir::LogicalResult buildIDUNthwNtk(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                    VPU::MPEMode& mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                    bool allowExtension) {
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (dpuTaskType == VPUIP::NCETaskType::ELTWISE) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_8x16;
    }
    // EISW-170316 - Hardware only supports MPE_Mode_CUBOID_16x16 for depthwise conv, maxPool and avgPool
    else if (dpuTaskType == VPUIP::NCETaskType::DWCONV || dpuTaskType == VPUIP::NCETaskType::MAXPOOL ||
             dpuTaskType == VPUIP::NCETaskType::AVEPOOL) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_16x16;
    }

    auto activationReuse = IDUNthwNtk::NTHW_NTK_8_8;

    switch (mpeFrequentMode) {
    case VPU::MPEMode::DCIM_IC128:
    case VPU::MPEMode::CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        activationReuse = IDUNthwNtk::NTHW_NTK_4_16;
        break;
    case VPU::MPEMode::CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    case VPU::MPEMode::VECTOR:
        activationReuse = IDUNthwNtk::NTHW_NTK_8_8;
        break;
    case VPU::MPEMode::CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        activationReuse = IDUNthwNtk::NTHW_NTK_16_4;
        break;
    default:
        // For enabling code reuse from future arches, check the parameter allowExtension
        if (!allowExtension) {
            log.error("IDU NTHW mode not supported for MPE mode {0}", mpeFrequentMode);
            return mlir::failure();
        }
    }

    if (activationReuse != IDUNthwNtk::NTHW_NTK_8_8) {
        IDUNthwNtkOp::create(builder, loc, activationReuse);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUPalletLutREAD(mlir::OpBuilder& builder, VPUASM::DPUVariantOp origVarOp) {
    if (origVarOp.getPalletLutRead()) {
        IDUpalletLUTReadOp::create(builder, origVarOp.getLoc());
    }
    return mlir::success();
}

mlir::LogicalResult buildIDUWeightsTableZpBitwidth(mlir::OpBuilder& builder, const mlir::Location& loc,
                                                   const Logger& log, const vpux::NDTypeInterface& weightsType) {
    const auto weightsElementType = weightsType.getElementType();

    VPUIPDPU::WeightsZpBitwidthType zpBitwidth;
    if (auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElementType)) {
        auto storageType = quantType.getStorageType();
        if (const auto qType = mlir::dyn_cast<mlir::quant::QuantileType>(storageType)) {
            storageType = qType.getStorageType();
        }
        if (auto storageTypeAsIntegerType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
            auto numberOfBitsInZeroPoint = storageTypeAsIntegerType.getWidth();
            if (numberOfBitsInZeroPoint == 8) {
                zpBitwidth = VPUIPDPU::WeightsZpBitwidthType::ZP_8b;
            } else if (numberOfBitsInZeroPoint == 4) {
                zpBitwidth = VPUIPDPU::WeightsZpBitwidthType::ZP_4b;
            } else if (numberOfBitsInZeroPoint == 2) {
                zpBitwidth = VPUIPDPU::WeightsZpBitwidthType::ZP_4b;
            } else {
                log.error("unsupported bitwidth for ZP; expecting ZP on 8bits, 4bits or 2bits");
                return mlir::failure();
            }
        } else {
            log.error("unsupported storage type; expecting an integer type");
            return mlir::failure();
        }
    } else {
        log.error("unsupported element type in case of weights; expecting a quantized type");
        return mlir::failure();
    }

    WeightsTableZpBitwidthOp::create(builder, loc, zpBitwidth);

    return mlir::success();
}

mlir::LogicalResult buildIDUGroupConv(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, const vpux::NDTypeInterface& inActType,
                                      vpux::VPUIP::NCETaskType taskType) {
    const auto groupSize = getRangeSize(inStartZ, inEndZ);
    const auto inFullSizeZ = inActType.getShape()[Dims4D::Act::C];

    if (taskType == VPUIP::NCETaskType::CONV && groupSize < inFullSizeZ) {
        if (inStartZ % 16 == 0 && groupSize % 16 == 0) {
            IDUGroupConvOp::create(builder, loc);
        } else {
            log.error(
                    "Unsupported start&size for group conv; Both start&size must be 16 aligned. Start: {0}; Size: {1}.",
                    inStartZ, groupSize);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, int64_t outStartZ, int64_t outEndZ,
                                      std::optional<int64_t> outChannelOffset, VPUIP::NCETaskType taskType,
                                      const vpux::NDTypeInterface& inActType, const vpux::NDTypeInterface& outActType,
                                      std::optional<mlir::ArrayAttr> kernelSize, bool hasWeightTable) {
    auto inSizeZ = getRangeSize(inStartZ, inEndZ);
    auto outSizeZ = getRangeSize(outStartZ, outEndZ);
    auto inFullSizeZ = inActType.getShape()[Dims4D::Act::C];

    if (taskType == VPUIP::NCETaskType::CONV && inSizeZ < inFullSizeZ) {
        // partial IC case
        int64_t weightStart = 0;
        if (hasWeightTable) {
            auto outputZ = outActType.getShape()[Dims4D::Act::C];
            weightStart = (outStartZ - outChannelOffset.value_or(0)) % outputZ;
            weightStart <<= 4;
        }

        auto weightNum = vpux::alignValUp(outSizeZ, static_cast<int64_t>(16));
        int64_t kernelX = 1, kernelY = 1;

        if (kernelSize.has_value()) {
            auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
            kernelX = kernelSizeArray[1];
            kernelY = kernelSizeArray[0];
        }
        int64_t weightSize = kernelX * kernelY;

        if (inSizeZ < 16) {
            weightSize *= 16;
        } else {
            weightSize *= inSizeZ;
        }
        IDUWeightSetOp::create(builder, loc, weightStart, weightNum, weightSize);

        return mlir::success();
    }

    return arch50xx::IDU::buildIDUWeightSet(builder, loc, log, inStartZ, inEndZ, outStartZ, outEndZ, outChannelOffset,
                                            taskType, inActType, outActType, kernelSize, hasWeightTable);
}

mlir::LogicalResult buildIDUWeightTableOffsets(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                               const std::optional<int64_t>& weightTableOffset, bool hasWeights,
                                               bool hasWeightTableDataPtr, bool hasWeightTableSpPtr,
                                               std::optional<mlir::MemRefType> weightTableScaleType,
                                               std::optional<mlir::MemRefType> weightTableBiasType,
                                               const vpux::NDTypeInterface& outActType, int64_t outStartZ,
                                               std::optional<int64_t> outChannelOffset) {
    mlir::IntegerAttr dataPtrZpOffsetAttr, spPtrOffsetAttr, scaleOffsetAttr, biasOffsetAttr;
    constexpr uint32_t wtAddrAlignment = 32;
    auto outputZ = outActType.getShape()[Dims4D::Act::C];
    auto channelOffset = (outStartZ - outChannelOffset.value_or(0)) % outputZ;

    auto getAlignedOffset = [&](int64_t byteOffset) -> mlir::FailureOr<int64_t> {
        if (byteOffset % wtAddrAlignment) {
            log.error("Weight table offset {0} is not aligned to {1} bytes", byteOffset, wtAddrAlignment);
            return mlir::failure();
        }
        return byteOffset / wtAddrAlignment;
    };

    if (weightTableOffset) {
        auto alignedOffset = getAlignedOffset(weightTableOffset.value());
        if (mlir::failed(alignedOffset)) {
            return mlir::failure();
        }
        dataPtrZpOffsetAttr = builder.getI64IntegerAttr(alignedOffset.value());
        if (hasWeightTableSpPtr) {
            spPtrOffsetAttr = dataPtrZpOffsetAttr;
        }
    }
    if (hasWeights && !hasWeightTableDataPtr && channelOffset > 0) {
        spPtrOffsetAttr = builder.getI64IntegerAttr(channelOffset);
    }

    if (weightTableScaleType) {
        auto alignedOffset = getAlignedOffset(
                channelOffset * Bit(weightTableScaleType.value().getElementTypeBitWidth()).to<Byte>().count());
        if (mlir::failed(alignedOffset)) {
            return mlir::failure();
        }
        scaleOffsetAttr = builder.getI64IntegerAttr(alignedOffset.value());
    }
    if (weightTableBiasType) {
        auto alignedOffset = getAlignedOffset(
                channelOffset * Bit(weightTableBiasType.value().getElementTypeBitWidth()).to<Byte>().count());
        if (mlir::failed(alignedOffset)) {
            return mlir::failure();
        }
        biasOffsetAttr = builder.getI64IntegerAttr(alignedOffset.value());
    }

    if (dataPtrZpOffsetAttr || spPtrOffsetAttr || scaleOffsetAttr || biasOffsetAttr) {
        IDUWeightTableOffsetsOp::create(builder, loc, dataPtrZpOffsetAttr, spPtrOffsetAttr, scaleOffsetAttr,
                                        biasOffsetAttr);
    }

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch60xx::IDU

using namespace vpux::VPUIPDPU::arch60xx::IDU;

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUVariantIDU(VPUASM::DPUVariantOp origVarOp,
                                                                 mlir::OpBuilder& builder, const Logger& log,
                                                                 ELF::SymbolReferenceMap& symRefMap,
                                                                 bool dpuSwizzlingEnabled) {
    auto origInvOp = mlir::cast<VPUASM::DPUInvariantOp>(symRefMap.lookupSymbol(origVarOp.getInvariant()));

    mlir::Operation* inAct = nullptr;
    if (origInvOp.getInput()) {
        inAct = symRefMap.lookupSymbol(origInvOp.getInput().value());
    }

    std::optional<int64_t> inSwizzlingKey;

    mlir::MemRefType outType;
    if (!origInvOp.getIsContinued() && origInvOp.getOutput()) {
        auto outBuffer = symRefMap.lookupSymbol(origInvOp.getOutput().value());
        outType = getBufferType(outBuffer);
    } else if (origInvOp.getIsContinued() && origInvOp.getOutputTypeContinued()) {
        auto outBufferType = origInvOp.getOutputTypeContinued().value();
        outType = outBufferType.getMemref();
    } else {
        outType = VPUASM::getMinMaxDataType(origInvOp, symRefMap);
    }

    if (!outType) {
        log.error("Expected either output buffer, output type for continued mode, or min/max output for reduce ops");
        return mlir::failure();
    }

    mlir::Type weightsType;
    std::optional<int64_t> weightsSwizzlingKey;
    if (origInvOp.getWeights()) {
        auto weights = symRefMap.lookupSymbol(origInvOp.getWeights().value());
        weightsType = getBufferType(weights);
        weightsSwizzlingKey = getSwizzlingKey(weights);
    }

    // arch40xx functions
    // IDUWorkloadSet
    if (arch40xx::IDU::buildIDUWorkloadSet(builder, origVarOp.getLoc(),
                                           parseIntArrayAttr<int64_t>(origVarOp.getInStart()),
                                           parseIntArrayAttr<int64_t>(origVarOp.getInEnd()))
                .failed()) {
        return mlir::failure();
    }

    auto outStartZ = parseIntArrayAttr<int64_t>(origVarOp.getStart())[2];
    if (inAct) {
        inSwizzlingKey = getSwizzlingKey(inAct);
        auto inStartZ = parseIntArrayAttr<int64_t>(origVarOp.getInStart())[2];
        auto inEndZ = parseIntArrayAttr<int64_t>(origVarOp.getInEnd())[2];
        auto outEndZ = parseIntArrayAttr<int64_t>(origVarOp.getEnd())[2];
        auto inActType = getBufferType(inAct);

        // IDUWeightSet
        if (arch60xx::IDU::buildIDUWeightSet(builder, origVarOp.getLoc(), log, inStartZ, inEndZ, outStartZ, outEndZ,
                                             origInvOp.getOutChannelOffset(), origInvOp.getNceTaskType(), inActType,
                                             outType, origInvOp.getKernelSize(), origVarOp.getWeightTable().has_value())
                    .failed()) {
            return mlir::failure();
        }

        // IDUGroupConv
        if (buildIDUGroupConv(builder, origVarOp.getLoc(), log, inStartZ, inEndZ, inActType, origInvOp.getNceTaskType())
                    .failed()) {
            return mlir::failure();
        }
    }

    // IDUPadding
    if (arch40xx::IDU::buildIDUPadding(builder, origVarOp.getLoc(), log, origVarOp.getPad()).failed()) {
        return mlir::failure();
    }

    if (dpuSwizzlingEnabled) {
        // IDUActSwizzle
        if (arch40xx::IDU::buildIDUActSwizzle(builder, origVarOp.getLoc(), log, inSwizzlingKey).failed()) {
            return mlir::failure();
        }

        // IDUWeightSwizzle
        if (arch40xx::IDU::buildIDUWeightSwizzle(builder, origVarOp.getLoc(), log, weightsSwizzlingKey).failed()) {
            return mlir::failure();
        }
    }

    // arch40xx function - rewritten for arch60xx to add DCIM case
    // IDUNthwNtk
    auto mpeFreqMode = origInvOp.getMpeFrequentMode();
    if (buildIDUNthwNtk(builder, origVarOp.getLoc(), log, mpeFreqMode, origInvOp.getNceTaskType()).failed()) {
        return mlir::failure();
    }

    // IDUSEDense
    if (arch40xx::IDU::buildIDUSEDense(builder, origVarOp.getLoc(), log,
                                       !origInvOp.getInputStorageElementTable().has_value())
                .failed()) {
        return mlir::failure();
    }

    // IDUConvContinue
    if (arch40xx::IDU::buildIDUConvContinue(builder, origVarOp.getLoc(), log, origInvOp.getIsContinued()).failed()) {
        return mlir::failure();
    }

    // IDUBinaryConfig
    // variant.offset_addr.offset_addr_bf.bin_cfg not set in nce_lib
    // leaving field value to reset initialization in lowering VPUASM2NPUReg40XX

    // arch50xx functions
    // IDUSEOnly
    if (arch50xx::IDU::buildIDUSEOnly(
                builder, origVarOp.getLoc(), log,
                origInvOp.getInputStorageElementTable().has_value() && !origInvOp.getInputSparsityMap().has_value())
                .failed()) {
        return mlir::failure();
    }

    // IDUPerOutputChannelScaling
    if (arch50xx::IDU::buildIDUPerOutputChannelScaling(
                builder, origVarOp.getLoc(), log, origInvOp.getNceTaskType(),
                origVarOp.getWeightTable().has_value() || origVarOp.getWeightTableScale().has_value(),
                origInvOp.getWeightsSparsityMap().has_value())
                .failed()) {
        return mlir::failure();
    }

    // IDUPalletLutRead
    if (buildIDUPalletLutREAD(builder, origVarOp).failed()) {
        return mlir::failure();
    }

    // IDUWeightsTableZpBitwidth
    if (origVarOp.getWeightZeroPoints()) {
        if (buildIDUWeightsTableZpBitwidth(builder, origVarOp.getLoc(), log, weightsType).failed()) {
            return mlir::failure();
        }
    }
    auto isMaxPoolWithHaloAndWeightTable = origVarOp.getNceTaskType() == VPUIP::NCETaskType::MAXPOOL &&
                                           origInvOp.getVariantCount() > 1 && origVarOp.getHaloRegions().has_value() &&
                                           origVarOp.getWeightTableScale().has_value() &&
                                           origVarOp.getWeightTableBias().has_value();

    // IDUWeightTableOffsets
    if (isMaxPoolWithHaloAndWeightTable || !origVarOp.getHaloRegions().has_value() ||
        (origVarOp.getWeightTableDataPtr() && origInvOp.getVariantCount() > 1)) {
        std::optional<mlir::MemRefType> weightTableScaleType;
        if (auto weightTableScale = origVarOp.getWeightTableScale()) {
            weightTableScaleType = getBufferType(symRefMap.lookupSymbol(weightTableScale.value()));
        }
        std::optional<mlir::MemRefType> weightTableBiasType;
        if (auto weightTableBias = origVarOp.getWeightTableBias()) {
            weightTableBiasType = getBufferType(symRefMap.lookupSymbol(weightTableBias.value()));
        };
        if (buildIDUWeightTableOffsets(builder, origVarOp.getLoc(), log, origVarOp.getWeightTableOffset(),
                                       origVarOp.getWeights().has_value(),
                                       origVarOp.getWeightTableDataPtr().has_value(),
                                       origVarOp.getWeightTableSpPtr().has_value(), weightTableScaleType,
                                       weightTableBiasType, outType, outStartZ, origInvOp.getOutChannelOffset())
                    .failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}
