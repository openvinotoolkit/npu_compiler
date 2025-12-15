//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_idu.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPUIPDPU::arch40xx::IDU {

mlir::LogicalResult buildIDUWorkloadSet(mlir::OpBuilder& builder, const mlir::Location& loc,
                                        const SmallVector<int64_t>&& inStart, const SmallVector<int64_t>&& inEnd) {
    builder.create<IDUWorkloadSetOp>(loc, inStart[0], inStart[1], inStart[2], getRangeSize(inStart[0], inEnd[0]),
                                     getRangeSize(inStart[1], inEnd[1]), getRangeSize(inStart[2], inEnd[2]));

    return mlir::success();
}

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& /*log*/,
                                      int64_t inStartZ, int64_t inEndZ, int64_t outStartZ, int64_t outEndZ,
                                      std::optional<int64_t> outChannelOffset, VPUIP::NCETaskType taskType,
                                      const vpux::NDTypeInterface& inActType, const vpux::NDTypeInterface& outActType,
                                      std::optional<mlir::ArrayAttr> kernelSize) {
    // weight_start is updated during run-time relocation by addition with weight_table address
    auto outputZ = outActType.getShape()[Dims4D::Act::C];
    auto weightStart = (outStartZ - outChannelOffset.value_or(0)) % outputZ;
    weightStart <<= 4;

    auto inputZ = inActType.getShape()[Dims4D::Act::C];
    auto inSizeZ = getRangeSize(inStartZ, inEndZ);
    auto outSizeZ = getRangeSize(outStartZ, outEndZ);
    auto weightNum = vpux::alignValUp(outSizeZ, static_cast<int64_t>(16));
    int64_t weightSize = 0;

    int64_t kernelX = 1, kernelY = 1;
    if (kernelSize.has_value()) {
        auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
        kernelX = kernelSizeArray[1];
        kernelY = kernelSizeArray[0];
    }

    switch (taskType) {
    case VPUIP::NCETaskType::CONV: {
        weightSize = kernelX * kernelY;
        if (inActType.getShape()[Dims4D::Act::C] < 16) {
            weightSize *= 16;
        } else {
            weightSize *= inputZ;
        }
    } break;
    case VPUIP::NCETaskType::DWCONV:
    case VPUIP::NCETaskType::AVEPOOL:
    case VPUIP::NCETaskType::MAXPOOL:
        weightSize = kernelX * kernelY * inSizeZ;
        break;
    case VPUIP::NCETaskType::ELTWISE: {
        auto inputX = inActType.getShape()[Dims4D::Act::W], inputY = inActType.getShape()[Dims4D::Act::H];
        weightSize = inputX * inputY * inputZ;
    } break;
    default:
        break;
    }

    builder.create<IDUWeightSetOp>(loc, weightStart, weightNum, weightSize);

    return mlir::success();
}

mlir::LogicalResult buildIDUPadding(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                    VPU::PaddingAttr pad) {
    builder.create<IDUPaddingOp>(loc, pad);

    return mlir::success();
}

mlir::LogicalResult buildIDUActSwizzle(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                       std::optional<int64_t> inSwizzling) {
    if (inSwizzling.has_value()) {
        auto swizzleKey = symbolizeDPUSwizzleKey(inSwizzling.value());
        if (!swizzleKey.has_value()) {
            log.error("Invalid input swizzle key '{0}'", inSwizzling.value());
            return mlir::failure();
        }
        if (swizzleKey.value() != DPUSwizzleKey::SWIZZLE_OFF) {
            builder.create<IDUActSwizzleOp>(loc, swizzleKey.value());
        }
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUWeightSwizzle(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                          std::optional<int64_t> weightsSwizzling) {
    if (weightsSwizzling.has_value()) {
        auto swizzleKey = symbolizeDPUSwizzleKey(weightsSwizzling.value());
        if (!swizzleKey.has_value()) {
            log.error("Invalid weights swizzle key '{0}'", weightsSwizzling.value());
            return mlir::failure();
        }
        if (swizzleKey.value() != DPUSwizzleKey::SWIZZLE_OFF) {
            builder.create<IDUWeightSwizzleOp>(loc, swizzleKey.value());
        }
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUNthwNtk(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                    VPU::MPEMode mpeFrequentMode, VPUIP::NCETaskType dpuTaskType) {
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (dpuTaskType == VPUIP::NCETaskType::ELTWISE) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_8x16;
    }

    auto activationReuse = IDUNthwNtk::NTHW_NTK_8_8;

    switch (mpeFrequentMode) {
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
        log.error("ODU NTHW mode not supported for MPE mode {0}", mpeFrequentMode);
        return mlir::failure();
    }

    if (activationReuse != IDUNthwNtk::NTHW_NTK_8_8) {
        builder.create<IDUNthwNtkOp>(loc, activationReuse);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUSEDense(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&, bool seDense) {
    if (seDense) {
        builder.create<IDUSEDenseOp>(loc);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUConvContinue(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                         std::optional<bool> isContinued) {
    if (isContinued.value_or(false)) {
        builder.create<IDUConvContinueOp>(loc);
    }

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch40xx::IDU

using namespace vpux::VPUIPDPU::arch40xx::IDU;

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUVariantIDU(VPUASM::DPUVariantOp origVarOp,
                                                                 mlir::OpBuilder& builder, const Logger& log,
                                                                 ELF::SymbolReferenceMap& symRefMap) {
    auto origInvOp = mlir::cast<VPUASM::DPUInvariantOp>(symRefMap.lookupSymbol(origVarOp.getInvariant()));

    mlir::Operation* inAct = nullptr;
    if (origInvOp.getInput()) {
        inAct = symRefMap.lookupSymbol(origInvOp.getInput().value());
    }

    std::optional<int64_t> inSwizzlingKey;

    mlir::MemRefType outActType;
    if (!origInvOp.getIsContinued() && origInvOp.getOutput()) {
        auto outBuffer = symRefMap.lookupSymbol(origInvOp.getOutput().value());
        outActType = getBufferType(outBuffer);
    } else if (origInvOp.getIsContinued() && origInvOp.getOutputTypeContinued()) {
        auto outBufferType = origInvOp.getOutputTypeContinued().value();
        outActType = outBufferType.getMemref();
    } else {
        log.error("Expected either output buffer or output type for continued mode");
        return mlir::failure();
    }

    std::optional<int64_t> weightsSwizzlingKey;
    if (origInvOp.getWeights()) {
        auto weights = symRefMap.lookupSymbol(origInvOp.getWeights().value());
        weightsSwizzlingKey = getSwizzlingKey(weights);
    }

    // IDUWorkloadSet
    if (buildIDUWorkloadSet(builder, origVarOp.getLoc(), parseIntArrayAttr<int64_t>(origVarOp.getInStart()),
                            parseIntArrayAttr<int64_t>(origVarOp.getInEnd()))
                .failed()) {
        return mlir::failure();
    }

    // IDUWeightSet
    if (inAct) {
        inSwizzlingKey = getSwizzlingKey(inAct);
        auto inStartZ = parseIntArrayAttr<int64_t>(origVarOp.getInStart())[2];
        auto inEndZ = parseIntArrayAttr<int64_t>(origVarOp.getInEnd())[2];
        auto outStartZ = parseIntArrayAttr<int64_t>(origVarOp.getStart())[2];
        auto outEndZ = parseIntArrayAttr<int64_t>(origVarOp.getEnd())[2];
        auto inActType = getBufferType(inAct);
        if (buildIDUWeightSet(builder, origVarOp.getLoc(), log, inStartZ, inEndZ, outStartZ, outEndZ,
                              origInvOp.getOutChannelOffset(), origInvOp.getNceTaskType(), inActType, outActType,
                              origInvOp.getKernelSize())
                    .failed()) {
            return mlir::failure();
        }
    }

    // IDUPadding
    if (buildIDUPadding(builder, origVarOp.getLoc(), log, origVarOp.getPad()).failed()) {
        return mlir::failure();
    }

    // IDUActSwizzle
    if (buildIDUActSwizzle(builder, origVarOp.getLoc(), log, inSwizzlingKey).failed()) {
        return mlir::failure();
    }

    // IDUWeightSwizzle
    if (buildIDUWeightSwizzle(builder, origVarOp.getLoc(), log, weightsSwizzlingKey).failed()) {
        return mlir::failure();
    }

    // IDUNthwNtk
    if (buildIDUNthwNtk(builder, origVarOp.getLoc(), log, origInvOp.getMpeFrequentMode(), origInvOp.getNceTaskType())
                .failed()) {
        return mlir::failure();
    }

    // IDUSEDense
    if (buildIDUSEDense(builder, origVarOp.getLoc(), log, !origInvOp.getInputStorageElementTable().has_value())
                .failed()) {
        return mlir::failure();
    }

    // IDUConvContinue
    if (buildIDUConvContinue(builder, origVarOp.getLoc(), log, origInvOp.getIsContinued()).failed()) {
        return mlir::failure();
    }

    // IDUBinaryConfig
    // variant.offset_addr.offset_addr_bf.bin_cfg not set in nce_lib
    // leaving field value to reset initialization in lowering VPUASM2NPUReg40XX

    return mlir::success();
}
