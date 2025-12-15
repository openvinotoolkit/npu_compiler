//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant_idu.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPUIPDPU::arch50xx::IDU {

mlir::LogicalResult buildIDUSEOnly(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&, bool seOnly) {
    if (seOnly) {
        builder.create<IDUSEOnlyOp>(loc);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUPerOutputChannelScaling(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                                    VPUIP::NCETaskType taskType, bool weightTableProvided,
                                                    bool weightsSparse) {
    // HW constraint: sb_read_en is not supported when dw_opt_en is set (currently never set in
    // dpu_invariant_idu_rewriter)

    // weight table provided means per output channel QDQ
    if (weightTableProvided && (taskType == VPUIP::NCETaskType::MAXPOOL || taskType == VPUIP::NCETaskType::AVEPOOL ||
                                taskType == VPUIP::NCETaskType::ELTWISE)) {
        const auto tensor2ActSparse = (taskType == VPUIP::NCETaskType::ELTWISE && weightsSparse);
        builder.create<IDUPerOutputChannelScalingOp>(loc, tensor2ActSparse);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t inStartZ, int64_t inEndZ, int64_t outStartZ, int64_t outEndZ,
                                      std::optional<int64_t> outChannelOffset, VPUIP::NCETaskType taskType,
                                      const vpux::NDTypeInterface& inActType, const vpux::NDTypeInterface& outActType,
                                      std::optional<mlir::ArrayAttr> kernelSize) {
    if (taskType == VPUIP::NCETaskType::REDUCEMEAN || taskType == VPUIP::NCETaskType::REDUCESUMSQUARE ||
        taskType == VPUIP::NCETaskType::REDUCESUM) {
        // weight_start is updated during run-time relocation by addition with weight_table address
        auto outputZ = outActType.getShape()[Dims4D::Act::C];
        auto weightStart = (outStartZ - outChannelOffset.value_or(0)) % outputZ;
        weightStart <<= 4;

        auto inputZ = inActType.getShape()[Dims4D::Act::C];
        auto outSizeZ = getRangeSize(outStartZ, outEndZ);
        auto weightNum = vpux::alignValUp(outSizeZ, static_cast<int64_t>(16));

        int64_t kernelX = 1, kernelY = 1;
        if (kernelSize.has_value()) {
            auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
            kernelX = kernelSizeArray[1];
            kernelY = kernelSizeArray[0];
        }

        auto weightSize = kernelX * kernelY;
        if (inActType.getShape()[Dims4D::Act::C] < 16) {
            weightSize *= 16;
        } else {
            weightSize *= inputZ;
        }

        builder.create<IDUWeightSetOp>(loc, weightStart, weightNum, weightSize);
    } else {
        if (arch40xx::IDU::buildIDUWeightSet(builder, loc, log, inStartZ, inEndZ, outStartZ, outEndZ, outChannelOffset,
                                             taskType, inActType, outActType, kernelSize)
                    .failed()) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch50xx::IDU

using namespace vpux::VPUIPDPU::arch50xx::IDU;

mlir::LogicalResult vpux::VPUIPDPU::arch50xx::buildDPUVariantIDU(VPUASM::DPUVariantOp origVarOp,
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
    if (arch40xx::IDU::buildIDUWorkloadSet(builder, origVarOp.getLoc(),
                                           parseIntArrayAttr<int64_t>(origVarOp.getInStart()),
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
    if (arch40xx::IDU::buildIDUPadding(builder, origVarOp.getLoc(), log, origVarOp.getPad()).failed()) {
        return mlir::failure();
    }

    // IDUActSwizzle
    if (arch40xx::IDU::buildIDUActSwizzle(builder, origVarOp.getLoc(), log, inSwizzlingKey).failed()) {
        return mlir::failure();
    }

    // IDUWeightSwizzle
    if (arch40xx::IDU::buildIDUWeightSwizzle(builder, origVarOp.getLoc(), log, weightsSwizzlingKey).failed()) {
        return mlir::failure();
    }

    // IDUNthwNtk
    if (arch40xx::IDU::buildIDUNthwNtk(builder, origVarOp.getLoc(), log, origInvOp.getMpeFrequentMode(),
                                       origInvOp.getNceTaskType())
                .failed()) {
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

    // IDUSEOnly
    if (buildIDUSEOnly(
                builder, origVarOp.getLoc(), log,
                origInvOp.getInputStorageElementTable().has_value() && !origInvOp.getInputSparsityMap().has_value())
                .failed()) {
        return mlir::failure();
    }

    // IDUPerOutputChannelScaling
    if (buildIDUPerOutputChannelScaling(
                builder, origVarOp.getLoc(), log, origInvOp.getNceTaskType(),
                origVarOp.getWeightTable().has_value() || origVarOp.getWeightTableScale().has_value(),
                origInvOp.getWeightsSparsityMap().has_value())
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
