//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_ppe.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_ppe.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace VPUIPDPU;

mlir::LogicalResult VPUIPDPU::arch50xx::PPE::configurePPE(VPUIPDPU::arch50xx::PPE::PPEConfig& config,
                                                          mlir::Type outDataType,
                                                          const vpux::NDTypeInterface& inActType,
                                                          VPUIP::NCETaskType dpuTaskType,
                                                          const arch50xx::PPE::PPETask& ppeTask,
                                                          bool isWeightTableProvided, bool isSprLookUpTableProvided) {
    // bias add & mult
    if (!isWeightTableProvided) {
        config.biasAdd.biasStatic = 0.0f;
        if (dpuTaskType == VPUIP::NCETaskType::ELTWISE || dpuTaskType == VPUIP::NCETaskType::AVEPOOL ||
            dpuTaskType == VPUIP::NCETaskType::REDUCEMEAN) {
            config.scaleMult.scaleStatic = 1.0f;
            if (ppeTask.fpScaleData.has_value()) {
                config.scaleMult.scaleStatic = ppeTask.fpScaleData;
            }
        } else if (dpuTaskType == VPUIP::NCETaskType::MAXPOOL) {
            config.scaleMult.scaleStatic = 1.0f;
            //
            // TODO: E#-156827 - fix ReduceSumSquare numericsbench operator; hw tests are currently failing with
            // config.scaleMult.scaleStatic = 1.0f for ReduceSumSquare
            //
        } else if (dpuTaskType == VPUIP::NCETaskType::REDUCESUMSQUARE) {
            config.scaleMult.scaleStatic = 1.0f / inActType.getShape()[Dims4D::Act::C];
        } else if (dpuTaskType == VPUIP::NCETaskType::REDUCESUM) {
            config.scaleMult.scaleStatic = 1.0f;
        }
        if (ppeTask.fpBias.has_value()) {
            config.biasAdd.biasStatic = ppeTask.fpBias.value();
        }
    }

    if (dpuTaskType == VPUIP::NCETaskType::CONV || dpuTaskType == VPUIP::NCETaskType::DWCONV) {
        auto ppeScale = ppeTask.fpScaleData.has_value();
        auto ppeBias = ppeTask.fpBias.has_value();

        if (ppeScale || ppeBias) {
            config.biasAdd.biasStatic = 0.0f;
            config.scaleMult.scaleStatic = 1.0f;

            if (ppeScale) {
                config.scaleMult.scaleStatic = ppeTask.fpScaleData;
            }
            if (ppeBias) {
                config.biasAdd.biasStatic = ppeTask.fpBias;
            }
        }
    }

    // spr lookup table
    if (isSprLookUpTableProvided) {
        config.sprLUT.enableLookUpTable = PPEsprLUTMode::ON;
    }

    // relu mult
    config.preluMult.preluAlpha = ppeTask.fpPreluAlpha.front();

    // clamp
    config.clamp.clampHigh = ppeTask.fixedFunction.fpClampHigh;
    config.clamp.clampLow = ppeTask.fixedFunction.fpClampLow;

    // convert
    config.convert.convertMode = PPEFpConvertMode::NONE;
    if (mlir::isa<mlir::quant::QuantizedType>(outDataType)) {
        auto quantType = mlir::cast<mlir::quant::QuantizedType>(outDataType);
        if (mlir::isa<mlir::Float8E4M3FNType>(quantType.getStorageType())) {
            config.convert.convertMode = PPEFpConvertMode::HF8;
        } else if (mlir::isa<mlir::Float8E5M2Type>(quantType.getStorageType())) {
            config.convert.convertMode = PPEFpConvertMode::BF8;
        } else {
            config.convert.convertMode = PPEFpConvertMode::I32;
        }
    } else if (mlir::isa<mlir::Float16Type>(outDataType)) {
        config.convert.convertMode = PPEFpConvertMode::FP16;
        config.convert.clampMode = PPEFpConvClampMode::ON;
    } else if (mlir::isa<mlir::BFloat16Type>(outDataType)) {
        config.convert.convertMode = PPEFpConvertMode::BF16;
        config.convert.bf16RoundMode = PPEFpConvBf16RoundMode::RNE;
    } else if (getIOType(outDataType) == IOType::INT) {
        config.convert.convertMode = PPEFpConvertMode::I32;
    }

    config.zeroPointOffset.zeroPointStatic = ppeTask.fpAdder;

    return mlir::success();
}

mlir::LogicalResult VPUIPDPU::arch50xx::PPE::buildPPEConfig(mlir::OpBuilder& builder, const mlir::Location& loc,
                                                            const Logger& log, const PPEConfig& config,
                                                            mlir::Value weightsTable) {
    // PPEFpBiasAdd
    auto biasStaticAttr = getF32FloatAttrOrNull(builder, config.biasAdd.biasStatic);
    if (biasStaticAttr) {
        builder.create<PPEFpBiasAddOp>(loc, nullptr, biasStaticAttr);
    } else {
        if (!weightsTable) {
            log.error("Expected weights_table operand in PPE FP pipeline");
            return mlir::failure();
        }
        builder.create<PPEFpBiasAddOp>(loc, weightsTable, nullptr);
    }

    // PPEFpScaleMult
    auto scaleStaticAttr = getF32FloatAttrOrNull(builder, config.scaleMult.scaleStatic);
    if (scaleStaticAttr) {
        builder.create<PPEFpScaleMultOp>(loc, nullptr, scaleStaticAttr);
    } else {
        if (!weightsTable) {
            log.error("Expected weights_table operand in PPE FP pipeline");
            return mlir::failure();
        }
        builder.create<PPEFpScaleMultOp>(loc, weightsTable, nullptr);
    }

    // PPEFpAddMultBypass
    builder.create<VPUIPDPU::PPEFpAddMultBypassOp>(loc, VPUIPDPU::PPEBypassMode::OFF);

    // PPEFpSprLUT
    builder.create<PPEFpSprLUTModeOp>(loc, config.sprLUT.enableLookUpTable);

    // PPEFpPreluMult
    auto preluMultAttr = getF32FloatAttrOrNull(builder, config.preluMult.preluAlpha);
    builder.create<PPEFpPreluMultOp>(loc, preluMultAttr);

    // PPEFpClamps
    auto clampLowAttr = getF32FloatAttrOrNull(builder, config.clamp.clampLow);
    auto clampHighAttr = getF32FloatAttrOrNull(builder, config.clamp.clampHigh);
    builder.create<PPEFpClampOp>(loc, clampLowAttr, clampHighAttr);

    // PPEFpConvert
    auto clampModeAttr = getEnumAttrOrNull<PPEFpConvClampModeAttr>(builder, config.convert.clampMode);
    auto ftzModeAttr = getEnumAttrOrNull<PPEFpConvFTZModeAttr>(builder, config.convert.ftzMode);
    auto bf16RoundModeAttr = getEnumAttrOrNull<PPEFpConvBf16RoundModeAttr>(builder, config.convert.bf16RoundMode);
    builder.create<PPEFpConvertOp>(loc, config.convert.convertMode, clampModeAttr, ftzModeAttr, bf16RoundModeAttr);

    // PPEIntZeroPointOffset
    builder.create<PPEIntZeroPointOffsetOp>(loc, config.zeroPointOffset.zeroPointStatic);

    return mlir::success();
}

mlir::FailureOr<arch50xx::PPE::PPETask> VPUIPDPU::arch50xx::PPE::evalPPETasks(const Logger& log,
                                                                              mlir::Region& ppeRegion) {
    PPETask ppeTask{};
    for (auto ppeTaskOp : ppeRegion.getOps<VPUASM::PPETaskOp>()) {
        const auto fpPpeAttr = mlir::dyn_cast<VPU::PPEFpAttr>(ppeTaskOp.getPpeAttr());
        VPUX_THROW_WHEN(fpPpeAttr == nullptr,
                        "Expected PPEFpAttr type but got {0}, make sure to use the right factory version",
                        ppeTaskOp.getPpeAttr());

        const auto ppeMode = fpPpeAttr.getMode().getValue();
        if (ppeMode != VPU::PPEMode::NOOP) {
            if (ppeTask.fixedFunction.ppeMode != VPU::PPEMode::NOOP) {
                log.error("Cannot set more than one PPE task");
                return mlir::failure();
            }
            ppeTask.fixedFunction.ppeMode = ppeMode;
        }

        // Note: mlir::FloatAttr's store values as f64, while PPE HW uses f32.
        // Computing the PPE attributes in higher precision and then casting to f32 should make the most use
        // (accuracy-wise) out of the already imposed FloatAttr storage.
        const auto castCb = [](double value) {
            return static_cast<float>(value);
        };

        ppeTask.fixedFunction.fpClampLow = castCb(fpPpeAttr.getClampLow().getValueAsDouble());
        ppeTask.fixedFunction.fpClampHigh = castCb(fpPpeAttr.getClampHigh().getValueAsDouble());
        ppeTask.fpPreluAlpha = SmallVector<float>();
        llvm::transform(parseFPArrayAttr<double>(fpPpeAttr.getPreluAlpha()), std::back_inserter(ppeTask.fpPreluAlpha),
                        castCb);
        ppeTask.fpAdder = castCb(fpPpeAttr.getAdder().getValueAsDouble());

        if (const auto scaleAttr = fpPpeAttr.getScale()) {
            ppeTask.fpScaleData = castCb(scaleAttr.getValueAsDouble());
        }
        if (const auto biasAttr = fpPpeAttr.getBias()) {
            ppeTask.fpBias = castCb(biasAttr.getValueAsDouble());
        }
    }

    return ppeTask;
}

mlir::LogicalResult vpux::VPUIPDPU::arch50xx::buildDPUInvariantPPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    if (!origInvOp.getPpe().hasOneBlock()) {
        log.error("VPUASM::DPUInvariant->PPE is not a single block region");
        return mlir::failure();
    }

    arch50xx::PPE::PPEConfig config;
    auto outDataType =
            mlir::cast<mlir::MemRefType>(getInvBlockArg(BlockArg::ACT_OUT, invBlock, invBlockArgsPos).getType())
                    .getElementType();

    auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos);
    auto dpuTaskType = origInvOp.getNceTaskType();

    auto ppeTask = arch50xx::PPE::evalPPETasks(log, origInvOp.getPpe());
    if (mlir::failed(ppeTask)) {
        return mlir::failure();
    }

    if (arch50xx::PPE::configurePPE(config, outDataType, inAct.getType(), dpuTaskType, ppeTask.value(),
                                    getInvBlockArg(BlockArg::WEIGHTS_TABLE, invBlock, invBlockArgsPos) != nullptr,
                                    getInvBlockArg(BlockArg::SPR_LOOKUP_TABLE, invBlock, invBlockArgsPos) != nullptr)
                .failed()) {
        return mlir::failure();
    }

    if (arch50xx::PPE::buildPPEConfig(builder, origInvOp.getLoc(), log, config,
                                      getInvBlockArg(BlockArg::WEIGHTS_TABLE, invBlock, invBlockArgsPos))
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
