//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_ppe.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace {

using namespace VPUIPDPU;
using namespace VPUIPDPU::arch40xx::PPE;

struct PPEConfig {
    struct FpPPE {
        struct BiasAdd {
            std::optional<float> biasStatic;
        } biasAdd;
        struct ScaleMult {
            std::optional<float> scaleStatic;
            std::optional<float> preluAlpha;
        } scaleMult;
        struct AddMultBypass {
            VPUIPDPU::PPEBypassMode bypassMode = VPUIPDPU::PPEBypassMode::ON;
        } addMultBypass;
        struct Convert {
            VPUIPDPU::PPEFpConvertMode convertMode = VPUIPDPU::PPEFpConvertMode::NONE;
            std::optional<VPUIPDPU::PPEFpConvClampMode> clampMode;
            std::optional<VPUIPDPU::PPEFpConvFTZMode> ftzMode;
            std::optional<VPUIPDPU::PPEFpConvBf16RoundMode> bf16RoundMode;
        } convert;
    } fpPPE;
    struct IntPPE {
        struct BiasAdd {
            std::optional<int64_t> biasStatic;
        } biasAdd;
        struct ScaleMult {
            std::optional<int64_t> scaleStatic;
        } scaleMult;
        struct PreluMult {
            int64_t preluMultStatic = 1;
        } preluMult;
        struct ScaleShift {
            std::optional<int64_t> shiftStatic;
        } scaleShift;
        struct PreluShift {
            int64_t preluShiftStatic = 0;
        } preluShift;
        struct Round {
            VPUIPDPU::PPEIntRoundMode roundMode = VPUIPDPU::PPEIntRoundMode::RNE;
        } round;
        struct ZeroPointOffset {
            int64_t zeroPointStatic = 0;
        } zeroPointOffset;
        struct Clamp {
            std::optional<int64_t> clampLow;
            int64_t clampHigh = 0;
        } clamp;
        struct Convert {
            VPUIPDPU::PPEIntConvertMode convertMode = VPUIPDPU::PPEIntConvertMode::NONE;
        } convert;
    } intPPE;
};

enum class PPEUseCase { INT_INT, INT_FP, FP_INT, FP_FP, Unsupported };

PPEUseCase detectPPEUseCase(mlir::Type inDataType, mlir::Type outDataType) {
    constexpr auto ioTypesNum = static_cast<size_t>(IOType::IOTypeNum);
    const PPEUseCase ppeUseCases[ioTypesNum][ioTypesNum] = {{PPEUseCase::INT_INT, PPEUseCase::INT_FP},
                                                            {PPEUseCase::FP_INT, PPEUseCase::FP_FP}};

    const auto inSelect = getIOType(inDataType), outSelect = getIOType(outDataType);

    if (inSelect == IOType::IOTypeNum || outSelect == IOType::IOTypeNum) {
        return PPEUseCase::Unsupported;
    }

    return ppeUseCases[static_cast<size_t>(inSelect)][static_cast<size_t>(outSelect)];
}

mlir::LogicalResult configureFpPPE(const Logger&, PPEConfig::FpPPE& config, VPUIP::NCETaskType dpuTaskType,
                                   const PPETask& ppeTask) {
    if (config.addMultBypass.bypassMode != VPUIPDPU::PPEBypassMode::ON) {
        if (dpuTaskType == VPUIP::NCETaskType::ELTWISE || dpuTaskType == VPUIP::NCETaskType::AVEPOOL) {
            config.biasAdd.biasStatic = 0.0f;
            config.scaleMult.scaleStatic = ppeTask.fpScaleData;
        } else if (dpuTaskType == VPUIP::NCETaskType::MAXPOOL) {
            config.biasAdd.biasStatic = 0.0f;
            config.scaleMult.scaleStatic = 1.0f;
        }

        if ((ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELU) ||
            (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELUX) ||
            (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LPRELU)) {
            config.scaleMult.preluAlpha = ppeTask.fpPreluAlpha;
        }
    }

    if (ppeTask.ppeFpScale.has_value()) {
        config.scaleMult.scaleStatic = ppeTask.ppeFpScale.value();
    }
    if (ppeTask.ppeFpBias.has_value()) {
        config.biasAdd.biasStatic = ppeTask.ppeFpBias.value();
    }
    return mlir::success();
}

mlir::LogicalResult getPPEQuantConfig(const Logger& log, mlir::Type type, const PPETask& ppeTask,
                                      SmallVector<int64_t>& quantMult, SmallVector<int64_t>& quantShift,
                                      SmallVector<uint8_t>& quantZero) {
    if (const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
        quantZero.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type)) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZero.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZero.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZero.push_back(0);
    }

    quantMult = ppeTask.ppeQuantMult.value();
    quantShift = ppeTask.ppeQuantShift.value();

    if (quantShift.size() != quantZero.size()) {
        log.error("Mismatch of size between quant shift/mult vector and quant ZP:  {0} != {1}", quantShift.size(),
                  quantZero.size());
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureIntPPE(const Logger& log, PPEConfig::IntPPE& config, mlir::Type outDataType,
                                    VPUIP::NCETaskType dpuTaskType, const PPETask& ppeTask, bool bypassPPEInt) {
    SmallVector<int64_t> quantMult;
    SmallVector<int64_t> quantShift;
    SmallVector<uint8_t> quantZero;

    const auto isQuantizationProvided = ppeTask.ppeQuantMult.has_value() && ppeTask.ppeQuantShift.has_value() &&
                                        ppeTask.ppeQuantPostShift.has_value();
    const auto isQuantizationNotProvided = !ppeTask.ppeQuantMult.has_value() && !ppeTask.ppeQuantShift.has_value() &&
                                           !ppeTask.ppeQuantPostShift.has_value();
    if (!isQuantizationProvided && !isQuantizationNotProvided) {
        log.error("Missing quantization scale settings.");
        return mlir::failure();
    }
    if (isQuantizationProvided) {
        if (getPPEQuantConfig(log, outDataType, ppeTask, quantMult, quantShift, quantZero).failed()) {
            return mlir::failure();
        }
    } else {
        if (getQuantConfig(log, outDataType, quantMult, quantShift, quantZero).failed()) {
            return mlir::failure();
        }
    }

    if (bypassPPEInt) {
        config.biasAdd.biasStatic = 0;
        config.scaleMult.scaleStatic = 1;
        config.scaleShift.shiftStatic = 0;
    } else {
        // bias&scale
        if (dpuTaskType == VPUIP::NCETaskType::ELTWISE || dpuTaskType == VPUIP::NCETaskType::AVEPOOL) {
            config.biasAdd.biasStatic = 0;
            // An ELTWISE with non neutral fpScaleData suggests that it represents a PermuteQuantize. PermuteQuantize
            // has the constraint that both input and output must have the same type, which applies also for quantized
            // types, details in IE::isPurePermuteCompatiblePrecision. In this case the non neutral value of fpScaleData
            // must be placed inside scaleStatic and shiftStatic and the output quantized type must be ignored in the
            // same manner the input quantized type is
            // TODO(E#193194): Refine this workaround by properly populating PPE earlier
            if (dpuTaskType == VPUIP::NCETaskType::ELTWISE && ppeTask.fpScaleData != 1.0) {
                const auto scaleApproximation = QuantizationApproximation(ppeTask.fpScaleData);
                config.scaleMult.scaleStatic = scaleApproximation.mult();
                config.scaleShift.shiftStatic = scaleApproximation.shift();
            } else {
                config.scaleMult.scaleStatic = !quantMult.empty() ? quantMult[0] : 1;
                config.scaleShift.shiftStatic = !quantShift.empty() ? quantShift[0] : 0;
            }
        } else if (dpuTaskType == VPUIP::NCETaskType::MAXPOOL) {
            config.biasAdd.biasStatic = 0;
            if (outDataType.isF16()) {
                config.scaleMult.scaleStatic = !quantMult.empty() ? quantMult[0] : 1;
                config.scaleShift.shiftStatic = !quantShift.empty() ? quantShift[0] : 0;
            } else {
                config.scaleMult.scaleStatic = 1;
                config.scaleShift.shiftStatic = 0;
            }
        }

        // prelu alpha
        if (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LPRELU) {  // leaky relu
            config.preluMult.preluMultStatic = ppeTask.fixedFunction.lReluMult;
            config.preluShift.preluShiftStatic = ppeTask.fixedFunction.lReluShift;
        } else if (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELU ||
                   ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELUX) {  // relu or relux
            // zero negative slope
            config.preluMult.preluMultStatic = 0;
            config.preluShift.preluShiftStatic = 0;
        } else {  // no activation function
            config.preluMult.preluMultStatic = 1;
            config.preluShift.preluShiftStatic = 0;
        }
    }

    // round
    if (dpuTaskType == VPUIP::NCETaskType::MAXPOOL) {
        config.round.roundMode = VPUIPDPU::PPEIntRoundMode::NONE;
    }

    // zero point offset
    if (getIOType(outDataType) == IOType::INT) {
        if (quantZero.size() && dpuTaskType != VPUIP::NCETaskType::MAXPOOL) {
            config.zeroPointOffset.zeroPointStatic = quantZero[0];
        }
    }

    // clamp
    config.clamp.clampHigh = ppeTask.fixedFunction.intClampHigh;
    if (config.convert.convertMode == VPUIPDPU::PPEIntConvertMode::NONE) {
        config.clamp.clampLow = ppeTask.fixedFunction.intClampLow;
    }

    return mlir::success();
}

mlir::LogicalResult configurePPE(const Logger& log, PPEConfig& config, mlir::Type inDataType, mlir::Type outDataType,
                                 VPUIP::NCETaskType dpuTaskType, const PPETask& ppeTask) {
    bool bypassPPEInt = false;
    switch (detectPPEUseCase(inDataType, outDataType)) {
    case PPEUseCase::INT_INT:
    case PPEUseCase::INT_FP:
        if (getBaseType(inDataType).isInteger(CHAR_BIT) && getBaseType(outDataType).isBF16()) {
            log.error("X8 in, I32 convolution, BF16 out is not supported by the hardware");
            return mlir::failure();
        }
        config.fpPPE.addMultBypass.bypassMode = VPUIPDPU::PPEBypassMode::ON;
        config.fpPPE.convert.convertMode = VPUIPDPU::PPEFpConvertMode::NONE;
        if (mlir::isa<mlir::FloatType>(outDataType)) {
            if (!outDataType.isF16()) {
                log.error("Activation data type conversion from integer to floating point only supported to FP16");
                return mlir::failure();
            }
            config.intPPE.convert.convertMode = VPUIPDPU::PPEIntConvertMode::FP16;
        }
        break;
    case PPEUseCase::FP_INT:
        config.fpPPE.convert.convertMode = VPUIPDPU::PPEFpConvertMode::I32;
        LLVM_FALLTHROUGH;  // intentional fallthrough
    case PPEUseCase::FP_FP:
        config.fpPPE.addMultBypass.bypassMode = VPUIPDPU::PPEBypassMode::OFF;
        bypassPPEInt = true;
        if (outDataType.isF16()) {
            config.fpPPE.convert.convertMode = VPUIPDPU::PPEFpConvertMode::FP16;
            config.fpPPE.convert.clampMode = VPUIPDPU::PPEFpConvClampMode::ON;
        } else if (outDataType.isBF16()) {
            config.fpPPE.convert.convertMode = VPUIPDPU::PPEFpConvertMode::BF16;
            config.fpPPE.convert.bf16RoundMode = VPUIPDPU::PPEFpConvBf16RoundMode::RNE;
        }
        if (dpuTaskType == VPUIP::NCETaskType::MAXPOOL && (inDataType.isF16() || inDataType.isBF16())) {
            config.fpPPE.addMultBypass.bypassMode = VPUIPDPU::PPEBypassMode::ON;
            config.fpPPE.convert.convertMode = VPUIPDPU::PPEFpConvertMode::NONE;
            config.fpPPE.convert.bf16RoundMode = std::nullopt;
            config.fpPPE.convert.clampMode = std::nullopt;
        }
        break;
    default:
        log.error("Unsupported PPE use case");
        return mlir::failure();
    }

    if (configureFpPPE(log, config.fpPPE, dpuTaskType, ppeTask).failed()) {
        return mlir::failure();
    }

    if (configureIntPPE(log, config.intPPE, outDataType, dpuTaskType, ppeTask, bypassPPEInt).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult buildFpPPEConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                     const PPEConfig::FpPPE& config, mlir::Value weightsTable) {
    if (config.addMultBypass.bypassMode != VPUIPDPU::PPEBypassMode::ON) {
        // PPEFpBiasAdd
        auto biasStaticAttr = getF32FloatAttrOrNull(builder, config.biasAdd.biasStatic);
        if (biasStaticAttr) {
            builder.create<VPUIPDPU::PPEFpBiasAddOp>(loc, nullptr, biasStaticAttr);
        } else {
            if (!weightsTable) {
                log.error("Expected weights_table operand in PPE FP pipeline");
                return mlir::failure();
            }
            builder.create<VPUIPDPU::PPEFpBiasAddOp>(loc, weightsTable, nullptr);
        }

        // PPEFpScaleMult
        auto preluAlphaAttr = getF32FloatAttrOrNull(builder, config.scaleMult.preluAlpha);
        auto scaleStaticAttr = getF32FloatAttrOrNull(builder, config.scaleMult.scaleStatic);
        if (scaleStaticAttr) {
            builder.create<VPUIPDPU::PPEFpScalePreluMultOp>(loc, nullptr, scaleStaticAttr, preluAlphaAttr);
        } else {
            if (!weightsTable) {
                log.error("Expected weights_table operand in PPE FP pipeline");
                return mlir::failure();
            }
            builder.create<VPUIPDPU::PPEFpScalePreluMultOp>(loc, weightsTable, nullptr, preluAlphaAttr);
        }
    }

    // PPEFpAddMultBypass
    builder.create<VPUIPDPU::PPEFpAddMultBypassOp>(loc, config.addMultBypass.bypassMode);

    // PPEFpConvert
    auto clampModeAttr = getEnumAttrOrNull<VPUIPDPU::PPEFpConvClampModeAttr>(builder, config.convert.clampMode);
    auto ftzModeAttr = getEnumAttrOrNull<VPUIPDPU::PPEFpConvFTZModeAttr>(builder, config.convert.ftzMode);
    auto bf16RoundModeAttr =
            getEnumAttrOrNull<VPUIPDPU::PPEFpConvBf16RoundModeAttr>(builder, config.convert.bf16RoundMode);
    builder.create<VPUIPDPU::PPEFpConvertOp>(loc, config.convert.convertMode, clampModeAttr, ftzModeAttr,
                                             bf16RoundModeAttr);

    return mlir::success();
}

mlir::LogicalResult buildIntPPEConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      const PPEConfig::IntPPE& config, mlir::Value weightsTable) {
    // PPEIntBiasAdd
    auto biasStaticAttr = getI64IntegerAttrOrNull(builder, config.biasAdd.biasStatic);
    if (biasStaticAttr) {
        builder.create<VPUIPDPU::PPEIntBiasAddOp>(loc, nullptr, biasStaticAttr);
    } else {
        if (!weightsTable) {
            log.error("Expected weights_table operand in PPE INT pipeline");
            return mlir::failure();
        }
        builder.create<VPUIPDPU::PPEIntBiasAddOp>(loc, weightsTable, nullptr);
    }

    // PPEIntScaleMult
    auto scaleStaticAttr = getI64IntegerAttrOrNull(builder, config.scaleMult.scaleStatic);
    if (scaleStaticAttr) {
        builder.create<VPUIPDPU::PPEIntScaleMultOp>(loc, nullptr, scaleStaticAttr);
    } else {
        if (!weightsTable) {
            log.error("Expected weights_table operand in PPE INT pipeline");
            return mlir::failure();
        }
        builder.create<VPUIPDPU::PPEIntScaleMultOp>(loc, weightsTable, nullptr);
    }

    // PPEIntScaleShift
    auto shiftStaticAttr = getI64IntegerAttrOrNull(builder, config.scaleShift.shiftStatic);
    if (shiftStaticAttr) {
        builder.create<VPUIPDPU::PPEIntScaleShiftOp>(loc, nullptr, shiftStaticAttr);
    } else {
        if (!weightsTable) {
            log.error("Expected weights_table operand in PPE INT pipeline");
            return mlir::failure();
        }
        builder.create<VPUIPDPU::PPEIntScaleShiftOp>(loc, weightsTable, nullptr);
    }

    // PPEIntPreluMult
    builder.create<VPUIPDPU::PPEIntPreluMultOp>(loc, config.preluMult.preluMultStatic);

    // PPEIntPreluShift
    builder.create<VPUIPDPU::PPEIntPreluShiftOp>(loc, config.preluShift.preluShiftStatic);

    // PPEIntRound
    builder.create<VPUIPDPU::PPEIntRoundOp>(loc, config.round.roundMode);

    // PPEIntZeroPointOffset
    builder.create<VPUIPDPU::PPEIntZeroPointOffsetOp>(loc, config.zeroPointOffset.zeroPointStatic);

    // PPEIntClamp
    auto clampLowAttr = getI64IntegerAttrOrNull(builder, config.clamp.clampLow);
    builder.create<VPUIPDPU::PPEIntClampOp>(loc, clampLowAttr, config.clamp.clampHigh);

    // PPEIntConvert
    builder.create<VPUIPDPU::PPEIntConvertOp>(loc, config.convert.convertMode);

    return mlir::success();
}

mlir::LogicalResult buildPPEConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const PPEConfig& config, mlir::Value weightsTable) {
    if (buildFpPPEConfig(builder, loc, log, config.fpPPE, weightsTable).failed()) {
        return mlir::failure();
    }

    if (buildIntPPEConfig(builder, loc, log, config.intPPE, weightsTable).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

mlir::FailureOr<PPETask> vpux::VPUIPDPU::arch40xx::PPE::evalPPETasks(const Logger& log, mlir::Region& ppeRegion) {
    PPETask ppeTask{};

    for (auto ppeTaskOp : ppeRegion.getOps<VPUASM::PPETaskOp>()) {
        auto ppeAttr = ppeTaskOp.getPpeAttr();
        auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
        VPUX_THROW_WHEN(intPpeAttr == nullptr,
                        "Expected PPEIntAttr type but got {0}, make sure to use the right factory version", ppeAttr);

        if (intPpeAttr.getMode()) {
            const auto ppeMode = intPpeAttr.getMode().getValue();
            if (ppeMode != VPU::PPEMode::NOOP) {
                if (ppeTask.fixedFunction.ppeMode != VPU::PPEMode::NOOP) {
                    log.error("Cannot set more than one PPE task");
                    return mlir::failure();
                }
                ppeTask.fixedFunction.ppeMode = ppeMode;
            }
        }

        if (const auto clampLowAttr = intPpeAttr.getClampLow()) {
            ppeTask.fixedFunction.intClampLow = checked_cast<int32_t>(clampLowAttr.getValue().getSExtValue());
        }
        if (const auto clampHighAttr = intPpeAttr.getClampHigh()) {
            ppeTask.fixedFunction.intClampHigh = checked_cast<int32_t>(clampHighAttr.getValue().getSExtValue());
        }
        if (const auto LreluMultAttr = intPpeAttr.getLreluMult()) {
            ppeTask.fixedFunction.lReluMult = checked_cast<int32_t>(LreluMultAttr.getValue().getSExtValue());
        }
        if (const auto LreluShiftAttr = intPpeAttr.getLreluShift()) {
            ppeTask.fixedFunction.lReluShift = checked_cast<uint32_t>(LreluShiftAttr.getValue().getSExtValue());
        }
        if (const auto quantMultAttr = intPpeAttr.getQuantMult()) {
            ppeTask.ppeQuantMult = parseIntArrayAttr<int64_t>(quantMultAttr);
        }
        if (const auto quantShiftAttr = intPpeAttr.getQuantShift()) {
            ppeTask.ppeQuantShift = parseIntArrayAttr<int64_t>(quantShiftAttr);
        }
        if (const auto quantPostShiftAttr = intPpeAttr.getQuantPostShift()) {
            ppeTask.ppeQuantPostShift = checked_cast<int64_t>(quantPostShiftAttr.getValue().getSExtValue());
        }
        // Note: For values like 0.1, checked_cast fails, due to loss in precision when converting
        // from double to float and back, due to the static_cast<double>(static_cast<float>(value)) == value
        // check; use static_cast instead
        if (const auto quantScaleAttr = intPpeAttr.getQuantScale()) {
            auto floatScaleAttr = quantScaleAttr.getValue()[0];
            ppeTask.fpScaleData =
                    static_cast<float>(mlir::dyn_cast<mlir::FloatAttr>(floatScaleAttr).getValueAsDouble());
        }
        if (const auto fpPReluAlphaAttr = intPpeAttr.getFpPreluAlpha()) {
            ppeTask.fpPreluAlpha = static_cast<float>(intPpeAttr.getFpPreluAlpha().getValueAsDouble());
        }
    }

    if ((ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELU) ||
        (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELUX)) {
        ppeTask.fpPreluAlpha = -0.0f;  // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                       // (0x00000000), not negative zero (0x80000000)
    }

    return ppeTask;
}

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUInvariantPPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    if (!origInvOp.getPpe().hasOneBlock()) {
        log.error("VPUASM::DPUInvariant->PPE is not a single block region");
        return mlir::failure();
    }

    PPEConfig config;
    auto inDataType =
            mlir::cast<mlir::MemRefType>(getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos).getType())
                    .getElementType();
    auto outDataType =
            mlir::cast<mlir::MemRefType>(getInvBlockArg(BlockArg::ACT_OUT, invBlock, invBlockArgsPos).getType())
                    .getElementType();
    auto dpuTaskType = origInvOp.getNceTaskType();

    auto ppeTask = evalPPETasks(log, origInvOp.getPpe());
    if (mlir::failed(ppeTask)) {
        return mlir::failure();
    }

    if (configurePPE(log, config, inDataType, outDataType, dpuTaskType, ppeTask.value()).failed()) {
        return mlir::failure();
    }

    if (buildPPEConfig(builder, origInvOp.getLoc(), log, config,
                       getInvBlockArg(BlockArg::WEIGHTS_TABLE, invBlock, invBlockArgsPos))
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
