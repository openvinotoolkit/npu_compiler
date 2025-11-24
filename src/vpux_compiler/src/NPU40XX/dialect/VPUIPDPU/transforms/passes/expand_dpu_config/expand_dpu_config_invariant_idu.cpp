//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_idu.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Support/LogicalResult.h>

namespace vpux::VPUIPDPU::arch40xx::IDU {

mlir::LogicalResult verifyInQuantConfig(const Logger& log, mlir::Type inType) {
    SmallVector<uint8_t> inQuantZero = getZeroPoints<uint8_t>(inType);

    if (inQuantZero.size() != 1) {
        log.error("Mismatch of size between input quant ZP and quant shift vector:  {0} != 1", inQuantZero.size());
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult getInQuantConfig(const Logger& log, mlir::Type in1Type, mlir::Type in2Type, const PPETask& ppeTask,
                                     SmallVector<int64_t>& in1QuantMult, SmallVector<int64_t>& in1QuantShift,
                                     SmallVector<int64_t>& in2QuantMult, SmallVector<int64_t>& in2QuantShift,
                                     SmallVector<float>& in1QuantMultFp, SmallVector<float>& in2QuantMultFp) {
    if (verifyInQuantConfig(log, in1Type).failed()) {
        return mlir::failure();
    }

    if (verifyInQuantConfig(log, in2Type).failed()) {
        return mlir::failure();
    }

    if (ppeTask.in1QuantMult.has_value()) {
        in1QuantMult = ppeTask.in1QuantMult.value();
    }
    if (ppeTask.in2QuantMult.has_value()) {
        in2QuantMult = ppeTask.in2QuantMult.value();
    }
    if (ppeTask.in1QuantMultFp.has_value()) {
        in1QuantMultFp = ppeTask.in1QuantMultFp.value();
    }
    if (ppeTask.in2QuantMultFp.has_value()) {
        in2QuantMultFp = ppeTask.in2QuantMultFp.value();
    }

    in1QuantShift = in2QuantShift = {0};

    return mlir::success();
}

mlir::LogicalResult configureInActivations(const Logger&, IDUConfig::InActivations& config, bool inSparse) {
    config.inSparse = inSparse;

    return mlir::success();
}

mlir::LogicalResult configurePalletization(const Logger& log, IDUConfig::Weights& config, mlir::Type weightsType) {
    if (llvm::isa_and_nonnull<mlir::quant::QuantileQuantizedType>(weightsType) ||
        llvm::isa_and_nonnull<mlir::quant::QuantileQuantizedPerAxisType>(weightsType)) {
        auto storageType = llvm::dyn_cast<mlir::quant::QuantizedType>(weightsType).getStorageType();

        const auto bitWidth = storageType.getIntOrFloatBitWidth();
        if (bitWidth != 1 && bitWidth != 2 && bitWidth != 4) {
            log.error("Unsupported palletization bitwidth {0} from type {1}. Only bitwidths of 1,2,4 are supported.",
                      bitWidth, storageType);
            return mlir::failure();
        }
        const std::map<uint32_t, IDUWeightPalletMode> palletBitWidthMap = {{1, IDUWeightPalletMode::ONE_BIT_PLT},
                                                                           {2, IDUWeightPalletMode::TWO_BIT_PLT},
                                                                           {4, IDUWeightPalletMode::FOUR_BIT_PLT}};
        config.pltMode = palletBitWidthMap.at(bitWidth);

        if (!config.quantileLUT) {
            config.quantileLUT.emplace();
        }

        if (auto quantileQuantizedType = llvm::dyn_cast<mlir::quant::QuantileQuantizedType>(weightsType)) {
            auto quantiles = quantileQuantizedType.getQuantiles();
            config.quantileLUT->assign(quantiles.begin(), quantiles.end());
        } else if (auto quantileQuantizedPerAxisType =
                           llvm::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(weightsType)) {
            auto quantiles = quantileQuantizedPerAxisType.getQuantiles();
            config.quantileLUT->assign(quantiles.begin(), quantiles.end());
        } else {
            log.error("Palletization mode is enabled but weight type doesn't contain quantile LUT");
            return mlir::failure();
        }
    } else {
        config.pltMode = IDUWeightPalletMode::NO_PLT;
    }

    return mlir::success();
}

mlir::LogicalResult configureWeights(const Logger& log, IDUConfig::Weights& config, VPUIP::NCETaskType taskType,
                                     mlir::Type inActType, mlir::Type weightsType, bool wtSparse) {
    if (taskType == VPUIP::NCETaskType::MAXPOOL || taskType == VPUIP::NCETaskType::AVEPOOL) {
        config.wMode = getBaseType(inActType);
    } else {
        if (!weightsType) {
            log.error("Missing weights data for DPU task {0}", VPUIP::stringifyNCETaskType(taskType));
            return mlir::failure();
        }
        const bool isPalletModeEnabled = llvm::isa_and_nonnull<mlir::quant::QuantileQuantizedType>(weightsType) ||
                                         llvm::isa_and_nonnull<mlir::quant::QuantileQuantizedPerAxisType>(weightsType);
        config.wMode = getBaseType(weightsType, isPalletModeEnabled);
    }

    if (taskType == VPUIP::NCETaskType::AVEPOOL) {
        if (config.wMode.isInteger(CHAR_BIT * sizeof(uint8_t))) {
            config.poolWtData = 0x0101;  // Two I8/U8 values => 0x0101;
        } else if (config.wMode.isF16()) {
            config.poolWtData = 0x3c00;  // fp16 1
        } else if (config.wMode.isBF16()) {
            config.poolWtData = 0x3f80;  // bf16 1
        } else if (mlir::isa<mlir::Float8E5M2Type>(config.wMode)) {
            config.poolWtData = 0x3c3c;  // bf8 1
        } else if (mlir::isa<mlir::Float8E4M3FNType>(config.wMode)) {
            config.poolWtData = 0x3838;  // hf8 1
        } else {
            log.error("Input data type not supported for AVEPOOL");
            return mlir::failure();
        }
    }

    config.wtSparse = (taskType == VPUIP::NCETaskType::MAXPOOL) || wtSparse;

    return mlir::success();
}

mlir::LogicalResult configureSparsityPattern(const Logger&, IDUConfig::InputLayerCfg& config,
                                             std::optional<int64_t> spPattern,
                                             std::optional<bool> inChannelsCompression) {
    if (spPattern.has_value()) {
        config.sparsityPattern = spPattern.value();
        if (config.sparsityPattern) {
            config.inputCompressed = inChannelsCompression.value_or(0);
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureStorageElement(const Logger& log, IDUConfig::StorageElement& config,
                                            VPUIP::NCETaskType taskType, const NDTypeInterface& inActType,
                                            bool inSparsityEnabled, std::optional<int64_t> seSize) {
    if (taskType == VPUIP::NCETaskType::CONV || taskType == VPUIP::NCETaskType::ELTWISE) {
        auto seSizeVal = seSize.value_or(0);
        if (inSparsityEnabled && seSizeVal) {
            auto inputZ = inActType.getShape()[Dims4D::Act::C];
            if ((taskType == VPUIP::NCETaskType::ELTWISE) && (seSizeVal != inputZ)) {
                log.warning("Storage_element_size ({0}) for eltwise != Z dim ({1}) ---- not tested", seSizeVal, inputZ);
            }
            config.seSize = seSizeVal;
            if (seSizeVal != 0) {
                auto numSEsInZDir = (inputZ / seSizeVal) - 1;
                if (inputZ % seSizeVal) {
                    ++numSEsInZDir;
                }
                config.numSEsInZDir = numSEsInZDir;
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureKernel(const Logger&, IDUConfig::Kernel& config,
                                    std::optional<mlir::ArrayAttr> kernelSize) {
    if (kernelSize.has_value()) {
        auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
        config.kernelX = kernelSizeArray[1];
        config.kernelY = kernelSizeArray[0];
    }

    return mlir::success();
}

mlir::LogicalResult configureStride(const Logger&, IDUConfig::Stride& config,
                                    std::optional<mlir::ArrayAttr> kernelStrides) {
    if (kernelStrides.has_value()) {
        auto kernelStridesArray = parseIntArrayAttr<int64_t>(kernelStrides.value());
        config.strideX = kernelStridesArray[1];
        config.strideY = kernelStridesArray[0];
    }

    return mlir::success();
}

mlir::LogicalResult configureWorkload(const Logger& log, IDUConfig::WorkloadCfg& config, VPUIP::NCETaskType taskType,
                                      int64_t kernelX, int64_t kernelY) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        config.workloadType = IDUWorkloadType::CONV;
        break;
    case VPUIP::NCETaskType::DWCONV:
        config.workloadType = IDUWorkloadType::DWCONV;
        break;
    case VPUIP::NCETaskType::MAXPOOL:
        config.workloadType = IDUWorkloadType::MAXPOOL;
        break;
    case VPUIP::NCETaskType::AVEPOOL:
        config.workloadType = IDUWorkloadType::AVEPOOL;
        break;
    case VPUIP::NCETaskType::ELTWISE: {
        if (kernelX != 1 || kernelY != 1) {
            log.error("Eltwise only supports 1x1 kernel. Got '{0}' x '{1}'", kernelX, kernelY);
            return mlir::failure();
        }
        config.workloadType = IDUWorkloadType::ELTWISE;
    } break;
    case VPUIP::NCETaskType::IDENTITY:
    default:
        log.error("Workload not supported '{0}'", VPUIP::stringifyNCETaskType(taskType));
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureDepthWiseCfg(const Logger&, IDUConfig::DepthWiseCfg& config, VPUIP::NCETaskType taskType,
                                          std::optional<bool> smallKernelOptimization) {
    if (taskType == VPUIP::NCETaskType::DWCONV && smallKernelOptimization.value_or(true) == false) {
        config.dw3x3s1OptDisable = true;
    } else if (taskType == VPUIP::NCETaskType::MAXPOOL) {
        // Must be set = 1 (true) for MAXPOOL ops
        config.dw3x3s1OptDisable = true;
    }
    return mlir::success();
}

mlir::LogicalResult configureEltwiseCfg(const Logger& log, IDUConfig::EltWiseCfg& config, VPUIP::NCETaskType taskType,
                                        mlir::Type inActType, mlir::Type weightsType, const PPETask& ppeTask) {
    if (taskType == VPUIP::NCETaskType::ELTWISE) {
        config.eltWiseCfgOp = true;

        const auto isInputQuantizationProvided =
                (ppeTask.in1QuantMult.has_value() && ppeTask.in2QuantMult.has_value()) ||
                (ppeTask.in1QuantMultFp.has_value() && ppeTask.in2QuantMultFp.has_value());
        SmallVector<int64_t> in1QuantMult, in2QuantMult;
        SmallVector<float> in1QuantMultFp, in2QuantMultFp;
        SmallVector<int64_t> in1QuantShift, in2QuantShift;
        if (isInputQuantizationProvided) {
            if (getInQuantConfig(log, inActType, weightsType, ppeTask, in1QuantMult, in1QuantShift, in2QuantMult,
                                 in2QuantShift, in1QuantMultFp, in2QuantMultFp)
                        .failed()) {
                return mlir::failure();
            }
        } else {
            SmallVector<uint8_t> in1QuantZero, in2QuantZero;
            if (getQuantConfig(log, inActType, in1QuantMult, in1QuantShift, in1QuantZero).failed()) {
                return mlir::failure();
            }
            if (getQuantConfig(log, weightsType, in2QuantMult, in2QuantShift, in2QuantZero).failed()) {
                return mlir::failure();
            }
        }

        if (!in1QuantMult.empty() && !in2QuantMult.empty() && !in1QuantShift.empty() && !in2QuantShift.empty()) {
            if (!in1QuantShift[0] && !in2QuantShift[0]) {
                config.elopScaleA = in1QuantMult[0];
                config.elopScaleB = in2QuantMult[0];
            }
        }

        if (!in1QuantMultFp.empty() && !in2QuantMultFp.empty() && !in1QuantShift.empty() && !in2QuantShift.empty()) {
            config.elopScapeFp = true;
            config.fpElopScaleA = in1QuantMultFp[0];
            config.fpElopScaleB = in2QuantMultFp[0];
        }

        auto inType = getBaseType(inActType);
        auto wtType = getBaseType(weightsType);
        if (mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(inType) ||
            mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(wtType)) {
            config.elopScapeFp = true;
        }

        if ((mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(inType) &&
             mlir::isa<mlir::BFloat16Type>(wtType)) ||
            (mlir::isa<mlir::BFloat16Type>(inType) &&
             mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(wtType))) {
            config.bf16FlowOn = true;
        }
    }

    return mlir::success();
}

PPETask evalPPETasks(mlir::Region& ppeRegion, std::optional<VPUIP::NCETaskType> taskType) {
    PPETask ppeTask;

    for (auto ppeTaskOp : ppeRegion.getOps<VPUASM::PPETaskOp>()) {
        auto ppeAttr = ppeTaskOp.getPpeAttr();
        auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
        VPUX_THROW_WHEN(intPpeAttr == nullptr,
                        "Expected PPEIntAttr type but got {0}, make sure to use the right factory version", ppeAttr);

        if (const auto in1QuantMultAttr = intPpeAttr.getIn1QuantMult()) {
            if (mlir::isa_and_nonnull<mlir::FloatAttr>(in1QuantMultAttr.getValue()[0])) {
                ppeTask.in1QuantMultFp = parseFPArrayAttr<float>(in1QuantMultAttr);
            } else if (mlir::isa_and_nonnull<mlir::IntegerAttr>(in1QuantMultAttr.getValue()[0])) {
                ppeTask.in1QuantMult = parseIntArrayAttr<int64_t>(in1QuantMultAttr);
            }
        }

        if (const auto in2QuantMultAttr = intPpeAttr.getIn2QuantMult()) {
            if (mlir::isa_and_nonnull<mlir::FloatAttr>(in2QuantMultAttr.getValue()[0])) {
                ppeTask.in2QuantMultFp = parseFPArrayAttr<float>(in2QuantMultAttr);
            } else if (mlir::isa_and_nonnull<mlir::IntegerAttr>(in2QuantMultAttr.getValue()[0])) {
                ppeTask.in2QuantMult = parseIntArrayAttr<int64_t>(in2QuantMultAttr);
            }
        }

        if (taskType.has_value() && taskType.value() == VPUIP::NCETaskType::ELTWISE) {
            const auto ppeMode = intPpeAttr.getMode().getValue();
            if (ppeMode != VPU::PPEMode::NOOP) {
                switch (ppeMode) {
                case VPU::PPEMode::ADD:
                    ppeTask.eltwiseType = IDUEltwiseType::ADD;
                    break;
                case VPU::PPEMode::SUB:
                    ppeTask.eltwiseType = IDUEltwiseType::SUBTRACT;
                    break;
                case VPU::PPEMode::MULT:
                    ppeTask.eltwiseType = IDUEltwiseType::MULT;
                    break;
                default:
                    break;
                }
            }
        }
    }

    return ppeTask;
}

mlir::LogicalResult configureIDU(const Logger& log, IDUConfig& config, const vpux::NDTypeInterface& inActType,
                                 mlir::Type weightsElementType, VPUIP::NCETaskType taskType,
                                 std::optional<int64_t> spPattern, std::optional<bool> inChannelsCompression,
                                 std::optional<bool> smallKernelOptimization, bool inActSparse, bool weightsSparse,
                                 std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 const PPETask& ppeTask) {
    // IDUInActivations
    if (configureInActivations(log, config.inActivations, inActSparse).failed()) {
        return mlir::failure();
    }

    // IDUWeights
    auto inActElementType = mlir::cast<mlir::MemRefType>(inActType).getElementType();
    if (configurePalletization(log, config.weights, weightsElementType).failed()) {
        return mlir::failure();
    }
    if (configureWeights(log, config.weights, taskType, inActElementType, weightsElementType, weightsSparse).failed()) {
        return mlir::failure();
    }

    // IDUInputLayerCfg
    if (configureSparsityPattern(log, config.inputLayerCfg, spPattern, inChannelsCompression).failed()) {
        return mlir::failure();
    }

    // IDUStorageElement
    if (configureStorageElement(log, config.storageElement, taskType, inActType, inActSparse, seSize).failed()) {
        return mlir::failure();
    }

    // IDUKernel
    if (configureKernel(log, config.kernel, kernelSize).failed()) {
        return mlir::failure();
    }

    // IDUStride
    if (configureStride(log, config.stride, kernelStrides).failed()) {
        return mlir::failure();
    }

    // IDUWorkloadCfg
    if (configureWorkload(log, config.workloadCfg, taskType, config.kernel.kernelX, config.kernel.kernelY).failed()) {
        return mlir::failure();
    }

    // IDUDepthWiseCfg
    if (configureDepthWiseCfg(log, config.depthWiseCfg, taskType, smallKernelOptimization).failed()) {
        return mlir::failure();
    }

    // IDUEltWiseCfg
    if (configureEltwiseCfg(log, config.eltWiseCfg, taskType, inActElementType, weightsElementType, ppeTask).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const IDUConfig& config,
                                   mlir::Value inAct) {
    // IDUInActivations
    builder.create<IDUInActivationsOp>(loc, inAct, config.inActivations.inSparse);

    // IDUWeights
    auto poolWtDataAttr = getI64IntegerAttrOrNull(builder, config.weights.poolWtData);
    auto quantileLUTAttr = getF64ArrayAttrOrNull(builder, config.weights.quantileLUT);
    builder.create<IDUWeightsOp>(loc, config.weights.wMode, poolWtDataAttr, config.weights.wtSparse,
                                 config.weights.pltMode, quantileLUTAttr);

    // IDUInputLayerCfg
    if (config.inputLayerCfg.sparsityPattern) {
        builder.create<IDUInputLayerCfgOp>(loc, config.inputLayerCfg.sparsityPattern,
                                           config.inputLayerCfg.inputCompressed);
    }

    // IDUStorageElement
    if (config.storageElement.seSize) {
        auto numSEsInZDirAttr = getI64IntegerAttrOrNull(builder, config.storageElement.numSEsInZDir);
        builder.create<IDUStorageElementOp>(loc, config.storageElement.seSize, numSEsInZDirAttr);
    }

    // IDUKernel
    builder.create<IDUKernelOp>(loc, config.kernel.kernelX, config.kernel.kernelY);

    // IDUStride
    builder.create<IDUStrideOp>(loc, config.stride.strideX, config.stride.strideY);

    // IDUWorkloadCfg
    builder.create<IDUWorkloadCfgOp>(loc, config.workloadCfg.workloadType);

    // IDUDepthWiseCfg
    if (config.depthWiseCfg.dw3x3s1OptDisable || config.depthWiseCfg.dwOptOffset.has_value()) {
        auto dwOptOffsetAttr = getI64IntegerAttrOrNull(builder, config.depthWiseCfg.dwOptOffset);
        builder.create<IDUDepthWiseCfgOp>(loc, config.depthWiseCfg.dw3x3s1OptDisable, dwOptOffsetAttr);
    }

    // IDUEltWiseCfg
    if (config.eltWiseCfg.eltWiseCfgOp) {
        if (config.eltWiseCfg.elopScapeFp) {
            auto fpElopScaleAAttr = builder.getF32FloatAttr(config.eltWiseCfg.fpElopScaleA);
            auto fpElopScaleBAttr = builder.getF32FloatAttr(config.eltWiseCfg.fpElopScaleB);
            builder.create<IDUEltWiseCfgOp>(loc, config.eltWiseCfg.bf16FlowOn, fpElopScaleAAttr, fpElopScaleBAttr);
        } else {
            auto elopScaleAAttr = getI64IntegerAttrOrNull(builder, config.eltWiseCfg.elopScaleA);
            auto elopScaleBAttr = getI64IntegerAttrOrNull(builder, config.eltWiseCfg.elopScaleB);
            builder.create<IDUEltWiseCfgOp>(loc, config.eltWiseCfg.bf16FlowOn, elopScaleAAttr, elopScaleBAttr);
        }
    }

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch40xx::IDU

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUInvariantIDU(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    IDU::IDUConfig config;
    auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos);
    mlir::Type weightsType;
    if (auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
        weightsType = mlir::cast<mlir::MemRefType>(weights.getType()).getElementType();
    }

    auto ppeTask = IDU::evalPPETasks(origInvOp.getPpe(), /*taskType*/ {});
    if (IDU::configureIDU(log, config, inAct.getType(), weightsType, origInvOp.getNceTaskType(),
                          origInvOp.getCmSpPattern(), origInvOp.getInputChannelsCompression(),
                          origInvOp.getIsSmallKernelOptimized(),
                          getInvBlockArg(BlockArg::ACT_SPARSE_MAP_IN, invBlock, invBlockArgsPos) != nullptr,
                          getInvBlockArg(BlockArg::WEIGHTS_SPARSE_MAP, invBlock, invBlockArgsPos) != nullptr,
                          origInvOp.getKernelSize(), origInvOp.getKernelStrides(), origInvOp.getInputSeSize(), ppeTask)
                .failed()) {
        return mlir::failure();
    }

    if (IDU::buildIDUConfig(builder, origInvOp.getLoc(), config, inAct).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
