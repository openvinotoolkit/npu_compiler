//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_idu.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_idu.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIPDPU::arch50xx::IDU {

mlir::LogicalResult getInQuantConfig(const Logger& log, mlir::Type in1Type, mlir::Type in2Type, const PPETask& ppeTask,
                                     SmallVector<float>& in1MultFp, SmallVector<float>& in2MultFp) {
    if (VPUIPDPU::arch40xx::IDU::verifyInQuantConfig(log, in1Type).failed()) {
        return mlir::failure();
    }

    if (VPUIPDPU::arch40xx::IDU::verifyInQuantConfig(log, in2Type).failed()) {
        return mlir::failure();
    }

    if (ppeTask.in1MultFp.has_value()) {
        in1MultFp = ppeTask.in1MultFp.value();
    }
    if (ppeTask.in2MultFp.has_value()) {
        in2MultFp = ppeTask.in2MultFp.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureEltwiseMode(VPUIPDPU::arch50xx::IDU::IDUConfig::EltwiseMode& config,
                                         VPUIP::NCETaskType taskType, std::optional<VPU::EltwiseType> eltwiseType) {
    if (taskType == VPUIP::NCETaskType::ELTWISE) {
        config.eltwiseModeOp = true;
        if (eltwiseType.has_value()) {
            switch (eltwiseType.value()) {
            case VPU::EltwiseType::ADD:
                config.eltwiseType = VPUIPDPU::IDUEltwiseType::ADD;
                break;
            case VPU::EltwiseType::SUBTRACT:
                config.eltwiseType = VPUIPDPU::IDUEltwiseType::SUBTRACT;
                break;
            case VPU::EltwiseType::MULTIPLY:
                config.eltwiseType = VPUIPDPU::IDUEltwiseType::MULT;
                break;
            default:
                VPUX_THROW("Eltwise type not supported: {0}", eltwiseType.value());
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureEltwiseCfg(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig::EltWiseCfg& config,
                                        VPUIP::NCETaskType taskType, mlir::Type inActType, mlir::Type weightsType,
                                        const PPETask& ppeTask) {
    if (taskType == VPUIP::NCETaskType::ELTWISE) {
        config.eltWiseCfgOp = true;

        const auto isInputQuantizationProvided = (ppeTask.in1MultFp.has_value() && ppeTask.in2MultFp.has_value());
        SmallVector<float> in1MultFp, in2MultFp;
        if (isInputQuantizationProvided) {
            if (getInQuantConfig(log, inActType, weightsType, ppeTask, in1MultFp, in2MultFp).failed()) {
                return mlir::failure();
            }
        }

        auto inType = getBaseType(inActType);
        auto wtType = getBaseType(weightsType);
        if (!in1MultFp.empty() && !in1MultFp.empty()) {
            if (!mlir::isa<mlir::FloatType>(inType)) {
                config.elopScaleA = static_cast<int64_t>(in1MultFp[0]);
                config.elopScaleB = static_cast<int64_t>(in2MultFp[0]);
            } else {
                config.elopScapeFp = true;
                config.fpElopScaleA = in1MultFp[0];
                config.fpElopScaleB = in2MultFp[0];
            }
        }

        if (mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(inType) ||
            mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(wtType)) {
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

mlir::LogicalResult configureWorkload(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig::WorkloadCfg& config,
                                      VPUIP::NCETaskType taskType, int64_t kernelX, int64_t kernelY) {
    switch (taskType) {
    case VPUIP::NCETaskType::REDUCEMEAN:
        config.workloadType = IDUWorkloadType::REDUCEMEAN;
        break;
    case VPUIP::NCETaskType::REDUCESUMSQUARE:
        config.workloadType = IDUWorkloadType::REDUCESUMSQUARE;
        break;
    case VPUIP::NCETaskType::REDUCESUM:
        config.workloadType = IDUWorkloadType::REDUCESUM;
        break;
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

mlir::LogicalResult configureStorageElement(const Logger& log,
                                            VPUIPDPU::arch50xx::IDU::IDUConfig::StorageElement& config,
                                            VPUIP::NCETaskType taskType, const NDTypeInterface& inActType,
                                            bool inSparsityEnabled, std::optional<int64_t> seSize) {
    if (taskType == VPUIP::NCETaskType::CONV || taskType == VPUIP::NCETaskType::ELTWISE ||
        taskType == VPUIP::NCETaskType::REDUCEMEAN || taskType == VPUIP::NCETaskType::REDUCESUMSQUARE ||
        taskType == VPUIP::NCETaskType::REDUCESUM) {
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

mlir::LogicalResult configureWeights(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig::Weights& config,
                                     VPUIP::NCETaskType taskType, mlir::Type inActType, mlir::Type weightsType,
                                     bool wtSparse) {
    if (taskType == VPUIP::NCETaskType::MAXPOOL || taskType == VPUIP::NCETaskType::AVEPOOL ||
        taskType == VPUIP::NCETaskType::REDUCEMEAN || taskType == VPUIP::NCETaskType::REDUCESUMSQUARE ||
        taskType == VPUIP::NCETaskType::REDUCESUM) {
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

    if (taskType == VPUIP::NCETaskType::AVEPOOL || taskType == VPUIP::NCETaskType::REDUCEMEAN ||
        taskType == VPUIP::NCETaskType::REDUCESUMSQUARE || taskType == VPUIP::NCETaskType::REDUCESUM) {
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

mlir::LogicalResult configureIDU(const Logger& log, VPUIPDPU::arch50xx::IDU::IDUConfig& config,
                                 const vpux::NDTypeInterface& inActType, mlir::Type weightsElementType,
                                 VPUIP::NCETaskType taskType, std::optional<int64_t> spPattern,
                                 std::optional<bool> inChannelsCompression, std::optional<bool> smallKernelOptimization,
                                 bool inActSparse, bool weightsSparse, std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 std::optional<VPU::EltwiseType> eltwiseType, const PPETask& ppeTask) {
    // IDUInActivations
    if (arch40xx::IDU::configureInActivations(log, config.inActivations, inActSparse).failed()) {
        return mlir::failure();
    }

    // IDUWeights
    auto inActElementType = mlir::cast<mlir::MemRefType>(inActType).getElementType();
    if (arch40xx::IDU::configurePalletization(log, config.weights, weightsElementType).failed()) {
        return mlir::failure();
    }
    if (arch50xx::IDU::configureWeights(log, config.weights, taskType, inActElementType, weightsElementType,
                                        weightsSparse)
                .failed()) {
        return mlir::failure();
    }

    // IDUInputLayerCfg
    if (arch40xx::IDU::configureSparsityPattern(log, config.inputLayerCfg, spPattern, inChannelsCompression).failed()) {
        return mlir::failure();
    }

    // IDUStorageElement
    if (arch50xx::IDU::configureStorageElement(log, config.storageElement, taskType, inActType, inActSparse, seSize)
                .failed()) {
        return mlir::failure();
    }

    // IDUKernel
    if (arch40xx::IDU::configureKernel(log, config.kernel, kernelSize).failed()) {
        return mlir::failure();
    }

    // IDUStride
    if (arch40xx::IDU::configureStride(log, config.stride, kernelStrides).failed()) {
        return mlir::failure();
    }

    // IDUWorkloadCfg
    if (arch50xx::IDU::configureWorkload(log, config.workloadCfg, taskType, config.kernel.kernelX,
                                         config.kernel.kernelY)
                .failed()) {
        return mlir::failure();
    }

    // IDUDepthWiseCfg
    if (arch40xx::IDU::configureDepthWiseCfg(log, config.depthWiseCfg, taskType, smallKernelOptimization).failed()) {
        return mlir::failure();
    }

    // IDUEltWiseCfg
    if (configureEltwiseCfg(log, config.eltWiseCfg, taskType, inActElementType, weightsElementType, ppeTask).failed()) {
        return mlir::failure();
    }

    // IDUEltwiseMode
    return configureEltwiseMode(config.eltwiseMode, taskType, eltwiseType);
}

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc,
                                   const VPUIPDPU::arch50xx::IDU::IDUConfig& config, mlir::Value inAct) {
    if (arch40xx::IDU::buildIDUConfig(builder, loc, config, inAct).failed()) {
        return mlir::failure();
    }

    // IDUEltWiseMode
    if (config.eltwiseMode.eltwiseModeOp) {
        builder.create<IDUEltWiseModeOp>(loc, config.eltwiseMode.eltwiseType);
    }

    return mlir::success();
}

PPETask evalPPETasks(mlir::Region& ppeRegion, std::optional<VPUIP::NCETaskType> taskType) {
    PPETask ppeTask{};

    for (auto ppeTaskOp : ppeRegion.getOps<VPUASM::PPETaskOp>()) {
        const auto fpPpeAttr = mlir::dyn_cast<VPU::PPEFpAttr>(ppeTaskOp.getPpeAttr());
        VPUX_THROW_WHEN(fpPpeAttr == nullptr,
                        "Expected PPEFpAttr type but got {0}, make sure to use the right factory version",
                        ppeTaskOp.getPpeAttr());

        // Note: mlir::FloatAttr's store values as f64, while PPE HW uses f32.
        // Computing the PPE attributes in higher precision and then casting to f32 should make the most use
        // (accuracy-wise) out of the already imposed FloatAttr storage.
        const auto castCb = [](double value) {
            return static_cast<float>(value);
        };

        if (const auto in1MultAttr = fpPpeAttr.getIn1Mult()) {
            ppeTask.in1MultFp = SmallVector<float>();
            llvm::transform(parseFPArrayAttr<double>(in1MultAttr), std::back_inserter(*ppeTask.in1MultFp), castCb);
        }
        if (const auto in2MultAttr = fpPpeAttr.getIn2Mult()) {
            ppeTask.in2MultFp = SmallVector<float>();
            llvm::transform(parseFPArrayAttr<double>(in2MultAttr), std::back_inserter(*ppeTask.in2MultFp), castCb);
        }

        if (taskType.has_value() && taskType.value() == VPUIP::NCETaskType::ELTWISE) {
            const auto ppeMode = fpPpeAttr.getMode().getValue();
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

}  // namespace vpux::VPUIPDPU::arch50xx::IDU

mlir::LogicalResult vpux::VPUIPDPU::arch50xx::buildDPUInvariantIDU(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    IDU::IDUConfig config;
    auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos);
    mlir::Type weightsType;
    if (auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
        weightsType = mlir::cast<mlir::MemRefType>(weights.getType()).getElementType();
    }

    auto inActSparseMap = getInvBlockArg(BlockArg::ACT_SPARSE_MAP_IN, invBlock, invBlockArgsPos);
    auto inActSE = getInvBlockArg(BlockArg::ACT_SE_IN, invBlock, invBlockArgsPos);
    auto isSEOnlyOp = inActSparseMap == nullptr && inActSE != nullptr;
    auto ppeTask = IDU::evalPPETasks(origInvOp.getPpe(), origInvOp.getNceTaskType());
    if (IDU::configureIDU(log, config, inAct.getType(), weightsType, origInvOp.getNceTaskType(),
                          origInvOp.getCmSpPattern(), origInvOp.getInputChannelsCompression(),
                          origInvOp.getIsSmallKernelOptimized(), isSEOnlyOp || inActSparseMap != nullptr,
                          getInvBlockArg(BlockArg::WEIGHTS_SPARSE_MAP, invBlock, invBlockArgsPos) != nullptr,
                          origInvOp.getKernelSize(), origInvOp.getKernelStrides(), origInvOp.getInputSeSize(),
                          origInvOp.getEltwiseType(), ppeTask)
                .failed()) {
        return mlir::failure();
    }

    if (IDU::buildIDUConfig(builder, origInvOp.getLoc(), config, inAct).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
