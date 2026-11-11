//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant_idu.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace {

using namespace vpux;
using namespace vpux::VPUIPDPU;

mlir::FailureOr<PPEScaleDataType> getPPEScaleDataType(const Logger& log, mlir::Type type) {
    if (mlir::isa<mlir::Float32Type>(type)) {
        return PPEScaleDataType::FP32;
    } else if (mlir::isa<mlir::Float8E5M2Type>(type)) {
        return PPEScaleDataType::BF8;
    } else if (mlir::isa<mlir::Float8E4M3FNType>(type)) {
        return PPEScaleDataType::HF8;
    } else if (mlir::isa<mlir::Float8E8M0FNUType>(type)) {
        return PPEScaleDataType::E8M0;
    }

    log.error("Unsupported PPE scale data type: {0}", type);
    return mlir::failure();
}

}  // namespace

namespace vpux::VPUIPDPU::arch60xx::IDU {

mlir::LogicalResult configureDepthWiseCfg(const Logger&, IDUConfig::DepthWiseCfg& config, VPUIP::NCETaskType taskType,
                                          bool smallKernelOptimization) {
    if (taskType == VPUIP::NCETaskType::DWCONV && !smallKernelOptimization) {
        config.dw3x3s1OptDisable = true;
    } else if (taskType == VPUIP::NCETaskType::MAXPOOL || taskType == VPUIP::NCETaskType::AVEPOOL) {
        // Must be set = 1 (true) for MAXPOOL ops and AVEPOOL ops
        config.dw3x3s1OptDisable = true;
    }

    return mlir::success();
}

mlir::LogicalResult configureWeightTableSplit([[maybe_unused]] mlir::Operation* op, const Logger& log,
                                              IDUConfig::WeightTableSplit& config, bool wtLegacy, bool wtDataPtr,
                                              bool wtSpPtr, bool wtZp, bool perChZp, mlir::Type scalesElementType,
                                              VPUIP::NCETaskType taskType) {
    if (wtLegacy) {
        return mlir::success();
    }

    if (!wtDataPtr && taskType == VPUIP::NCETaskType::DWCONV) {
        log.error("Weight table split invalid configuration - weight data pointer must be set for DWCONV!");
        return mlir::failure();
    }

    if (!wtDataPtr && !wtSpPtr && !wtZp && !perChZp) {
        config.wtSplitMode = WtSplitMode::NO_WT_READS;
    } else if (wtDataPtr && !wtZp && !perChZp) {
        config.wtSplitMode = WtSplitMode::WEIGHTS_ONLY;
    } else if (!wtDataPtr && !wtSpPtr && wtZp && perChZp) {
        config.wtSplitMode = WtSplitMode::ZERO_POINTS_ONLY;
    } else if (wtDataPtr && !wtZp && perChZp) {
        config.wtSplitMode = WtSplitMode::WEIGHTS_AND_ZPS;
    } else {
        log.error("Weight table split invalid configuration!");
        return mlir::failure();
    }

    if (scalesElementType) {
        auto scaleDtype = getPPEScaleDataType(log, scalesElementType);
        if (mlir::failed(scaleDtype)) {
            return mlir::failure();
        }
        config.scaleDtype = scaleDtype.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureWeightTableLegacyZp([[maybe_unused]] mlir::Operation* op,
                                                 IDUConfig::WeightTableLegacyZp& config, mlir::Type weightsType,
                                                 bool wtLegacy) {
    if (!wtLegacy) {
        return mlir::success();
    }

    if (!weightsType) {
        return mlir::success();
    }

    if (mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(weightsType)) {
        config.wtLegacyZpMode = WtLegacyZpMode::WEIGHTS_PTR_AND_ZPS;
    } else {
        config.wtLegacyZpMode = WtLegacyZpMode::WEIGHTS_PTR;
    }

    return mlir::success();
}

mlir::LogicalResult configurePalletization(const Logger& log, IDUConfig::Weights& config, mlir::Type weightsType) {
    if (mlir::isa_and_present<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(
                weightsType)) {
        auto quantized = mlir::cast<mlir::quant::QuantizedType>(weightsType);
        if (const auto quantileStorageType = mlir::dyn_cast<mlir::quant::QuantileType>(quantized.getStorageType())) {
            const auto bitWidth = quantileStorageType.getStorageWidth();

            if (bitWidth < 1 || bitWidth > 6) {
                log.error("Unsupported palletization bitwidth {0} from type {1}. Only bitwidths of 1-6 are supported.",
                          bitWidth, quantileStorageType);
                return mlir::failure();
            }
            const std::map<uint32_t, IDUWeightPalletMode> palletBitWidthMap = {
                    {1, IDUWeightPalletMode::ONE_BIT_PLT},   {2, IDUWeightPalletMode::TWO_BIT_PLT},
                    {3, IDUWeightPalletMode::THREE_BIT_PLT}, {4, IDUWeightPalletMode::FOUR_BIT_PLT},
                    {5, IDUWeightPalletMode::FIVE_BIT_PLT},  {6, IDUWeightPalletMode::SIX_BIT_PLT}};
            config.pltMode = palletBitWidthMap.at(bitWidth);

            if (!config.quantileLUT) {
                config.quantileLUT.emplace();
            }
            auto quantiles = quantileStorageType.getQuantiles();
            config.quantileLUT->assign(quantiles.begin(), quantiles.end());
        } else {
            config.pltMode = IDUWeightPalletMode::NO_PLT;
        }
    } else {
        config.pltMode = IDUWeightPalletMode::NO_PLT;
    }
    return mlir::success();
}

mlir::LogicalResult configureSubtractMpeWtBias(IDUConfig::SubtractMpeWtBias& config, mlir::Type inActType,
                                               mlir::Type weightsType) {
    if (mlir::isa_and_nonnull<mlir::quant::UniformQuantizedPerAxisType>(weightsType) ||
        mlir::isa_and_nonnull<mlir::quant::UniformQuantizedType>(weightsType)) {
        auto isPalletModeEnabled = [weightsType]() -> bool {
            auto quantized = mlir::cast<mlir::quant::QuantizedType>(weightsType);
            return mlir::isa_and_present<mlir::quant::QuantileType>(quantized.getStorageType());
        }();
        auto inType = getBaseType(inActType);
        auto wtType = getBaseType(weightsType, isPalletModeEnabled);
        const auto isSupportedWtType = wtType.isSignedInteger(8) || wtType.isSignlessInteger(8) ||
                                       (isPalletModeEnabled && wtType.isUnsignedInteger(8)) || wtType.isInteger(4) ||
                                       (isPalletModeEnabled && wtType.isF16());

        if (inType.isF16() && isSupportedWtType) {
            config.subtractMpeWtBias = true;
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureWorkload(const Logger& log, IDUConfig::WorkloadCfg& config, VPUIP::NCETaskType taskType,
                                      int64_t kernelX, int64_t kernelY, mlir::Type inActType, mlir::Type outActType,
                                      VPU::MPEMode mpeFrequentMode, bool isContinued) {
    constexpr int64_t accuShapeW = 16;
    constexpr int64_t accuShapeH = 16;
    constexpr int64_t accuShapeC = 32;

    if (!inActType) {
        if (!outActType) {
            log.error("Invalid DPU workload: no input activations and also no output provided");
            return mlir::failure();
        }

        // check supported nthw_ntk settings
        // CUBOID_16x16 implies setting NTHW_NTK_16_4
        if (mpeFrequentMode != VPU::MPEMode::CUBOID_16x16) {
            log.error("Invalid DPU workload: only NTHW_NTK_16_4 supported for accumulators contexts dumping "
                      "(mpeFrequentMode = {0})",
                      mpeFrequentMode);
            return mlir::failure();
        }

        auto outShape = mlir::cast<vpux::NDTypeInterface>(outActType).getShape();
        if (outShape[Dims4D::Act::W] != accuShapeW || outShape[Dims4D::Act::H] != accuShapeH ||
            outShape[Dims4D::Act::C] != accuShapeC) {
            log.error("Invalid DPU workload: output activations shape {0} not supported for accumulators contexts "
                      "dumping",
                      outShape);
            return mlir::failure();
        }

        auto outActElementType = mlir::cast<mlir::MemRefType>(outActType).getElementType();
        auto outBaseType = getBaseType(outActElementType);
        if (outBaseType.isInteger(CHAR_BIT * sizeof(int32_t))) {
            config.workloadType = IDUWorkloadType::ACCU_STORE_I32;
        } else if (outBaseType.isF32()) {
            config.workloadType = IDUWorkloadType::ACCU_STORE_FP32;
        } else {
            log.error("Unsupported out_activations type {0} for accumulators contexts dumping", outBaseType);
            return mlir::failure();
        }

        return mlir::success();
    } else {
        auto inActElementType = mlir::cast<mlir::MemRefType>(inActType).getElementType();
        auto inBaseType = getBaseType(inActElementType);
        bool doAccuLoad = false;
        if (inBaseType.isInteger(CHAR_BIT * sizeof(int32_t))) {
            config.workloadType = IDUWorkloadType::ACCU_LOAD_I32;
            doAccuLoad = true;
        } else if (inBaseType.isF32()) {
            config.workloadType = IDUWorkloadType::ACCU_LOAD_FP32;
            doAccuLoad = true;
        }
        if (doAccuLoad) {
            if (mpeFrequentMode != VPU::MPEMode::CUBOID_16x16) {
                log.error("Invalid DPU workload: only NTHW_NTK_16_4 supported for accumulators contexts loading "
                          "(mpeFrequentMode = {0})",
                          mpeFrequentMode);
                return mlir::failure();
            }
            auto inShape = mlir::cast<vpux::NDTypeInterface>(inActType).getShape();
            if (inShape[Dims4D::Act::W] != accuShapeW || inShape[Dims4D::Act::H] != accuShapeH ||
                inShape[Dims4D::Act::C] != accuShapeC) {
                log.error("Invalid DPU workload: input activations shape {0} not supported for accumulators contexts "
                          "loading",
                          inShape);
                return mlir::failure();
            }
            if (kernelX != 1 || kernelY != 1) {
                log.error("Invalid DPU workload: kernel size {0}x{1}, but only 1x1 is supported for accumulators "
                          "contexts loading",
                          kernelX, kernelY);
                return mlir::failure();
            }
            if (!isContinued) {
                log.error("Invalid DPU workload: must be continued for accumulators contexts loading");
                return mlir::failure();
            }

            auto outActElementType = mlir::cast<mlir::MemRefType>(outActType).getElementType();
            auto outBaseType = getBaseType(outActElementType);
            if (outBaseType != inBaseType) {
                log.error("Unsupported out_activations type {0} for accumulators contexts loading", outBaseType);
                return mlir::failure();
            }

            return mlir::success();
        }
    }

    return arch50xx::IDU::configureWorkload(log, config, taskType, kernelX, kernelY);
}

mlir::LogicalResult configureWeights(const Logger& log, IDUConfig::Weights& config, VPUIP::NCETaskType taskType,
                                     mlir::Type inActType, mlir::Type weightsType, bool wtSparse) {
    auto inBaseType = getBaseType(inActType);
    if (inBaseType.isInteger(CHAR_BIT * sizeof(int32_t)) || inBaseType.isF32()) {
        config.wMode = inBaseType;
        // in case of IDUWorkloadType::ACCU_LOAD wt_dense should be 0 (sparse)
        config.wtSparse = true;
        return mlir::success();
    }

    return arch50xx::IDU::configureWeights(log, config, taskType, inActType, weightsType, wtSparse);
}

mlir::LogicalResult configureIDU(mlir::Operation* op, const Logger& log, IDU::IDUConfig& config, mlir::Type inActType,
                                 mlir::Type weightsElementType, mlir::Type outActType, VPUIP::NCETaskType taskType,
                                 std::optional<int64_t> spPattern, std::optional<bool> inChannelsCompression,
                                 bool smallKernelOptimization, bool inActSparse, bool weightsSparse,
                                 std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 std::optional<VPU::EltwiseType> eltwiseType, const arch50xx::IDU::PPETask& ppeTask,
                                 bool wtLegacy, bool wtDataPtr, bool wtSpPtr, bool wtZp, bool perChZp,
                                 mlir::Type scalesElementType, VPU::MPEMode mpeFrequentMode, bool isContinued) {
    if (!inActType) {
        if (weightsElementType) {
            log.error("Invalid DPU workload: no input activations but weights provided");
            return mlir::failure();
        }

        // IDUWorkloadCfg
        return arch60xx::IDU::configureWorkload(log, config.workloadCfg, taskType, 0, 0, inActType, outActType,
                                                mpeFrequentMode, isContinued);
    }

    // IDUInActivations
    if (arch40xx::IDU::configureInActivations(log, config.inActivations, inActSparse).failed()) {
        return mlir::failure();
    }

    // IDUWeights
    auto inActElementType = mlir::cast<mlir::MemRefType>(inActType).getElementType();
    if (arch60xx::IDU::configurePalletization(log, config.weights, weightsElementType).failed()) {
        return mlir::failure();
    }
    if (arch60xx::IDU::configureWeights(log, config.weights, taskType, inActElementType, weightsElementType,
                                        weightsSparse)
                .failed()) {
        return mlir::failure();
    }

    // IDUInputLayerCfg
    if (arch40xx::IDU::configureSparsityPattern(log, config.inputLayerCfg, spPattern, inChannelsCompression).failed()) {
        return mlir::failure();
    }

    // IDUStorageElement
    auto seSizeVal = seSize.value_or(0);
    if (inActSparse && seSizeVal) {
        auto inputZ = mlir::cast<vpux::NDTypeInterface>(inActType).getShape()[Dims4D::Act::C];
        if ((taskType == VPUIP::NCETaskType::ELTWISE) && (seSizeVal != inputZ)) {
            // E#170667: se_size != input_Z is not supported due to a HW limitation
            log.error("Storage element size ({0}) for eltwise != Z dim ({1}) is not supported.", seSizeVal, inputZ);
            return mlir::failure();
        }
    }

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
    if (arch60xx::IDU::configureWorkload(log, config.workloadCfg, taskType, config.kernel.kernelX,
                                         config.kernel.kernelY, inActType, outActType, mpeFrequentMode, isContinued)
                .failed()) {
        return mlir::failure();
    }

    // IDUDepthWiseCfg
    if (arch60xx::IDU::configureDepthWiseCfg(log, config.depthWiseCfg, taskType, smallKernelOptimization).failed()) {
        return mlir::failure();
    }

    // IDUEltWiseCfg
    if (arch50xx::IDU::configureEltwiseCfg(log, config.eltWiseCfg, taskType, inActElementType, weightsElementType,
                                           ppeTask)
                .failed()) {
        return mlir::failure();
    }

    // IDUEltwiseMode
    if (arch50xx::IDU::configureEltwiseMode(config.eltwiseMode, taskType, eltwiseType).failed()) {
        return mlir::failure();
    }

    if (arch60xx::IDU::configureWeightTableSplit(op, log, config.weightTableSplit, wtLegacy, wtDataPtr, wtSpPtr, wtZp,
                                                 perChZp, scalesElementType, taskType)
                .failed()) {
        return mlir::failure();
    }

    if (arch60xx::IDU::configureWeightTableLegacyZp(op, config.weightTableLegacyZp, weightsElementType, wtLegacy)
                .failed()) {
        return mlir::failure();
    }

    if (arch60xx::IDU::configureSubtractMpeWtBias(config.subtractMpeWeightsBias, inActElementType, weightsElementType)
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const IDUConfig& config,
                                   mlir::Value inAct, bool wtLegacy, VPUIP::NCETaskType taskType,
                                   bool useWorkloadTypeInWT, bool isZpInUint4Range) {
    if (!inAct) {
        // IDUWorkloadCfg
        IDUWorkloadCfgOp::create(builder, loc, config.workloadCfg.workloadType);

        return mlir::success();
    }

    if (arch50xx::IDU::buildIDUConfig(builder, loc, config, inAct).failed()) {
        return mlir::failure();
    }

    if (!wtLegacy) {
        // WeightTableSplit
        auto scaleDtypeAttr = getEnumAttrOrNull<PPEScaleDataTypeAttr>(builder, config.weightTableSplit.scaleDtype);
        vpux::VPUIPDPU::IDUWorkloadTypeAttr workloadTypeAttr;
        if (useWorkloadTypeInWT) {
            workloadTypeAttr = IDUWorkloadTypeAttr::get(builder.getContext(), config.workloadCfg.workloadType);
        }
        WeightTableSplitOp::create(builder, loc, config.weightTableSplit.wtSplitMode, workloadTypeAttr, scaleDtypeAttr);
    } else if (isZpInUint4Range && taskType == VPUIP::NCETaskType::DWCONV && wtLegacy) {
        // WeightTableLegacyZp
        WeightTableLegacyZpOp::create(builder, loc, config.weightTableLegacyZp.wtLegacyZpMode);
    }

    // IDUSubtractMpeWtBias
    if (config.subtractMpeWeightsBias.subtractMpeWtBias) {
        IDUSubtractMpeWtBiasOp::create(builder, loc);
    }

    return mlir::success();
}

}  //  namespace vpux::VPUIPDPU::arch60xx::IDU

mlir::LogicalResult vpux::VPUIPDPU::arch60xx::buildDPUInvariantIDU(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    IDU::IDUConfig config;
    mlir::Type inActType, weightsElementType, outActType, scaleType;
    auto isZpInUint4Range = false;
    auto wtLegacy = getInvBlockArg(BlockArg::WEIGHTS_TABLE, invBlock, invBlockArgsPos) != nullptr;
    auto wtZp = getInvBlockArg(BlockArg::WEIGHTS_ZERO_POINTS, invBlock, invBlockArgsPos) != nullptr;
    auto wtDataPtr = getInvBlockArg(BlockArg::WEIGHTS_TABLE_DATA_PTR, invBlock, invBlockArgsPos) != nullptr;

    // If zero-point table was created, that means that weights have per-channel zero points. Since
    // buildDPUInvariantIDU is called per invariant, there's a chance that this particular invariant got a part of
    // weights that have the same zero points, due to SEGMENTED distribution tensor mode. Still weights can be
    // per-channel quantized, due to different scales. Because we are not taking to consideration here whether weights
    // should have different scales but zero points, we will set this flag to true always, if zero-point table was
    // created.
    auto perChWeightsZp = wtZp;

    auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos);
    if (inAct) {
        inActType = inAct.getType();
    }
    if (auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
        weightsElementType = mlir::cast<mlir::MemRefType>(weights.getType()).getElementType();

        if (const auto uniformQuantPerAxisType =
                    mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(weightsElementType)) {
            if (wtLegacy) {
                perChWeightsZp = !areAllZeroPointsEqual(uniformQuantPerAxisType);
                auto zeroPoints = uniformQuantPerAxisType.getZeroPoints();

                // In hardware, legacy format used for DW conv allows us to store per-channel zero points only on 4 MS
                // bits. So, ZPs should fit into UINT4 range (0-15).
                const uint32_t uint4Max = 15;
                if (llvm::all_of(zeroPoints, [](int64_t zp) {
                        return zp >= 0 && zp <= uint4Max;
                    })) {
                    isZpInUint4Range = true;
                }
            }
        }
        if (wtDataPtr && mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElementType)) {
            auto mpeEngineAttr = mlir::dyn_cast_or_null<VPU::MPEEngineAttr>(origInvOp.getMpeEngineAttr());
            // mpeEngineAttr with per-tensor zero-point is created during conversion from IE to VPU dialect, and set
            // if one of the following conditions is met:
            // - weights are per-tensor quantized with per-tensor zero-point
            // - weights are per-channel quantized with the same zp on each channel, but different per-channel scale
            // Since the flag is called "perChWeightsZp", we ensure that the mpeEngine attr is either doesn't exist
            // or weightZp attribute is not set.
            perChWeightsZp = mpeEngineAttr == nullptr || mpeEngineAttr.getWeightZp() == nullptr;
        }
    }
    if (auto outAct = getInvBlockArg(BlockArg::ACT_OUT, invBlock, invBlockArgsPos)) {
        outActType = outAct.getType();
    }
    if (auto scales = getInvBlockArg(BlockArg::WEIGHTS_TABLE_SCALE, invBlock, invBlockArgsPos)) {
        scaleType = mlir::cast<mlir::MemRefType>(scales.getType()).getElementType();
    }

    auto inActSparseMap = getInvBlockArg(BlockArg::ACT_SPARSE_MAP_IN, invBlock, invBlockArgsPos);
    auto inActSE = getInvBlockArg(BlockArg::ACT_SE_IN, invBlock, invBlockArgsPos);
    auto isSEOnlyOp = inActSparseMap == nullptr && inActSE != nullptr;
    auto ppeTask = arch50xx::IDU::evalPPETasks(origInvOp.getPpe(), origInvOp.getNceTaskType());

    if (IDU::configureIDU(origInvOp.getOperation(), log, config, inActType, weightsElementType, outActType,
                          origInvOp.getNceTaskType(), origInvOp.getCmSpPattern(),
                          origInvOp.getInputChannelsCompression(), origInvOp.getIsSmallKernelOptimized(),
                          isSEOnlyOp || inActSparseMap != nullptr,
                          getInvBlockArg(BlockArg::WEIGHTS_SPARSE_MAP, invBlock, invBlockArgsPos) != nullptr,
                          origInvOp.getKernelSize(), origInvOp.getKernelStrides(), origInvOp.getInputSeSize(),
                          origInvOp.getEltwiseType(), ppeTask, wtLegacy, wtDataPtr,
                          getInvBlockArg(BlockArg::WEIGHTS_TABLE_SP_PTR, invBlock, invBlockArgsPos) != nullptr, wtZp,
                          perChWeightsZp, scaleType, origInvOp.getMpeFrequentMode(), origInvOp.getIsContinued())
                .failed()) {
        return mlir::failure();
    }

    if (IDU::buildIDUConfig(builder, origInvOp.getLoc(), config, inAct, wtLegacy, origInvOp.getNceTaskType(),
                            /*useWorkloadTypeInWT*/ false, isZpInUint4Range)
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}
