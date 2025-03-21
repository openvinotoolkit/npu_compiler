//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model_data.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

namespace {

ArrayRef<char> getCostModelData(VPU::ArchKind archKind, bool isFastModel) {
    switch (archKind) {
    case VPU::ArchKind::NPU37XX:
        if (isFastModel) {
            return ArrayRef(VPU::COST_MODEL_2_7_FAST, VPU::COST_MODEL_2_7_FAST_SIZE);
        }
        return ArrayRef(VPU::COST_MODEL_2_7, VPU::COST_MODEL_2_7_SIZE);
    default:
        if (isFastModel) {
            return ArrayRef(VPU::COST_MODEL_4_0_FAST, VPU::COST_MODEL_4_0_FAST_SIZE);
        }
        return ArrayRef(VPU::COST_MODEL_4_0, VPU::COST_MODEL_4_0_SIZE);
    }
}

}  // namespace

std::shared_ptr<VPUNN::VPUCostModel> vpux::VPU::createCostModel(ArchKind arch) {
    // Track [E#70055]
    // TODO: Do not switch vpunn model to FAST temporarily, need to investigate the impact for workloads generation pass
    bool isFastModel = false;
    const auto costModelData = getCostModelData(arch, isFastModel);
    return std::make_shared<VPUNN::VPUCostModel>(costModelData.data(), costModelData.size(), false);
}

std::shared_ptr<VPUNN::VPULayerCostModel> vpux::VPU::createLayerCostModel(ArchKind arch) {
    // VPUNN provides two models - default and fast.
    // Currently use default model for workload generation. Ticket to explore moving to fast model [E#70055].
    // Currently use fast model for per layer evaluation in multi-cluster strategy selection
    bool isFastModel = true;
    const auto costModelData = getCostModelData(arch, isFastModel);
    auto layerCostModel = std::make_shared<VPUNN::VPULayerCostModel>(costModelData.data(), costModelData.size(), false);
    if (VPU::isArchVPUX3XXX(arch)) {
        // keep same per tile workload channel limit on 37XX after new vpunn software update
        layerCostModel->set_maxWorkloadsPerIntraTileSplit(50U);
    }
    return layerCostModel;
}

///@brief Validate vpunn cost. If cost is not the defined error code then return it
/// Else print and return error code (an uint32 value in [max-100, max]) to user.
/// Please report to E#80022 if any error code found in compilation log.
uint32_t vpux::VPU::checkAndReturnCost(const VPUNN::CyclesInterfaceType& cost, vpux::Logger log, bool beSilent) {
    if (VPUNN::Cycles::isErrorCode(cost)) {
        auto errorCode = VPUNN::Cycles::toErrorText(cost);
        if (beSilent) {
            log.trace("VPUNN error code {0} is caught, code val {1}", errorCode, cost);
        } else {
            log.warning("VPUNN error code {0} is caught, code val {1}", errorCode, cost);
        }
        return (cost == VPUNN::Cycles::ERROR_INPUT_TOO_BIG) ? VPU::ERROR_INPUT_TOO_BIG : VPU::INVALID_COST_BASE;
    }
    return cost;
}

///@brief Print vpunn config info
void vpux::VPU::printVPUNNLayerConfig(const VPUNN::DPULayer& layer, const VPUNN::VPULayerStrategy& strategy,
                                      vpux::Logger log) {
    std::ostringstream layerStream;
    layerStream << layer;
    log.trace("[VPUNN LOG] Layer config: {0}", layerStream.str());
    std::ostringstream strategyStream;
    strategyStream << strategy;
    log.trace("[VPUNN LOG] Strategy config: {0}", strategyStream.str());
}

/// @brief Print vpunn dpu workload for debug
/// @warning Default logCb is Trace level
void vpux::VPU::printVPUNNWorkloadConfig(const VPUNN::DPUWorkload& wl, LogCb logCb) {
    std::ostringstream wlStream;
    wlStream << wl;
    logCb(formatv("[VPUNN LOG] DPU workload config: {0}", wlStream.str()));
}

float vpux::getWeightsSparsityRatio(vpux::NDTypeInterface weightsType, int64_t compressedSize) {
    auto originalSize = weightsType.getShape().totalSize();
    auto elemType = weightsType.getElementType();
    auto elemByteSize = vpux::getElemTypeSize(elemType).to<Byte>().count();
    auto originalAllocSize = originalSize * elemByteSize;

    // This check is to pass UNINIT.STACK.MUST check for "weightsSparsityRatio" in klocwork
    VPUX_THROW_WHEN(originalAllocSize == 0, "Denominator should be non-zero when doing division");
    float weightsSparsityRatio = 1.0F - (checked_cast<float>(compressedSize) / checked_cast<float>(originalAllocSize));

    VPUX_THROW_UNLESS(weightsSparsityRatio >= 0.0 && weightsSparsityRatio <= 1.0,
                      "weightsSparsityRatio should be in range [0.0 , 1.0] however get {0}", weightsSparsityRatio);
    return weightsSparsityRatio;
}

///@brief Weights sparsity ratio basically is the math sparsity (the ratio of zero values) but considering the 16 Bytes
/// alignment for weights sets.
///@details A storage element is allocated to a weights set (ICxHxW), which has 16 Bytes alignment HW constraint.
/// Each weights set will be compressed to only include dense values and align to 16 B
/// And the total compressed_size stored in sparsityCompressionAttr, which is calculated by sparsify-weights pass.
/// So ratio can be calculated by 1 - (compressed_size / total_size)
float vpux::VPU::getWeightsSparsityRatio(mlir::Value weights) {
    const auto sparseType = weights.getType().dyn_cast<VPU::SparseTensorType>();
    VPUX_THROW_WHEN(sparseType == nullptr, "Not a sparse type");
    const auto sparsityCompressionAttr = sparseType.getSparsityCompression();
    VPUX_THROW_WHEN(sparsityCompressionAttr == nullptr, "sparsity_compressionAttr shouldn't be a nullptr");

    auto log = vpux::Logger("[calculate-sparstiy-ratio-vpunn]", LogLevel::None);
    log.trace("Calculate weights sparsity ratio for Weights {0}", weights.getLoc());
    auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();
    auto elemType = weightsType.getElementType();
    auto compressedSize = sparsityCompressionAttr.getAllocSize(elemType).count();

    auto weightsSparsityRatio = getWeightsSparsityRatio(weightsType, compressedSize);

    log.trace(" Sparsity ratio: {0}", weightsSparsityRatio);
    return weightsSparsityRatio;
}

VPUNN::VPUDevice vpux::VPU::getVPUDeviceType(VPU::ArchKind archKind) {
    switch (archKind) {
    case VPU::ArchKind::NPU37XX:
        return VPUNN::VPUDevice::VPU_2_7;
    case VPU::ArchKind::NPU40XX:
        return VPUNN::VPUDevice::VPU_4_0;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

bool vpux::VPU::isVPUNNSupportedElementType(mlir::Type type) {
    if (type.isBF16()) {
        return true;
    } else if (type.isF16()) {
        return true;
    } else if (type.isInteger(CHAR_BIT * sizeof(int8_t))) {
        return true;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return true;
    } else if (auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qType.getStorageTypeIntegralWidth() == 8) {
            return true;
        } else if (qType.getStorageTypeIntegralWidth() == 4) {
            // Temporary enablement; follow up E#103211
            return true;
        }
    }
    return false;
}

std::optional<VPUNN::DataType> vpux::VPU::getVPUNNElementType(mlir::Type type) {
    if (type.isBF16()) {
        return VPUNN::DataType::BFLOAT16;
    } else if (type.isF16()) {
        return VPUNN::DataType::FLOAT16;
    } else if (type.isInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::INT8;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::UINT8;
    } else if (auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qType.getStorageTypeIntegralWidth() == 8) {
            return qType.isSigned() ? VPUNN::DataType::INT8 : VPUNN::DataType::UINT8;
        } else if (qType.getStorageTypeIntegralWidth() == 4) {
            return qType.isSigned() ? VPUNN::DataType::INT4 : VPUNN::DataType::UINT4;
        }
    } else if (type.isF32()) {
        // Temporary enablement; follow up E#149202
        return VPUNN::DataType::BFLOAT16;
    }

    return std::nullopt;
}

VPUNN::Layout vpux::VPU::getVPUNNLayout(VPUIPDPU::ODUPermuteDataMode oduPermutation) {
    switch (oduPermutation) {
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZXY:
        return VPUNN::Layout::ZXY;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZYX:
        return VPUNN::Layout::ZYX;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_YZX:
        return VPUNN::Layout::YZX;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_YXZ:
        return VPUNN::Layout::YXZ;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_XZY:
        return VPUNN::Layout::XZY;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_XYZ:
        return VPUNN::Layout::XYZ;
    default:
        VPUX_THROW("Unsupported ODU permute mode: '{0}'", oduPermutation);
    }
}

VPUNN::VPUTensor vpux::VPU::getVPUTensor(ShapeRef shape, mlir::Type elemType,
                                         VPUIPDPU::ODUPermuteDataMode oduPermutation) {
    const auto nnType = VPU::getVPUNNElementType(elemType);
    VPUX_THROW_UNLESS(nnType.has_value(), "Unsupported data type: '{0}'", elemType);

    if (shape.size() == 5) {
        return VPUNN::VPUTensor(
                {
                        static_cast<unsigned int>(shape[DimsGroups5D::Act::W]),
                        static_cast<unsigned int>(shape[DimsGroups5D::Act::H]),
                        static_cast<unsigned int>(shape[DimsGroups5D::Act::C]),
                        static_cast<unsigned int>(shape[DimsGroups5D::Act::G] * shape[DimsGroups5D::Act::N]),
                },
                nnType.value(), getVPUNNLayout(oduPermutation));
    } else if (shape.size() == 4) {
        return VPUNN::VPUTensor(
                {
                        static_cast<unsigned int>(shape[Dims4D::Act::W]),
                        static_cast<unsigned int>(shape[Dims4D::Act::H]),
                        static_cast<unsigned int>(shape[Dims4D::Act::C]),
                        static_cast<unsigned int>(shape[Dims4D::Act::N]),
                },
                nnType.value(), getVPUNNLayout(oduPermutation));
    } else {
        VPUX_THROW("Not supported shape, with number of dimensions = {0}", shape.size());
    }
}

VPUNN::ExecutionMode vpux::VPU::getExecutionMode(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR:
        return VPUNN::ExecutionMode::VECTOR;
    case VPU::MPEMode::MATRIX:
        return VPUNN::ExecutionMode::MATRIX;
    case VPU::MPEMode::VECTOR_FP16:
        return VPUNN::ExecutionMode::VECTOR_FP16;
    case VPU::MPEMode::CUBOID_16x16:
        return VPUNN::ExecutionMode::CUBOID_16x16;
    case VPU::MPEMode::CUBOID_8x16:
        return VPUNN::ExecutionMode::CUBOID_8x16;
    case VPU::MPEMode::CUBOID_4x16:
        return VPUNN::ExecutionMode::CUBOID_4x16;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
}

/*
 * Determines the appropriate SOK layer strategy based on the distribution mode and architecture.
 *
 * SOK_NO_BROADCAST is used for specific architectures (>NPU40XX) to provide more accurate cost calculations
 * Mode DistributionMode::DUPLICATED | DistributionMode::SEGMENTED maps to VPUNN::VPUTilingStrategy::SOK
 * Mode DistributionMode::SEGMENTED maps to VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST
 *
 * For other architectures, both SOK modes map to VPUNN::VPUTilingStrategy::SOK
 * SOK_NO_BROADCAST is only utilized when the VPUNN cost is invalid for SOK to avoid performance regressions
 */
inline VPUNN::VPUTilingStrategy getSOKLayerStrategy(vpux::VPU::DistributionMode distributionMode,
                                                    vpux::VPU::ArchKind arch) {
    if (distributionMode == vpux::VPU::DistributionMode::SEGMENTED && arch > vpux::VPU::ArchKind::NPU40XX) {
        return VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST;
    }
    return VPUNN::VPUTilingStrategy::SOK;
}

/**
 * @param nTiles the number of CMX tiles
 * @param nDPUs Number of DPU per CMX tile
 * @param nSHVs the number of Act_Shave per CMX tiles
 * @param prefetching a boolean to determine whether to prefetch
 * @param distributionMode the tensor distribution mode
 */
VPUNN::VPULayerStrategy vpux::VPU::getVPULayerStrategy(VPU::MultiClusterStrategy strategy, size_t nDPUs, size_t nTiles,
                                                       ArchKind arch, size_t nSHVs, bool prefetching,
                                                       DistributionMode distributionMode) {
    VPUNN::VPULayerStrategy VPUNNStrategy;
    VPUNNStrategy.nDPUs = static_cast<unsigned int>(nDPUs);
    VPUNNStrategy.nSHVs = static_cast<unsigned int>(nSHVs);
    VPUNNStrategy.nTiles = static_cast<unsigned int>(nTiles);
    VPUNNStrategy.prefetching = prefetching;

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
    case VPU::MultiClusterStrategy::HKSwitch:
    // TODO:[E-122321] Investigate if VPUNN Cost Model supports multiple batch query.
    // As a workaround, we set SOB MC to SOH tiling strategy for now.
    case VPU::MultiClusterStrategy::SplitOverBatch:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH_Overlapped;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        VPUNNStrategy.tiling_strategy = getSOKLayerStrategy(distributionMode, arch);
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::Clustering:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOW;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::SplitOverHeightKernel:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOHK;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::SplitOverHeightWidth:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOHW;
        return VPUNNStrategy;
    // TODO: [E-126102] Cost model for Grouped MatMul
    case VPU::MultiClusterStrategy::SplitOverGroup:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;
        return VPUNNStrategy;
    default:
        VPUX_THROW("Unsupported cluster-tiling strategy: '{0}' in VPUNN", strategy);
    }
}

VPUNN::DPULayer vpux::VPU::getDPULayer(const VPUIP::WorkloadCostParams& params) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "Kernel array size less than 2");
    const unsigned int KY = checked_cast<unsigned int>(params.kernelSize[Dims4D::Kernel::Y.ind()]);
    const unsigned int KX = checked_cast<unsigned int>(params.kernelSize[Dims4D::Kernel::X.ind()]);

    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "Kernel stride array size less than 2");
    const unsigned int SY = checked_cast<unsigned int>(params.kernelStride[Dims4D::Strides::Y.ind()]);
    const unsigned int SX = checked_cast<unsigned int>(params.kernelStride[Dims4D::Strides::X.ind()]);

    const auto opType = getOperationType(params.nceTaskType);

    const auto outputTensor = VPU::getVPUTensor(params.outputShape, params.outDataType);
    const auto inputTensor = VPU::getVPUTensor(params.inputShape, params.inDataType);

    auto vpunnLayer = VPUNN::DPULayer(
            getVPUDeviceType(params.arch), opType, {inputTensor}, {outputTensor}, {KX, KY}, {SX, SY},
            {static_cast<unsigned int>(params.padInfo.top), static_cast<unsigned int>(params.padInfo.bottom),
             static_cast<unsigned int>(params.padInfo.left), static_cast<unsigned int>(params.padInfo.right)});
    vpunnLayer.set_weight_sparsity(params.isWeightsSparsityEnabled, params.weightsSparsityRatio);

    if (params.weightsDataType.has_value()) {
        vpunnLayer.weight_type = getVPUNNElementType(params.weightsDataType.value());
    }

    // set superdense
    if (vpux::VPU::NCESparsity::isSuperdenseRequired(params.outOrder, params.outputShape, params.outDataType)) {
        vpunnLayer.superdense_memory = true;
    }

    return vpunnLayer;
}

/// @brief Build VPUNN DPUWorkload
/// @param tileParams WorkloadCostParams inputShape & outputShape items are per tile.
/// @param wl A workload
/// @return VPUNN DPUWorkload
VPUNN::DPUWorkload vpux::VPU::getDPUWorkload(const VPUIP::WorkloadCostParams& tileParams,
                                             const VPUIP::WorkloadTile& wl) {
    VPUX_THROW_WHEN(tileParams.kernelSize.size() < 2, "Kernel array size less than 2");
    const auto KY = tileParams.kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = tileParams.kernelSize[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(tileParams.kernelStride.size() < 2, "Kernel stride array size less than 2");
    const auto SY = tileParams.kernelStride[Dims4D::Strides::Y.ind()];
    const auto SX = tileParams.kernelStride[Dims4D::Strides::X.ind()];

    const auto opType = getOperationType(tileParams.nceTaskType);

    const auto& outputTile = std::get<0>(wl);
    const auto mpeMode = std::get<1>(wl);

    auto padsTileConf = backInferPadsTile(outputTile, tileParams.fullInputShape, tileParams.padInfo, ArrayRef({KY, KX}),
                                          ArrayRef({SY, SX}));

    const auto OW = outputTile.shape[Dims4D::Act::W];
    const auto OH = outputTile.shape[Dims4D::Act::H];
    auto OC = outputTile.shape[Dims4D::Act::C];
    const auto ON = outputTile.shape[Dims4D::Act::N];

    const auto IW = (OW - 1) * SX + KX - padsTileConf.left - padsTileConf.right;
    const auto IH = (OH - 1) * SY + KY - padsTileConf.top - padsTileConf.bottom;
    auto IC = tileParams.nceTaskType == VPUIP::NCETaskType::CONV ? tileParams.inputShape[Dims4D::Act::C] : OC;
    const auto IN = ON;

    auto inputTensorShape = Shape({IN, IC, IH, IW});
    auto outputTensorShape = Shape({ON, OC, OH, OW});

    // [VPUNN error code fix] Correct pad and align compute shape for NCE.Permute workloads
    if (tileParams.nceTaskType == VPUIP::NCETaskType::ELTWISE &&
        tileParams.oduPermutation == VPUIPDPU::ODUPermuteDataMode::PERMUTE_YZX) {
        // Bottom_pad is for output channel alignment(align to 4/16) in NCE Permute
        // workloads. We don't need it in final workloads and must set zero before passing to VPUNN
        padsTileConf.bottom = 0;

        // IC is the true compute shape for NCE Permute workloads
        // Need keep OC == IC for eltwise workloads check in VPUNN
        // E.g., nce permute : in {6, 120, 640} out {16, 120, 640}. The real OC = IC = 6
        IC = tileParams.inputShape[Dims4D::Act::C];
        OC = IC;

        // Correct input and output compute shape for NCE.Permute workloads
        // In this case the input&output layouts are NCHW->NHWC. We need to use the shape casting
        // to NHWC->NWCH for VPUNN cost calculation.
        if (tileParams.inOrder == DimsOrder::NCHW && tileParams.outOrder == DimsOrder::NHWC) {
            inputTensorShape[Dims4D::Act::C] = IW;
            inputTensorShape[Dims4D::Act::H] = IC;
            inputTensorShape[Dims4D::Act::W] = IH;
            outputTensorShape[Dims4D::Act::C] = OW;
            outputTensorShape[Dims4D::Act::H] = OC;
            outputTensorShape[Dims4D::Act::W] = OH;
        }
    }

    // TODO: Input and output VPUTensor need set corresponding layout & activation sparsity fields once VPUNN support
    // them. See ticket E#89715 & E#90004
    const auto inputTensor = getVPUTensor(inputTensorShape, tileParams.inDataType);
    const auto outputTensor = getVPUTensor(outputTensorShape, tileParams.outDataType, tileParams.oduPermutation);

    VPUNN::DPUWorkload vpunnDPUWorkload{
            getVPUDeviceType(tileParams.arch),
            opType,
            {inputTensor},
            {outputTensor},
            {static_cast<unsigned int>(KX), static_cast<unsigned int>(KY)},
            {static_cast<unsigned int>(SX), static_cast<unsigned int>(SY)},
            {static_cast<unsigned int>(padsTileConf.top), static_cast<unsigned int>(padsTileConf.bottom),
             static_cast<unsigned int>(padsTileConf.left), static_cast<unsigned int>(padsTileConf.right)},
            getExecutionMode(mpeMode)};

    vpunnDPUWorkload.weight_sparsity_enabled = tileParams.isWeightsSparsityEnabled;
    vpunnDPUWorkload.weight_sparsity = tileParams.weightsSparsityRatio;

    if (tileParams.weightsDataType.has_value()) {
        vpunnDPUWorkload.weight_type = getVPUNNElementType(tileParams.weightsDataType.value());
    }

    auto getISIStrategy = [&](VPU::MultiClusterStrategy layerStrategy) {
        if (layerStrategy == VPU::MultiClusterStrategy::HKSwitch) {
            if (tileParams.arch >= VPU::ArchKind::NPU40XX) {
                layerStrategy = VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
            } else {
                layerStrategy = VPU::MultiClusterStrategy::SplitOverHeight;
            }
        }

        switch (layerStrategy) {
        // Tile_input which has halos need map to SPLIT_OVER_H
        case VPU::MultiClusterStrategy::SplitOverHeight:
            return VPUNN::ISIStrategy::SPLIT_OVER_H;
        case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
        case VPU::MultiClusterStrategy::Clustering:
            return VPUNN::ISIStrategy::CLUSTERING;
        case VPU::MultiClusterStrategy::SplitOverKernel:
            return VPUNN::ISIStrategy::SPLIT_OVER_K;
        default:
            VPUX_THROW("Unsupported strategy {0} to convert to ISI_Strategy", layerStrategy);
        }
    };
    vpunnDPUWorkload.isi_strategy = getISIStrategy(tileParams.layerStrategy);

    if (tileParams.layerStrategy == MultiClusterStrategy::SplitOverKernel ||
        tileParams.layerStrategy == MultiClusterStrategy::HKSwitch) {
        // Assign actual used tiles in parent layer especially for SOK
        vpunnDPUWorkload.output_write_tiles = checked_cast<unsigned int>(tileParams.numTiles);
    }

    // set activation
    if (tileParams.ppeAttr != nullptr) {
        vpunnDPUWorkload.activation_function = getVPUNNActivationFunction(tileParams.ppeAttr);
    }

    // set superdense
    if (vpux::VPU::NCESparsity::isSuperdenseRequired(tileParams.outOrder, tileParams.outputShape,
                                                     tileParams.outDataType)) {
        vpunnDPUWorkload.superdense_memory = true;
    }

    return vpunnDPUWorkload;
}

VPUIP::WorkloadCostParams vpux::VPU::getWorkloadCostParam(VPU::NCEOpInterface nceOp, VPU::ArchKind arch, int64_t numDPU,
                                                          int64_t numTiles) {
    const auto inputType = nceOp->getOperand(0).getType().cast<NDTypeInterface>();
    const auto outputType = nceOp->getResult(0).getType().cast<NDTypeInterface>();
    const auto inElemType = inputType.getElementType();
    const auto outElemType = outputType.getElementType();

    const auto inputOrder = inputType.getDimsOrder();
    const auto outputOrder = outputType.getDimsOrder();

    const auto inputShape = inputType.getShape();
    const auto outputShape = outputType.getShape();

    const auto pads = nceOp.getPad();

    VPUIP::WorkloadCostParams params = {};
    params.inDataType = inElemType;
    params.outDataType = outElemType;
    if (nceOp.getWeightsOperand() != nullptr) {
        params.weightsDataType = nceOp.getWeightsOperand().getType().cast<NDTypeInterface>().getElementType();
    }
    params.inOrder = inputOrder;
    params.outOrder = outputOrder;
    params.numDPU = numDPU;
    params.numTiles = numTiles;
    params.arch = arch;
    params.fullInputShape = inputShape.raw();
    params.inputShape = inputShape.raw();
    params.outputShape = outputShape.raw();
    params.padInfo = VPU::toPadInfo(pads);
    params.kernelSize = nceOp.getKernelSizeVal();
    params.kernelStride = nceOp.getStridesVal();
    params.weightsSparsityRatio = 0;
    params.isWeightsSparsityEnabled = false;

    // set ppe for workload activation
    params.ppeAttr = nceOp.getPPE();

    // set MC strategy
    auto op = nceOp.getOperation();
    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        auto strategy = clusteredOp.getMultiClusterStrategy();

        if (strategy.has_value()) {
            params.layerStrategy = strategy.value();
        } else if (op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
            // It shows this is a cluster tiling op and its MC strategy attribute has been removed
            // We need judge it from the input/ output distributed mode
            auto clusterOp = op->getParentOfType<VPU::NCEClusterTilingOp>();
            auto inputType = (*clusterOp.getOperands().begin()).getType().cast<VPU::DistributedTypeInterface>();
            auto outputType = (*clusterOp.getResults().begin()).getType().cast<VPU::DistributedTypeInterface>();
            auto distributedInput = inputType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
            auto distributedOutput = outputType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
            VPUX_THROW_WHEN(
                    distributedInput == nullptr || distributedOutput == nullptr,
                    "Input or output type should be DistributedTensorType but got input type - {0}, output type - {1}",
                    inputType, outputType);
            auto distributionInAttr = distributedInput.getDistribution();
            auto distributionOutAttr = distributedOutput.getDistribution();
            SmallVector<int64_t> numTilesIn = {1, 1, 1, 1}, numTilesOut = {1, 1, 1, 1};
            // DUPLICATED tensor has no numTiles item
            if (distributionInAttr.getNumTiles() != nullptr) {
                numTilesIn = vpux::parseIntArrayAttr<int64_t>(distributionInAttr.getNumTiles());
            }
            if (distributionOutAttr.getNumTiles() != nullptr) {
                numTilesOut = vpux::parseIntArrayAttr<int64_t>(distributionOutAttr.getNumTiles());
            }
            auto modeIn = distributionInAttr.getMode().getValue();
            auto modeOut = distributionOutAttr.getMode().getValue();

            // Consider SOK on DW conv ops, the modes may also be SEGMENTED
            // We need distinguish it with numTiles.
            if (modeIn == VPU::DistributionMode::SEGMENTED && modeOut == VPU::DistributionMode::SEGMENTED &&
                (numTilesIn[Dims4D::Act::H.ind()] > 1)) {
                params.layerStrategy = VPU::MultiClusterStrategy::SplitOverHeight;
            } else if (modeIn == VPU::DistributionMode::OVERLAPPED) {
                // Set SplitOverHeightOverlapped to be different from SplitOverHeight for VPUNN even on NPU40XX
                params.layerStrategy = VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
            } else if (modeOut == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
                params.layerStrategy = VPU::MultiClusterStrategy::HKSwitch;
            } else if (modeOut == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED) ||
                       (numTilesOut[Dims4D::Act::C.ind()] > 1)) {
                params.layerStrategy = VPU::MultiClusterStrategy::SplitOverKernel;
            }
        }
    }

    // Considering weights sparsity. For CONV, DW_CONV ops
    const auto weights = nceOp.getWeightsOperand();
    if (weights != nullptr && weights.getType().isa<VPU::SparseTensorType>()) {
        params.weightsSparsityRatio = getWeightsSparsityRatio(weights);
        params.isWeightsSparsityEnabled = true;
    }

    llvm::TypeSwitch<mlir::Operation*, void>(nceOp.getOperation())
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::DWCONV;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::MAXPOOL;
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::AVEPOOL;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
            })
            .Case<VPU::NCEInterpolateOp>([&](VPU::NCEInterpolateOp) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
            })
            .Case<VPU::NCEMatMulOp>([&](auto) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
            })
            .Case<VPU::NCEReduceOp>([&](VPU::NCEReduceOp origOp) {
                params.nceTaskType = VPU::configureNCEReduceTaskType(origOp);
            })
            // Only for VPUNN L1 DPU API
            // For L2 API, the strategy SOW is not supported by VPUNN, refer to #86188
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
                params.oduPermutation = VPUIPDPU::ODUPermuteDataMode::PERMUTE_YZX;
            })
            .Default([](mlir::Operation* op) {
                VPUX_THROW("Unsupported NCE operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
    return params;
}
