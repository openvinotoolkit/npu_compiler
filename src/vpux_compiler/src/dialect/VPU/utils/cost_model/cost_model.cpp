//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>

#include <vpu/layer.h>
#include <vpu_layer_strategy.h>

using namespace vpux;

bool vpux::VPU::hasVPUNNPreSplit(mlir::Operation* op) {
    return VPU::getConstraint<bool>(op, VPU::VPUNN_PRE_SPLIT);
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

///@brief Print vpunn layers
void vpux::VPU::printVPUNNLayers(ArrayRef<VPUNN::DPULayer> layers, vpux::Logger log) {
    for (auto& layer : layers) {
        std::ostringstream layerStream;
        layerStream << layer;
        log.warning("[VPUNN LOG] Layer config: {0}", layerStream.str());
    }
}

/// @brief Print vpunn dpu workload for debug
/// @warning Default logCb is Trace level
void vpux::VPU::printVPUNNWorkloadConfig(const VPUNN::DPUWorkload& wl, LogCb logCb) {
    std::ostringstream wlStream;
    wlStream << wl;
    logCb(formatv("[VPUNN LOG] DPU workload config: {0}", wlStream.str()));
}

///@brief Print vpunn workload split info
void vpux::VPU::printLayerSplitInfo(const VPUNN::LayerSplitInfo& info, const Logger& log) {
    log.trace("[VPUNN LOG] split info of size {0}", info.size());
    for (auto item : info) {
        auto workloadCost = item.best_intra_tile_split;
        log.nest(1).trace("[VPUNN LOG] cost {0} with MPEMode {1}", checkAndReturnCost(workloadCost.first, log),
                          getMPEMode(workloadCost.second[0].execution_order));
        for (auto perClusterSplit : workloadCost.second) {
            log.nest(2).trace("[VPUNN LOG] split offsets {0}", perClusterSplit.offsets);
            log.nest(2).trace("[VPUNN LOG] split shape {0}", perClusterSplit.outputs[0].get_shape());
        }
    }
}

///@brief Map VPU::MPEMode from VPUNN::ExecutionMode
VPU::MPEMode vpux::VPU::getMPEMode(VPUNN::ExecutionMode executionMode) {
    switch (executionMode) {
    case VPUNN::ExecutionMode::VECTOR:
        return MPEMode::VECTOR;
    case VPUNN::ExecutionMode::MATRIX:
        return MPEMode::MATRIX;
    case VPUNN::ExecutionMode::VECTOR_FP16:
        return MPEMode::VECTOR_FP16;
    case VPUNN::ExecutionMode::CUBOID_16x16:
        return MPEMode::CUBOID_16x16;
    case VPUNN::ExecutionMode::CUBOID_8x16:
        return MPEMode::CUBOID_8x16;
    case VPUNN::ExecutionMode::CUBOID_4x16:
        return MPEMode::CUBOID_4x16;
    default:  // do not handle __size
        return MPEMode::NOP;
    }
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
    const auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(weights.getType());
    VPUX_THROW_WHEN(sparseType == nullptr, "Not a sparse type");
    const auto sparsityCompressionAttr = sparseType.getSparsityCompression();
    VPUX_THROW_WHEN(sparsityCompressionAttr == nullptr, "sparsity_compressionAttr shouldn't be a nullptr");

    auto log = vpux::Logger("[calculate-sparstiy-ratio-vpunn]", LogLevel::None);
    log.trace("Calculate weights sparsity ratio for Weights {0}", weights.getLoc());
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
    auto elemType = weightsType.getElementType();
    auto compressedSize = sparsityCompressionAttr.getAllocSize(elemType).count();

    auto weightsSparsityRatio = getWeightsSparsityRatio(weightsType, compressedSize);

    log.trace(" Sparsity ratio: {0}", weightsSparsityRatio);
    return weightsSparsityRatio;
}

VPUNN::VPUDevice vpux::VPU::getVPUDeviceType(config::ArchKind archKind) {
    switch (archKind) {
    case config::ArchKind::NPU37XX:
        return VPUNN::VPUDevice::VPU_2_7;
    case config::ArchKind::NPU40XX:
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
    } else if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        if (qType.getStorageTypeIntegralWidth() == 8) {
            return true;
        } else if (qType.getStorageTypeIntegralWidth() == 4) {
            // Temporary enablement; follow up E#103211
            return true;
        }
    } else if (type.isFloat8E5M2()) {  // FP8
        return true;
    } else if (type.isFloat8E4M3FN()) {  // HF8
        return true;
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
    } else if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        auto storageType = qType.getStorageType();
        if (storageType.isFloat8E5M2()) {
            return VPUNN::DataType::BF8;
        } else if (storageType.isFloat8E4M3FN()) {
            return VPUNN::DataType::HF8;
        }

        if (qType.getStorageTypeIntegralWidth() == 8) {
            return qType.isSigned() ? VPUNN::DataType::INT8 : VPUNN::DataType::UINT8;
        } else if (qType.getStorageTypeIntegralWidth() == 4) {
            return qType.isSigned() ? VPUNN::DataType::INT4 : VPUNN::DataType::UINT4;
        } else if (qType.getStorageTypeIntegralWidth() == 2) {
            return qType.isSigned() ? VPUNN::DataType::INT2 : VPUNN::DataType::UINT2;
            // To do: provide proper cost for I16/U16: #E-160697
        } else if (qType.getStorageTypeIntegralWidth() == 16) {
            return VPUNN::DataType::FLOAT16;
        }
    } else if (type.isF32()) {
        return VPUNN::DataType::FLOAT32;
    } else if (type.isFloat8E5M2()) {
        return VPUNN::DataType::BF8;
    } else if (type.isFloat8E4M3FN()) {
        return VPUNN::DataType::HF8;
    }

    return std::nullopt;
}

VPUNN::Layout vpux::VPU::getVPUNNLayout(vpux::DimsOrder vpuxLayout) {
    if (vpuxLayout == vpux::DimsOrder::NHWC || vpuxLayout == vpux::DimsOrder::GNHWC) {
        return VPUNN::Layout::ZXY;
    } else if (vpuxLayout == vpux::DimsOrder::NWHC) {
        return VPUNN::Layout::ZYX;
    } else if (vpuxLayout == vpux::DimsOrder::NWCH) {
        return VPUNN::Layout::YZX;
    } else if (vpuxLayout == vpux::DimsOrder::NCWH) {
        return VPUNN::Layout::YXZ;
    } else if (vpuxLayout == vpux::DimsOrder::NHCW) {
        return VPUNN::Layout::XZY;
    } else if (vpuxLayout == vpux::DimsOrder::NCHW) {
        return VPUNN::Layout::XYZ;
    } else {
        Logger::global().warning("Unsupported vpux layout '{0}' is detected, use default VPUNN Layout 'ZXY'",
                                 vpuxLayout);
    }

    return VPUNN::Layout::ZXY;
}

VPUNN::VPUTensor vpux::VPU::getVPUTensor(ShapeRef shape, mlir::Type elemType, DimsOrder layout) {
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
                nnType.value(), getVPUNNLayout(layout));
    } else if (shape.size() == 4) {
        return VPUNN::VPUTensor(
                {
                        static_cast<unsigned int>(shape[Dims4D::Act::W]),
                        static_cast<unsigned int>(shape[Dims4D::Act::H]),
                        static_cast<unsigned int>(shape[Dims4D::Act::C]),
                        static_cast<unsigned int>(shape[Dims4D::Act::N]),
                },
                nnType.value(), getVPUNNLayout(layout));
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
                                                    vpux::config::ArchKind arch) {
    if (distributionMode == vpux::VPU::DistributionMode::SEGMENTED && arch > vpux::config::ArchKind::NPU40XX) {
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
                                                       config::ArchKind arch, size_t nSHVs, bool prefetching,
                                                       DistributionMode distributionMode, mlir::Operation* op) {
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
        VPUNNStrategy.tiling_strategy = mlir::isa_and_nonnull<VPU::NCEPermuteOp>(op)
                                                ? VPUNN::VPUTilingStrategy::SOW
                                                : VPUNN::VPUTilingStrategy::SOH_Overlapped;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        VPUNNStrategy.tiling_strategy = mlir::isa_and_nonnull<VPU::NCEPermuteOp>(op)
                                                ? VPUNN::VPUTilingStrategy::SOH_Overlapped
                                                : getSOKLayerStrategy(distributionMode, arch);
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

void correctParamsForNcePermute(Shape& inputShape, Shape& outputShape, PadInfo& padding) {
    // Bottom_pad is for output channel alignment(align to 4/16) in NCE Permute
    // workloads. We don't need it in final workloads and must set zero before passing to VPUNN
    padding.bottom = 0;

    // IC is the true compute shape for NCE Permute workloads
    // Need keep OC == IC for eltwise workloads check in VPUNN
    // E.g., nce permute : in {6, 120, 640} out {16, 120, 640}. The real OC = IC = 6
    auto IH = inputShape[Dims4D::Act::H];
    auto IW = inputShape[Dims4D::Act::W];
    auto IC = inputShape[Dims4D::Act::C];

    auto OH = outputShape[Dims4D::Act::H];
    auto OW = outputShape[Dims4D::Act::W];
    auto OC = IC;

    // Correct input and output compute shape for NCE.Permute workloads
    // The original input&output layouts are NCHW->NHWC. We need to use the shape casting
    // to NHWC->NWCH for VPUNN cost calculation.
    inputShape[Dims4D::Act::C] = IW;
    inputShape[Dims4D::Act::H] = IC;
    inputShape[Dims4D::Act::W] = IH;
    outputShape[Dims4D::Act::C] = OW;
    outputShape[Dims4D::Act::H] = OC;
    outputShape[Dims4D::Act::W] = OH;
}

void setAutopadFields(VPUNN::DPUWorkload& workload, const VPUIP::WorkloadCostParams& params) {
    // The workloads that use the IDU / ODU autopad features must be explicitly marked for VPUNN to correctly calculate
    // their cost.
    // Note: Compressed Convolutions are an alternative way to avoid padding the input channels to 16, by only
    // padding them to 4. For these workloads, VPUNN does not expect the IDU autopad to be marked as enabled
    const auto usesIDUAutopad =
            workload.inputs[0].z() < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT && !params.isNceCompressConv;
    const auto usesODUAutopad = workload.outputs[0].z() < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    if (usesIDUAutopad) {
        workload.input_autopad = true;
    }
    if (usesODUAutopad) {
        workload.output_autopad = true;
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

    auto padsConf = params.padInfo;

    const auto OW = params.outputShape[Dims4D::Act::W];
    const auto OH = params.outputShape[Dims4D::Act::H];
    auto OC = params.outputShape[Dims4D::Act::C];
    const auto ON = params.outputShape[Dims4D::Act::N];

    const auto IW = (OW - 1) * SX + KX - padsConf.left - padsConf.right;
    const auto IH = (OH - 1) * SY + KY - padsConf.top - padsConf.bottom;
    auto IC = params.inputShape[Dims4D::Act::C];
    const auto IN = ON;

    auto inputTensorShape = Shape({IN, IC, IH, IW});
    auto outputTensorShape = Shape({ON, OC, OH, OW});

    // [VPUNN error code fix] Correct pad and align compute shape for NCE.Permute workloads
    if (params.isNcePermute) {
        correctParamsForNcePermute(inputTensorShape, outputTensorShape, padsConf);
    }

    const auto outputTensor = VPU::getVPUTensor(outputTensorShape, params.outDataType, params.outOrder);
    const auto inputTensor = VPU::getVPUTensor(inputTensorShape, params.inDataType, params.inOrder);

    auto vpunnLayer =
            VPUNN::DPULayer(getVPUDeviceType(params.arch), opType, {inputTensor}, {outputTensor}, {KX, KY}, {SX, SY},
                            {static_cast<unsigned int>(padsConf.top), static_cast<unsigned int>(padsConf.bottom),
                             static_cast<unsigned int>(padsConf.left), static_cast<unsigned int>(padsConf.right)});

    VPUX_THROW_WHEN(params.isWeightsSparsityEnabled && (params.weightsSparsityRatio == 0.),
                    "Invalid sparsity ratio zero");
    vpunnLayer.set_weight_sparsity(params.isWeightsSparsityEnabled, params.weightsSparsityRatio);

    if (params.weightsDataType.has_value()) {
        vpunnLayer.weight_type = getVPUNNElementType(params.weightsDataType.value());
    }

    if (params.sepInfo.has_value()) {
        vpunnLayer.sep_activators = getSEPModeInfo(params.sepInfo.value());
    }

    // set superdense
    if (vpux::VPU::NCESparsity::isSuperdenseRequired(params.outOrder, params.outputShape, params.outDataType)) {
        vpunnLayer.superdense_memory = true;
    }

    setAutopadFields(vpunnLayer, params);

    return vpunnLayer;
}

std::vector<VPUNN::DPULayer> vpux::VPU::getPerClusterDPULayers(VPU::NCEOpInterface nceOp,
                                                               const VPUIP::WorkloadCostParams& params, Logger log) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "Kernel array size less than 2");
    const auto KY = params.kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = params.kernelSize[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "Kernel stride array size less than 2");
    const auto SY = params.kernelStride[Dims4D::Strides::Y.ind()];
    const auto SX = params.kernelStride[Dims4D::Strides::X.ind()];

    const auto opType = getOperationType(params.nceTaskType);

    const auto getPerClusterShapes = [&](VPU::DistributedTensorType distributedType,
                                         bool isOutput = false) -> SmallVector<Shape> {
        if (distributedType != nullptr) {
            // For output tensor, compute shape is required to get correct shapes for computation
            // For input tensor, memory shape is required to ignore HALO region
            return isOutput ? distributedType.getPerClusterComputeShapes()
                            : distributedType.getPerClusterMemoryShapes();
        }
        return isOutput ? SmallVector({params.outputShape}) : SmallVector({params.inputShape});
    };

    // OutputTensors and InputTensors
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp == nullptr) {
        return std::vector<VPUNN::DPULayer>({getDPULayer(params)});
    }
    auto outputDistributedType = getDistributedTensor(clusteredOp->getResult(0));
    auto actInputDistributedType = getDistributedTensor(clusteredOp->getOperand(0));
    if (outputDistributedType == nullptr || actInputDistributedType == nullptr) {
        // When the distributedTypes are not created, generate distributedTypes from strategy
        auto strategy = params.layerStrategy;
        auto numClusters = params.numTiles;
        const auto offsets = Shape(params.outputShape.size(), 0);
        auto outType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
        if (mlir::isa<VPU::SparseTensorType, VPUIP::SparseBufferType>(outType)) {
            outType = mlir::cast<vpux::NDTypeInterface>(getEffectiveSparseOutputType(outType));
        }
        outType = outType.extractDenseTile(offsets, params.outputShape);
        outputDistributedType = mlir::cast<VPU::DistributedTensorType>(
                getDistributedOutputTypeFromOp(clusteredOp, outType, numClusters, strategy));
        auto inType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        if (mlir::isa<VPU::SparseTensorType, VPUIP::SparseBufferType>(inType)) {
            inType = mlir::cast<vpux::NDTypeInterface>(getEffectiveSparseOutputType(inType));
        }
        inType = inType.extractDenseTile(offsets, params.inputShape);
        actInputDistributedType = mlir::cast<VPU::DistributedTensorType>(getDistributedActivationTypeFromOp(
                clusteredOp, clusteredOp->getOperand(0), inType, numClusters, strategy));
    }
    auto outputPerClusterShapes = getPerClusterShapes(outputDistributedType, true);
    auto actInputPerClusterShapes = getPerClusterShapes(actInputDistributedType);
    auto outputPerClusterPaddings = SmallVector<PadInfo>(outputPerClusterShapes.size(), params.padInfo);
    const auto numClusters = outputPerClusterShapes.size();

    if (outputDistributedType.getDistribution().getMode().getValue() != VPU::DistributionMode::DUPLICATED) {
        auto outputPerClusterOffsets = outputDistributedType.getPerClusterComputeShapeOffsets();
        auto numTiles = vpux::parseIntArrayAttr<int64_t>(outputDistributedType.getDistribution().getNumTiles());
        auto numTilesShape = Shape(numTiles);

        for (auto index : irange(numClusters)) {
            TileInfo outputTile(outputPerClusterShapes[index], outputPerClusterOffsets[index], numTilesShape);
            auto padsTileConf = backInferPadsTile(outputTile, params.fullInputShape, params.padInfo, ArrayRef({KY, KX}),
                                                  ArrayRef({SY, SX}));
            outputPerClusterPaddings[index] = padsTileConf;
        }
    }

    auto adjustInputOutputShape = [&]() {
        // For non-conv operations, the compute IC equal to OC
        // Input memory shapes' IC might be bigger than actual compute IC because of possible broadcast
        const auto useOCAsIC =
                (!mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEPermuteOp>(
                        nceOp.getOperation())) &&
                (outputPerClusterShapes[0][Dims4D::Act::C] != actInputPerClusterShapes[0][Dims4D::Act::C]) &&
                !canAutopadOutput(nceOp.getOperation());
        const auto inputMode = actInputDistributedType.getDistribution().getMode().getValue();
        // Only have memory shapes for input
        // Memory shapes could be bigger than actual compute shapes because of possible broadcast
        // Back infer input shapes from output shapes for these cases
        const auto useBackInferredHW =
                inputMode == VPU::DistributionMode::OVERLAPPED || inputMode == VPU::DistributionMode::DUPLICATED;
        for (auto clusterId : irange(outputPerClusterShapes.size())) {
            if (useOCAsIC) {
                actInputPerClusterShapes[clusterId][Dims4D::Act::C] = outputPerClusterShapes[clusterId][Dims4D::Act::C];
            }
            if (useBackInferredHW) {
                actInputPerClusterShapes[clusterId][Dims4D::Act::W] =
                        (outputPerClusterShapes[clusterId][Dims4D::Act::W] - 1) * SX + KX -
                        outputPerClusterPaddings[clusterId].left - outputPerClusterPaddings[clusterId].right;
                actInputPerClusterShapes[clusterId][Dims4D::Act::H] =
                        (outputPerClusterShapes[clusterId][Dims4D::Act::H] - 1) * SY + KY -
                        outputPerClusterPaddings[clusterId].top - outputPerClusterPaddings[clusterId].bottom;
            }

            // [VPUNN error code fix] Correct pad and align compute shape for NCE.Permute workloads
            if (params.isNcePermute) {
                auto adjustActInputShape = actInputPerClusterShapes[clusterId];
                auto adjustOutputShape = outputPerClusterShapes[clusterId];
                auto adjustPadding = outputPerClusterPaddings[clusterId];
                correctParamsForNcePermute(adjustActInputShape, adjustOutputShape, adjustPadding);
                outputPerClusterPaddings[clusterId] = std::move(adjustPadding);
                actInputPerClusterShapes[clusterId] = std::move(adjustActInputShape);
                outputPerClusterShapes[clusterId] = std::move(adjustOutputShape);
            }
        }
    };

    adjustInputOutputShape();

    VPUX_THROW_UNLESS(outputPerClusterShapes.size() == outputPerClusterPaddings.size() &&
                              outputPerClusterShapes.size() == actInputPerClusterShapes.size(),
                      "Invalid per cluster split, shape size {0} but padding size {1}, act input size {2}",
                      outputPerClusterShapes.size(), outputPerClusterPaddings.size(), actInputPerClusterShapes.size());

    log.trace("Split op {0} into {1} clusters", nceOp->getName(), numClusters);

    std::vector<VPUNN::VPUTensor> outputTensors;
    std::vector<VPUNN::VPUTensor> actInputTensors;
    std::vector<PadInfo> outputPaddings;
    outputTensors.reserve(numClusters);
    actInputTensors.reserve(numClusters);
    outputPaddings.reserve(numClusters);

    for (auto index : irange(numClusters)) {
        const auto outputOneClusterShape = outputPerClusterShapes[index];
        auto inputOneClusterShape = actInputPerClusterShapes[index];
        outputTensors.push_back(VPU::getVPUTensor(outputOneClusterShape, params.outDataType, params.outOrder));
        actInputTensors.push_back(VPU::getVPUTensor(inputOneClusterShape, params.inDataType, params.inOrder));
        outputPaddings.push_back(outputPerClusterPaddings[index]);
    }

    std::vector<VPUNN::DPULayer> vpunnLayers;
    vpunnLayers.reserve(numClusters);

    // Set VPUNN layer attributes
    unsigned int outputWriteTiles = 1;  // how many clusters the workload is broadcast to. 1 if no broadcast
    VPUNN::ISIStrategy isiStrategy = getISIStrategyForType(
            outputDistributedType, outputWriteTiles);  // if the workload's output needs broadcast per channel

    for (auto index : irange(numClusters)) {
        auto vpunnLayer =
                VPUNN::DPULayer(getVPUDeviceType(params.arch), opType, {actInputTensors[index]}, {outputTensors[index]},
                                {static_cast<unsigned int>(KX), static_cast<unsigned int>(KY)},
                                {static_cast<unsigned int>(SX), static_cast<unsigned int>(SY)},
                                {static_cast<unsigned int>(outputPerClusterPaddings[index].top),
                                 static_cast<unsigned int>(outputPerClusterPaddings[index].bottom),
                                 static_cast<unsigned int>(outputPerClusterPaddings[index].left),
                                 static_cast<unsigned int>(outputPerClusterPaddings[index].right)});
        // act_sparsity is not set in compiler, because the act input sparsity is unknown to compiler
        // SEP attributes unset. Track E#158943
        // halo not required for accurate cost, but better to have it. Track E#158946
        vpunnLayer.set_weight_sparsity(params.isWeightsSparsityEnabled, params.weightsSparsityRatio);
        vpunnLayer.isi_strategy = isiStrategy;
        vpunnLayer.output_write_tiles = outputWriteTiles;
        auto inputTwoType = nceOp->getNumOperands() > 1 ? nceOp->getOperand(1).getType() : nullptr;
        auto input1Swizzling = getVPUNNSwizzlingKey(actInputDistributedType);
        auto input2Swizzling = getVPUNNSwizzlingKey(inputTwoType);
        vpunnLayer.input_swizzling = {input1Swizzling, input2Swizzling};
        vpunnLayer.output_swizzling = {getVPUNNSwizzlingKey(outputDistributedType)};
        if (params.weightsDataType.has_value()) {
            vpunnLayer.weight_type = getVPUNNElementType(params.weightsDataType.value());
        }
        if (params.ppeAttr != nullptr) {
            vpunnLayer.activation_function = getVPUNNActivationFunction(params.ppeAttr);
        }
        if (VPU::NCESparsity::isSuperdenseRequired(params.outOrder, outputPerClusterShapes[index],
                                                   params.outDataType)) {
            vpunnLayer.superdense_memory = true;
        }

        setAutopadFields(vpunnLayer, params);

        vpunnLayers.push_back(std::move(vpunnLayer));
    }
    return vpunnLayers;
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
    auto IC = tileParams.inputShape[Dims4D::Act::C];
    const auto IN = ON;

    auto inputTensorShape = Shape({IN, IC, IH, IW});
    auto outputTensorShape = Shape({ON, OC, OH, OW});

    // [VPUNN error code fix] Correct pad and align compute shape for NCE.Permute workloads
    if (tileParams.isNcePermute) {
        correctParamsForNcePermute(inputTensorShape, outputTensorShape, padsTileConf);
    }

    // TODO: Input and output VPUTensor need set corresponding layout & activation sparsity fields once VPUNN
    // support them. See ticket E#89715 & E#90004
    const auto inputTensor = getVPUTensor(inputTensorShape, tileParams.inDataType, tileParams.inOrder);
    const auto outputTensor = getVPUTensor(outputTensorShape, tileParams.outDataType, tileParams.outOrder);

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
            if (tileParams.arch >= config::ArchKind::NPU40XX) {
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

    // set sep
    if (tileParams.sepInfo.has_value()) {
        vpunnDPUWorkload.sep_activators = getSEPModeInfo(tileParams.sepInfo.value());
    }

    // set superdense
    if (vpux::VPU::NCESparsity::isSuperdenseRequired(tileParams.outOrder, tileParams.outputShape,
                                                     tileParams.outDataType)) {
        vpunnDPUWorkload.superdense_memory = true;
    }

    setAutopadFields(vpunnDPUWorkload, tileParams);

    return vpunnDPUWorkload;
}

VPUIP::WorkloadCostParams vpux::VPU::getWorkloadCostParam(VPU::NCEOpInterface nceOp, config::ArchKind arch,
                                                          int64_t numDPU, int64_t numTiles) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto inElemType = inputType.getElementType();
    const auto outElemType = outputType.getElementType();

    const auto inputOrder = inputType.getDimsOrder();
    const auto outputOrder = outputType.getDimsOrder();

    const auto inputShape = getBoundedShape(inputType);
    const auto outputShape = getBoundedShape(outputType);

    const auto pads = nceOp.getPad();

    VPUIP::WorkloadCostParams params = {};
    params.inDataType = inElemType;
    params.outDataType = outElemType;
    if (nceOp.getWeightsOperand() != nullptr) {
        params.weightsDataType =
                mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getElementType();
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

    // set sep
    if (VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
        auto input = nceOp->getOperand(0);
        auto inputSparseTensorOp = input.getDefiningOp<VPU::GroupSparseTensorOp>();
        auto inputs = inputSparseTensorOp.getOperands();
        auto sepDataShape = mlir::cast<vpux::NDTypeInterface>((*inputs.begin()).getType()).getShape();
        auto sepTableShape = mlir::cast<vpux::NDTypeInterface>((*std::prev(inputs.end())).getType()).getShape();
        SmallVector<int64_t> sepDataShapeVec(sepDataShape.begin(), sepDataShape.end());
        SmallVector<int64_t> sepTableShapeVec(sepTableShape.begin(), sepTableShape.end());
        params.sepInfo = VPUIP::SEPInfo{vpux::Shape(sepTableShape.begin(), sepTableShape.end()),
                                        vpux::Shape(sepDataShapeVec.begin(), sepDataShapeVec.end())};
    }

    // set MC strategy
    auto op = nceOp.getOperation();
    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        auto strategy = clusteredOp.getMultiClusterStrategy();

        if (strategy.has_value()) {
            params.layerStrategy = strategy.value();
        } else if (hasDistributedTypesIO(op)) {
            // It shows this is a cluster tiling op and its MC strategy attribute has been removed
            // We need judge it from the input/ output distributed mode
            auto inputType = mlir::cast<vpux::VPU::DistributedTypeInterface>((*op->getOperands().begin()).getType());
            auto outputType = mlir::cast<vpux::VPU::DistributedTypeInterface>((*op->getResults().begin()).getType());
            auto distributedInput =
                    mlir::cast<vpux::VPU::DistributedTensorType>(inputType.getDistributedTypes().front());
            auto distributedOutput =
                    mlir::cast<vpux::VPU::DistributedTensorType>(outputType.getDistributedTypes().front());
            VPUX_THROW_WHEN(distributedInput == nullptr || distributedOutput == nullptr,
                            "Input or output type should be DistributedTensorType but got input type - {0}, output "
                            "type - {1}",
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
                // Set SplitOverHeightOverlapped to be different from SplitOverHeight for VPUNN even on VPUX40XX
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
    if (weights != nullptr && mlir::isa<vpux::VPU::SparseTensorType>(weights.getType())) {
        params.weightsSparsityRatio = getWeightsSparsityRatio(weights);
        params.isWeightsSparsityEnabled = true;
    }

    llvm::TypeSwitch<mlir::Operation*, void>(nceOp.getOperation())
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
                params.isNceCompressConv = true;
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
                params.isNcePermute = true;
                // NCEPermuteOp is an intermediate representation of eltwise-add with ODU permute
                // The input layout is NHWC and output layout is NWCH after lowering to eltwise-add
                params.inOrder = DimsOrder::NHWC;
                params.outOrder = DimsOrder::NWCH;
            })
            .Default([](mlir::Operation* op) {
                VPUX_THROW("Unsupported NCE operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
    return params;
}

vpux::VPU::LayerCostModelAnalysis::LayerCostModelAnalysis(mlir::ModuleOp module) {
    auto arch = config::getArch(module);
    _layerCostModel = VPU::CostModelConfig::createLayerCostModel(arch);
}

std::shared_ptr<VPUNN::VPULayerCostModel> vpux::VPU::LayerCostModelAnalysis::getVPUNNLayerCostModel() {
    return _layerCostModel;
}

bool vpux::VPU::LayerCostModelAnalysis::isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&) {
    return !_preserved;
}

void vpux::VPU::LayerCostModelAnalysis::invalidate() {
    _preserved = false;
}

std::shared_ptr<VPUNN::VPULayerCostModel> vpux::VPU::LayerCostModelAnalysis::getOrCreateLayerCostModel(
        std::optional<std::reference_wrapper<vpux::VPU::LayerCostModelAnalysis>> analysis, config::ArchKind arch,
        Logger log) {
    if (analysis.has_value()) {
        log.trace("Load preserved layer cost model");
        return analysis.value().get().getVPUNNLayerCostModel();
    }
    log.warning("Create new layer cost model instance");
    return VPU::CostModelConfig::createLayerCostModel(arch);
}

vpux::VPU::CostModelAnalysis::CostModelAnalysis(mlir::ModuleOp module) {
    auto arch = config::getArch(module);
    _costModel = VPU::CostModelConfig::createCostModel(arch);
}

std::shared_ptr<VPUNN::VPUCostModel> vpux::VPU::CostModelAnalysis::getVPUNNCostModel() {
    return _costModel;
}

bool vpux::VPU::CostModelAnalysis::isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&) {
    return !_preserved;
}

void vpux::VPU::CostModelAnalysis::invalidate() {
    _preserved = false;
}

std::shared_ptr<VPUNN::VPUCostModel> vpux::VPU::CostModelAnalysis::getOrCreateCostModel(
        std::optional<std::reference_wrapper<vpux::VPU::CostModelAnalysis>> analysis, config::ArchKind arch,
        Logger log) {
    if (analysis.has_value()) {
        log.trace("Load preserved cost model");
        return analysis.value().get().getVPUNNCostModel();
    }
    log.warning("Create new cost model instance");
    return VPU::CostModelConfig::createCostModel(arch);
}

vpux::VPU::ICostModelUtilsInterface* vpux::VPU::getICostModelUtilsInterface(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<vpux::VPU::VPUDialect>();
    assert(dialect != nullptr && "VPU Dialect must be present in the context");

    auto iface = dialect->getRegisteredInterface<vpux::VPU::ICostModelUtilsInterface>();
    assert(iface != nullptr && "The requested interface must be registered in the context");
    return iface;
}
