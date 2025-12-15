//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <vpu/dma_types.h>
#include <vpu/dpu_types.h>
#include <vpu/layer.h>
#include <vpu/shave/layers.h>
#include <vpu/vpu_tiling_strategy.h>
#include <vpu_cost_model.h>

#include <bitset>
#include <limits>

using namespace vpux;

VPUNN::VPUTensor getVPUNNTensor(ShapeRef tensorShape, VPUNN::DataType dataType) {
    // Track E#160854. More generic support for 5D tensors
    if (tensorShape.size() >= 4) {
        return VPUNN::VPUTensor({static_cast<unsigned int>(tensorShape[Dims4D::Act::W]),
                                 static_cast<unsigned int>(tensorShape[Dims4D::Act::H]),
                                 static_cast<unsigned int>(tensorShape[Dims4D::Act::C]),
                                 static_cast<unsigned int>(tensorShape[Dims4D::Act::N])},
                                dataType);
    } else {
        return VPUNN::VPUTensor({static_cast<unsigned int>(tensorShape.totalSize()), 1, 1, 1}, dataType);
    }
}

VPUNN::VPUTensor getVPUNNTensorMultiCluster(ArrayRef<Shape> tensorShapes, VPUNN::DataType dataType) {
    unsigned int totalShape = 0;
    for (size_t idx = 0; idx < tensorShapes.size(); idx++) {
        totalShape += static_cast<unsigned int>(tensorShapes[idx].totalSize());
    }
    return VPUNN::VPUTensor({totalShape, 1, 1, 1}, dataType);
}

// This function convert Arch kind to VPUNN VPUDevice directly and faithfully
VPUNN::VPUDevice getVPUNNDevice(config::ArchKind archKind) {
    switch (archKind) {
    case config::ArchKind::NPU37XX:
        return VPUNN::VPUDevice::VPU_2_7;
    case config::ArchKind::NPU40XX:
        return VPUNN::VPUDevice::VPU_4_0;
    case config::ArchKind::NPU50XX:
        return VPUNN::VPUDevice::NPU_5_0;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

VPUNN::DataType getElementType(mlir::Type type, [[maybe_unused]] VPUNN::VPUDevice vpuDevice) {
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
        if (mlir::isa<mlir::Float8E5M2Type>(storageType)) {
            return VPUNN::DataType::BF8;
        } else if (mlir::isa<mlir::Float8E4M3FNType>(storageType)) {
            return VPUNN::DataType::HF8;
        }

        if (qType.getStorageTypeIntegralWidth() == 8) {
            return qType.isSigned() ? VPUNN::DataType::INT8 : VPUNN::DataType::UINT8;
        } else if (qType.getStorageTypeIntegralWidth() == 4 && vpuDevice > VPUNN::VPUDevice::VPU_4_0) {
            // set INT4 only for 50XX+, see E#152912
            return qType.isSigned() ? VPUNN::DataType::INT4 : VPUNN::DataType::UINT4;
        }
    } else if (type.isF32() && vpuDevice > VPUNN::VPUDevice::VPU_4_0) {
        // set FP32 only for 50XX+, see E#158088
        return VPUNN::DataType::FLOAT32;
    } else if (mlir::isa<mlir::Float8E5M2Type>(type)) {
        return VPUNN::DataType::BF8;
    } else if (mlir::isa<mlir::Float8E4M3FNType>(type)) {
        return VPUNN::DataType::HF8;
    }

    // default until support for more types introduced
    return VPUNN::DataType::BFLOAT16;
}

VPUNN::MemoryLocation vpux::getMemoryLocation(mlir::Type type) {
    auto memKind = mlir::cast<vpux::NDTypeInterface>(type).getMemoryKind();
    if (memKind == VPU::MemoryKind::CMX_NN) {
        return VPUNN::MemoryLocation::CMX;
    }

    return VPUNN::MemoryLocation::DRAM;
}

VPUNN::Swizzling vpux::getVPUNNSwizzlingKey(mlir::Type type) {
    SmallVector<VPUNN::Swizzling> swizzlingKeyVPUNN = {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_1,
                                                       VPUNN::Swizzling::KEY_2, VPUNN::Swizzling::KEY_3,
                                                       VPUNN::Swizzling::KEY_4, VPUNN::Swizzling::KEY_5};

    auto swizzlingKey = vpux::getSwizzlingKey(type);
    VPUX_THROW_UNLESS(checked_cast<size_t>(swizzlingKey) < swizzlingKeyVPUNN.size(), "Unsupported swizzling key: '{0}'",
                      swizzlingKey);

    return swizzlingKeyVPUNN[swizzlingKey];
}

VPUNN::ActivationFunction vpux::getVPUNNActivationFunction(VPU::PPEAttr ppeAttr) {
    const auto ppeMode = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterMode>().getMode(ppeAttr);
    const auto clampLow = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterClamp>().getClamps(ppeAttr).first;

    switch (ppeMode) {
    case VPU::PPEMode::LRELU:
        return VPUNN::ActivationFunction::LRELU;
    case VPU::PPEMode::ADD:
        return VPUNN::ActivationFunction::ADD;
    case VPU::PPEMode::SUB:
        return VPUNN::ActivationFunction::SUB;
    case VPU::PPEMode::MULT:
        return VPUNN::ActivationFunction::MULT;
    default:
        if (isDoubleEqual(clampLow, 0)) {
            return VPUNN::ActivationFunction::RELU;
        }
        return VPUNN::ActivationFunction::NONE;
    }
}

namespace {

bool isFullyBroadCastDPUTask(VPUIP::ITIBufferType outType, ArrayRef<int64_t> workloadOutShape) {
    if (outType == nullptr) {
        return false;
    }
    auto outwardHaloRegions = outType.getOutwardHaloRegions();
    if (outwardHaloRegions.empty()) {
        return false;
    }
    for (const auto& attr : outwardHaloRegions) {
        auto outwardHaloRegion = mlir::cast<VPUIP::OutwardHaloRegionAttr>(attr);
        auto outwardHaloRegionShape = parseIntArrayAttr<int64_t>(outwardHaloRegion.getShape());
        if (outwardHaloRegionShape[Dims4D::Act::C.ind()] != workloadOutShape[2] ||
            outwardHaloRegionShape[Dims4D::Act::H.ind()] != workloadOutShape[1] ||
            outwardHaloRegionShape[Dims4D::Act::W.ind()] != workloadOutShape[0]) {
            return false;
        }
    }

    return true;
}

SmallVector<int64_t> getDPUOutputWorkloadSize(ArrayRef<VPUIP::DPUTaskOp> dpuTaskOps) {
    // DPU workload has three dimensions
    SmallVector<int64_t> outEnd(3, std::numeric_limits<int64_t>::min());
    SmallVector<int64_t> outStart(3, std::numeric_limits<int64_t>::max());
    for (auto dpuTaskOp : dpuTaskOps) {
        const auto curOutStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutStart());
        const auto curoutEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutEnd());
        VPUX_THROW_WHEN(outStart.size() != 3 || outEnd.size() != 3, "Unexpected size of outStart/End attributes");
        for (auto i : irange(3)) {
            outStart[i] = std::min(outStart[i], curOutStart[i]);
            outEnd[i] = std::max(outEnd[i], curoutEnd[i]);
        }
    }
    return {outEnd[0] - outStart[0] + 1, outEnd[1] - outStart[1] + 1, outEnd[2] - outStart[2] + 1};
}

VPUNN::ISIStrategy getVPUNNISIStrategyFromITTOutputType(VPUIP::DPUTaskOp dpuTaskOp, unsigned int& outputWriteTiles) {
    auto nceClusterOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceClusterOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    const auto dpuTaskOps = to_small_vector(nceClusterOp.getVariants().getOps<VPUIP::DPUTaskOp>());
    const auto dpuWorkloadSize = getDPUOutputWorkloadSize(dpuTaskOps);
    auto outputITTType = mlir::dyn_cast<VPUIP::ITIBufferType>(nceClusterOp->getResult(0).getType());
    auto isFullyBroadCasted = isFullyBroadCastDPUTask(outputITTType, dpuWorkloadSize);

    if (!isFullyBroadCasted) {
        outputWriteTiles = 1;
        return VPUNN::ISIStrategy::CLUSTERING;
    }
    auto outwardHaloRegions = outputITTType.getOutwardHaloRegions();
    auto inwardHaloRegion = mlir::cast<VPUIP::OutwardHaloRegionAttr>(outwardHaloRegions[0]).getInwardHaloRegions();
    outputWriteTiles = inwardHaloRegion.size() + 1;
    const auto outShape = getShape(nceClusterOp->getResult(0));
    auto isSplitOverC = outShape[Dims4D::Act::C] != dpuWorkloadSize[2];
    return isSplitOverC ? VPUNN::ISIStrategy::SPLIT_OVER_K : VPUNN::ISIStrategy::CLUSTERING;
}

// Keep the original logic of assigning VPUNN ISIStrategy for NPU37XX and NPU40XX because the DPU cost model will not be
// updated for them.
VPUNN::ISIStrategy getVPUNNISIStrategyForNPU40XXAndBelow(VPUIP::DPUTaskOp dpuTaskOp, unsigned int& outputWriteTiles) {
    auto nceClusterOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceClusterOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    VPUNN::ISIStrategy isiStrategy = VPUNN::ISIStrategy::CLUSTERING;
    outputWriteTiles = 1;

    // Check if output is broadcasted to multiple tiles (e.g. HKSwitch and SOK)
    auto outputType = nceClusterOp->getResult(0).getType();
    auto distributedOutput = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);
    if (distributedOutput) {
        const auto distributionAttr = distributedOutput.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED) ||
            mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
            outputWriteTiles = distributionAttr.getNumClusters().getInt();
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        }
    }

    // Check if input is distributed - segmented between multiple tiles (e.g. SOH)
    auto distributedInput = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(nceClusterOp.getParentInput().getType());
    if (distributedInput) {
        const auto distributionAttr = distributedInput.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED) {
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        }
    }

    return isiStrategy;
}

// Fix the logic of assigning VPUNN ISIStrategy for NPU50XX+
// ISIStrategy describes how the dpu output looks like when we need to broadcast it from the current cluster to others
// 1. For SOK the dpu output is not contiguous because C is the innermost dimension, we assign ISIStrategy::SPLIT_OVER_K
// 2. For HKSwitch the dpu output is contiguous because H is the outermost dimension, we assign ISIStrategy::CLUSTERING
// 3. For other cases when there's no broadcast, ISIStrategy is not used actually
VPUNN::ISIStrategy getVPUNNISIStrategyForNPU50XXAndAbove(VPUIP::DPUTaskOp dpuTaskOp, unsigned int& outputWriteTiles) {
    auto nceClusterOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceClusterOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    VPUNN::ISIStrategy isiStrategy = VPUNN::ISIStrategy::CLUSTERING;
    if (auto distributedOutput = mlir::dyn_cast<VPUIP::DistributedBufferType>(nceClusterOp->getResult(0).getType())) {
        const auto outputDistributionAttr = distributedOutput.getDistribution();
        const auto outputMode = outputDistributionAttr.getMode().getValue();
        if (outputMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
            outputWriteTiles = outputDistributionAttr.getNumClusters().getInt();
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        } else if (outputMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
            outputWriteTiles = outputDistributionAttr.getNumClusters().getInt();
            isiStrategy = VPUNN::ISIStrategy::CLUSTERING;
        } else {
            outputWriteTiles = 1;
            isiStrategy = VPUNN::ISIStrategy::CLUSTERING;
        }
        return isiStrategy;
    }
    return getVPUNNISIStrategyFromITTOutputType(dpuTaskOp, outputWriteTiles);
}

bool isConstDeclareOpFilledAllOne(Const::DeclareOp op) {
    const auto content = op.getContent();
    return content.read([](auto values) {
        if (values.size() == 0) {
            return false;
        }

        for (const auto& value : values) {
            if (checked_cast<int>(value) != 1) {
                return false;
            }
        }
        return true;
    });
}
}  // namespace

VPUNN::SEPModeInfo vpux::getSEPModeInfo(VPUIP::SEPInfo sepInfo) {
    const auto getWHCBShape = [](ShapeRef shape) {
        VPUX_THROW_UNLESS(shape.size() == 4, "Shape '{0}' has illegal rank: {1}, expected: 4", shape, shape.size());
        return VPUNN::WHCBTensorShape(
                static_cast<unsigned int>(shape[Dims4D::Act::W]), static_cast<unsigned int>(shape[Dims4D::Act::H]),
                static_cast<unsigned int>(shape[Dims4D::Act::C]), static_cast<unsigned int>(shape[Dims4D::Act::N]));
    };
    return VPUNN::SEPModeInfo{true, getWHCBShape(sepInfo.sepTableShape), getWHCBShape(sepInfo.sepActShape)};
}

VPUNN::DPUWorkload vpux::getDPUWorkload(VPUIP::DPUTaskOp dpuTaskOp, config::ArchKind arch) {
    auto nceClusterOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceClusterOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    auto inputOneType = nceClusterOp->getOperand(0).getType();
    auto outputType = nceClusterOp->getResult(0).getType();
    auto inputTwoType = nceClusterOp->getNumOperands() > 1 ? nceClusterOp->getOperand(1).getType() : nullptr;

    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(inputOneType).getElementType();
    auto inputTwoElemType =
            inputTwoType != nullptr ? mlir::cast<vpux::NDTypeInterface>(inputTwoType).getElementType() : nullptr;
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(outputType).getElementType();

    // CostModel does not support F32/SI32 layers
    // TODO: Support FP32 output element type E#149202
    if (inputElemType.isF32()) {
        VPUX_THROW("Can't convert a F32/SI32 workload as CostModel does not support");
    }
    if (inputElemType.isSignedInteger(32) || outputElemType.isSignedInteger(32)) {
        VPUX_THROW("Can't convert a F32/SI32 workload as CostModel does not support");
    }

    auto input1Swizzling = getVPUNNSwizzlingKey(inputOneType);
    auto input2Swizzling = getVPUNNSwizzlingKey(inputTwoType);
    auto outputSwizzling = getVPUNNSwizzlingKey(outputType);

    VPUNN::ActivationFunction activationFunction = VPUNN::ActivationFunction::NONE;
    auto ppeOps = to_small_vector(nceClusterOp.getPpe().getOps<VPUIP::PPETaskOp>());
    if (!ppeOps.empty()) {
        activationFunction = getVPUNNActivationFunction(ppeOps.front().getPpeAttr());
    }

    unsigned int outputWriteTiles = 1;
    VPUNN::ISIStrategy isiStrategy = arch <= config::ArchKind::NPU40XX
                                             ? getVPUNNISIStrategyForNPU40XXAndBelow(dpuTaskOp, outputWriteTiles)
                                             : getVPUNNISIStrategyForNPU50XXAndAbove(dpuTaskOp, outputWriteTiles);

    bool isWeightsSparsityEnabled = false;
    float weightsSparsityRatio = 0;
    auto weightsSparsityMap = nceClusterOp.getWeightsSparsityMap();
    if (weightsSparsityMap != nullptr && nceClusterOp.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
        isWeightsSparsityEnabled = true;

        auto weightsType = mlir::cast<vpux::NDTypeInterface>(nceClusterOp.getWeights().getType());
        auto weightsElemType = weightsType.getElementType();

        const auto sparsityCompressionAttr = VPUIP::getSparsityCompressionAttr(weightsType);
        VPUX_THROW_WHEN(sparsityCompressionAttr == nullptr, "sparsity_compressionAttr shouldn't be a nullptr");

        auto compressedSize = sparsityCompressionAttr.getAllocSize(weightsElemType).count();
        weightsSparsityRatio = vpux::getWeightsSparsityRatio(weightsType, compressedSize);
    }

    auto isInputSparsityEnabled = (nceClusterOp.getInputSparsityMap() != nullptr);
    // check if SEP data is truly dense or sparse (e.g., sep with 0 padding)
    if ((nceClusterOp.getInputStorageElementTable() != nullptr) && isInputSparsityEnabled) {
        auto declareOp = nceClusterOp.getInputSparsityMap().getDefiningOp<Const::DeclareOp>();
        isInputSparsityEnabled = declareOp && !isConstDeclareOpFilledAllOne(declareOp);
    }
    auto isOutputSparsityEnabled = (nceClusterOp.getOutputSparsityMap() != nullptr);

    auto nceTaskType = nceClusterOp.getTaskType();
    auto opType = getOperationType(nceTaskType);

    int64_t KX = 1, KY = 1;
    int64_t SX = 1, SY = 1;

    if (auto kernelSizeAttr = nceClusterOp.getKernelSizeAttr()) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);
        KX = kernelSize[Dims4D::Kernel::X.ind()];
        KY = kernelSize[Dims4D::Kernel::Y.ind()];
    }

    if (auto kernelStridesAttr = nceClusterOp.getKernelStridesAttr()) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(kernelStridesAttr);
        SX = kernelStrides[Dims4D::Kernel::X.ind()];
        SY = kernelStrides[Dims4D::Kernel::Y.ind()];
    }

    auto mpeMode = dpuTaskOp.getMpeMode();

    const auto paddingAttr = dpuTaskOp.getPad();

    const auto left = paddingAttr.getLeft().getValue().getSExtValue();
    const auto right = paddingAttr.getRight().getValue().getSExtValue();
    const auto top = paddingAttr.getTop().getValue().getSExtValue();
    const auto bottom = paddingAttr.getBottom().getValue().getSExtValue();

    const auto outStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutStart());
    const auto outEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutEnd());

    VPUX_THROW_WHEN(outStart.size() != 3 || outEnd.size() != 3, "Unexpected size of outStart/End attributes");

    // DPUTask workload description is expected to have 3 elements: [W, H, C]
    const int64_t OC = outEnd[2] - outStart[2] + 1;
    const int64_t OH = outEnd[1] - outStart[1] + 1;
    const int64_t OW = outEnd[0] - outStart[0] + 1;

    auto IW = (OW - 1) * SX + KX - left - right;
    auto IH = (OH - 1) * SY + KY - top - bottom;
    auto IC = nceTaskType == VPUIP::NCETaskType::CONV
                      ? mlir::cast<vpux::NDTypeInterface>(inputOneType).getShape()[Dims4D::Act::C]
                      : OC;

    if (dpuTaskOp.getInStart().has_value() && dpuTaskOp.getInEnd().has_value()) {
        const auto inStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getInStart().value());
        const auto inEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getInEnd().value());

        IC = inEnd[2] - inStart[2] + 1;
        IH = inEnd[1] - inStart[1] + 1;
        IW = inEnd[0] - inStart[0] + 1;
    }

    // Set actual IC for compress conv, to pass compute shape to VPUNN
    if (nceClusterOp.getInputChannelsCompressionAttr() != nullptr) {
        if (nceClusterOp.getCmSpPatternAttr() != nullptr) {
            auto cm_sp_pattern = checked_cast<uint16_t>(nceClusterOp.getCmSpPatternAttr().getValue().getSExtValue());
            std::bitset<16> cm_sp_pattern_bits(cm_sp_pattern);
            IC = cm_sp_pattern_bits.count();
        }
    }

    const auto inputOrder = mlir::cast<vpux::NDTypeInterface>(inputOneType).getDimsOrder();
    const auto outputOrder = mlir::cast<vpux::NDTypeInterface>(outputType).getDimsOrder();
    auto inputLayout = vpux::VPU::getVPUNNLayout(inputOrder);
    auto outputLayout = vpux::VPU::getVPUNNLayout(outputOrder);

    // As there's no activation sparsity ratio in compiler, set it to false to assure vpunn sanity check
    // TODO: remove it once activation sparsity ratio is supported in compiler, see E#159669
    isInputSparsityEnabled = false;
    isOutputSparsityEnabled = false;

    const auto inputTensor = VPUNN::VPUTensor(
            {static_cast<unsigned int>(IW), static_cast<unsigned int>(IH), static_cast<unsigned int>(IC), 1},
            getElementType(inputElemType, getVPUNNDevice(arch)), inputLayout, isInputSparsityEnabled);
    const auto outputTensor = VPUNN::VPUTensor(
            {static_cast<unsigned int>(OW), static_cast<unsigned int>(OH), static_cast<unsigned int>(OC), 1},
            getElementType(outputElemType, getVPUNNDevice(arch)), outputLayout, isOutputSparsityEnabled);

    VPUNN::DPUWorkload vpunnDPUWorkload;
    if (inputTwoElemType != nullptr) {
        vpunnDPUWorkload.weight_type = getElementType(inputTwoElemType, getVPUNNDevice(arch));
    }
    vpunnDPUWorkload.device = VPU::getVPUDeviceType(arch);
    vpunnDPUWorkload.op = opType;
    vpunnDPUWorkload.inputs = {inputTensor};
    vpunnDPUWorkload.outputs = {outputTensor};
    vpunnDPUWorkload.kernels = {static_cast<unsigned int>(KX), static_cast<unsigned int>(KY)};
    vpunnDPUWorkload.strides = {static_cast<unsigned int>(SX), static_cast<unsigned int>(SY)};
    vpunnDPUWorkload.padding = {static_cast<unsigned int>(top), static_cast<unsigned int>(bottom),
                                static_cast<unsigned int>(left), static_cast<unsigned int>(right)};
    vpunnDPUWorkload.execution_order = VPU::getExecutionMode(mpeMode);
    vpunnDPUWorkload.activation_function = activationFunction;
    vpunnDPUWorkload.input_swizzling = {input1Swizzling, input2Swizzling};
    vpunnDPUWorkload.output_swizzling = {outputSwizzling};
    vpunnDPUWorkload.output_write_tiles = outputWriteTiles;
    vpunnDPUWorkload.weight_sparsity = weightsSparsityRatio;
    vpunnDPUWorkload.weight_sparsity_enabled = isWeightsSparsityEnabled;
    vpunnDPUWorkload.isi_strategy = isiStrategy;
    vpunnDPUWorkload.superdense_memory = nceClusterOp.getIsSuperdense();

    // set sep info
    if (auto seTable = nceClusterOp.getInputStorageElementTable()) {
        auto shapeVec = mlir::cast<vpux::NDTypeInterface>(inputOneType).getShape().raw();
        auto dataShape = vpux::Shape(shapeVec.begin(), shapeVec.end());
        if (auto distributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputOneType)) {
            auto perClusterMemoryShapes = distributedType.getPerClusterMemoryShapes();
            dataShape = perClusterMemoryShapes[0];
            if (dpuTaskOp.getClusterId().has_value()) {
                auto clusterId = dpuTaskOp.getClusterId().value();
                dataShape = perClusterMemoryShapes[clusterId];
            }
        }
        vpunnDPUWorkload.sep_activators =
                getSEPModeInfo(VPUIP::SEPInfo{vpux::Shape({1, 1, IH, IW}), std::move(dataShape)});
    }

    // The workloads that use the IDU / ODU autopad features must be explicitly marked for VPUNN to correctly calculate
    // their cost.
    // Note: Compressed Convolutions are an alternative way to avoid padding the input channels to 16, by only
    // padding them to 4. For these workloads, VPUNN does not expect the IDU autopad to be marked as enabled
    const auto usesIDUAutopad = vpunnDPUWorkload.inputs[0].z() < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT &&
                                nceClusterOp.getInputChannelsCompressionAttr() == nullptr;
    const auto usesODUAutopad = vpunnDPUWorkload.outputs[0].z() < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    if (usesIDUAutopad) {
        vpunnDPUWorkload.input_autopad = true;
    }
    if (usesODUAutopad) {
        vpunnDPUWorkload.output_autopad = true;
    }

    return vpunnDPUWorkload;
}

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                    config::ArchKind archKind, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                    [[maybe_unused]] int64_t numDMAPorts) {
    auto operandType = innerOperand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);
    auto vpuDevice = VPU::getVPUDeviceType(archKind);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    // Only apply to 50XX, as only arch with accurate DMA cost model
    VPUX_THROW_WHEN(numDMAPorts <= 0, "Invalid number of DMA ports; should be > 0, but actual value is {0}",
                    numDMAPorts);
    if (archKind == config::ArchKind::NPU50XX) {
        size_t cost = 0;
        for (auto shape : perClusterShapes) {
            auto vpuTensorInput = getVPUNNTensor(shape, inElemType);
            auto vpuTensorOutput = getVPUNNTensor(shape, outElemType);
            cost += costModel->DMA(vpuDevice, vpuTensorInput, vpuTensorOutput);
        }
        return cost / numDMAPorts;
    }

    return static_cast<size_t>(costModel->DMA(vpuDevice, {getVPUNNTensorMultiCluster(perClusterShapes, inElemType)},
                                              {getVPUNNTensorMultiCluster(perClusterShapes, outElemType)}));
}

bool extraDMAsRequired(mlir::Value innerOperand) {
    if (auto inputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(innerOperand.getType())) {
        auto distribution = inputType.getDistribution();
        auto distributionMode = distribution.getMode().getValue();
        return distributionMode == VPU::DistributionMode::SEGMENTED ||
               distributionMode == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

size_t vpux::getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    auto inElemType =
            getElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), getVPUNNDevice(archKind));
    auto outElemType =
            getElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), getVPUNNDevice(archKind));

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType) && extraDMAsRequired(input)) {
        return calculateMultiClusterDMACost(input, inElemType, outElemType, archKind, costModel, numDMAPorts);
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType) && extraDMAsRequired(output)) {
        return calculateMultiClusterDMACost(output, inElemType, outElemType, archKind, costModel, numDMAPorts);
    }

    auto inputShape = getShape(input);
    auto outputShape = getShape(output);

    // TODO: add layout info to VPUNN tensors
    auto cost = costModel->DMA(VPU::getVPUDeviceType(archKind), {getVPUNNTensor(inputShape, inElemType)},
                               {getVPUNNTensor(outputShape, outElemType)}, getMemoryLocation(inputType),
                               getMemoryLocation(outputType));

    return static_cast<size_t>(cost);
}

size_t getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "DMA ports is at least one but got {0}", numDMAPorts);
    auto distributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tensorType);
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    auto elemType = tensorType.getElementType();

    SmallVector<Shape> shapes;
    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    // Only apply to 50XX, as only arch with accurate DMA cost model
    if (vpuDevice == VPUNN::VPUDevice::NPU_5_0) {
        shapes = distributedTensorType.getPerClusterMemoryShapes();
        size_t cost = 0;
        for (auto shape : shapes) {
            auto vpuTensor = getVPUNNTensor(shape, getElementType(elemType, vpuDevice));
            cost += costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
        }
        return cost / numDMAPorts;
    }
    if (numDMAPorts > 1) {
        // For distributed segmented DMA, transaction will be split between ports and executing
        // in parallel when there are multiple DMA ports available.
        // When enabling architectures whose number of tiles is not equal to number of DMA ports, using
        // simply the largest size in tiles to calculate cost is not accurate, see E#84432
        shapes.push_back(distributedTensorType.getLargestCompactShape());
    } else {
        shapes = distributedTensorType.getPerClusterComputeShapes();
    }
    auto vpuTensor = getVPUNNTensorMultiCluster(shapes, getElementType(elemType, vpuDevice));
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

size_t getSpillingCostForSegmented(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                   VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                   int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "DMA ports is at least one but got {0}", numDMAPorts);

    auto inDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);
    VPUX_THROW_WHEN(inDistributedTensorType == nullptr || outDistributedTensorType == nullptr, "Invalid type.");

    auto inElemType = inTensorType.getElementType();
    auto outElemType = outTensorType.getElementType();

    auto getShapes = [&](auto distributedType, auto plainType) -> SmallVector<Shape> {
        if (distributedType) {
            // For distributed segmented DMA, transaction will be split between ports and executing
            // in parallel when there are multiple DMA ports available.
            // When enabling architectures whose number of tiles is not equal to number of DMA ports, using
            // simply the largest size in tiles to calculate cost is not accurate, see E#84432
            return (numDMAPorts > 1) ? SmallVector<Shape>{distributedType.getLargestCompactShape()}
                                     : distributedType.getPerClusterComputeShapes();
        }
        return SmallVector<Shape>{plainType.getShape().raw()};
    };
    SmallVector<Shape> inShapes = getShapes(inDistributedTensorType, inTensorType);
    SmallVector<Shape> outShapes = getShapes(outDistributedTensorType, outTensorType);

    auto inTensor = inDistributedTensorType
                            ? getVPUNNTensorMultiCluster(inShapes, getElementType(inElemType, vpuDevice))
                            : getVPUNNTensor(inShapes[0], getElementType(inElemType, vpuDevice));
    auto outTensor = outDistributedTensorType
                             ? getVPUNNTensorMultiCluster(outShapes, getElementType(outElemType, vpuDevice))
                             : getVPUNNTensor(outShapes[0], getElementType(outElemType, vpuDevice));

    return costModel->DMA(vpuDevice, inTensor, outTensor, getMemoryLocation(inTensorType),
                          getMemoryLocation(outTensorType));
}

size_t getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t /*numDMAPorts*/) {
    auto shape = tensorType.getShape();
    auto elemType = tensorType.getElementType();
    auto vpuTensor = getVPUNNTensor(shape, getElementType(elemType, vpuDevice));
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

size_t getSpillingCostForDuplicated(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                    VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                    int64_t /*numDMAPorts*/) {
    auto inVpuTensor =
            getVPUNNTensor(inTensorType.getShape(), getElementType(inTensorType.getElementType(), vpuDevice));
    auto outVpuTensor =
            getVPUNNTensor(outTensorType.getShape(), getElementType(outTensorType.getElementType(), vpuDevice));
    return costModel->DMA(vpuDevice, inVpuTensor, outVpuTensor, getMemoryLocation(inTensorType),
                          getMemoryLocation(outTensorType));
}

using GetDMAOnVPUNN = size_t (*)(vpux::NDTypeInterface tensortType, VPUNN::VPUDevice vpuDevice,
                                 const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetDMAOnVPUNN> spillingCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getSpillingCostForDuplicated},
        {VPU::DistributionMode::SEGMENTED, getSpillingCostForSegmented},
        {VPU::DistributionMode::OVERLAPPED, getSpillingCostForSegmented},
        {VPU::DistributionMode::MULTICASTED, getSpillingCostForDuplicated},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
};

using GetIODMAOnVPUNN = size_t (*)(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                   VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                   int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetIODMAOnVPUNN> spillingIOCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getSpillingCostForDuplicated},
        {VPU::DistributionMode::SEGMENTED, getSpillingCostForSegmented},
        {VPU::DistributionMode::OVERLAPPED, getSpillingCostForSegmented},
        {VPU::DistributionMode::MULTICASTED, getSpillingCostForDuplicated},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
};

// Used by VPU dialect
size_t vpux::getDMACost(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tensorType)) {
        tensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }

    auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tensorType);

    const auto elementType = tensorType.getElementType();

    if (distributedType != nullptr) {
        const auto dmaCostFunc = spillingCostMapVPUNN.at(distributedType.getDistribution().getMode().getValue());
        return dmaCostFunc(tensorType, vpuDevice, costModel, numDMAPorts);
    }

    const auto vpunnTensor = getVPUNNTensor(tensorType.getShape(), getElementType(elementType, vpuDevice));
    return costModel->DMA(vpuDevice, vpunnTensor, vpunnTensor);
}

size_t vpux::getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                        VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                        int64_t numDMAPorts) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(inTensorType)) {
        inTensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(outTensorType)) {
        outTensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }

    const auto inElementType = inTensorType.getElementType();
    const auto outElementType = outTensorType.getElementType();

    auto inDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);

    if (inDistributedType || outDistributedType) {
        auto distributionMode = inDistributedType ? inDistributedType.getDistribution().getMode().getValue()
                                                  : outDistributedType.getDistribution().getMode().getValue();
        const auto dmaCostFunc = spillingIOCostMapVPUNN.at(distributionMode);
        return dmaCostFunc(inTensorType, outTensorType, vpuDevice, costModel, numDMAPorts);
    }

    const auto inVpunnTensor = getVPUNNTensor(inTensorType.getShape(), getElementType(inElementType, vpuDevice));
    const auto outVpunnTensor = getVPUNNTensor(outTensorType.getShape(), getElementType(outElementType, vpuDevice));
    auto cost = costModel->DMA(vpuDevice, inVpunnTensor, outVpunnTensor, getMemoryLocation(inTensorType),
                               getMemoryLocation(outTensorType));
    return static_cast<size_t>(cost);
}

size_t vpux::getDPUCost(mlir::Operation* op) {
    // costs for DPU calculated during workload generation, re-use

    if (op->hasAttr(DPUCost)) {
        auto cost = mlir::cast<mlir::IntegerAttr>(op->getAttr(DPUCost)).getValue().getSExtValue();
        return checked_cast<size_t>(cost);
    }

    VPUX_THROW("Op {0} has no atrribute {1}", op->getLoc(), DPUCost);
}

size_t vpux::getAsyncExecuteCycleBegin(mlir::async::ExecuteOp op) {
    if (!op->hasAttr(cycleBegin)) {
        Logger::global().trace("Attribute '{0}' not present in async.execute '{1}'", cycleBegin, op);
        return 0;
    }
    return checked_cast<size_t>(mlir::cast<mlir::IntegerAttr>(op->getAttr(cycleBegin)).getValue().getSExtValue());
}

size_t vpux::getAsyncExecuteCycleEnd(mlir::async::ExecuteOp op) {
    if (!op->hasAttr(cycleEnd)) {
        Logger::global().trace("Attribute '{0}' not present in async.execute '{1}'", cycleEnd, op);
        return 0;
    }
    return checked_cast<size_t>(mlir::cast<mlir::IntegerAttr>(op->getAttr(cycleEnd)).getValue().getSExtValue());
}

size_t vpux::calculateCopyCycles(mlir::Operation* innerOp, config::ArchKind archKind,
                                 const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::NNDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::DepthToSpaceDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::SpaceToDepthDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::PerAxisTileDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::TimestampOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getOutput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::PermuteDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::ExpandDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::UpsamplingDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.getInput(), copyOp.getOutput(), archKind, costModel));
    } else if (auto convertDMAOp = mlir::dyn_cast<VPUIP::ConvertDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(convertDMAOp.getInput(), convertDMAOp.getOutput(), archKind, costModel));
    }
    return 0;
}

vpux::Byte vpux::getSwKernelRunTotalAllocSize(VPUIP::SwKernelRun swKernelRun, ArrayRef<mlir::Value> inputs,
                                              ArrayRef<mlir::Value> outputBuffs,
                                              SmallVector<mlir::Value>& inputsForKernelRun,
                                              SmallVector<mlir::Value>& outputsForKernelRun) {
    const auto insSize = inputs.size();
    const auto outsSize = outputBuffs.size();
    const auto kernelOpArgsCount = insSize + outsSize;
    auto totalSwKernelRunSize = vpux::Byte(0);

    for (auto arg : swKernelRun.getArgs()) {
        auto blkArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(arg);
        if (blkArg == nullptr) {
            continue;
        }

        auto id = blkArg.getArgNumber();
        VPUX_THROW_UNLESS(id < kernelOpArgsCount,
                          "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", id,
                          kernelOpArgsCount);
        mlir::Value buffer;
        if (id < insSize) {
            buffer = inputs[id];
            inputsForKernelRun.push_back(buffer);
        } else {
            buffer = outputBuffs[id - insSize];
            outputsForKernelRun.push_back(buffer);
        }
        totalSwKernelRunSize += mlir::cast<vpux::NDTypeInterface>(buffer.getType()).getCompactAllocSize();
    }
    return totalSwKernelRunSize;
}

std::string getSwKernelOperationName(VPUIP::SwKernelOp swKernelOp) {
    auto strKernelOp = swKernelOp.getKernelFunction().getLeafReference().str();

    // cut kernel_entry name if is added
    auto vpuNameEndIdx = strKernelOp.find(".", 0);
    if (vpuNameEndIdx != std::string::npos) {
        strKernelOp = strKernelOp.substr(0, vpuNameEndIdx);
    }

    size_t prefEndIndex = 0;
    auto prefIndex = strKernelOp.find(vpux::VPUIP::SW_KERNEL_NAME_PREFIX.str());
    if (prefIndex != std::string::npos) {
        prefEndIndex = prefIndex + vpux::VPUIP::SW_KERNEL_NAME_PREFIX.size();
    } else {
        StringLiteral generated = "generated_";
        auto prefIndexGenerated = strKernelOp.find(generated);
        VPUX_THROW_WHEN(prefIndexGenerated == std::string::npos, "Not a valid swKernelOp name - {0}", strKernelOp);
        prefEndIndex = prefIndexGenerated + generated.size();
    }

    VPUX_THROW_WHEN(prefEndIndex > strKernelOp.size(), "Not a valid swKernelOp name length - {0}", strKernelOp);

    auto nameSize = std::string::npos;
    auto nameEndIndex = strKernelOp.find("_", prefEndIndex);
    if (nameEndIndex != std::string::npos) {
        nameSize = nameEndIndex - prefIndex;
    }

    return strKernelOp.substr(prefEndIndex, nameSize);
}

// Define a map for operation handlers used only for Shave2 API
// Some functions might have the default parameters but we extract and profile the exact parameters via
// ExtraParameters which save the exact attrs and let us know the exact workload which is done in
// If modifications happen into getKernelInfo(mlir::Operation* origOp) that has the transforming switch-case from
// VPU to VPUIP dialect, or any other optimization place that will change the order of the parameters, please revise
// accordingly in this place and request a cache update.
// getShaveWorkloadFunction
std::map<std::string, std::function<void(VPUNN::SHAVEWorkload::Parameters&, std::string&, VPUIP::SwKernelOp)>>
        operationHandlersVPUIP = {
                {"MVN",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string& shaveFunction,
                    VPUIP::SwKernelOp swKernelOp) {
                     auto swKernelRunOps = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
                     auto swKernelRunOp = *swKernelRunOps.begin();
                     auto attributeArray = mlir::dyn_cast_or_null<mlir::ArrayAttr>(swKernelRunOp->getAttr("attrs"));
                     VPUX_THROW_WHEN(!attributeArray || attributeArray.empty(),
                                     "MVN operation does not have valid attributes");

                     auto acrossChannels = mlir::dyn_cast_or_null<mlir::BoolAttr>(attributeArray[0]);
                     VPUX_THROW_WHEN(!acrossChannels, "MVN operation does not have the accross channels attribute");

                     shaveFunction = acrossChannels.getValue() ? "MVN_3Ax" : "MVN_2Ax";
                     VPUNN::SHAVEWorkload::Param param =
                             acrossChannels.getValue() ? 3 : 2;  // how many axes are selected
                     params = std::vector{param};
                 }},
                {"MVN6",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPUIP::SwKernelOp) {
                     VPUNN::SHAVEWorkload::Param param = 1;  // how many axes are selected
                     params = std::vector{param};
                 }},
                {"softmax",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPUIP::SwKernelOp) {
                     VPUNN::SHAVEWorkload::Param param = 1;  // select dimension (N(0), C(1), H(2), W(3))
                     params = std::vector{param};
                 }},
                {"gather",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPUIP::SwKernelOp) {
                     VPUNN::SHAVEWorkload::Param paramAxis = 1;
                     VPUNN::SHAVEWorkload::Param paramBatches = 1;
                     params = std::vector{paramAxis, paramBatches};
                 }},
                {"normalizel2onlyc",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPUIP::SwKernelOp) {
                     VPUNN::SHAVEWorkload::Param param = 1;  // select dimension (N(0), C(1), H(2), W(3))
                     params = std::vector{param};
                 }},
};

std::map<std::string, std::function<void(VPUNN::SHAVEWorkload::Parameters&, std::string&, VPU::SWOpInterface)>>
        operationHandlersVPU = {
                {"MVN",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string& shave_function,
                    VPU::SWOpInterface operation) {
                     auto acrossChannelsAttr = operation->getAttr("across_channels");
                     auto acrossChannels = mlir::dyn_cast_or_null<mlir::BoolAttr>(acrossChannelsAttr);
                     VPUX_THROW_WHEN(!acrossChannels, "MVN operation does not have the across_channels attribute");
                     shave_function = acrossChannels.getValue() ? "MVN_3Ax" : "MVN_2Ax";
                     VPUNN::SHAVEWorkload::Param param =
                             acrossChannels.getValue() ? 3 : 2;  // how many axes are selected
                     params = std::vector{param};
                 }},
                {"MVN6",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPU::SWOpInterface operation) {
                     auto axesAttr = operation->getAttr("axes");
                     auto axes = mlir::dyn_cast_or_null<mlir::ArrayAttr>(axesAttr);
                     VPUX_THROW_WHEN(!axes, "MVN6 operation does not have the axes attribute");
                     VPUNN::SHAVEWorkload::Param param = static_cast<int>(axes.size());  // how many axes are selected
                     params = std::vector{param};
                 }},
                {"softmax",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPU::SWOpInterface operation) {
                     auto axisIndAttr = operation->getAttr("axisInd");
                     auto axisInd = mlir::dyn_cast_or_null<mlir::IntegerAttr>(axisIndAttr);
                     VPUX_THROW_WHEN(!axisInd, "softmax operation does not have the axisInd attribute");
                     VPUNN::SHAVEWorkload::Param param =
                             static_cast<int>(axisInd.getInt());  // select dimension (N(0), C(1), H(2), W(3))
                     params = std::vector{param};
                 }},
                {"gather",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPU::SWOpInterface operation) {
                     auto gatherOp = mlir::dyn_cast<vpux::VPU::GatherOp>(operation.getOperation());
                     VPUX_THROW_WHEN(!gatherOp, "Operation is not a GatherOp");

                     // Access GatherOp-specific attributes
                     auto axis = gatherOp.getAxisValueAttr();
                     auto batchDims = gatherOp.getBatchDimsAttr();

                     VPUNN::SHAVEWorkload::Param paramAxis = axis ? static_cast<int>(axis.getInt()) : 1;
                     VPUNN::SHAVEWorkload::Param paramBatches = batchDims ? static_cast<int>(batchDims.getInt()) : 1;
                     params = std::vector{paramAxis, paramBatches};
                 }},
                {"normalizel2onlyc",
                 [](VPUNN::SHAVEWorkload::Parameters& params, std::string&, VPU::SWOpInterface) {
                     VPUNN::SHAVEWorkload::Param param = 1;  // select dimension (N(0), C(1), H(2), W(3))
                     params = std::vector{param};
                 }},
};

std::vector<VPUNN::VPUTensor> getVPUNNTensorFromArrayRef(ArrayRef<mlir::Value> values, VPUNN::VPUDevice vpuDev) {
    std::vector<VPUNN::VPUTensor> tensors;
    for (const auto& value : values) {
        auto ndType = mlir::cast<vpux::NDTypeInterface>(value.getType());
        tensors.push_back(getVPUNNTensor(ndType.getShape(), getElementType(ndType.getElementType(), vpuDev)));
    }
    return tensors;
}

std::unique_ptr<VPUNN::SHAVEWorkload> getShaveWorkloadFunction(VPUIP::SwKernelOp swKernelOp, config::ArchKind arch,
                                                               ArrayRef<mlir::Value> inputs,
                                                               ArrayRef<mlir::Value> outputs, bool isShave2ApiUsed) {
    auto swKernelName = getSwKernelOperationName(swKernelOp);
    auto vpuDev = vpux::VPU::getVPUDeviceType(arch);
    const auto& shaveUtilIntf = vpux::VPU::CostModelConfig::getShaveCostModelUtilsInterface(arch);

    VPUNN::SHAVEWorkload::Parameters params = {};
    VPUNN::SHAVEWorkload::ExtraParameters extraParams = {};
    std::string baseLevelString{"VPUIP"};
    extraParams["level"] = baseLevelString;

    VPUX_THROW_WHEN(inputs.empty(), "No inputs identified for op {0}", swKernelName);
    VPUX_THROW_WHEN(outputs.empty(), "No outputs identified for op {0}", swKernelName);

    // Getting the input and output tensors
    std::vector<VPUNN::VPUTensor> inputTensors = getVPUNNTensorFromArrayRef(inputs, vpuDev);
    std::vector<VPUNN::VPUTensor> outputTensors = getVPUNNTensorFromArrayRef(outputs, vpuDev);

    // search operation
    std::string shaveFunction;
    auto mapOfKernels = shaveUtilIntf.getSwKernelContainer();
    auto it = mapOfKernels.find(swKernelName);

    if (it == mapOfKernels.end()) {
        return nullptr;
    }

    shaveFunction = it->second;
    // Execute the handler for the current operation
    if (isShave2ApiUsed) {
        auto handlerIt = operationHandlersVPUIP.find(shaveFunction);
        if (handlerIt != operationHandlersVPUIP.end()) {
            // Since for some operations that are influenced by some extraparamteres since we cannot figure out
            // them from the operation attributes, we are going to take that specific array, then we will get the values
            // and add them into a single string which will be passed to VPUNN as extra parameter. VPUNN profiling
            // system will cover the specific implementation of the higher level operation and will retrieve from cache
            // the correct result.
            auto swKernelRunOps = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
            auto swKernelRunOp = *swKernelRunOps.begin();
            auto attributeArray = mlir::dyn_cast_or_null<mlir::ArrayAttr>(swKernelRunOp->getAttr("attrs"));

            std::string attrValuesStr;
            if (attributeArray != nullptr) {
                llvm::raw_string_ostream rso(attrValuesStr);
                for (auto attr : attributeArray) {
                    std::string attrStr;
                    llvm::raw_string_ostream attrStream(attrStr);
                    attr.print(attrStream);
                    attrStream.flush();
                    std::replace(attrStr.begin(), attrStr.end(), ',', '.');
                    rso << attrStr << ";";
                }
                rso.flush();
            }
            extraParams["attrs"] = attrValuesStr;
            handlerIt->second(params, shaveFunction, swKernelOp);
        }
    }

    auto swwl = std::make_unique<VPUNN::SHAVEWorkload>(shaveFunction, vpuDev, inputTensors, outputTensors, params,
                                                       extraParams);
    return swwl;
}

std::unique_ptr<VPUNN::SHAVEWorkload> getShaveWorkloadFunction(VPU::SWOpInterface operation, config::ArchKind arch,
                                                               std::vector<VPUNN::VPUTensor> inputTensors,
                                                               std::vector<VPUNN::VPUTensor> outputTensors,
                                                               bool isShave2ApiUsed) {
    auto swKernelName = operation->getName().stripDialect().str();
    auto vpuDev = vpux::VPU::getVPUDeviceType(arch);
    const auto& shaveUtilIntf = vpux::VPU::CostModelConfig::getShaveCostModelUtilsInterface(arch);

    VPUNN::SHAVEWorkload::Param param;
    VPUNN::SHAVEWorkload::Parameters params = {};

    VPUNN::SHAVEWorkload::ExtraParameters extraParams = {};
    std::string baseLevelString{"VPU"};
    extraParams["level"] = baseLevelString;

    auto mapOfKernels = shaveUtilIntf.getSwKernelContainer();
    auto it = mapOfKernels.find(swKernelName);

    if (it == mapOfKernels.end()) {
        return nullptr;
    }

    std::string shaveFunction = it->second;
    if (isShave2ApiUsed) {
        auto handlerIt = operationHandlersVPU.find(shaveFunction);
        if (handlerIt != operationHandlersVPU.end()) {
            // Process attributes and populate extraParams
            for (auto attr : operation->getAttrs()) {
                auto attrName = attr.getName().str();
                if (attrName == "multiClusterStrategy") {
                    continue;
                }
                extraParams[attrName] = formatv("{0}", attr.getValue()).str();
            }
            handlerIt->second(params, shaveFunction, operation);
        }
    }

    auto swwl = std::make_unique<VPUNN::SHAVEWorkload>(shaveFunction, vpuDev, inputTensors, outputTensors, params,
                                                       extraParams);
    return swwl;
}

std::unique_ptr<VPUNN::SHAVEWorkload> getShaveWorkloadFunction(VPU::SWOpInterface operation, config::ArchKind arch,
                                                               ArrayRef<mlir::Value> inputs,
                                                               ArrayRef<mlir::Value> outputs, bool isShave2ApiUsed) {
    auto vpuDev = vpux::VPU::getVPUDeviceType(arch);

    std::vector<VPUNN::VPUTensor> inputTensors = getVPUNNTensorFromArrayRef(inputs, vpuDev);
    std::vector<VPUNN::VPUTensor> outputTensors = getVPUNNTensorFromArrayRef(outputs, vpuDev);

    return getShaveWorkloadFunction(operation, arch, std::move(inputTensors), std::move(outputTensors),
                                    isShave2ApiUsed);
}

size_t getShaveActCycleForSwKernelFunc(VPUIP::SwKernelOp swKernelOp, config::ArchKind arch,
                                       ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                       const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto log = vpux::Logger::global().nest("Get Shave Act Cycle For SW Kernel Function", 0);

    auto swwl = getShaveWorkloadFunction(swKernelOp, arch, inputs, outputs, costModel->isShave2ApiUsed());
    if (swwl == nullptr) {
        return 1;
    }

    std::string infoOut;
    auto cost = VPU::checkAndReturnCost(costModel->SHAVE(*swwl, infoOut), log, true);

    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    if (cost >= VPU::INVALID_COST_BASE) {
        log.trace("[VPUNN LOG] getShaveActCycleForSwKernelFunc: INVALID_COST is caught. Please check possible VPUNN "
                  "debug info: {0}",
                  infoOut);
        VPU::printVPUNNWorkloadConfig(*swwl, logCb);
        return 1;
    }
    return cost;
}

std::unique_ptr<VPUNN::SHAVEWorkload> vpux::getVPUNNSWKernelOp(VPUIP::SwKernelOp swKernelOp, bool isShave2ApiUsed) {
    // Exclude strange sw ops produced by compiler like cache_flush_invalidate op
    if (swKernelOp.getInputs().empty() || swKernelOp.getOutputBuffs().empty()) {
        return nullptr;
    }

    auto inputs = to_small_vector(swKernelOp.getInputs());
    auto outputs = to_small_vector(swKernelOp.getOutputBuffs());

    SmallVector<mlir::Value> smallVecInputs(inputs.begin(), inputs.end());
    SmallVector<mlir::Value> smallVecOutputs{outputs[0]};

    return getShaveWorkloadFunction(swKernelOp, config::getArch(swKernelOp.getOperation()), smallVecInputs,
                                    smallVecOutputs, isShave2ApiUsed);
}

std::unique_ptr<VPUNN::SHAVEWorkload> vpux::getVPUNNSWKernelOp(VPU::SWOpInterface operation, bool isShave2ApiUsed) {
    auto vpuDev = config::getArch(operation);
    const auto operName = operation->getName().stripDialect().str();

    auto inputs = to_small_vector(operation->getOperands());
    auto output = operation->getResult(0);

    return getShaveWorkloadFunction(operation, vpuDev, inputs, {output}, isShave2ApiUsed);
}

std::unique_ptr<VPUNN::SHAVEWorkload> vpux::getVPUNNSWKernelOp(VPU::SWOpInterface operation,
                                                               vpux::NDTypeInterface outputNDType,
                                                               ArrayRef<vpux::NDTypeInterface> inputTypes,
                                                               bool isShave2APIused) {
    auto arch = config::getArch(operation);
    auto vpuDev = vpux::VPU::getVPUDeviceType(arch);

    const auto operName = operation->getName().stripDialect().str();
    auto outputTensor = getVPUNNTensor(outputNDType.getShape(), getElementType(outputNDType.getElementType(), vpuDev));
    std::vector<VPUNN::VPUTensor> outputTensors = {outputTensor};

    std::vector<VPUNN::VPUTensor> inputTensors;
    for (auto inputNd : inputTypes) {
        inputTensors.push_back(getVPUNNTensor(inputNd.getShape(), getElementType(inputNd.getElementType(), vpuDev)));
    }

    return getShaveWorkloadFunction(operation, arch, std::move(inputTensors), std::move(outputTensors),
                                    isShave2APIused);
}

std::unique_ptr<VPUNN::SHAVEWorkload> vpux::getVPUNNSWKernelOp(VPU::SWOpInterface operation,
                                                               std::vector<VPUNN::VPUTensor> outputTensors,
                                                               std::vector<VPUNN::VPUTensor> inputTensors,
                                                               bool isShave2ApiUsed) {
    auto arch = config::getArch(operation);

    const auto operName = operation->getName().stripDialect().str();

    return getShaveWorkloadFunction(operation, arch, std::move(inputTensors), std::move(outputTensors),
                                    isShave2ApiUsed);
}

size_t vpux::calculateShaveActCycles(VPUIP::SwKernelOp swKernelOp,
                                     const std::shared_ptr<VPUNN::VPUCostModel>& costModel, config::ArchKind arch) {
    if (swKernelOp.getInputs().empty() || swKernelOp.getOutputBuffs().empty()) {
        return 1;
    }

    auto inputNdType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(0).getType());
    auto outputNdType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
    auto inputElemType = inputNdType.getElementType();
    auto outputElemType = outputNdType.getElementType();

    // CostModel does not support F32/SI32 layers
    if (!costModel->isShave2ApiUsed()) {
        if (inputElemType.isF32() || outputElemType.isF32()) {
            return 1;
        }
        if (inputElemType.isSignedInteger(32) || outputElemType.isSignedInteger(32)) {
            return 1;
        }
    }
    auto inputs = to_small_vector(swKernelOp.getInputs());
    auto outputs = to_small_vector(swKernelOp.getOutputBuffs());

    SmallVector<mlir::Value> inputsForLargestKernelRun(inputs.begin(), inputs.end());
    SmallVector<mlir::Value> outputsForLargestKernelRun{outputs[0]};
    auto largestSwKernelRunSize = vpux::Byte(0);
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();

    // SwKernelOp can have multiple SWKernelRun which could further be distributed on 2 ACTShaves in parallel
    // In such case use the largest SwKernelRun to calculate the cycle cost
    if (std::distance(swKernelRuns.begin(), swKernelRuns.end()) > 1) {
        for (auto&& kernelRun : swKernelRuns) {
            SmallVector<mlir::Value> inputsForKernelRun;
            SmallVector<mlir::Value> outputsForKernelRun;
            auto swKernelRunSize =
                    getSwKernelRunTotalAllocSize(kernelRun, inputs, outputs, inputsForKernelRun, outputsForKernelRun);
            if (largestSwKernelRunSize < swKernelRunSize) {
                largestSwKernelRunSize = swKernelRunSize;
                inputsForLargestKernelRun = std::move(inputsForKernelRun);
                outputsForLargestKernelRun = std::move(outputsForKernelRun);
            }
        }
    }

    return getShaveActCycleForSwKernelFunc(swKernelOp, arch, inputsForLargestKernelRun, outputsForLargestKernelRun,
                                           costModel);
}

size_t vpux::getDPUTaskOpCost(VPUIP::DPUTaskOp dpuTaskOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                              config::ArchKind arch, vpux::Logger log) {
    auto nceOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    auto inputOneType = nceOp->getOperand(0).getType();
    auto outputType = nceOp->getResult(0).getType();

    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(inputOneType).getElementType();
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(outputType).getElementType();

    // CostModel does not support F32/SI32 layers
    if (inputElemType.isF32() || outputElemType.isF32()) {
        return 1;
    }
    if (inputElemType.isSignedInteger(32) || outputElemType.isSignedInteger(32)) {
        return 1;
    }

    auto vpunnDPUWorkload = vpux::getDPUWorkload(dpuTaskOp, arch);

    // TODO: Should RUNTIME_OVERHEAD_PER_WORKLOAD be added?
    std::string vpunnInputCheckInfo;
    auto cost = VPU::checkAndReturnCost(costModel->DPU(vpunnDPUWorkload, vpunnInputCheckInfo), log, true);
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    if (cost >= VPU::INVALID_COST_BASE) {
        log.trace("[VPUNN LOG] getDPUTaskOpCost: INVALID_COST is caught. Please check possible VPUNN debug info: {0}",
                  vpunnInputCheckInfo);
        VPU::printVPUNNWorkloadConfig(vpunnDPUWorkload, logCb);
    }
    return cost;
}

std::vector<std::pair<int64_t, size_t>> vpux::calculateNceVariantCycles(
        VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel, config::ArchKind arch,
        vpux::Logger log) {
    std::vector<std::pair<int64_t, size_t>> nceVariantCyclePerCluster;
    for (auto dpuTaskOp : nceOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        auto clusterId = dpuTaskOp.getClusterId().value_or(0);
        nceVariantCyclePerCluster.push_back({clusterId, getDPUTaskOpCost(dpuTaskOp, costModel, arch, log)});
    }
    return nceVariantCyclePerCluster;
}

size_t vpux::calculateNceCycles(VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                config::ArchKind arch, vpux::Logger log, int64_t numDPU) {
    auto variantCostVec = calculateNceVariantCycles(nceOp, costModel, arch, log);

    // Group costs by cluster ID and find the maximum cost for each cluster
    std::unordered_map<int64_t, std::vector<size_t>> clusterCosts;
    for (const auto& entry : variantCostVec) {
        clusterCosts[entry.first].push_back(entry.second);
    }
    size_t maxCost = 0;
    for (const auto& entry : clusterCosts) {
        size_t actualCost = VPUNN::dpu_schedule(numDPU, entry.second);
        if (actualCost > maxCost) {
            maxCost = actualCost;
        }
    }
    return maxCost;
}

std::string vpux::stringifyVPUNNStrategy(VPUNN::VPUTilingStrategy strategy) {
    const auto& enumMap = VPUNN::mapToText<VPUNN::VPUTilingStrategy>();
    auto it = enumMap.find(static_cast<int>(strategy));
    if (it != enumMap.end()) {
        return it->second;
    } else {
        return "WRONG strategy";
    }
}
