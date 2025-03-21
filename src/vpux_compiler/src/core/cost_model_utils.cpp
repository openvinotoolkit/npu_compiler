//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <vpu/shave/layers.h>
#include <bitset>

using namespace vpux;

VPUNN::VPUTensor getVPUNNTensor(ShapeRef tensorShape, VPUNN::DataType dataType) {
    return VPUNN::VPUTensor({static_cast<unsigned int>(tensorShape.totalSize()), 1, 1, 1}, dataType);
}

VPUNN::VPUTensor getVPUNNTensorMultiCluster(ArrayRef<Shape> tensorShapes, VPUNN::DataType dataType) {
    unsigned int totalShape = 0;
    for (size_t idx = 0; idx < tensorShapes.size(); idx++) {
        totalShape += static_cast<unsigned int>(tensorShapes[idx].totalSize());
    }
    return VPUNN::VPUTensor({totalShape, 1, 1, 1}, dataType);
}

VPUNN::DataType getElementType(mlir::Type type) {
    if (type.isF16()) {
        return VPUNN::DataType::FLOAT16;
    } else if (type.isInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::INT8;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::UINT8;
    } else if (auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qType.getStorageTypeIntegralWidth() == 8) {
            return qType.isSigned() ? VPUNN::DataType::INT8 : VPUNN::DataType::UINT8;
        }
    }
    // default until support for more types introduced
    return VPUNN::DataType::BFLOAT16;
}

VPUNN::MemoryLocation vpux::getMemoryLocation(mlir::Type type) {
    auto memKind = type.cast<vpux::NDTypeInterface>().getMemoryKind();
    if (memKind == VPU::MemoryKind::CMX_NN) {
        return VPUNN::MemoryLocation::CMX;
    }

    return VPUNN::MemoryLocation::DRAM;
}

VPUNN::Swizzling getVPUNNSwizzlingKey(mlir::Type type) {
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

VPUNN::DPUWorkload vpux::getDPUWorkload(VPUIP::DPUTaskOp dpuTaskOp, VPU::ArchKind arch) {
    auto nceClusterOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceClusterOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    auto inputOneType = nceClusterOp->getOperand(0).getType();
    auto outputType = nceClusterOp->getResult(0).getType();
    auto inputTwoType = nceClusterOp->getNumOperands() > 1 ? nceClusterOp->getOperand(1).getType() : nullptr;

    // E#73766: delete this check in the end of epic
    if (auto nceTilingParent = nceClusterOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        inputOneType = nceTilingParent.getInputs()[0].getType();
        outputType = nceTilingParent.getOutputs()[0].getType();
        inputTwoType = nceTilingParent.getInputs().size() > 1 ? nceTilingParent.getInputs()[1].getType() : nullptr;
    }

    auto inputElemType = inputOneType.cast<vpux::NDTypeInterface>().getElementType();
    auto outputElemType = outputType.cast<vpux::NDTypeInterface>().getElementType();

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
    VPUNN::ISIStrategy isiStrategy = VPUNN::ISIStrategy::CLUSTERING;

    // Check if output is distributed (multicasted) to multiple tiles (e.g. SOK)
    // The output type of NCEClusterTaskOp must be DistributedBufferType whether it's unrolled or not
    // Thus it's no need to get DistributedBufferType from its parent (Actually that will be failed in
    // CalculateAsyncRegionCycleCost pass)
    auto distributedOutput = outputType.dyn_cast<VPUIP::DistributedBufferType>();
    bool isOutputCMajor = false;
    if (distributedOutput) {
        const auto distributionAttr = distributedOutput.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED) ||
            mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
            outputWriteTiles = distributionAttr.getNumClusters().getInt();
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        }
        if (distributedOutput.getDimsOrder() == DimsOrder::NCHW) {
            isOutputCMajor = true;
        }
    }

    // Check if input is distributed - segmented between multiple tiles (e.g. SOH)
    // TODO: Get DistributedBufferType from parent input will fail in CalculateAsyncRegionCycleCost pass, ticket see
    // #104324
    if (auto distributedInput = nceClusterOp.getParentInput().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        const auto distributionAttr = distributedInput.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED) {
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        }
    }

    bool isWeightsSparsityEnabled = false;
    float weightsSparsityRatio = 0;
    auto weightsSparsityMap = nceClusterOp.getWeightsSparsityMap();
    if (weightsSparsityMap != nullptr && nceClusterOp.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
        isWeightsSparsityEnabled = true;

        auto weightsType = nceClusterOp.getWeights().getType().cast<vpux::NDTypeInterface>();
        auto weightsElemType = weightsType.getElementType();

        const auto sparsityCompressionAttr = VPUIP::getSparsityCompressionAttr(weightsType);
        VPUX_THROW_WHEN(sparsityCompressionAttr == nullptr, "sparsity_compressionAttr shouldn't be a nullptr");

        auto compressedSize = sparsityCompressionAttr.getAllocSize(weightsElemType).count();
        weightsSparsityRatio = vpux::getWeightsSparsityRatio(weightsType, compressedSize);
    }

    auto isInputSparsityEnabled =
            (nceClusterOp.getInputSparsityMap() != nullptr || nceClusterOp.getInputStorageElementTable() != nullptr);
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
                      ? inputOneType.cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C]
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

    // Set input layout - in general input layout is always ZXY even for permuteQuantize
    auto inputLayout = VPUNN::Layout::ZXY;
    // Set output layout
    auto outputLayout = VPUNN::Layout::ZXY;
    if (nceClusterOp.getIsPermuteQuantizeAttr() != nullptr) {
        outputLayout = VPUNN::Layout::YZX;
    } else if (isOutputCMajor) {
        outputLayout = VPUNN::Layout::XYZ;
    }

    const auto inputTensor = VPUNN::VPUTensor(
            {static_cast<unsigned int>(IW), static_cast<unsigned int>(IH), static_cast<unsigned int>(IC), 1},
            getElementType(inputElemType), inputLayout, isInputSparsityEnabled);
    const auto outputTensor = VPUNN::VPUTensor(
            {static_cast<unsigned int>(OW), static_cast<unsigned int>(OH), static_cast<unsigned int>(OC), 1},
            getElementType(outputElemType), outputLayout, isOutputSparsityEnabled);

    VPUNN::DPUWorkload vpunnDPUWorkload;
    vpunnDPUWorkload.device = getVPUDeviceType(arch);
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

    return vpunnDPUWorkload;
}

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                    VPU::ArchKind archKind, const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto operandType = innerOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    return static_cast<size_t>(costModel->DMA(getVPUDeviceType(archKind),
                                              {getVPUNNTensorMultiCluster(perClusterShapes, inElemType)},
                                              {getVPUNNTensorMultiCluster(perClusterShapes, outElemType)}));
}

bool extraDMAsRequired(mlir::Value innerOperand) {
    if (auto inputType = innerOperand.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        auto distribution = inputType.getDistribution();
        auto distributionMode = distribution.getMode().getValue();
        return distributionMode == VPU::DistributionMode::SEGMENTED ||
               distributionMode == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

size_t vpux::getDMACost(mlir::Value input, mlir::Value output, VPU::ArchKind archKind,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    auto inElemType = getElementType(inputType.cast<vpux::NDTypeInterface>().getElementType());
    auto outElemType = getElementType(outputType.cast<vpux::NDTypeInterface>().getElementType());

    if (inputType.dyn_cast<VPUIP::DistributedBufferType>() && extraDMAsRequired(input)) {
        return calculateMultiClusterDMACost(input, inElemType, outElemType, archKind, costModel);
    }

    if (outputType.dyn_cast<VPUIP::DistributedBufferType>() && extraDMAsRequired(output)) {
        return calculateMultiClusterDMACost(output, inElemType, outElemType, archKind, costModel);
    }

    auto inputShape = getShape(input);
    auto outputShape = getShape(output);

    // TODO: add layout info to VPUNN tensors
    auto cost = costModel->DMA(getVPUDeviceType(archKind), {getVPUNNTensor(inputShape, inElemType)},
                               {getVPUNNTensor(outputShape, outElemType)}, getMemoryLocation(inputType),
                               getMemoryLocation(outputType));

    return static_cast<size_t>(cost);
}

size_t getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "DMA ports is at least one but got {0}", numDMAPorts);
    auto distributedTensorType = tensorType.dyn_cast<VPU::DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    auto elemType = tensorType.getElementType();

    SmallVector<Shape> shapes;
    if (numDMAPorts > 1) {
        // For distributed segmented DMA, transaction will be split between ports and executing
        // in parallel when there are multiple DMA ports available.
        // When enabling NPU40XX whose number of tiles is not equal to number of DMA ports, using
        // simply the largest size in tiles to calculate cost is not accurate, see E#84432
        shapes.push_back(distributedTensorType.getLargestCompactShape());
    } else {
        shapes = distributedTensorType.getPerClusterComputeShapes();
    }
    auto vpuTensor = getVPUNNTensorMultiCluster(shapes, getElementType(elemType));
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

size_t getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t /*numDMAPorts*/) {
    auto shape = tensorType.getShape();
    auto elemType = tensorType.getElementType();
    auto vpuTensor = getVPUNNTensor(shape, getElementType(elemType));
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
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

size_t vpux::getDMACost(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    if (auto sparseTensorType = tensorType.dyn_cast<VPU::SparseTensorType>()) {
        tensorType = sparseTensorType.getData().cast<vpux::NDTypeInterface>();
    }

    auto distributedType = tensorType.dyn_cast<VPU::DistributedTensorType>();

    const auto elementType = tensorType.getElementType();

    if (distributedType != nullptr) {
        const auto dmaCostFunc = spillingCostMapVPUNN.at(distributedType.getDistribution().getMode().getValue());
        return dmaCostFunc(tensorType, vpuDevice, costModel, numDMAPorts);
    }

    const auto vpunnTensor = getVPUNNTensor(tensorType.getShape(), getElementType(elementType));
    return costModel->DMA(vpuDevice, vpunnTensor, vpunnTensor);
}

size_t vpux::getDPUCost(mlir::Operation* op) {
    // costs for DPU calculated during workload generation, re-use

    if (op->hasAttr(DPUCost)) {
        auto cost = op->getAttr(DPUCost).cast<mlir::IntegerAttr>().getValue().getSExtValue();
        return checked_cast<size_t>(cost);
    }

    VPUX_THROW("Op {0} has no atrribute {1}", op->getLoc(), DPUCost);
}

size_t vpux::getAsyncExecuteCycleBegin(mlir::async::ExecuteOp op) {
    if (!op->hasAttr(cycleBegin)) {
        Logger::global().trace("Attribute '{0}' not present in async.execute '{1}'", cycleBegin, op);
        return 0;
    }
    return checked_cast<size_t>(op->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getValue().getSExtValue());
}

size_t vpux::getAsyncExecuteCycleEnd(mlir::async::ExecuteOp op) {
    if (!op->hasAttr(cycleEnd)) {
        Logger::global().trace("Attribute '{0}' not present in async.execute '{1}'", cycleEnd, op);
        return 0;
    }
    return checked_cast<size_t>(op->getAttr(cycleEnd).cast<mlir::IntegerAttr>().getValue().getSExtValue());
}

size_t vpux::calculateCopyCycles(mlir::Operation* innerOp, VPU::ArchKind archKind,
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
        auto blkArg = arg.dyn_cast_or_null<mlir::BlockArgument>();
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
        totalSwKernelRunSize += buffer.getType().cast<vpux::NDTypeInterface>().getCompactAllocSize();
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

#define SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT(_NAME_STR_, _VPUNN_TYPE_)                                         \
    {                                                                                                          \
        _NAME_STR_, [](VPUNN::VPUDevice vpuDev, VPUNN::VPUTensor inputTensor, VPUNN::VPUTensor outputTensor) { \
            return std::make_unique<_VPUNN_TYPE_>(vpuDev, inputTensor, outputTensor);                          \
        }                                                                                                      \
    }

#define SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT(_NAME_STR_, _VPUNN_TYPE_)                           \
    {                                                                                              \
        _NAME_STR_, [](VPUNN::VPUDevice vpuDev, const std::vector<VPUNN::VPUTensor>& inputTensors, \
                       VPUNN::VPUTensor outputTensor) {                                            \
            return std::make_unique<_VPUNN_TYPE_>(vpuDev, inputTensors, outputTensor);             \
        }                                                                                          \
    }

std::map<std::string,
         std::function<std::unique_ptr<VPUNN::SWOperation>(VPUNN::VPUDevice, VPUNN::VPUTensor, VPUNN::VPUTensor)>>
        swKernelNameToVpunn1InputFuncMap = {
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sigmoid", VPUNN::SHVSigmoid),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Elu", VPUNN::SHVELU),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("HardSigmoid", VPUNN::SHVHardSigmoid),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("SoftMax", VPUNN::SHVSoftmax),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Clamp", VPUNN::SHVClamp),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("FakeQuantize", VPUNN::SHVFakeQuantize),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Quantize", VPUNN::SHVQuantizeCast),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Tanh", VPUNN::SHVTanh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sin", VPUNN::SHVSin),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Cos", VPUNN::SHVCos),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sqrt", VPUNN::SHVSqrt),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sinh", VPUNN::SHVSinh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Cosh", VPUNN::SHVCosh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Asinh", VPUNN::SHVAsinh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Acosh", VPUNN::SHVAcosh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Abs", VPUNN::SHVAbs),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Atan", VPUNN::SHVAtan),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Asin", VPUNN::SHVAsin),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Acos", VPUNN::SHVAcos),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Atanh", VPUNN::SHVAtanh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Log", VPUNN::SHVLog),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Selu", VPUNN::SHVSelu),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Gelu", VPUNN::SHVGelu),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Exp", VPUNN::SHVExp),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Floor", VPUNN::SHVFloor),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Round", VPUNN::SHVRound),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Mish", VPUNN::SHVMish),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Erf", VPUNN::SHVErf),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Negative", VPUNN::SHVNegative),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sign", VPUNN::SHVSign),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("YuvToRgb", VPUNN::SHVYuvToRgb),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("SoftPlus", VPUNN::SHVSoftPlus),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Swish", VPUNN::SHVSwish),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("MVN", VPUNN::SHVMVN),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Ceiling", VPUNN::SHVCeiling),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Roll", VPUNN::SHVRoll),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Gather", VPUNN::SHVGather),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("ScatterNDUpdate", VPUNN::SHVScatterNDUpdate),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("ScatterUpdate", VPUNN::SHVScatterUpdate),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("PermuteQuantize", VPUNN::SHVPermuteQuantize),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("DepthToSpace", VPUNN::SHVDepthToSpace),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("SpaceToDepth", VPUNN::SHVSpaceToDepthOp),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("MemPermute", VPUNN::SHVMemPermute)};

std::map<std::string, std::function<std::unique_ptr<VPUNN::SWOperation>(
                              VPUNN::VPUDevice, const std::vector<VPUNN::VPUTensor>&, VPUNN::VPUTensor)>>
        swKernelNameToVpunnVecInputFuncMap = {
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Power", VPUNN::SHVPower),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Add", VPUNN::SHVAdd),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Divide", VPUNN::SHVDivide),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("SquaredDifference", VPUNN::SHVSquaredDiff),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("FloorMod", VPUNN::SHVFloorMod),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Less", VPUNN::SHVLess),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LessEqual", VPUNN::SHVLessEqual),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Greater", VPUNN::SHVGreater),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("GreaterEqual", VPUNN::SHVGreaterEqual),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LogicalOr", VPUNN::SHVLogicalOr),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LogicalNot", VPUNN::SHVLogicalNot),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LogicalXor", VPUNN::SHVLogicalXor),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Multiply", VPUNN::SHVMultiply),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("And", VPUNN::SHVAnd),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Minimum", VPUNN::SHVMinimum),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Maximum", VPUNN::SHVMaximum),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Subtract", VPUNN::SHVSubtract),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("NotEqual", VPUNN::SHVNotEqual),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Equal", VPUNN::SHVEqual)};

std::unique_ptr<VPUNN::SWOperation> queryKernelMap(const std::string& swKernelName, VPUNN::VPUDevice vpuDev,
                                                   ArrayRef<vpux::NDTypeInterface> inputNdTypes,
                                                   vpux::NDTypeInterface outputNdType) {
    VPUX_THROW_WHEN(inputNdTypes.empty(), "No inputs identified for op {0}", swKernelName);

    auto outputTensor = getVPUNNTensor(outputNdType.getShape(), getElementType(outputNdType.getElementType()));

    if (swKernelNameToVpunn1InputFuncMap.find(swKernelName) != swKernelNameToVpunn1InputFuncMap.end()) {
        auto input0NdType = inputNdTypes.front();
        auto inputTensor = getVPUNNTensor(input0NdType.getShape(), getElementType(input0NdType.getElementType()));

        return swKernelNameToVpunn1InputFuncMap[swKernelName](vpuDev, inputTensor, outputTensor);
    } else if (swKernelNameToVpunnVecInputFuncMap.find(swKernelName) != swKernelNameToVpunnVecInputFuncMap.end()) {
        std::vector<VPUNN::VPUTensor> inputTensors;
        for (auto inputNd : inputNdTypes) {
            inputTensors.push_back(getVPUNNTensor(inputNd.getShape(), getElementType(inputNd.getElementType())));
        }

        return swKernelNameToVpunnVecInputFuncMap[swKernelName](vpuDev, inputTensors, outputTensor);
    }

    return nullptr;
}

std::unique_ptr<VPUNN::SWOperation> queryKernelMap(const std::string& swKernelName, VPUNN::VPUDevice vpuDev,
                                                   ArrayRef<mlir::Value> inputs, mlir::Value output) {
    SmallVector<vpux::NDTypeInterface> inputTypes;
    inputTypes.reserve(inputs.size());
    llvm::transform(inputs, std::back_inserter(inputTypes), [](mlir::Value value) {
        return value.getType().cast<vpux::NDTypeInterface>();
    });
    return queryKernelMap(swKernelName, vpuDev, inputTypes, output.getType().cast<vpux::NDTypeInterface>());
}

size_t getShaveActCycleForSwKernelFunc(const std::string& swKernelName, VPU::ArchKind arch,
                                       ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                       const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    VPUX_THROW_WHEN(inputs.empty(), "No inputs identified for op {0}", swKernelName);
    VPUX_THROW_WHEN(outputs.empty(), "No outputs identified for op {0}", swKernelName);

    auto vpuDev = getVPUDeviceType(arch);

    std::unique_ptr<VPUNN::SWOperation> vpunnLayer = queryKernelMap(swKernelName, vpuDev, inputs, outputs[0]);

    return vpunnLayer != nullptr ? costModel->SHAVE(*vpunnLayer) : 1;
}

std::unique_ptr<VPUNN::SWOperation> vpux::getVPUNNSWKernelOp(VPUIP::SwKernelOp swKernelOp) {
    // E#73766: delete this check in the end of epic
    VPUX_THROW_WHEN(mlir::isa<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp()),
                    "Only single cluster task is supported by this method, op - {0}", swKernelOp->getLoc());
    // Exclude strange sw ops produced by compiler like cache_flush_invalidate op
    if (swKernelOp.getInputs().empty() || swKernelOp.getOutputBuffs().empty()) {
        return nullptr;
    }
    const auto swKernelName = getSwKernelOperationName(swKernelOp);
    auto vpuDev = getVPUDeviceType(VPU::getArch(swKernelOp.getOperation()));

    auto inputs = to_small_vector(swKernelOp->getOperands());
    auto output = swKernelOp->getResult(0);

    std::unique_ptr<VPUNN::SWOperation> vpunnLayer = queryKernelMap(swKernelName, vpuDev, inputs, output);

    return vpunnLayer;
}

std::unique_ptr<VPUNN::SWOperation> vpux::getVPUNNSWKernelOp(VPU::SWOpInterface operation) {
    auto vpuDev = VPU::getVPUDeviceType(VPU::getArch(operation));
    const auto operName = operation->getName().stripDialect().str();

    auto inputs = to_small_vector(operation->getOperands());
    auto output = operation->getResult(0);

    std::unique_ptr<VPUNN::SWOperation> vpunnLayer = queryKernelMap(operName, vpuDev, inputs, output);

    return vpunnLayer;
}

std::unique_ptr<VPUNN::SWOperation> vpux::getVPUNNSWKernelOp(VPU::SWOpInterface operation,
                                                             vpux::NDTypeInterface outputNDType,
                                                             ArrayRef<vpux::NDTypeInterface> types) {
    auto vpuDev = VPU::getVPUDeviceType(VPU::getArch(operation));
    const auto operName = operation->getName().stripDialect().str();

    std::unique_ptr<VPUNN::SWOperation> vpunnLayer = queryKernelMap(operName, vpuDev, types, outputNDType);

    return vpunnLayer;
}

size_t vpux::calculateShaveActCycles(VPUIP::SwKernelOp swKernelOp,
                                     const std::shared_ptr<VPUNN::VPUCostModel>& costModel, VPU::ArchKind arch) {
    if (swKernelOp.getInputs().empty() || swKernelOp.getOutputBuffs().empty()) {
        return 1;
    }
    auto inputNdType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto outputNdType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto inputElemType = inputNdType.getElementType();
    auto outputElemType = outputNdType.getElementType();

    // CostModel does not support F32/SI32 layers
    if (inputElemType.isF32() || outputElemType.isF32()) {
        return 1;
    }
    if (inputElemType.isSignedInteger(32) || outputElemType.isSignedInteger(32)) {
        return 1;
    }

    auto inputs = to_small_vector(swKernelOp.getInputs());
    auto outputs = to_small_vector(swKernelOp.getOutputBuffs());

    // E#73766: delete this check in the end of epic
    if (auto parentOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        inputs = to_small_vector(parentOp.getInputs());
        outputs = to_small_vector(parentOp.getOutputBuffs());
    }

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

    auto swKernelName = getSwKernelOperationName(swKernelOp);

    return getShaveActCycleForSwKernelFunc(swKernelName, arch, inputsForLargestKernelRun, outputsForLargestKernelRun,
                                           costModel);
}

size_t vpux::getDPUTaskOpCost(VPUIP::DPUTaskOp dpuTaskOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                              VPU::ArchKind arch, vpux::Logger log) {
    auto nceOp = dpuTaskOp->getParentOfType<VPUIP::NCEClusterTaskOp>();
    VPUX_THROW_WHEN(nceOp == nullptr, "The parent of dpuTaskOp {0} must be a NCEClusterTaskOp but not",
                    dpuTaskOp->getLoc());
    auto inputOneType = nceOp->getOperand(0).getType();
    auto outputType = nceOp->getResult(0).getType();

    // E#73766: delete this check in the end of epic
    if (auto nceTilingParent = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        inputOneType = nceTilingParent.getInputs()[0].getType();
        outputType = nceTilingParent.getOutputs()[0].getType();
    }

    auto inputElemType = inputOneType.cast<vpux::NDTypeInterface>().getElementType();
    auto outputElemType = outputType.cast<vpux::NDTypeInterface>().getElementType();

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
        VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel, VPU::ArchKind arch,
        vpux::Logger log) {
    std::vector<std::pair<int64_t, size_t>> nceVariantCyclePerCluster;
    for (auto dpuTaskOp : nceOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        auto clusterId = dpuTaskOp.getClusterId().value_or(0);
        nceVariantCyclePerCluster.push_back({clusterId, getDPUTaskOpCost(dpuTaskOp, costModel, arch, log)});
    }
    return nceVariantCyclePerCluster;
}

size_t vpux::calculateNceCycles(VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                VPU::ArchKind arch, vpux::Logger log, int64_t numDPU) {
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
