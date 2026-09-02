//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/dma_cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <vpu/dma_types.h>
#include <vpu/dpu_types.h>
#include <vpu_cost_model.h>

#include <algorithm>
#include <climits>
#include <cmath>

using namespace vpux;

// To be updated regularly depending on latest VPUNN cost model accuracy progress
bool vpux::isDMACostModelAccurate(config::ArchKind arch) {
    // E#226235: To be updated to include all arches
    return arch == config::ArchKind::NPU37XX || arch == config::ArchKind::NPU50XX;
}

bool vpux::isDMAPortSplittingSupported(config::ArchKind arch) {
    // Enable DMA port splitting cost estimation only for 50XX for now
    return arch == config::ArchKind::NPU50XX;
}

// Stride DMA cost is inaccurate by cost model, so use this variable to help correct the cost value
// TODO: Ticket E#213641, remove this variable when stride DMA cost is accurate by VPUNN cost model
constexpr double strideDMACorrectionThresholdInBitsV1 = 512;
constexpr double strideDMACorrectionThresholdInBitsV2 = 1024;
// Adjusted stride DMA correction threshold for temporal-tiling full-search,
// to resolve auto-tuning performance issues caused by inaccurate stride DMA cost modeling.
constexpr double strideDMACorrectionThresholdInBitsFullSearch = 2048;

double vpux::getStrideDMACorrectionThresholdByArch(config::ArchKind arch) {
    // Experimental threshold to correct 50XX DMA cost
    if (arch == config::ArchKind::NPU50XX) {
        return strideDMACorrectionThresholdInBitsV2;
    }
    return strideDMACorrectionThresholdInBitsV1;
}

static double getStrideDMACorrectionThresholdByArchFullSearchVersion(config::ArchKind arch) {
    // Experimental threshold to correct 50XX+ DMA cost
    if (arch >= config::ArchKind::NPU50XX) {
        return strideDMACorrectionThresholdInBitsFullSearch;
    }
    return strideDMACorrectionThresholdInBitsV1;
}

// Apply stride DMA cost correction for a single tile type.
// Returns true if a correction was applied, false otherwise.
// TODO: Ticket E#213641, remove this after stride DMA cost is accurate
// isFullSearchVersion: Stride DMA threshold is adjusted for the tiling full search version
// To resolve the performance issue in auto-tuning which are caused by the inaccurate stride DMA cost
bool vpux::applyStrideDMACorrectionForTile(vpux::NDTypeInterface tileType, bool isStridedDMA, uint32_t& cost,
                                           config::ArchKind arch, bool isFullSearchVersion) {
    if (!isStridedDMA) {
        return false;
    }
    const auto dimOrder = tileType.getDimsOrder();
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    const Bit elemSize = tileType.getElemTypeSize();
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tileType)) {
        tileType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }

    Bit continuousBitsOnLowestDim;
    if (auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tileType)) {
        continuousBitsOnLowestDim = distributedType.getLargestCompactShape()[lowestDim] * elemSize;
    } else {
        continuousBitsOnLowestDim = tileType.getShape()[lowestDim] * elemSize;
    }

    auto curStrideDMACorrectionThreshold = isFullSearchVersion
                                                   ? getStrideDMACorrectionThresholdByArchFullSearchVersion(arch)
                                                   : getStrideDMACorrectionThresholdByArch(arch);
    if (continuousBitsOnLowestDim.count() < curStrideDMACorrectionThreshold) {
        auto factor = curStrideDMACorrectionThreshold / continuousBitsOnLowestDim.count();
        cost = checked_cast<uint32_t>(std::floor(factor * cost));
        return true;
    }
    return false;
}

// Apply stride DMA cost correction across all tiles.
bool vpux::correctStrideDMACostOnAllTiles(
        ArrayRef<std::vector<std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>>> tilesTypes,
        const std::function<vpux::NDTypeInterface(
                ArrayRef<std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>>)>& tileTypeGetter,
        SmallVector<uint32_t>& dmaCost, bool isStridedDMA, config::ArchKind arch) {
    if (!isStridedDMA) {
        return false;
    }
    VPUX_THROW_WHEN(dmaCost.size() != tilesTypes.size(), "DMA costs size mismatches with tiled types");

    for (auto tileId : irange(tilesTypes.size())) {
        auto currentTileType = tileTypeGetter(tilesTypes[tileId]);
        applyStrideDMACorrectionForTile(currentTileType, isStridedDMA, dmaCost[tileId], arch);
    }

    return true;
}

VPUNN::MemoryLocation vpux::getMemoryLocation(mlir::Type type) {
    auto memKind = mlir::cast<vpux::NDTypeInterface>(type).getMemoryKind();
    if (memKind == VPU::MemoryKind::CMX_NN) {
        return VPUNN::MemoryLocation::CMX;
    }

    return VPUNN::MemoryLocation::DRAM;
}

// Forward declarations — defined in cost_model_utils.cpp (same library, external linkage).
VPUNN::DataType getElementType(mlir::Type type, VPUNN::VPUDevice vpuDevice);
VPUNN::VPUTensor getVPUNNTensor(ShapeRef tensorShape, VPUNN::DataType dataType);

namespace {

VPUNN::VPUTensor getVPUNNTensorMultiCluster(ArrayRef<Shape> tensorShapes, VPUNN::DataType dataType) {
    unsigned int totalShape = 0;
    for (size_t idx = 0; idx < tensorShapes.size(); idx++) {
        totalShape += static_cast<unsigned int>(tensorShapes[idx].totalSize());
    }
    return VPUNN::VPUTensor({totalShape, 1, 1, 1}, dataType);
}

// Per-cluster shape selection for a segmented/overlapped distributed DMA: with
// multiple DMA ports the transaction is approximated by the largest compact tile, otherwise by the
// per-cluster compute shapes.
SmallVector<Shape> getLegacySegmentedShapes(vpux::VPU::DistributedTensorType distributedType, int64_t numDMAPorts) {
    if (numDMAPorts > 1) {
        return SmallVector<Shape>{distributedType.getLargestCompactShape()};
    }
    return distributedType.getPerClusterComputeShapes();
}

size_t getDistributedSegmentedDMACost(vpux::NDTypeInterface tensorType, config::ArchKind archKind,
                                      VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                      int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "Number of DMA ports must be at least 1");
    auto distributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tensorType);
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    auto elemType = tensorType.getElementType();

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        const auto shapes = distributedTensorType.getPerClusterMemoryShapes();
        size_t cost = 0;
        for (const auto& shape : shapes) {
            auto vpuTensor = getVPUNNTensor(shape, getElementType(elemType, vpuDevice));
            // With no mem location specified the DMA function will default to DDR->CMX
            cost += costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
        }
        return cost / numDMAPorts;
    }

    const auto shapes = getLegacySegmentedShapes(distributedTensorType, numDMAPorts);
    auto vpuTensor = getVPUNNTensorMultiCluster(shapes, getElementType(elemType, vpuDevice));
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

size_t getDistributedSegmentedDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      config::ArchKind /*archKind*/, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "Number of DMA ports must be at least 1");

    auto inDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);
    VPUX_THROW_WHEN(inDistributedTensorType == nullptr || outDistributedTensorType == nullptr,
                    "Expected distributed tensor types for segmented DMA cost, but got in={0}, out={1}", inTensorType,
                    outTensorType);

    auto inElemType = inTensorType.getElementType();
    auto outElemType = outTensorType.getElementType();

    // E#219110: current behaviour is inconsistent with DMA port splitting in other functions,
    // where it's only applied in npu5XX at the moment.
    auto getShapes = [&](auto distributedType, auto plainType) -> SmallVector<Shape> {
        if (distributedType) {
            // For distributed segmented DMA, transaction will be split between ports and executing
            // in parallel when there are multiple DMA ports available.
            // When enabling architectures whose number of tiles is not equal to number of DMA ports, using
            // simply the largest size in tiles to calculate cost is not accurate, see E#84432
            return getLegacySegmentedShapes(distributedType, numDMAPorts);
        }
        return SmallVector<Shape>{plainType.getShape().raw()};
    };
    SmallVector<Shape> inShapes = getShapes(inDistributedTensorType, inTensorType);
    SmallVector<Shape> outShapes = getShapes(outDistributedTensorType, outTensorType);

    auto inTensor = getVPUNNTensorMultiCluster(inShapes, getElementType(inElemType, vpuDevice));
    auto outTensor = getVPUNNTensorMultiCluster(outShapes, getElementType(outElemType, vpuDevice));

    return costModel->DMA(vpuDevice, inTensor, outTensor, getMemoryLocation(inTensorType),
                          getMemoryLocation(outTensorType));
}

size_t getDistributedBroadcastDMACost(vpux::NDTypeInterface tensorType, config::ArchKind /*archKind*/,
                                      VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                      int64_t /*numDMAPorts*/) {
    auto shape = tensorType.getShape();
    auto elemType = tensorType.getElementType();
    auto vpuTensor = getVPUNNTensor(shape, getElementType(elemType, vpuDevice));
    // With no mem location specified the DMA function will default to DDR->CMX
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

size_t getDistributedBroadcastDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      config::ArchKind /*archKind*/, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t /*numDMAPorts*/) {
    auto inVpuTensor =
            getVPUNNTensor(inTensorType.getShape(), getElementType(inTensorType.getElementType(), vpuDevice));
    auto outVpuTensor =
            getVPUNNTensor(outTensorType.getShape(), getElementType(outTensorType.getElementType(), vpuDevice));
    return costModel->DMA(vpuDevice, inVpuTensor, outVpuTensor, getMemoryLocation(inTensorType),
                          getMemoryLocation(outTensorType));
}

using GetDMAOnVPUNN = size_t (*)(vpux::NDTypeInterface tensortType, config::ArchKind archKind,
                                 VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                 int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetDMAOnVPUNN> spillingCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::SEGMENTED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::MULTICASTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
};

using GetIODMAOnVPUNN = size_t (*)(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                   config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetIODMAOnVPUNN> spillingIOCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::SEGMENTED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::MULTICASTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
};

// Unwraps a sparse tensor type to its data component, leaving non-sparse types unchanged.
vpux::NDTypeInterface unwrapSparseData(vpux::NDTypeInterface type) {
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(type)) {
        return mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    return type;
}

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                    config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto operandType = innerOperand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    VPUX_THROW_WHEN(numDMAPorts <= 0, "Invalid number of DMA ports; should be > 0, but actual value is {0}",
                    numDMAPorts);

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
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

}  // namespace

// Used by VPU dialect
// Single-type overload. Use for a same-type DMA when only the NDTypeInterface is known, not an mlir::Value.
// Used by generic VPU cost paths that do not know the transfer direction: it defaults to DDR->CMX in
// VPUCostModel::DMA
size_t vpux::getDMACost(vpux::NDTypeInterface tensorType, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    tensorType = unwrapSparseData(tensorType);

    auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tensorType);

    const auto elementType = tensorType.getElementType();

    if (distributedType != nullptr) {
        const auto dmaCostFunc = spillingCostMapVPUNN.at(distributedType.getDistribution().getMode().getValue());
        return dmaCostFunc(tensorType, archKind, vpuDevice, costModel, numDMAPorts);
    }

    const auto vpunnTensor = getVPUNNTensor(tensorType.getShape(), getElementType(elementType, vpuDevice));
    // no inLoc/outLoc provided, use default DMA direction (DDR->CMX) in VPUCostModel::DMA
    return costModel->DMA(vpuDevice, vpunnTensor, vpunnTensor);
}

// Input/output-type overload. Use when only the types are known (no mlir::Value) and the input and
// output types differ, e.g. a DDR<->CMX copy. Used by VPU dialect passes (MoveReflectPadToCMX) and
// LayerVPUNNCost.
size_t vpux::getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                        config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    inTensorType = unwrapSparseData(inTensorType);
    outTensorType = unwrapSparseData(outTensorType);

    const auto inElementType = inTensorType.getElementType();
    const auto outElementType = outTensorType.getElementType();

    auto inDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);

    if (inDistributedType || outDistributedType) {
        auto distributionMode = inDistributedType ? inDistributedType.getDistribution().getMode().getValue()
                                                  : outDistributedType.getDistribution().getMode().getValue();
        const auto dmaCostFunc = spillingIOCostMapVPUNN.at(distributionMode);
        return dmaCostFunc(inTensorType, outTensorType, archKind, vpuDevice, costModel, numDMAPorts);
    }

    const auto inVpunnTensor = getVPUNNTensor(inTensorType.getShape(), getElementType(inElementType, vpuDevice));
    const auto outVpunnTensor = getVPUNNTensor(outTensorType.getShape(), getElementType(outElementType, vpuDevice));
    auto cost = costModel->DMA(vpuDevice, inVpunnTensor, outVpunnTensor, getMemoryLocation(inTensorType),
                               getMemoryLocation(outTensorType));
    return static_cast<size_t>(cost);
}

// Used in VPUIP
// Per-operation overload. Use when the actual DMA/Copy op operands (mlir::Value) are available: the
// VPUIP op cycle-cost methods (Copy/NNDMA/Convert DMA in copy.cpp/dma.cpp), the per-op dispatch
// below, and the FeasibleMemoryScheduler spill-cost path. It inspects the operands for distributed
// buffer types and whether extra per-cluster DMAs are required.
size_t vpux::getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    auto inElemType = getElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), vpuDevice);
    auto outElemType = getElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), vpuDevice);

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType) && extraDMAsRequired(input)) {
        return calculateMultiClusterDMACost(input, inElemType, outElemType, archKind, vpuDevice, costModel,
                                            numDMAPorts);
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType) && extraDMAsRequired(output)) {
        return calculateMultiClusterDMACost(output, inElemType, outElemType, archKind, vpuDevice, costModel,
                                            numDMAPorts);
    }

    auto inputShape = getShape(input);
    auto outputShape = getShape(output);

    // TODO: add layout info to VPUNN tensors
    auto cost = costModel->DMA(vpuDevice, {getVPUNNTensor(inputShape, inElemType)},
                               {getVPUNNTensor(outputShape, outElemType)}, getMemoryLocation(inputType),
                               getMemoryLocation(outputType));

    return static_cast<size_t>(cost);
}

namespace {

// Legacy analytical (non-VPUNN) spill cost. Reached only through `spillingAnalyticalCostMap` from
// LayerCostModel::getDMACostOfType, whose _arch guard routes every other arch to the VPUNN path.
double getAnalyticalSpillingCostForNonMultiCluster(vpux::NDTypeInterface tensorType, const VPU::DistributionInfo&,
                                                   double ddrLatency, double ddrBandwidth, int64_t /*numDMAPorts*/) {
    // calculate the data byte size need copy from cmx to ddr or vice versa
    const auto totalSize = static_cast<double>(tensorType.getTotalAllocSize().count());
    return ddrLatency + totalSize / ddrBandwidth;
}

// Legacy analytical (non-VPUNN) spill cost. A DUPLICATED/MULTICASTED broadcast is a single transfer,
// so numDMAPorts is intentionally ignored here for the same reason: single-DMA port splitting is credited only on the
// VPUNN path.
double getAnalyticalSpillingCostForDuplicated(vpux::NDTypeInterface tensorType,
                                              const VPU::DistributionInfo& distribution, double ddrLatency,
                                              double ddrBandwidth, int64_t /*numDMAPorts*/) {
    VPU::TensorDistributionMap distributionMap;
    distributionMap.insert(std::make_pair(tensorType, distribution));
    const auto totalSize = VPU::getTotalAllocSizeWithDistribution(tensorType, distributionMap);
    return ddrLatency + totalSize.count() / ddrBandwidth;
}

// Analytical (non-VPUNN) segmented spill cost.
//
// It models the ATOMIC whole-segment round-robin: the N per-cluster segments are distributed across
// the P DMA ports (segment i -> port i % P) and the wall-clock is the busiest port (ceil(N/P) worth
// of segments). This deliberately does NOT credit tail-splitting: SplitDMAToBalanceLoad can split the
// N % P leftover segments to reach a fully balanced sum/P, but that balance pass is exercised on
// NPU5 (P=2), whose cost is computed by the VPUNN path (see modelSegmentedDMAPortCost in
// cost_model_utils.cpp). Keeping this estimate conservative avoids over-crediting parallelism for the
// arch handled here.
double getAnalyticalSpillingCostForSegmented(vpux::NDTypeInterface tensorType,
                                             const VPU::DistributionInfo& distribution, double ddrLatency,
                                             double ddrBandwidth, int64_t numDMAPorts) {
    SmallVector<Shape> perClusterMemShapes{};
    if (distribution.getMemoryShapes().size() == 0) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(tensorType.getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Shape {0}, Unsupported distribution: {1}",
                          tensorType.getShape(), distribution);
        perClusterMemShapes = optionalPerClusterMemoryShapes.value();
    } else {
        for (auto& shape : distribution.getMemoryShapes()) {
            perClusterMemShapes.push_back(Shape(shape));
        }
    }

    // Aggregate the total size which needs to be transferred on each DMA port
    auto totalSizeOnPorts = SmallVector<int64_t>(numDMAPorts, 0);
    for (size_t i = 0; i < perClusterMemShapes.size(); ++i) {
        totalSizeOnPorts[i % numDMAPorts] += perClusterMemShapes[i].totalSize();
    }
    // Considering multiple ports used in parallel, only take into account the largest size to transfer
    auto totalSize = *std::max_element(totalSizeOnPorts.begin(), totalSizeOnPorts.end());

    const Bit elemSize = tensorType.getElemTypeSize();
    totalSize = alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>().count();
    return ddrLatency + static_cast<double>(totalSize) / ddrBandwidth;
}

using GetSpillingCostCB = double (*)(vpux::NDTypeInterface, const VPU::DistributionInfo& distribution,
                                     double ddrLatency, double ddrBandwidth, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetSpillingCostCB> spillingAnalyticalCostMap{
        // using  DistributionMode::NONE for single clustering case
        {VPU::DistributionMode::NONE, getAnalyticalSpillingCostForNonMultiCluster},
        {VPU::DistributionMode::DUPLICATED, getAnalyticalSpillingCostForDuplicated},
        {VPU::DistributionMode::SEGMENTED, getAnalyticalSpillingCostForSegmented},
        {VPU::DistributionMode::OVERLAPPED, getAnalyticalSpillingCostForSegmented},
        {VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, getAnalyticalSpillingCostForSegmented},
        {VPU::DistributionMode::MULTICASTED, getAnalyticalSpillingCostForDuplicated},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getAnalyticalSpillingCostForDuplicated},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getAnalyticalSpillingCostForDuplicated},
};

}  // namespace

double vpux::getAnalyticalDMACost(vpux::NDTypeInterface srcType, const VPU::DistributionInfo& distribution,
                                  double ddrLatency, double ddrBandwidth, int64_t numDMAPorts) {
    auto srcMode = distribution.getDistributionMode();
    auto spillingReadCostFunc = spillingAnalyticalCostMap.at(srcMode);
    return spillingReadCostFunc(srcType, distribution, ddrLatency, ddrBandwidth, numDMAPorts);
}
