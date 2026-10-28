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
#include <numeric>

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

// This function throws to help discover the cases in which the memKind is actually not known
// and the DMA direction should be passed explicitly to getDMACost util
VPUNN::MemoryLocation vpux::getMemoryLocation(mlir::Type type) {
    auto memKind = mlir::cast<vpux::NDTypeInterface>(type).getMemoryKind();
    switch (memKind) {
    case VPU::MemoryKind::DDR:
        return VPUNN::MemoryLocation::DRAM;
    case VPU::MemoryKind::CMX_NN:
        return VPUNN::MemoryLocation::CMX;
    // HW register / MMIO accesses (e.g. profiling timer reads, barrier-config and workpoint DMAs) are
    // off-CMX transfers. VPUNN has no dedicated Register location, so model them as DRAM, which also
    // matches the historical behavior of mapping every non-CMX kind to DRAM.
    case VPU::MemoryKind::Register:
        return VPUNN::MemoryLocation::DRAM;
    default:
        VPUX_THROW("Unsupported memory kind {0} for type {1}", memKind, type);
    }
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

SmallVector<size_t> getVPUNNMultiClusterCosts(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes,
                                              VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                              VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                              VPUNN::VPUDevice vpuDevice,
                                              const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    VPUX_THROW_UNLESS(inShapes.size() == outShapes.size(),
                      "Inconsistent number of per-cluster shapes for in={0} and out={1}", inShapes.size(),
                      outShapes.size());

    SmallVector<size_t> costs;
    costs.reserve(inShapes.size());
    for (auto idx : irange(inShapes.size())) {
        auto inVpuTensor = getVPUNNTensor(inShapes[idx], inElemType);
        auto outVpuTensor = getVPUNNTensor(outShapes[idx], outElemType);
        costs.push_back(costModel->DMA(vpuDevice, inVpuTensor, outVpuTensor, srcMemLoc, dstMemLoc));
    }
    return costs;
}

size_t getVPUNNMultiClusterCostSum(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes, VPUNN::DataType inElemType,
                                   VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                                   VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto costs = getVPUNNMultiClusterCosts(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                           vpuDevice, costModel);
    return std::accumulate(costs.begin(), costs.end(), static_cast<size_t>(0));
}

// Optimistic multi-port estimate: cost = (sum of all per-cluster transfer costs) / numDMAPorts.
// The per-cluster transfers are treated as a single divisible workload spread perfectly evenly over the
// ports, i.e. the leftover "tail" (when the number of clusters is not a multiple of numDMAPorts) is
// assumed to be splittable at no extra cost. This is a optimistic, since it doesn't consider all the
// conditions necessary for the split to be possible (see SplitDmaToBalanceLoad VPUIP pass)
size_t trivialTailSplitDMACost(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes, VPUNN::DataType inElemType,
                               VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                               VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                               const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto mcCostSum = getVPUNNMultiClusterCostSum(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                                 vpuDevice, costModel);
    return mcCostSum / numDMAPorts;
}

// Conservative multi-port estimate: each per-cluster transfer is assigned whole to one port in
// round-robin order (cluster i -> port i % numDMAPorts), and the cost is the load of the busiest port.
// Since the ports run in parallel, the makespan is the longest port, and an odd tail transfer is charged
// in full to whichever port receives it rather than being split. This is an upper bound on the real cost, since it
// assumes always no splitting of the tail DMAs. Example, 3 clusters costing {100, 100, 100} on 2 ports -> port0 = 200,
// port1 = 100 -> longestPathCost = 200.
size_t roundRobinUnsplitTailDMACost(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes, VPUNN::DataType inElemType,
                                    VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                                    VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto costs = getVPUNNMultiClusterCosts(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                           vpuDevice, costModel);

    // Aggregate the total size which needs to be transferred on each DMA port
    auto totalCostOnPorts = SmallVector<size_t>(numDMAPorts, 0);
    for (size_t i = 0; i < costs.size(); ++i) {
        totalCostOnPorts[i % numDMAPorts] += costs[i];
    }
    // Considering multiple ports used in parallel and the remaining tail on the last port, the total cost is determined
    // by the longest path (no tail splitting); the cost estimation is conservative (pessimistic).
    auto longestPathCost = *std::max_element(totalCostOnPorts.begin(), totalCostOnPorts.end());

    return longestPathCost;
}

// Strategy used to estimate the cost of a distributed DMA spread across multiple DMA ports
enum class DMAPortSplitCostMode {
    // Optimistic: the per-cluster costs are summed and divided evenly across all ports,
    // assuming the tail on the last port can be freely split conditionlessly.
    TrivialTailSplit,
    // Conservative: per-cluster transfers are assigned round-robin to ports and the cost is
    // the longest-loaded port, without splitting the tail.
    RoundRobinUnsplitTail,
};

StringLiteral stringifyEnum(DMAPortSplitCostMode val) {
    switch (val) {
    case DMAPortSplitCostMode::TrivialTailSplit:
        return "TrivialTailSplit";
    case DMAPortSplitCostMode::RoundRobinUnsplitTail:
        return "RoundRobinUnsplitTail";
    default:
        return "<UNKNOWN>";
    }
}

// Dispatches to the requested multi-port DMA cost estimation strategy.
// TODO E#219110 : to be extended with accurate tail split mode for segmented/overlapped distributed DMA and with
// DUPLICATED buffers for broadcast distributed DMA
size_t getMultiPortDMACost(DMAPortSplitCostMode mode, ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes,
                           VPUNN::DataType inElemType, VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                           VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                           const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    switch (mode) {
    case DMAPortSplitCostMode::TrivialTailSplit:
        return trivialTailSplitDMACost(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc, vpuDevice,
                                       costModel, numDMAPorts);
    case DMAPortSplitCostMode::RoundRobinUnsplitTail:
        return roundRobinUnsplitTailDMACost(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                            vpuDevice, costModel, numDMAPorts);
    }
    VPUX_THROW("Unsupported DMA port split cost mode: {0}", stringifyEnum(mode));
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

size_t getDistributedSegmentedDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                      config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "Number of DMA ports must be at least 1");

    auto inDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);
    VPUX_THROW_WHEN(inDistributedTensorType == nullptr,
                    "Expected input distributed tensor type for segmented DMA cost, but got in={0}", inTensorType);
    VPUX_THROW_WHEN(outDistributedTensorType == nullptr,
                    "Expected output distributed tensor type for segmented DMA cost, but got out={0}", outTensorType);

    auto inElemType = inTensorType.getElementType();
    auto outElemType = outTensorType.getElementType();

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        const auto inShapes = inDistributedTensorType.getPerClusterMemoryShapes();
        const auto outShapes = outDistributedTensorType.getPerClusterMemoryShapes();
        VPUX_THROW_WHEN(inShapes.size() != outShapes.size(),
                        "Inconsistent number of per-cluster shapes for in (num={0}) and out (num={1})", inShapes.size(),
                        outShapes.size());

        return getMultiPortDMACost(DMAPortSplitCostMode::TrivialTailSplit, inShapes, outShapes,
                                   getElementType(inElemType, vpuDevice), getElementType(outElemType, vpuDevice),
                                   srcMemLoc, dstMemLoc, vpuDevice, costModel, numDMAPorts);
    }

    SmallVector<Shape> inShapes = getLegacySegmentedShapes(inDistributedTensorType, numDMAPorts);
    SmallVector<Shape> outShapes = getLegacySegmentedShapes(outDistributedTensorType, numDMAPorts);

    auto inTensor = getVPUNNTensorMultiCluster(inShapes, getElementType(inElemType, vpuDevice));
    auto outTensor = getVPUNNTensorMultiCluster(outShapes, getElementType(outElemType, vpuDevice));

    return costModel->DMA(vpuDevice, inTensor, outTensor, srcMemLoc, dstMemLoc);
}

size_t getDistributedBroadcastDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                      config::ArchKind /*archKind*/, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t /*numDMAPorts*/) {
    auto inVpuTensor =
            getVPUNNTensor(inTensorType.getShape(), getElementType(inTensorType.getElementType(), vpuDevice));
    auto outVpuTensor =
            getVPUNNTensor(outTensorType.getShape(), getElementType(outTensorType.getElementType(), vpuDevice));
    return costModel->DMA(vpuDevice, inVpuTensor, outVpuTensor, srcMemLoc, dstMemLoc);
}

using GetIODMAOnVPUNN = size_t (*)(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                   VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                   config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetIODMAOnVPUNN> spillingIOCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::SEGMENTED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
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

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, mlir::Type inputType, mlir::Type outputType,
                                    VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                    config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto operandType = innerOperand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_WHEN(distributedType == nullptr, "Unsupported operand type {0}", operandType);

    auto inVpunnElemType = getElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), vpuDevice);
    auto outVpunnElemType = getElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), vpuDevice);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    VPUX_THROW_WHEN(numDMAPorts <= 0, "Invalid number of DMA ports; should be > 0, but actual value is {0}",
                    numDMAPorts);

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        return getMultiPortDMACost(DMAPortSplitCostMode::TrivialTailSplit, perClusterShapes, perClusterShapes,
                                   inVpunnElemType, outVpunnElemType, srcMemLoc, dstMemLoc, vpuDevice, costModel,
                                   numDMAPorts);
    }

    return static_cast<size_t>(
            costModel->DMA(vpuDevice, {getVPUNNTensorMultiCluster(perClusterShapes, inVpunnElemType)},
                           {getVPUNNTensorMultiCluster(perClusterShapes, outVpunnElemType)}, srcMemLoc, dstMemLoc));
}

bool extraDMAsRequired(mlir::Value innerOperand) {
    if (auto inputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(innerOperand.getType())) {
        auto distribution = inputType.getDistribution();
        auto distributionMode = distribution.getMode().getValue();
        // E#228872 : missing SEGMENTED|OVERLAPPED combination case
        return distributionMode == VPU::DistributionMode::SEGMENTED ||
               distributionMode == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

StringLiteral stringifyEnum(vpux::DmaDirectionType val) {
    switch (val) {
    case vpux::DmaDirectionType::CMX_TO_DDR:
        return "CMX_TO_DDR";
    case vpux::DmaDirectionType::DDR_TO_CMX:
        return "DDR_TO_CMX";
    case vpux::DmaDirectionType::CMX_TO_CMX:
        return "CMX_TO_CMX";
    case vpux::DmaDirectionType::DDR_TO_DDR:
        return "DDR_TO_DDR";
    default:
        return "<UNKNOWN>";
    }
}

// Returns [srcMemLoc, dstMemLoc] for the given DMA direction. Throws if the direction is unsupported.
std::pair<VPUNN::MemoryLocation, VPUNN::MemoryLocation> getVPUNNMemLocations(vpux::DmaDirectionType dmaDirection) {
    switch (dmaDirection) {
    case vpux::DmaDirectionType::CMX_TO_DDR:
        return {VPUNN::MemoryLocation::CMX, VPUNN::MemoryLocation::DRAM};
    case vpux::DmaDirectionType::DDR_TO_CMX:
        return {VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX};
    case vpux::DmaDirectionType::CMX_TO_CMX:
        return {VPUNN::MemoryLocation::CMX, VPUNN::MemoryLocation::CMX};
    case vpux::DmaDirectionType::DDR_TO_DDR:
        return {VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::DRAM};
    default:
        VPUX_THROW("Unsupported DMA direction: {0}", stringifyEnum(dmaDirection));
    }
}

// Core of the VPU type-based DMA cost. Takes the VPUNN in/out memory locations explicitly so callers can
// either derive them from the operand memory kinds or force them from a DMA direction.
size_t getTypeDMACostWithMemLoc(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);

    if (inDistributedType || outDistributedType) {
        const auto distributionMode = inDistributedType ? inDistributedType.getDistribution().getMode().getValue()
                                                        : outDistributedType.getDistribution().getMode().getValue();
        const auto dmaCostFunc = spillingIOCostMapVPUNN.at(distributionMode);
        return dmaCostFunc(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                           numDMAPorts);
    }

    const auto inVpunnTensor =
            getVPUNNTensor(inTensorType.getShape(), getElementType(inTensorType.getElementType(), vpuDevice));
    const auto outVpunnTensor =
            getVPUNNTensor(outTensorType.getShape(), getElementType(outTensorType.getElementType(), vpuDevice));
    // TODO E#227047: add layout info to VPUNN tensors
    return static_cast<size_t>(costModel->DMA(vpuDevice, inVpunnTensor, outVpunnTensor, srcMemLoc, dstMemLoc));
}

// Core of the VPUIP per-operation DMA cost. Takes the VPUNN in/out memory locations explicitly so
// callers can either derive them from the operand types or force them from a DMA direction.
size_t getValueDMACostWithMemLoc(mlir::Value input, mlir::Value output, VPUNN::MemoryLocation srcMemLoc,
                                 VPUNN::MemoryLocation dstMemLoc, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                 const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    // E#228695 WA: Multi-cluster CMX->CMX transfers keep the legacy VPUNN default (DRAM->CMX) rather than their
    // real memory locations. VPUNN prices any transfer that touches DRAM the same in both
    // directions, so CMX->CMX is the only direction whose cost actually changes here and it is
    // cheaper.
    // The override is deliberately scoped to the multi-cluster path below: the plain path keeps the
    // caller's locations, which is what a CMX->CMX spill of a non-distributed buffer relies on.
    // The effective behaviour coincides with the legacy one, and it needs to be fixed alongside
    // the scheduler to remove this workaround.
    auto mcSrcMemLoc = srcMemLoc;
    auto mcDstMemLoc = dstMemLoc;
    if (srcMemLoc == VPUNN::MemoryLocation::CMX && dstMemLoc == VPUNN::MemoryLocation::CMX) {
        mcSrcMemLoc = VPUNN::MemoryLocation::DRAM;
        mcDstMemLoc = VPUNN::MemoryLocation::CMX;
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType) && extraDMAsRequired(input)) {
        return calculateMultiClusterDMACost(input, inputType, outputType, mcSrcMemLoc, mcDstMemLoc, archKind, vpuDevice,
                                            costModel, numDMAPorts);
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType) && extraDMAsRequired(output)) {
        return calculateMultiClusterDMACost(output, inputType, outputType, mcSrcMemLoc, mcDstMemLoc, archKind,
                                            vpuDevice, costModel, numDMAPorts);
    }

    auto inElemType = getElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), vpuDevice);
    auto outElemType = getElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), vpuDevice);

    auto inputShape = getShape(input);
    auto outputShape = getShape(output);

    // TODO E#227047: add layout info to VPUNN tensors
    auto cost = costModel->DMA(vpuDevice, {getVPUNNTensor(inputShape, inElemType)},
                               {getVPUNNTensor(outputShape, outElemType)}, srcMemLoc, dstMemLoc);

    return static_cast<size_t>(cost);
}

}  // namespace

// Used by VPU dialect
// Single-type overload. Use for a same-type DMA when only the NDTypeInterface is known in generic VPU cost paths
size_t vpux::getDMACost(vpux::NDTypeInterface tensorType, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                        vpux::DmaDirectionType dmaDirection) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    tensorType = unwrapSparseData(tensorType);
    const auto [srcMemLoc, dstMemLoc] = getVPUNNMemLocations(dmaDirection);
    // Same-type transfer: pass the type for both sides; only the memory locations differ.
    return getTypeDMACostWithMemLoc(tensorType, tensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                                    numDMAPorts);
}

// Input/output-type overload. Use when only the types are known and the input and
// output types differ, e.g. a DRAM<->CMX copy. Used by VPU dialect passes MoveReflectPadToCMX
size_t vpux::getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                        config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    inTensorType = unwrapSparseData(inTensorType);
    outTensorType = unwrapSparseData(outTensorType);
    const auto srcMemLoc = getMemoryLocation(inTensorType);
    const auto dstMemLoc = getMemoryLocation(outTensorType);
    return getTypeDMACostWithMemLoc(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                                    numDMAPorts);
}

// Used in VPUIP
// Per-operation overload. Use when the actual DMA/Copy Op operands (mlir::Value) are available (Copy/NNDMA/Convert DMA)
size_t vpux::getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    const auto srcMemLoc = getMemoryLocation(input.getType());
    const auto dstMemLoc = getMemoryLocation(output.getType());
    return getValueDMACostWithMemLoc(input, output, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel, numDMAPorts);
}

// DMA-direction overload. The input and output are the same spilled buffer, so pass the dma direction explicitly.
// Used in FeasibleMemoryScheduler spill-cost path
size_t vpux::getDMACost(mlir::Value input, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                        vpux::DmaDirectionType dmaDirection) {
    const auto [srcMemLoc, dstMemLoc] = getVPUNNMemLocations(dmaDirection);
    return getValueDMACostWithMemLoc(input, input, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel, numDMAPorts);
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
