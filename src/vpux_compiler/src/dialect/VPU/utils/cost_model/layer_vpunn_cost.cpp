//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

using namespace vpux;
using namespace VPU;

namespace {

// Stride DMA cost is inaccurate by cost model, so use this variable to help correct the cost value
// TODO: Ticket E#135490, remove this variable when stride DMA cost is accurate by VPUNN cost model
constexpr auto strideDMACorrectionThresholdInBits = 512;

bool isTiledOnLowestDim(ShapeRef tileAxis, DimsOrder dimOrder) {
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    const auto axis = tileAxis[lowestDim];
    return axis > 1;
}

}  // namespace

MultiClusterStrategySetter::MultiClusterStrategySetter(mlir::Operation* operation, VPU::MultiClusterStrategy strategy)
        : _operation(operation) {
    setTemporaryStrategy(strategy);
}

MultiClusterStrategySetter::~MultiClusterStrategySetter() {
    removeTemporaryStrategy();
}

void MultiClusterStrategySetter::removeTemporaryStrategy() {
    if (auto childClusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(_operation)) {
        if (_origStrategy.has_value()) {
            childClusterOp.setMultiClusterStrategy(_origStrategy.value());
        } else {
            _operation->removeAttr(multiClusterStrategy);
        }
    }
}

void MultiClusterStrategySetter::setTemporaryStrategy(VPU::MultiClusterStrategy tempStrategy) {
    if (auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(_operation)) {
        _origStrategy = clusterOp.getMultiClusterStrategy();
        clusterOp.setMultiClusterStrategy(tempStrategy);
    }
}

StrategyCost LayerVPUNNCost::getStrategyCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const {
    if (mlir::isa<VPU::NCEPermuteOp>(operation)) {
        return getSimpleLayerCost(operation->getResult(0).getType().cast<vpux::NDTypeInterface>(), parameters);
    } else if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation)) {
        return getNCELayerCost(nceOp, parameters);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(operation)) {
        return getSWLayerCost(swOp, parameters);
    } else if (VPU::isPureViewOp(operation)) {
        return 0.0;
    } else {
        _log.trace("Unsupported op type {0} at {1}", operation->getName(), operation->getLoc());
        return getSimpleLayerCost(operation->getResult(0).getType().cast<vpux::NDTypeInterface>(), parameters);
    }
}

StrategyCost LayerVPUNNCost::getSpillingTypeCost(vpux::NDTypeInterface type,
                                                 const std::optional<ShapeRef>& tileAxis) const {
    StrategyCost cost = getDMACost(type, _vpuDevice, _vpunnCostModel, _numDMAPorts);

    if (tileAxis.has_value() && isTiledOnLowestDim(tileAxis.value(), type.getDimsOrder())) {
        cost = correctStrideDMACost(type, cost);
    }
    return cost;
}

SmallVector<StrategyCost> LayerVPUNNCost::getSpillingWriteCostsForAllTiles(
        mlir::Operation* operation, const VPUNNCostParameters& parameters,
        std::function<vpux::NDTypeInterface(const TileInfo&)> getOutputType) const {
    if (VPU::isPureViewOp(operation)) {
        return {0};
    }

    SmallVector<StrategyCost> spillWriteCosts;
    if (getOutputType == nullptr) {
        getOutputType = [&](const auto& tileInfo) {
            auto outputType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType());
            auto tiledType = outputType.extractDenseTile(tileInfo.offsets, tileInfo.shape);
            if (auto parentClusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
                auto numClusters = parentClusterOp.getOptimalNumClusters(outputType.getShape(), parameters._strategy);

                tiledType = VPU::getDistributedOutputTypeFromOp(parentClusterOp, tiledType, numClusters,
                                                                parameters._strategy);
            }
            return tiledType;
        };
    }

    auto outputType = mlir::cast<NDTypeInterface>(operation->getResult(0).getType());

    const auto tiling =
            parameters._tiling.empty() ? OutputTiling({TileInfo(outputType.getShape())}) : parameters._tiling;
    spillWriteCosts.reserve(tiling.size());
    for (const auto& tileInfo : tiling) {
        auto tiledType = getOutputType(tileInfo);
        spillWriteCosts.emplace_back(getSpillingTypeCost(tiledType, tiling[0].axis));
    }
    return spillWriteCosts;
}

StrategyCost LayerVPUNNCost::getSpillingWriteCost(
        mlir::Operation* operation, const VPUNNCostParameters& parameters,
        std::function<vpux::NDTypeInterface(const TileInfo&)> getOutputType /*nullptr*/) const {
    auto spillingWriteCosts = getSpillingWriteCostsForAllTiles(operation, parameters, std::move(getOutputType));
    auto writeCost =
            std::accumulate(spillingWriteCosts.begin(), spillingWriteCosts.end(), 0, std::plus<StrategyCost>());
    return writeCost;
}

SmallVector<StrategyCost> LayerVPUNNCost::getSpillingReadCostsForAllTiles(
        mlir::Operation* operation, const VPUNNCostParameters& parameters, mlir::Operation* parentOp,
        std::function<bool(mlir::Value value)> findOperand,
        std::function<vpux::NDTypeInterface(const TileInfo&)> getOperandType) const {
    VPUX_THROW_WHEN(parentOp == nullptr && findOperand == nullptr,
                    "Either parent operation or functor for operands must be passed");

    if (VPU::isPureViewOp(operation)) {
        return {0};
    }

    SmallVector<StrategyCost> spillReadCosts;
    if (findOperand == nullptr) {
        findOperand = [&](auto value) {
            auto operation = value.getDefiningOp();
            while (operation != nullptr) {
                if (operation == parentOp) {
                    return true;
                } else if (VPU::isPureViewOp(operation)) {
                    operation = operation->getOperand(0).getDefiningOp();
                    continue;
                }
                return false;
            }
            return false;
        };
    }

    const auto operandItr = llvm::find_if(operation->getOperands(), std::move(findOperand));
    VPUX_THROW_WHEN(operandItr == operation->getOperands().end(),
                    "Operation {0} has no common tensors with operation {1}", *parentOp, *operation);
    const size_t operandInd = std::distance(operation->getOperands().begin(), operandItr);
    const auto childTiling = parameters._tiling.empty() ? OutputTiling({TileInfo(getShape(operation->getResult(0)))})
                                                        : parameters._tiling;

    MultiClusterStrategySetter mcSetter(operation, parameters._strategy);

    if (getOperandType == nullptr) {
        getOperandType = [&](const auto& tileInfo) {
            auto tiling = getTileTypes(operation, tileInfo);
            return tiling[operandInd];
        };
    }

    spillReadCosts.reserve(childTiling.size());
    for (const auto& tileInfo : childTiling) {
        const auto childOperandsTiling = getOperandType(tileInfo);
        spillReadCosts.emplace_back(getSpillingTypeCost(childOperandsTiling, tileInfo.axis));
    }
    return spillReadCosts;
}

StrategyCost LayerVPUNNCost::getSpillingReadCost(
        mlir::Operation* operation, const VPUNNCostParameters& parameters, mlir::Value operand,
        std::function<vpux::NDTypeInterface(const TileInfo&)> getOperandType /*nullptr*/) const {
    return getSpillingReadCost(
            operation, parameters, nullptr,
            [&](mlir::Value item) {
                return item == operand;
            },
            std::move(getOperandType));
}

StrategyCost LayerVPUNNCost::getSpillingReadCost(
        mlir::Operation* operation, const VPUNNCostParameters& parameters, mlir::Operation* parentOp /*nullptr*/,
        std::function<bool(mlir::Value value)> findOperand /*nullptr*/,
        std::function<vpux::NDTypeInterface(const TileInfo&)> getOperandType /*nullptr*/) const {
    auto spillReadCost = getSpillingReadCostsForAllTiles(operation, parameters, parentOp, std::move(findOperand),
                                                         std::move(getOperandType));
    auto readCost = std::accumulate(spillReadCost.begin(), spillReadCost.end(), 0, std::plus<StrategyCost>());
    return readCost;
}

StrategyCost LayerVPUNNCost::getSpillingCost(mlir::Operation* parentOp, const VPUNNCostParameters& parentParameters,
                                             mlir::Operation* childOp,
                                             const VPUNNCostParameters& childParameters) const {
    /*
     Spilling cost is computed as sum of cyclecost of dma of parent operation from CMX->DDR (write cost)
     and cyclecost of dma of child operation from DDR->CMX (read cost)
     In case one of operations is pure view like, it's supposed to be in DDR already, no write/read cost
     is needed from/to it
    */

    return getSpillingWriteCost(parentOp, parentParameters) + getSpillingReadCost(childOp, childParameters, parentOp);
}

size_t LayerVPUNNCost::getNumClusterCorrectionSize(VPU::MultiClusterStrategy strategy) const {
    return strategy != MultiClusterStrategy::Clustering ? _numTiles : 1;
}

StrategyCost LayerVPUNNCost::getSimpleLayerCost(vpux::NDTypeInterface outputType,
                                                const VPUNNCostParameters& parameters) const {
    return outputType.getTotalAllocSize().count() / getNumClusterCorrectionSize(parameters._strategy);
}

StrategyCost LayerVPUNNCost::getNCELayerCost(VPU::NCEOpInterface nceOp, const VPUNNCostParameters& parameters) const {
    // Types for each tile
    std::vector<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes;

    auto isPrefetchTilingEnabled = (parameters._mode != TilingMode::ISOLATED);

    _log.trace("Start calculating vpunn cost for Op {0} with strategy {1}", nceOp.getLoc(), parameters._strategy);

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
    MultiClusterStrategySetter mcSetter(nceOp, parameters._strategy);
    // Set prefetching to be true to ignore the DMA cost and only get the execution DPU cost
    // According to the VPUNN API definition,
    //      when prefetching is false, the returned cost is the sum of DPU + weights DMA
    //      when prefetching is true, the returned cost is just DPU because it considers the weights are prefetched
    auto distributionMode = DistributionMode::NONE;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp != nullptr) {
        auto outputType = clusteredOp->getResult(0).getType().cast<NDTypeInterface>();
        distributionMode = getOutputTensorDistributionMode(clusteredOp, costParams.layerStrategy, outputType);
    }
    const auto vpunnStrategy = VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numTiles, _arch, _numShaveActs,
                                                        true, distributionMode);
    auto vpunnLayerDPUCosts = getDPUCostForNCEOp(nceOp, parameters._strategy, parameters._tiling, costParams,
                                                 vpunnStrategy, _vpunnCostModel, _log);
    _log.trace("VPUNN DPU layer costs {0}", vpunnLayerDPUCosts);

    if (vpunnLayerDPUCosts.empty()) {
        _log.trace("DPU cost is empty, return COST_MAX");
        return std::numeric_limits<VPU::StrategyCost>::max();
    }

    // Accumulate DPU costs
    auto vpunnCost = std::accumulate(vpunnLayerDPUCosts.begin(), vpunnLayerDPUCosts.end(), 0);

    if (!parameters._withDMAs) {
        return vpunnCost;
    }

    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
    auto siblingsAnalysis = SiblingOpsAnalysis(nceOp.getOperation());
    for (auto& outTile : parameters._tiling) {
        auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, _log);
        tilesTypes.push_back(getTileDistributions(nceOp.getOperation(), siblingsAnalysis, outTile, inTiles));
    }

    // Add extra weights DMA costs
    const auto getSpillingReadCost = [&](NDTypeInterface srcType,
                                         const TensorDistributionMap& distributions) -> uint32_t {
        auto distributedType = getDistributedTypeFromDistributionMap(srcType, distributions);
        return checked_cast<uint32_t>(getDMACost(distributedType, _vpuDevice, _vpunnCostModel, _numDMAPorts));
    };
    auto vpunnLayerWeightsCosts = getPerTileWeightsDMACosts(nceOp, siblingsAnalysis, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    vpunnCost += getWeightsDMACostForNCEOp(nceOp, parameters._tiling, vpunnLayerDPUCosts, vpunnLayerWeightsCosts,
                                           isPrefetchTilingEnabled, _log);

    _log.trace("VPUNN total layer cost {0}", vpunnCost);
    return vpunnCost;
}

StrategyCost LayerVPUNNCost::getSWLayerCost(VPU::SWOpInterface swOp, const VPUNNCostParameters& parameters) const {
    auto outputType = swOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto outputTiling = parameters._tiling;
    if (outputTiling.empty()) {
        outputTiling.push_back(TileInfo(outputType.getShape()));
    }

    StrategyCost fullCost = 0;
    for (auto index : irange(outputTiling.size())) {
        SmallVector<vpux::NDTypeInterface> inputNDTypes;
        SmallVector<TileInfo> inputTiles;
        if (parameters._operandsTiling.empty()) {
            if (auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation())) {
                inputTiles = tilingBuilderOp.backInferTileInfo(outputTiling[index], _log).tiles;
            } else {
                continue;
            }
        } else {
            inputTiles = parameters._operandsTiling[index];
        }

        for (auto typeIndex : irange(inputTiles.size())) {
            inputNDTypes.push_back(swOp->getOperand(typeIndex).getType().cast<vpux::NDTypeInterface>().extractDenseTile(
                    inputTiles[typeIndex].offsets, inputTiles[typeIndex].shape));
        }

        auto outputTiledType = outputType.extractDenseTile(outputTiling[index].offsets, outputTiling[index].shape);
        const auto vpunnLayer = getVPUNNSWKernelOp(swOp, outputTiledType, inputNDTypes);

        if (!vpunnLayer) {
            fullCost += getSimpleLayerCost(outputTiledType, parameters);
        } else {
            auto distributionMode = DistributionMode::NONE;
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation());
            if (clusteredOp != nullptr) {
                distributionMode = getOutputTensorDistributionMode(clusteredOp, parameters._strategy, outputTiledType);
            }

            auto vpunnStrategy = VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numTiles, _arch,
                                                          _numShaveActs, false, distributionMode);
            fullCost += _vpunnCostModel->Layer(*vpunnLayer, vpunnStrategy);
        }
    }

    return fullCost;
}

// Correct the DMA cost for a given type by considering the stride of the tensor.
// It calculates the number of continuous bytes on the lowest dimension and compares it with a predefined
// threshold. If the continuous bytes are less than the threshold, the cost is adjusted accordingly.
StrategyCost LayerVPUNNCost::correctStrideDMACost(vpux::NDTypeInterface type, StrategyCost cost) const {
    const auto dimOrder = type.getDimsOrder();
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    const Bit elemSize = type.getElemTypeSize();
    if (auto sparseTensorType = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
        type = sparseTensorType.getData().cast<NDTypeInterface>();
    }
    Bit continuousBytesOnLowestDim;
    if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(type)) {
        continuousBytesOnLowestDim = distributedType.getLargestCompactShape()[lowestDim] * elemSize;
    } else {
        continuousBytesOnLowestDim = type.getShape()[lowestDim] * elemSize;
    }
    if (continuousBytesOnLowestDim.count() < strideDMACorrectionThresholdInBits) {
        auto factor = checked_cast<double>(strideDMACorrectionThresholdInBits) / continuousBytesOnLowestDim.count();
        return checked_cast<uint32_t>(std::floor(factor * cost));
    }
    return cost;
}
