//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_driver.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <vpu/shave/layers.h>
#include <vpu_layer_cost_model.h>

using namespace vpux;
using namespace VPU;

namespace {

mlir::Value getInputFromClusteredOp(VPU::ClusteredOpInterface clusteredOp, mlir::Operation* parentOp) {
    for (auto operand : clusteredOp->getOperands()) {
        auto parent = operand.getDefiningOp();
        if (parent == parentOp) {
            return operand;
        }
        while (mlir::isa_and_present<VPU::DistributedCastOpInterface, VPU::GroupSparseTensorOp>(parent)) {
            // propagate cast ops
            parent = parent->getOperand(0).getDefiningOp();
            if (parent == parentOp) {
                return operand;
            }
        }
    }

    VPUX_THROW("Cannot find input from op: {0}, parent op: {1}", clusteredOp, parentOp);
}

bool isSOHAlignmentCompatibleOrAdjustedCompatible(std::pair<mlir::Type, VPU::DistributionInfo>& srcTypeDistribution,
                                                  std::pair<mlir::Type, VPU::DistributionInfo>& dstTypeDistribution) {
    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcTypeDistribution.first);
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstTypeDistribution.first);

    if (srcType.getShape() != dstType.getShape()) {
        return false;
    }
    if (srcType.getDimsOrder() != dstType.getDimsOrder() || srcType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }

    auto srcDistribution = srcTypeDistribution.second;
    auto dstDistribution = dstTypeDistribution.second;
    if ((srcDistribution.getDistributionMode() != DistributionMode::SEGMENTED) ||
        (dstDistribution.getDistributionMode() != DistributionMode::SEGMENTED) ||
        (srcDistribution.getNumTiles() != dstDistribution.getNumTiles())) {
        return false;
    }

    return true;
}

bool isDistributedCastCompatible(std::pair<mlir::Type, VPU::DistributionInfo>& srcTypeDistribution,
                                 std::pair<mlir::Type, VPU::DistributionInfo>& dstTypeDistribution) {
    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcTypeDistribution.first);
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstTypeDistribution.first);
    auto srcDistribution = srcTypeDistribution.second;
    auto dstDistribution = dstTypeDistribution.second;
    if (getBoundedShape(srcType) != getBoundedShape(dstType)) {
        return false;
    }

    if (areDistributionElementTypesCompatible(srcType.getElementType(), dstType.getElementType()).failed()) {
        return false;
    }

    if (srcType.getMemSpace() != dstType.getMemSpace()) {
        return false;
    }

    if (srcType.getDimsOrder() != dstType.getDimsOrder()) {
        return false;
    }

    if (areDistributionsCompatible(srcType, srcDistribution, dstType, dstDistribution).failed()) {
        return false;
    }

    return true;
}

bool isTargetTensorTypeCompatible(std::pair<mlir::Type, VPU::DistributionInfo>& srcType,
                                  std::pair<mlir::Type, VPU::DistributionInfo>& dstType) {
    const auto& srcTypeDistribution = srcType.second;
    const auto& dstTypeDistribution = dstType.second;
    const auto srcTypeIsDistributed = srcTypeDistribution.getDistributionMode() != DistributionMode::NONE;
    const auto dstTypeIsDistributed = dstTypeDistribution.getDistributionMode() != DistributionMode::NONE;
    if (srcTypeIsDistributed ^ dstTypeIsDistributed) {
        return false;
    }
    if (srcTypeIsDistributed && dstTypeIsDistributed) {
        if (!isDistributedCastCompatible(srcType, dstType)) {
            return false;
        }
    }
    return true;
}

uint32_t getPrefetchDMACostOverlappsWithPreviousDPU(SmallVector<uint32_t>& layerDPUCosts,
                                                    ArrayRef<uint32_t> layerDMACosts, bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx + 1] > layerDPUCosts[tileIdx]) {
                // Prefetched and all the DPU cycles are overlapped with DMA
                totalDMACost += (layerDMACosts[tileIdx + 1] - layerDPUCosts[tileIdx]);
                layerDPUCosts[tileIdx] = 0;
            } else {
                // Prefetched and some DPU cycles are still not overlapped
                layerDPUCosts[tileIdx] -= layerDMACosts[tileIdx + 1];
            }
        }
    }
    totalDMACost += layerDMACosts[0];
    return totalDMACost;
}

uint32_t getOutputDMACostOverlappsWithNextDPU(SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                              bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx] > layerDPUCosts[tileIdx + 1]) {
                // Prefetched and all the DPU cycles are overlapped with DMA
                totalDMACost += (layerDMACosts[tileIdx] - layerDPUCosts[tileIdx + 1]);
                layerDPUCosts[tileIdx + 1] = 0;
            } else {
                // Prefetched and some DPU cycles are still not overlapped
                layerDPUCosts[tileIdx + 1] -= layerDMACosts[tileIdx];
            }
        }
    }
    totalDMACost += layerDMACosts.back();
    return totalDMACost;
}

bool isTiledOnLowestDim(const TileInfo& outTile, const DimsOrder& dimOrder) {
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    for (auto item : outTile.axis | indexed) {
        const auto index = item.index();
        const auto axis = item.value();
        if (axis > 1 && Dim(index) == lowestDim) {
            return true;
        }
    }
    return false;
}
}  // namespace

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, Logger log,
                               SiblingOpsAnalysis& siblingsOpsAnalysis)
        : _func(func),
          _enablePrefetchTiling(enablePrefetchTiling),
          _log(log),
          _siblingsOpsAnalysis(siblingsOpsAnalysis) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto tileOp = config::getTileExecutor(module)) {
        auto dpuExec = tileOp.getSubExecutor(config::ExecutorKind::DPU);
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _DMABandwidth = getDMABandwidth(config::getPlatform(tileOp), config::getRevisionID(module));
        if (auto shaveActExec = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    }
    _numDMAPorts = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN).getCount();
    _arch = config::getArch(module);
    _vpuDeviceType = getVPUDeviceType(module);
    _layerCostModel = VPU::CostModelConfig::createLayerCostModel(module);
}

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling,
                               SiblingOpsAnalysis& siblingsOpsAnalysis,
                               std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModelPtr, Logger log)
        : _layerCostModel(std::move(layerCostModelPtr)),
          _func(func),
          _enablePrefetchTiling(enablePrefetchTiling),
          _log(log),
          _siblingsOpsAnalysis(siblingsOpsAnalysis) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto tileOp = config::getTileExecutor(module)) {
        auto dpuExec = tileOp.getSubExecutor(config::ExecutorKind::DPU);
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _DMABandwidth = getDMABandwidth(config::getPlatform(tileOp), config::getRevisionID(module));
        if (auto shaveActExec = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    }
    _numDMAPorts = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN).getCount();
    _arch = config::getArch(module);
    _vpuDeviceType = getVPUDeviceType(module);
}

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, bool enableTilingFullSearchSpace,
                               SiblingOpsAnalysis& siblingsOpsAnalysis,
                               std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModelPtr, Logger log)
        : LayerCostModel(func, enablePrefetchTiling, siblingsOpsAnalysis, std::move(layerCostModelPtr), log) {
    _enableTilingFullSearchSpace = enableTilingFullSearchSpace;
}

void LayerCostModel::resetNNCacheCounter() {
    _layerCostModel->getDPUPreloadedCacheCounter().reset();
}

void LayerCostModel::printNNCacheStatistics() const {
    _log.info("[NN Cache statistics]  {0}", _layerCostModel->getDPUPreloadedCacheCounter().printString());
}

std::pair<mlir::Type, TensorDistributionMap> LayerCostModel::getInputWithDistribution(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
        VPU::MultiClusterStrategy specifiedStrategy) const {
    const auto& input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr =
            VPU::getOptimalNumClusters(origOp, getBoundedShape(origOp->getResult(0).getType()), specifiedStrategy);
    if (auto nceOp = mlir::dyn_cast_if_present<VPU::NCEOpInterface>(origOp.getOperation())) {
        auto isFilter = isWeightsLikeOperand(nceOp, input);
        if (isFilter) {
            return std::make_pair(input.getType(), getFilterDistributionAttrFromOp(nceOp, input.getType(),
                                                                                   numClustersAttr, specifiedStrategy));
        }
    }
    return std::make_pair(input.getType(),
                          getActivationDistributionAttrFromOp(origOp, input, input.getType(), numClustersAttr,
                                                              specifiedStrategy, _siblingsOpsAnalysis));
}

std::pair<mlir::Type, TensorDistributionMap> LayerCostModel::getInputWithDistribution(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp, VPU::MultiClusterStrategy specifiedStrategy,
        mlir::ArrayRef<int64_t> customAlignment) const {
    const auto& input = getInputFromClusteredOp(origOp, parentOp);
    const auto resultShape = getBoundedShape(origOp->getResult(0).getType());
    auto numClustersAttr = VPU::getOptimalNumClusters(origOp, resultShape, specifiedStrategy);
    return std::make_pair(input.getType(), getActivationDistributionAttrFromOp(origOp, input, input.getType(),
                                                                               numClustersAttr, specifiedStrategy,
                                                                               _siblingsOpsAnalysis, customAlignment));
}

std::pair<mlir::Type, TensorDistributionMap> LayerCostModel::getOutputWithDistribution(
        VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const {
    auto numClustersAttr =
            VPU::getOptimalNumClusters(origOp, getBoundedShape(origOp->getResult(0).getType()), specifiedStrategy);
    return std::make_pair(origOp->getResult(0).getType(),
                          getOutputDistributionAttrFromOp(origOp, origOp->getResult(0).getType(), numClustersAttr,
                                                          specifiedStrategy, _siblingsOpsAnalysis));
}

/*
 * Get the spilling cost
 * srcTensorType is the output of parent op (current op)
 * dstTensorType is the input of child op
 * return spilling write cost and spilling read cost
 */
LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             vpux::NDTypeInterface dstTensorType,
                                                             VPU::ClusteredOpInterface parentOp,
                                                             VPU::ClusteredOpInterface userOp) const {
    return getSpillingCost(srcTensorType, getDistributionMapFromDistributedType(srcTensorType), dstTensorType,
                           getDistributionMapFromDistributedType(dstTensorType), parentOp, userOp);
}

LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             const TensorDistributionMap& srcDistribution,
                                                             vpux::NDTypeInterface dstTensorType,
                                                             const TensorDistributionMap& dstDistribution,
                                                             VPU::ClusteredOpInterface parentOp,
                                                             VPU::ClusteredOpInterface userOp) const {
    // Concat is on DDR memory if there's spilling. So we don't need copy from CMX to DDR if Concat is parent. Also we
    // don't need copy from DDR to CMX if Concat is user.
    if (mlir::isa<VPU::ConcatOp>(parentOp)) {
        return {0.0, getSpillingDMACost(dstTensorType, dstDistribution)};
    }

    if (mlir::isa<VPU::ConcatOp>(userOp)) {
        return {getSpillingDMACost(srcTensorType, srcDistribution), 0.0};
    }

    return {getSpillingDMACost(srcTensorType, srcDistribution), getSpillingDMACost(dstTensorType, dstDistribution)};
}

double LayerCostModel::getDMACostOfType(vpux::NDTypeInterface srcType) const {
    auto distributedSrcType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(srcType);
    auto distribution = distributedSrcType != nullptr
                                ? DistributionInfo::getClassFromAttr(distributedSrcType.getDistribution())
                                : DistributionInfo();
    return getDMACostOfType(srcType, distribution);
}

double LayerCostModel::getDMACostOfType(vpux::NDTypeInterface srcType,
                                        const VPU::DistributionInfo& distribution) const {
    if (isDMACostModelAccurate(_arch)) {
        TensorDistributionMap distributionMap;
        distributionMap.insert(std::make_pair(srcType, distribution));
        auto distributedType = getDistributedTypeFromDistributionMap(srcType, distributionMap);
        return static_cast<double>(getDMACost(distributedType, _arch, _vpuDeviceType,
                                              _layerCostModel->get_TheoreticalDMA_cost_model_shared(), _numDMAPorts));
    }
    // When cost model is not accurate, use a manual analytical calculation to estimate the cost
    return getAnalyticalDMACost(srcType, distribution, _DDRLatency, _DMABandwidth, _numDMAPorts);
}

double LayerCostModel::getSpillingDMACost(vpux::NDTypeInterface srcTensorType) const {
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(srcTensorType)) {
        srcTensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    return getDMACostOfType(srcTensorType);
}

double LayerCostModel::getSpillingDMACost(vpux::NDTypeInterface srcTensorType,
                                          const TensorDistributionMap& distributions) const {
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(srcTensorType)) {
        srcTensorType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    VPU::DistributionInfo distribution{};
    if (distributions.contains(srcTensorType)) {
        distribution = distributions.at(srcTensorType);
    }
    return getDMACostOfType(srcTensorType, distribution);
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double LayerCostModel::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR: {
        mpeHeight = 1;
        mpeWidth = 16;
        break;
    }
    case VPU::MPEMode::VECTOR_FP16: {
        mpeHeight = 1;
        mpeWidth = 4;
        break;
    }
    case VPU::MPEMode::MATRIX:
    // These different mpe modes on VPUX37XX have impact on the reuse of activation and weights. We can't estimate reuse
    // cost with current cost equation. In the future we will integrate VPUNN to estimate the cost.
    case VPU::MPEMode::CUBOID_4x16:
    case VPU::MPEMode::CUBOID_8x16:
    case VPU::MPEMode::CUBOID_16x16: {
        mpeHeight = 4;
        mpeWidth = 4;
        break;
    }
    default:
        VPUX_THROW("Unsupported mpeMode '{0}'", mpeMode);
    }

    return static_cast<double>(
            _numDPUs * divUp((mpeHeight * divUp(shape[Dims4D::Act::H], mpeHeight) * mpeWidth *
                              divUp(shape[Dims4D::Act::W], mpeWidth) * VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT *
                              divUp(shape[Dims4D::Act::C], VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT)),
                             _numDPUs));
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
double LayerCostModel::computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    auto numClusters = getOptimalNumClusters(
            clusteredOp, mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getShape(), strategy);
    const auto distributedOutputTensorType = VPU::getDistributedOutputTypeFromOp(
            clusteredOp, mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()), numClusters, strategy);

    VPUX_THROW_UNLESS(distributedOutputTensorType.containsDistributedTypes(), "Missing output distributed types");
    const auto distributedOutputDataType =
            mlir::cast<vpux::VPU::DistributedTensorType>(distributedOutputTensorType.getDistributedTypes().front());
    const auto perClusterShape = distributedOutputDataType.getLargestCompactShape();
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return std::max({static_cast<double>(perClusterOutputTensorVolume) /
                             calculateMPEVolume(VPU::MPEMode::CUBOID_4x16, perClusterShape),
                     static_cast<double>(perClusterOutputTensorVolume) /
                             calculateMPEVolume(VPU::MPEMode::CUBOID_8x16, perClusterShape),
                     static_cast<double>(perClusterOutputTensorVolume) /
                             calculateMPEVolume(VPU::MPEMode::CUBOID_16x16, perClusterShape)});
}

/// @brief A switcher to select time-cost or efficiency-cost for greedy strategy assignment
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                       bool useTimeBasedCost) {
    if (!useTimeBasedCost) {
        return getEfficiencyCost(nceOp, strategy);
    }

    double basicDPUandDMACost = COST_MAX;

    const auto it = _costCache.find(nceOp);
    if (it == _costCache.end()) {
        // Case 1 - Op costs are not found in cache:
        // 1.Calculate cost value with VPUNN
        // 2.Create new op costs
        // 3.Store the new op costs in cost cache
        basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
        SmallVector newOpCosts((getMaxEnumValForMultiClusterStrategy() + 1), COST_MAX);
        newOpCosts[static_cast<uint64_t>(strategy)] = basicDPUandDMACost;
        _costCache.insert({nceOp, newOpCosts});
    } else {
        auto strategyCostIt = it->second.begin() + static_cast<uint64_t>(strategy);
        if (strategyCostIt != nullptr && *strategyCostIt != COST_MAX) {
            // Case 2 - Strategy cost is found in cache:
            // Retrieve the cost value directly
            basicDPUandDMACost = *strategyCostIt;
        } else {
            // Case 3 - Op costs are found but op strategy cost is not found:
            // 1.Calculate cost value with VPUNN
            // 2.Update op strategy cost value in cache
            basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
            *strategyCostIt = basicDPUandDMACost;
        }
    }

    return basicDPUandDMACost;
}

/// @brief Time-cost : return the shave computation time (cycles)
/// @details use vpunn cost model to get the shave cost of sw layer
double LayerCostModel::getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy) {
    const auto allTypesSupported = [](mlir::ValueRange values) -> bool {
        return llvm::all_of(values, [](mlir::Value value) {
            const auto valueType = mlir::cast<vpux::NDTypeInterface>(value.getType());
            const auto nnType = VPU::getVPUNNElementType(valueType.getElementType());
            return nnType.has_value();
        });
    };

    if (!allTypesSupported(swOp->getOperands()) || !allTypesSupported(swOp->getResults())) {
        _log.warning("'{0}': Unable to compute SW layer cost due to unsupported data type", swOp->getLoc());
        return 0.0;
    }

    if (_enableTilingFullSearchSpace) {
        if (auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation())) {
            if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation())) {
                const auto origStrategy = clusteredOp.getMultiClusterStrategy();
                clusteredOp.setMultiClusterStrategy(strategy);
                VPUX_SCOPE_EXIT {
                    if (origStrategy.has_value()) {
                        clusteredOp.setMultiClusterStrategy(*origStrategy);
                    } else {
                        clusteredOp->removeAttr(VPU::multiClusterStrategy);
                    }
                };

                // Scope online profiling to this full-search query.
                const VPU::ScopedProfilingForAutoHint profilingScope(_layerCostModel->get_cost_model_shared());

                if (auto bestTilingInfo = VPU::TemporalTilingDriver::getBestTilingStrategy(tilingOp, *this, _log)) {
                    return bestTilingInfo->costInfo.overallCost;
                }
            }
        }
    }

    uint32_t fullCost = 0;
    OutputTiling outTiles({TileInfo(getShape(swOp->getResult(0)))});

    const auto& shaveUtilIntf = VPU::getShaveCostModelUtils(swOp->getContext());
    auto isShave2APIUsed = shaveUtilIntf.isShave2ApiUsed();

    if (!isShave2APIUsed) {
        auto vpunnLayer = getVPUNNSWKernelOp(swOp);

        auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, _arch, _numShaveActs, false);
        return _layerCostModel->Layer(*vpunnLayer, vpunnStrategy);
    }

    const auto swCostParam = getShaveWorkloadCostParam(swOp, _arch, _numShaveActs, _numTiles);
    SmallVector<uint32_t> vpunnLayerSWCosts;
    vpunnLayerSWCosts = getSHAVECostForSwOpPreSplit(swOp, outTiles, swCostParam, _layerCostModel, _numShaveActs, _log);
    for (const auto& cost : vpunnLayerSWCosts) {
        fullCost += cost;
    }

    return fullCost;
}

/// @brief get computation cost
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getLayerCost(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                    bool useTimeBasedCost) {
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
        return getNCELayerCost(nceOp, strategy, useTimeBasedCost);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        return getSWLayerCost(swOp, strategy);
    } else if (mlir::isa<VPU::ConcatOp>(clusteredOp.getOperation())) {
        // Concat has no computation cost
        return 0.0;
    } else if (mlir::isa<VPU::GatherDMAOp>(clusteredOp.getOperation())) {
        // GatherDMAOp has no computation cost
        return 0.0;
    } else {
        VPUX_THROW("Unsupported op type {0} at {1}", clusteredOp->getName(), clusteredOp->getLoc());
    }
}

// For sw op, if strategy is set for it's parent or user nce op, check if sw op can avoid spilling with the op
bool isSoftwareOpCustomStrategyIncompatibleWithOtherNceOp(mlir::Operation* swOp, VPU::MultiClusterStrategy strategy,
                                                          int64_t numTiles) {
    if (!mlir::isa_and_nonnull<VPU::SWOpInterface>(swOp)) {
        return false;
    }

    // Rank need to be 4 when checking strategy compatibility
    for (const auto& input : swOp->getOperands()) {
        const auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
        if (inputShape.size() != RANK_REQUIRED_FOR_TILING) {
            return false;
        }
    }
    for (const auto& output : swOp->getResults()) {
        const auto outputShape = mlir::cast<vpux::NDTypeInterface>(output.getType()).getShape();
        if (outputShape.size() != RANK_REQUIRED_FOR_TILING) {
            return false;
        }
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        // For Clustering, Sw op can always set Clustering to avoid spilling
        return false;
    }

    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp)) {
        // If SOH liked strategy can not be assigned for SW op, there will be spilling between them
        // If SOK strategy can not be assigned for SW op, the only way to avoid spilling is setting Clustering. But
        // Clustering has bad compute efficiency, so we can assume there's spilling.
        if (strategy == VPU::MultiClusterStrategy::HKSwitch) {
            strategy = VPU::MultiClusterStrategy::SplitOverHeight;
        }
        return !clusteredOp.checkStrategyCompatibility(strategy, numTiles);
    }

    return false;
}

// This function, introduced for Tiling rework, has a lot of similarities to getDPUandDMATimeCostWithCustomTiling,
// but input and output types are different and internal assumptions and several details diverged.
// It would be best to unify the functionality, which is going to be the subject of E#215636
ComputeAndDMATimeCost LayerCostModel::getComputeAndDataCosts(mlir::Operation* op,
                                                             std::optional<VPU::MultiClusterStrategy> strategy,
                                                             const OutputTiling& outTiles) const {
    VPUX_THROW_WHEN(outTiles.empty(), "Empty output tiles");

    _log.trace("[Cost Analysis] {0} [{1}]  Get VPUNN cost for {2} {3}", op->getLoc(), op->getName(), strategy,
               outTiles[0].axis);

    // Step 1: Compute costs — get per-tile DPU or SHAVE execution cost from VPUNN
    SmallVector<uint32_t> vpunnLayerComputeCosts;
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
        auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
        if (strategy.has_value()) {
            mcStrategy = strategy.value();
        }

        const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs, _numTiles, mcStrategy);

        auto distributionMode = DistributionMode::NONE;
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        if (clusteredOp != nullptr) {
            auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
            auto tiledOutputType = outputType.extractDenseTile(outTiles[0].offsets, outTiles[0].shape);
            distributionMode = getOutputTensorDistributionMode(clusteredOp, mcStrategy, tiledOutputType);
        }

        const auto vpunnStrategy = VPU::getVPULayerStrategy(mcStrategy, _numDPUs, _numTiles, _arch, /*nSHVs=*/1,
                                                            /*prefetching=*/true, distributionMode, nceOp);

        if (mlir::isa<VPU::NCEMatMulOp>(nceOp)) {
            // TODO E#126102: cost not supported for MatMul
            vpunnLayerComputeCosts = SmallVector<uint32_t>(outTiles.size(), 1);
        } else if (config::hasVPUNNPreSplit(nceOp)) {
            vpunnLayerComputeCosts = getDPUCostForNCEOpPreSplit(
                    nceOp, outTiles, costParams, vpunnStrategy.tiling_strategy, _layerCostModel, _numDPUs, _log);
        } else {
            vpunnLayerComputeCosts =
                    getDPUCostForNCEOp(nceOp, mcStrategy, outTiles, costParams, vpunnStrategy, _layerCostModel, _log);
        }
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op)) {
        const auto swCostParam = getShaveWorkloadCostParam(swOp, _arch, _numShaveActs, _numTiles);
        vpunnLayerComputeCosts = getSHAVECostForSwOpPreSplit(swOp, strategy, outTiles, swCostParam, _layerCostModel,
                                                             _numShaveActs, _log);
    } else {
        // TODO E#216536: support compute DMA case
        Logger::global().error("Unsupported op type {0} at {1}", op->getName(), op->getLoc());
        vpunnLayerComputeCosts = SmallVector<uint32_t>(outTiles.size(), 1);
    }

    _log.trace("VPUNN compute layer costs {0}", vpunnLayerComputeCosts);

    // Step 2: Tile type distributions — for each output tile, back-infer input tiles and resolve
    // the multi-cluster distribution for each operand/result. Result is a 2D structure:
    //   tilesTypes[tileIdx][operandIdx] = {tiled NDType, cluster distribution map}
    // where operandIdx ordering is [activation, weights..., output]
    std::vector<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes;
    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "Expected operation to implement TilingBuilderOpInterface");

    for (auto& outTile : outTiles) {
        auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, _log);
        // TODO E#216912: Add support for multi-output ops in getTileDistributions
        // TODO E#217919: Support missing tile distributions (and cost) for non act/wgt operands in
        // getTileDistributionsWithFilter
        tilesTypes.push_back(getTileDistributions(op, _siblingsOpsAnalysis, outTile, strategy, inTiles));
    }

    // Step 3: DMA costs -> iterate tilesTypes and classify each operand position:
    //   operandIdx == 0        : activation input (DMA read)
    //   0 < operandIdx < last  : weights/secondary inputs (DMA read)
    //   operandIdx == last     : output (DMA write)
    SmallVector<uint32_t> vpunnLayerActCosts;
    SmallVector<uint32_t> vpunnLayerWeightsCosts;
    SmallVector<uint32_t> vpunnLayerOutputCosts;

    bool correctedStrideActDMACost = false;
    bool correctedStrideOutputDMACost = false;
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
    const auto outOrder = DimsOrder::fromValue(op->getResult(0));
    const auto inputTiledOnLowestDim = isActivationTiledOnLowestDim(tilingBuilderOp, outTiles[0], inOrder);
    const auto outputTiledOnLowestDim = isTiledOnLowestDim(outTiles[0], outOrder);
    const auto arch = config::getArch(op);

    // TODO: E#216912 Add support for multi-output ops
    for (const auto& tileTypes : tilesTypes) {
        for (auto operandIdx : irange(tileTypes.size())) {
            const auto tileType =
                    getDistributedTypeFromDistributionMap(tileTypes[operandIdx].first, tileTypes[operandIdx].second);
            if (operandIdx == 0) {
                auto dmaCost = checked_cast<uint32_t>(
                        getSpillingDMACost(tileTypes[operandIdx].first, tileTypes[operandIdx].second));
                correctedStrideActDMACost |= applyStrideDMACorrectionForTile(tileType, inputTiledOnLowestDim, dmaCost,
                                                                             arch, /*isFullSearchVersion=*/true);
                vpunnLayerActCosts.push_back(dmaCost);
            } else if (operandIdx == tileTypes.size() - 1) {
                auto dmaCost = checked_cast<uint32_t>(
                        getSpillingDMACost(tileTypes[operandIdx].first, tileTypes[operandIdx].second));
                correctedStrideOutputDMACost |= applyStrideDMACorrectionForTile(
                        tileType, outputTiledOnLowestDim, dmaCost, arch, /*isFullSearchVersion=*/true);
                vpunnLayerOutputCosts.push_back(dmaCost);
            } else {
                auto dmaCost = checked_cast<uint32_t>(
                        getSpillingDMACost(tileTypes[operandIdx].first, tileTypes[operandIdx].second));
                vpunnLayerWeightsCosts.push_back(dmaCost);
            }
        }
    }

    if (correctedStrideActDMACost || correctedStrideOutputDMACost) {
        _log.nest(4).trace(
                "[Cost Analysis] Note: DMA costs have been corrected for stride DMA. Activation {0}, Output {1}",
                correctedStrideActDMACost, correctedStrideOutputDMACost);
    }

    return {std::move(vpunnLayerActCosts), std::move(vpunnLayerWeightsCosts), std::move(vpunnLayerComputeCosts),
            std::move(vpunnLayerOutputCosts)};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> LayerCostModel::getTiledOpOperands(
        mlir::Operation* op, const TileInfo& outTile, std::optional<VPU::MultiClusterStrategy> customStrategy) {
    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (tilingOp == nullptr) {
        std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> ndTypes;
        ndTypes.reserve(op->getNumOperands() + op->getNumResults());
        for (auto type : op->getOperandTypes()) {
            ndTypes.push_back(std::make_pair(mlir::cast<NDTypeInterface>(type), TensorDistributionMap{}));
        }
        for (auto type : op->getResultTypes()) {
            ndTypes.push_back(std::make_pair(mlir::cast<NDTypeInterface>(type), TensorDistributionMap{}));
        }
        return ndTypes;
    }

    auto inTiles = tilingOp.backInferTileInfo(outTile, _log);
    const auto& inputTiles = inTiles.tiles;
    const auto outputTiles = tilingOp.getOutputTiling(outTile, _log);

    VPUX_THROW_UNLESS(inputTiles.size() == op->getNumOperands(),
                      "Unexpected inputTile size '{0}' and Op operands size '{1}'", inputTiles.size(),
                      op->getNumOperands());

    VPUX_THROW_UNLESS(outputTiles.size() == op->getNumResults(),
                      "Unexpected outputTile size '{0}' and Op results size '{1}'", outputTiles.size(),
                      op->getNumResults());

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr || !customStrategy.has_value()) {
        std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> ndTypes;
        for (const auto& [input, inputTile] : zip(op->getOperands(), inputTiles)) {
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
            const auto type = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
            ndTypes.push_back(std::make_pair(type, TensorDistributionMap{}));
        }
        for (const auto& [output, outputTile] : zip(op->getResults(), outputTiles)) {
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
            const auto type = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
            ndTypes.push_back(std::make_pair(type, TensorDistributionMap{}));
        }
        return ndTypes;
    }

    return getTileDistributions(op, _siblingsOpsAnalysis, outTile, customStrategy, inTiles);
}

HwLayerTilingStrategyCosts LayerCostModel::getDPUandDMATimeCostWithCustomTiling(VPU::NCEOpInterface nceOp,
                                                                                VPU::MultiClusterStrategy strategy,
                                                                                const OutputTiling& outTiles) const {
    // Types for each tile
    std::vector<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes;

    _log.trace("[Cost Analysis] {0} [{1}]  Get VPUNN cost for {2} {3}", nceOp.getLoc(), nceOp->getName(), strategy,
               outTiles[0].axis);

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs, _numTiles, strategy);

    auto distributionMode = DistributionMode::NONE;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp != nullptr) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
        auto tiledOutputType = outputType.extractDenseTile(outTiles[0].offsets, outTiles[0].shape);
        distributionMode = getOutputTensorDistributionMode(clusteredOp, strategy, tiledOutputType);
    }

    SmallVector<uint32_t> vpunnLayerDPUCosts;
    SmallVector<uint32_t> vpunnOriginalLayerDPUCosts;
    const auto vpunnStrategy =
            VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, _arch, 1, true, distributionMode, nceOp);

    const auto enableVPUNNPreSplit = config::hasVPUNNPreSplit(nceOp);
    if (enableVPUNNPreSplit && !isActSparseOp(nceOp)) {
        // Track E#160972. Activation sparse op accuracy issue
        vpunnLayerDPUCosts = getDPUCostForNCEOpPreSplit(nceOp, outTiles, costParams, vpunnStrategy.tiling_strategy,
                                                        _layerCostModel, _numDPUs, _log);
    } else {
        vpunnLayerDPUCosts =
                getDPUCostForNCEOp(nceOp, strategy, outTiles, costParams, vpunnStrategy, _layerCostModel, _log);
    }
    if (vpunnLayerDPUCosts.empty()) {
        return {COST_MAX, COST_MAX};
    }
    _log.trace("VPUNN DPU layer costs {0}", vpunnLayerDPUCosts);
    vpunnOriginalLayerDPUCosts = vpunnLayerDPUCosts;

    const auto getSpillingReadCostFunc = [&](NDTypeInterface srcType,
                                             const TensorDistributionMap& distributions) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingDMACost(srcType, distributions));
    };

    const auto getSpillingWriteCostFunc = [&](NDTypeInterface srcType,
                                              const TensorDistributionMap& distributions) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingDMACost(srcType, distributions));
    };

    VPUX_THROW_WHEN(outTiles.empty(), "Empty output tiles");
    const auto inOrder = DimsOrder::fromValue(nceOp->getOperand(0));
    const auto outOrder = DimsOrder::fromValue(nceOp->getResult(0));
    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
    // [E#221093] Fix the inputTiledOnLowestDim with correct inTiles instead of outTiles
    const auto inputTiledOnLowestDim = isTiledOnLowestDim(outTiles[0], inOrder);
    const auto outputTiledOnLowestDim = isTiledOnLowestDim(outTiles[0], outOrder);
    bool correctedStrideActDMACost = false;
    bool correctedStrideOutputDMACost = false;

    // When tiles are split on the lowest dim, the DMA can become strided and VPUNN underestimates it.
    // Correct with an architecture-dependent threshold until NNDMA model is fully integrated.
    auto activationTileTypeGetter = [](ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> tilesType) {
        auto [srcType, distributionMap] = tilesType.front();
        return getDistributedTypeFromDistributionMap(srcType, distributionMap);
    };

    auto outputTileTypeGetter = [](ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> tilesType) {
        auto [srcType, distributionMap] = tilesType.back();
        return getDistributedTypeFromDistributionMap(srcType, distributionMap);
    };

    // Accumulate all the DPU costs
    double cost = std::accumulate(vpunnLayerDPUCosts.begin(), vpunnLayerDPUCosts.end(), 0.0);
    double costWithPrefetching = cost;
    for (auto& outTile : outTiles) {
        auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, _log);
        tilesTypes.push_back(
                getTileDistributions(nceOp.getOperation(), _siblingsOpsAnalysis, outTile, strategy, inTiles));
    }
    _log.trace(" VPUNN accumulated DPU cost {0}", cost);

    // Add weights DMA costs, which considers DMA prefetching and pipelining and only accumulates extra DMA costs based
    // on DPU cost
    auto vpunnLayerWeightsCosts =
            getPerTileWeightsDMACosts(nceOp, strategy, _siblingsOpsAnalysis, tilesTypes, getSpillingReadCostFunc);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    _log.trace("vpunnLayerDPUCosts {0}", vpunnLayerDPUCosts);
    const auto [weightsCost, weightsCostWithPrefetching] = getWeightsDMACostForNCEOp(
            nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerWeightsCosts, _enablePrefetchTiling, _log);
    _log.trace(" VPUNN accumulated weights DMA cost {0} with prefetchTiling option {1}", weightsCost,
               _enablePrefetchTiling);
    cost += weightsCost;
    costWithPrefetching += weightsCostWithPrefetching;
    _log.trace("Include Weights DMA cost {0}, Full layer cost now {1}", weightsCost, cost);
    _log.trace("Include Weights Prefetch DMA cost {0}, Full Prefetch layer cost now {1}", weightsCostWithPrefetching,
               costWithPrefetching);

    auto getParentOp = [&]() -> mlir::Operation* {
        mlir::Operation* parentOp = nceOp->getOperand(0).getDefiningOp();
        while (parentOp && isPureViewOp(parentOp)) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }
        return parentOp;
    };
    // Add activation DMA costs
    // Subgraph optimization will calculate input spilling cost instead of activation dma cost
    SmallVector<uint32_t> vpunnLayerActCosts;
    if (!isUnderSubgraphOpt() || getParentOp() == nullptr) {
        vpunnLayerActCosts =
                getPerTileActivationDMACosts(nceOp, tilesTypes, getSpillingReadCostFunc, strategy, _numTiles);
        // TODO: Ticket E#135490, remove this after stride DMA cost is accurate
        correctedStrideActDMACost =
                correctStrideDMACostOnAllTiles(tilesTypes, activationTileTypeGetter, vpunnLayerActCosts,
                                               inputTiledOnLowestDim, config::getArch(nceOp));
        _log.trace("VPUNN activation DMA costs {0}", vpunnLayerActCosts);
        _log.trace("vpunnLayerDPUCosts {0}", vpunnLayerDPUCosts);
        const auto actCost = getActivationDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerActCosts,
                                                          _enablePrefetchTiling, _log);
        cost += actCost;
        costWithPrefetching += actCost;
        _log.trace("Include Activation DMA cost {0}, Full layer cost now {1}", actCost, cost);
        _log.trace("Include Activation DMA cost {0}, Prefetched full layer cost now {1}", actCost, costWithPrefetching);
    }

    // Add output spilling cost
    // for non clusteredOp, must be ops that requires tiling
    VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op at '{0}' is not a clustered op", nceOp.getLoc());
    SmallVector<uint32_t> vpunnLayerOutputCosts;
    if (!clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0))) {
        // Consider output spilling pipelining with the next tile's DPU
        // Might be inaccurate when the DPU time is smaller than the sum of DMA time (input + weights + output)
        vpunnLayerOutputCosts = getPerTileOutputDMACosts(nceOp, tilesTypes, getSpillingWriteCostFunc);
        // TODO: Ticket E#135490, remove this after stride DMA cost is accurate
        correctedStrideOutputDMACost =
                correctStrideDMACostOnAllTiles(tilesTypes, outputTileTypeGetter, vpunnLayerOutputCosts,
                                               outputTiledOnLowestDim, config::getArch(nceOp));
        _log.trace("VPUNN output DMA costs {0}", vpunnLayerOutputCosts);
        _log.trace("vpunnLayerDPUCosts {0}", vpunnLayerDPUCosts);
        const auto outCost = getOutputDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerOutputCosts,
                                                      _enablePrefetchTiling, _log);
        cost += outCost;
        costWithPrefetching += outCost;
        _log.trace("Include Output DMA cost {0}, Full layer cost now {1}", outCost, cost);
        _log.trace("Include Output DMA cost {0}, Prefetched full layer cost now {1}", outCost, costWithPrefetching);
    }

    _log.nest(2).trace("[Cost Analysis] Full DPU costs {0}", vpunnOriginalLayerDPUCosts);
    _log.nest(2).trace("[Cost Analysis] Weights costs {0}", vpunnLayerWeightsCosts);
    _log.nest(2).trace("[Cost Analysis] Activation costs {0}", vpunnLayerActCosts);
    _log.nest(2).trace("[Cost Analysis] Output costs {0}", vpunnLayerOutputCosts);
    _log.nest(2).trace("[Cost Analysis] Non-Pipelined DPU cost remainder {0}", vpunnLayerDPUCosts);
    if (correctedStrideActDMACost || correctedStrideOutputDMACost) {
        _log.nest(4).trace(
                "[Cost Analysis] Note: DMA costs have been corrected for stride DMA. Activation {0}, Output {1}",
                correctedStrideActDMACost, correctedStrideOutputDMACost);
    }
    _log.trace("[Cost Analysis] Final Full Cost with No Prefetch: {0}", cost);
    _log.trace("[Cost Analysis] Final Full Cost with Prefetching: {0}", costWithPrefetching);

    return {cost, costWithPrefetching};
}

/// @brief Time-cost : return a sum of layer DPU time and weights DMA time (cycles)
/// @details DPU time calculation also considers the impact of workloads split efficiency
double LayerCostModel::getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) {
    {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op {0} at {1} should be a clustered op", nceOp->getName(),
                        nceOp.getLoc());

        // Set customized strategy to the op to get corresponding output tiles when tiling
        // Save and restore original strategy if needed
        const auto origStrategy = clusteredOp.getMultiClusterStrategy();
        clusteredOp.setMultiClusterStrategy(strategy);
        VPUX_SCOPE_EXIT {
            if (origStrategy.has_value()) {
                clusteredOp.setMultiClusterStrategy(*origStrategy);
            } else {
                clusteredOp->removeAttr(VPU::multiClusterStrategy);
            }
        };

        if (_enableTilingFullSearchSpace) {
            if (auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation())) {
                // Scope online profiling to this full-search query.
                const VPU::ScopedProfilingForAutoHint profilingScope(_layerCostModel->get_cost_model_shared());

                if (auto bestTilingInfo = VPU::TemporalTilingDriver::getBestTilingStrategy(tilingOp, *this, _log)) {
                    return bestTilingInfo->costInfo.overallCost;
                }
            }
        }

        // Output tiling for each tile
        OutputTiling outTiles({TileInfo(getShape(nceOp->getResult(0)))});

        // Check CMX memory as VPUNN works with layer which fits CMX memory
        // If not, tiling big layer to fit into CMX
        if (!(clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0)))) {
            _log.trace("Tiling op {0} to fit into cmx before passing to VPUNN Layer API", nceOp.getLoc());
            auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
            VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                            nceOp.getLoc());
            auto costModel = std::make_shared<vpux::VPU::LayerCostModel>(*this);
            auto tiles = getHWLayerTilingStrategy(tilingBuilderOp, _enablePrefetchTiling, costModel, _log);

            if (mlir::failed(tiles)) {
                _log.trace("Invalid tiling strategy for {0}", nceOp->getName());
                return COST_MAX;
            }
            outTiles = tiles.value();
        }

        // If the output size of nceOp on dimension C exceeds VPU_DIMENSION_LIMIT,
        // we tile it again to avoid potential VPUNN errors in MC strategy assignment.
        // Tracking Number: E#140892
        int tileNum = outTiles.size();
        for (int tileIdx = 0; tileIdx < tileNum; tileIdx++) {
            ShapeRef shape = outTiles[tileIdx].shape;
            if (shape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                _log.debug("Tiled Op size {0} is still over VPU_DIMENSION_LIMIT on C, tile again to avoid VPUNN error.",
                           shape);
                // tile the oversized ops into 16-aligned tiles under VPU_DIMENSION_LIMIT
                int shapeOnC = shape[Dims4D::Act::C];
                int oversizeFactor = shapeOnC / VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
                if (shapeOnC % VPU::NCEInvariant::VPU_DIMENSION_LIMIT != 0) {
                    oversizeFactor += 1;
                }
                int tiledShapeOnC = shapeOnC / oversizeFactor;
                tiledShapeOnC += (tiledShapeOnC % 16 == 0) ? 0 : 16 - tiledShapeOnC % 16;
                int remainderOnC = shapeOnC - (oversizeFactor - 1) * tiledShapeOnC;

                // update the original tile
                outTiles[tileIdx].shape[Dims4D::Act::C] = remainderOnC;
                for (int i = 1; i < oversizeFactor; i++) {
                    TileInfo newTile = TileInfo(shape);
                    newTile.shape[Dims4D::Act::C] = tiledShapeOnC;
                    outTiles.push_back(newTile);
                }
            }
        }

        auto [costWithoutPrefetching, costWithPrefetching] =
                getDPUandDMATimeCostWithCustomTiling(nceOp, strategy, outTiles);
        auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
        double cost = 0;
        if (_enablePrefetchTiling && tilingInfoOp != nullptr &&
            tilingInfoOp.isSupportedTiling(outTiles, vpux::TilingMode::PREFETCHING, _log)) {
            cost = costWithPrefetching;
        } else {
            cost = costWithoutPrefetching;
        }

        _log.trace("VPUNN total layer cost for {0} and {1} is {2}", strategy, outTiles[0].axis, cost);

        return cost;
    }
}

///@brief Effi-cost : A simple cost considering DPU computing efficiency
double LayerCostModel::getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    return 1.0 / computeSplitEfficiency(nceOp, strategy);
}

bool LayerCostModel::hasMultiClusterStrategy(mlir::Operation* op) const {
    if (auto clusteringOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        return clusteringOp.getMultiClusterStrategy().has_value();
    }

    return false;
}

VPU::MultiClusterStrategy LayerCostModel::getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const {
    auto strategy = clusteredOp.getMultiClusterStrategy();
    if (!strategy.has_value()) {
        VPUX_THROW("NCE operation {0} doesn't have multiClusterStrategy attribute", clusteredOp->getLoc());
    }

    return strategy.value();
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface /*clusteredOp*/,
                                 std::pair<mlir::Type, TensorDistributionMap>& srcTensorType,
                                 std::pair<mlir::Type, TensorDistributionMap>& dstTensorType) const {
    auto getActivationTypeFromSparseType = [](std::pair<mlir::Type, TensorDistributionMap>& tensorType) {
        VPU::DistributionInfo distribution{};
        if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tensorType.first)) {
            if (tensorType.second.contains(sparseTensorType.getData())) {
                distribution = tensorType.second.at(sparseTensorType.getData());
            }
            // interested in activation spills so use data for compatibility
            return std::make_pair(sparseTensorType.getData(), distribution);
        }
        if (tensorType.second.contains(tensorType.first)) {
            distribution = tensorType.second.at(tensorType.first);
        }
        return std::make_pair(tensorType.first, distribution);
    };

    auto srcTensor = getActivationTypeFromSparseType(srcTensorType);
    auto dstTensor = getActivationTypeFromSparseType(dstTensorType);

    if (isTargetTensorTypeCompatible(srcTensor, dstTensor) ||
        isSOHAlignmentCompatibleOrAdjustedCompatible(srcTensor, dstTensor)) {
        return false;
    }
    return true;
}

/// Anywhere if you need to consider possible spilling, please call me!
/// srcTensorType is the output of parent origOp
/// dstTensorType is the input of child NCE op
bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface op, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const {
    auto srcTensorDistributionMap =
            std::make_pair(mlir::cast<mlir::Type>(srcTensorType), getDistributionMapFromDistributedType(srcTensorType));
    auto dstTensorDistributionMap =
            std::make_pair(mlir::cast<mlir::Type>(dstTensorType), getDistributionMapFromDistributedType(dstTensorType));
    return hasSpilling(op, srcTensorDistributionMap, dstTensorDistributionMap);
}

std::pair<std::pair<mlir::Type, TensorDistributionMap>, std::pair<mlir::Type, TensorDistributionMap>>
LayerCostModel::getDistributionsWithStrategy(VPU::ClusteredOpInterface parentOp,
                                             VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userOp,
                                             VPU::MultiClusterStrategy userStrategy) const {
    // Set the custom strategy to the op to get the accurate distributed type
    // The distribution mode depends on the neighboring op's strategy
    // e.g., Conv (SOK) -> SW (SOK), the output of the Conv would be SEGMENTED
    // Conv (SOK) -> SW (Clustering), the output of the Conv would be DUPLICATED|SEGMENTED
    // The DistributedType is decided by the ops multiCluster strategy attributes
    auto greedyStrategyParentOp = getMultiClusterStrategyValue(parentOp);
    auto greedyStrategyUserOp = getMultiClusterStrategyValue(userOp);
    parentOp.setMultiClusterStrategy(parentStrategy);
    userOp.setMultiClusterStrategy(userStrategy);
    auto targetOutput = getOutputWithDistribution(parentOp, parentStrategy);
    auto targetInput = getInputWithDistribution(userOp, parentOp, userStrategy);
    parentOp.setMultiClusterStrategy(greedyStrategyParentOp);
    userOp.setMultiClusterStrategy(greedyStrategyUserOp);

    VPU::DistributionInfo outputDistribution{};
    auto outputType = mlir::isa<VPU::SparseTensorType>(targetOutput.first)
                              ? mlir::cast<vpux::VPU::SparseTensorType>(targetOutput.first).getData()
                              : targetOutput.first;
    if (targetOutput.second.contains(outputType)) {
        outputDistribution = targetOutput.second.at(outputType);
    }

    VPU::DistributionInfo inputDistribution{};
    auto inputType = mlir::isa<VPU::SparseTensorType>(targetInput.first)
                             ? mlir::cast<vpux::VPU::SparseTensorType>(targetInput.first).getData()
                             : targetInput.first;
    if (targetInput.second.contains(inputType)) {
        inputDistribution = targetInput.second.at(inputType);
    }

    // Adjust inputType alignment for SW op
    // e.g., Conv (SOK) -> SW (SOK), the input of SW can have a same alignment with Conv
    // to avoid spilling

    auto parentOutAlignment = outputDistribution.getAlignment();
    auto UserInAlignment = inputDistribution.getAlignment();
    if (!parentOutAlignment.empty() && UserInAlignment.empty() &&
        mlir::isa<VPU::SWOpInterface>(userOp.getOperation()) &&
        isSWOpChannelAlignmentCompatible(userOp, targetInput.first,
                                         mlir::cast<vpux::NDTypeInterface>(userOp->getResult(0).getType()))) {
        targetInput = getInputWithDistribution(userOp, parentOp, userStrategy, parentOutAlignment);
    }
    return std::make_pair(targetOutput, targetInput);
}

std::pair<vpux::NDTypeInterface, vpux::NDTypeInterface> LayerCostModel::getDistributionTypesWithStrategy(
        VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userOp,
        VPU::MultiClusterStrategy userStrategy) const {
    auto [outDist, inDist] = getDistributionsWithStrategy(parentOp, parentStrategy, userOp, userStrategy);
    return std::make_pair(getDistributedTypeFromDistributionMap(outDist.first, outDist.second),
                          getDistributedTypeFromDistributionMap(inDist.first, inDist.second));
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const {
    auto targetTypes = getDistributionsWithStrategy(origOp, origOpStrategy, userOp, userOpStrategy);
    auto targetOutputType = targetTypes.first;
    auto targetInputType = targetTypes.second;
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::doesLayerRequireTiling(VPU::ClusteredOpInterface clusteredOp,
                                            VPU::MultiClusterStrategy strategy) const {
    return !(clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0)));
}

bool LayerCostModel::doesLayerHaveVPUNNSupportedTypes(VPU::ClusteredOpInterface clusteredOp) const {
    const auto arch = _arch;
    const bool hasSupportedOperandTypes = llvm::all_of(clusteredOp->getOperands(), [arch](const mlir::Value val) {
        return vpux::VPU::isVPUNNSupportedElementType(mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType(),
                                                      arch);
    });
    const bool hasSupportedResultTypes = llvm::all_of(clusteredOp->getResults(), [arch](const mlir::Value val) {
        return vpux::VPU::isVPUNNSupportedElementType(mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType(),
                                                      arch);
    });
    return hasSupportedOperandTypes && hasSupportedResultTypes;
}

LayerCostModel::SpillingCost LayerCostModel::calculateSpillingCost(VPU::ClusteredOpInterface parentOp,
                                                                   VPU::ClusteredOpInterface userOp,
                                                                   VPU::MultiClusterStrategy parentStrategy,
                                                                   VPU::MultiClusterStrategy userStrategy) const {
    auto [targetOutput, targetInput] = getDistributionsWithStrategy(parentOp, parentStrategy, userOp, userStrategy);
    return getSpillingCost(targetOutput.first, targetOutput.second, targetInput.first, targetInput.second, parentOp,
                           userOp);
}

bool LayerCostModel::useSOKWhenMVNSideOrFitCMX(VPU::ClusteredOpInterface clusteredOp,
                                               ArrayRef<StrategyInfoPair> bucket) {
    // This check is a W/A for NPU40 or lower arch, where VPUNN cost is not accurate or not available (error code)
    // For newer platforms, a more accurate VPUNN cost model is provided so that this check should be dropped
    auto arch = config::getArch(clusteredOp.getOperation());
    if (arch > config::ArchKind::NPU40XX) {
        return false;
    }

    if (!mlir::isa<mlir::BlockArgument>(clusteredOp->getOperand(0)) &&
        mlir::isa<VPU::MVNOp>(clusteredOp->getOperand(0).getDefiningOp())) {
        // MVN producer
        return true;
    }
    for (auto* user : clusteredOp->getResult(0).getUsers()) {
        // MVN consumer
        if (mlir::isa<VPU::MVNOp>(user)) {
            return true;
        }
    }

    // Only needed for pre-NPU4 arch. If SOK is the only strategy that fits in CMX, choose it.
    if (arch == config::ArchKind::NPU37XX) {
        int numFitsIntoCMX = llvm::count_if(bucket, [](const auto& p) {
            return p.second.fitsIntoCMX;
        });

        if (numFitsIntoCMX != 1) {
            return false;
        }

        const bool sokFitsIntoCMX = llvm::any_of(bucket, [](const auto& p) {
            return p.first == MultiClusterStrategy::SplitOverKernel && p.second.fitsIntoCMX;
        });

        return sokFitsIntoCMX;
    }

    // arch == config::ArchKind::NPU40XX
    return false;
}

bool LayerCostModel::preferChannelSplitting(VPU::ClusteredOpInterface clusteredOp) {
    bool preferChannels = false;
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
        if (nceOp.getWeightsOperand() != nullptr) {
            auto activationSize = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType()).getTotalAllocSize();
            activationSize += mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getTotalAllocSize();
            auto weightSize =
                    mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType()).getTotalAllocSize();

            if (activationSize < weightSize) {
                preferChannels = true;
            }
        }
    }
    return preferChannels;
}

bool LayerCostModel::isCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) {
    bool isOperationCompatible = false;

    switch (strategy) {
    case MultiClusterStrategy::SplitOverHeight:
        isOperationCompatible =
                clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()));
        break;
    case MultiClusterStrategy::SplitOverKernel:
        isOperationCompatible =
                clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                 /*axis=*/ShapeRef());
        break;
    default:
        VPUX_THROW("Invalid strategy {0} considered for operation {1}", strategy, clusteredOp->getName());
    }

    return isOperationCompatible && clusteredOp.checkStrategyCompatibility(strategy, _numTiles);
}

bool LayerCostModel::usesFullTiles(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) {
    return _numTiles ==
           VPU::getOptimalNumClusters(clusteredOp, getBoundedShape(clusteredOp->getResult(0).getType()), strategy);
}

VPU::MultiClusterStrategy LayerCostModel::getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    // #E-190170 enable non-SOH/SOK cases as allowed by the op
    SmallVector<VPU::MultiClusterStrategy> potentialStrategies = {VPU::MultiClusterStrategy::SplitOverHeight,
                                                                  VPU::MultiClusterStrategy::SplitOverKernel};

    _log.trace("Optimal Layer Strategy for {0}", clusteredOp->getLoc());

    // Define PriorityBuckets into which we will sort potential multi-cluster strategies
    // If we find a strategy in a higher priority bucket, no need to look at any further buckets
    llvm::DenseMap<PriorityBucket, SmallVector<StrategyInfoPair>> priorityBuckets;

    // Map used for npu37xx workaround based on heuristics, for soh vs sok selection depending on cmx fitting
    llvm::DenseMap<VPU::MultiClusterStrategy, MultiClusterStrategyInfo> strategyInfoMap;

    const auto optimalMCTiling = [&](VPU::MultiClusterStrategy strategy) {
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
            return mlir::isa<vpux::VPU::NCECompressConvolutionOp, vpux::VPU::NCEPermuteOp>(clusteredOp)
                           ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                           : VPU::MultiClusterStrategy::SplitOverHeight;
        }
        // For any other strategy, return it unchanged
        return strategy;
    };

    auto getPriority = [](const auto& strategy) -> PriorityBucket {
        if (strategy.cost < COST_MAX) {
            return strategy.fitsIntoCMX ? PriorityBucket::COST_BASED_FITS_CMX : PriorityBucket::COST_BASED_NOT_FITS_CMX;
        }
        if (strategy.fitsIntoCMX && strategy.usesFullTiles) {
            return PriorityBucket::HEURISTIC_FITS_CMX_FULL_TILES;
        }
        return PriorityBucket::HEURISTIC_COMPATIBLE_ONLY;
    };

    // Step 1: Sort all strategies into the correct bucket by priority
    for (auto strategy : potentialStrategies) {
        _log.trace("Consider strategy {0}", strategy);
        if (!isCompatible(clusteredOp, strategy)) {
            _log.trace("Strategy {0} is incompatible, discard", strategy);
            continue;
        }

        MultiClusterStrategyInfo strategyInfo;

        strategyInfo.fitsIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(strategy, _siblingsOpsAnalysis, /*reservedMem=*/Byte(0));
        strategyInfo.cost = getLayerCost(clusteredOp, strategy);
        strategyInfo.usesFullTiles = usesFullTiles(clusteredOp, strategy);
        strategyInfoMap[strategy] = strategyInfo;

        auto bucketKey = getPriority(strategyInfo);
        priorityBuckets[bucketKey].emplace_back(strategy, strategyInfo);
        _log.trace("Strategy {0} saved for evaluation in priority bucket {1}", strategy, static_cast<int>(bucketKey));
    }

    auto arch = config::getArch(clusteredOp.getOperation());
    if (arch == config::ArchKind::NPU37XX) {
        // In NPU37XX the cost model is not accurate, so implementation relies more on heuristics.
        // In this case, give precedence to the case in which one strategy fits in CMX, regardless of costs
        auto sohIt = strategyInfoMap.find(VPU::MultiClusterStrategy::SplitOverHeight);
        auto sokIt = strategyInfoMap.find(VPU::MultiClusterStrategy::SplitOverKernel);

        if (sohIt != strategyInfoMap.end() && sokIt != strategyInfoMap.end()) {
            const auto& sohInfo = sohIt->second;
            const auto& sokInfo = sokIt->second;
            // the case in which one or both strategies are not compatible (hence keys not in the map) is managed by the
            // general implementation
            if (sohInfo.fitsIntoCMX && !sokInfo.fitsIntoCMX) {
                _log.trace("Strategy SplitOverHeight chosen as it fits into CMX");
                return optimalMCTiling(sohIt->first);
            }
            if (!sohInfo.fitsIntoCMX && sokInfo.fitsIntoCMX && sokInfo.usesFullTiles) {
                _log.trace("Strategy SplitOverKernel chosen as it fits into CMX");
                return VPU::MultiClusterStrategy::SplitOverKernel;
            }
        }
    }

    // Step 2: Choose best strategy
    // All strategies have been sorted into priority buckets.
    // The two highest priority buckets contains only strategies with valid VPUNN cost.
    // If they are non-empty, return the strategy with lowest cost.

    // Buckets 0|1: pick lowest cost
    // Use same evaluation criteria, but look in bucket 0 first (fits in cmx, lowest cost)
    for (PriorityBucket key : {PriorityBucket::COST_BASED_FITS_CMX, PriorityBucket::COST_BASED_NOT_FITS_CMX}) {
        auto bucketIt = priorityBuckets.find(key);
        if (bucketIt == priorityBuckets.end() || bucketIt->second.empty()) {
            continue;
        }
        auto& costBucket = bucketIt->second;
        // W/A for 40XX and earlier. In this case choose SOK if exists in the bucket, regardless of cost
        if (useSOKWhenMVNSideOrFitCMX(clusteredOp, costBucket)) {
            auto sokIt = std::find_if(costBucket.begin(), costBucket.end(), [](const auto& s) {
                return s.first == VPU::MultiClusterStrategy::SplitOverKernel;
            });
            if (sokIt != costBucket.end()) {
                _log.trace("Strategy SplitOverKernel chosen from cost based priority bucket due to MVN workaround");
                return sokIt->first;
            }
        }

        auto bestStrategy = std::min_element(costBucket.begin(), costBucket.end(), [](auto& a, auto& b) {
                                return a.second.cost < b.second.cost;
                            })->first;
        _log.trace("Strategy {0} chosen from priority bucket {1}", bestStrategy, static_cast<int>(key));
        return optimalMCTiling(bestStrategy);
    }

    // Step 2a - no strategies have valid cost, choose best strategy
    // Buckets 2 and 3: heuristic selection.
    // Use same evaluation criteria, but look in bucket 2 first

    // Decide if we prefer splitting activations or weights based on size
    bool preferChannels = preferChannelSplitting(clusteredOp);

    for (PriorityBucket key :
         {PriorityBucket::HEURISTIC_FITS_CMX_FULL_TILES, PriorityBucket::HEURISTIC_COMPATIBLE_ONLY}) {
        auto bucketIt = priorityBuckets.find(key);
        if (bucketIt == priorityBuckets.end() || bucketIt->second.empty()) {
            continue;
        }

        auto& bucket = bucketIt->second;
        auto preferredStrategy = preferChannels ? VPU::MultiClusterStrategy::SplitOverKernel
                                                : VPU::MultiClusterStrategy::SplitOverHeight;

        auto it = llvm::find_if(bucket, [&](const StrategyInfoPair& pair) {
            return pair.first == preferredStrategy;
        });
        if (it != bucket.end()) {
            _log.trace("Preferred strategy {0} chosen from bucket {1}", preferredStrategy, static_cast<int>(key));
            return optimalMCTiling(it->first);
        }

        // Fallback: take the first valid strategy
        auto chosenStrategy = optimalMCTiling(bucket.front().first);
        _log.trace("Fallback: Strategy {0} chosen from bucket {1}", chosenStrategy, static_cast<int>(key));
        return chosenStrategy;
    }

    // If no compatibile strategies found in any bucket, fallback to clustering
    _log.trace("No compatible strategy found, fallback to Clustering");
    return VPU::MultiClusterStrategy::Clustering;
}

bool LayerCostModel::isUnderSubgraphOpt() const {
    return _underSubgraphOpt;
}
void LayerCostModel::setUnderSubgraphOpt(bool underSubgraphOpt) {
    _underSubgraphOpt = underSubgraphOpt;
}

bool vpux::VPU::isStrategySOXCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                        size_t numTiles) {
    if (clusteredOp == nullptr) {
        return false;
    }

    if (clusteredOp.checkStrategyCompatibility(strategy, numTiles)) {
        switch (strategy) {
        case VPU::MultiClusterStrategy::SplitOverHeight:
        case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
            return clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()));
        case VPU::MultiClusterStrategy::SplitOverKernel:
            return clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(),
                                                                    /*offset=*/ShapeRef(),
                                                                    /*axis=*/ShapeRef());
        case VPU::MultiClusterStrategy::SplitOverWidth:
            return clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(),
                                                                   /*offset=*/ShapeRef(),
                                                                   /*axis=*/ShapeRef());
        default:
            return false;
        }
    }

    return false;
}

// For a clustered op which doesn't support cycle cost calculation the priority for strategies is parent strategy >
// SOH/SOHOverlapped > SOK > SOW > Clustering
// Note: parent strategy will take precedence, but it is delayed just before subgraph optimization to allow
// multi-threading of this pass in the future
std::optional<VPU::MultiClusterStrategy> vpux::VPU::getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    auto module = clusteredOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(module);
    const auto numTiles = tileOp.getCount();

    // Only the highest dimension is prioritized
    // Need to investigate if the complete layout order could be optimal
    // Track E#124146
    auto strategyOrder = SmallVector(
            {VPU::MultiClusterStrategy::SplitOverHeight, VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
             VPU::MultiClusterStrategy::SplitOverBatch, VPU::MultiClusterStrategy::SplitOverKernel,
             VPU::MultiClusterStrategy::SplitOverWidth});
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
    const auto outputShape = getBoundedShape(outputType);
    const auto highestDim = vpux::getHighestNonTrivialDim(outputShape, outputType.getDimsOrder()).value_or(Dim(0));
    const auto numActShaves = config::getTotalNumOfEngines(clusteredOp, config::ExecutorKind::SHAVE_ACT);

    DenseMap<int64_t, SmallVector<VPU::MultiClusterStrategy>> dimToStrategyMap{
            {Dims4D::Act::N.ind(), {VPU::MultiClusterStrategy::SplitOverBatch}},
            {Dims4D::Act::H.ind(),
             {VPU::MultiClusterStrategy::SplitOverHeight, VPU::MultiClusterStrategy::SplitOverHeightOverlapped}},
            {Dims4D::Act::W.ind(), {VPU::MultiClusterStrategy::SplitOverWidth}},
            {Dims4D::Act::C.ind(), {VPU::MultiClusterStrategy::SplitOverKernel}},
    };

    auto updateStrategyOrder = [&](Dim priorityDim) {
        const auto& strategies = dimToStrategyMap[priorityDim.ind()];
        for (const auto& strategy : strategies) {
            strategyOrder.erase(llvm::find(strategyOrder, strategy));
        }
        strategyOrder.insert(strategyOrder.begin(), strategies.begin(), strategies.end());
    };

    // For EltwiseOp and DynamicQuantizeOp, any layout is acceptable. Prioritize the highest dimension for MC and MS
    // tiling
    const auto isEltwiseSwOp =
            clusteredOp->hasTrait<VPU::EltwiseOp>() && mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation());
    const auto isDQ = mlir::isa_and_nonnull<VPU::DynamicQuantizeOp>(clusteredOp.getOperation());
    const auto isMaxPool8 = mlir::isa_and_nonnull<VPU::MaxPool8Op>(clusteredOp.getOperation());
    if ((isEltwiseSwOp || isDQ) && outputShape[highestDim] >= numActShaves) {
        updateStrategyOrder(highestDim);
    }
    if (isMaxPool8) {
        SmallVector<Dim> dimsAscending = {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
        std::sort(dimsAscending.begin(), dimsAscending.end(), [&outputShape](Dim a, Dim b) {
            return outputShape[a] < outputShape[b];
        });
        // Order the strategy based on size of dims, biggest dim will be prioritized
        for (auto dim : dimsAscending) {
            updateStrategyOrder(dim);
        }
    }

    // For DequantizeOp with layout [OC, IC, KY, KX], SOK segmentation is on OC
    const auto isWeightDequant =
            mlir::isa_and_nonnull<VPU::DequantizeOp>(clusteredOp.getOperation()) && highestDim == Dims4D::Filter::OC;
    if (highestDim == Dims4D::Act::C || isWeightDequant) {
        updateStrategyOrder(Dims4D::Act::C);
    }

    auto findDimCanSupportShaveTiling = [&]() -> std::optional<Dim> {
        if (!isEltwiseSwOp) {
            return std::nullopt;
        }
        const auto dimOrder = outputType.getDimsOrder();
        if (dimOrder.dimPos(highestDim) == dimOrder.numDims() - 1) {
            return std::nullopt;
        }

        if (outputShape[highestDim] >= numActShaves) {
            return std::nullopt;
        }

        auto dimRange = irange(dimOrder.dimPos(highestDim) + 1, dimOrder.numDims());
        auto shaveTilingSupportDim = llvm::find_if(dimRange, [&](size_t dimIdx) {
            return outputShape[dimOrder.dimAt(dimIdx)] >= numActShaves;
        });
        if (shaveTilingSupportDim == dimRange.end()) {
            return std::nullopt;
        }
        const auto tileDim = dimOrder.dimAt(*shaveTilingSupportDim);

        // Sigmoid with small data size are not performant with shave tiling, so we need to add related check here.
        // It's supposed to be same one in TileActShaveKernelTaskPass.
        if (auto sigmoidOp = mlir::dyn_cast<VPU::SigmoidOp>(clusteredOp.getOperation())) {
            // E#92211: Measurements for the performance profiling, see this ticket for details.
            auto tileOp = config::getTileExecutor(clusteredOp->getParentOfType<mlir::ModuleOp>());
            auto dpuCount = tileOp.getCount();
            VPUX_THROW_WHEN(dpuCount <= 0, "Invalid number of clusters: {0}", dpuCount);
            auto outType = mlir::cast<vpux::NDTypeInterface>(sigmoidOp.getOutput().getType());
            Shape tiledOutShape = Shape(outputShape);
            tiledOutShape[tileDim] = tiledOutShape[tileDim] / dpuCount;
            auto tiledOutType = outType.changeShape(tiledOutShape);
            auto tiledSize = tiledOutType.getTotalAllocSize() / dpuCount;
            if (tiledSize < VPUIP::SIGMOID_SW_KERNEL_TILING_THRESHOLD) {
                return std::nullopt;
            }
        }
        return tileDim;
    };
    auto shaveTilingDim = findDimCanSupportShaveTiling();
    if (shaveTilingDim.has_value()) {
        // The compiler tend to use strategy which also supports shave tiling, since the gain of multi shaves
        // usually is greater than the loss of stride DMA.
        updateStrategyOrder(shaveTilingDim.value());
    }

    for (auto strategy : strategyOrder) {
        if (!clusteredOp.checkStrategyCompatibility(strategy, numTiles)) {
            continue;
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            if (clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            if (clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                 /*axis=*/ShapeRef())) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
            if (clusteredOp.isOperationSplitOverBatchCompatible(/*outputShape=*/ShapeRef())) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
            if (mlir::isa<VPU::SoftMaxOp, VPU::DepthToSpaceOp, VPU::PadOp, VPU::MVN1NormalizeOp, VPU::SwishOp,
                          VPU::MultiplyOp, VPU::SelectOp, VPU::DynamicDequantizeOp, VPU::DynamicQuantizeOp,
                          VPU::GreaterEqualOp, VPU::DeformableConvolutionOp, VPU::MaximumOp,
                          VPU::ScatterElementsUpdateOp, VPU::SubtractOp, VPU::AddOp, VPU::SquaredDifferenceOp,
                          VPU::EqualOp>(clusteredOp.getOperation()) &&
                clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                /*axis=*/ShapeRef())) {
                return strategy;
            }
        }
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering, numTiles)) {
        return VPU::MultiClusterStrategy::Clustering;
    }

    return std::nullopt;
}

bool vpux::VPU::isStrategyCompatibleShape(VPU::ClusteredOpInterface clusteredOp, const vpux::TileInfo& outputTile,
                                          VPU::MultiClusterStrategy strategy, Logger log) {
    auto shape = ShapeRef(outputTile.shape);

    if (shape.size() != RANK_REQUIRED_FOR_TILING && shape.size() != DimsGroups5D::Act::numDims) {
        log.trace("Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3} or {4}.",
                  clusteredOp->getName(), clusteredOp->getLoc(), shape.size(), RANK_REQUIRED_FOR_TILING,
                  DimsGroups5D::Act::numDims);
        return false;
    }
    switch (strategy) {
    case MultiClusterStrategy::SplitOverHeight:
    case MultiClusterStrategy::SplitOverHeightOverlapped:
    case MultiClusterStrategy::HKSwitch: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile);
    }
    case MultiClusterStrategy::SplitOverHeightKernel: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverWidth: {
        return clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverKernel: {
        return clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverHeightWidth: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverBatch: {
        return clusteredOp.isOperationSplitOverBatchCompatible(outputTile.shape);
    }
    case MultiClusterStrategy::Clustering: {
        return true;
    }
    case MultiClusterStrategy::SplitOverGroup: {
        return clusteredOp.isOperationSplitOverGroupCompatible(outputTile);
    }
    default: {
        VPUX_THROW("Unknown multi cluster strategy {0}", strategy);
    }
    }
}

SmallVector<uint32_t> vpux::VPU::getDPUCostForNCEOpPreSplit(
        VPU::NCEOpInterface nceOp, const OutputTiling& outTiles, const VPUIP::WorkloadCostParams& costParams,
        VPUNN::VPUTilingStrategy vpunnTilingStrategy, const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
        int64_t numDPU, Logger log) {
    auto& cache = VPU::getGlobalOpTilingCache();
    std::optional<llvm::hash_code> opHash = std::nullopt;
    const auto useCache = cache.isCacheSupported();
    if (useCache) {
        auto mcStrategyAttr = VPU::MultiClusterStrategyAttr::get(nceOp->getContext(), costParams.layerStrategy);
        opHash = cache.calculateOpHash(nceOp.getOperation(), std::nullopt, outTiles, mcStrategyAttr);
        auto cachedCost = cache.getOpDpuCost(opHash.value());
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
    }

    std::vector<std::vector<VPUNN::DPULayer>> preSplitVPUNNLayers;

    // E#160175 For not supported SEP layer costs - optimize activation spills
    if (VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
        return SmallVector<uint32_t>(outTiles.size(), 1);
    }

    if (!outTiles.empty()) {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());
        const auto tilingVPUNNLayer =
                [&](const VPUIP::WorkloadCostParams& basicCostParams) -> std::vector<std::vector<VPUNN::DPULayer>> {
            std::vector<std::vector<VPUNN::DPULayer>> tiledVpunnLayers;
            tiledVpunnLayers.reserve(outTiles.size());
            for (auto& outTile : outTiles) {
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                auto curCostParams = basicCostParams;
                curCostParams.inputShape = inputTile.shape;
                curCostParams.outputShape = outTile.shape;
                if (inPad.has_value()) {
                    curCostParams.padInfo = {
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right),
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom)};
                }
                tiledVpunnLayers.push_back(VPU::getPerClusterDPULayers(nceOp, curCostParams, log));
            }
            return tiledVpunnLayers;
        };
        preSplitVPUNNLayers = tilingVPUNNLayer(costParams);
    } else {
        preSplitVPUNNLayers = {VPU::getPerClusterDPULayers(nceOp, costParams, log)};
    }

    // Exclude multiClusterStrategy from hash code for VPUNN statistic
    // to make sure that one operation's hash code won't be changed by temporal multiClusterStrategy attribute
    const auto opHashWoStrategy = VPU::getGlobalOpTilingCache().calculateOpHashIncludingTilingExcludingAttr(
            nceOp, multiClusterStrategy, std::nullopt, outTiles);
    auto getVPUNNLayersCostFromCache = [&](const std::vector<VPUNN::DPULayer>& vpunnLayers,
                                           VPUNN::LayerSplitInfo& layerSplitInfo) {
        if (!useCache) {
            return checkAndReturnCost(
                    vpunnCostModel->LayersPreSplit(vpunnLayers, numDPU, /*input_in_ddr=*/false,
                                                   /*output_in_ddr=*/false, /*prefetching=*/true, layerSplitInfo,
                                                   static_cast<size_t>(opHashWoStrategy), vpunnTilingStrategy),
                    log);
        }
        auto hash = cache.calculateVPUNNLayersHash(vpunnLayers);
        auto cachedCost = cache.getVPUNNLayerCost(hash);
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
        auto cost = checkAndReturnCost(
                vpunnCostModel->LayersPreSplit(vpunnLayers, numDPU, /*input_in_ddr=*/false,
                                               /*output_in_ddr=*/false, /*prefetching=*/true, layerSplitInfo,
                                               static_cast<size_t>(opHashWoStrategy), vpunnTilingStrategy),
                log);
        // Track E#158789, cache layerSplitInfo
        cache.updateVPUNNLayerCost(hash, cost);
        return cost;
    };

    SmallVector<uint32_t> layerDPUCosts;
    for (auto& vpunnLayers : preSplitVPUNNLayers) {
        VPUNN::LayerSplitInfo layerSplitInfo;
        auto cost = getVPUNNLayersCostFromCache(vpunnLayers, layerSplitInfo);
        log.debug("Called L2 API with hash {0} strategy {1}", static_cast<size_t>(opHashWoStrategy),
                  stringifyVPUNNStrategy(vpunnTilingStrategy));
        if (cost >= VPU::INVALID_COST_BASE) {
            log.warning("Cost is invalid");
            printVPUNNLayers(vpunnLayers, log);
        }
        layerDPUCosts.push_back(cost);
    }
    if (useCache) {
        cache.updateOpDPUCost(opHash.value(), layerDPUCosts);
    }
    return layerDPUCosts;
}

namespace {

SmallVector<uint32_t> collectLayerShaveCosts(
        VPU::SWOpInterface swOp, std::vector<std::vector<VPUNN::SHAVEWorkload>>& preSplitVPUNNLayers,
        const llvm::DenseMap<int, SmallVector<vpux::NDTypeInterface>>& inputNDTypesMap,
        const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel, int64_t numSHV, Logger log) {
    log.trace("Pre-split of op {0} VPUNN layers size {1}", swOp->getName(), preSplitVPUNNLayers.size());

    SmallVector<uint32_t> layerShaveCosts;
    for (size_t index = 0; index < preSplitVPUNNLayers.size(); ++index) {
        auto& vpunnLayers = preSplitVPUNNLayers[index];
        auto cost = checkAndReturnCost(
                vpunnCostModel->LayersPreSplit(vpunnLayers, numSHV, /*input_in_ddr=*/false, /*output_in_ddr=*/false),
                log);
        auto it = inputNDTypesMap.find(index);
        ArrayRef<vpux::NDTypeInterface> tiledInputTypes = it != inputNDTypesMap.end()
                                                                  ? ArrayRef<vpux::NDTypeInterface>(it->second)
                                                                  : ArrayRef<vpux::NDTypeInterface>();
        cost = vpux::VPU::correctSwOpCost(swOp, tiledInputTypes, cost);
        if (cost >= VPU::INVALID_COST_BASE) {
            log.warning("Cost is invalid");
            printVPUNNLayers(vpunnLayers, log);
        }
        layerShaveCosts.push_back(cost);
    }
    return layerShaveCosts;
}

}  // namespace

SmallVector<uint32_t> vpux::VPU::getSHAVECostForSwOpPreSplit(
        VPU::SWOpInterface swOp, std::optional<VPU::MultiClusterStrategy> strategy, const OutputTiling& outTiles,
        const VPUIP::ShaveWorkloadCostParams& costParams,
        const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel, int64_t numSHV, Logger log) {
    std::vector<std::vector<VPUNN::SHAVEWorkload>> preSplitVPUNNLayers;
    llvm::DenseMap<int, SmallVector<vpux::NDTypeInterface>> inputNDTypesMap;

    if (!outTiles.empty()) {
        preSplitVPUNNLayers.reserve(outTiles.size());
        for (const auto& tileIndex : irange(outTiles.size())) {
            const auto& outTile = outTiles[tileIndex];
            auto curCostParams = costParams;
            curCostParams.inputShapes.clear();
            curCostParams.outputShapes.clear();

            const auto& curTileTypes = VPU::getTileDistributions(swOp.getOperation(), outTile, strategy);
            for (const auto& tileTypeIdx : irange(curTileTypes.size())) {
                const auto& curTileType = curTileTypes[tileTypeIdx];
                if (tileTypeIdx < curTileTypes.size() - 1) {
                    curCostParams.inputShapes.push_back(curTileType.first.getShape().raw());
                } else {
                    curCostParams.outputShapes.push_back(curTileType.first.getShape().raw());
                }
                inputNDTypesMap[tileIndex].push_back(curTileType.first);
            }

            preSplitVPUNNLayers.push_back(VPU::getPerClusterShaveWorkloads(swOp, curCostParams, log, curTileTypes));
        }
    } else {
        preSplitVPUNNLayers = {VPU::getPerClusterShaveWorkloads(swOp, costParams, log)};
    }

    return collectLayerShaveCosts(swOp, preSplitVPUNNLayers, inputNDTypesMap, vpunnCostModel, numSHV, log);
}

SmallVector<uint32_t> vpux::VPU::getSHAVECostForSwOpPreSplit(
        VPU::SWOpInterface swOp, const OutputTiling& outTiles, const VPUIP::ShaveWorkloadCostParams& costParams,
        const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel, int64_t numSHV, Logger log) {
    std::vector<std::vector<VPUNN::SHAVEWorkload>> preSplitVPUNNLayers;
    llvm::DenseMap<int, SmallVector<vpux::NDTypeInterface>> inputNDTypesMap;
    if (!outTiles.empty()) {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "SW op {0} at {1} should be a tiling op", swOp->getName(),
                        swOp.getLoc());
        const auto tilingVPUNNLayer = [&](const VPUIP::ShaveWorkloadCostParams& basicCostParams)
                -> std::vector<std::vector<VPUNN::SHAVEWorkload>> {
            std::vector<std::vector<VPUNN::SHAVEWorkload>> tiledVpunnLayers;
            tiledVpunnLayers.reserve(outTiles.size());
            for (size_t outTileIdx = 0; outTileIdx < outTiles.size(); ++outTileIdx) {
                auto& outTile = outTiles[outTileIdx];
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                auto& inputTiles = inTiles.tiles;

                if (mlir::isa<VPU::MemPermuteOp>(swOp.getOperation())) {
                    SmallVector<vpux::NDTypeInterface> inputNDTypes;
                    inputNDTypes.reserve(inputTiles.size());
                    for (size_t typeIndex = 0; typeIndex < inputTiles.size(); ++typeIndex) {
                        inputNDTypes.push_back(
                                mlir::cast<vpux::NDTypeInterface>(swOp->getOperand(typeIndex).getType())
                                        .extractDenseTile(inputTiles[typeIndex].offsets, inputTiles[typeIndex].shape));
                    }
                    inputNDTypesMap[outTileIdx] = std::move(inputNDTypes);
                }

                auto curCostParams = basicCostParams;
                curCostParams.inputShapes.clear();
                curCostParams.outputShapes.clear();

                for (auto inTile : inputTiles) {
                    curCostParams.inputShapes.push_back(inTile.shape);
                }

                // Handle multiple outputs - extract shape for each output result
                // For devices up to 40XX, preserve old behavior: only push first output shape
                auto arch = config::getArch(swOp.getOperation());
                if (arch <= config::ArchKind::NPU40XX) {
                    curCostParams.outputShapes.push_back(outTile.shape);
                } else {
                    for (size_t outIdx = 0; outIdx < swOp->getNumResults(); ++outIdx) {
                        curCostParams.outputShapes.push_back(outTile.shape);
                    }
                }

                tiledVpunnLayers.push_back(VPU::getPerClusterShaveWorkloads(swOp, curCostParams, log));
            }
            return tiledVpunnLayers;
        };
        preSplitVPUNNLayers = tilingVPUNNLayer(costParams);
    } else {
        preSplitVPUNNLayers = {VPU::getPerClusterShaveWorkloads(swOp, costParams, log)};
    }

    return collectLayerShaveCosts(swOp, preSplitVPUNNLayers, inputNDTypesMap, vpunnCostModel, numSHV, log);
}

SmallVector<uint32_t> vpux::VPU::getDPUCostForNCEOp(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy mcStrategy,
                                                    const OutputTiling& outTiles,
                                                    const VPUIP::WorkloadCostParams& costParams,
                                                    VPUNN::VPULayerStrategy vpunnStrategy,
                                                    const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                    Logger log) {
    // E#160175 Apply workaround only for VPUX4XXX architecture
    if (costParams.arch == config::ArchKind::NPU40XX && VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
        return SmallVector<uint32_t>(outTiles.size(), 1);
    }

    auto& cache = VPU::getGlobalOpTilingCache();
    std::optional<llvm::hash_code> opHash = std::nullopt;
    const auto useCache = cache.isCacheSupported();
    if (useCache) {
        auto mcStrategyAttr = VPU::MultiClusterStrategyAttr::get(nceOp->getContext(), mcStrategy);
        opHash = cache.calculateOpHash(nceOp.getOperation(), std::nullopt, outTiles, mcStrategyAttr);
        auto cachedCost = cache.getOpDpuCost(opHash.value());
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
    }

    // Cost model does not support Grouped MatMul: group dim is stripped and cost is scaled by groups-per-cluster
    const auto outputShape = getShape(nceOp->getResult(0));
    const bool hasGroupDim = outputShape.size() == 5;
    auto vpunnCostParams = costParams;
    if (hasGroupDim) {
        vpunnCostParams.fullInputShape = VPU::stripGroupDim(costParams.fullInputShape);
        vpunnCostParams.inputShape = VPU::stripGroupDim(costParams.inputShape);
        vpunnCostParams.outputShape = VPU::stripGroupDim(costParams.outputShape);
        vpunnCostParams.inOrder = DimsOrder::NHWC;
        vpunnCostParams.outOrder = DimsOrder::NHWC;
    }
    std::vector<VPUNN::DPULayer> vpunnLayers;
    SmallVector<int64_t> groupScales;
    if (outTiles.empty()) {
        vpunnLayers.push_back(VPU::getDPULayer(vpunnCostParams));
        if (hasGroupDim) {
            groupScales.push_back(divUp(outputShape[DimsGroups5D::Act::G], vpunnCostParams.numTiles));
        }
    } else {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());
        const auto tilingVPUNNLayer =
                [&](const VPUIP::WorkloadCostParams& basicCostParams) -> std::vector<VPUNN::DPULayer> {
            std::vector<VPUNN::DPULayer> vpunnLayers;
            vpunnLayers.reserve(outTiles.size());
            for (auto& outTile : outTiles) {
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                auto curCostParams = basicCostParams;
                curCostParams.inputShape = inputTile.shape;
                curCostParams.outputShape = outTile.shape;
                if (hasGroupDim) {
                    curCostParams.inputShape = VPU::stripGroupDim(inputTile.shape);
                    curCostParams.outputShape = VPU::stripGroupDim(outTile.shape);
                    groupScales.push_back(divUp(outTile.shape[DimsGroups5D::Act::G], curCostParams.numTiles));
                }
                if (inPad.has_value()) {
                    curCostParams.padInfo = {
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right),
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom)};
                }
                if (VPU::isNCEWithSEPActivation(nceOp.getOperation())) {
                    auto input = nceOp->getOperand(0);
                    auto inputSparseTensorOp = input.getDefiningOp<VPU::GroupSparseTensorOp>();
                    auto tilingViewOp = mlir::cast<VPU::TilingViewLikeOpInterface>(inputSparseTensorOp.getOperation());
                    auto sepInTiles = tilingViewOp.backInferTileInfo(inputTile, log);
                    auto sepDataTile = sepInTiles.tiles.front();
                    auto sepTableTile = sepInTiles.tiles.back();

                    // Check if sparsity map is present
                    bool hasSparseMap = inputSparseTensorOp.getSparsityMap() != nullptr;

                    curCostParams.sepInfo = VPUIP::SEPInfo{sepTableTile.shape, sepDataTile.shape, hasSparseMap};
                }
                vpunnLayers.push_back(VPU::getDPULayer(curCostParams));
            }
            return vpunnLayers;
        };
        vpunnLayers = tilingVPUNNLayer(vpunnCostParams);
    }

    auto getVPUNNLayerCostFromCache = [&](VPUNN::DPULayer& vpunnLayer) {
        if (!useCache) {
            return checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
        }
        auto hash = cache.calculateVPUNNLayerHash(vpunnLayer, vpunnStrategy);
        auto cachedCost = cache.getVPUNNLayerCost(hash);
        if (cachedCost.has_value()) {
            return cachedCost.value();
        }
        auto cost = checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
        cache.updateVPUNNLayerCost(hash, cost);
        return cost;
    };

    SmallVector<uint32_t> layerDPUCosts;
    for (auto [i, vpunnLayer] : llvm::enumerate(vpunnLayers)) {
        auto cost = getVPUNNLayerCostFromCache(vpunnLayer);
        if (cost >= VPU::INVALID_COST_BASE) {
            printVPUNNLayerConfig(vpunnLayer, vpunnStrategy, log);
            // A workaround to resolve new introduced INPUT_TOO_BIG error code after vpunn software update
            // We should set correct SOK & SOK_NO_BROADCAST when real nn model is updated in future
            if ((cost == VPU::ERROR_INPUT_TOO_BIG) &&
                (vpunnStrategy.tiling_strategy == VPUNN::VPUTilingStrategy::SOK)) {
                auto clusteredOpNceOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
                auto outputType = mlir::cast<NDTypeInterface>(clusteredOpNceOp->getResult(0).getType());
                auto distributionMode = getOutputTensorDistributionMode(clusteredOpNceOp, mcStrategy, outputType);
                if (distributionMode == DistributionMode::SEGMENTED) {
                    vpunnStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK_NO_BROADCAST;
                    cost = checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
                    vpunnStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;
                }
            }

            if (cost == VPU::ERROR_INPUT_TOO_BIG && !layerDPUCosts.empty()) {
                log.trace(" Use the first availabe layer cost to estimate the layer with ERROR_INPUT_TOO_BIG");
                cost = layerDPUCosts.front();
            } else if (cost >= VPU::INVALID_COST_BASE) {
                layerDPUCosts.clear();
                break;
            } else {
                cost = cost * SOK_NO_BROADCAST_DPU_COST_RATIO;
                log.trace(" Use SOK_NO_BROADCAST to estimate the layer with ERROR_INPUT_TOO_BIG, get cost {0}", cost);
            }
        }

        if (mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation()) &&
            (mcStrategy == VPU::MultiClusterStrategy::Clustering)) {
            // The VPUNN cost of NCEEltwiseOp is inaccurate
            // Multiply a ratio to correct the cost
            // Track [E#98656]
            log.trace("Using NCEELTWISE_DPU_COST_RATIO for DPU cost");
            cost *= NCEELTWISE_DPU_COST_RATIO;
        }

        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        const auto arch = config::getArch(clusteredOp);
        if (mlir::isa<VPU::NCEConvolutionOp>(nceOp.getOperation()) && arch == config::ArchKind::NPU40XX) {
            if (mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel && outTiles.size() > 1) {
                auto nTiles = vpunnStrategy.nTiles;
                auto outShape = mlir::cast<vpux::NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                        .getShape();
                auto outputChannelsDim = outShape[Dims4D::Act::C];
                auto perTileChannels = static_cast<double>(outputChannelsDim) / nTiles;
                auto maxSpatialDim = std::max(outShape[Dims4D::Act::H], outShape[Dims4D::Act::W]);
                if (perTileChannels / static_cast<double>(maxSpatialDim) < NCECONV_DPU_SOK_OC_TO_SPATIAL_RATIO) {
                    cost *= NCECONV_DPU_SOK_COST_RATIO;
                }
            }
        }
        if (mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp.getOperation())) {
            auto nTiles = vpunnStrategy.nTiles;
            if ((mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel) && (!config::isArchVPUX3XXX(arch))) {
                auto modeIn = VPU::getActivationTensorDistributionMode(clusteredOp, mcStrategy);
                auto modeOut = VPU::getOutputTensorDistributionMode(clusteredOp, mcStrategy, nullptr);

                // DUP -> DUP case
                if ((VPU::bitEnumContainsAny(modeIn, VPU::DistributionMode::DUPLICATED) ||
                     VPU::bitEnumContainsAny(modeIn, VPU::DistributionMode::MULTICASTED)) &&
                    ((VPU::bitEnumContainsAny(modeOut, VPU::DistributionMode::DUPLICATED) ||
                      VPU::bitEnumContainsAny(modeOut, VPU::DistributionMode::MULTICASTED)))) {
                    VPUX_THROW_WHEN(nTiles <= 0, "nTiles should be positive but got {0}", nTiles);
                    auto outputChannels =
                            mlir::cast<vpux::NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                    .getShape()[Dims4D::Act::C];
                    size_t perTileCh = outputChannels / nTiles;
                    if (perTileCh <= 32) {
                        // The VPUNN cost of NCEDWCONV is inaccurate
                        // Multiply a ratio to correct the cost
                        // Track [E#117314]
                        log.trace("Using NCEDWCONV_DPU_COST_RATIO for SOK DPU cost");
                        cost *= NCEDWCONV_DPU_COST_RATIO;
                    }
                }
            } else if (mcStrategy == VPU::MultiClusterStrategy::HKSwitch) {
                auto outputHeight =
                        mlir::cast<vpux::NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                .getShape()[Dims4D::Act::H];
                size_t perTileH = outputHeight / nTiles;
                if (perTileH < 6) {
                    // The VPUNN cost of NCEDWCONV is inaccurate
                    // Multiply a ratio to correct the cost
                    // Track [E#144661]
                    log.trace("Using NCEDWCONV_HK_DPU_COST_RATIO for HK DPU cost");
                    cost *= NCEDWCONV_HK_DPU_COST_RATIO;
                }
            }
        }

        if ((mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceOp->getOperand(0).getType()) ||
             mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceOp->getResult(0).getType())) &&
            (mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel)) {
            // op with SEP activation should not use this ratio
            if (!VPU::isNCEWithSEPActivation(nceOp.getOperation()) && !config::isArchVPUX3XXX(arch)) {
                // The VPUNN cost of ACT-SPARSITY is inaccurate
                // Multiply a ratio to correct the cost
                // Track [E#117195]
                log.trace("Using ACTSPARSE_DPU_COST_RATIO for DPU cost");
                cost *= ACTSPARSE_DPU_COST_RATIO;
            }
        }

        if (hasGroupDim) {
            cost = checked_cast<uint32_t>(cost * groupScales[i]);
        }
        layerDPUCosts.push_back(cost);
    }
    if (useCache) {
        cache.updateOpDPUCost(opHash.value(), layerDPUCosts);
    }
    return layerDPUCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileWeightsDMACosts(
        VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
        ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc) {
    auto weightsOperand = nceOp.getWeightsOperand();
    if (weightsOperand == nullptr) {
        return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
    }

    const auto inferredTileTypes =
            std::vector<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>>{getTileDistributions(
                    nceOp.getOperation(), siblingsAnalysis, TileInfo(getBoundedShape(nceOp->getResult(0))), strategy)};
    ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> inferredTileTypesRef{inferredTileTypes};
    const auto typesList = tilesTypes.empty() ? inferredTileTypesRef : tilesTypes;

    SmallVector<uint32_t> perTileWeightsCosts;
    for (const auto& tileTypes : typesList) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto weightsDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1].first, tileTypes[1].second));
        perTileWeightsCosts.push_back(weightsDMACost);
    }

    return perTileWeightsCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileActivationDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc,
        VPU::MultiClusterStrategy strategy, int64_t numTiles) {
    bool permuteCastInMiddle = false;
    auto getParentOp = [&]() -> mlir::Operation* {
        mlir::Operation* parentOp = nceOp->getOperand(0).getDefiningOp();
        while (parentOp && isPureViewOp(parentOp)) {
            // PermuteCast op will change MC axis so that we assume there must be a spill between parent sw op and
            // current nce op. And activation dma cost should be calculated. More general solution need
            // backInferTilingDim interface implementation for permuteCast op. See E#106960
            if (mlir::isa<VPU::PermuteCastOp>(parentOp)) {
                permuteCastInMiddle = true;
            }
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }
        return parentOp;
    };
    auto parentOp = getParentOp();

    // If nceOp can fit into CMX and it has a parent op, we assume that act spilling can be avoided by adjusting
    // strategy of the parent op, hence we return a per tile activation DMA cost of zero
    if ((parentOp != nullptr) && (tilesTypes.size() == 1)) {
        if (!mlir::isa<VPU::SWOpInterface>(parentOp)) {
            return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
        }

        if (!isSoftwareOpCustomStrategyIncompatibleWithOtherNceOp(parentOp, strategy, numTiles)) {
            if (!permuteCastInMiddle) {
                return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
            }
        }
    }

    bool isEltwiseOpWithDiffInputs =
            (mlir::isa<VPU::NCEEltwiseOp>(nceOp) && nceOp->getOperand(0) != nceOp->getOperand(1));

    SmallVector<uint32_t> perTileActCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto actDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[0].first, tileTypes[0].second));
        if (isEltwiseOpWithDiffInputs) {
            actDMACost += checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1].first, tileTypes[1].second));
        }
        perTileActCosts.push_back(actDMACost);
    }

    return perTileActCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileOutputDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingWriteCostFunc) {
    SmallVector<uint32_t> perTileOutputCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected > 1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto outputDMACost =
                checked_cast<uint32_t>(getSpillingWriteCostFunc(tileTypes.back().first, tileTypes.back().second));
        perTileOutputCosts.push_back(outputDMACost);
    }

    return perTileOutputCosts;
}

std::pair<uint32_t, uint32_t> vpux::VPU::getWeightsDMACostForNCEOp(VPU::NCEOpInterface nceOp,
                                                                   const OutputTiling& outTiles,
                                                                   SmallVector<uint32_t>& layerDPUCosts,
                                                                   ArrayRef<uint32_t> layerDMACosts,
                                                                   bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;
    auto tilingStrategy = tiles.front().axis;
    const bool is5D = tilingStrategy.size() == DimsGroups5D::Act::numDims;
    const auto isWeightsSharedNestedTiling = isWeightsFirstNestedTiling(nceOp.getOperation(), tilingStrategy);
    SmallVector<uint32_t> filteredDMACosts;
    SmallVector<uint32_t> filteredDPUCosts;
    if (isWeightsSharedNestedTiling) {
        log.trace("[Cost Analysis] Assumption: Weights First nested tiling");
        // Unroll channel first
        // weights are partially shared. Every tile_H * tile_W weights are shared
        int64_t temporalSize;
        if (is5D) {
            temporalSize = tilingStrategy[DimsGroups5D::Act::G] * tilingStrategy[DimsGroups5D::Act::C];
        } else {
            temporalSize = tilingStrategy[Dims4D::Act::C];
        }

        for (int64_t i = 0; i < temporalSize; i++) {
            filteredDPUCosts.push_back(layerDPUCosts[i]);
            filteredDMACosts.push_back(layerDMACosts[i]);
        }
    } else {
        filteredDMACosts = SmallVector<uint32_t>(layerDMACosts);
        filteredDPUCosts = layerDPUCosts;
    }

    auto weightsOperand = nceOp.getWeightsOperand();
    const auto isWeightsTileOnGroup = is5D && tilingStrategy[DimsGroups5D::Act::G] > 1;
    const auto isWeightsTileOnChannel =
            is5D ? tilingStrategy[DimsGroups5D::Act::C] > 1 : tilingStrategy[Dims4D::Act::C] > 1;
    bool isWeightsDMASplitOnEachTile = (weightsOperand != nullptr && (isWeightsTileOnChannel || isWeightsTileOnGroup));
    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // If the DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;
    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Weights DMA from second tile on will be overlapped with DPU of previous tile
        log.trace("[Cost Analysis] Assumption: Weights DMA will pipeline with DPU");
        totalDMACost += getPrefetchDMACostOverlappsWithPreviousDPU(filteredDPUCosts, ArrayRef(filteredDMACosts),
                                                                   isWeightsDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If weights DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If weights DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isWeightsDMASplitOnEachTile
                                ? std::accumulate(filteredDMACosts.begin(), filteredDMACosts.end(), 0U)
                                : filteredDMACosts.front();
    }

    return std::make_pair(totalDMACost, totalDMACost - filteredDMACosts.front());
}

uint32_t vpux::VPU::getActivationDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                                 SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                                 bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto tilingStrategy = tiles.front().axis;
    [[maybe_unused]] const auto isActSharedNestedTiling =
            isSpatialFirstNestedTiling(nceOp.getOperation(), tilingStrategy);
    SmallVector<uint32_t> filteredDMACosts;
    SmallVector<uint32_t> filteredDPUCosts;
    auto useFilteredCosts = false;
    const auto temporalSize = tilingStrategy[Dims4D::Act::C];
    // Consider activation sharing in cost calculation for nested tiling, reduces total DMA
    // Only apply to 50XX, as only arch with accurate DMA cost model
    if (config::getArch(nceOp) == config::ArchKind::NPU50XX && isActSharedNestedTiling) {
        useFilteredCosts = true;
        // Unroll channel first
        // Activation DMAs are partially shared. Every tile_C activation are shared
        for (size_t i = 0; i < tiles.size(); i += temporalSize) {
            filteredDPUCosts.push_back(layerDPUCosts[i]);
            filteredDMACosts.push_back(layerDMACosts[i]);
        }
    }
    if (!useFilteredCosts) {
        filteredDMACosts = SmallVector<uint32_t>(layerDMACosts);
        filteredDPUCosts = layerDPUCosts;
    }

    auto weightsOperand = nceOp.getWeightsOperand();
    // If the activation needs to be split:
    //      no weights operand - like Eltwise
    //      tiling on spatial dimension
    //      or DepthConvolution, the activation input is always split for tile, and the DMAs should be accumulated
    const auto nonOneDims = getNonOneDim(tiles.front().axis);
    const auto actSplit = llvm::any_of(nonOneDims, [](Dim tileDim) {
        return tileDim != Dims4D::Act::C;
    });
    bool isActDMASplitOnEachTile =
            (weightsOperand == nullptr || actSplit || mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp.getOperation()));

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Act DMA from second tile on will be overlapped with DPU of previous tile
        log.trace("[Cost Analysis] Assumption: Activation DMA will pipeline with DPU");
        totalDMACost += getPrefetchDMACostOverlappsWithPreviousDPU(filteredDPUCosts, ArrayRef(filteredDMACosts),
                                                                   isActDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If act DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If act DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isActDMASplitOnEachTile ? std::accumulate(filteredDMACosts.begin(), filteredDMACosts.end(), 0U)
                                                : filteredDMACosts.front();
    }

    // overlapped part should be removed from DPU cost list
    // because it can't be overlapped with other DMA costs anymore
    if (!useFilteredCosts) {
        layerDPUCosts = std::move(filteredDPUCosts);
    } else {
        for (size_t i = 0; i < tiles.size(); i += temporalSize) {
            layerDPUCosts[i] = filteredDPUCosts[i / temporalSize];
        }
    }

    return totalDMACost;
}

uint32_t vpux::VPU::getOutputDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                             SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                             bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA of the current tile will overlap with DPU of the next tile
    nceOp->setAttr(outputPipelining, mlir::BoolAttr::get(nceOp->getContext(), true));
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;
    nceOp->removeAttr(outputPipelining);

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Output DMA expect for the last tile will be overlapped with DPU of the next tile
        log.trace("[Cost Analysis] Assumption: Output DMA will pipeline with DPU");
        totalDMACost += getOutputDMACostOverlappsWithNextDPU(layerDPUCosts, layerDMACosts, true);
    } else {
        totalDMACost += std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U);
    }

    return totalDMACost;
}

size_t vpux::VPU::getNumNonConstantOperands(mlir::Operation* op) {
    return std::count_if(op->operand_begin(), op->operand_end(), [](mlir::Value operand) {
        return !mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp());
    });
}

bool vpux::VPU::hasLayerWithMultipleInputs(mlir::Operation* op) {
    return std::any_of(op->user_begin(), op->user_end(), [](mlir::Operation* user) {
        return getNumNonConstantOperands(user) > 1 || hasLayerWithMultipleInputs(user);
    });
}

bool vpux::VPU::isSingleBatchRequired(mlir::Operation* op) {
    return !mlir::isa<VPU::MVNOp, VPU::MVN1NormalizeOp, VPU::MVN1SumOp, VPU::LSTMSequenceOp, VPU::DequantizeOp,
                      VPU::ReverseOp, VPU::ReverseSequenceOp, VPU::GridSampleOp, VPU::DeformableConvolutionOp,
                      VPU::MemPermuteOp>(op);
}

bool vpux::VPU::setSOKForRuntimeDequantConvolution(VPU::NCEOpInterface nceOp, LayerCostModel& costModel) {
    // change to SOK if depends on SOK dequantize and does not fit into CMX
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (!clusteredOp) {
        return false;
    }
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    if (!clusteredOp.isOperationSplitOverKernelCompatible(outputType.getShape(), ShapeRef(), ShapeRef())) {
        return false;
    }
    // If Tiling is not possible with SOK , cost model returns COST_MAX, avoid assigning SOK when tiling cant be
    // done. Heuristic should be improved by considering SOH + DMA cost vs SOK when costs are more reliable :
    // E#163827
    auto SOKCost = costModel.getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
    if (SOKCost == costModel.COST_MAX) {
        return false;
    }

    // check if weights fit into CMX (force SOK only for large weights)
    auto numClusters = VPU::getOptimalNumClusters(nceOp.getOperation(), outputType.getShape(),
                                                  VPU::MultiClusterStrategy::SplitOverKernel);
    auto weightType = nceOp.getWeightsOperand().getType();
    auto filterSize = VPU::getTotalAllocSizeWithDistribution(
                              weightType, getFilterDistributionAttrFromOp(nceOp, weightType, numClusters,
                                                                          VPU::MultiClusterStrategy::SplitOverKernel))
                              .count();
    auto totalAvailableCMXSize = VPU::getTotalCMXSize(nceOp.getOperation()).count();

    if (filterSize <= totalAvailableCMXSize) {
        return false;
    }

    clusteredOp.setMultiClusterStrategy(MultiClusterStrategy::SplitOverKernel);
    return true;
}

bool vpux::VPU::alignStrategyWithParentRuntimeDequant(VPU::ClusteredOpInterface clusteredOp,
                                                      LayerCostModel& costModel) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());
    if (!nceOp) {
        return false;
    }
    auto weights = nceOp.getWeightsOperand();
    if (!weights) {
        return false;
    }
    auto dequantizeOp = weights.getDefiningOp<VPU::DequantizeOp>();
    if (!dequantizeOp) {
        return false;
    }
    auto opStrategy = clusteredOp.getMultiClusterStrategy();
    auto dequantizeOpStrategy = dequantizeOp.getMultiClusterStrategy();

    if (!opStrategy.has_value() || !dequantizeOpStrategy.has_value()) {
        return false;
    }
    if (opStrategy != MultiClusterStrategy::SplitOverHeight && opStrategy != MultiClusterStrategy::SplitOverWidth &&
        opStrategy != MultiClusterStrategy::SplitOverHeightOverlapped && opStrategy != MultiClusterStrategy::HKSwitch) {
        return false;
    }
    if (dequantizeOpStrategy != MultiClusterStrategy::SplitOverKernel) {
        return false;
    }

    return setSOKForRuntimeDequantConvolution(nceOp, costModel);
}

bool vpux::VPU::isNCEOpUsedAsConvScale(mlir::Operation* op) {
    if (!mlir::isa<VPU::NCEMaxPoolOp, VPU::NCEEltwiseOp>(op)) {
        return false;
    }

    auto isConvWeightTableScaleConsumer = [](mlir::Operation* user, mlir::Value value) {
        auto convOp = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(user);
        if (convOp == nullptr) {
            return false;
        }

        auto convStrategy = convOp.getMultiClusterStrategy();
        return convOp.getWeightTableScale() == value && convStrategy.has_value();
    };

    auto output = op->getResult(0);
    while (output.hasOneUse()) {
        auto user = *output.getUsers().begin();
        if (isConvWeightTableScaleConsumer(user, output)) {
            return true;
        }

        if (!mlir::isa<VPU::ViewLikeOpInterface>(user)) {
            break;
        }

        output = user->getResult(0);
    }

    if (output.use_empty()) {
        return false;
    }

    return llvm::all_of(output.getUsers(), [&](mlir::Operation* user) {
        auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
        if (sliceOp == nullptr) {
            return false;
        }

        auto sliceOutput = sliceOp.getResult();
        if (sliceOutput.use_empty()) {
            return false;
        }

        return llvm::all_of(sliceOutput.getUsers(), [&](mlir::Operation* sliceUser) {
            return isConvWeightTableScaleConsumer(sliceUser, sliceOutput);
        });
    });
}

std::optional<VPU::MultiClusterStrategy> vpux::VPU::getMultiClusterStrategyFromOp(mlir::Operation* op) {
    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        return clusteredOp.getMultiClusterStrategy();
    }
    return std::nullopt;
}
