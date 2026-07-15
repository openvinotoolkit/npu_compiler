//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <vpu_layer_cost_model.h>

using namespace vpux;
using namespace VPU;

namespace {
bool isTiledOnLowestDim(ShapeRef tileAxis, const DimsOrder& dimOrder) {
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    const auto axis = tileAxis[lowestDim];
    return axis > 1;
}

}  // namespace

//
// CostModelConfig
//

std::shared_ptr<VPUNN::VPULayerCostModel> CostModelConfig::createLayerCostModel(mlir::MLIRContext* context) {
    const auto& costModelFactory = getCostModelFactory(context);
    return costModelFactory.createLayerCostModel();
}

std::shared_ptr<VPUNN::VPULayerCostModel> CostModelConfig::createLayerCostModel(mlir::Operation* op) {
    return createLayerCostModel(op->getContext());
}

//
// LayerCostModelAnalysis
//

vpux::VPU::LayerCostModelAnalysis::LayerCostModelAnalysis(mlir::ModuleOp module) {
    _layerCostModel = VPU::CostModelConfig::createLayerCostModel(module);
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
        std::optional<std::reference_wrapper<vpux::VPU::LayerCostModelAnalysis>> analysis, mlir::MLIRContext* context,
        Logger log) {
    if (analysis.has_value()) {
        log.trace("Load preserved layer cost model");
        return analysis.value().get().getVPUNNLayerCostModel();
    }
    log.warning("Create new layer cost model instance");
    return VPU::CostModelConfig::createLayerCostModel(context);
}

// For SW ops like VPU.MemPermuteOp, there are usually multiple possible implementations - optimized and non-optimized
// versions. The cost for the non-optimized version usually is much larger than the other version, and in these cases
// the VPUNN cost is not accurate. The non-optimized cost needs to be corrected (increased) by multiplying it by this
// factor.
//
// E#171770: We should remove this factor when the cost model is updated to support the non-optimized version of
// MemPermute shave kernel.
//
// E#171772: This cost factor will vary by arch and layer type, so we need different factor per arch and layer type. The
// implementation should be refactored after cost model refactoring is done [E#166371].

constexpr int64_t SW_COST_CORRECTION_FACTOR_FOR_MEM_PERMUTE = 10;
constexpr int64_t DEPTH_TO_SPACE_MIN_OPTIMIZED_WIDTH = 128;

// E#154343: Inaccurate VPUNN cost. For some small SW ops, the cost model usually underestimates the actual cost. This
// experimental value is set as the minimum cost for those ops
constexpr StrategyCost SMALL_SW_COST_THRESHOLD = 12000;

StrategyCost vpux::VPU::correctSwOpCost(VPU::SWOpInterface swOp, ArrayRef<vpux::NDTypeInterface> tiledInputTypes,
                                        StrategyCost cost) {
    if (auto memPermute = mlir::dyn_cast<VPU::MemPermuteOp>(swOp.getOperation())) {
        // Currently only MemPermuteOp needs cost correction
        auto inputType = mlir::cast<NDTypeInterface>(memPermute.getInput().getType());
        auto outputType = mlir::cast<NDTypeInterface>(memPermute.getOutput().getType());
        if (VPUIP::satisfiesOptimizedMemPermute(config::getArch(swOp.getOperation()), inputType, outputType)) {
            VPUX_THROW_WHEN(tiledInputTypes.empty(), "Cannot get tiled input");
            auto tiledInputType = tiledInputTypes.front();
            auto tiledOutputType = tiledInputType.changeDimsOrder(outputType.getDimsOrder());
            if (!VPUIP::satisfiesOptimizedMemPermute(config::getArch(swOp.getOperation()), tiledInputType,
                                                     tiledOutputType)) {
                cost *= SW_COST_CORRECTION_FACTOR_FOR_MEM_PERMUTE;
            }
        }
    } else if (mlir::isa<VPU::MultiplyOp, VPU::SoftMaxOp, VPU::ConvertOp>(swOp.getOperation())) {
        // E#154343: Inaccurate VPUNN cost for those ops when the size is small. So we set a minimum cost for them.
        cost = std::max(SMALL_SW_COST_THRESHOLD, cost);
    } else if (auto d2s = mlir::dyn_cast<VPU::DepthToSpaceOp>(swOp.getOperation())) {
        if (VPU::satisfiesOptimizedDepthToSpace(d2s, config::getArch(swOp.getOperation()))) {
            auto inShape = tiledInputTypes.front().getShape();
            auto origInShape = getShape(d2s.getInput());
            if (origInShape[Dims4D::Act::W] > DEPTH_TO_SPACE_MIN_OPTIMIZED_WIDTH &&
                inShape[Dims4D::Act::W] < DEPTH_TO_SPACE_MIN_OPTIMIZED_WIDTH) {
                cost *= origInShape[Dims4D::Act::W] / DEPTH_TO_SPACE_MIN_OPTIMIZED_WIDTH;
            }
        }
    }
    return cost;
}

MultiClusterStrategySetter::MultiClusterStrategySetter(mlir::Operation* operation, VPU::MultiClusterStrategy strategy)
        : _operation(operation) {
    setTemporaryStrategy(strategy);
}

MultiClusterStrategySetter::~MultiClusterStrategySetter() {
    removeTemporaryStrategy();
}

void MultiClusterStrategySetter::removeTemporaryStrategy() {
    if (!_isStrategyChanged) {
        return;
    }
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
        _isStrategyChanged = !_origStrategy.has_value() || _origStrategy.value() != tempStrategy;
        if (_isStrategyChanged) {
            clusterOp.setMultiClusterStrategy(tempStrategy);
        }
    }
}

LayerVPUNNCost::LayerVPUNNCost(mlir::func::FuncOp func, std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModel,
                               Logger log)
        : _vpunnCostModel(std::move(layerCostModel)), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    _arch = config::getArch(module);
    auto vpuDev = getVPUDeviceType(module);
    auto tileOp = config::getTileExecutor(module);
    auto dpuExec = tileOp.getSubExecutor(config::ExecutorKind::DPU);
    _numTiles = tileOp.getCount();
    _numDPUs = dpuExec.getCount();
    _vpuDevice = vpuDev;
    _numShaveActs = 0;
    _numDMAPorts = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN).getCount();
    if (auto shaveActExec = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT)) {
        _numShaveActs = shaveActExec.getCount();
    }
};

LayerVPUNNCost::LayerVPUNNCost(mlir::func::FuncOp func, Logger log)
        : LayerVPUNNCost(func, VPU::CostModelConfig::createLayerCostModel(func), log) {};

void LayerVPUNNCost::resetNNCacheCounter() {
    _vpunnCostModel->getDPUPreloadedCacheCounter().reset();
}

void LayerVPUNNCost::printNNCacheStatistics() const {
    _log.info("[NN Cache statistics]  {0}", _vpunnCostModel->getDPUPreloadedCacheCounter().printString());
}

LayerVPUNNCost::LayerVPUNNCost(LayerVPUNNCost&& other) noexcept
        : _arch(other._arch),
          _numTiles(other._numTiles),
          _numDPUs(other._numDPUs),
          _numShaveActs(other._numShaveActs),
          _numDMAPorts(other._numDMAPorts),
          _vpuDevice(other._vpuDevice),
          _vpunnCostModel(std::move(other._vpunnCostModel)),
          _log(std::move(other._log)) {
}

StrategyCost LayerVPUNNCost::getStrategyCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const {
    if (mlir::isa<VPU::NCEPermuteOp>(operation)) {
        return getSimpleLayerCost(mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()), parameters);
    } else if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation)) {
        return getNCELayerCost(nceOp, parameters);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(operation)) {
        return getSWLayerCost(swOp, parameters);
    } else if (VPU::isPureViewOp(operation)) {
        return 0.0;
    } else {
        _log.trace("Unsupported op type {0} at {1}", operation->getName(), operation->getLoc());
        return getSimpleLayerCost(mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()), parameters);
    }
}

StrategyCost LayerVPUNNCost::getSpillingTypeCost(vpux::NDTypeInterface type,
                                                 const std::optional<ShapeRef>& tileAxis) const {
    StrategyCost cost =
            getDMACost(type, _vpuDevice, _vpunnCostModel->get_TheoreticalDMA_cost_model_shared(), _numDMAPorts);

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

    /* Check if the op has single slice like user op. If the slice like op will slice on the inner most dimension, which
    may cause the stride DMA inefficient,in this case, we need to compare the number of continuous bytes on the lowest
    dimension and compares it with a predefined hreshold. If the continuous bytes are less than the threshold, the cost
    need to be adjusted accordingly.
               Op
               |
          SliceLikeOp
               |
            Other op
    Ticket to apply this correct for other related DMC cost calculation: E#167797
    */

    auto hasInefficientSliceLikeSingleUserOp = [&]() -> mlir::FailureOr<double> {
        mlir::Operation* userOp = nullptr;
        if (auto vfOp = operation->getParentOfType<VPU::VerticalFusionOp>()) {
            auto innerOps = vfOp.getBody()->without_terminator();
            auto lastOp = &(*std::prev(innerOps.end()));
            if (vfOp->hasOneUse() && lastOp == operation) {
                userOp = *vfOp->user_begin();
            }
        } else if (operation->hasOneUse()) {
            userOp = *operation->user_begin();
        }
        if (userOp == nullptr || !VPU::isPureViewOp(userOp)) {
            return mlir::failure();
        }
        auto inShape = getShape(userOp->getOperand(0));
        auto outShape = getShape(userOp->getResult(0));
        if (inShape.totalSize() <= outShape.totalSize()) {
            return mlir::failure();
        }

        auto dimOrder = outputType.getDimsOrder();
        auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
        const Bit elemSize = outputType.getElemTypeSize();
        auto continousBytes = outShape[lowestDim] * elemSize.count();

        auto strideDMACorrectionThreshold = getStrideDMACorrectionThresholdByArch(_arch);
        if (continousBytes >= strideDMACorrectionThreshold) {
            return mlir::failure();
        }
        return checked_cast<double>(strideDMACorrectionThreshold) / continousBytes;
    };

    auto factor = hasInefficientSliceLikeSingleUserOp();

    const auto tiling =
            parameters._tiling.empty() ? OutputTiling({TileInfo(outputType.getShape())}) : parameters._tiling;
    spillWriteCosts.reserve(tiling.size());
    for (const auto& tileInfo : tiling) {
        auto tiledType = getOutputType(tileInfo);
        auto cost = getSpillingTypeCost(tiledType, tiling[0].axis);
        if (mlir::succeeded(factor)) {
            cost = checked_cast<uint32_t>(std::floor(factor.value() * cost));
        }
        spillWriteCosts.emplace_back(cost);
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

    if (getOperandType == nullptr) {
        getOperandType = [&](const auto& tileInfo) {
            auto tiling = getTileTypes(operation, tileInfo, parameters._strategy);
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

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs, _numTiles, parameters._strategy);
    // Set prefetching to be true to ignore the DMA cost and only get the execution DPU cost
    // According to the VPUNN API definition,
    //      when prefetching is false, the returned cost is the sum of DPU + weights DMA
    //      when prefetching is true, the returned cost is just DPU because it considers the weights are prefetched
    auto distributionMode = DistributionMode::NONE;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp != nullptr) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
        distributionMode = getOutputTensorDistributionMode(clusteredOp, costParams.layerStrategy, outputType);
    }
    const auto vpunnStrategy = VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numTiles, _arch, _numShaveActs,
                                                        true, distributionMode, nceOp);
    SmallVector<uint32_t> vpunnLayerDPUCosts;
    const auto enableVPUNNPreSplit = config::hasVPUNNPreSplit(nceOp);
    const auto isMatMul = mlir::isa<VPU::NCEMatMulOp>(nceOp.getOperation());
    if (enableVPUNNPreSplit && !isActSparseOp(nceOp) && !isMatMul) {
        // Track E#160972. Activation sparse op accuracy issue
        vpunnLayerDPUCosts = getDPUCostForNCEOpPreSplit(nceOp, parameters._tiling, costParams,
                                                        vpunnStrategy.tiling_strategy, _vpunnCostModel, _numDPUs, _log);
    } else {
        vpunnLayerDPUCosts = getDPUCostForNCEOp(nceOp, parameters._strategy, parameters._tiling, costParams,
                                                vpunnStrategy, _vpunnCostModel, _log);
    }
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
        tilesTypes.push_back(
                getTileDistributions(nceOp.getOperation(), siblingsAnalysis, outTile, parameters._strategy, inTiles));
    }

    // Add extra weights DMA costs
    const auto getSpillingReadCost = [&](NDTypeInterface srcType,
                                         const TensorDistributionMap& distributions) -> uint32_t {
        auto distributedType = getDistributedTypeFromDistributionMap(srcType, distributions);
        return checked_cast<uint32_t>(getDMACost(
                distributedType, _vpuDevice, _vpunnCostModel->get_TheoreticalDMA_cost_model_shared(), _numDMAPorts));
    };
    auto vpunnLayerWeightsCosts =
            getPerTileWeightsDMACosts(nceOp, parameters._strategy, siblingsAnalysis, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    auto [cost, costWithPrefetching] = getWeightsDMACostForNCEOp(nceOp, parameters._tiling, vpunnLayerDPUCosts,
                                                                 vpunnLayerWeightsCosts, isPrefetchTilingEnabled, _log);
    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = parameters._tiling.empty() ? OutputTiling({TileInfo(outShape)}) : parameters._tiling;
    if (isPrefetchTilingEnabled && tilingInfoOp != nullptr &&
        tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PREFETCHING, _log)) {
        vpunnCost += costWithPrefetching;
    } else {
        vpunnCost += cost;
    }

    _log.trace("VPUNN total layer cost {0}", vpunnCost);
    return vpunnCost;
}

StrategyCost LayerVPUNNCost::getSWLayerCost(VPU::SWOpInterface swOp, const VPUNNCostParameters& parameters) const {
    auto swKernelName = swOp->getName().stripDialect().str();
    auto outputTiling = parameters._tiling;
    const auto& shaveUtilInterface = VPU::getShaveCostModelUtils(swOp->getContext());
    auto isShave2APIUsed = shaveUtilInterface.isShave2ApiUsed();

    // Handle all outputs
    SmallVector<vpux::NDTypeInterface> outputTypes;
    // For devices up to 40XX, preserve old behavior: only process first output (index 0)
    if (_arch <= config::ArchKind::NPU40XX) {
        outputTypes.push_back(mlir::cast<vpux::NDTypeInterface>(swOp->getResult(0).getType()));
    } else {
        for (auto result : swOp->getResults()) {
            outputTypes.push_back(mlir::cast<vpux::NDTypeInterface>(result.getType()));
        }
    }

    if (outputTiling.empty()) {
        outputTiling.push_back(TileInfo(outputTypes[0].getShape()));
    }

    StrategyCost fullCost = 0;
    _log.trace("Op {0} with strategy {1} has {2} output tiles", swOp->getLoc(), parameters._strategy,
               outputTiling.size());
    if (shaveUtilInterface.isSwKernelOpSupported(swKernelName) && isShave2APIUsed) {
        _log.trace("SW kernel {0} is supported, calculating cost using VPUNN cost model", swKernelName);
        const auto swCostParam = getShaveWorkloadCostParam(swOp, _arch, _numShaveActs, _numTiles);
        SmallVector<uint32_t> vpunnLayerSWCosts;
        vpunnLayerSWCosts =
                getSHAVECostForSwOpPreSplit(swOp, outputTiling, swCostParam, _vpunnCostModel, _numShaveActs, _log);
        for (const auto& cost : vpunnLayerSWCosts) {
            fullCost += cost;
        }
    } else {
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
                inputNDTypes.push_back(
                        mlir::cast<vpux::NDTypeInterface>(swOp->getOperand(typeIndex).getType())
                                .extractDenseTile(inputTiles[typeIndex].offsets, inputTiles[typeIndex].shape));
            }

            std::optional<VPUNN::SHAVEWorkload> vpunnLayer;
            StrategyCost currentCost = 0;

            auto outputTiledType =
                    outputTypes[0].extractDenseTile(outputTiling[index].offsets, outputTiling[index].shape);
            // For devices up to 40XX, preserve old behavior: single output tile extraction
            if (_arch <= config::ArchKind::NPU40XX) {
                auto workloadPtr = getVPUNNSWKernelOp(swOp, outputTiledType, inputNDTypes);
                if (workloadPtr) {
                    vpunnLayer = std::move(*workloadPtr);
                }
            } else {
                // Extract tiled types for all outputs
                SmallVector<vpux::NDTypeInterface> outputTiledTypes;
                for (auto outputType : outputTypes) {
                    outputTiledTypes.push_back(
                            outputType.extractDenseTile(outputTiling[index].offsets, outputTiling[index].shape));
                }
                auto workloadPtr = getVPUNNSWKernelOp(swOp, outputTiledTypes, inputNDTypes);
                if (workloadPtr) {
                    vpunnLayer = std::move(*workloadPtr);
                }
            }

            if (!vpunnLayer.has_value()) {
                currentCost = getSimpleLayerCost(
                        outputTypes[0].extractDenseTile(outputTiling[index].offsets, outputTiling[index].shape),
                        parameters);

            } else {
                auto distributionMode = DistributionMode::NONE;
                auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation());
                if (clusteredOp != nullptr) {
                    distributionMode =
                            getOutputTensorDistributionMode(clusteredOp, parameters._strategy, outputTiledType);
                }

                auto vpunnStrategy = VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numTiles, _arch,
                                                              _numShaveActs, false, distributionMode, swOp);
                currentCost = _vpunnCostModel->Layer(vpunnLayer.value(), vpunnStrategy);
            }
            fullCost += vpux::VPU::correctSwOpCost(swOp, inputNDTypes, currentCost);
        }
    }
    _log.trace("VPUNN SW layer costs {0}", fullCost);
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
        type = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    Bit continuousBytesOnLowestDim;
    if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(type)) {
        continuousBytesOnLowestDim = distributedType.getLargestCompactShape()[lowestDim] * elemSize;
    } else {
        continuousBytesOnLowestDim = type.getShape()[lowestDim] * elemSize;
    }

    auto strideDMACorrectionThreshold = getStrideDMACorrectionThresholdByArch(_arch);
    if (continuousBytesOnLowestDim.count() < strideDMACorrectionThreshold) {
        auto factor = checked_cast<double>(strideDMACorrectionThreshold) / continuousBytesOnLowestDim.count();
        return checked_cast<uint32_t>(std::floor(factor * cost));
    }
    return cost;
}
