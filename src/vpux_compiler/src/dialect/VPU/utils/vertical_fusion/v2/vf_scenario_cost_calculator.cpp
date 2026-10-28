//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_scenario_cost_calculator.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/hash_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_hash.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/SmallPtrSet.h>

#include <cmath>
#include <numeric>

// Experimental number for the VF internal slice copy cost ratio
static constexpr double VF_INTERNAL_SLICE_DMA_COST_RATIO = 0.6;

namespace {
struct OpIndexWithCost {
    size_t tileIdx;
    size_t opIdx;
    std::optional<vpux::VPU::StrategyCost> cost;
};
}  // namespace

namespace vpux::VPU::VF::v2 {
namespace {
ParentVFTilingInfo& getParentVFTilingInfo(mlir::Operation* parentOp, ParentVFTilingInfoCache& cache) {
    auto cachedInfo = llvm::find_if(cache, [&](const auto& info) {
        return info.parentOp == parentOp;
    });
    if (cachedInfo != cache.end()) {
        return *cachedInfo;
    }

    auto parentCache = std::make_unique<VFCacheAnalysis>(parentOp);
    std::unique_ptr<VFConfig> parentConfig;
    mlir::ArrayAttr tilingStrategyAttr;
    if (auto parentVFOp = mlir::dyn_cast<VPU::VerticalFusionOp>(parentOp)) {
        parentConfig = std::make_unique<VFConfig>(parentVFOp, *parentCache);
        tilingStrategyAttr = parentVFOp.getTilingStrategy();
    } else {
        // scf path
        llvm::SetVector<mlir::Operation*> parentOps;
        parentOps.insert(parentOp);
        parentConfig = std::make_unique<VFConfig>(parentOps, *parentCache);
        tilingStrategyAttr = parentOp->getAttrOfType<mlir::ArrayAttr>(tilingStrategy);
    }

    VPUX_THROW_WHEN(tilingStrategyAttr == nullptr, "Cannot get tiling strategy for parent {0}", *parentOp);
    auto tilingInfo = std::make_unique<TilingOperationStorage>();
    auto tilingDims = parseIntArrayAttr<int64_t>(tilingStrategyAttr);
    auto tilingStorage = calculateTilingRegions(*parentConfig, tilingDims, Logger::global(), tilingInfo);
    VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles", *parentOp,
                    tilingDims);

    cache.emplace_back(parentOp, std::move(parentCache), std::move(parentConfig), std::move(tilingInfo));
    return cache.back();
}

mlir::Value getVFOperand(mlir::Operation* operation, int64_t operandIdx, VFConfig& config) {
    if (config.getSubgraph() == nullptr) {
        return operation->getOperand(operandIdx);
    }

    auto argument = getVFBlockArgument(operation->getOperand(operandIdx));
    if (argument == nullptr) {
        return nullptr;
    }

    return config.getSubgraph()->getOperand(argument.getArgNumber());
}

mlir::Operation* getParentVFOperation(mlir::Value operand, VFConfig& config) {
    if (operand == nullptr) {
        return nullptr;
    }

    if (config.getSubgraph() == nullptr) {
        return operand.getDefiningOp();
    }

    return operand.getDefiningOp<VPU::VerticalFusionOp>().getOperation();
}

/**
 * @brief Determines if a given operation has a prefetched DMA for a specific block argument.
 *  Differences from v1 implementation:
 * - For elementwise operations with more than one operand, the function checks if the current parent operation
 *   is scheduled before another parent operation in the block. It also calculates memory requirements to ensure
 *   they fit within the available CMX memory.
 */
bool hasPrefetchedDMA(mlir::Operation* operation, int64_t operandIdx, VFConfig& config,
                      const VPUNNCostParameters& parameters, const bool isInput,
                      ParentVFTilingInfoCache& parentVFTilingInfoCache) {
    if (operation->hasTrait<VPU::EltwiseOp>() && operation->getNumOperands() > 1) {
        auto* curParentOp = getParentVFOperation(getVFOperand(operation, operandIdx, config), config);
        auto otherOperandIdx = operandIdx == 0 ? 1 : 0;
        if (auto otherParentOperand = getVFOperand(operation, otherOperandIdx, config)) {
            auto* otherParentOp = getParentVFOperation(otherParentOperand, config);
            if (curParentOp != nullptr && otherParentOp != nullptr && curParentOp->isBeforeInBlock(otherParentOp)) {
                auto operandSize = config.getOperationTypes(operation, parameters._tiling[0],
                                                            parameters._operandsTiling[0])[operandIdx]
                                           .getTotalAllocSize();

                auto& parentInfo = getParentVFTilingInfo(otherParentOp, parentVFTilingInfoCache);
                auto parentOutputOps = parentInfo.config->getOutputs();
                VPUX_THROW_WHEN(parentOutputOps.empty(), "There are no output operations for parent {0}",
                                *otherParentOp);
                auto lastOp = parentOutputOps.back();
                auto inputOutputTiling = parentInfo.tilingInfo->getRef(lastOp, 0);
                const auto& inputOutputTilingValue = inputOutputTiling.value().get();
                auto lastTileSize = parentInfo.config->getOperationRequiredCMX(lastOp, inputOutputTilingValue.second,
                                                                               inputOutputTilingValue.first.tiles);
                return operandSize + lastTileSize > getTotalCMXFragmentationAwareSize(operation);
            }
        }
        return true;
    }

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);

    const auto operand = operation->getOperand(operandIdx);
    if (nceOp != nullptr) {
        if (config.getSubgraph() != nullptr) {
            const auto blockArgument = getVFBlockArgument(operand);
            if (blockArgument != getVFBlockArgument(nceOp.getWeightsOperand()) &&
                blockArgument != getVFBlockArgument(nceOp->getOperand(0))) {
                return false;
            }
        } else if (operand != nceOp.getWeightsOperand() && operand != nceOp->getOperand(0)) {
            return false;
        }
    }

    if (isInput) {
        return true;
    }

    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        return false;
    }

    return config.getSubgraph() != nullptr
                   ? getVFBlockArgument(operand) == getVFBlockArgument(nceOp.getWeightsOperand())
                   : operand == nceOp.getWeightsOperand();
}

}  // namespace

llvm::hash_code VFScenarioCostCalculator::getStrategyCostHash(mlir::Operation* operation,
                                                              const VPUNNCostParameters& parameters) {
    auto hash = llvm::hash_combine(VPU::hashOperationForTiling(operation), parameters._strategy, parameters._mode,
                                   parameters._withDMAs, parameters._tiling.size(), parameters._operandsTiling.size());
    const auto hashTileShapeOnly = !mlir::isa<VPU::NCEOpInterface>(operation) ||
                                   VPU::hasZeroPadding(mlir::cast<VPU::NCEOpInterface>(operation).getPad());

    for (const auto& tile : parameters._tiling) {
        hash = llvm::hash_combine(hash, getTileHash(tile, hashTileShapeOnly));
    }
    for (const auto& operandTiling : parameters._operandsTiling) {
        hash = llvm::hash_combine(hash, operandTiling.size());
        for (const auto& tile : operandTiling) {
            hash = llvm::hash_combine(hash, getTileHash(tile, hashTileShapeOnly));
        }
    }

    return hash;
}

VPUNNCostParameters VFScenarioCostCalculator::fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                                              const SmallVector<TileInfo>& inputTiles) {
    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
        mcStrategy = mcOperation.getMultiClusterStrategy().value_or(mcStrategy);
    }

    auto mode = TilingMode::ISOLATED;
    SmallVector<OutputTiling> inputAllTiles;
    if (!inputTiles.empty()) {
        inputAllTiles.push_back(inputTiles);
    }
    return VPUNNCostParameters(mcStrategy, tiling, mode, inputAllTiles, false);
}

VPUNNCostParameters VFScenarioCostCalculator::fillInCostParam(mlir::Operation* operation,
                                                              const TilingOperationStorage::UPtr& opStorage,
                                                              size_t index) {
    auto inputOutputTiling = opStorage->getRef(operation, index);

    OutputTiling outputTiling;
    SmallVector<TileInfo> inputTiling;

    if (inputOutputTiling.has_value()) {
        const auto& inputOutputTilingValue = inputOutputTiling.value().get();
        outputTiling = {inputOutputTilingValue.second};
        inputTiling = inputOutputTilingValue.first.tiles;
    }

    return fillInCostParam(operation, outputTiling, inputTiling);
}

StrategyCost VFScenarioCostCalculator::getStrategyCost(VFConfig& config, mlir::Operation* operation,
                                                       const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                       const VPUNNCostParameters& parameters) {
    const auto hash = getStrategyCostHash(operation, parameters);
    auto cachedCost = config.getCachedStrategyCost(hash);
    if (cachedCost.has_value()) {
        return cachedCost.value();
    }

    auto cost = costFunction->getStrategyCost(operation, parameters);
    config.cacheStrategyCost(hash, cost);
    return cost;
}

StrategyCost VFScenarioCostCalculator::getParentCost(mlir::Operation* operation,
                                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost) {
    StrategyCost parentCost = 0;
    auto* parent = findParent(operation->getOperand(0));
    if (parent == nullptr && (operation->getNumOperands() > 1 && operation->hasTrait<VPU::EltwiseOp>())) {
        parent = findParent(operation->getOperand(1));
    }
    if (parent != nullptr) {
        auto foundCost = isolatedOperCost.find(parent);
        VPUX_THROW_WHEN(foundCost == isolatedOperCost.end(), "Cannot find the cost for {0}", *parent);
        parentCost = foundCost->second;
    }

    return parentCost;
}

std::optional<StrategyCost> VFScenarioCostCalculator::getViewLikeOpDMACost(
        mlir::Operation* operation, VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo, size_t tileIdx,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) {
    auto blockArg = getVFBlockArgument(operation->getOperand(0));
    if (blockArg != nullptr) {
        return std::nullopt;
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType());
    // Two kinds of view-like operations may change the total allocation size.
    // 1. For view-like ops (e.g. Slice op), there is DMA involved in the tiled case which should be taken into account
    // when scheduling.
    // 2. For GroupSparseTensor, VPUNN cost is not accurate for SEP ops; use DMA spilling cost as a penalty
    if (inType.getTotalAllocSize() == outType.getTotalAllocSize()) {
        return std::nullopt;
    }

    // View op can't get distributed type directly. Use the clustered consumer to calculate the DMA cost.
    auto userToOperand = findFirstNonViewUser(operation);
    if (!userToOperand.has_value()) {
        return std::nullopt;
    }
    auto user = userToOperand->first;
    auto operandIdx = userToOperand->second;

    auto userCostParameters = fillInCostParam(user, tilingInfo, tileIdx);
    const auto& userTileTypes =
            config.getOperationTypes(user, userCostParameters._tiling[0], userCostParameters._operandsTiling[0]);
    VPUX_THROW_WHEN(operandIdx >= static_cast<int64_t>(userTileTypes.size()), "Invalid operand index {0} for user {1}",
                    operandIdx, user->getLoc());
    auto userTileType = userTileTypes[operandIdx];
    return costFunction->getSpillingTypeCost(userTileType, DmaDirectionType::DDR_TO_CMX,
                                             userCostParameters._tiling[0].axis);
}

StrategyCost VFScenarioCostCalculator::getPrefetchingCost(const VFScheduling& scheduling, mlir::Operation* operation,
                                                          VFConfig& config, TypeCostFunction typeCostFunction,
                                                          const VPUNNCostParameters& parameters, const bool isInput,
                                                          const TilingOperationStorage::UPtr& tilingInfo,
                                                          const int64_t index,
                                                          ParentVFTilingInfoCache& parentVFTilingInfoCache) {
    StrategyCost prefetchedCost = 0;
    auto inputTiling = tilingInfo->getRef(operation, index);
    if (!inputTiling.has_value()) {
        return prefetchedCost;
    }

    const auto sharedWeightsEnabled = scheduling.isSharedWeightsSupported(config);
    const auto& inputTilingValue = inputTiling.value().get();
    auto tileAxis = inputTilingValue.second.axis;
    if (tileAxis.size() == DimsGroups5D::Act::numDims) {
        tileAxis[DimsGroups5D::Act::C] = 1;
        tileAxis[DimsGroups5D::Act::G] = 1;
    } else {
        tileAxis[Dims4D::Act::C] = 1;
    }

    auto tileSizeOnSpatialDims = tileAxis.totalSize();

    auto isWeightsSharedWithPreviousTile =
            index == 0 ? false : index / tileSizeOnSpatialDims == (index - 1) / tileSizeOnSpatialDims;

    ArrayRef<NDTypeInterface> operationTypes;
    for (auto input : operation->getOperands() | indexed) {
        if (input.index() >= inputTilingValue.first.tiles.size()) {
            break;
        }
        auto inputOperand = input.value();
        const auto& inTile = inputTilingValue.first.tiles[input.index()];
        auto isAlreadyShared = isWeightsSharedWithPreviousTile && sharedWeightsEnabled &&
                               isOperandSharedWeightsForTiling(operation, inputOperand, inTile);
        if (isAlreadyShared) {
            continue;
        }

        if (auto blockArg = getVFBlockArgument(inputOperand)) {
            if (hasPrefetchedDMA(operation, input.index(), config, parameters, isInput, parentVFTilingInfoCache)) {
                if (operationTypes.empty()) {
                    operationTypes =
                            config.getOperationTypes(operation, parameters._tiling[0], parameters._operandsTiling[0]);
                }
                prefetchedCost += typeCostFunction(operationTypes[input.index()],
                                                   parameters._operandsTiling[0][input.index()].axis);
            }
        }
    }

    return prefetchedCost;
}

/*
   For the pattern below,
    VF{
        Conv1[Kernel = [1, 1]]
         |  \
         |   Conv2 [Kernel = [3, 3]]
         \       /
          Eltwise
    }
   there will be slice inserted between Conv1 and Eltwise after VF is unrolled. And the slice will be further converted
   into related DMA ops. So the scheduler need to take the slice copy cost into account.


*/
StrategyCost VFScenarioCostCalculator::getInternalSliceCopyCost(
        mlir::Operation* op, VFConfig& config, const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
        const VPUNNCostParameters& parameters, const bool isInput, const TilingOperationStorage::UPtr& tilingInfo,
        const int64_t index) {
    const auto eltwiseLikeOp =
            op->getNumOperands() > 1 && op->hasTrait<VPU::EltwiseOp>() && op->getOperand(0) != op->getOperand(1);
    StrategyCost cost = 0;
    if (!eltwiseLikeOp || isInput) {
        return cost;
    }

    SmallVector<mlir::Operation*> parents;
    parents.push_back(findParent(op->getOperand(0)));
    parents.push_back(findParent(op->getOperand(1)));

    auto* parentLeft = parents.front();
    auto* parentRight = parents.back();

    const auto& ops = config.getOperationsForTiling();
    auto allParentsAreInVF = llvm::all_of(parents, [&](auto* parent) {
        return llvm::find(ops, parent) != ops.end();
    });

    if (!allParentsAreInVF || parentLeft == parentRight) {
        return cost;
    }

    auto* earliestParent = parentLeft->isBeforeInBlock(parentRight) ? parentLeft : parentRight;
    auto* closestParent = earliestParent == parentLeft ? parentRight : parentLeft;

    // Only NCE-with-weights parents drive the slice-cost heuristic; SW eltwise ops (Elu,
    // HSigmoid, Erf, ...) are now valid VF parents and must be skipped.
    SmallVector<mlir::Operation*> chain;
    const auto isNceWithWeights = [](mlir::Operation* candidate) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(candidate);
        return nceOp != nullptr && nceOp.getWeightsOperand() != nullptr;
    };
    const auto isParentOperation = [&]() {
        if (isNceWithWeights(closestParent)) {
            chain.emplace_back(closestParent);
        }
        auto curOp = closestParent;
        while ((curOp = findParent(curOp->getOperand(0))) != nullptr) {
            if (curOp == earliestParent) {
                return true;
            }
            if (isNceWithWeights(curOp)) {
                chain.emplace_back(curOp);
            }
        }
        return false;
    };
    if (!isParentOperation()) {
        return cost;
    }

    auto origOutShape = getShape(op->getResult(0));
    auto opTilingInfo = tilingInfo->getRef(op, index);
    const auto& opTiling = opTilingInfo.value().get();
    const auto isSliceOpNeeded = [&]() {
        auto tileOnH = opTiling.second.shape[Dims4D::Act::H] != origOutShape[Dims4D::Act::H];
        auto tileOnW = opTiling.second.shape[Dims4D::Act::W] != origOutShape[Dims4D::Act::W];
        if (!tileOnH && !tileOnW) {
            return false;
        }

        return llvm::any_of(chain, [&](auto* parent) {
            auto nceOp = mlir::cast<VPU::NCEOpInterface>(parent);
            if (auto filter = nceOp.getWeightsOperand()) {
                auto filterShape = getShape(filter);
                return (filterShape[Dims4D::Filter::KY] != 1 && tileOnH) ||
                       (filterShape[Dims4D::Filter::KX] != 1 && tileOnW);
            }
            return false;
        });
    };

    if (isSliceOpNeeded()) {
        const auto& tileTypes = config.getOperationTypes(op, opTiling.second, opTiling.first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "Can not get tiled types for tile of operation {0}", op->getLoc());
        auto type = tileTypes.front();
        auto inputSize = type.getTotalAllocSize();
        auto closestParentOpTilingInfo = tilingInfo->getRef(closestParent, index);
        const auto& closestParentOpTiling = closestParentOpTilingInfo.value().get();
        auto requiredCMXSize = inputSize + config.getOperationRequiredCMX(closestParent, closestParentOpTiling.second,
                                                                          closestParentOpTiling.first.tiles);

        auto sliceCanBeOverlappedWithClosestParent = requiredCMXSize < getTotalCMXFragmentationAwareSize(op);
        if (!sliceCanBeOverlappedWithClosestParent) {
            cost = std::ceil(VF_INTERNAL_SLICE_DMA_COST_RATIO *
                             (costFunction->getSpillingReadCost(op, parameters, op->getOperand(0),
                                                                [&](const auto& tileInfo) {
                                                                    return config.getOperationTypes(op, tileInfo,
                                                                                                    {})[0];
                                                                }) +
                              costFunction->getSpillingWriteCost(op, parameters, [&](const auto& tileInfo) {
                                  return config.getOperationTypes(op, tileInfo, {})[0];
                              })));
        }
    }
    return cost;
}

void VFScenarioCostCalculator::reduceCostWithPrefetchedDMA(StrategyCost& parentCost, const StrategyCost& prefetchCost,
                                                           StrategyCost& accumuatedPrefetchCost) {
    // reduce the cost with dmas prefetched for this tile
    if (accumuatedPrefetchCost <= parentCost) {
        parentCost -= accumuatedPrefetchCost;
    }
    accumuatedPrefetchCost += prefetchCost;
}

StrategyCost PerformanceVFScenarioCostCalculator::calculate(VFScenarioCostCalculationContext& context) const {
    auto intervals = getTimeIntervals(context);
    if (intervals.empty()) {
        return std::numeric_limits<StrategyCost>::max();
    }

    const auto maxInterval = llvm::max_element(intervals, [](const auto& lhs, const auto& rhs) {
        return lhs._mEnd < rhs._mEnd;
    });
    return maxInterval->_mEnd == 0 ? std::numeric_limits<StrategyCost>::max() : maxInterval->_mEnd;
}

SmallVector<TimelineInterval> PerformanceVFScenarioCostCalculator::getTimeIntervals(
        VFScenarioCostCalculationContext& context) const {
    VFLinearContainer linearTimeIntervals;
    SmallVector<StrategyCost> prefetchCostList(context.tilesNumber + 1, 0);
    DenseMap<mlir::Operation*, StrategyCost> isolatedOperCost;
    ParentVFTilingInfoCache parentVFTilingInfoCache;
    StrategyCost fullCost = 0;
    auto sortedOps = SmallVector<mlir::Operation*>(context.config.getVFOperations().begin(),
                                                   context.config.getVFOperations().end());
    llvm::sort(sortedOps, [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    for (auto index : irange(context.tilesNumber)) {
        for (auto item : sortedOps | indexed) {
            auto lastEndTime = fullCost;
            auto* operation = item.value();
            auto parameters = fillInCostParam(operation, context.tilingInfo, index);
            if (parameters._tiling.empty()) {
                context.log.warning("No tiling information for VF op at '{0}'", operation->getLoc());
                linearTimeIntervals.invalidate();
                return linearTimeIntervals.getAllIntervals();
            }

            if (mlir::isa<VPU::TilingViewLikeOpInterface>(operation)) {
                auto copyCost = getViewLikeOpDMACost(operation, context.config, context.tilingInfo, index,
                                                     context.costFunction);
                if (copyCost.has_value()) {
                    linearTimeIntervals.addDMA(operation, index, lastEndTime, copyCost.value());
                    fullCost += copyCost.value();
                }
                continue;
            }

            auto isolatedCost = getStrategyCost(context.config, operation, context.costFunction, parameters);
            if (isolatedCost >= std::numeric_limits<StrategyCost>::max()) {
                linearTimeIntervals.invalidate();
                return linearTimeIntervals.getAllIntervals();
            }
            isolatedOperCost[operation] = isolatedCost;
            fullCost += isolatedCost;

            StrategyCost outputCost = 0;
            StrategyCost correctedOutputCost = 0;
            if (llvm::find(context.config.getOutputs(), operation) != context.config.getOutputs().end() &&
                context.tilesNumber > 1) {
                outputCost = context.costFunction->getSpillingTypeCost(
                        context.config
                                .getOperationTypes(operation, parameters._tiling[0], parameters._operandsTiling[0])
                                .back(),
                        DmaDirectionType::CMX_TO_DDR, parameters._tiling[0].axis);
                if (outputCost >= std::numeric_limits<StrategyCost>::max()) {
                    linearTimeIntervals.invalidate();
                    return linearTimeIntervals.getAllIntervals();
                }
                correctedOutputCost = outputCost;
                correctOutputSpillCost(correctedOutputCost, context.config, isolatedOperCost, prefetchCostList, index,
                                       context.tilesNumber);
                fullCost += correctedOutputCost;
            }

            const bool isInput = llvm::find(context.config.getInputs(), operation) != context.config.getInputs().end();
            auto prefetchedCost = getPrefetchingCost(
                    context.scheduling, operation, context.config,
                    [&](NDTypeInterface type, const std::optional<ShapeRef>& tileAxis) {
                        return context.costFunction->getSpillingTypeCost(type, DmaDirectionType::DDR_TO_CMX, tileAxis);
                    },
                    parameters, isInput, context.tilingInfo, index, parentVFTilingInfoCache);
            auto correctedPrefetchedCost = prefetchedCost;
            if (prefetchedCost >= std::numeric_limits<StrategyCost>::max()) {
                linearTimeIntervals.invalidate();
                return linearTimeIntervals.getAllIntervals();
            }

            if (context.prefetching && prefetchedCost > 0) {
                correctInputPrefetchingCost(correctedPrefetchedCost, operation, context.config, isolatedOperCost,
                                            prefetchCostList, index);
            }
            fullCost += correctedPrefetchedCost;

            auto internalSliceCost = getInternalSliceCopyCost(operation, context.config, context.costFunction,
                                                              parameters, isInput, context.tilingInfo, index);
            fullCost += internalSliceCost;
            if (prefetchedCost > 0) {
                linearTimeIntervals.addDMA(operation, index, lastEndTime - prefetchedCost + correctedPrefetchedCost,
                                           prefetchedCost);
                lastEndTime += correctedPrefetchedCost;
            }
            if (internalSliceCost > 0) {
                linearTimeIntervals.addDMA(operation, index, lastEndTime, internalSliceCost);
                lastEndTime += internalSliceCost;
            }
            linearTimeIntervals.addOperation(operation, index, lastEndTime, isolatedCost);
            if (outputCost > 0) {
                linearTimeIntervals.addDMA(operation, index, lastEndTime + isolatedCost, outputCost);
            }
        }
    }
    return linearTimeIntervals.getAllIntervals();
}

SmallVector<TimelineInterval> PipeliningVFScenarioCostCalculator::getTimeIntervals(
        VFScenarioCostCalculationContext& context) const {
    return getPipelining(context).getAllIntervals();
}

StrategyCost PipeliningVFScenarioCostCalculator::calculate(VFScenarioCostCalculationContext& context) const {
    return getPipelining(context).maxCost();
}

VFPipelineContainer PipeliningVFScenarioCostCalculator::getPipelining(VFScenarioCostCalculationContext& context) const {
    auto operations = context.config.getVFOperations().getArrayRef();
    auto pipelinedStructure = VFPipelineContainer();
    ParentVFTilingInfoCache parentVFTilingInfoCache;
    const size_t totalTiledOpSize = operations.size() * checked_cast<size_t>(context.tilesNumber);
    size_t executedOpSize = 0;
    size_t currentTileIdx = 0;
    size_t pipelinedTileIdx = currentTileIdx + 1;
    size_t currentTileOpIdx = 0;
    size_t pipelinedTileOpIdx = 0;

    SmallVector<config::ExecutorKind> executorKinds;
    auto getNextComputeOp = [&](size_t opIdx) -> config::ExecutorKind {
        for (auto idx = opIdx; idx < operations.size(); ++idx) {
            auto* operation = operations[idx];
            if (mlir::isa<VPU::SWOpInterface>(operation)) {
                return config::ExecutorKind::SHAVE_ACT;
            }
            if (mlir::isa<VPU::NCEOpInterface>(operation)) {
                return config::ExecutorKind::DPU;
            }
        }
        return config::ExecutorKind::UNKNOWN;
    };
    for (auto opIdx : irange(operations.size())) {
        executorKinds.push_back(getNextComputeOp(opIdx));
    }

    auto addOutputSpill = [&](mlir::Operation* operation, int64_t index, const VPUNNCostParameters& costParameters) {
        if (llvm::find(context.config.getOutputs(), operation) == context.config.getOutputs().end()) {
            return;
        }
        VPUX_THROW_WHEN(costParameters._tiling.empty() || costParameters._operandsTiling.empty(),
                        "Empty tiling for operation {0} at {1}", operation->getName(), operation->getLoc());
        const auto& opTypes = context.config.getOperationTypes(operation, costParameters._tiling[0],
                                                               costParameters._operandsTiling[0]);
        VPUX_THROW_WHEN(opTypes.empty(), "getOperationTypes returned empty for operation {0} at {1}",
                        operation->getName(), operation->getLoc());
        auto spillCost = context.costFunction->getSpillingTypeCost(opTypes.back(), DmaDirectionType::CMX_TO_DDR,
                                                                   costParameters._tiling[0].axis);
        pipelinedStructure.addDMA(operation, index, spillCost, true);
    };

    auto getNextOp = [&]() -> OpIndexWithCost {
        if (currentTileOpIdx >= operations.size()) {
            currentTileIdx = pipelinedTileIdx;
            currentTileOpIdx = pipelinedTileOpIdx;
            VPUX_THROW_UNLESS(currentTileOpIdx < operations.size(),
                              "Pipelined index {0} is out of range for operations size {1}", currentTileOpIdx,
                              operations.size());
            pipelinedTileIdx += 1;
            pipelinedTileOpIdx = 0;
        }
        if (currentTileIdx == 0 && currentTileOpIdx == 0) {
            return {currentTileIdx, currentTileOpIdx++, std::nullopt};
        }
        if (pipelinedTileOpIdx < currentTileOpIdx && pipelinedTileIdx < checked_cast<size_t>(context.tilesNumber) &&
            pipelinedTileOpIdx < operations.size()) {
            auto* operation = operations[pipelinedTileOpIdx];
            auto costParameters = fillInCostParam(operation, context.tilingInfo, pipelinedTileIdx);
            auto isolatedCost = getStrategyCost(context.config, operation, context.costFunction, costParameters);
            if (isolatedCost >= VPU::INVALID_COST_BASE) {
                context.log.trace("Op {0} has invalid cost {1}, force to {2}", operation->getLoc(), isolatedCost,
                                  VPU::UNIT_COST);
                isolatedCost = VPU::UNIT_COST;
            }
            const auto isPotentialDMAOp = [&]() {
                if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(operation)) {
                    return swOp.supportLoweringAsDMA();
                }
                return false;
            }();
            const auto lastTile = static_cast<size_t>(pipelinedStructure.getLastIntervalIndex().value_or(0));
            const auto isPipelineBetterCandidate =
                    lastTile == currentTileIdx || executorKinds[currentTileOpIdx] != executorKinds[pipelinedTileOpIdx];
            if (!isPotentialDMAOp && isPipelineBetterCandidate &&
                pipelinedStructure.isPipelineAvailable(pipelinedTileIdx, operation, isolatedCost)) {
                return {pipelinedTileIdx, pipelinedTileOpIdx++, isolatedCost};
            }
        }
        return {currentTileIdx, currentTileOpIdx++, std::nullopt};
    };

    while (executedOpSize < totalTiledOpSize) {
        const auto opIndexWithCost = getNextOp();
        const auto tileIdx = opIndexWithCost.tileIdx;
        const auto opIdx = opIndexWithCost.opIdx;
        auto* operation = operations[opIdx];
        auto costParameters = fillInCostParam(operation, context.tilingInfo, tileIdx);
        auto isolatedCost = opIndexWithCost.cost.has_value()
                                    ? opIndexWithCost.cost.value()
                                    : getStrategyCost(context.config, operation, context.costFunction, costParameters);
        if (mlir::isa<VPU::TilingViewLikeOpInterface>(operation)) {
            auto copyCost =
                    getViewLikeOpDMACost(operation, context.config, context.tilingInfo, tileIdx, context.costFunction);
            if (copyCost.has_value()) {
                pipelinedStructure.addDMA(operation, tileIdx, copyCost.value());
            }
            ++executedOpSize;
            continue;
        }
        if (isolatedCost >= VPU::INVALID_COST_BASE) {
            context.log.trace("Op {0} has invalid cost {1}, force to {2}", operation->getLoc(), isolatedCost,
                              VPU::UNIT_COST);
            isolatedCost = VPU::UNIT_COST;
        }
        const auto isInput = llvm::find(context.config.getInputs(), operation) != context.config.getInputs().end();
        auto prefetchedCost = getPrefetchingCost(
                context.scheduling, operation, context.config,
                [&](NDTypeInterface type, const std::optional<ShapeRef>& tileAxis) {
                    return context.costFunction->getSpillingTypeCost(type, DmaDirectionType::DDR_TO_CMX, tileAxis);
                },
                costParameters, isInput, context.tilingInfo, tileIdx, parentVFTilingInfoCache);
        pipelinedStructure.addDMA(operation, tileIdx, prefetchedCost);
        pipelinedStructure.addOperation(operation, tileIdx, isolatedCost);
        addOutputSpill(operation, tileIdx, costParameters);
        ++executedOpSize;
    }
    return pipelinedStructure;
}

void PerformanceVFScenarioCostCalculator::correctOutputSpillCost(StrategyCost&, VFConfig&,
                                                                 const DenseMap<mlir::Operation*, StrategyCost>&,
                                                                 SmallVector<StrategyCost>&, int64_t, int64_t) const {
}

void PerformanceVFScenarioCostCalculator::correctInputPrefetchingCost(StrategyCost&, mlir::Operation*, VFConfig&,
                                                                      const DenseMap<mlir::Operation*, StrategyCost>&,
                                                                      SmallVector<StrategyCost>&, size_t) const {
}

void MinimalVFScenarioCostCalculator::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, SmallVector<StrategyCost>& prefetchCostList,
        size_t index) const {
    if (llvm::find(config.getInputs(), operation) != config.getInputs().end()) {
        return;
    }

    VPUX_THROW_WHEN(index >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}", index,
                    prefetchCostList.size());
    auto parentCost = getParentCost(operation, isolatedOperCost);
    reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index]);
    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}

void LastOpPrefetchingVFScenarioCostCalculator::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, SmallVector<StrategyCost>& prefetchCostList,
        size_t index) const {
    StrategyCost parentCost = 0;
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();
    VPUX_THROW_WHEN(config.getOutputs().empty(), "Cannot find outputs for VF {0}", config.getSubgraph());
    auto* lastOp = config.getOutputs().back();
    if (isInput) {
        if (index == 0) {
            return;
        }
        auto foundCost = isolatedOperCost.find(lastOp);
        VPUX_THROW_WHEN(foundCost == isolatedOperCost.end(), "Cannot find the cost for {0}", *lastOp);
        parentCost = foundCost->second;
        VPUX_THROW_WHEN(index - 1 >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}",
                        index - 1, prefetchCostList.size());
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index - 1]);
    } else {
        parentCost = getParentCost(operation, isolatedOperCost);
        VPUX_THROW_WHEN(index >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}",
                        index, prefetchCostList.size());
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index]);
    }
    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}

void WeightsPrefetchingVFScenarioCostCalculator::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, SmallVector<StrategyCost>& prefetchCostList,
        size_t index) const {
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();
    StrategyCost parentCost = 0;
    if (isInput) {
        if (index == 0) {
            return;
        }
        parentCost = std::accumulate(isolatedOperCost.begin(), isolatedOperCost.end(), static_cast<StrategyCost>(0),
                                     [](StrategyCost previous, const auto& item) {
                                         return previous + item.second;
                                     });
        VPUX_THROW_WHEN(index - 1 >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}",
                        index - 1, prefetchCostList.size());
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index - 1]);
    } else {
        parentCost = getParentCost(operation, isolatedOperCost);
        VPUX_THROW_WHEN(index >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}",
                        index, prefetchCostList.size());
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index]);
    }
    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}

void FullPrefetchingVFScenarioCostCalculator::correctOutputSpillCost(
        StrategyCost& spillCost, VFConfig& config, const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
        SmallVector<StrategyCost>& prefetchCostList, int64_t index, int64_t tilesNumber) const {
    VPUX_THROW_WHEN(static_cast<size_t>(index + 1) >= prefetchCostList.size(),
                    "Index {0} out of range for prefetchCostList of size {1}", index + 1, prefetchCostList.size());
    prefetchCostList[index + 1] = spillCost;
    StrategyCost nextTileOpCost = 0;
    if (index != tilesNumber - 1) {
        for (auto* input : config.getInputs()) {
            auto foundCost = isolatedOperCost.find(input);
            VPUX_THROW_WHEN(foundCost == isolatedOperCost.end(), "Cannot find the cost for {0}", *input);
            nextTileOpCost += foundCost->second;
        }
    }
    spillCost = nextTileOpCost < spillCost ? spillCost - nextTileOpCost : 0U;
}

void FullPrefetchingVFScenarioCostCalculator::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, SmallVector<StrategyCost>& prefetchCostList,
        size_t index) const {
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();
    StrategyCost parentCost = 0;
    if (isInput) {
        if (index == 0) {
            return;
        }
        parentCost = std::accumulate(isolatedOperCost.begin(), isolatedOperCost.end(), static_cast<StrategyCost>(0),
                                     [](StrategyCost previous, const auto& item) {
                                         return previous + item.second;
                                     });
        VPUX_THROW_WHEN(index - 1 >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}",
                        index - 1, prefetchCostList.size());
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index - 1]);
    } else {
        const auto getPrevOp = [&](mlir::Operation* currentOp) -> mlir::Operation* {
            if (llvm::find(config.getInputs(), currentOp) != config.getInputs().end()) {
                return nullptr;
            }
            auto* parent = findParent(currentOp->getOperand(0));
            if (parent == nullptr && currentOp->getNumOperands() > 1 && operation->hasTrait<VPU::EltwiseOp>()) {
                parent = findParent(currentOp->getOperand(1));
            }
            return parent;
        };
        auto* current = operation;
        while (current != nullptr) {
            parentCost += getParentCost(current, isolatedOperCost);
            current = getPrevOp(current);
        }
        VPUX_THROW_WHEN(index >= prefetchCostList.size(), "Index {0} out of range for prefetchCostList of size {1}",
                        index, prefetchCostList.size());
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, prefetchCostList[index]);
    }
    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}

StrategyCost PowerVFScenarioCostCalculator::calculate(VFScenarioCostCalculationContext& context) const {
    const auto& scheduling = context.scheduling;
    auto& config = context.config;
    StrategyCost powerCost = 0;
    const auto& inputs = config.getInputs();
    const auto& outputs = config.getOutputs();
    llvm::SmallPtrSet<mlir::Operation*, 16> inputSet(inputs.begin(), inputs.end());
    llvm::SmallPtrSet<mlir::Operation*, 16> outputSet(outputs.begin(), outputs.end());
    ParentVFTilingInfoCache parentVFTilingInfoCache;

    SmallVector<mlir::Operation*> sortedOps(config.getVFOperations().begin(), config.getVFOperations().end());
    llvm::sort(sortedOps, [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    for (auto index : irange(context.tilesNumber)) {
        for (auto* operation : sortedOps) {
            if (mlir::isa<VPU::TilingViewLikeOpInterface>(operation)) {
                continue;
            }
            auto costParameters = fillInCostParam(operation, context.tilingInfo, index);
            if (costParameters._tiling.empty()) {
                context.log.warning("No tiling information for VF op at '{0}'", operation->getLoc());
                return std::numeric_limits<StrategyCost>::max();
            }

            const auto isInput = inputSet.contains(operation);
            powerCost += getPrefetchingCost(
                    scheduling, operation, config,
                    [](NDTypeInterface type, const std::optional<ShapeRef>&) {
                        return type.getTotalAllocSize().count();
                    },
                    costParameters, isInput, context.tilingInfo, index, parentVFTilingInfoCache);
            if (outputSet.contains(operation)) {
                powerCost += config.getOperationTypes(operation, costParameters._tiling[0],
                                                      costParameters._operandsTiling[0])
                                     .back()
                                     .getTotalAllocSize()
                                     .count();
            }
        }
    }

    return powerCost;
}

std::unique_ptr<VFScenarioCostCalculator> PerformanceVFScenarioCostCalculatorFactory::create(VFScenario scenario) {
    switch (scenario) {
    case VFScenario::MINIMAL:
        return std::make_unique<MinimalVFScenarioCostCalculator>();
    case VFScenario::LASTOP_PREFETCHING:
        return std::make_unique<LastOpPrefetchingVFScenarioCostCalculator>();
    case VFScenario::WEIGHTS_PREFETCHING:
        return std::make_unique<WeightsPrefetchingVFScenarioCostCalculator>();
    case VFScenario::FULL_PREFETCHING:
        return std::make_unique<FullPrefetchingVFScenarioCostCalculator>();
    case VFScenario::VF_PIPELINING:
        return std::make_unique<PipeliningVFScenarioCostCalculator>();
    default:
        VPUX_THROW("Unsupported VF scenario for performance cost calculation: {0}", scenario);
    }
}

SmallVector<TimelineInterval> getVFTimeIntervals(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                                                 const TilingOperationStorage::UPtr& tilingInfo,
                                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log,
                                                 bool prefetching) {
    VFScenarioCostCalculationContext context(scheduling, config, tilesNumber, tilingInfo, costFunction, log,
                                             prefetching);
    auto calculator = PerformanceVFScenarioCostCalculatorFactory::create(scheduling.getType());
    return calculator->getTimeIntervals(context);
}

}  // namespace vpux::VPU::VF::v2
