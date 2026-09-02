//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/hash_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_hash.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <deque>

// Experimental number for the VF internal slice copy cost ratio
static constexpr double VF_INTERNAL_SLICE_DMA_COST_RATIO = 0.6;

namespace vpux::VPU::VF::v2 {
namespace {
ParentVFTilingInfo& getParentVFTilingInfo(VPU::VerticalFusionOp parentOp, ParentVFTilingInfoCache& cache) {
    auto* parentOperation = parentOp.getOperation();
    auto cachedInfo = llvm::find_if(cache, [&](const auto& info) {
        return info.parentOp == parentOperation;
    });
    if (cachedInfo != cache.end()) {
        return *cachedInfo;
    }

    auto parentCache = std::make_unique<VFCacheAnalysis>(parentOp.getOperation());
    auto parentConfig = std::make_unique<VFConfig>(parentOp, *parentCache);
    auto tilingInfo = std::make_unique<TilingOperationStorage>();
    auto tilingDims = parseIntArrayAttr<int64_t>(parentOp.getTilingStrategy());
    auto tilingStorage = calculateTilingRegions(parentOp, tilingDims, Logger::global(), tilingInfo);
    VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles", parentOp,
                    tilingDims);

    cache.emplace_back(parentOperation, std::move(parentCache), std::move(parentConfig), std::move(tilingInfo));
    return cache.back();
}

}  // namespace

llvm::hash_code getStrategyCostHash(mlir::Operation* operation, const VPUNNCostParameters& parameters) {
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

VFScheduling::VFScheduling(Logger log, bool prefetching /*true*/): _log(log), _prefetching(prefetching) {
}

const std::deque<std::shared_ptr<IVFScheduling<VFConfig>>>& VFScheduling::nextChecks() const {
    return _dependents;
}

void VFScheduling::addNext(std::shared_ptr<IVFScheduling<VFConfig>> check) {
    _dependents.emplace_back(check);
}

Byte VFScheduling::getInputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    const auto index = 0;
    auto inputSize = Byte(0);

    auto inputOperands = llvm::SmallPtrSet<mlir::Value, 4>();
    for (auto op : config.getInputs()) {
        auto tileInfo = tilingInfo->getRef(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        const auto& tileInfoValue = tileInfo.value().get();
        const auto& tileTypes = config.getOperationTypes(op, tileInfoValue.second, tileInfoValue.first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There are not enough types for tile of operation {0}", *op);

        // avoid repeatedly counting if inputOps share the same operand
        const auto inputTypesCount = tileTypes.size() - op->getNumResults();
        VPUX_THROW_UNLESS(inputTypesCount <= op->getNumOperands(), "Mismatch between tile types and operands");
        for (size_t i = 0; i < inputTypesCount; ++i) {
            auto operand = op->getOperand(i);
            if (inputOperands.contains(operand)) {
                continue;
            }
            inputOperands.insert(operand);
            inputSize += tileTypes[i].getTotalAllocSize();
        }
    }

    return inputSize;
}

Byte VFScheduling::getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    auto outputSize = Byte(0);
    const auto index = 0;

    for (auto op : config.getOutputs()) {
        auto tileInfo = tilingInfo->getRef(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        const auto& tileInfoValue = tileInfo.value().get();
        const auto& tileTypes = config.getOperationTypes(op, tileInfoValue.second, tileInfoValue.first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There is no output type for tile of operation {0}", *op);

        auto type = tileTypes.back();
        outputSize += type.getTotalAllocSize();
    }

    return outputSize;
}

VPUNNCostParameters VFScheduling::fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                                  const SmallVector<TileInfo>& inputTiles) const {
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

VPUNNCostParameters VFScheduling::fillInCostParam(mlir::Operation* operation,
                                                  const TilingOperationStorage::UPtr& opStorage, size_t index) const {
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

StrategyCost VFScheduling::getStrategyCost(VFConfig& config, mlir::Operation* operation,
                                           const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                           const VPUNNCostParameters& parameters) const {
    const auto hash = getStrategyCostHash(operation, parameters);
    auto cachedCost = config.getCachedStrategyCost(hash);
    if (cachedCost.has_value()) {
        return cachedCost.value();
    }

    auto cost = costFunction->getStrategyCost(operation, parameters);
    config.cacheStrategyCost(hash, cost);
    return cost;
}

/**
 * @brief Determines if a given operation has a prefetched DMA for a specific block argument.
 *  Differences from v1 implementation:
 * - For elementwise operations with more than one operand, the function checks if the current parent operation
 *   is scheduled before another parent operation in the block. It also calculates memory requirements to ensure
 *   they fit within the available CMX memory.
 */
bool hasPrefetchedDMA(mlir::Operation* operation, int64_t operandIdx, mlir::BlockArgument arg, VFConfig& config,
                      const VPUNNCostParameters& parameters, const bool isInput,
                      ParentVFTilingInfoCache& parentVFTilingInfoCache) {
    if (operation->hasTrait<VPU::EltwiseOp>() && operation->getNumOperands() > 1) {
        auto vfOp = operation->getParentOfType<VPU::VerticalFusionOp>();
        auto curParentOp = vfOp->getOperand(arg.getArgNumber()).getDefiningOp<VPU::VerticalFusionOp>();
        auto otherOperandIdx = operandIdx == 0 ? 1 : 0;
        if (auto otherOperandblockArg = getVFBlockArgument(operation->getOperand(otherOperandIdx))) {
            auto otherParentOp =
                    vfOp->getOperand(otherOperandblockArg.getArgNumber()).getDefiningOp<VPU::VerticalFusionOp>();
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

    if (nceOp != nullptr &&
        (arg != getVFBlockArgument(nceOp.getWeightsOperand()) && arg != getVFBlockArgument(nceOp->getOperand(0)))) {
        return false;
    }

    if (isInput) {
        return true;
    }

    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        return false;
    }

    return arg == getVFBlockArgument(nceOp.getWeightsOperand());
}

bool VFScheduling::isSharedWeightsSupported(VFConfig& config) const {
    // OptimizeParallelCopies may keep parallel copies when inplace eltwise are involved. Scenario becomes more complex
    // when there are multi eltwise ops inside. Skip this for now.
    auto hasMultiInPlaceEltwise = llvm::count_if(config.getVFOperations(), [](mlir::Operation* op) {
                                      return op->hasTrait<VPU::EltwiseOp>() && op->hasAttr(isInPlace);
                                  }) > 1;
    return !hasMultiInPlaceEltwise;
}

Byte VFScheduling::calculateSharedSize(VFConfig& config, mlir::Operation* operation,
                                       const vpux::VPU::VFOperationTiling& inputOutputTiling) const {
    const auto& inTile = inputOutputTiling.first;
    const auto& outTile = inputOutputTiling.second;
    Byte size(0);
    const auto& tiledTypes = config.getOperationTypes(operation, outTile, inTile.tiles);
    for (auto operandIdx : irange(tiledTypes.size() - operation->getNumResults())) {
        auto operand = operation->getOperand(operandIdx);
        if (operandIdx >= inTile.tiles.size()) {
            break;
        }
        if (mlir::isa<mlir::BlockArgument>(operand)) {
            auto canBeShared = isOperandSharedWeightsForTiling(operation, operand, inTile.tiles[operandIdx]);
            if (canBeShared) {
                size += tiledTypes[operandIdx].getTotalAllocSize();
            }
        }
    }
    return size;
};

Byte VFScheduling::getSharedSizeByAllTiles(ArrayRef<mlir::Operation*> operations, VFConfig& config,
                                           const TilingOperationStorage::UPtr& tilingInfo) const {
    Byte reservedSize(0);
    if (!isSharedWeightsSupported(config)) {
        return reservedSize;
    }

    const auto index = 0;
    for (auto* operation : operations) {
        auto opTiling = tilingInfo->getRef(operation, index);
        if (!opTiling.has_value()) {
            continue;
        }
        reservedSize += calculateSharedSize(config, operation, opTiling.value().get());
    }
    return reservedSize;
}

mlir::Operation* findParent(mlir::Value operand) {
    auto parent = operand.getDefiningOp();

    while (parent != nullptr && mlir::isa<VPU::TilingViewLikeOpInterface>(parent)) {
        parent = parent->getOperand(0).getDefiningOp();
    }

    return parent;
}

StrategyCost VFScheduling::getParentCost(mlir::Operation* operation,
                                         const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost) const {
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

void VFScheduling::correctOutputSpillCost(StrategyCost& /*spillCost*/, VFConfig& /*config*/,
                                          const DenseMap<mlir::Operation*, StrategyCost>& /*isolatedOperCost*/,
                                          SmallVector<StrategyCost>& /*prefetchCostList*/, const int64_t /*index*/,
                                          const int64_t /*tilesNumber*/) const {
}

void VFScheduling::correctInputPrefetchingCost(StrategyCost& /*prefetchCost*/, mlir::Operation* /*operation*/,
                                               VFConfig& /*config*/,
                                               const DenseMap<mlir::Operation*, StrategyCost>& /*isolatedOperCost*/,
                                               SmallVector<StrategyCost>& /*prefetchCostList*/,
                                               const size_t /*index*/) const {
}

std::optional<StrategyCost> VFScheduling::getViewLikeOpDMACost(
        mlir::Operation* operation, VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo, size_t tileIdx,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
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
    return costFunction->getSpillingTypeCost(userTileType, userCostParameters._tiling[0].axis);
}

StrategyCost VFScheduling::getCost(VFConfig& config, int64_t tilesNumber,
                                   const TilingOperationStorage::UPtr& tilingInfo,
                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    auto linearTimeIntervals = calculateLinearTimeIntervals(config, tilesNumber, tilingInfo, costFunction);
    return linearTimeIntervals.maxCost() == 0 ? std::numeric_limits<StrategyCost>::max()
                                              : linearTimeIntervals.maxCost();
}

StrategyCost VFScheduling::getPrefetchingCost(mlir::Operation* operation, VFConfig& config,
                                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                              const VPUNNCostParameters& parameters, const bool isInput,
                                              const TilingOperationStorage::UPtr& tilingInfo, const int64_t index,
                                              ParentVFTilingInfoCache& parentVFTilingInfoCache) const {
    StrategyCost prefetchedCost = 0;
    auto inputTiling = tilingInfo->getRef(operation, index);
    if (!inputTiling.has_value()) {
        return prefetchedCost;
    }

    const auto sharedWeightsEnabled = isSharedWeightsSupported(config);
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
            if (hasPrefetchedDMA(operation, input.index(), blockArg, config, parameters, isInput,
                                 parentVFTilingInfoCache)) {
                if (operationTypes.empty()) {
                    operationTypes =
                            config.getOperationTypes(operation, parameters._tiling[0], parameters._operandsTiling[0]);
                }
                prefetchedCost += costFunction->getSpillingTypeCost(operationTypes[input.index()],
                                                                    parameters._operandsTiling[0][input.index()].axis);
            }
        }
    }

    return prefetchedCost;
}

VFLinearContainer VFScheduling::calculateLinearTimeIntervals(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    // prefetch cost per tiles, size is tilesNumber + 1 to avoid out of range access when handling last tile
    SmallVector<StrategyCost> prefetchCostList(tilesNumber + 1, 0);
    StrategyCost fullCost = 0;
    VFLinearContainer linearTimeIntervals;
    const auto& inputs = config.getInputs();
    DenseMap<mlir::Operation*, StrategyCost> isolatedOperCost;
    ParentVFTilingInfoCache parentVFTilingInfoCache;
    _log.trace("Calculate linear cost for merged VF at {0} with tiles number {1}, op number {2}",
               config.getOutputs().front()->getLoc(), tilesNumber, config.getOperationsForTiling().size());

    // Sort operations in topological order (inputs first, outputs last) to ensure
    // that isolatedOperCost is populated for all producers before consumers reference them.
    SmallVector<mlir::Operation*> sortedOps(config.getVFOperations().begin(), config.getVFOperations().end());
    llvm::sort(sortedOps, [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    for (auto index : irange(tilesNumber)) {
        for (auto item : sortedOps | indexed) {
            auto lastEndTime = fullCost;
            auto opIndex = item.index();
            auto op = item.value();
            auto costParameters = fillInCostParam(op, tilingInfo, index);
            if (costParameters._tiling.empty()) {
                _log.warning("No tiling information for VF op at '{0}'", op->getLoc());
                linearTimeIntervals.invalidate();
                return linearTimeIntervals;
            }

            if (auto viewOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(op)) {
                // The view op has different input and output tile size, which means it will be converted to copy
                // after tiling. And the copy should be taken into account when calculating the cost of the VF.
                // Currently only view ops like slice ops will meet this condition. And the Spill write and read DMA
                // pairs will be further optimized into a single CMX2CMX DMA instead.
                auto copyCost = getViewLikeOpDMACost(op, config, tilingInfo, index, costFunction);
                if (copyCost.has_value()) {
                    linearTimeIntervals.addDMA(op, index, lastEndTime, copyCost.value());
                    lastEndTime += copyCost.value();
                    fullCost += copyCost.value();
                }
                continue;
            }

            // isolated operation cost
            auto isolatedCost = getStrategyCost(config, op, costFunction, costParameters);
            _log.trace("opIndex {0} isolated cost {1}", opIndex, isolatedCost);
            if (isolatedCost >= std::numeric_limits<StrategyCost>::max()) {
                _log.warning("Invalid VPUNN cost");
                linearTimeIntervals.invalidate();
                return linearTimeIntervals;
            }
            isolatedOperCost[op] = isolatedCost;
            fullCost += isolatedCost;

            StrategyCost outputCost = 0;
            StrategyCost correctedOutputCost = 0;
            if (llvm::find(config.getOutputs(), op) != config.getOutputs().end() && tilesNumber > 1) {
                // add the cost of output dma
                outputCost = costFunction->getSpillingTypeCost(
                        config.getOperationTypes(op, costParameters._tiling[0], costParameters._operandsTiling[0])
                                .back(),
                        costParameters._tiling[0].axis);
                if (outputCost >= std::numeric_limits<StrategyCost>::max()) {
                    _log.warning("Invalid VPUNN cost");
                    linearTimeIntervals.invalidate();
                    return linearTimeIntervals;
                }
                _log.trace("opIndex {0} original output spill cost {1}", opIndex, outputCost);
                correctedOutputCost = outputCost;
                correctOutputSpillCost(correctedOutputCost, config, isolatedOperCost, prefetchCostList, index,
                                       tilesNumber);
                _log.trace("opIndex {0} corrected output spill cost {1}", opIndex, correctedOutputCost);
                fullCost += correctedOutputCost;
            }
            const bool isInput = llvm::find(inputs, op) != inputs.end();
            StrategyCost prefetchedCost = getPrefetchingCost(op, config, costFunction, costParameters, isInput,
                                                             tilingInfo, index, parentVFTilingInfoCache);
            StrategyCost correctedPrefetchedCost = prefetchedCost;

            if (prefetchedCost >= std::numeric_limits<StrategyCost>::max()) {
                _log.warning("Invalid VPUNN cost");
                linearTimeIntervals.invalidate();
                return linearTimeIntervals;
            }
            if (_prefetching && prefetchedCost > 0) {
                _log.trace("opIndex {0} original prefetch spill cost {1}", opIndex, prefetchedCost);
                correctInputPrefetchingCost(correctedPrefetchedCost, op, config, isolatedOperCost, prefetchCostList,
                                            index);
                _log.trace("opIndex {0} corrected prefetch spill cost {1}", opIndex, correctedPrefetchedCost);
            }
            fullCost += correctedPrefetchedCost;

            auto internalSliceCost =
                    getInternalSliceCopyCost(op, config, costFunction, costParameters, isInput, tilingInfo, index);

            if (internalSliceCost > 0) {
                _log.trace("opIndex {0} internal slice spill cost {1}", opIndex, internalSliceCost);
                fullCost += internalSliceCost;
            }

            if (prefetchedCost > 0) {
                linearTimeIntervals.addDMA(op, index, lastEndTime - prefetchedCost + correctedPrefetchedCost,
                                           prefetchedCost);
                lastEndTime += correctedPrefetchedCost;
            }
            if (internalSliceCost > 0) {
                linearTimeIntervals.addDMA(op, index, lastEndTime, internalSliceCost);
                lastEndTime += internalSliceCost;
            }
            linearTimeIntervals.addOperation(op, index, lastEndTime, isolatedCost);
            lastEndTime += isolatedCost;
            if (outputCost > 0) {
                linearTimeIntervals.addDMA(op, index, lastEndTime, outputCost);
            }
        }
    }
    _log.trace("Total linear cost: {0}", fullCost);
    return linearTimeIntervals;
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
StrategyCost VFScheduling::getInternalSliceCopyCost(mlir::Operation* op, VFConfig& config,
                                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                    const VPUNNCostParameters& parameters, const bool isInput,
                                                    const TilingOperationStorage::UPtr& tilingInfo,
                                                    const int64_t index) const {
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

void VFScheduling::reduceCostWithPrefetchedDMA(StrategyCost& parentCost, const StrategyCost& prefetchCost,
                                               StrategyCost& accumuatedPrefetchCost) const {
    // reduce the cost with dmas prefetched for this tile
    if (accumuatedPrefetchCost <= parentCost) {
        parentCost -= accumuatedPrefetchCost;
    }
    accumuatedPrefetchCost += prefetchCost;
}

SmallVector<TimelineInterval> VFScheduling::getTimeIntervals(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    auto linearTimeIntervals = calculateLinearTimeIntervals(config, tilesNumber, tilingInfo, costFunction);
    return linearTimeIntervals.getAllIntervals();
}
}  // namespace vpux::VPU::VF::v2
