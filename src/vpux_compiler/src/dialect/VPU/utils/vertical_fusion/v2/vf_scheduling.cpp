//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <deque>

// Experimental number for the VF internal slice copy cost ratio
static constexpr double VF_INTERNAL_SLICE_DMA_COST_RATIO = 0.6;

namespace vpux::VPU::VF::v2 {
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

    for (auto op : config.getInputs()) {
        auto tileInfo = tilingInfo->get(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1} {2}", index,
                        *op, config.getSubgraph());

        auto tileTypes = config.getOperationTypes(op, tileInfo.value().second, tileInfo.value().first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There are not enough types for tile of operation {0}", *op);

        // exclude output type information
        tileTypes.pop_back();
        for (auto type : tileTypes) {
            inputSize += type.getTotalAllocSize();
        }
    }

    return inputSize;
}

Byte VFScheduling::getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    auto outputSize = Byte(0);
    const auto index = 0;

    for (auto op : config.getOutputs()) {
        auto tileInfo = tilingInfo->get(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        auto tileTypes = config.getOperationTypes(op, tileInfo.value().second, tileInfo.value().first.tiles);
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
    auto inputOutputTiling = opStorage->get(operation, index);

    OutputTiling outputTiling;
    SmallVector<TileInfo> inputTiling;

    if (inputOutputTiling.has_value()) {
        outputTiling = {inputOutputTiling.value().second};
        inputTiling = inputOutputTiling.value().first.tiles;
    }

    return fillInCostParam(operation, outputTiling, inputTiling);
}

/**
 * @brief Determines if a given operation has a prefetched DMA for a specific block argument.
 *  Differences from v1 implementation:
 * - For elementwise operations with more than one operand, the function checks if the current parent operation
 *   is scheduled before another parent operation in the block. It also calculates memory requirements to ensure
 *   they fit within the available CMX memory.
 */
bool hasPrefetchedDMA(mlir::Operation* operation, mlir::BlockArgument arg, VFConfig& config,
                      const VPUNNCostParameters& parameters, const bool isInput) {
    if (operation->hasTrait<VPU::EltwiseOp>() && operation->getNumOperands() > 1) {
        auto getOperandIdx = [&]() {
            auto uses = arg.getUses();
            for (auto& use : uses) {
                if (use.getOwner() == operation) {
                    return use.getOperandNumber();
                }
            }
            VPUX_THROW("Cannot find the operand index for {0}", operation->getLoc());
        };

        auto vfOp = operation->getParentOfType<VPU::VerticalFusionOp>();
        auto curParentOp = vfOp->getOperand(arg.getArgNumber()).getDefiningOp<VPU::VerticalFusionOp>();

        auto operandIdx = getOperandIdx();
        auto otherOperandIdx = operandIdx == 0 ? 1 : 0;
        if (auto otherOperandblockArg = mlir::dyn_cast<mlir::BlockArgument>(operation->getOperand(otherOperandIdx))) {
            auto otherParentOp =
                    vfOp->getOperand(otherOperandblockArg.getArgNumber()).getDefiningOp<VPU::VerticalFusionOp>();
            if (curParentOp != nullptr && otherParentOp != nullptr && curParentOp->isBeforeInBlock(otherParentOp)) {
                auto operandSize = config.getOperationTypes(operation, parameters._tiling[0],
                                                            parameters._operandsTiling[0])[operandIdx]
                                           .getTotalAllocSize();

                auto parentVfConfig = VFConfig(otherParentOp);
                auto tilingInfo = std::make_unique<TilingOperationStorage>();
                auto tilingDims = parseIntArrayAttr<int64_t>(otherParentOp.getTilingStrategy());
                auto tilingStorage = calculateTilingRegions(otherParentOp, tilingDims, Logger::global(), tilingInfo);
                VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles",
                                curParentOp, tilingDims);
                auto lastOp = parentVfConfig.getOperationsForTiling().back();
                auto inputOutputTiling = tilingInfo->get(lastOp, 0);
                auto lastTileSize = VPU::getRequiredCMX(
                        lastOp, parentVfConfig.getOperationTypes(lastOp, inputOutputTiling.value().second,
                                                                 inputOutputTiling.value().first.tiles));
                return operandSize + lastTileSize > getTotalCMXFragmentationAwareSize(operation);
            }
        }
        return true;
    }

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);

    if (nceOp != nullptr && (arg != nceOp.getWeightsOperand() && arg != nceOp->getOperand(0))) {
        return false;
    }

    if (isInput) {
        return true;
    }

    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        return false;
    }

    return arg == nceOp.getWeightsOperand();
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
    auto tiledTypes = config.getOperationTypes(operation, outTile, inTile.tiles);
    for (auto operandIdx : irange(tiledTypes.size() - operation->getNumResults())) {
        auto operand = operation->getOperand(operandIdx);
        if (operandIdx >= inTile.tiles.size()) {
            break;
        }
        if (mlir::isa<mlir::BlockArgument>(operand)) {
            auto canBeShared = isOperandSharedWeightsForTiling(operation, operand, tiledTypes[operandIdx].getShape());
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
        auto opTiling = tilingInfo->get(operation, index);
        if (!opTiling.has_value()) {
            continue;
        }
        reservedSize += calculateSharedSize(config, operation, opTiling.value());
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
                                          const int64_t /*index*/, const int64_t /*tilesNumber*/) const {
}

void VFScheduling::correctInputPrefetchingCost(StrategyCost& /*prefetchCost*/, mlir::Operation* /*operation*/,
                                               VFConfig& /*config*/,
                                               const DenseMap<mlir::Operation*, StrategyCost>& /*isolatedOperCost*/,
                                               const size_t /*index*/) const {
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
                                              const TilingOperationStorage::UPtr& tilingInfo,
                                              const int64_t index) const {
    StrategyCost prefetchedCost = 0;
    auto inputTiling = tilingInfo->get(operation, index);
    if (!inputTiling.has_value()) {
        return prefetchedCost;
    }

    const auto sharedWeightsEnabled = isSharedWeightsSupported(config);

    for (auto input : operation->getOperands() | indexed) {
        if (input.index() >= inputTiling.value().first.tiles.size()) {
            break;
        }
        auto inputOperand = input.value();
        auto inTiledShape = inputTiling.value().first.tiles[input.index()].shape;
        auto isAlreadyShared = index != 0 && sharedWeightsEnabled &&
                               isOperandSharedWeightsForTiling(operation, inputOperand, inTiledShape);
        if (isAlreadyShared) {
            continue;
        }

        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(inputOperand)) {
            if (hasPrefetchedDMA(operation, blockArg, config, parameters, isInput)) {
                prefetchedCost += costFunction->getSpillingTypeCost(
                        config.getOperationTypes(operation, parameters._tiling[0],
                                                 parameters._operandsTiling[0])[input.index()],
                        parameters._operandsTiling[0][input.index()].axis);
            }
        }
    }

    return prefetchedCost;
}

VFLinearContainer VFScheduling::calculateLinearTimeIntervals(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    _prefetchedCost.clear();
    StrategyCost fullCost = 0;
    VFLinearContainer linearTimeIntervals;
    auto inputs = config.getInputs();
    DenseMap<mlir::Operation*, StrategyCost> isolatedOperCost;
    _log.trace("Calculate linear cost for merged VF at {0} with tiles number {1}, op number {2}",
               config.getSubgraph().getLoc(), tilesNumber, config.getOperationsForTiling().size());
    for (auto index : irange(tilesNumber)) {
        for (auto item : config.getOperationsForTiling() | indexed) {
            auto lastEndTime = fullCost;
            auto opIndex = item.index();
            auto op = item.value();
            auto costParameters = fillInCostParam(op, tilingInfo, index);
            if (costParameters._tiling.empty()) {
                _log.warning("No tiling information for VF op at '{0}'", op->getLoc());
                linearTimeIntervals.invalidate();
                return linearTimeIntervals;
            }

            // isolated operation cost
            auto isolatedCost = costFunction->getStrategyCost(op, costParameters);
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
                correctOutputSpillCost(correctedOutputCost, config, isolatedOperCost, index, tilesNumber);
                _log.trace("opIndex {0} corrected output spill cost {1}", opIndex, correctedOutputCost);
                fullCost += correctedOutputCost;
            }
            const bool isInput = llvm::find(inputs, op) != inputs.end();
            StrategyCost prefetchedCost =
                    getPrefetchingCost(op, config, costFunction, costParameters, isInput, tilingInfo, index);
            StrategyCost correctedPrefetchedCost = prefetchedCost;

            if (prefetchedCost >= std::numeric_limits<StrategyCost>::max()) {
                _log.warning("Invalid VPUNN cost");
                linearTimeIntervals.invalidate();
                return linearTimeIntervals;
            }
            if (_prefetching && prefetchedCost > 0) {
                _log.trace("opIndex {0} original prefetch spill cost {1}", opIndex, prefetchedCost);
                correctInputPrefetchingCost(correctedPrefetchedCost, op, config, isolatedOperCost, index);
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

    auto ops = config.getOperationsForTiling();
    auto allParentsAreInVF = llvm::all_of(parents, [&](auto* parent) {
        return llvm::find(ops, parent) != ops.end();
    });

    if (!allParentsAreInVF || parentLeft == parentRight) {
        return cost;
    }

    auto* earliestParent = parentLeft->isBeforeInBlock(parentRight) ? parentLeft : parentRight;
    auto* closestParent = earliestParent == parentLeft ? parentRight : parentLeft;

    SmallVector<mlir::Operation*> chain;
    const auto isParentOperation = [&]() {
        chain.emplace_back(closestParent);
        auto curOp = closestParent;
        while ((curOp = findParent(curOp->getOperand(0))) != nullptr) {
            if (curOp == earliestParent) {
                return true;
            }
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(curOp);
            if (nceOp != nullptr && nceOp.getWeightsOperand() != nullptr) {
                chain.emplace_back(curOp);
            }
        }
        return false;
    };
    if (!isParentOperation()) {
        return cost;
    }

    auto origOutShape = getShape(op->getResult(0));
    auto opTiling = tilingInfo->get(op, index).value();
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
        auto tileTypes = config.getOperationTypes(op, opTiling.second, opTiling.first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "Can not get tiled types for tile of operation {0}", op->getLoc());
        auto type = tileTypes.front();
        auto inputSize = type.getTotalAllocSize();
        auto closestParentOpTiling = tilingInfo->get(closestParent, index).value();
        auto requiredCMXSize =
                inputSize +
                VPU::getRequiredCMX(closestParent, config.getOperationTypes(closestParent, closestParentOpTiling.second,
                                                                            closestParentOpTiling.first.tiles));

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
                                               const size_t index) const {
    // reduce the cost with dmas prefetched for this tile
    if (_prefetchedCost[index] <= parentCost) {
        parentCost -= _prefetchedCost[index];
    }
    _prefetchedCost[index] += prefetchCost;
}

SmallVector<TimelineInterval> VFScheduling::getTimeIntervals(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    auto linearTimeIntervals = calculateLinearTimeIntervals(config, tilesNumber, tilingInfo, costFunction);
    return linearTimeIntervals.getAllIntervals();
}
}  // namespace vpux::VPU::VF::v2
