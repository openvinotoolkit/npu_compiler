//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_pass_config_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <deque>

namespace vpux::VPU::VF::v2 {

std::optional<int64_t> findOptimalTilingStrategyInRange(
        const std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>& scheduling, const Dim dim, int64_t minNTiles,
        int64_t& maxNTiles, std::unique_ptr<IVFAxisIncrement>& axisIncrement, ArrayRef<int64_t> origTilingArray,
        TilingOperationStorage::UPtr& minStorage, TilingOperationStorage::UPtr& maxStorage, VFConfig& config,
        Logger log) {
    auto returnFailure = [&minStorage]() -> std::optional<int64_t> {
        minStorage.reset();
        return std::nullopt;
    };

    const auto origMaxTile = maxNTiles;
    SmallVector<int64_t> tilingMaxStrategy(origTilingArray.begin(), origTilingArray.end());
    SmallVector<int64_t> tilingArray(origTilingArray.begin(), origTilingArray.end());

    // Binary search: find minimal valid tiling number within [minNTiles, maxNTiles)
    while (minNTiles < maxNTiles) {
        // Check convergence: if next increment equals max, it is the boundary
        auto nextValueFromMin = minNTiles;
        axisIncrement->increasedValue(nextValueFromMin, maxNTiles);
        if (maxNTiles == nextValueFromMin) {
            // Use maxStorage if returning origMaxTile, otherwise minStorage already set
            if (maxNTiles == origMaxTile) {
                minStorage.reset(maxStorage.release());
            }
            return maxNTiles;
        }

        // Calculate middle point
        auto currentNTiles = axisIncrement->getMiddleValue(minNTiles, maxNTiles);
        if (currentNTiles == minNTiles) {
            return returnFailure();
        }

        // Get valid tiling strategy for currentNTiles
        tilingMaxStrategy[dim.ind()] = maxNTiles;
        tilingArray[dim.ind()] = currentNTiles;

        auto opStorage = std::make_unique<TilingOperationStorage>();
        auto validStrategy =
                getMinimalValidTilingStrategyFromRange(config, tilingArray, tilingMaxStrategy, dim, opStorage, log);
        if (mlir::failed(validStrategy)) {
            return returnFailure();
        }

        currentNTiles = validStrategy.value()[dim.ind()];

        // Handle reaching upper bound
        if (currentNTiles == maxNTiles) {
            minStorage.reset(opStorage.release());
            return currentNTiles;
        }

        // Update search range based on validation result
        if (scheduling->validate(config, opStorage)) {
            maxNTiles = currentNTiles;
            minStorage.reset(opStorage.release());
            log.trace("Valid tiling found: tiles={0}", currentNTiles);
        } else {
            minNTiles = currentNTiles;
            log.trace("Invalid tiling: tiles={0}, searching higher", currentNTiles);
        }
    }

    return returnFailure();
};

std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>> getSchedulingScenarios(VFCase::VFConfigType& config,
                                                                                        Logger log) {
    std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>> vfChecks;
    VFSchedulingFactory vfFactory(true);

    auto minimalCheck = vfFactory.createVFScenario(VFScenario::MINIMAL, log);

    if (config.isPipelined()) {
        auto pipeliningChecks = vfFactory.createVFScenario(VFScenario::VF_PIPELINING, log);
        minimalCheck->addNext(std::move(pipeliningChecks));
    }
    auto prefetchingCheck = vfFactory.createVFScenario(VFScenario::LASTOP_PREFETCHING, log);
    auto weightsCheck = vfFactory.createVFScenario(VFScenario::WEIGHTS_PREFETCHING, log);
    auto fullPrefetching = vfFactory.createVFScenario(VFScenario::FULL_PREFETCHING, log);
    weightsCheck->addNext(std::move(fullPrefetching));
    prefetchingCheck->addNext(std::move(weightsCheck));
    minimalCheck->addNext(std::move(prefetchingCheck));
    vfChecks.emplace_back(std::move(minimalCheck));

    return vfChecks;
}

std::optional<int64_t> getOptimalTilingStrategy(const std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>& scheduling,
                                                const Dim dim, const VFSplit& split, const int64_t minTiles,
                                                int64_t& maxTiles, TilingOperationStorage::UPtr& minStorage,
                                                TilingOperationStorage::UPtr& maxStorage, VFCase::VFConfigType& config,
                                                Logger log) {
    if (minTiles > maxTiles || maxTiles == 1) {
        return std::nullopt;
    }

    auto minNTiles = minTiles;
    auto maxNTiles = maxTiles;

    std::optional<int64_t> result;
    auto outType = mlir::cast<vpux::NDTypeInterface>(config.getOutputs().back()->getResult(0).getType());
    auto tilingArray = restoreTilingBySplit(outType.getRank(), split);
    tilingArray[dim.ind()] = minNTiles;
    if (minTiles == maxTiles) {
        if (minStorage == nullptr) {
            minStorage = std::make_unique<TilingOperationStorage>();
            auto tilingRegions = calculateTilingRegions(config, tilingArray, log, minStorage);

            if (mlir::failed(tilingRegions)) {
                minStorage.reset();
                return std::nullopt;
            }
        }

        if (scheduling->validate(config, minStorage)) {
            result = minTiles;
        }
        return result;
    }

    auto tilingMaxStrategy = restoreTilingBySplit(outType.getRank(), split);
    tilingMaxStrategy[dim.ind()] = maxNTiles;

    if (minStorage == nullptr) {
        minStorage = std::make_unique<TilingOperationStorage>();
        auto getValidStrategy =
                getMinimalValidTilingStrategyFromRange(config, tilingArray, tilingMaxStrategy, dim, minStorage, log);

        if (mlir::failed(getValidStrategy)) {
            minStorage.reset();
            return std::nullopt;
        }

        tilingArray = getValidStrategy.value();
        minNTiles = tilingArray[dim.ind()];
    }

    if (scheduling->validate(config, minStorage)) {
        result = minNTiles;
        return result;
    }

    auto axisIncrement = getVFAxisIncrement(dim);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", dim);

    if (maxStorage == nullptr) {
        maxStorage = std::make_unique<TilingOperationStorage>();
        // When maxNTiles is too large,  to avoid spending too much time on calculating, try to check if the cube root
        // of the max tile is valid or not.
        auto cbrtMaxTile = getCbrtMaxTileCandidate(minNTiles, maxNTiles);
        mlir::FailureOr<SmallVector<int64_t>> getValidStrategy = mlir::failure();
        if (cbrtMaxTile.has_value()) {
            auto tilingCbrtMaxStrategy = tilingMaxStrategy;
            tilingCbrtMaxStrategy[dim.ind()] = cbrtMaxTile.value();
            getValidStrategy = getMaximalValidTilingStrategyFromRange(config, tilingArray, tilingCbrtMaxStrategy, dim,
                                                                      maxStorage, log);

            auto useCbrtMaxTileStrategy = mlir::succeeded(getValidStrategy) && scheduling->validate(config, maxStorage);
            if (useCbrtMaxTileStrategy) {
                maxNTiles = getValidStrategy.value()[dim.ind()];
                result = findOptimalTilingStrategyInRange(scheduling, dim, minNTiles, maxNTiles, axisIncrement,
                                                          tilingArray, minStorage, maxStorage, config, log);
                maxStorage.reset();
                return result;
            }
            maxStorage.reset();
        }

        getValidStrategy =
                getMaximalValidTilingStrategyFromRange(config, tilingArray, tilingMaxStrategy, dim, maxStorage, log);
        if (mlir::failed(getValidStrategy)) {
            maxStorage.reset();
            return std::nullopt;
        }
        maxTiles = tilingMaxStrategy[dim.ind()];
        tilingMaxStrategy = getValidStrategy.value();
        maxNTiles = tilingMaxStrategy[dim.ind()];
    }

    if (!scheduling->validate(config, maxStorage)) {
        return std::nullopt;
    }
    return findOptimalTilingStrategyInRange(scheduling, dim, minNTiles, maxNTiles, axisIncrement, tilingArray,
                                            minStorage, maxStorage, config, log);
}

VPU::VF::v2::VFCase getVFCaseWithTiling(
        VPU::VF::v2::VFConfig& config, Dim dim, const VPU::VF::v2::VFSplit& split,
        const std::function<int64_t(Dim, const VFSplit&)>& minNumCalc,
        const std::function<int64_t(Dim, const VFSplit&)>& maxNumCalc, Logger log,
        const std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>>& vfSchedulingChecks) {
    auto minTiles = minNumCalc(dim, split);
    auto maxTiles = maxNumCalc(dim, split);
    auto mergedCase = VPU::VF::v2::VFCase(config, split);

    auto schedulingChecks = vfSchedulingChecks;

    TilingOperationStorage::UPtr maxStorage = nullptr;
    TilingOperationStorage::UPtr minStorage = nullptr;

    while (!schedulingChecks.empty()) {
        auto currentCheck = schedulingChecks.front();
        schedulingChecks.pop_front();
        auto numTiles = getOptimalTilingStrategy(currentCheck, dim, split, minTiles, maxTiles, minStorage, maxStorage,
                                                 mergedCase.getConfig(), log);

        if (numTiles.has_value()) {
            mergedCase.setTilingNumber(dim, numTiles.value());
            mergedCase.setScheduling(currentCheck);

            if (currentCheck->nextChecks().empty()) {
                mergedCase.setTilingStorage(std::move(minStorage));
                return mergedCase;
            }

            for (const auto& check : currentCheck->nextChecks() | reversed) {
                schedulingChecks.push_front(check);
            }
            minTiles = numTiles.value();
        }
    }
    mergedCase.setTilingStorage(std::move(minStorage));

    return mergedCase;
}

std::optional<VFCase> findBestVFCaseFromSplits(VFConfig& config, ArrayRef<VFSplit> splits,
                                               const std::function<int64_t(Dim, const VFSplit&)>& minNumCalc,
                                               const std::function<int64_t(Dim, const VFSplit&)>& maxNumCalc,
                                               const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log,
                                               mlir::MLIRContext* ctx, const SplitFilterFn& splitFilter) {
    if (splits.empty()) {
        return std::nullopt;
    }

    auto vfSchedulingChecks = getSchedulingScenarios(config, log);

    std::vector<std::optional<VFCase>> vfCases(splits.size(), std::nullopt);
    std::vector<StrategyCost> vfCosts(splits.size(), std::numeric_limits<StrategyCost>::max());

    loop_1d(LoopExecPolicy::Parallel, ctx, splits.size(), [&](auto splitIndex) {
        auto dimOpt = getVFOptimizedDim(splits[splitIndex]);
        if (!dimOpt.has_value()) {
            return;
        }
        auto dim = dimOpt.value();

        if (splitFilter != nullptr && splitFilter(dim, splits[splitIndex])) {
            return;
        }

        auto vfCase =
                getVFCaseWithTiling(config, dim, splits[splitIndex], minNumCalc, maxNumCalc, log, vfSchedulingChecks);
        if (!vfCase.isInitialized()) {
            return;
        }

        // If the split contains channel and one spatial dim, try to adjust the channel tile size
        if (splits[splitIndex].size() == 2) {
            auto iter = llvm::find_if(splits[splitIndex], [&](const std::pair<Dim, std::optional<int64_t>>& item) {
                return item.first != dim;
            });
            if (iter != splits[splitIndex].end()) {
                auto otherDim = iter->first;
                if (!isSpatialDim(otherDim)) {
                    auto newSplit = splits[splitIndex];
                    newSplit[dim] = parseIntArrayAttr<int64_t>(vfCase.getTiling())[dim.ind()];
                    newSplit[otherDim] = std::nullopt;
                    vfCase = getVFCaseWithTiling(config, otherDim, newSplit, minNumCalc, maxNumCalc, log,
                                                 vfSchedulingChecks);
                    if (!vfCase.isInitialized()) {
                        return;
                    }
                }
            }
        }

        vfCosts[splitIndex] = vfCase.getCost(costFunction, log.nest());
        vfCases[splitIndex] = std::move(vfCase);
    });

    auto bestCostIter = std::min_element(vfCosts.begin(), vfCosts.end());
    if (*bestCostIter == std::numeric_limits<StrategyCost>::max()) {
        return std::nullopt;
    }
    auto bestIndex = std::distance(vfCosts.begin(), bestCostIter);
    return std::move(vfCases[bestIndex]);
}

bool isVFMergeProfitable(VFCase& mergedCase, StrategyCost prevCost, StrategyCost currentCost,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    if (!mergedCase.isInitialized()) {
        return false;
    }

    auto mergedVFCost = mergedCase.getCost(costFunction, log);

    if (mergedVFCost > VPUNN::Cycles::cost_adder(prevCost, currentCost)) {
        log.trace("VF merge not profitable: mergedVFCost ({0}) > prevCost ({1}) + currentCost ({2})", mergedVFCost,
                  prevCost, currentCost);
        return false;
    }
    log.trace("VF merge profitable: mergedVFCost ({0}) <= prevCost ({1}) + currentCost ({2})", mergedVFCost, prevCost,
              currentCost);

    mergedCase.approveScheduling();
    return true;
}

VPU::VF::v2::VFSplit getVFSplit(vpux::NDTypeInterface outputType, mlir::Operation* outputOperation,
                                DimArrRef allowedDims, VPU::VF::v2::VFConfig& config) {
    auto shapeType = mlir::cast<mlir::ShapedType>(outputType);
    if (!shapeType.hasStaticShape()) {
        VPU::VF::v2::VFSplit vfSplit;
        auto countDynDims = shapeType.getNumDynamicDims();
        // allowedDims to memoryOrder
        auto outputOrder = outputType.getDimsOrder();
        auto allowedDimsInMemoryOrder = allowedDims.vec();
        llvm::sort(allowedDimsInMemoryOrder, [&](const Dim& d1, const Dim& d2) {
            return outputOrder.dimPos(d1) < outputOrder.dimPos(d2);
        });
        const auto hasDynAlignment = VPU::hasDynamicDimAlignment(outputOperation);
        auto boundedShape = getBoundedShape(outputType);
        std::optional<SmallVector<int64_t>> alignmentValues;
        auto innerMostDynamicDim = getInnermostDynamicDim(outputType.getShape(), outputType.getDimsOrder());
        if (hasDynAlignment) {
            alignmentValues = getAlignment(outputOperation, {}, boundedShape, true);
        }

        for (auto dim : allowedDimsInMemoryOrder) {
            if (!shapeType.isDynamicDim(dim.ind())) {
                continue;
            }

            VPUX_THROW_WHEN(dim == Dims4D::Act::C, "Dynamic channels are not supported");
            if (hasDynAlignment && alignmentValues.has_value() && innerMostDynamicDim.has_value()) {
                if (dim == innerMostDynamicDim.value() && countDynDims > 1) {
                    vfSplit[dim] = divUp(boundedShape[dim], alignmentValues.value()[dim.ind()]);
                } else {
                    vfSplit[dim] = std::nullopt;
                }
                continue;
            }
            if (countDynDims == 1) {
                vfSplit[dim] = std::nullopt;
                break;
            }

            vfSplit[dim] = getTilingLimit(dim, config, true);
            --countDynDims;
        }
        return vfSplit;
    }

    VPU::SiblingOpsAnalysis siblingAnalisys(outputOperation);
    auto outputShape = outputType.getShape().toValues();
    if (auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(outputOperation)) {
        if (clusterOp.getMultiClusterStrategy().has_value()) {
            outputType = clusterOp.getDistributedTypeForOpResult(
                    outputOperation->getResult(0), clusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false);

            auto distribution = VPU::DistributionInfo::getClassFromAttr(
                    mlir::cast<VPU::DistributedTensorType>(outputType).getDistribution());
            if (distribution.getMemoryShapes().empty()) {
                auto optMemoryShapes =
                        VPU::getPerClusterMemoryShapes(outputShape, distribution, outputType.getElementType());
                if (optMemoryShapes.has_value()) {
                    outputShape = Shape(optMemoryShapes.value().front());
                }
            } else {
                outputShape = Shape(distribution.getMemoryShapes().front());
            }
        }
    }

    const auto compareDims = [&](auto dimLeft, auto dimRight) {
        return outputShape[dimLeft] < outputShape[dimRight];
    };
    auto maxDim = std::max_element(allowedDims.begin(), allowedDims.end(), compareDims);

    if (maxDim == allowedDims.end()) {
        return {};
    }

    const auto parseTilingStrategy = [](mlir::Operation* op) {
        return Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(tilingStrategy))));
    };

    const auto maxDimMap = buildOpDimMap(config, *maxDim);

    int64_t maxTilingOnMaxDim = 1;
    for (auto* op : config.getVFOperations()) {
        maxTilingOnMaxDim =
                std::max(maxTilingOnMaxDim, op->hasAttr(tilingStrategy)
                                                    ? parseTilingStrategy(op)[maxDimMap.find(op)->second]
                                                    : getTilingLimit(*maxDim, config, /*multiDimTiling=*/false));
    }

    VPU::VF::v2::VFSplit vfSplit;
    const auto maxAllowedTiles1D = getTilingLimit(*maxDim, config, /*multiDimTiling=*/false);
    if (maxTilingOnMaxDim > maxAllowedTiles1D) {
        for (auto dim : allowedDims) {
            if (dim == *maxDim) {
                continue;
            }
            const auto dimMap = buildOpDimMap(config, dim);
            const bool anyOpTilesOnDim = llvm::any_of(config.getVFOperations(), [&](mlir::Operation* op) {
                if (op == outputOperation || !op->hasAttr(tilingStrategy)) {
                    return false;
                }
                const auto strategy = parseTilingStrategy(op);
                return strategy[dimMap.find(op)->second] > 1;
            });
            if (anyOpTilesOnDim) {
                vfSplit[dim] = getTilingLimit(dim, config, /*multiDimTiling=*/true);
            }
        }
    }
    vfSplit[*maxDim] = std::nullopt;
    return vfSplit;
}

VPU::VF::v2::VFSplit getVFSplitForSCF(VF::v2::VFConfig& config, DimArrRef allowedDims) {
    auto outputs = config.getOutputs();
    if (outputs.empty()) {
        return {};
    }
    auto* lastOp = outputs.back();
    auto outputType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType());
    return getVFSplit(outputType, lastOp, allowedDims, config);
}

VPU::VF::v2::VFCase computeVFSCFCase(vpux::NDTypeInterface outputType, const VPU::VF::v2::VFSplit& vfSplit,
                                     VPU::VF::v2::VFConfig& config, mlir::Operation* rootOp, Logger log) {
    VPU::VF::v2::VFCase emptyVFCase(config, {});
    if (vfSplit.empty()) {
        return emptyVFCase;
    }

    auto optDim = VPU::VF::v2::getNonTiledDimForVFOptimization(vfSplit);
    if (!optDim.has_value()) {
        return emptyVFCase;
    }

    const auto getMinTiles = [&](auto dim, const VPU::VF::v2::VFSplit& split) {
        if ((outputType.getShape().isDynamic() && dim == optDim.value()) || split.size() > 1) {
            return VPU::MIN_REQUIRED_TILES;
        }

        const auto getDimValue = [&dim](auto* oper) -> int64_t {
            return oper->hasAttr(tilingStrategy) ? Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(
                                                           oper->getAttr(tilingStrategy))))[dim]
                                                 : 1;
        };

        std::set<int64_t> minTilesSet;
        llvm::copy(config.getVFOperations() | transformed(getDimValue), std::inserter(minTilesSet, minTilesSet.end()));
        return *minTilesSet.rbegin();
    };

    const auto getMaxTiles = [&](auto dim, const VPU::VF::v2::VFSplit& split) -> int64_t {
        auto maxTiles = getTilingLimit(dim, config);
        if (!VPU::hasDynamicDimAlignment(rootOp) && split.size() > 1) {
            auto otherDimSum = VPU::VF::v2::getVFTilesLen(split);
            maxTiles = divUp(maxTiles, otherDimSum);

            if (outputType.getShape().isDynamic()) {
                maxTiles = std::max(maxTiles, VPU::MINIMUM_LENGTH_TILING);
            }
        }
        return maxTiles;
    };

    return VPU::VF::v2::getVFCaseWithTiling(config, optDim.value(), vfSplit, getMinTiles, getMaxTiles, log,
                                            VPU::VF::v2::getSchedulingScenarios(config, log));
}

bool isViewLikeOpVFCompatible(mlir::Operation* op) {
    // extend the check to be the same as it is in default pipeline
    // E-218728
    auto viewLikeOp = mlir::dyn_cast_or_null<VPU::TilingViewLikeOpInterface>(op);
    if (viewLikeOp == nullptr) {
        return false;
    }
    if (!VPU::isPureViewOp(op) || !viewLikeOp.isVFSupported()) {
        return false;
    }
    // Reject ops sourced exclusively from block arguments or constants
    if (llvm::all_of(op->getOperands(), [](mlir::Value value) {
            return mlir::isa_and_nonnull<mlir::BlockArgument>(value) ||
                   mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
        })) {
        return false;
    }
    return true;
}

VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                    ArrayRef<TileInfo> inputTiles, const bool enablePrefetching) {
    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
        mcStrategy = mcOperation.getMultiClusterStrategy().value_or(mcStrategy);
    }

    auto mode = enablePrefetching ? TilingMode::PREFETCHING : TilingMode::ISOLATED;

    SmallVector<OutputTiling> inputAllTiles;
    if (!inputTiles.empty()) {
        inputAllTiles.emplace_back(inputTiles);
    }
    return VPUNNCostParameters(mcStrategy, tiling, mode, inputAllTiles);
}

// Check if there is a spill between parent op and current op due to incompatible distributed type. This function is
// used to help VF op calculate related spilling status around its parent or user op
bool hasSpillDueToIncompatibleDistributedType(mlir::Operation* parentOp, mlir::Operation* currentOp,
                                              mlir::Value currentOpOperand) {
    if (isPureViewOp(parentOp)) {
        return false;
    }

    auto producerValue = findProducerValue(currentOpOperand);
    if (producerValue == nullptr || parentOp != producerValue.getOwner()) {
        return false;
    }

    auto outShapeSize = getShape(producerValue).totalSize();
    auto inShapeSize = getShape(currentOpOperand).totalSize();
    if (outShapeSize != inShapeSize) {
        return false;
    }

    auto distributedOutType =
            mlir::dyn_cast_if_present<VPU::DistributedTensorType>(getDistributedOutputType(parentOp, producerValue));
    if (distributedOutType == nullptr) {
        return false;
    }

    auto distributedInType =
            mlir::dyn_cast_or_null<VPU::DistributedTensorType>(getDistributedInputType(currentOp, currentOpOperand));
    if (distributedInType == nullptr) {
        return false;
    }
    return VPU::hasSpillDueToIncompatibleDistributionMode(distributedInType, distributedOutType);
}

bool hasSpill(mlir::Operation* parentOp, mlir::Operation* currentOp, mlir::Value currentOpOperand) {
    if (hasSpillDueToIncompatibleDistributedType(parentOp, currentOp, currentOpOperand)) {
        return true;
    }
    auto parentTiling = parentOp->getAttr(vpux::tilingStrategy);
    auto currentTiling = currentOp->getAttr(vpux::tilingStrategy);
    if (parentTiling == currentTiling) {
        return false;
    }
    return outputTileAxisIsSameAsMultiClusterStrategy(parentOp) ||
           inputTileAxisIsSameAsMultiClusterStrategy(currentOp, currentOpOperand);
}

StrategyCost extractBaselineCost(mlir::Operation* operation, ShapeRef tilingDims,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    auto vfSplit = getVFTilingSplit(tilingDims);

    OutputTiling tiles;

    auto hasTiling = !vfSplit.empty();

    // Determine VF context: operation may be inside a VF region or be a VF op itself
    auto parentVFOp = operation->getParentOfType<VPU::VerticalFusionOp>();
    if (parentVFOp == nullptr) {
        parentVFOp = mlir::dyn_cast<VPU::VerticalFusionOp>(operation);
    }
    std::optional<VFConfig> vfConfigOpt;
    if (parentVFOp != nullptr) {
        vfConfigOpt.emplace(parentVFOp);
    }

    if (hasTiling) {
        auto tiling = fillDividedTiles(operation, ShapeRef(tilingDims), getShape(operation->getResult(0)));
        VPUX_THROW_WHEN(mlir::failed(tiling), "Incorrect tiling {0} for vf {1}", tilingDims, operation->getLoc());
        tiles = tiling.value();
    }

    const auto costParameters = fillInCostParam(operation, tiles, {}, true);
    auto cost = costFunction->getStrategyCost(operation, costParameters);
    log.trace("Original VF {0} has strategy cost {1}", operation->getLoc(), cost);

    SmallVector<mlir::Value> operands = {operation->getOperand(0)};
    auto eltwiseLikeOp = false;
    if (operation->getNumOperands() > 1 && operation->template hasTrait<VPU::EltwiseOp>() &&
        operation->getOperand(0) != operation->getOperand(1)) {
        operands.emplace_back(operation->getOperand(1));
        eltwiseLikeOp = true;
    }

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);
    auto vfOperation = mlir::dyn_cast<VPU::VerticalFusionOpInterface>(operation);
    SmallVector<Dim> restrictedAxes;
    if (vfOperation != nullptr) {
        restrictedAxes = vfOperation.restrictedFusionAxes();
    }

    auto spilling = hasTiling &&
                    (isSpatialTiling(tilingDims) || nceOp == nullptr || nceOp.getWeightsOperand() == nullptr ||
                     (nceOp.getWeightsOperand() != nullptr &&
                      (restrictedAxes.empty() || llvm::find(restrictedAxes, Dims4D::Act::C) == restrictedAxes.end())));

    // Resolves an operand to the value outside the VF region for parent lookup
    auto resolveOuterOperand = [&](mlir::Value value) -> mlir::Value {
        if (vfConfigOpt.has_value()) {
            if (auto arg = getVFBlockArgument(value)) {
                return vfConfigOpt->getSubgraph().getOperand(arg.getArgNumber());
            }
        }
        return value;
    };

    auto* checkedOp = vfConfigOpt.has_value() ? vfConfigOpt->getSubgraph().getOperation() : operation;

    auto checkSpillParent = [&](mlir::Value value) -> bool {
        auto parentOperand = resolveOuterOperand(value);
        auto parentOp = findParent(parentOperand);
        if (parentOp == nullptr) {
            return false;
        }
        return !VF::v2::isCmxOperation(parentOp, false) || isPrevOperationEarlyScheduled(parentOp, checkedOp) ||
               hasBeforeDDRUsers(parentOp, checkedOp) || hasSpill(parentOp, checkedOp, parentOperand);
    };

    auto hasSpilledParents = llvm::any_of(operands, [&](mlir::Value value) {
        if (vfConfigOpt.has_value()) {
            // VF case: only block arguments can have external parents
            if (getVFBlockArgument(value) != nullptr) {
                return checkSpillParent(value);
            }
            return false;
        }
        // Non-VF case: check direct operand parent
        return checkSpillParent(value);
    });
    auto hasSpilledUsers =
            checkedOp->getUsers().empty() || hasOutputSpilledForDifferentDataSizeUses(checkedOp) ||
            llvm::any_of(findUses(checkedOp), [&checkedOp](auto* use) {
                return !VF::v2::isCmxOperation(use->getOwner(), true) ||
                       isPrevOperationEarlyScheduled(checkedOp, use->getOwner()) ||
                       hasSpill(checkedOp, use->getOwner(), use->getOwner()->getOperand(use->getOperandNumber()));
            });
    auto multiDimTilingWithChannelTiling = vfSplit.size() > 1 && vfSplit.find(Dims4D::Act::C) != vfSplit.end();
    auto spillReadWriteCanBeOverlapped = [&]() {
        if (costParameters._tiling.size() <= 1) {
            return false;
        }

        const auto arch = config::getArch(operation);
        if (!VPU::spillingCopyOpsCanBeOverlapped(arch)) {
            return false;
        }
        if (multiDimTilingWithChannelTiling) {
            return true;
        }

        if (auto tilingOpInterface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(operation)) {
            return tilingOpInterface.isSupportedTiling(tiles, TilingMode::PIPELINING, log);
        }
        return false;
    }();

    log.trace("Original VF {0} spill status: spill {1}, spill parent {2} spill user {3}", operation->getLoc(), spilling,
              hasSpilledParents, hasSpilledUsers);

    // Helper to get operation types for a given tile (uses VFConfig cache when available)
    std::function<SmallVector<NDTypeInterface>(const TileInfo&)> getOpTypes =
            [&](const TileInfo& tileInfo) -> SmallVector<NDTypeInterface> {
        if (vfConfigOpt.has_value()) {
            return vfConfigOpt->getOperationTypes(operation, tileInfo, {});
        }
        auto strategy = getMultiClusterStrategyFromOp(operation);
        return getTileTypes(operation, tileInfo, strategy);
    };

    SmallVector<StrategyCost> perTileSpillReadCost(costParameters._tiling.size());
    SmallVector<StrategyCost> perTileSpillWriteCost(costParameters._tiling.size());

    if (spilling || hasSpilledParents) {
        auto isOperandNotTiled = [&](auto operandIdx) {
            if (vfSplit.size() != 1 || vfSplit.find(Dims4D::Act::C) == vfSplit.end()) {
                return false;
            }
            auto tilingBuilderOp = mlir::dyn_cast_if_present<VPU::TilingBuilderOpInterface>(operation);
            if (tilingBuilderOp == nullptr || costParameters._tiling.empty()) {
                return false;
            }
            auto inTile = tilingBuilderOp.backInferTileInfo(costParameters._tiling.front(), log);
            return inTile.tiles.size() > operandIdx &&
                   inTile.tiles[operandIdx].shape == getShape(operation->getOperand(operandIdx));
        };
        for (auto operandValue : operands | indexed) {
            auto operand = operandValue.value();
            if (!spilling) {
                if (vfConfigOpt.has_value()) {
                    auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand);
                    if (arg == nullptr || !checkSpillParent(operand)) {
                        continue;
                    }
                } else {
                    if (!checkSpillParent(operand)) {
                        continue;
                    }
                }
            }
            auto getOperandType = [&](const TileInfo& tileInfo) -> vpux::NDTypeInterface {
                return getOpTypes(tileInfo)[operandValue.index()];
            };

            auto spillReadCostForCurOperand = costFunction->getSpillingReadCostsForAllTiles(
                    operation, costParameters, nullptr,
                    [&](mlir::Value item) {
                        return item == operand;
                    },
                    getOperandType);

            if (!spillReadWriteCanBeOverlapped) {
                if (operandValue.index() == 0 && isOperandNotTiled(operandValue.index())) {
                    cost += spillReadCostForCurOperand.front();
                } else {
                    cost += std::accumulate(spillReadCostForCurOperand.begin(), spillReadCostForCurOperand.end(),
                                            static_cast<StrategyCost>(0));
                }
            } else {
                llvm::transform(irange(spillReadCostForCurOperand.size()), perTileSpillReadCost.begin(),
                                [&](size_t index) {
                                    return spillReadCostForCurOperand[index] + perTileSpillReadCost[index];
                                });
            }
        }
    }

    if (spilling || hasSpilledUsers) {
        auto getOutputType = [&](const TileInfo& tileInfo) -> vpux::NDTypeInterface {
            auto types = getOpTypes(tileInfo);
            VPUX_THROW_WHEN(types.empty(), "Cannot get types for {0}", *operation);
            return types.back();
        };
        if (!spillReadWriteCanBeOverlapped) {
            cost += costFunction->getSpillingWriteCost(operation, costParameters, getOutputType);
        } else {
            perTileSpillWriteCost =
                    costFunction->getSpillingWriteCostsForAllTiles(operation, costParameters, getOutputType);
        }
    }
    if (spillReadWriteCanBeOverlapped) {
        if (multiDimTilingWithChannelTiling) {
            cost += perTileSpillReadCost.front() + perTileSpillWriteCost.back();
        } else {
            for (auto tileInd : irange(perTileSpillReadCost.size())) {
                if (tileInd == 0) {
                    cost += perTileSpillReadCost[tileInd];
                } else {
                    cost += std::max(perTileSpillReadCost[tileInd], perTileSpillWriteCost[tileInd - 1]);
                }
            }
            // Add the spilling write cost for the last tile's output
            if (!perTileSpillWriteCost.empty()) {
                cost += perTileSpillWriteCost.back();
            }
        }
    }

    if (!spilling && eltwiseLikeOp) {
        SmallVector<mlir::Operation*> parents;
        parents.reserve(operands.size());

        const auto getOutsideParents = [&](mlir::Value operand) {
            auto outerOperand = resolveOuterOperand(operand);
            auto parentOp = findParent(outerOperand);
            if (parentOp != nullptr) {
                parents.emplace_back(parentOp);
            }
        };

        llvm::for_each(operands, getOutsideParents);
        if (parents.size() <= 1) {
            return cost;
        }
        // check long term spill

        auto* parentLeft = parents.front();
        auto* parentRight = parents.back();

        if (parentLeft == nullptr || parentRight == nullptr) {
            return cost;
        }

        // Parents must be in the same block for ordering comparison
        if (parentLeft->getBlock() != parentRight->getBlock()) {
            return cost;
        }

        auto* earliestParent = parentLeft->isBeforeInBlock(parentRight) ? parentLeft : parentRight;
        auto* closestParent = earliestParent == parentLeft ? parentRight : parentLeft;

        SmallVector<mlir::Operation*> chain;
        const auto isParentOperation = [&]() {
            auto prevOp = closestParent;
            while ((prevOp = findParent(prevOp->getOperand(0)))) {
                if (prevOp == earliestParent) {
                    return true;
                }

                if (mlir::isa<VPU::TilingInfoOpInterface, VPU::VerticalFusionOp>(prevOp)) {
                    chain.emplace_back(prevOp);
                }

                if (mlir::isa_and_nonnull<Const::DeclareOp>(prevOp)) {
                    return false;
                }
            }

            return false;
        }();

        if (parentLeft != parentRight && !earliestParent->hasOneUse() &&
            VF::v2::isCmxOperation(earliestParent, false) && !mlir::isa_and_nonnull<Const::DeclareOp>(closestParent) &&
            isParentOperation && chain.size() > 1) {
            auto operandType = mlir::cast<vpux::NDTypeInterface>(earliestParent->getResult(0).getType());
            auto operandSize = operandType.getTotalAllocSize();
            if (auto distributedOutType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(
                        VPU::getDistributedOutputType(earliestParent, earliestParent->getResult(0)))) {
                operandSize = distributedOutType.getTotalAllocSize();
            }
            const auto hasLongSpilling = [&](mlir::Operation* op) {
                if (!VF::v2::isCmxOperation(op, true)) {
                    return false;
                }
                if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(op)) {
                    auto vfOperations = to_small_vector(vfOp.getBody()->getOps<VPU::VerticalFusionOpInterface>());
                    if (vfOperations.size() > 1) {
                        return false;
                    }

                    op = vfOperations.back().getOperation();
                }
                return getRequiredCMX(op, TileInfo(getShape(op->getResult(0))), log) + operandSize >
                       getTotalCMXFragmentationAwareSize(op);
            };
            if (hasLongSpilling(chain.back()) || hasLongSpilling(chain.front())) {
                auto getSecondOperandType = [&](const TileInfo& tileInfo) -> vpux::NDTypeInterface {
                    return getOpTypes(tileInfo)[1];
                };
                auto longSpillingCost =
                        costFunction->getSpillingReadCost(operation, costParameters, operands.back(),
                                                          getSecondOperandType) +
                        costFunction->getSpillingWriteCost(operation, costParameters, getSecondOperandType);

                log.trace("Original VF {0} has long spill cost {1}", operation->getLoc(), longSpillingCost);
                cost += longSpillingCost;
            }
        } else if (!isParentOperation && vfConfigOpt.has_value() &&
                   cmxSizeExceedForEltwiseOpWithSwOpUser(*vfConfigOpt, parents, log)) {
            // If the eltwise and its user exhaust entire CMX memory, we should
            // consider that there will very likely be dynamic spilling for its shared input
            auto getSecondOperandType = [&](const TileInfo& tileInfo) -> vpux::NDTypeInterface {
                return getOpTypes(tileInfo)[1];
            };
            auto inSpillingCost =
                    costFunction->getSpillingReadCost(operation, costParameters, operands.back(), getSecondOperandType);
            log.trace("Original VF {0} has shared input spill cost {1}", operation->getLoc(), inSpillingCost);
            cost += inSpillingCost;
        }
    }

    return cost;
}

}  // namespace vpux::VPU::VF::v2
