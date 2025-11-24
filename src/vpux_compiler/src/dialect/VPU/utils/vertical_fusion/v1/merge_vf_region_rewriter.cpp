//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/merge_vf_region_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/IRMapping.h>

#include <deque>

namespace vpux::VPU::VF::v1 {

std::optional<int64_t> findOptimalTilingStrategyInRange(const MergeVFRegionRewriter::IVFSchedulingPtr& scheduling,
                                                        const Dim dim, int64_t minNTiles, int64_t& maxNTiles,
                                                        std::unique_ptr<IVFAxisIncrement>& axisIncrement,
                                                        ArrayRef<int64_t> origTilingArray,
                                                        TilingOperationStorage::UPtr& minStorage,
                                                        TilingOperationStorage::UPtr& maxStorage, VFConfig& config,
                                                        Logger log) {
    std::optional<int64_t> result = std::nullopt;
    const auto origMaxTile = maxNTiles;
    auto nextValueFromMin = minNTiles;
    axisIncrement->increasedValue(nextValueFromMin, maxNTiles);
    SmallVector<int64_t> tilingMaxStrategy(origTilingArray.begin(), origTilingArray.end());
    SmallVector<int64_t> tilingArray(origTilingArray.begin(), origTilingArray.end());

    while (minNTiles < maxNTiles) {
        auto currentNTiles = axisIncrement->getMiddleValue(minNTiles, maxNTiles);

        if (maxNTiles == nextValueFromMin) {
            result = maxNTiles;
            if (maxNTiles == origMaxTile) {
                minStorage.reset(maxStorage.release());
            }
            break;
        }

        if (currentNTiles == minNTiles) {
            return std::nullopt;
        }

        tilingMaxStrategy[dim.ind()] = maxNTiles;
        tilingArray[dim.ind()] = currentNTiles;

        auto opStorage = std::make_unique<TilingOperationStorage>();
        auto getValidTilingStrategy = getMinimalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray,
                                                                             tilingMaxStrategy, dim, opStorage, log);
        if (mlir::failed(getValidTilingStrategy)) {
            return std::nullopt;
        }

        tilingArray = getValidTilingStrategy.value();
        currentNTiles = tilingArray[dim.ind()];
        result = currentNTiles;

        if (currentNTiles == maxNTiles) {
            break;
        }

        if (scheduling->validate(config, opStorage)) {
            maxNTiles = currentNTiles;
            minStorage.reset(opStorage.release());
        } else {
            minNTiles = currentNTiles;
        }

        nextValueFromMin = minNTiles;
        axisIncrement->increasedValue(nextValueFromMin, maxNTiles);
    }
    return result;
};

std::optional<int64_t> MergeVFRegionRewriter::getOptimalTilingStrategy(
        const IVFSchedulingPtr& scheduling, const Dim dim, const int64_t minTiles, int64_t& maxTiles,
        TilingOperationStorage::UPtr& minStorage, TilingOperationStorage::UPtr& maxStorage, VFConfig& config) const {
    if (minTiles > maxTiles || maxTiles == 1) {
        return std::nullopt;
    }

    auto minNTiles = minTiles;
    auto maxNTiles = maxTiles;

    std::optional<int64_t> result;
    auto outType = mlir::cast<vpux::NDTypeInterface>(config.getSubgraph()->getResult(0).getType());
    auto tilingArray = SmallVector<int64_t>(outType.getRank(), 1);
    tilingArray[dim.ind()] = minNTiles;
    if (minTiles == maxTiles) {
        if (minStorage == nullptr) {
            minStorage = std::make_unique<TilingOperationStorage>();
            auto tilingRegions = calculateTilingRegions(config.getSubgraph(), tilingArray, _log, minStorage);

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

    auto tilingMaxStrategy = SmallVector<int64_t>(outType.getRank(), 1);
    tilingMaxStrategy[dim.ind()] = maxNTiles;

    if (minStorage == nullptr) {
        minStorage = std::make_unique<TilingOperationStorage>();
        auto getValidStrategy = getMinimalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray,
                                                                       tilingMaxStrategy, dim, minStorage, _log);

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
        maxStorage = std::make_unique<TilingOperationStorage>(config.getOperationsForTiling(), maxNTiles);
        // When maxNTiles is too large,  to avoid spending too much time on calculating, try to check if the cube root
        // of the max tile is valid or not.
        mlir::FailureOr<SmallVector<int64_t>> getValidStrategy = mlir::failure();
        auto cbrtMaxTile = getCbrtMaxTileCandidate(minNTiles, maxNTiles, axisIncrement);
        if (cbrtMaxTile.has_value()) {
            auto tilingCbrtMaxStrategy = tilingMaxStrategy;
            tilingCbrtMaxStrategy[dim.ind()] = cbrtMaxTile.value();
            getValidStrategy = getMaximalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray,
                                                                      tilingCbrtMaxStrategy, dim, maxStorage, _log);

            auto useCbrtMaxTileStrategy = mlir::succeeded(getValidStrategy) && scheduling->validate(config, maxStorage);
            if (useCbrtMaxTileStrategy) {
                maxNTiles = getValidStrategy.value()[dim.ind()];
                result = findOptimalTilingStrategyInRange(scheduling, dim, minNTiles, maxNTiles, axisIncrement,
                                                          tilingArray, minStorage, maxStorage, config, _log);
                maxStorage.reset();
                return result;
            }
        }

        maxStorage.reset();
        getValidStrategy = getMaximalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray, tilingMaxStrategy,
                                                                  dim, maxStorage, _log);
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
                                            minStorage, maxStorage, config, _log);
}

StrategyCost MergeVFRegionRewriter::extractVFCost(VFConfig& vfConfig) const {
    auto vfOp = vfConfig.getSubgraph();
    auto tilingDims = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategyAttr());

    const auto dim = getVFTilingDim(tilingDims);
    auto operations = vfConfig.getOperationsForTiling();
    if (operations.empty()) {
        return 0;
    }

    if (!dim.has_value() || operations.size() == 1) {
        OutputTiling tiles;
        auto* operation = operations.front();
        if (dim.has_value()) {
            auto tiling = fillDividedTiles(operation, ShapeRef(tilingDims), getShape(operation->getResult(0)));
            VPUX_THROW_WHEN(mlir::failed(tiling), "Incorrect tiling {0} for vf {1}", tilingDims, vfOp);
            tiles = tiling.value();
        }

        const auto costParameters = fillInCostParam(operation, tiles, {}, _enablePrefetchTiling);
        auto cost = _vpunnCostFunction->getStrategyCost(operation, costParameters);

        SmallVector<mlir::Value> operands = {operation->getOperand(0)};
        if (operation->getNumOperands() > 1 && operation->template hasTrait<VPU::EltwiseOp>() &&
            operation->getOperand(0) != operation->getOperand(1)) {
            operands.emplace_back(operation->getOperand(1));
        }

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);

        auto spilling = dim.has_value() &&
                        (isSpatialTiling(tilingDims) || (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr));
        auto hasSpilledParents = llvm::any_of(operands, [&](mlir::Value value) {
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
                auto parentOperand = vfConfig.getSubgraph().getOperand(arg.getArgNumber());
                auto parentOp = findParent(parentOperand);
                return !isCmxOperation(parentOp, false) ||
                       isPrevOperationEarlyScheduled(parentOp, vfConfig.getSubgraph());
            }
            return false;
        });
        auto hasSpilledUsers =
                vfConfig.getSubgraph()->getUsers().empty() ||
                llvm::any_of(findUses(vfConfig.getSubgraph()), [&vfConfig](auto* use) {
                    return !isCmxOperation(use->getOwner(), true) ||
                           isPrevOperationEarlyScheduled(vfConfig.getSubgraph().getOperation(), use->getOwner());
                });

        auto spillReadWriteCanBeOverlapped = [&]() {
            if (operations.size() != 1 || costParameters._tiling.size() <= 1) {
                return false;
            }

            const auto arch = config::getArch(operation);
            if (!VPU::spillingCopyOpsCanBeOverlapped(arch)) {
                return false;
            }
            if (auto tilingOpInterface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(operation)) {
                return tilingOpInterface.isSupportedTiling(tiles, TilingMode::PIPELINING, _log);
            }
            return false;
        }();

        SmallVector<StrategyCost> perTileSpillReadCost(costParameters._tiling.size());
        SmallVector<StrategyCost> perTileSpillWriteCost(costParameters._tiling.size());

        if (spilling || hasSpilledParents) {
            for (auto operandValue : operands | indexed) {
                auto operand = operandValue.value();
                auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand);
                if (!spilling && arg != nullptr) {
                    auto parentOp = vfConfig.getSubgraph().getOperand(arg.getArgNumber()).getDefiningOp();
                    if (isCmxOperation(parentOp, false) &&
                        !isPrevOperationEarlyScheduled(parentOp, vfConfig.getSubgraph())) {
                        continue;
                    }
                }
                auto getOperandType = [&](const auto& tileInfo) {
                    return vfConfig.getOperationTypes(operation, tileInfo, {})[operandValue.index()];
                };

                if (!spillReadWriteCanBeOverlapped) {
                    cost += _vpunnCostFunction->getSpillingReadCost(operation, costParameters, operand, getOperandType);
                } else {
                    auto spillReadCostForCurOperand = _vpunnCostFunction->getSpillingReadCostsForAllTiles(
                            operation, costParameters, nullptr,
                            [&](mlir::Value item) {
                                return item == operand;
                            },
                            getOperandType);
                    llvm::transform(irange(spillReadCostForCurOperand.size()), perTileSpillReadCost.begin(),
                                    [&](size_t index) {
                                        return spillReadCostForCurOperand[index] + perTileSpillReadCost[index];
                                    });
                }
            }
        }

        if (spilling || hasSpilledUsers) {
            auto getOutputType = [&](const auto& tileInfo) {
                auto types = vfConfig.getOperationTypes(operation, tileInfo, {});
                VPUX_THROW_WHEN(types.empty(), "Cannot get types for {0}", *operation);
                return types.back();
            };
            if (!spillReadWriteCanBeOverlapped) {
                cost += _vpunnCostFunction->getSpillingWriteCost(operation, costParameters, getOutputType);
            } else {
                perTileSpillWriteCost =
                        _vpunnCostFunction->getSpillingWriteCostsForAllTiles(operation, costParameters, getOutputType);
            }
        }
        if (spillReadWriteCanBeOverlapped) {
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

        return cost;
    }

    auto vfCase = VFCase(vfConfig, dim.value());
    vfCase.setTilingNumber(tilingDims[dim.value().ind()]);

    auto scenario = detectScenario(vfConfig);

    vfCase.setScheduling(std::move(scenario));
    return vfCase.getCost(_vpunnCostFunction, _log);
}

bool MergeVFRegionRewriter::canMergeVFOpsWithoutCostCheck(VFCase& mergedCase) const {
    auto& vfConfig = mergedCase.getConfig();
    if (mergedCase.getTilingNumber() == 1 && vfConfig.isPotentiallyPipelined()) {
        mergedCase.approveScheduling();
        return true;
    }
    return false;
}

bool MergeVFRegionRewriter::canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const {
    if (opsNeedTiling || vfConfig.isPipelined()) {
        return false;
    }
    const auto filterNCENotEltwiseLike = [](mlir::Operation* op) {
        return mlir::isa<VPU::NCEOpInterface>(op) && !op->template hasTrait<VPU::EltwiseOp>();
    };
    const auto filterSWKernels = [](mlir::Operation* op) {
        return mlir::isa<VPU::SWOpInterface>(op);
    };
    // when pipeline case is generic this check is enough to prevent VF
    // but now we check additionally that there are no operations
    // with different executors
    auto checkedOperations = vfConfig.getOperationsForTiling();
    return (llvm::all_of(checkedOperations, filterNCENotEltwiseLike) ||
            llvm::all_of(checkedOperations, filterSWKernels));
}

std::deque<MergeVFRegionRewriter::IVFSchedulingPtr> MergeVFRegionRewriter::getVFSchedulingChecks(
        VFConfig& config) const {
    std::deque<IVFSchedulingPtr> vfChecks;
    VFSchedulingFactory vfFactory(_enablePrefetchTiling);

    auto minimalCheck = vfFactory.createVFScenario(VFScenario::MINIMAL, _log);

    if (config.isPipelined()) {
        auto pipeliningChecks = vfFactory.createVFScenario(VFScenario::VF_PIPELINING, _log);
        minimalCheck->addNext(std::move(pipeliningChecks));
    }

    auto prefetchingCheck = vfFactory.createVFScenario(VFScenario::LASTOP_PREFETCHING, _log);
    auto weightsCheck = vfFactory.createVFScenario(VFScenario::WEIGHTS_PREFETCHING, _log);
    auto fullPrefetching = vfFactory.createVFScenario(VFScenario::FULL_PREFETCHING, _log);
    weightsCheck->addNext(std::move(fullPrefetching));
    prefetchingCheck->addNext(std::move(weightsCheck));
    minimalCheck->addNext(std::move(prefetchingCheck));

    vfChecks.emplace_back(std::move(minimalCheck));

    return vfChecks;
}

MergeVFRegionRewriter::IVFSchedulingPtr MergeVFRegionRewriter::detectScenario(VFConfig& vfConfig) const {
    VFSchedulingFactory costFactory(_enablePrefetchTiling);
    auto scenarioKind = vfConfig.getSubgraph().getScenario().has_value() ? vfConfig.getSubgraph().getScenario().value()
                        : _enablePrefetchTiling                          ? VFScenario::WEIGHTS_PREFETCHING
                                                                         : VFScenario::MINIMAL;
    return costFactory.createVFScenario(scenarioKind, _log);
}

std::optional<VFCase> MergeVFRegionRewriter::findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                                          VPU::VerticalFusionOp currentOp) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    VPUX_THROW_WHEN(currentTiling.size() != prevTiling.size(),
                    "Tiling info rank of current block {0} is not equal to tiling info rank of previous block {1}",
                    currentTiling.size(), prevTiling.size());
    VFConfig currentConfig(currentOp, _enableVerticalFusionPipelining);
    VFConfig prevConfig(prevOp, _enableVerticalFusionPipelining);

    auto curAxis = getVFTilingDim(currentTiling, currentConfig.getVFOperations());
    auto prevAxis = getVFTilingDim(prevTiling, prevConfig.getVFOperations());

    if (mlir::failed(curAxis) || mlir::failed(prevAxis)) {
        return std::nullopt;
    }

    bool curHasTiling = hasTiling(currentTiling);
    bool prevHasTiling = hasTiling(prevTiling);
    // in case both subgraphs have tiling, check if they match
    // if there is only one subgraph with tiling, check if it's allowed
    // to tile second one with such axis
    // if both doesn't have tiling, check if there is at least one
    // allowed axis for both of them
    VFConfig vfConfig(mergedOp, _enableVerticalFusionPipelining, prevHasTiling, curHasTiling);

    bool opsNeedTiling = prevHasTiling || curHasTiling;
    if (canSkipMergeVF(vfConfig, opsNeedTiling)) {
        return std::nullopt;
    }

    // Record the operation and its corresponding tiling dim when back-infer subgraph
    std::unordered_map<mlir::Operation*, vpux::Dim> opDimMap;
    // Only for current VF Op check to skip restricted dims
    // E.g., VF{conv} -> VF{conv}, the first VF can support CTiling, but the second cannot
    const auto isRegionRestrictedDim = [&](const std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
        // Skip the case of ConvolutionOp whose weights has split over output channel tiling strategy
        if (isTileOverOutputChannel(currentConfig)) {
            return false;
        }

        for (auto* operation : currentConfig.getOperationsForTiling()) {
            auto vfOperation = mlir::cast<VPU::VerticalFusionOpInterface>(operation);
            auto restrictedAxes = vfOperation.restrictedFusionAxes();
            if (restrictedAxes.empty()) {
                continue;
            }

            if (llvm::find(currentConfig.getInputs(), operation) != currentConfig.getInputs().end()) {
                // skip inputs which has no connection with previous operation
                if (llvm::none_of(operation->getOperands(), [&](mlir::Value value) {
                        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
                            return currentOp.getOperand(argument.getArgNumber()).getDefiningOp() == prevOp;
                        }
                        return false;
                    })) {
                    continue;
                }
            }
            VPUX_THROW_WHEN(opDimMap.find(operation) == opDimMap.end(), "Operation {0} is not in the map",
                            operation->getLoc());
            auto dim = opDimMap.at(operation);
            if (llvm::find(restrictedAxes, dim) != restrictedAxes.end()) {
                return true;
            }
        }
        return false;
    };

    auto vfSchedulingChecks = getVFSchedulingChecks(vfConfig);

    VPU::VFSubgraphUserSetter setter(currentOp, mergedOp);

    auto getVFCaseWithTiling = [&](const Dim curDim, const Dim prevDim) {
        auto maxTiles = getTilingLimit(curDim, vfConfig.getVFOperations());
        auto minTiles = std::max(currentTiling[curDim.ind()], prevTiling[prevDim.ind()]);

        VFCase mergedCase(vfConfig, curDim);

        auto schedulingChecks = vfSchedulingChecks;

        TilingOperationStorage::UPtr maxStorage = nullptr;
        TilingOperationStorage::UPtr minStorage = nullptr;

        while (!schedulingChecks.empty()) {
            auto currentCheck = schedulingChecks.front();
            schedulingChecks.pop_front();
            auto numTiles = getOptimalTilingStrategy(currentCheck, curDim, minTiles, maxTiles, minStorage, maxStorage,
                                                     vfConfig);

            if (numTiles.has_value()) {
                mergedCase.setTilingNumber(numTiles.value());
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

        return mergedCase;
    };

    const auto linkNumber = getLinkNumber(currentOp, prevOp);
    std::optional<Dim> checkedDim;
    if (curHasTiling && prevHasTiling) {
        auto curInputAxesResult = backInferVFTilingDim(currentConfig, curAxis.value(), opDimMap);
        VPUX_THROW_UNLESS(mlir::succeeded(curInputAxesResult),
                          "Cannot backinfer tiling dim for current VF {0} with axis {1}", currentOp, curAxis.value());
        auto curInputAxes = curInputAxesResult.value();
        if (curInputAxes[linkNumber] == prevAxis.value() && !isRegionRestrictedDim(opDimMap)) {
            auto areAllAligned = llvm::all_of(vfConfig.getOperationsForTiling(), [](auto* operation) {
                return mlir::isa<IE::AlignedChannelsOpInterface>(operation);
            });
            if (prevAxis.value() != Dims4D::Act::C || !areAllAligned) {
                // try to use current axis, otherwise try to find other axis
                auto mergedCase = getVFCaseWithTiling(curAxis.value(), prevAxis.value());
                checkedDim = curAxis.value();
                if (mergedCase.isInitialized()) {
                    return mergedCase;
                }
            }
        }
    }

    DimArr allowedDims = getAllowedDims(vfConfig.getVFOperations(), _log);
    if (allowedDims.empty()) {
        return std::nullopt;
    }

    StrategyCost bestCost = std::numeric_limits<StrategyCost>::max();
    std::optional<VFCase> mergedCase = std::nullopt;
    for (auto dim : allowedDims) {
        // in order not to check twice dim which has been handled unsuccessfully
        if (checkedDim.has_value() && checkedDim.value() == dim) {
            continue;
        }
        // E.g., prevTiling [1, 3, 1, 1] -> permuteCast -> currentTiling [1, 1, 2, 1]
        // Thus we need dim backinfer to get correct axis to compare
        // As Vf inputs may be more than one, we need backinfer dim for each of them and use correct one
        auto curInputDimsResult = backInferVFTilingDim(currentConfig, dim, opDimMap);
        VPUX_THROW_UNLESS(mlir::succeeded(curInputDimsResult),
                          "Cannot backinfer tiling dim for current VF {0} with axis {1}", currentOp, dim);
        auto curInputDims = curInputDimsResult.value();

        if (isRegionRestrictedDim(opDimMap)) {
            continue;
        }

        auto currentVFCase = getVFCaseWithTiling(dim, curInputDims[linkNumber]);

        // calculate optimal number of tiles for that dim
        if (!currentVFCase.isInitialized()) {
            continue;
        }

        // get vpunncost
        StrategyCost cost = currentVFCase.getCost(_vpunnCostFunction, _log.nest());
        // compare cost, choose best strategy
        if (cost < bestCost) {
            bestCost = cost;
            mergedCase = std::move(currentVFCase);
        }
    }
    return mergedCase;
}

mlir::LogicalResult MergeVFRegionRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Starting vertical fusion for region with VerticalFusionOp {0} at location {1}", vfOp, vfOp->getLoc());

    VPU::VerticalFusionOp vfBlock = nullptr;
    VPU::VerticalFusionOp parentVFOp = nullptr;
    for (auto operand : vfOp->getOperands()) {
        parentVFOp = operand.getDefiningOp<VPU::VerticalFusionOp>();
        vfBlock = nullptr;

        if (parentVFOp == nullptr) {
            continue;
        }

        _log.trace("Analyzing vertical fusion region with parent VerticalFusionOp {0} at location {1}", parentVFOp,
                   parentVFOp->getLoc());

        const bool allInOldBlock = llvm::all_of(parentVFOp->getUsers(), [&](auto user) {
            return user == vfOp;
        });
        // if not all user of current parent VF go to the same block
        // check if all users are waiting for to be merged with same VF
        // For situations
        // Operation1
        //   |      |
        //   |     Operation2
        //     Eltwise
        // in case Operation1's user goes to Operation2, which can be fused with Eltwise
        // switch to Operation2, try to merge it first and then come back later to this case
        if (!allInOldBlock) {
            continue;
        }

        vfBlock = fuseOpsInBlock(rewriter, vfOp, parentVFOp.getOperation());
        auto vfCase = findVFCase(parentVFOp, vfOp, vfBlock);
        if (!vfCase.has_value() || !checkVFCostFunction(parentVFOp, vfOp, vfCase.value())) {
            // Drop all references to vfBlock to avoid it being added back to the rewriter
            // worklist.
            vfBlock->dropAllReferences();
            rewriter.eraseOp(vfBlock);
            vfBlock = nullptr;
            // Add support for NCE task, if merging activation failed, continue to merge weights.
            // E-141686: A general solution to merge more subgraph for more VF ops.
            if (checkOtherVFInput(vfOp, parentVFOp)) {
                continue;
            }
            return mlir::failure();
        }

        break;
    }

    if (vfBlock == nullptr) {
        return mlir::failure();
    }

    _log.trace("Merged subgraph {0}", vfBlock);
    fuseBlocks(rewriter, vfOp, vfBlock);

    return mlir::success();
}
}  // namespace vpux::VPU::VF::v1
