//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_vf_region_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPU::VF::v2 {

// Check if there is a spill between parent op and current op due to incompatible distributed type. This function is
// used to help VF op calculate related spilling status around its parent or user op
bool hasSpillDueToIncompatibleDistributedType(mlir::Operation* parentOp, mlir::Operation* currentOp,
                                              mlir::Value currentOpOperand) {
    if (isPureViewOp(parentOp)) {
        return false;
    }

    if (parentOp != findParent(currentOpOperand)) {
        return false;
    }
    auto outShapeSize = getShape(parentOp->getResult(0)).totalSize();
    auto inShapeSize = getShape(currentOpOperand).totalSize();
    if (outShapeSize != inShapeSize) {
        return false;
    }

    auto distributedOutType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(getDistributedOutputType(parentOp));
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

bool tileOnSameDims(const VFSplit& curVFSplit, const VFSplit& preVFSplit, const int64_t linkNumber,
                    VFConfig& currentConfig, std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
    if (curVFSplit.size() != preVFSplit.size()) {
        return false;
    }
    for (const auto& item : curVFSplit) {
        auto curTilingDim = item.first;
        auto curInputAxesResult = VPU::backInferVFTilingDim(currentConfig, curTilingDim, opDimMap);
        if (mlir::failed(curInputAxesResult)) {
            return false;
        }
        auto curInputAxes = curInputAxesResult.value();
        const auto isTiledOnPreVF = preVFSplit.find(curInputAxes[linkNumber]) != preVFSplit.end();
        if (!isTiledOnPreVF) {
            return false;
        }
    }
    return true;
}

SmallVector<VFSplit> getSplitFromDimArr(DimArrRef dimsToCheck, DimArrRef allowedDims, VFConfig& vfConfig) {
    SmallVector<VFSplit> splits;
    for (auto dim : dimsToCheck) {
        VFSplit singleSplit = {{dim, std::nullopt}};
        splits.emplace_back(singleSplit);

        if (dim.ind() <= Dims4D::Act::C.ind()) {
            // Only enable 2D tiling for H and W
            continue;
        }
        for (auto otherDim : allowedDims) {
            if (dim.ind() > otherDim.ind() && otherDim.ind() > Dims4D::Act::C.ind()) {
                VFSplit doubleSplit = {{otherDim, getTilingLimit(otherDim, vfConfig, true)}, {dim, std::nullopt}};
                splits.emplace_back(doubleSplit);
            }
        }
    }
    return splits;
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
        _log.trace("Original VF {0} has strategy cost {1}", operation->getLoc(), cost);

        SmallVector<mlir::Value> operands = {operation->getOperand(0)};
        auto eltwiseLikeOp = false;
        if (operation->getNumOperands() > 1 && operation->template hasTrait<VPU::EltwiseOp>() &&
            operation->getOperand(0) != operation->getOperand(1)) {
            operands.emplace_back(operation->getOperand(1));
            eltwiseLikeOp = true;
        }

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);
        auto vfOperation = mlir::cast<VPU::VerticalFusionOpInterface>(operation);
        auto restrictedAxes = vfOperation.restrictedFusionAxes();

        auto spilling =
                dim.has_value() &&
                (isSpatialTiling(tilingDims) || nceOp == nullptr || nceOp.getWeightsOperand() == nullptr ||
                 (nceOp.getWeightsOperand() != nullptr &&
                  (restrictedAxes.empty() || llvm::find(restrictedAxes, Dims4D::Act::C) == restrictedAxes.end())));

        auto checkSpillParent = [&](mlir::BlockArgument arg) {
            auto parentOperand = vfConfig.getSubgraph().getOperand(arg.getArgNumber());
            auto parentOp = findParent(parentOperand);
            return !VF::v2::isCmxOperation(parentOp, false) ||
                   isPrevOperationEarlyScheduled(parentOp, vfConfig.getSubgraph()) ||
                   hasBeforeDDRUsers(parentOp, vfConfig.getSubgraph()) ||
                   hasSpill(parentOp, vfConfig.getSubgraph(), parentOperand);
        };

        auto hasSpilledParents = llvm::any_of(operands, [&](mlir::Value value) {
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
                return checkSpillParent(arg);
            }
            return false;
        });
        auto hasSpilledUsers =
                vfConfig.getSubgraph()->getUsers().empty() ||
                hasOutputSpilledForDifferentDataSizeUses(vfConfig.getSubgraph()) ||
                llvm::any_of(findUses(vfConfig.getSubgraph()), [&vfConfig](auto* use) {
                    return !VF::v2::isCmxOperation(use->getOwner(), true) ||
                           isPrevOperationEarlyScheduled(vfConfig.getSubgraph().getOperation(), use->getOwner()) ||
                           hasSpill(vfConfig.getSubgraph(), use->getOwner(),
                                    use->getOwner()->getOperand(use->getOperandNumber()));
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

        _log.trace("Original VF {0} spill status: spill {1}, spill parent {2} spill user {3}", operation->getLoc(),
                   spilling, hasSpilledParents, hasSpilledUsers);

        SmallVector<StrategyCost> perTileSpillReadCost(costParameters._tiling.size());
        SmallVector<StrategyCost> perTileSpillWriteCost(costParameters._tiling.size());

        if (spilling || hasSpilledParents) {
            for (auto operandValue : operands | indexed) {
                auto operand = operandValue.value();
                auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand);
                if (!spilling && arg != nullptr) {
                    if (!checkSpillParent(arg)) {
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

        if (!spilling && eltwiseLikeOp) {
            SmallVector<mlir::Operation*> parents;
            parents.reserve(operands.size());

            const auto getOutsideParents = [&](mlir::Value operand) {
                if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                    auto parentOperand = vfConfig.getSubgraph().getOperand(arg.getArgNumber());
                    auto parentOp = findParent(parentOperand);

                    if (parentOp != nullptr) {
                        parents.emplace_back(parentOp);
                    }
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
                VF::v2::isCmxOperation(earliestParent, false) &&
                !mlir::isa_and_nonnull<Const::DeclareOp>(closestParent) && isParentOperation && chain.size() > 1) {
                auto operandType = mlir::cast<vpux::NDTypeInterface>(earliestParent->getResult(0).getType());
                auto operandSize = operandType.getTotalAllocSize();
                if (auto distributedOutType = VPU::getDistributedOutputType(earliestParent)) {
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
                    return getRequiredCMX(op, TileInfo(getShape(op->getResult(0))), _log) + operandSize >
                           getTotalCMXFragmentationAwareSize(op);
                };
                if (hasLongSpilling(chain.back()) || hasLongSpilling(chain.front())) {
                    auto longSpillingCost = _vpunnCostFunction->getSpillingReadCost(
                                                    operation, costParameters, operands.back(),
                                                    [&](const auto& tileInfo) {
                                                        return vfConfig.getOperationTypes(operation, tileInfo, {})[1];
                                                    }) +
                                            _vpunnCostFunction->getSpillingWriteCost(
                                                    operation, costParameters, [&](const auto& tileInfo) {
                                                        return vfConfig.getOperationTypes(operation, tileInfo, {})[1];
                                                    });

                    _log.trace("Original VF {0} has long spill cost {1}", operation->getLoc(), longSpillingCost);
                    cost += longSpillingCost;
                }
            } else if (!isParentOperation && cmxSizeExceedForEltwiseOpWithSwOpUser(vfConfig, parents, _log)) {
                // If the eltwise and its user exhaust entire CMX memory, we should
                // consider that there will very likely be dynamic spilling for its shared input
                auto inSpillingCost = _vpunnCostFunction->getSpillingReadCost(
                        operation, costParameters, operands.back(), [&](const auto& tileInfo) {
                            return vfConfig.getOperationTypes(operation, tileInfo, {})[1];
                        });
                _log.trace("Original VF {0} has shared input spill cost {1}", operation->getLoc(), inSpillingCost);
                cost += inSpillingCost;
            }
        }

        return cost;
    }

    auto vfSplit = getVFTilingSplit(tilingDims);
    auto vfCase = VFCase(vfConfig, vfSplit);

    auto scenario = detectScenario(vfConfig);

    vfCase.setScheduling(std::move(scenario));
    return vfCase.getCost(_vpunnCostFunction, _log);
}

bool MergeVFRegionRewriter::cmxSizeExceedForEltwiseOpWithSwOpUser(VFConfig& currentConfig,
                                                                  ArrayRef<mlir::Operation*> parents,
                                                                  Logger log) const {
    /*
        Check the pattern below:
                         ParentVF0
                           /   \
                 EltwiseOp    SiblingOp
                     |
                   SWOp
                     |
    The execution order of EltWiseOp, SiblingOp and SWOp will be EltwiseOp -> SwOp -> SiblingOp, in which SiblingOp is
    expected to be overlapped with SwOp, but if may result in dynamic spilling when the cmx size of EltwiseOp and SWOp
    is greater than the available CMX Size.
    */
    auto currentVFOp = currentConfig.getSubgraph();
    auto uses = findUses(currentVFOp);
    if (uses.size() != 1) {
        return false;
    }
    auto userVFOp = mlir::dyn_cast_or_null<VPU::VerticalFusionOp>((*uses.begin())->getOwner());
    if (userVFOp == nullptr) {
        return false;
    }
    VFConfig userConfig(userVFOp, _enableVerticalFusionPipelining);
    auto swOpUser = mlir::dyn_cast<VPU::SWOpInterface>(userConfig.getOperationsForTiling().front());
    if (swOpUser == nullptr) {
        return false;
    }
    auto parentHasMultiUses = llvm::any_of(parents, [&](auto* parent) {
        auto parentUses = findUses(parent);
        auto otherUserCount = llvm::count_if(parentUses, [&](auto* use) {
            auto userOp = use->getOwner();
            return userOp != nullptr && userOp != currentVFOp && VF::v2::isCmxOperation(userOp, false);
        });
        return otherUserCount > 0;
    });
    if (!parentHasMultiUses) {
        return false;
    }

    const auto currentTiling = parseIntArrayAttr<int64_t>(currentVFOp.getTilingStrategy());
    const auto userTiling = parseIntArrayAttr<int64_t>(userVFOp.getTilingStrategy());
    auto hasTiling = [&](const auto& tiling) {
        return llvm::any_of(tiling, [](auto i) {
            return i != 1;
        });
    };
    if (hasTiling(currentTiling) || hasTiling(userTiling)) {
        // Skipp this complex scenario
        return false;
    }

    auto eltwiseOp = currentConfig.getOperationsForTiling().front();
    auto getUsedSize = [&](mlir::Operation* operation) {
        auto usedSize = getRequiredCMX(operation, TileInfo(getShape(operation->getResult(0))), log);
        return usedSize;
    };
    auto types = getTileTypes(eltwiseOp, TileInfo(getShape(eltwiseOp->getResult(0))));
    auto sharedInputSize = types.front().getTotalAllocSize();
    auto totalAvailableCMXSize = getTotalCMXVFPipelineFragmentationAwareSize(currentVFOp);
    // Caclulate the required size for the eltwise op and swOp user
    auto usedSize = getUsedSize(swOpUser) + sharedInputSize;
    return usedSize > totalAvailableCMXSize;
}

bool MergeVFRegionRewriter::canMergeVFOpsWithoutCostCheck(VFCase&) const {
    return false;
}

bool MergeVFRegionRewriter::canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const {
    return !opsNeedTiling && !vfConfig.isPipelined();
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

    auto curVFSplit = getVFTilingSplit(currentTiling);
    auto preVFSplit = getVFTilingSplit(prevTiling);

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

    bool checkIfNextMergeBetter = false;
    if (!vfConfig.isPipelined() && currentOp->hasOneUse() &&
        mlir::isa<VPU::VerticalFusionOp>(*currentOp->getUsers().begin())) {
        checkIfNextMergeBetter = true;
    }

    // If mergedOp can not pipeline, but currentOp + userOp can pipeline, and mergedOp's tile dim is userOp's
    // restricted axes, then we will block the mergeOp. For example Conv + Add + Softmax, the three operations
    // can not be vertically fused due to restricted axes on Softmax. Conv + Add can not pipeline while
    // Add + Softmax can. We prefer Add vertical fusion with Softmax.
    const auto isNextMergeCanBePipelined = [&](const std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
        auto nextVFOp = mlir::cast<VPU::VerticalFusionOp>(*mergedOp->getUsers().begin());

        llvm::SetVector<mlir::Operation*> operations;
        const auto currentOps = currentConfig.getOperationsForTiling();
        operations.insert(currentOps.begin(), currentOps.end());

        VFConfig nextConfig(nextVFOp, _enableVerticalFusionPipelining);
        const auto nextOps = nextConfig.getOperationsForTiling();
        operations.insert(nextOps.begin(), nextOps.end());

        VFConfig mergeCurrWithNextConfig(operations);
        if (!mergeCurrWithNextConfig.isPipelined()) {
            return false;
        }

        for (auto* operation : nextConfig.getOperationsForTiling()) {
            auto vfOperation = mlir::cast<VPU::VerticalFusionOpInterface>(operation);
            auto restrictedAxes = vfOperation.restrictedFusionAxes();
            if (restrictedAxes.empty()) {
                continue;
            }
            if (llvm::find(restrictedAxes, opDimMap.at(currentConfig.getOutputs().back())) != restrictedAxes.end()) {
                return true;
            }
        }
        return false;
    };

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

    auto vfSchedulingChecks = getSchedulingScenarios(vfConfig, _log);
    const auto linkNumber = getLinkNumber(currentOp, prevOp);

    const auto getMinimalNumber = [&](auto dim, const VFSplit& split) -> int64_t {
        if (split.size() == 1) {
            // 1D tiling
            auto curInputAxes = backInferVFTilingDim(currentConfig, dim, opDimMap);
            return std::max(currentTiling[dim.ind()], prevTiling[curInputAxes.value()[linkNumber].ind()]);
        } else {
            // 2D tiling
            return MINIMUM_LENGTH_TILING;
        }
    };

    const auto getMaximalNumber = [&](auto dim, const VFSplit& split) -> int64_t {
        auto maxTiles = getTilingLimit(dim, vfConfig);
        if (split.size() > 1) {
            // 2D tiling
            auto otherDimSum = getVFTilesLen(split);
            maxTiles = divUp(maxTiles, otherDimSum);
        }
        return maxTiles;
    };

    VPU::VFSubgraphUserSetter setter(currentOp, mergedOp);

    DimArr allowedDims = getAllowedDims(vfConfig.getVFOperations().getArrayRef(), _log);
    if (allowedDims.empty()) {
        return std::nullopt;
    }

    DimArr dimsToCheck;
    if (tileOnSameDims(curVFSplit, preVFSplit, linkNumber, currentConfig, opDimMap)) {
        // If the current and previous VF splits are on the same dimensions, we can try to check the common dimensions
        // first
        for (auto& item : curVFSplit) {
            dimsToCheck.push_back(item.first);
        }
    } else {
        // Otherwise, we check all allowed dimensions
        dimsToCheck = allowedDims;
    }

    auto getVFCaseFromSplits = [&](ArrayRef<VFSplit> splits) -> std::optional<VFCase> {
        StrategyCost bestCost = std::numeric_limits<StrategyCost>::max();
        std::optional<VFCase> mergedCase = std::nullopt;
        for (auto split : splits) {
            auto dim = split.rbegin()->first;
            if (mlir::failed(backInferVFTilingDim(currentConfig, dim, opDimMap))) {
                continue;
            }

            // Skip current merge if a better (pipelined) merge with the next VF block is possible.
            if (checkIfNextMergeBetter && isNextMergeCanBePipelined(opDimMap)) {
                continue;
            }

            if (isRegionRestrictedDim(opDimMap)) {
                continue;
            }

            auto currentVFCase = VPU::VF::v2::getVFCaseWithTiling(vfConfig, dim, split, getMinimalNumber,
                                                                  getMaximalNumber, _log, vfSchedulingChecks);

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
    };

    auto splits = getSplitFromDimArr(dimsToCheck, allowedDims, vfConfig);
    auto mergedCase = getVFCaseFromSplits(splits);
    if (mergedCase.has_value()) {
        return mergedCase;
    }

    // If no valid case found, try to check dims that are not in dimsToCheck. For example, if the current VF and
    // previous VF has tiled on same dimensions W. Then the allowedDims will only contains dimW instead. If merge on
    // dimW is not optimal, the compiler can still have the change to merge on other supported dimensions like dimH,
    // dimH&dimW, etc.
    DimArr restAllowedDims;
    llvm::copy_if(allowedDims, std::back_inserter(restAllowedDims), [&](const Dim& dim) {
        return llvm::find(dimsToCheck, dim) == dimsToCheck.end();
    });
    auto splitsWithLowPriority = getSplitFromDimArr(restAllowedDims, allowedDims, vfConfig);
    mergedCase = getVFCaseFromSplits(splitsWithLowPriority);
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
        // try further with next operand
        // For situations
        // Operation1
        //   |      |          |
        //   |     Operation2
        //     Eltwise         Operation3
        // in case Operation1's user goes to Operation3, which cannot be fused with Eltwise
        // switch to Operation2, try to merge it with Eltwise
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
}  // namespace vpux::VPU::VF::v2
