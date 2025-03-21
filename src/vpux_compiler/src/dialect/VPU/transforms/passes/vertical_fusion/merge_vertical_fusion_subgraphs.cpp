//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduling_factory.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>

#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MERGEVFSUBGRAPHS
#define GEN_PASS_DEF_MERGEVFSUBGRAPHS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

// This function tries to find mergeable input for the currentOp
// Currently only support NCE task with weights
// E-141686: A general solution to merge more subgraph for VFOp.
mlir::FailureOr<VPU::VerticalFusionOp> findMergeableVFInput(VFConfig& vfConfig) {
    auto currentOp = vfConfig.getSubgraph();
    for (auto* op : vfConfig.getOperationsForTiling()) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
            continue;
        }
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(nceOp.getWeightsOperand())) {
            auto parentOp = currentOp.getOperand(blockArg.getArgNumber()).getDefiningOp<VPU::VerticalFusionOp>();
            if (parentOp != nullptr) {
                return parentOp;
            }
        }
    }
    return mlir::failure();
}

// This function checks if other inputs of currentOp can be merged
// If prevOp was tried to merge with currentOp, return false
bool checkOtherVFInput(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) {
    // Check if currentOp has mergeable input
    auto vfConfig = VFConfig(currentOp);
    auto mergeableOp = findMergeableVFInput(vfConfig);
    if (mlir::failed(mergeableOp)) {
        return false;
    }
    // prevOp was tried to merge with currentOp
    return mergeableOp.value() != prevOp;
}

// This function checks the weights tilingStrategy is split over output channel
bool isTileOverOutputChannel(VFConfig& vfConfig) {
    // Check if nceTaskOp has mergeable input, weights of NCE task
    auto nceTaskOp = vfConfig.getSubgraph();
    auto weightsOp = findMergeableVFInput(vfConfig);
    if (mlir::failed(weightsOp)) {
        return false;
    }

    const auto moreThanOne = [](auto value) {
        return value > 1;
    };

    // weights, tiles on OutputChannel dim > 1
    // NCE task, tiles on activation Channel dim > 1
    const auto weightsTilingStrategy = parseIntArrayAttr<int64_t>(weightsOp.value().getTilingStrategy());
    const auto nceTaskTilingStrategy = parseIntArrayAttr<int64_t>(nceTaskOp.getTilingStrategy());
    return weightsTilingStrategy[Dims4D::Filter::OC.ind()] > 1 || llvm::any_of(nceTaskTilingStrategy, moreThanOne);
}

// set VF users for correct tensor size calculation
class VFSubgraphUserSetter {
public:
    VFSubgraphUserSetter(VerticalFusionOp original, VerticalFusionOp candidate)
            : _mOriginalSubgraph(original), _mCandidateSubgraph(candidate) {
        moveUsers(_mOriginalSubgraph, _mCandidateSubgraph);
    }
    ~VFSubgraphUserSetter() {
        moveUsers(_mCandidateSubgraph, _mOriginalSubgraph);
    }

private:
    void moveUsers(VerticalFusionOp from, VerticalFusionOp to);

    VerticalFusionOp _mOriginalSubgraph;
    VerticalFusionOp _mCandidateSubgraph;
};

void VFSubgraphUserSetter::moveUsers(VerticalFusionOp from, VerticalFusionOp to) {
    from.getResult(0).replaceAllUsesWith(to.getResult(0));
}

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, bool enablePrefetchTiling,
                          const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _enablePrefetchTiling(enablePrefetchTiling),
              _vpunnCostFunction(costFunction),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp, VFCase& mergedCase) const;
    int64_t getMaxDimLimit(const Dim axis, ArrayRef<mlir::Operation*> operation) const;
    std::optional<int64_t> getOptimalTilingStrategy(const std::shared_ptr<IVFScheduling>& scheduling, const Dim dim,
                                                    const int64_t minTiles, int64_t& maxTiles,
                                                    TilingOperationStorage::UPtr& minStorage,
                                                    TilingOperationStorage::UPtr& maxStorage, VFConfig& config) const;
    bool waitOtherUsers(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    mlir::FailureOr<VFCase> findVFCase(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp,
                                       VPU::VerticalFusionOp mergedVFOp) const;
    bool alignMCTiling(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    mlir::FailureOr<VFCase> findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                         VPU::VerticalFusionOp currentOp) const;
    std::deque<std::shared_ptr<IVFScheduling>> getVFSchedulingChecks(VFConfig& config) const;
    StrategyCost extractVFCost(VFConfig& vfConfig) const;
    std::shared_ptr<IVFScheduling> detectScenario(VFConfig& vfConfig) const;

    void fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp currentOp,
                    VPU::VerticalFusionOp mergedOp) const;

    bool _enableVerticalFusionPipelining = false;
    bool _enablePrefetchTiling = true;
    const std::unique_ptr<VPU::LayerVPUNNCost>& _vpunnCostFunction;
    Logger _log;
};

inline bool hasTiling(const ArrayRef<int64_t> tilingInfo) {
    return llvm::any_of(tilingInfo, [](auto i) {
        return i != 1;
    });
}

bool MergeVFRegionRewriter::alignMCTiling(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const {
    const auto prevBlock = prevOp.getBody();
    const auto parentVFOp = currentOp.getBody();

    auto newOps = prevBlock->getOps<VPU::VerticalFusionOpInterface>();
    auto oldOps = parentVFOp->getOps<VPU::VerticalFusionOpInterface>();

    if (newOps.empty() || oldOps.empty()) {
        return false;
    }

    const auto getCurrInputArgument = [](VPU::VerticalFusionOp currentOp,
                                         VPU::VerticalFusionOp prevOp) -> mlir::BlockArgument {
        for (auto blockArg : currentOp.getBody()->getArguments()) {
            auto operand = currentOp.getOperand(blockArg.getArgNumber());
            if (operand.getDefiningOp() == prevOp.getOperation()) {
                return blockArg;
            }
        }
        return nullptr;
    };

    // Get output op of previous vf region
    auto prevOutputOp = prevOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    // Get input arg of current vf region corresponding to previous vf op
    auto currInputArg = getCurrInputArgument(currentOp, prevOp);
    VPUX_THROW_UNLESS(currInputArg != nullptr,
                      "No corresponding input argument found for current VF region with previous VF region");

    const auto isClusteredOpWithMCStrategy = [](mlir::Operation* op) {
        auto clusterOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    const auto getOutputDistributedType = [](VPU::ClusteredOpInterface clusteredOp) {
        const auto outputType = clusteredOp->getResult(0).getType().cast<NDTypeInterface>();
        const auto numClusters =
                clusteredOp.getOptimalNumClusters(outputType.getShape(), clusteredOp.getMultiClusterStrategy().value());

        auto ndType = VPU::getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters).cast<NDTypeInterface>();
        if (auto sparseTensorType = ndType.dyn_cast<VPU::SparseTensorType>()) {
            ndType = sparseTensorType.getData().cast<NDTypeInterface>();
        }

        return ndType.dyn_cast_or_null<VPU::DistributedTensorType>();
    };

    const auto getInputDistributedType = [](VPU::ClusteredOpInterface clusteredOp, mlir::Value inputOperand,
                                            bool& isSparsed) {
        const auto inputType = inputOperand.getType().cast<NDTypeInterface>();
        isSparsed = false;
        const auto numClusters =
                clusteredOp.getOptimalNumClusters(inputType.getShape(), clusteredOp.getMultiClusterStrategy().value());

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());

        auto ndType =
                (nceOp != nullptr && nceOp.getWeightsOperand() == inputOperand)
                        ? VPU::getDistributedFilterTypeFromOp(nceOp, inputType, numClusters).cast<NDTypeInterface>()
                        : VPU::getDistributedActivationTypeFromOp(clusteredOp, inputType, numClusters)
                                  .cast<NDTypeInterface>();
        if (auto sparseTensorType = ndType.dyn_cast<VPU::SparseTensorType>()) {
            ndType = sparseTensorType.getData().cast<NDTypeInterface>();
            isSparsed = true;
        }

        return ndType.dyn_cast_or_null<VPU::DistributedTensorType>();
    };

    const auto inferInputDistributedType = [](VPU::DistributedTensorType srcDistType,
                                              ArrayRef<mlir::Operation*> inputViewOps) {
        auto inputDistType = mlir::cast<vpux::NDTypeInterface>(srcDistType);
        auto distribution = VPU::DistributionInfo::getClassFromAttr(srcDistType.getDistribution());
        for (auto viewOp : inputViewOps) {
            if (auto distCastOp = mlir::dyn_cast<VPU::DistributedCastOpInterface>(viewOp)) {
                auto castedTypeWithDistribution =
                        distCastOp.inferCastedTypeAndDistribution(inputDistType, distribution);
                if (mlir::succeeded(castedTypeWithDistribution)) {
                    inputDistType = mlir::cast<vpux::NDTypeInterface>(castedTypeWithDistribution.value().first);
                    distribution = castedTypeWithDistribution.value().second;
                }
            }
        }
        TensorDistributionMap distributionMap;
        distributionMap.insert(std::make_pair(inputDistType, distribution));
        return mlir::cast<VPU::DistributedTensorType>(
                getDistributedTypeFromDistributionMap(inputDistType, distributionMap));
    };

    // Check if previous output op has MC strategy
    const auto isPrevOutOpWithMCStrategy = isClusteredOpWithMCStrategy(prevOutputOp);
    const auto prevOutDistType = isPrevOutOpWithMCStrategy
                                         ? getOutputDistributedType(mlir::cast<VPU::ClusteredOpInterface>(prevOutputOp))
                                         : nullptr;

    const auto hasTrueOverlappedParams = [](VPU::DistributedTensorType tensor) {
        if (tensor == nullptr) {
            return false;
        }
        if (tensor.getDistribution().getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
            return false;
        }
        if (tensor.getPerClusterMemoryShapes() != tensor.getPerClusterComputeShapes()) {
            return true;
        }
        if (tensor.getPerClusterMemoryShapeOffsets() != tensor.getPerClusterComputeShapeOffsets()) {
            return true;
        }
        return false;
    };
    bool outputTrueOverlapped = hasTrueOverlappedParams(prevOutDistType);
    bool isSWLayer = mlir::isa<VPU::SWOpInterface>(prevOutputOp);

    // Here we need to ensure either all current input ops and previous output op have no mc strategy,
    // or all have mc stratgy with compatible distributed tensor types
    for (auto currInputOp : currInputArg.getUsers()) {
        SmallVector<mlir::Operation*> currInputViewLikeOps;
        while (mlir::isa<VPU::TilingViewLikeOpInterface>(currInputOp) && currInputOp->hasOneUse()) {
            currInputViewLikeOps.push_back(currInputOp);
            currInputOp = *(currInputOp->getUsers().begin());
        }
        const auto isCurrInOpWithMCStrategy = isClusteredOpWithMCStrategy(currInputOp);
        if (isPrevOutOpWithMCStrategy != isCurrInOpWithMCStrategy) {
            return false;
        }
        if (isPrevOutOpWithMCStrategy && isCurrInOpWithMCStrategy) {
            auto currInputOperand = currInputViewLikeOps.empty() ? currInputArg.cast<mlir::Value>()
                                                                 : currInputViewLikeOps.back()->getResult(0);
            bool isSparsed = false;
            auto actualCurrInDistType = getInputDistributedType(mlir::cast<VPU::ClusteredOpInterface>(currInputOp),
                                                                currInputOperand, isSparsed);

            auto inputTrueOverlapped = hasTrueOverlappedParams(actualCurrInDistType);

            //  E#112803 will handle sparse consumers
            if (inputTrueOverlapped && isSparsed) {
                return false;
            }

            // TODO E#92130 extend Shave operations with OVERLAPPED param propagation
            if ((outputTrueOverlapped && mlir::isa<VPU::SWOpInterface>(currInputOp)) ||
                (isSWLayer && inputTrueOverlapped)) {
                return false;
            }

            auto inferredCurrInDistType = inferInputDistributedType(prevOutDistType, currInputViewLikeOps);

            if (areDistributionAttrsCompatible(inferredCurrInDistType, actualCurrInDistType, true).failed()) {
                return false;
            }
        }
    }

    return true;
}

// Get operandNumber for prevOp output in currentOp inputs
size_t getLinkNumber(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) {
    auto operands = currentOp->getOperands();
    auto operandIt = llvm::find_if(operands, [&](auto operand) {
        return operand.getDefiningOp() == prevOp;
    });
    VPUX_THROW_WHEN(operandIt == operands.end(),
                    "Cannot find the operand number for the operation {0} in the current block {1}", prevOp, currentOp);
    return std::distance(operands.begin(), operandIt);
}

VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                    const SmallVector<TileInfo>& inputTiles, const bool enablePrefetching) {
    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
        mcStrategy = mcOperation.getMultiClusterStrategy().value_or(mcStrategy);
    }

    auto mode = enablePrefetching ? TilingMode::PREFETCHING : TilingMode::ISOLATED;

    SmallVector<OutputTiling> inputAllTiles;
    if (!inputTiles.empty()) {
        inputAllTiles.push_back(inputTiles);
    }
    return VPUNNCostParameters(mcStrategy, tiling, mode, inputAllTiles);
}

std::shared_ptr<IVFScheduling> MergeVFRegionRewriter::detectScenario(VFConfig& vfConfig) const {
    VFSchedulingFactory costFactory(_enablePrefetchTiling);
    auto scenarioKind = vfConfig.getSubgraph().getScenario().has_value()
                                ? vfConfig.getSubgraph().getScenario().value()
                                : _enablePrefetchTiling ? VFScenario::WEIGHTS_PREFETCHING : VFScenario::MINIMAL;
    return costFactory.createVFScenario(scenarioKind, _log);
}

SmallVector<mlir::Operation*> findUsers(mlir::Operation* operation) {
    SmallVector<mlir::Operation*> users;

    for (auto* user : operation->getUsers()) {
        if (!isPureViewOp(user)) {
            users.emplace_back(user);
            continue;
        }

        auto usersBelow = findUsers(user);
        if (!usersBelow.empty()) {
            llvm::copy(usersBelow, std::back_inserter(users));
        }
    }

    return users;
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
            auto tiling = fillDividedTiles(operation, Shape(tilingDims), getShape(operation->getResult(0)));
            VPUX_THROW_WHEN(mlir::failed(tiling), "Incorrect tiling {0} for vf {1}", tilingDims, vfOp);
            tiles = tiling.value();
        }

        const auto costParameters = fillInCostParam(operation, tiles, {}, _enablePrefetchTiling);
        auto cost = _vpunnCostFunction->getStrategyCost(operation, costParameters);

        SmallVector<mlir::Value> operands = {operation->getOperand(0)};
        if (operation->getNumOperands() > 1 && operation->hasTrait<VPU::EltwiseOp>() &&
            operation->getOperand(0) != operation->getOperand(1)) {
            operands.emplace_back(operation->getOperand(1));
        }

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);

        auto spilling = dim.has_value() &&
                        (isSpatialTiling(tilingDims) || (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr));
        auto hasSpilledParents = llvm::any_of(operands, [&](mlir::Value value) {
            if (auto arg = value.dyn_cast<mlir::BlockArgument>()) {
                auto parentOperand = vfConfig.getSubgraph().getOperand(arg.getArgNumber());
                auto parentOp = findParent(parentOperand);
                return !isCmxOperation(parentOp, false) ||
                       isPrevOperationEarlyScheduled(parentOp, vfConfig.getSubgraph());
            }
            return false;
        });
        auto hasSpilledUsers = vfConfig.getSubgraph()->getUsers().empty() ||
                               llvm::any_of(findUsers(vfConfig.getSubgraph()), [&vfConfig](auto* user) {
                                   return !isCmxOperation(user, true) ||
                                          isPrevOperationEarlyScheduled(vfConfig.getSubgraph().getOperation(), user);
                               });

        auto spillReadWriteCanBeOverlapped = [&]() {
            if (operations.size() != 1 || costParameters._tiling.size() <= 1) {
                return false;
            }

            const auto arch = VPU::getArch(operation);
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
                auto arg = operand.dyn_cast<mlir::BlockArgument>();
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

/*
 Function checks if two blocks suit to be merged in one on following criterias:
 1. Number of operations doesn't exceed the limit
 2. In case there is only one operation in the block, it might be merged as first op in the block
 3. All multicluster strategies are same for both blocks if there are any
 4. Required CMX memory by constant weights shouldn't exceed the size of the whole memory
*/
bool MergeVFRegionRewriter::checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                VFCase& mergedCase) const {
    VPUX_THROW_WHEN(!mergedCase.isInitialized(), "Incorrect tiling strategy for VF");

    if (mergedCase.getTilingNumber() == 1 && mergedCase.getConfig().isPotentiallyPipelined()) {
        mergedCase.approveScheduling();
        return true;
    }

    // compare the cost between merged VF Subgraph and 2 subgraphs with the spill
    auto prevOpConfig = VFConfig(prevOp, _enablePrefetchTiling);
    auto currentOpConfig = VFConfig(currentOp, _enablePrefetchTiling);

    const auto prevCost = extractVFCost(prevOpConfig);
    const auto currentCost = extractVFCost(currentOpConfig);

    // simply decide if there is tiling for parents
    const auto prevTilingStrategy = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());
    const auto currentTilingStrategy = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());

    StrategyCost mergedVFCost = 0;
    {
        // change the IR so that merged VF substitutes current operation and previous op to
        // calculate correct cost
        // the IR will change back when the setter is destroyed
        VFSubgraphUserSetter setter(currentOp, mergedCase.getConfig().getSubgraph());
        mergedCase.getConfig().invalidatePointers();
        prevOpConfig.invalidatePointers();
        currentOpConfig.invalidatePointers();
        mergedVFCost = mergedCase.getCost(_vpunnCostFunction, _log);
    }
    mergedCase.getConfig().invalidatePointers();

    if (mergedVFCost > prevCost + currentCost) {
        _log.trace("Failed to merge VerticalFusionOp due to higher cost: mergedVFCost ({0}) > prevCost ({1}) + "
                   "currentCost ({2})",
                   mergedVFCost, prevCost, currentCost);
        return false;
    }

    mergedCase.approveScheduling();
    return true;
}

/*
 As soon as we don't have logic right now for excluding operations or break subgraph
 check in advance that all users or previous block will be merged to current one
*/
bool MergeVFRegionRewriter::waitOtherUsers(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp) const {
    if (prevOp->hasOneUse()) {
        return true;
    }

    for (auto user : prevOp->getUsers()) {
        if (!mlir::isa<VPU::VerticalFusionOp>(user)) {
            return false;
        }
        if (user == currentOp) {
            continue;
        }

        const auto userGoToRegion = llvm::any_of(user->getUsers(), [&](auto current) {
            return current != currentOp;
        });

        if (userGoToRegion) {
            return false;
        }
    }

    return true;
}

std::optional<int64_t> MergeVFRegionRewriter::getOptimalTilingStrategy(
        const std::shared_ptr<IVFScheduling>& scheduling, const Dim dim, const int64_t minTiles, int64_t& maxTiles,
        TilingOperationStorage::UPtr& minStorage, TilingOperationStorage::UPtr& maxStorage, VFConfig& config) const {
    if (minTiles > maxTiles || maxTiles == 1) {
        return std::nullopt;
    }

    auto minNTiles = minTiles;
    auto maxNTiles = maxTiles;

    std::optional<int64_t> result;
    auto outType = config.getSubgraph()->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto tilingArray = SmallVector<int64_t>(outType.getRank(), 1);
    tilingArray[dim.ind()] = minNTiles;
    if (minTiles == maxTiles) {
        if (minStorage == nullptr) {
            minStorage = std::make_unique<TilingOperationStorage>();
            auto tilingRegions = VPU::calculateTilingRegions(config.getSubgraph(), tilingArray, _log, minStorage);

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
        auto getValidStrategy = VPU::getMinimalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray,
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

    if (maxStorage == nullptr) {
        maxStorage = std::make_unique<TilingOperationStorage>();
        auto getValidStrategy = VPU::getMaximalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray,
                                                                            tilingMaxStrategy, dim, maxStorage, _log);

        if (mlir::failed(getValidStrategy)) {
            maxStorage.reset();
            return std::nullopt;
        }

        tilingMaxStrategy = getValidStrategy.value();
        maxNTiles = tilingMaxStrategy[dim.ind()];
        maxTiles = tilingMaxStrategy[dim.ind()];
    }

    if (!scheduling->validate(config, maxStorage)) {
        return std::nullopt;
    }

    auto axisIncrement = getVFAxisIncrement(dim);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", dim);

    auto nextValueFromMin = minNTiles;
    axisIncrement->increasedValue(nextValueFromMin, maxNTiles);

    while (minNTiles < maxNTiles) {
        auto currentNTiles = axisIncrement->getMiddleValue(minNTiles, maxNTiles);

        if (maxNTiles == nextValueFromMin) {
            result = maxNTiles;
            if (maxNTiles == maxTiles) {
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
        auto getValidTilingStrategy = VPU::getMinimalValidTilingStrategyFromRange(
                config.getSubgraph(), tilingArray, tilingMaxStrategy, dim, opStorage, _log);
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
}

std::deque<std::shared_ptr<IVFScheduling>> MergeVFRegionRewriter::getVFSchedulingChecks(VFConfig& config) const {
    std::deque<std::shared_ptr<IVFScheduling>> vfChecks;
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

mlir::FailureOr<VFCase> MergeVFRegionRewriter::findVFTiling(VPU::VerticalFusionOp mergedOp,
                                                            VPU::VerticalFusionOp prevOp,
                                                            VPU::VerticalFusionOp currentOp) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    VPUX_THROW_WHEN(currentTiling.size() != prevTiling.size(),
                    "Tiling info rank of current block {0} is not equal to tiling info rank of previous block {1}",
                    currentTiling.size(), prevTiling.size());
    auto currentConfig = VFConfig(currentOp, _enableVerticalFusionPipelining);
    auto prevConfig = VFConfig(prevOp, _enableVerticalFusionPipelining);

    auto curAxis = getVFTilingDim(currentTiling, currentConfig.getVFOperations());
    auto prevAxis = getVFTilingDim(prevTiling, prevConfig.getVFOperations());

    if (mlir::failed(curAxis) || mlir::failed(prevAxis)) {
        return mlir::failure();
    }

    bool curHasTiling = hasTiling(currentTiling);
    bool prevHasTiling = hasTiling(prevTiling);
    // in case both subgraphs have tiling, check if they match
    // if there is only one subgraph with tiling, check if it's allowed
    // to tile second one with such axis
    // if both doesn't have tiling, check if there is at least one
    // allowed axis for both of them

    auto vfConfig = VFConfig(mergedOp, _enableVerticalFusionPipelining);

    if (!prevHasTiling && !curHasTiling && !vfConfig.isPipelined()) {
        const auto filterNCENotEltwiseLike = [](mlir::Operation* op) {
            return mlir::isa<VPU::NCEOpInterface>(op) && !op->hasTrait<VPU::EltwiseOp>();
        };
        const auto filterSWKernels = [](mlir::Operation* op) {
            return mlir::isa<VPU::SWOpInterface>(op);
        };
        // when pipeline case is generic this check is enough to prevent VF
        // but now we check additionally that there are no operations
        // with different executors
        auto checkedOperations = vfConfig.getOperationsForTiling();
        if ((llvm::all_of(checkedOperations, filterNCENotEltwiseLike) ||
             llvm::all_of(checkedOperations, filterSWKernels))) {
            return mlir::failure();
        }
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

    VFSubgraphUserSetter setter(currentOp, mergedOp);

    auto getVFCaseWithTiling = [&](const Dim curDim, const Dim prevDim) -> VFCase {
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
        auto curInputAxes = backInferVFTilingDim(currentConfig, curAxis.value(), opDimMap);
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
        return mlir::failure();
    }

    StrategyCost bestCost = std::numeric_limits<StrategyCost>::max();
    mlir::FailureOr<VFCase> mergedCase;
    for (auto dim : allowedDims) {
        // in order not to check twice dim which has been handled unsuccessfully
        if (checkedDim.has_value() && checkedDim.value() == dim) {
            continue;
        }
        // E.g., prevTiling [1, 3, 1, 1] -> permuteCast -> currentTiling [1, 1, 2, 1]
        // Thus we need dim backinfer to get correct axis to compare
        // As Vf inputs may be more than one, we need backinfer dim for each of them and use correct one
        auto curInputDims = backInferVFTilingDim(currentConfig, dim, opDimMap);

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

mlir::FailureOr<VFCase> MergeVFRegionRewriter::findVFCase(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                          VPU::VerticalFusionOp mergedVFOp) const {
    if (!alignMCTiling(currentOp, prevOp)) {
        return mlir::failure();
    }
    return findVFTiling(mergedVFOp, prevOp, currentOp);
}

void MergeVFRegionRewriter::fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp currentOp,
                                       VPU::VerticalFusionOp mergedOp) const {
    rewriter.replaceOp(currentOp, mergedOp.getResult(0));
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
        if (!allInOldBlock) {
            if (waitOtherUsers(parentVFOp, vfOp)) {
                continue;
            }
            return mlir::failure();
        }

        vfBlock = fuseOpsInBlock(rewriter, vfOp, parentVFOp.getOperation());
        auto vfCase = findVFCase(parentVFOp, vfOp, vfBlock);
        if (mlir::failed(vfCase) || !checkVFCostFunction(parentVFOp, vfOp, vfCase.value())) {
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

//
// MergeVfSubgraphsPass
//

class MergeVfSubgraphsPass final : public VPU::impl::MergeVfSubgraphsBase<MergeVfSubgraphsPass> {
public:
    explicit MergeVfSubgraphsPass(bool enableVerticalFusionPipelining, bool enablePrefetchTiling, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult MergeVfSubgraphsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading MergeVfSubgraphsPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_enablePrefetchTiling' field to the value {1} of the "
                   "pass option 'tilingMode' generated by MLIR",
                   _enablePrefetchTiling, tilingMode.getValue());
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void MergeVfSubgraphsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto costFunction = std::make_unique<VPU::LayerVPUNNCost>(func);

    // TODO rewrite with largest spill detection on each step
    SmallVector<int64_t> maxTiling;
    func->walk([&](VPU::VerticalFusionOp op) {
        const auto tiling = parseIntArrayAttr<int64_t>(op.getTilingStrategy());
        auto dim = getVFTilingDim(tiling);
        if (dim.has_value()) {
            maxTiling.emplace_back(tiling[dim.value().ind()]);
        }
    });

    auto maxTilingOp = std::max_element(maxTiling.begin(), maxTiling.end());

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeVFRegionRewriter>(&ctx, _enableVerticalFusionPipelining, _enablePrefetchTiling, costFunction,
                                        _log);

    auto config = getDefaultGreedyRewriteConfig();
    config.useTopDownTraversal = maxTilingOp == maxTiling.end() || maxTilingOp != std::prev(maxTiling.end());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeVfSubgraphsPass
//

std::unique_ptr<mlir::Pass> VPU::createMergeVfSubgraphsPass(bool enableVerticalFusionPipelining,
                                                            bool enablePrefetchTiling, Logger log) {
    return std::make_unique<MergeVfSubgraphsPass>(enableVerticalFusionPipelining, enablePrefetchTiling, log);
}
