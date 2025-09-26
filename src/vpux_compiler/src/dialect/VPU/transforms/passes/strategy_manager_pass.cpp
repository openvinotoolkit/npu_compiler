//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/mc_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_opt_alg.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_state_provider.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_STRATEGYMANAGERIMPL
#define GEN_PASS_DEF_STRATEGYMANAGERIMPL
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

namespace vpux::VPU {
namespace {

//
// StrategyManagerImplPass
//

class StrategyManagerImplPass final : public VPU::impl::StrategyManagerImplBase<StrategyManagerImplPass> {
public:
    explicit StrategyManagerImplPass(bool enablePrefetchTiling, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    SmallVector<std::pair<Strategy, StrategyCost>> getOperationOptions(mlir::Operation* operation,
                                                                       SiblingOpsAnalysis& siblingsAnalysis,
                                                                       size_t numTiles);
    SmallVector<VPU::MultiClusterStrategy> getAvailiableStrategies(config::ArchKind arch) const;
    bool checkDefaultStrategy(MultiClusterStrategy strategy) const;
    void fillInOptions(TilingOptions& options) const;
    bool mcTilingNeeded() const;
    bool hasNoUsersOrAllViewLike(mlir::Operation* op) const;
    bool lastOpExceptionCase(mlir::Operation* operation, MultiClusterStrategy strategy) const;

    std::shared_ptr<LayerVPUNNCost> _costModel;
    SmallVector<VPU::MultiClusterStrategy> _archStrategies;
    bool _enablePrefetchTiling = true;
    int64_t _numTiles;
};

void StrategyManagerImplPass::fillInOptions(TilingOptions& options) const {
    options.enablePrefetchTiling = _enablePrefetchTiling;
}

bool StrategyManagerImplPass::mcTilingNeeded() const {
    return _numTiles > 1;
}

bool StrategyManagerImplPass::hasNoUsersOrAllViewLike(mlir::Operation* op) const {
    // Return true if the operation has no users
    // For "ViewLike" users, recursively check if they also have no users or all "ViewLike" users.
    return op->getUsers().empty() || llvm::all_of(op->getUsers(), [&](mlir::Operation* user) {
               return VPU::isPureViewOp(user) && hasNoUsersOrAllViewLike(user);
           });
}

/*
If the op is last op emptying to DDR and strategy is HKSwitch then consider that as an exception case, and we need to
delete HKSwitch as strategy from operation options.
    * How to find last op :
        1. Op has no users or only has Return op as users
        2. All users of op are "ViewLike" operations or chain of view like ops or if all users' users have no users
themselves or are Return ops.
*/
bool StrategyManagerImplPass::lastOpExceptionCase(mlir::Operation* operation, MultiClusterStrategy strategy) const {
    if (strategy != MultiClusterStrategy::HKSwitch) {
        return false;
    }
    auto userOps = operation->getResult(0).getUsers();
    auto isUserReturnOp = [](mlir::Operation* op) {
        return mlir::isa<mlir::func::ReturnOp>(op);
    };
    // check if the operation has no users or if all users are "ViewLike" operations (including chains of "ViewLike"
    // operations) or all users are Return ops
    if (llvm::all_of(userOps, isUserReturnOp) || hasNoUsersOrAllViewLike(operation)) {
        return true;
    }
    return false;
}

mlir::LogicalResult StrategyManagerImplPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading enablePrefetchTiling with an MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

bool StrategyManagerImplPass::checkDefaultStrategy(MultiClusterStrategy strategy) const {
    // check if strategy is default
    return strategy == MultiClusterStrategy::Clustering;
}

SmallVector<std::pair<Strategy, StrategyCost>> StrategyManagerImplPass::getOperationOptions(
        mlir::Operation* operation, SiblingOpsAnalysis& siblingsAnalysis, size_t numTiles) {
    SmallVector<std::pair<Strategy, StrategyCost>> strategies;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);

    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation);

    if (clusteredOp == nullptr && tilingBuilderOp == nullptr) {
        return strategies;
    }

    auto operationStrategies = _archStrategies;

    for (auto strategy : operationStrategies) {
        if (clusteredOp != nullptr) {
            if (!(isStrategyCompatibleShape(clusteredOp,
                                            TileInfo(getShape(clusteredOp->getResult(0)), ShapeRef(), ShapeRef()),
                                            strategy, _log) &&
                  clusteredOp.checkStrategyCompatibility(strategy, numTiles))) {
                continue;
            }
        } else if (!checkDefaultStrategy(strategy)) {
            continue;
        }
        if (lastOpExceptionCase(operation, strategy)) {
            continue;  // Skip HKSwitch strategy for the last operation
        }
        bool hasPrefetch = false;
        do {
            mlir::ArrayAttr tilingStrategy;
            OutputTiling operationTiling;

            if (clusteredOp != nullptr) {
                clusteredOp.setMultiClusterStrategy(strategy);
            }

            auto mode = TilingMode::ISOLATED;
            bool prefetchTiling = hasPrefetch;
            // in case prefetch tiling is enabled, both cases should be taken into account
            // with prefetching and without
            hasPrefetch ^= _enablePrefetchTiling;
            bool tilingNeeded = false;
            auto tilingModeOpt = getTilingMode(operation, prefetchTiling, _log);
            if (tilingModeOpt.has_value()) {
                auto [tilingMode, prefetchingCandidate] = tilingModeOpt.value();
                if (tilingMode != TilingMode::PREFETCHING ||
                    (prefetchingCandidate && VPU::prefetchTilingConditionSatisfied(operation, _log))) {
                    tilingNeeded = true;
                }
            }
            if (tilingNeeded) {
                const auto layerTilingResult = getLayerTilingStrategy(tilingBuilderOp, prefetchTiling, mode, _log);

                if (mlir::failed(layerTilingResult)) {
                    continue;
                }

                operationTiling = layerTilingResult.value();
                VPUX_THROW_WHEN(operationTiling.empty(), "Couldn't get valid tiling for operation {0} in mode {1}",
                                operation->getLoc(), getTilingModeStr(mode));

                tilingStrategy = getIntArrayAttr(operation->getContext(), operationTiling[0].axis);
            } else if (clusteredOp != nullptr &&
                       !clusteredOp.doesLayerFitIntoCMX(strategy, siblingsAnalysis, Byte(0))) {
                _log.trace("Layer {0} doesn't fit in CMX with strategy {1}", operation->getLoc(), strategy);
                break;
            }

            if (operation->hasAttr(multiClusterStrategy)) {
                operation->removeAttr(multiClusterStrategy);
            }

            if (!tilingNeeded) {
                if (clusteredOp != nullptr && !clusteredOp.doesLayerFitIntoCMX(strategy, siblingsAnalysis, Byte(0))) {
                    _log.trace("Layer {0} doesn't fit in CMX with strategy {1}", operation->getLoc(), strategy);
                    continue;
                } else if (prefetchTiling) {
                    continue;
                }
            } else if (prefetchTiling && mode == TilingMode::ISOLATED) {
                continue;
            }

            const auto cost =
                    _costModel->getStrategyCost(operation, VPUNNCostParameters(strategy, operationTiling, mode));

            _log.trace("For operation {0} with MC strategy {1}-{2} vpunn returns cost {3}", operation->getLoc(),
                       strategy, tilingStrategy, cost);

            strategies.emplace_back(Strategy(strategy, tilingStrategy, mode), cost);

        } while (hasPrefetch);
    }

    return strategies;
}

SmallVector<VPU::MultiClusterStrategy> StrategyManagerImplPass::getAvailiableStrategies(config::ArchKind arch) const {
    auto mcListGetter = createMCStrategyGetter(arch, _numTiles);

    SmallVector<VPU::MultiClusterStrategy> strategies;
    mcListGetter->getMCStrategies(strategies);
    return strategies;
}

void StrategyManagerImplPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto siblingsAnalysis = getAnalysis<SiblingOpsAnalysis>();
    _costModel = std::make_shared<LayerVPUNNCost>(func);
    _numTiles = config::getTileExecutor(module).getCount();
    _archStrategies = getAvailiableStrategies(config::getArch(module));

    // calculate cost for all possible strategies
    // assign strategy with min cost
    auto operationStrategies = std::make_shared<OperationStrategies>();

    const auto findStrategyCallback = [&](mlir::Operation* operation) {
        auto strategies = getOperationOptions(operation, siblingsAnalysis, _numTiles);

        if (strategies.empty()) {
            return;
        }

        for (auto& [strategy, cost] : strategies) {
            auto operationStrategy = std::make_pair(operation, strategy);
            operationStrategies->addStrategy(operationStrategy, cost);
        }

        // set current and best one
        auto maxCostIt = std::max_element(strategies.begin(), strategies.end(), [](auto lhs, auto rhs) {
            return lhs.second < rhs.second;
        });

        if (maxCostIt == strategies.end()) {
            return;
        }

        const auto maxCostStrategy = std::make_pair(operation, maxCostIt->first);
        operationStrategies->setCurrentStrategy(maxCostStrategy);
        operationStrategies->setBestStrategy(maxCostStrategy);
    };

    func.walk(findStrategyCallback);

    auto operations = operationStrategies->getAllOperations();
    if (operations.empty()) {
        return;
    }

    // optimization
    auto stateProvider = std::make_shared<DefaultStateProvider>(operationStrategies, _costModel);

    if (operations.size() > 1) {
        TilingOptions options;
        fillInOptions(options);
        const auto optAlgorithm = createAlgorithm(options, stateProvider, operationStrategies);
        optAlgorithm->optimize();
    } else {
        auto* operation = operationStrategies->getAllOperations().front();
        auto strategies = operationStrategies->getAllStrategies(operation);
        auto minCostIt = std::min_element(strategies.begin(), strategies.end(), [](auto lhs, auto rhs) {
            return lhs.strategyCost < rhs.strategyCost;
        });

        if (minCostIt == strategies.end()) {
            return;
        }

        const auto minCostStrategy = std::make_pair(operation, minCostIt->strategy);
        operationStrategies->setCurrentStrategy(minCostStrategy);
        operationStrategies->setBestStrategy(minCostStrategy);
    }

    // set best strategy to each operation
    const auto setStrategyCallback = [&](mlir::Operation* operation) {
        if (VPU::isPureViewOp(operation) || !operationStrategies->hasAnyStrategy(operation)) {
            return;
        }

        const auto bestResult = operationStrategies->getBestStrategy(operation);

        if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
            if (mcTilingNeeded()) {
                clusteredOp.setMultiClusterStrategy(bestResult.getMCStrategy());
            }
        }

        if (bestResult.getTilingStrategy() != nullptr) {
            if (auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation)) {
                tilingOp->setAttr(tilingStrategy, bestResult.getTilingStrategy());
            }
        }
    };

    func.walk(setStrategyCallback);
}

}  // namespace

//
// createStrategyManagerImplPass
//

std::unique_ptr<mlir::Pass> createStrategyManagerImplPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<StrategyManagerImplPass>(enablePrefetchTiling, log);
}

}  // namespace vpux::VPU
