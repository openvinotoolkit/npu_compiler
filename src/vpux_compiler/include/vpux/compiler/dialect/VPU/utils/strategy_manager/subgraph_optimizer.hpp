//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"

#include <deque>

namespace vpux {
namespace VPU {
//
// Subgraph optimizer
//
class SubgraphOptimizer final {
public:
    using ShortcutMapTy =
            std::unordered_map<mlir::Operation*, std::pair<mlir::Operation*, SmallVector<mlir::Operation*>>>;
    SubgraphOptimizer(mlir::func::FuncOp func, bool enablePrefetchTiling, SiblingOpsAnalysis& siblingsOpsAnalysis,
                      std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModelPtr, Logger log);
    void optimizeStrategyAvoidSpillingOnModel();

private:
    struct SubgraphOptConfig {
        // indicate if using rollback strategy for cost calculation
        bool useRollbackStrategy;
        // indicate if checking concat related spilling cost
        bool checkConcatRelatedSpilling;
    };

    VPU::MultiClusterStrategy getRollbackStrategy(VPU::ClusteredOpInterface op);

    bool isValidStrategy(VPU::ClusteredOpInterface op, VPU::MultiClusterStrategy strategy);
    bool isStrategySOHLike(VPU::ClusteredOpInterface op);
    bool isStrategySOKLike(VPU::ClusteredOpInterface op);
    VPU::MultiClusterStrategy getBestInSOKLikeStrategies(VPU::ClusteredOpInterface op);
    VPU::MultiClusterStrategy getBestInSOHLikeStrategies(VPU::ClusteredOpInterface op);
    bool hasSpillingAroundConcat(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy clusteredStrategy,
                                 mlir::Operation* excludedOp = nullptr);
    bool hasSpillingCausedByStridedCMXConcat(VPU::ClusteredOpInterface clusteredOp,
                                             VPU::MultiClusterStrategy clusteredStrategy,
                                             const SubgraphOptConfig& config);
    bool hasSpillingRelatedToConcat(VPU::ClusteredOpInterface parentClusteredOp,
                                    VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userClusteredOp,
                                    VPU::MultiClusterStrategy userStrategy, const SubgraphOptConfig& config);
    ShortcutMapTy detectShortcuts();
    bool hasLongTermSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface parentOp,
                             VPU::MultiClusterStrategy parentOpStrategy, const SubgraphOptConfig& config);
    bool hasInputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origClusteredOp,
                                             VPU::MultiClusterStrategy origStrategy, const SubgraphOptConfig& config);
    bool hasInputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origClusteredOp, mlir::Value input,
                                             VPU::MultiClusterStrategy origStrategy, const SubgraphOptConfig& config);
    bool hasOutputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origClusteredOp,
                                              VPU::MultiClusterStrategy origStrategy, const SubgraphOptConfig& config);
    bool hasOutputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origClusteredOp, mlir::Operation* userOp,
                                              VPU::MultiClusterStrategy origStrategy, const SubgraphOptConfig& config);
    double getInputSpillingCostToMultiClusterLayer(VPU::ClusteredOpInterface clusteredOp,
                                                   VPU::MultiClusterStrategy strategy, const SubgraphOptConfig& config);
    double getInputSpillingCostToMultiClusterLayer(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                   VPU::MultiClusterStrategy strategy, const SubgraphOptConfig& config);
    double getOutputSpillingCostToMultiClusterLayer(VPU::ClusteredOpInterface clusteredOp,
                                                    VPU::MultiClusterStrategy strategy,
                                                    const SubgraphOptConfig& config);
    void optimizeStrategyAvoidSpillingOnSubgraph(VPU::ClusteredOpInterface op);
    void removeClusteringStrategyAvoidSpillingOnSubgraph(VPU::ClusteredOpInterface op);
    void tryInheritStrategyFromParent(VPU::ClusteredOpInterface clusteredOp);

    std::deque<VPU::ClusteredOpInterface> layersNeedRollback;
    std::map<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy> layersWithRollbackStrategy;
    // Layers which got same rollback strategy twice
    std::set<mlir::Operation*> layersWithConvergedStrategy;
    // Keeps a record of operations where spill cost has already being taken into account
    std::set<mlir::Operation*> opsWithSpillWriteCounted;
    mlir::func::FuncOp _func;
    Logger _log;
    LayerCostModel _layerCostModel;
    VPU::ClusteredOpInterface _subgraphOptBeginOp;
    // Map pattern: {ResBlock_endpoint : {ResBlock_startpoint : [Middle ops in residual block]}}
    // Shortcut is the direct connection from ResBlock_startpoint to ResBlock_endpoint, which is a special structure we
    // want to catch in long-term spilling
    ShortcutMapTy _shortcutsMap;
    // The configuration of finding a initial node for subgraph strategy optimization
    SubgraphOptConfig _configForFindingStartNode = {false, false};
    // The configuration of finding best rollback strategy
    SubgraphOptConfig _configForFindingRollbackStrategy = {true, false};
    // The configuration of checking whether neighbouring layers need be added into rollback list
    SubgraphOptConfig _configForFindingNeighbourNodes = {true, true};
    // The configuration of calculating the original cost of subgraph
    SubgraphOptConfig _configForCalcOrigCost = {false, true};
    // The configuration of calculating the rollback cost of subgraph
    SubgraphOptConfig _configForCalcRollbackCost = {true, true};
    SiblingOpsAnalysis& _siblingsOpsAnalysis;
};

}  // namespace VPU
}  // namespace vpux
