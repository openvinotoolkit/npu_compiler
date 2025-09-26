//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_MULTICLUSTERSTRATEGYASSIGNMENT
#define GEN_PASS_DEF_MULTICLUSTERSTRATEGYASSIGNMENT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// MultiClusterStrategyAssignmentPass
//

class MultiClusterStrategyAssignmentPass final :
        public VPU::impl::MultiClusterStrategyAssignmentBase<MultiClusterStrategyAssignmentPass> {
public:
    explicit MultiClusterStrategyAssignmentPass(bool enablePrefetchTiling, const int clusteredOpThreshold,
                                                StringRef mcOptimizationScope, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling),
              _clusteredOpThreshold(clusteredOpThreshold),
              _mcOptimizationScope(getMCOptimizationScope(mcOptimizationScope)) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    vpux::VPU::MCOptimizationScope getMCOptimizationScope(StringRef mcOptimizationScope);
    void safeRunOnFunc() final;

private:
    bool _enablePrefetchTiling = true;
    int _clusteredOpThreshold;
    VPU::MCOptimizationScope _mcOptimizationScope;
};

mlir::LogicalResult MultiClusterStrategyAssignmentPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading enablePrefetchTiling with an MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }

    if (mcOptimizationScope.hasValue()) {
        _mcOptimizationScope = getMCOptimizationScope(mcOptimizationScope);
    }

    return mlir::success();
}

vpux::VPU::MCOptimizationScope MultiClusterStrategyAssignmentPass::getMCOptimizationScope(
        StringRef mcOptimizationScope) {
    VPUX_THROW_WHEN(
            mcOptimizationScope.empty(),
            "Multi-cluster strategy optmization scope not provided. Please try 'mc-optimization-scope=subgraph'");
    const auto parsed = VPU::symbolizeMCOptimizationScope(mcOptimizationScope.upper());
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported multi-cluster strategy optimization scope '{0}'",
                      mcOptimizationScope.str());
    return parsed.value();
}

//
// safeRunOnFunc
//

void MultiClusterStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto tileOp = config::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");

    if (tileOp.getCount() < 2) {
        return;
    }

    const auto arch = config::getArch(module);
    auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
    auto layerCostModel =
            VPU::LayerCostModelAnalysis::getOrCreateLayerCostModel(maybeLayerCostModelAnalysis, arch, _log);

    auto clusteredOps = func.getOps<ClusteredOpInterface>();
    auto clusteredOpCount = std::distance(clusteredOps.begin(), clusteredOps.end());
    auto& cache = VPU::OpTilingCache::instance();
    cache.enableIfNecessary(clusteredOpCount > _clusteredOpThreshold);

    auto& siblingAnalysis = getAnalysis<SiblingOpsAnalysis>();
    StrategyManager strategyManager(func, tileOp.getCount(), _enablePrefetchTiling, _mcOptimizationScope,
                                    siblingAnalysis, std::move(layerCostModel), _log.nest());
    _log.debug("Greedy Strategy Assignment");
    auto enableMultiClusterForSWLayer = config::getAvailableExecutor(module, VPU::ExecutorKind::SHAVE_ACT) != nullptr;
    strategyManager.assignMultiClusterStrategy(enableMultiClusterForSWLayer);

    _log.debug("Execute Subgraph Optimization");
    strategyManager.optimizeMulticlusterStrategy();

    _log.trace("Remove Temporary Strategy");
    strategyManager.removeTemporaryMulticlusterStrategy();

    VPU::OpTilingCache::instance().printStats(_log);
}

}  // namespace

//
// createMultiClusterStrategyAssignmentPass
//

std::unique_ptr<mlir::Pass> VPU::createMultiClusterStrategyAssignmentPass(bool enablePrefetchTiling,
                                                                          const int clusteredOpThreshold,
                                                                          StringRef mcOptimizationScope, Logger log) {
    return std::make_unique<MultiClusterStrategyAssignmentPass>(enablePrefetchTiling, clusteredOpThreshold,
                                                                mcOptimizationScope, log);
}
