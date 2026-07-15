//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_driver.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/isolated_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/prefetch_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"

namespace vpux::VPU {

// Core logic for full search tiling space
// Search valid tile counts on one dimension while keeping all other dimensions fixed in baseTiles.
//
// Increasing the tile count reduces per-tile memory. Therefore each scenario becomes feasible after a threshold, and
// the next scenario starts when its own memory constraint becomes feasible.
// More advanced scenario is always better than lower scenarios when the tiling number is the same. So each tiling
// number only belongs to one scenario. The search walks the inclusive range [minTiles, maxTiles] once, partitioning it
// into scenario-specific candidate intervals:
//
//   minTiles                                                           maxTiles
//     |---------------------------------------------------------------------|
//     |  memory too high for ISO |  ISOLATED |
//     |--------------------------|-----------|
//                                 first PREFETCH feasible
//     |        memory too high for PREFETCH  | PREFETCH  |
//     |--------------------------------------|-----------|
//                                             first PIPELINE feasible  inefficient(end)
//     |              memory too high for PIPELINE        | PIPELINE options |
//     |--------------------------------------------------|------------------|
//
// For each scenario:
//   1. advance left until the current scenario satisfies its memory constraint;
//   2. record candidates for that scenario until the next scenario becomes feasible.
//
// The last scenario has no next scenario. Instead, collection stops once further tiling is
// considered inefficient by config.lastScenarioSearchStopPredicate.
// Returns an empty map if no scenario has valid candidates within the inclusive maxTiles bound.
static std::unordered_map<llvm::StringRef, SmallVector<int64_t>> findTileOptionsForDim(
        mlir::Operation* op, const std::vector<std::unique_ptr<TemporalTilingScenarioBase>>& scenarios,
        const Shape& baseTiles, Dim tileDim, int64_t minTiles, int64_t maxTiles, Dim dimToAlign, int64_t dimAlignment,
        ShapeRef outputShape, VPU::LayerCostModel& costModel, Logger log,
        const TemporalTilingSearchSpaceConfig& config) {
    if (maxTiles <= 1) {
        return {};
    }

    Shape right = baseTiles;
    right[tileDim] = maxTiles;
    Shape left = baseTiles;
    left[tileDim] = minTiles;

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outShape = getBoundedShape(outputType);
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    VPUX_THROW_WHEN(config.lastScenarioSearchStopPredicate == nullptr,
                    "Last scenario search stop predicate must be set");

    std::unordered_map<llvm::StringRef, SmallVector<int64_t>> validOptionsMap;
    for (const auto& scenarioIdx : irange(scenarios.size())) {
        const auto& scenario = scenarios[scenarioIdx];
        const bool isLastScenario = (scenarioIdx == scenarios.size() - 1);

        // Phase 1: advance left until current scenario's memory constraint is satisfied
        while (left[tileDim] <= right[tileDim]) {
            if (mlir::failed(
                        ensureNTilesIsCompatibleWithMultiClusterUp(op, left, tileDim, outputShape, right.raw(), log))) {
                break;
            }
            if (scenario->satisfyMemoryConstraint(op, left, costModel, log)) {
                break;
            }
            dimPlus(left, tileDim, dimToAlign, dimAlignment, outputShape, right.raw(), log);
        }
        // Phase 2: collect valid tile options for this scenario until the next one starts
        // For the last scenario (now it's PipelineTiling), terminate when tiling becomes inefficient according to
        // config.lastScenarioSearchStopPredicate. This reuses the last scenario's own calculatePeakMemory with a
        // stricter threshold, replacing the former InefficientTiling sentinel class.
        while (left[tileDim] <= right[tileDim]) {
            if (mlir::failed(
                        ensureNTilesIsCompatibleWithMultiClusterUp(op, left, tileDim, outputShape, right.raw(), log))) {
                break;
            }
            if (isLastScenario) {
                auto tiles = fillDividedTiles(op, left, outShape);
                if (mlir::succeeded(tiles) && tiles.value().size() >= 2) {
                    const auto peakMemory = scenario->calculatePeakMemory(op, tiles.value(), costModel);
                    if (config.lastScenarioSearchStopPredicate(peakMemory, cmxSize, config)) {
                        break;
                    }
                }
            } else {
                if (scenarios[scenarioIdx + 1]->satisfyMemoryConstraint(op, left, costModel, log)) {
                    break;
                }
            }
            validOptionsMap[scenario->getName()].push_back(left[tileDim]);
            dimPlus(left, tileDim, dimToAlign, dimAlignment, outputShape, right.raw(), log);
        }
    }

    return validOptionsMap;
}

// Enumerate valid temporal tiling shapes grouped by scenario name.
// Iteration hierarchy:
//   maxNumTiles from getTilingMaxLimits()
//     tileable dimensions under this maxNumTiles
//       Phase 1: 1D search; Phase 2: 2D search; Phase 3: 3D fallback
std::unordered_map<llvm::StringRef, SmallVector<Shape>> TemporalTilingDriver::getAllValidTilingStrategies(
        mlir::Operation* op, const std::vector<std::unique_ptr<TemporalTilingScenarioBase>>& scenarios,
        VPU::LayerCostModel& costModel, Logger log, const TemporalTilingSearchSpaceConfig& config) {
    if (vpux::isCostInaccurate(op)) {
        return {};
    }

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!tilingBuilder) {
        return {};
    }

    const auto dimAlignInfo = getAlignDimAndSize(op);
    auto dimToAlign = dimAlignInfo.first;
    auto dimAlignment = dimAlignInfo.second;

    auto minNumTiles = getMinNumTiles(op);
    auto validDims = getDimsForTiling(op, log);
    const auto outputShape = getBoundedShape(op->getResult(0));
    const auto maxTilingLimits = getTilingMaxLimits(op, config);

    std::unordered_map<llvm::StringRef, SmallVector<Shape>> validStrategiesMap;

    for (const auto& limit : maxTilingLimits) {
        const auto maxNumTiles = limit;

        // Filter to only tileable dimensions (max > 1 and allowed)
        SmallVector<Dim> tileableDims;
        for (const auto& dim : validDims) {
            if (maxNumTiles[dim.ind()] > 1) {
                tileableDims.push_back(dim);
            }
        }

        if (tileableDims.empty()) {
            continue;
        }

        Shape baseTiles(outputShape.size(), 1);

        // Helper to check duplicates
        auto isDuplicate = [&](llvm::StringRef scenarioName, const Shape& shape) {
            const auto it = validStrategiesMap.find(scenarioName);
            if (it == validStrategiesMap.end()) {
                return false;
            }

            return llvm::any_of(it->second, [&](const Shape& existing) {
                return existing == shape;
            });
        };

        auto addIfValid = [&](llvm::StringRef scenarioName, const Shape& tiles) {
            if (isDuplicate(scenarioName, tiles)) {
                return;
            }

            if (scenarioName != "ISOLATED") {
                bool hasTiling = llvm::any_of(tiles, [](int64_t t) {
                    return t > 1;
                });
                if (!hasTiling) {
                    return;
                }
            }

            // Reject strategies where any tile dimension exceeds VPU_DIMENSION_LIMIT
            auto tilingResult = fillDividedTiles(op, tiles, outputShape);
            if (mlir::failed(tilingResult) || tilingResult.value().empty()) {
                return;
            }

            const auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(op);
            for (const auto& tile : tilingResult.value()) {
                if (VPU::NCEInvariant::hasDimensionExceedingVPULimit(tile.shape)) {
                    return;
                }
                if (isNCEOp && !VPU::doesNCEOpChannelSatisfyWorkload(op, tile)) {
                    return;
                }
            }

            validStrategiesMap[scenarioName].push_back(tiles);
        };

        auto collectStrategiesForDim = [&](const Shape& baseShape, Dim dim) {
            const auto tileOptions =
                    findTileOptionsForDim(op, scenarios, baseShape, dim, minNumTiles[dim.ind()], maxNumTiles[dim.ind()],
                                          dimToAlign, dimAlignment, outputShape, costModel, log, config);

            for (const auto& [scenarioName, options] : tileOptions) {
                for (const auto& option : options) {
                    if (option > 0) {
                        Shape tiles = baseShape;
                        tiles[dim] = option;
                        addIfValid(scenarioName, tiles);
                    }
                }
            }
        };

        // === PHASE 1: Try single-dimension tiling (1D) ===
        // For each dimension, find minimum tiles needed
        for (const auto& dim : tileableDims) {
            collectStrategiesForDim(baseTiles, dim);
        }

        // If we found valid 1D strategies, we might not need multi-dim (legacy tiling strategy)
        // But we still explore 2D for potentially better cost

        // === PHASE 2: Try two-dimension tiling (2D) ===
        // Strategy: For each primary dim at max, find min tiles on secondary dim
        for (size_t i = 0; i < tileableDims.size(); ++i) {
            Dim dim1 = tileableDims[i];

            // Set dim1 to max
            Shape tilesWithPrimaryMax = baseTiles;
            tilesWithPrimaryMax[dim1] = maxNumTiles[dim1.ind()];
            ensureNTilesIsCompatibleWithMultiClusterDown(op, tilesWithPrimaryMax, dim1, outputShape, log);
            if (tilesWithPrimaryMax[dim1] < 2) {
                continue;
            }

            // Try each secondary dimension
            for (size_t j = 0; j < tileableDims.size(); ++j) {
                if (i == j) {
                    continue;
                }

                Dim secondaryDim = tileableDims[j];
                collectStrategiesForDim(tilesWithPrimaryMax, secondaryDim);
            }
        }

        // === PHASE 3: Try three-dimension tiling (3D) - only if 2D insufficient ===
        // Only explore if we have enough dimensions and previous phases didn't find solutions
        if (tileableDims.size() >= 3 && validStrategiesMap.empty()) {
            for (size_t i = 0; i < tileableDims.size(); ++i) {
                Dim dim1 = tileableDims[i];
                // Set dim1 to max
                Shape tilesWithOneMax = baseTiles;
                tilesWithOneMax[dim1] = maxNumTiles[dim1.ind()];
                ensureNTilesIsCompatibleWithMultiClusterDown(op, tilesWithOneMax, dim1, outputShape, log);
                if (tilesWithOneMax[dim1] < 2) {
                    continue;
                }

                for (size_t j = 0; j < tileableDims.size(); ++j) {
                    if (i == j) {
                        continue;
                    }
                    Dim dim2 = tileableDims[j];
                    // Set dim2 to max
                    Shape tilesWithTwoMax = tilesWithOneMax;
                    tilesWithTwoMax[dim2] = maxNumTiles[dim2.ind()];
                    ensureNTilesIsCompatibleWithMultiClusterDown(op, tilesWithTwoMax, dim2, outputShape, log);
                    if (tilesWithTwoMax[dim2] < 2) {
                        continue;
                    }

                    for (size_t k = 0; k < tileableDims.size(); ++k) {
                        if (k == i || k == j) {
                            continue;
                        }
                        Dim dim3 = tileableDims[k];

                        collectStrategiesForDim(tilesWithTwoMax, dim3);
                    }
                }
            }
        }
    }

    return validStrategiesMap;
}

std::optional<TemporalTilingInfo> TemporalTilingDriver::getBestTilingStrategy(
        VPU::TilingBuilderOpInterface op, VPU::LayerCostModel& costModel, Logger log,
        const TemporalTilingSearchSpaceConfig& config) {
    if (vpux::isCostInaccurate(op.getOperation())) {
        // [E#219502] Support for temporal tiling on cost-inaccurate ops can be added in the future
        // but currently we skip to avoid misleading results and potential regressions.
        return std::nullopt;
    }

    const auto outputShape = getBoundedShape(op->getResult(0));
    if (outputShape.size() != 4 && outputShape.size() != 5) {
        return std::nullopt;
    }

    auto& cache = VPU::getGlobalOpTilingCache();
    auto baseHash = cache.calculateOpHash(op);
    // Include output distribution mode in hash to avoid cache collisions between
    // ops with identical shapes/attrs but different consumer graphs that produce
    // different distribution modes (SEGMENTED vs DUPLICATED|SEGMENTED).
    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation())) {
        auto mcStrategy = clusteredOp.getMultiClusterStrategy();
        if (mcStrategy.has_value()) {
            auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
            auto outputDistMode = VPU::getOutputTensorDistributionMode(clusteredOp, mcStrategy.value(), outputType);
            baseHash = llvm::hash_combine(baseHash, static_cast<uint64_t>(mcStrategy.value()),
                                          static_cast<uint64_t>(outputDistMode));
        }
    }
    const auto opHash = baseHash;
    auto cachedResult = cache.getTemporalTilingInfo(opHash);
    if (cachedResult.has_value()) {
        return cachedResult.value();
    }

    // Create scenarios
    std::vector<std::unique_ptr<TemporalTilingScenarioBase>> temporalTilingScenarios;
    temporalTilingScenarios.push_back(std::make_unique<IsolatedTiling>());
    temporalTilingScenarios.push_back(std::make_unique<PrefetchTiling>());
    temporalTilingScenarios.push_back(std::make_unique<PipelineTiling>());

    std::unordered_map<llvm::StringRef, TemporalTilingScenarioBase*> scenarioMap;
    for (const auto& scenario : temporalTilingScenarios) {
        scenarioMap[scenario->getName()] = scenario.get();
    }

    if (log.isActive(LogLevel::Trace)) {
        log.trace("Op: {0}", op->getLoc());
        auto strategy = getMultiClusterStrategyFromOp(op.getOperation());
        if (strategy.has_value()) {
            log.trace("MC strategy: {0}", strategy.value());
        }
        for (auto operand : op.getOperation()->getOperands()) {
            auto ndType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
            log.nest().trace("Input shape: {0}, element size: {1}", ndType.getShape(),
                             vpux::getElemTypeSize(ndType.getElementType()));
        }
        for (auto result : op.getOperation()->getResults()) {
            auto ndType = mlir::cast<vpux::NDTypeInterface>(result.getType());
            log.nest().trace("Output shape: {0}, element size: {1}", ndType.getShape(),
                             vpux::getElemTypeSize(ndType.getElementType()));
        }
        log.trace("Searching for optimal temporal tiling strategy");
    }

    const auto validStrategies =
            getAllValidTilingStrategies(op.getOperation(), temporalTilingScenarios, costModel, log, config);

    SmallVector<TemporalTilingInfo> strategies;
    for (const auto& [scenarioName, options] : validStrategies) {
        log.nest(2).trace("Scenario: {0}, validStrategies: {1}", scenarioName, options.size());
        const auto scenario = scenarioMap[scenarioName];
        for (const auto& nTilesOnDim : options) {
            auto tilingResult = fillDividedTiles(op.getOperation(), nTilesOnDim, outputShape);

            if (mlir::failed(tilingResult) || tilingResult.value().empty()) {
                continue;
            }

            auto costInfo = scenario->calculateCost(op.getOperation(), tilingResult.value(), costModel);
            log.nest(3).trace(
                    "Scenario: {0}, Tiling: {1}, DMA cost: {2}, Compute cost: {3}, Overall cost: {4}, Efficiency {5}",
                    scenario->getName(), nTilesOnDim, costInfo.dmaCost, costInfo.computeCost, costInfo.overallCost,
                    costInfo.efficiency);

            strategies.push_back(TemporalTilingInfo{nTilesOnDim, costInfo, scenario->getName()});
        }
    }

    // Find best performance strategy
    std::optional<TemporalTilingInfo> bestStrategy;
    const auto bestIt = std::min_element(strategies.begin(), strategies.end(),
                                         [](const TemporalTilingInfo& a, const TemporalTilingInfo& b) {
                                             if (a.costInfo.overallCost != b.costInfo.overallCost) {
                                                 return a.costInfo.overallCost < b.costInfo.overallCost;
                                             }
                                             // Make equal-cost selection deterministic for stable blob generation.
                                             return a.tilingShape < b.tilingShape;
                                         });
    if (bestIt != strategies.end()) {
        bestStrategy = *bestIt;
    }

    if (bestStrategy.has_value()) {
        const auto& [tilingShape, tilingCost, tilingScenario] = bestStrategy.value();
        log.nest().warning("Total search space {0}. Best strategy: {1} {2} DMA cost {3} Compute cost {4} Total cost "
                           "{5} Efficiency {6}",
                           strategies.size(), tilingScenario, tilingShape, tilingCost.dmaCost, tilingCost.computeCost,
                           tilingCost.overallCost, tilingCost.efficiency);
    } else {
        log.nest().trace("Failed to find tiling strategy");
    }

    cache.updateTemporalTilingInfo(opHash, bestStrategy);
    return bestStrategy;
}

}  // namespace vpux::VPU
