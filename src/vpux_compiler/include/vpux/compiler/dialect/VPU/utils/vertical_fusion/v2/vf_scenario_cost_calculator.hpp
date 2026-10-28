//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"

#include <functional>
#include <memory>

namespace vpux::VPU::VF::v2 {

// Keeps the derived tiling state for one parent VF while evaluating prefetch and spill costs.
// The cache avoids rebuilding VFConfig and tiling information for the same parent across tiles.
struct ParentVFTilingInfo final {
    ParentVFTilingInfo(mlir::Operation* parentOp, std::unique_ptr<VFCacheAnalysis> cache,
                       std::unique_ptr<VFConfig> config, TilingOperationStorage::UPtr tilingInfo)
            : parentOp(parentOp),
              cache(std::move(cache)),
              config(std::move(config)),
              tilingInfo(std::move(tilingInfo)) {
    }

    mlir::Operation* parentOp = nullptr;
    std::unique_ptr<VFCacheAnalysis> cache;
    std::unique_ptr<VFConfig> config;
    TilingOperationStorage::UPtr tilingInfo;
};

using ParentVFTilingInfoCache = SmallVector<ParentVFTilingInfo, 2>;

/*
 The context shared by a scenario cost calculation.
*/
struct VFScenarioCostCalculationContext final {
    VFScenarioCostCalculationContext(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                                     const TilingOperationStorage::UPtr& tilingInfo,
                                     const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log,
                                     bool prefetching)
            : scheduling(scheduling),
              config(config),
              tilesNumber(tilesNumber),
              tilingInfo(tilingInfo),
              costFunction(costFunction),
              log(log),
              prefetching(prefetching) {
    }

    const VFScheduling& scheduling;
    VFConfig& config;
    int64_t tilesNumber;
    const TilingOperationStorage::UPtr& tilingInfo;
    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction;
    Logger log;
    bool prefetching;
};

class VFScenarioCostCalculator {
public:
    // Function type used to calculate the cost of a data type and optional shape.
    using TypeCostFunction = llvm::function_ref<StrategyCost(NDTypeInterface, const std::optional<ShapeRef>&)>;

    virtual ~VFScenarioCostCalculator() = default;

    // Calculates the total cost for the vertical-fusion scenario.
    virtual StrategyCost calculate(VFScenarioCostCalculationContext& context) const = 0;

    // Returns timeline intervals associated with the scenario, if available.
    virtual SmallVector<TimelineInterval> getTimeIntervals(VFScenarioCostCalculationContext&) const {
        return {};
    }

    // Hashes an operation's cost-model parameters for cost-cache lookup.
    static llvm::hash_code getStrategyCostHash(mlir::Operation* operation, const VPUNNCostParameters& parameters);

    // Builds cost-model parameters from an operation's output tiling and input tiles.
    static VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                               const SmallVector<TileInfo>& inputTiles);

    // Builds cost-model parameters from a stored tiling entry.
    static VPUNNCostParameters fillInCostParam(mlir::Operation* operation,
                                               const TilingOperationStorage::UPtr& opStorage, size_t index);

    // Calculates an operation's strategy cost using the supplied configuration and cost model.
    static StrategyCost getStrategyCost(VFConfig& config, mlir::Operation* operation,
                                        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                        const VPUNNCostParameters& parameters);

    // Retrieves the isolated cost associated with a parent operation.
    static StrategyCost getParentCost(mlir::Operation* operation,
                                      const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost);

    // Calculates the DMA cost of prefetching an operation input or output.
    static StrategyCost getPrefetchingCost(const VFScheduling& scheduling, mlir::Operation* operation, VFConfig& config,
                                           TypeCostFunction typeCostFunction, const VPUNNCostParameters& parameters,
                                           bool isInput, const TilingOperationStorage::UPtr& tilingInfo, int64_t index,
                                           ParentVFTilingInfoCache& parentVFTilingInfoCache);

    // Calculates DMA cost for a view-like operation, when such a cost applies.
    static std::optional<StrategyCost> getViewLikeOpDMACost(mlir::Operation* operation, VFConfig& config,
                                                            const TilingOperationStorage::UPtr& tilingInfo,
                                                            size_t tileIdx,
                                                            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction);

    // Calculates the cost of copying an internal slice for an input or output tile.
    static StrategyCost getInternalSliceCopyCost(mlir::Operation* operation, VFConfig& config,
                                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                 const VPUNNCostParameters& parameters, bool isInput,
                                                 const TilingOperationStorage::UPtr& tilingInfo, int64_t index);

    // Reduces spill cost by the portion covered by already accumulated prefetch cost.
    static void reduceCostWithPrefetchedDMA(StrategyCost& parentCost, const StrategyCost& prefetchCost,
                                            StrategyCost& accumulatedPrefetchCost);
};

/*
The performance-oriented scenario cost calculator.
*/
class PerformanceVFScenarioCostCalculator : public VFScenarioCostCalculator {
public:
    PerformanceVFScenarioCostCalculator() = default;

    // Calculates scenario cost using execution performance and DMA estimates.
    StrategyCost calculate(VFScenarioCostCalculationContext& context) const override;

    // Builds the execution timeline for the scenario.
    SmallVector<TimelineInterval> getTimeIntervals(VFScenarioCostCalculationContext& context) const override;

protected:
    // Adjusts output spill cost for the selected prefetching strategy.
    virtual void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                        SmallVector<StrategyCost>& prefetchCostList, int64_t index,
                                        int64_t tilesNumber) const;

    // Adjusts input prefetch cost for the selected prefetching strategy.
    virtual void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                             const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                             SmallVector<StrategyCost>& prefetchCostList, size_t index) const;
};

class MinimalVFScenarioCostCalculator final : public PerformanceVFScenarioCostCalculator {
public:
    using PerformanceVFScenarioCostCalculator::PerformanceVFScenarioCostCalculator;

protected:
    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     SmallVector<StrategyCost>& prefetchCostList, size_t index) const override;
};

class LastOpPrefetchingVFScenarioCostCalculator final : public PerformanceVFScenarioCostCalculator {
public:
    using PerformanceVFScenarioCostCalculator::PerformanceVFScenarioCostCalculator;

protected:
    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     SmallVector<StrategyCost>& prefetchCostList, size_t index) const override;
};

class WeightsPrefetchingVFScenarioCostCalculator final : public PerformanceVFScenarioCostCalculator {
public:
    using PerformanceVFScenarioCostCalculator::PerformanceVFScenarioCostCalculator;

protected:
    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     SmallVector<StrategyCost>& prefetchCostList, size_t index) const override;
};

class FullPrefetchingVFScenarioCostCalculator final : public PerformanceVFScenarioCostCalculator {
public:
    using PerformanceVFScenarioCostCalculator::PerformanceVFScenarioCostCalculator;

protected:
    void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                SmallVector<StrategyCost>& prefetchCostList, int64_t index,
                                int64_t tilesNumber) const override;

    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     SmallVector<StrategyCost>& prefetchCostList, size_t index) const override;
};

class PipeliningVFScenarioCostCalculator final : public PerformanceVFScenarioCostCalculator {
public:
    using PerformanceVFScenarioCostCalculator::PerformanceVFScenarioCostCalculator;

    StrategyCost calculate(VFScenarioCostCalculationContext& context) const override;

    SmallVector<TimelineInterval> getTimeIntervals(VFScenarioCostCalculationContext& context) const override;

    VFPipelineContainer getPipelining(VFScenarioCostCalculationContext& context) const;
};

class PerformanceVFScenarioCostCalculatorFactory final {
public:
    static std::unique_ptr<VFScenarioCostCalculator> create(VFScenario scenario);
};

/*
The power-oriented VF scenario cost calculator.
*/
class PowerVFScenarioCostCalculator final : public VFScenarioCostCalculator {
public:
    StrategyCost calculate(VFScenarioCostCalculationContext& context) const override;
};

// Calculates timeline intervals for a vertical-fusion scenario.
SmallVector<TimelineInterval> getVFTimeIntervals(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                                                 const TilingOperationStorage::UPtr& tilingInfo,
                                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log,
                                                 bool prefetching);

}  // namespace vpux::VPU::VF::v2
