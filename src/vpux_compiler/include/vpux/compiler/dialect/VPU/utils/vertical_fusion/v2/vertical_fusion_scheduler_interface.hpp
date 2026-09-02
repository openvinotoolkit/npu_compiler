//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <llvm/ADT/Hashing.h>

#include <deque>

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

llvm::hash_code getStrategyCostHash(mlir::Operation* operation, const VPUNNCostParameters& parameters);

/*
  Base implementation of scheduling scenario features
*/
class VFScheduling : public IVFScheduling<VFConfig> {
public:
    VFScheduling(Logger log, bool prefetching = true);
    virtual ~VFScheduling() = default;

    /*
      Calculate VF cost
    */
    StrategyCost getCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    /*
      Get all time intervals
    */
    virtual SmallVector<TimelineInterval> getTimeIntervals(
            VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const;

    /*
      Dependent checks
    */
    const std::deque<std::shared_ptr<IVFScheduling<VFConfig>>>& nextChecks() const override;

    /*
      Add dependent check
    */
    void addNext(std::shared_ptr<IVFScheduling<VFConfig>> check) override;

protected:
    /*
      Calculate input sizes
    */
    Byte getInputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const;

    /*
      Calculate output sizes
    */
    Byte getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const;

    /*
      Get parameters for cost calculation
    */
    VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                        const SmallVector<TileInfo>& inputTiles) const;

    /*
      Get parameters for cost calculation
    */
    VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const TilingOperationStorage::UPtr& opStorage,
                                        size_t index) const;

    StrategyCost getStrategyCost(VFConfig& config, mlir::Operation* operation,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                 const VPUNNCostParameters& parameters) const;

    /*
      Prefetch output spill
    */
    virtual void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                        SmallVector<StrategyCost>& prefetchCostList, const int64_t index,
                                        const int64_t tilesNumber) const;

    /*
      Prefetch input dmas
    */
    virtual void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                             const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                             SmallVector<StrategyCost>& prefetchCostList, const size_t index) const;

    /*
      Get cost of parent operation
    */
    StrategyCost getParentCost(mlir::Operation* operation,
                               const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost) const;

    /*
      calculate related timeline interval of common case
    */
    VFLinearContainer calculateLinearTimeIntervals(VFConfig& config, int64_t tilesNumber,
                                                   const TilingOperationStorage::UPtr& tilingInfo,
                                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const;

    /*
      Get input DMA cost
    */
    StrategyCost getPrefetchingCost(mlir::Operation* operation, VFConfig& config,
                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                    const VPUNNCostParameters& parameters, const bool isInput,
                                    const TilingOperationStorage::UPtr& tilingInfo, const int64_t index,
                                    ParentVFTilingInfoCache& parentVFTilingInfoCache) const;

    /*
      Get output DMA cost
    */
    StrategyCost getOutputSpillCost(mlir::Operation* operation, VFConfig& config,
                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                    const VPUNNCostParameters& parameters, const bool isOutput) const;

    /*
      Get internal slice copy dma cost
    */
    StrategyCost getInternalSliceCopyCost(mlir::Operation* operation, VFConfig& config,
                                          const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                          const VPUNNCostParameters& parameters, const bool isInput,
                                          const TilingOperationStorage::UPtr& tilingInfo, const int64_t index) const;

    /*
      Reduce the cost with already prefetched dmas from previous tile
    */
    void reduceCostWithPrefetchedDMA(StrategyCost& parentCost, const StrategyCost& prefetchCost,
                                     StrategyCost& accumulatedPrefetchCost) const;

    /*
      Calculate the DMA copy cost for a view-like op with different input and output sizes.
      Returns std::nullopt if no DMA copy is needed.
    */
    std::optional<StrategyCost> getViewLikeOpDMACost(mlir::Operation* operation, VFConfig& config,
                                                     const TilingOperationStorage::UPtr& tilingInfo, size_t tileIdx,
                                                     const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const;

    /*
      Check if the VF can support shared weights
    */
    virtual bool isSharedWeightsSupported(VFConfig& config) const;

    /*
      Calculate shared size across different tiles
    */
    Byte calculateSharedSize(VFConfig& config, mlir::Operation* operation,
                             const vpux::VPU::VFOperationTiling& inputOutputTiling) const;

    /*
      Calculate total shared size cross different tiles
    */
    Byte getSharedSizeByAllTiles(ArrayRef<mlir::Operation*> operations, VFConfig& config,
                                 const TilingOperationStorage::UPtr& tilingInfo) const;

protected:
    Logger _log;
    bool _prefetching = true;

private:
    std::deque<std::shared_ptr<IVFScheduling<VFConfig>>> _dependents;
};
}  // namespace vpux::VPU::VF::v2
