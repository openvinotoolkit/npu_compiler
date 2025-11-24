//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

#include <deque>

namespace vpux::VPU::VF::v2 {
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
      Dependend checks
    */
    const std::deque<std::shared_ptr<IVFScheduling<VFConfig>>>& nextChecks() const override;

    /*
      Add dependend check
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

    /*
      Prefetch output spill
    */
    virtual void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                        const int64_t index, const int64_t tilesNumber) const;

    /*
      Prefetch input dmas
    */
    virtual void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                             const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                             const size_t index) const;

    /*
      Get cost of parent operation
    */
    StrategyCost getParentCost(mlir::Operation* operation,
                               const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost) const;

    /*
      Get cost of common case
    */
    StrategyCost getLinearCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                               const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const;

    /*
      Get input dmas cost
    */
    StrategyCost getPrefetchingCost(mlir::Operation* operation, VFConfig& config,
                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                    const VPUNNCostParameters& parameters, const bool isInput,
                                    const TilingOperationStorage::UPtr& tilingInfo, const int64_t index) const;

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
                                     const size_t index) const;

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
    // the cost of DMAs which VF tile has already had prefetched
    mutable DenseMap<size_t, StrategyCost> _prefetchedCost;

private:
    std::deque<std::shared_ptr<IVFScheduling<VFConfig>>> _dependents;
};
}  // namespace vpux::VPU::VF::v2
