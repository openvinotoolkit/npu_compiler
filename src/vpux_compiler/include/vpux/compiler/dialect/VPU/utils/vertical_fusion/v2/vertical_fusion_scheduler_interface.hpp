//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_cost_strategy.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <llvm/ADT/Hashing.h>

#include <deque>

namespace vpux::VPU::VF::v2 {

class VFScheduling;

/*
  Base implementation of scheduling scenario features
*/
class VFScheduling : public IVFScheduling<VFConfig> {
public:
    VFScheduling(Logger log, bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy);
    virtual ~VFScheduling();

    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    /*
      Calculate VF cost
    */
    StrategyCost getCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    /*
      Dependent checks
    */
    const std::deque<std::shared_ptr<IVFScheduling<VFConfig>>>& nextChecks() const override;

    /*
      Add dependent check
    */
    void addNext(std::shared_ptr<IVFScheduling<VFConfig>> check) override;

    virtual bool validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                  const Byte reservedMemory) const = 0;

    /*
      Calculate input sizes
    */
    Byte getInputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const;

    /*
      Calculate output sizes
    */
    Byte getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const;

    /*
      Calculate the peak size of intermediate outputs that are live together.
      An operation may be excluded when its output is accounted for elsewhere.
    */
    Byte getResidentOutputSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                               mlir::Operation* excludedOperation = nullptr) const;

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
    friend class PerformanceVFCostStrategy;
    friend class PowerVFCostStrategy;

    Logger _log;
    bool _prefetching = true;
    std::shared_ptr<const VFCostStrategy> _costStrategy;

private:
    std::deque<std::shared_ptr<IVFScheduling<VFConfig>>> _dependents;
};
}  // namespace vpux::VPU::VF::v2
