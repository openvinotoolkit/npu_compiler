//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/tiling_scenario_interface.hpp"

namespace vpux::VPU {

// Base class for Vertical Fusion (LoopType::VF) compute region schedulers.
// Shares the TilingScenarioInterface contract with temporal tiling scenarios.
// Phase 1 (E#202070): VF regions do not use tiling cost analysis yet, so the cost/memory
// methods provide trivial defaults here; derived VF scenarios only implement getName and
// getScheduleStrategy. VF-specific shared implementation can be added here in later phases.
class VFScenarioBase : public TilingScenarioInterface {
public:
    VFScenarioBase() = default;
    virtual ~VFScenarioBase();
    VFScenarioBase(const VFScenarioBase&) = delete;
    VFScenarioBase& operator=(const VFScenarioBase&) = delete;
    VFScenarioBase(VFScenarioBase&&) = delete;
    VFScenarioBase& operator=(VFScenarioBase&&) = delete;

    llvm::StringRef getName() const override = 0;
    LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                           vpux::AddressType memorySize) const override = 0;

    // Phase 1 stub (E#202070): VF regions do not use tiling cost analysis yet
    bool satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim, VPU::LayerCostModel& costModel,
                                 Logger log) const override;
    CostInfo calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                           VPU::LayerCostModel& costModel) const override;
    Byte calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                             VPU::LayerCostModel& costModel) const override;
};

}  // namespace vpux::VPU
