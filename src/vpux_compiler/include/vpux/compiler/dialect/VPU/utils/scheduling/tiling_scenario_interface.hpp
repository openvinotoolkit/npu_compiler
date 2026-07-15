//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"

#include <llvm/ADT/StringRef.h>

#include <memory>

namespace vpux::VPU {

class LayerCostModel;

// Pure-virtual contract consumed by generateLoopSchedules() to produce a predefined loop schedule
// for a single compute region. Implementations specialize cost/feasibility queries and schedule
// production per LoopType (temporal tiling scenarios for LoopType::Tiling, VF scenarios for
// LoopType::VF).
class TilingScenarioInterface {
public:
    TilingScenarioInterface() = default;
    virtual ~TilingScenarioInterface();
    TilingScenarioInterface(const TilingScenarioInterface&) = delete;
    TilingScenarioInterface& operator=(const TilingScenarioInterface&) = delete;
    TilingScenarioInterface(TilingScenarioInterface&&) = delete;
    TilingScenarioInterface& operator=(TilingScenarioInterface&&) = delete;

    // Returns a human-readable identifier for this scheduling scenario
    virtual llvm::StringRef getName() const = 0;
    // Checks whether the given tile count per dimension fits within CMX memory for this scenario
    virtual bool satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim, VPU::LayerCostModel& costModel,
                                         Logger log) const = 0;
    // Estimates the execution cost (DMA + compute + stall cycles) for the given tiling
    virtual CostInfo calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                                   VPU::LayerCostModel& costModel) const = 0;
    // Computes the peak CMX memory footprint required to execute the given tiling
    virtual Byte calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                                     VPU::LayerCostModel& costModel) const = 0;
    // Determines the loop schedule for a given compute region
    virtual LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                                   vpux::AddressType memorySize) const = 0;
};

}  // namespace vpux::VPU
