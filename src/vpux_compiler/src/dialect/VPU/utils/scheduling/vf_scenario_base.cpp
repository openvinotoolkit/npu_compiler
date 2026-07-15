//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/vf_scenario_base.hpp"

using namespace vpux;
using namespace vpux::VPU;

// Out-of-line destructor anchors the vtable in a single translation unit and provides a
// user-defined definition required by the rule of five.
VFScenarioBase::~VFScenarioBase() {
}

bool VFScenarioBase::satisfyMemoryConstraint(mlir::Operation* /*op*/, const Shape& /*nTilesOnDim*/,
                                             VPU::LayerCostModel& /*costModel*/, Logger /*log*/) const {
    // Phase 1 stub (E#202070): VF regions do not use tiling cost analysis yet. Fail fast to expose
    // unintended call paths instead of silently reporting infeasibility.
    VPUX_THROW("VFScenarioBase::satisfyMemoryConstraint is not supported yet");
}

CostInfo VFScenarioBase::calculateCost(mlir::Operation* /*op*/, const OutputTiling& /*tiling*/,
                                       VPU::LayerCostModel& /*costModel*/) const {
    // Phase 1 stub (E#202070): VF regions do not use tiling cost analysis yet. Fail fast to expose
    // unintended call paths instead of silently returning a zeroed cost.
    VPUX_THROW("VFScenarioBase::calculateCost is not supported yet");
}

Byte VFScenarioBase::calculatePeakMemory(mlir::Operation* /*op*/, const OutputTiling& /*tiling*/,
                                         VPU::LayerCostModel& /*costModel*/) const {
    // Phase 1 stub (E#202070): VF regions do not use tiling cost analysis yet. Fail fast to expose
    // unintended call paths instead of silently returning a zeroed footprint.
    VPUX_THROW("VFScenarioBase::calculatePeakMemory is not supported yet");
}
