//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_scenario_base.hpp"

namespace vpux::VPU {

class TemporalTilingDriver {
public:
    // Get all valid tiling strategies for a given scenario and operation
    static std::unordered_map<llvm::StringRef, SmallVector<Shape>> getAllValidTilingStrategies(
            mlir::Operation* op, const std::vector<std::unique_ptr<TemporalTilingScenarioBase>>& scenarios,
            VPU::LayerCostModel& costModel, Logger log,
            const TemporalTilingSearchSpaceConfig& config = getDefaultTemporalTilingSearchSpaceConfig());

    // Find the best tiling strategy across all scenarios and dimension combinations
    static std::optional<TemporalTilingInfo> getBestTilingStrategy(
            VPU::TilingBuilderOpInterface op, VPU::LayerCostModel& costModel, Logger log = Logger::global(),
            const TemporalTilingSearchSpaceConfig& config = getDefaultTemporalTilingSearchSpaceConfig());
};

}  // namespace vpux::VPU
