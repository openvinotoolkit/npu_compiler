//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_cost_strategy_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPU::VF::v2 {

/*
  Factory which creates VF scheduling scenario
*/

class VFSchedulingFactory {
public:
    VFSchedulingFactory(bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy =
                                                  VFCostStrategyFactory::create(VFMergeCostMode::PERFORMANCE));
    /*
      create scheduling scenario
    */
    std::shared_ptr<IVFScheduling<VFConfig>> createVFScenario(VFScenario scenarioCode, Logger log) const;

private:
    bool _prefetching = true;
    std::shared_ptr<const VFCostStrategy> _costStrategy;
};

}  // namespace vpux::VPU::VF::v2
