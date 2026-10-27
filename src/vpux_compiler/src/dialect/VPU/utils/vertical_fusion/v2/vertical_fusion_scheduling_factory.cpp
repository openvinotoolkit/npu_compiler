//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/full_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/minimal_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/pipelining_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/prefetch_lastop_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/weights_prefetch_vf_scheduling.hpp"

namespace vpux::VPU::VF::v2 {
VFSchedulingFactory::VFSchedulingFactory(bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy)
        : _prefetching(prefetching), _costStrategy(std::move(costStrategy)) {
}

std::shared_ptr<IVFScheduling<VFConfig>> VFSchedulingFactory::createVFScenario(VFScenario scenarioCode,
                                                                               Logger log) const {
    switch (scenarioCode) {
    case VFScenario::MINIMAL: {
        return std::make_shared<VPU::VF::v2::MinimalRequirementsVFScheduling>(log, _prefetching, _costStrategy);
    }
    case VFScenario::LASTOP_PREFETCHING: {
        return std::make_shared<VPU::VF::v2::PrefetchingLastOpVFScheduling>(log, _prefetching, _costStrategy);
    }
    case VFScenario::WEIGHTS_PREFETCHING: {
        return std::make_shared<VPU::VF::v2::WeightsPrefetchingVFScheduling>(log, _prefetching, _costStrategy);
    }
    case VFScenario::FULL_PREFETCHING: {
        return std::make_shared<VPU::VF::v2::FullPrefetchingVFScheduling>(log, _prefetching, _costStrategy);
    }
    case VFScenario::VF_PIPELINING: {
        return std::make_shared<VPU::VF::v2::PipeliningVFScheduling>(log, _prefetching, _costStrategy);
    }
    default: {
        VPUX_THROW("No scheduling implemented for {0}", scenarioCode);
    }
    }
}
}  // namespace vpux::VPU::VF::v2
