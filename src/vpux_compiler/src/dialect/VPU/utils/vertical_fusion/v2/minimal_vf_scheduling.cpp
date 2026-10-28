//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/minimal_vf_scheduling.hpp"

namespace vpux::VPU::VF::v2 {
MinimalRequirementsVFScheduling::MinimalRequirementsVFScheduling(Logger log, bool prefetching,
                                                                 std::shared_ptr<const VFCostStrategy> costStrategy)
        : v2::VFScheduling(log, prefetching, std::move(costStrategy)) {
}

bool MinimalRequirementsVFScheduling::validateScenario(VFConfig&, const TilingOperationStorage::UPtr& tilingInfo,
                                                       Byte) const {
    return tilingInfo != nullptr;
}

VFScenario MinimalRequirementsVFScheduling::getType() const {
    return VFScenario::MINIMAL;
}

bool MinimalRequirementsVFScheduling::isSharedWeightsSupported(VFConfig&) const {
    return false;
}

}  // namespace vpux::VPU::VF::v2
