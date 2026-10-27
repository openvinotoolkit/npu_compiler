//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v2 {
/*
  Scheduling scenario with minimal requirements
*/
class MinimalRequirementsVFScheduling : public VFScheduling {
public:
    MinimalRequirementsVFScheduling(Logger log, bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy);

    VFScenario getType() const override;

protected:
    bool validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                          Byte reservedMemory) const override;

    bool isSharedWeightsSupported(VFConfig& config) const override;
};
}  // namespace vpux::VPU::VF::v2
