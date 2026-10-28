//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux::VPU::VF::v2 {
/*
  Scheduling scenario with weights prefetching with last operation
*/
class PrefetchingLastOpVFScheduling : public v2::VFScheduling {
public:
    PrefetchingLastOpVFScheduling(Logger log, bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy);

    VFScenario getType() const override;

protected:
    bool validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                          Byte reservedMemory) const override;
};

}  // namespace vpux::VPU::VF::v2
