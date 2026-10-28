//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/weights_prefetch_vf_scheduling.hpp"

namespace vpux::VPU::VF::v2 {

/*
  Scheduling scenario with full prefetching input and output dmas
*/
class FullPrefetchingVFScheduling : public WeightsPrefetchingVFScheduling {
public:
    FullPrefetchingVFScheduling(Logger log, bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy);

    VFScenario getType() const override;

protected:
    bool validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                          Byte reservedMemory) const override;
};
}  // namespace vpux::VPU::VF::v2
