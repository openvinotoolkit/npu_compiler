//
// Copyright (C) 2025 Intel Corporation.
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
    MinimalRequirementsVFScheduling(Logger log, bool prefetching);

    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    VFScenario getType() const override;

protected:
    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     const size_t index) const override;

    bool isSharedWeightsSupported(VFConfig& config) const override;
};
}  // namespace vpux::VPU::VF::v2
