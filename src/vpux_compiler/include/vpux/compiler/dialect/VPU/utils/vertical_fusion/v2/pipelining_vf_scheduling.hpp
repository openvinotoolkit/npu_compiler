//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v2 {
/*
  Scheduling scenario for pipelining dpu and act shave operations
*/
class PipeliningVFScheduling : public VFScheduling, public IVFPipelinedScheduling<VFConfig> {
public:
    PipeliningVFScheduling(Logger log, bool prefetching);

    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    VFScenario getType() const override;

    StrategyCost getCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    VFPipelineContainer getPipelining(VFConfig& config, int64_t tilesNumber,
                                      const TilingOperationStorage::UPtr& tilingInfo,
                                      const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    SmallVector<TimelineInterval> getTimeIntervals(
            VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

protected:
    void addOutputSpill(VFConfig& config, mlir::Operation* operation, VFPipelineContainer& pipelinedStructure,
                        int64_t index, const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                        const VPUNNCostParameters& costParameters) const;

    bool isSharedWeightsSupported(VFConfig& config) const override;

    SmallVector<VPU::ExecutorKind> getExecutorForVFOps(ArrayRef<mlir::Operation*> ops) const;
};

}  // namespace vpux::VPU::VF::v2
