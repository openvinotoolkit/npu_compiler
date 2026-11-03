//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/pipelining_vf_scheduling.hpp"

namespace vpux::VPU::VF::v2 {

/*
  Scheduling scenario which overlaps dpu and act shave operations and, on top of that, fetches the inputs of the
  next tile while the current pair is executing. It reuses the pipelining cost model and only tightens the memory
  requirement, so it is limited to subgraphs of two operations placed on different executors.
*/
class PipeliningDoubleBufferingVFScheduling final : public PipeliningVFScheduling {
public:
    PipeliningDoubleBufferingVFScheduling(Logger log, bool prefetching,
                                          std::shared_ptr<const VFCostStrategy> costStrategy);

    VFScenario getType() const override;

    VFPipelineContainer getPipelining(VFConfig& config, int64_t tilesNumber,
                                      const TilingOperationStorage::UPtr& tilingInfo,
                                      const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

protected:
    bool validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                          const Byte reservedMemory = Byte(0)) const override;

    // The scenario hides the input dma of the next tile behind a dpu/act shave pair, which is only meaningful when
    // the subgraph holds exactly two operations and they run on different executors.
    bool isSupportedPattern(const VFConfig& config) const;
};

}  // namespace vpux::VPU::VF::v2
