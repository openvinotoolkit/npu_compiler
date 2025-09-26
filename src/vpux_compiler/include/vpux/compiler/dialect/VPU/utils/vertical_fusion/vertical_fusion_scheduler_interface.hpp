//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

#include <deque>

namespace vpux {
namespace VPU {

/*
  Interface to provide scheduling scenarios for VF
*/
template <typename VFConfigType>
class IVFScheduling {
public:
    virtual ~IVFScheduling() = default;

    /*
      Validate memory requirements
    */
    virtual bool validate(VFConfigType& config, const TilingOperationStorage::UPtr& tilingInfo,
                          const Byte reservedMemory = Byte(0)) const = 0;

    /*
      Calculate VF cost
    */
    virtual StrategyCost getCost(VFConfigType& config, int64_t tilesNumber,
                                 const TilingOperationStorage::UPtr& tilingInfo,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const = 0;

    /*
      Type of scenario
    */
    virtual VFScenario getType() const = 0;

    /*
      Dependend checks
    */
    virtual const std::deque<std::shared_ptr<IVFScheduling>>& nextChecks() const = 0;

    /*
      Add dependend check
    */
    virtual void addNext(std::shared_ptr<IVFScheduling> check) = 0;
};

/*
  Interface to provide pipelined
*/
template <typename VFConfigType>
class IVFPipelinedScheduling {
public:
    virtual ~IVFPipelinedScheduling() = default;

    /*
      Get the structure of pipelined operations
    */
    virtual VFPipelineContainer getPipelining(VFConfigType& config, int64_t tilesNumber,
                                              const TilingOperationStorage::UPtr& tilingInfo,
                                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const = 0;
};
}  // namespace VPU
}  // namespace vpux
