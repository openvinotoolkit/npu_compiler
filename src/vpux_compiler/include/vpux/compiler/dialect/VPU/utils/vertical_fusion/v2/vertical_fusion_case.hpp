//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v2 {
/*
  Candidate for VF
*/

class VFCase final {
public:
    using VFConfigType = VFConfig;

    /*
     Constructor of VF case
    */
    VFCase(VFConfigType& config, const VFSplit& split);

    /*
     Destructor of VF case
    */
    ~VFCase();

    /*
     Move constructor
    */
    VFCase(VFCase&& vfCase);

    /*
     Move assignment operator
    */
    VFCase& operator=(VFCase&& other);

    /*
     Copy constructor
    */
    VFCase(const VFCase& vfCase);

    /*
     Copy assignment operator
    */
    VFCase& operator=(const VFCase& other);

    /*
     Set number of tiles for given dimension
    */
    void setTilingNumber(Dim dim, int64_t number);

    /*
     Set VF scheduling
    */
    void setScheduling(std::shared_ptr<IVFScheduling<VFConfigType>> vfScheduling);

    /*
     Set VF tiling storage
    */
    void setTilingStorage(std::unique_ptr<TilingOperationStorage> vfStorage);

    /*
     Get VF cost calculated based on the cost function and current scheduling
    */
    StrategyCost getCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

    /*
     Get cached VF cost if available
    */
    StrategyCost getCost() const;

    /*
     Check if VF case has been initialized with scheduling
    */
    bool isInitialized();

    /*
     Get VF config
    */
    VFConfigType& getConfig();

    /*
     Get selected VF scheduling scenario
    */
    std::optional<VFScenario> getScenario() const;

    /*
     Generate VF tiling
    */
    mlir::ArrayAttr getTiling();

    /*
     Set Scheduling and tiling to VF
    */
    void approveScheduling();

    /*
     Get VF tiling storage smart pointer (ownership is retained by VFCase)
    */
    const TilingOperationStorage::UPtr& getTilingStorage() const;

private:
    /*
    Add CMX write spills
    */
    void addCMXWriteSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

    /*
    Add CMX read spills
    */
    void addCMXReadSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

    /*
     Clear cached data
    */
    void clearCache();

    /*
     VF data
    */
    VFConfigType _config;

    /*
     VF Split
    */
    VFSplit _split;

    /*
     VF Scheduling
    */
    std::shared_ptr<IVFScheduling<VFConfigType>> _vfScheduling;

    /*
     VF TilingOperationStorage
    */
    std::unique_ptr<TilingOperationStorage> _vfTilingStorage;

    /*
     Cached VF cost
    */
    std::optional<StrategyCost> _cachedCost;
};
}  // namespace vpux::VPU::VF::v2
