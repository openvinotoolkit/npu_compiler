//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <memory>

namespace vpux::VPU::VF::v2 {

class VFScheduling;

struct VFBaselineCostContext final {
    VFBaselineCostContext(mlir::Operation* operation, const VPUNNCostParameters& costParameters,
                          const VPU::LayerVPUNNCost& costFunction)
            : operation(operation), costParameters(costParameters), costFunction(costFunction) {
    }

    mlir::Operation* operation;
    const VPUNNCostParameters& costParameters;
    const VPU::LayerVPUNNCost& costFunction;
    SmallVector<std::optional<ShapeRef>> tileAxes;
    SmallVector<SmallVector<NDTypeInterface>> spillReadTypes;
    SmallVector<bool> untiledOperands;
    SmallVector<NDTypeInterface> spillWriteTypes;
    bool spillReadWriteCanBeOverlapped = false;
    bool multiDimTilingWithChannelTiling = false;
};

class VFCostStrategy {
public:
    virtual ~VFCostStrategy() = default;

    virtual StrategyCost calculate(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                                   const TilingOperationStorage::UPtr& tilingInfo,
                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const = 0;

    virtual bool validate(const VFScheduling& scheduling, VFConfig& config,
                          const TilingOperationStorage::UPtr& tilingInfo, Byte reservedMemory) const = 0;

    virtual StrategyCost calculateBaselineCost(const VFBaselineCostContext& context) const = 0;
};

class PerformanceVFCostStrategy final : public VFCostStrategy {
public:
    StrategyCost calculate(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                           const TilingOperationStorage::UPtr& tilingInfo,
                           const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    bool validate(const VFScheduling& scheduling, VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  Byte reservedMemory) const override;

    StrategyCost calculateBaselineCost(const VFBaselineCostContext& context) const override;
};

class PowerVFCostStrategy final : public VFCostStrategy {
public:
    StrategyCost calculate(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                           const TilingOperationStorage::UPtr& tilingInfo,
                           const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    bool validate(const VFScheduling& scheduling, VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  Byte reservedMemory) const override;

    StrategyCost calculateBaselineCost(const VFBaselineCostContext& context) const override;
};

}  // namespace vpux::VPU::VF::v2
