//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils_interface.hpp"

#include <mutex>

namespace vpux {
namespace VPU {
namespace arch50xx {

/**
 * @brief NPU50 implementation of the Cost Model Factory
 */
class CostModelFactory final : public ICostModelFactory {
public:
    /**
     * @brief Create a VPUNN Cost Model for NPU50 architecture
     *
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    std::shared_ptr<VPUNN::VPUCostModel> createCostModel() const override;

    /**
     * @brief Create a VPUNN Layer Cost Model for NPU50 architecture
     *
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel() const override;

    /**
     * @brief Will create a unique pointer to the Shave Cost Model Utils
     *
     * @return std::unique_ptr<IShaveCostModelUtils>
     */
    std::unique_ptr<IShaveCostModelUtils> createShaveCostModelUtil() const override;

private:
    /**
     * @brief Creates bundled VPUNN Cost Model and Layer Cost Model
     *
     * This method ensures that both the Cost Model (l1) and Layer Cost Model (l2)
     * The Layer Cost Model (l2) depends on the Cost Model (l1) such that it shares the same state (eg. cache and cache
     * counter)
     */
    void ensureBundledCostModels() const;

    mutable std::shared_ptr<VPUNN::VPUCostModel> _costModel{nullptr};            // cached cost model instance
    mutable std::shared_ptr<VPUNN::VPULayerCostModel> _layerCostModel{nullptr};  // cached layer cost model instance
    mutable std::mutex _mutex;  // mutex to protect access to the cost model instances

    static constexpr bool _isShave2ApiUsedInVPUNN{
            false};  // cached flag indicating if Shave2Api is used should be set here
};

}  // namespace arch50xx
}  // namespace VPU
}  // namespace vpux
