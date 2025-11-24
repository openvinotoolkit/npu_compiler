//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils_interface.hpp"

namespace vpux {
namespace VPU {
namespace arch37xx {

/**
 * @brief NPU37XX implementation of the Cost Model Factory
 */
class CostModelFactory final : public ICostModelFactory {
public:
    /**
     * @brief Create a VPUNN Cost Model for NPU37XX architecture
     *
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    std::shared_ptr<VPUNN::VPUCostModel> createCostModel() const override;

    /**
     * @brief Create a VPUNN Layer Cost Model for NPU37XX architecture
     *
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel() const override;

    /**
     * @brief Each Cost Model Factory knows from compile time if the Shave2Api will be used
     *
     * @return boolean that shave2ApiIsUsed
     */
    std::unique_ptr<IShaveCostModelUtils> createShaveCostModelUtil() const override;

private:
    static constexpr bool _isShave2ApiUsedInVPUNN{
            false};  // cached flag indicating if Shave2Api is used should be set here
};

}  // namespace arch37xx
}  // namespace VPU
}  // namespace vpux
