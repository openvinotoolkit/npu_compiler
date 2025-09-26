//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace VPUNN {
class VPUCostModel;
class VPULayerCostModel;
}  // namespace VPUNN

namespace vpux {
namespace VPU {

static constexpr unsigned int VPUNN_CACHE_SIZE = 8192U;

/**
 * @brief Interface for creating VPUNN Cost Models
 */
class ICostModelFactory {
public:
    virtual ~ICostModelFactory() = default;

    /**
     * @brief Create a VPUNN Cost Model
     *
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    virtual std::shared_ptr<VPUNN::VPUCostModel> createCostModel() const = 0;

    /**
     * @brief Create a VPUNN Layer Cost Model
     *
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    virtual std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel() const = 0;
};

}  // namespace VPU
}  // namespace vpux
