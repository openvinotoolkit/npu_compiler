//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mutex>

namespace vpux::VPU {

/**
 * @brief Static class for encapsulating cost model factory-related objects.
 *
 * This class manages the singleton instance of the cost model factory,
 * ensuring thread safety and proper initialization.
 */
class CostModelConfig {
private:
    static std::map<config::ArchKind, std::unique_ptr<ICostModelFactory>>& _getFactories();
    static std::map<config::ArchKind, std::unique_ptr<IShaveCostModelUtils>>& _getCMShaveUtils();

    static std::mutex& _getCostModelFactoryMutex() {
        static std::mutex mtx;
        return mtx;
    }

    /**
     * @brief Get the factory for the specified architecture
     *
     * @param arch Architecture kind
     * @return const ICostModelFactory&
     */
    static const ICostModelFactory& getFactory(config::ArchKind arch);

public:
    /**
     * @brief Set the factory for the specified architecture
     *
     * @param arch Architecture kind
     */
    static void setFactory(config::ArchKind arch);

    /**
     * @brief Set the cmUtils for the specified architecture
     *
     * @param arch Architecture kind
     */
    static void setCMShaveUtils(config::ArchKind arch);

    /**
     * @brief Create a cost model for the specified architecture
     *
     * @param arch Architecture kind
     * @return std::shared_ptr<VPUNN::VPUCostModel>
     */
    static std::shared_ptr<VPUNN::VPUCostModel> createCostModel(config::ArchKind arch) {
        return getFactory(arch).createCostModel();
    }

    /**
     * @brief Create a layer cost model for the specified architecture
     *
     * @param arch Architecture kind
     * @return std::shared_ptr<VPUNN::VPULayerCostModel>
     */
    static std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(config::ArchKind arch) {
        return getFactory(arch).createLayerCostModel();
    }

    /**
     * @brief Get the Shave Cost Model Utils Interface for the specified architecture
     *
     * @param arch Architecture kind
     * @return const IShaveCostModelUtils&
     */
    static const IShaveCostModelUtils& getShaveCostModelUtilsInterface(config::ArchKind arch);
};

}  // namespace vpux::VPU
