//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/NPU50XX/dialect/VPU/utils/cost_model_factory.hpp"

#include <vpu_cost_model.h>

using namespace vpux::config;
namespace vpux::VPU {

std::map<ArchKind, std::unique_ptr<ICostModelFactory>>& CostModelConfig::_getFactories() {
    static std::map<ArchKind, std::unique_ptr<ICostModelFactory>> factories;
    return factories;
}

/**
 * @brief Set the factory for the specified architecture
 *
 * @param arch Architecture kind
 */
void CostModelConfig::setFactory(ArchKind arch) {
    std::lock_guard lock(_getCostModelFactoryMutex());
    auto& factories = _getFactories();

    // create it if not existed
    if (factories.find(arch) == factories.end()) {
        switch (arch) {
        case ArchKind::NPU37XX:
            factories[arch] = std::make_unique<arch37xx::CostModelFactory>();
            break;
        case ArchKind::NPU40XX:
            factories[arch] = std::make_unique<arch40xx::CostModelFactory>();
            break;
        case ArchKind::NPU50XX:
            factories[arch] = std::make_unique<arch50xx::CostModelFactory>();
            break;
        default:
            VPUX_THROW("Unsupported VPU arch type: '{0}'", arch);
        }

        // Check if already initialized with the correct factory type
        Logger::global().debug("Set CostModelFactory instance for architecture {0}", arch);
    }
}

const ICostModelFactory& CostModelConfig::getFactory(ArchKind arch) {
    auto& factories = _getFactories();

    VPUX_THROW_UNLESS(factories[arch] != nullptr, "CostModelFactory for architecture {0} is not initialized", arch);

    return *factories[arch];
}

std::map<ArchKind, std::unique_ptr<IShaveCostModelUtils>>& CostModelConfig::_getCMShaveUtils() {
    static std::map<ArchKind, std::unique_ptr<IShaveCostModelUtils>> cmShaveUtils;
    return cmShaveUtils;
}

/**
 * @brief Set the cmUtils for the specified architecture
 * Should be used only once the factories are created, it depends on the factories to be instantiated first.
 *
 * @param arch Architecture kind
 */
void CostModelConfig::setCMShaveUtils(ArchKind arch) {
    std::lock_guard lock(_getCostModelFactoryMutex());
    auto& cmShaveUtils = _getCMShaveUtils();

    // create it if not existed
    if (cmShaveUtils.find(arch) == cmShaveUtils.end()) {
        auto& factory = getFactory(arch);
        cmShaveUtils[arch] = factory.createShaveCostModelUtil();

        // Check if already initialized with the correct shave util type
        Logger::global().debug("Set CostModelShaveUtil instance for architecture {0}", arch);
    }
}

const IShaveCostModelUtils& CostModelConfig::getShaveCostModelUtilsInterface(config::ArchKind arch) {
    auto& cmUtils = _getCMShaveUtils();

    VPUX_THROW_UNLESS(cmUtils[arch] != nullptr, "CostModelShaveUtil for architecture {0} is not initialized", arch);

    return *cmUtils[arch];
}

}  // namespace vpux::VPU
