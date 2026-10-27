//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"  // for getVPUDeviceType
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model_data.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <mutex>

namespace vpux {
namespace VPU {
namespace arch50xx {

namespace {

/**
 * @brief Get cost model data for NPU50 architecture
 *
 * @param isFastModel Whether to use the fast model
 * @return ArrayRef<char>
 */
ArrayRef<char> getCostModelData([[maybe_unused]] bool isFastModel,
                                [[maybe_unused]] std::optional<config::Platform> platform) {
    if (platform.has_value()) {
        if (platform.value() == config::Platform::NPU5020) {
            return ArrayRef(VPU::COST_MODEL_5_2, VPU::COST_MODEL_5_2_SIZE);
        }
    }
    return ArrayRef(VPU::COST_MODEL_5_1, VPU::COST_MODEL_5_1_SIZE);
}

ArrayRef<char> getCostModelCacheData([[maybe_unused]] std::optional<config::Platform> platform) {
    if (platform.has_value()) {
        if (platform.value() == config::Platform::NPU5020) {
            return ArrayRef(VPU::COST_MODEL_CACHE_5_2, VPU::COST_MODEL_CACHE_5_2_SIZE);
        }
    }
    return ArrayRef(VPU::COST_MODEL_CACHE_5_1, VPU::COST_MODEL_CACHE_5_1_SIZE);
}

}  // namespace

std::shared_ptr<VPUNN::VPUCostModel> CostModelFactory::createCostModel() const {
    std::lock_guard<std::mutex> lock(_mutex);
    ensureBundledCostModels();
    return _costModel;
}

std::shared_ptr<VPUNN::VPULayerCostModel> CostModelFactory::createLayerCostModel() const {
    std::lock_guard<std::mutex> lock(_mutex);
    ensureBundledCostModels();
    return _layerCostModel;
}

// Note: This method is not inherently thread-safe and assumes _mutex is already held by the caller
void vpux::VPU::arch50xx::CostModelFactory::ensureBundledCostModels() const {
    if (_costModel != nullptr && _layerCostModel != nullptr) {
        // Check if already created models are also bundled
        if (_layerCostModel->get_cost_model_shared() == _costModel) {
            return;  // Already created and bundled
        }
    }

    // If both are missing or just layer exists, create paired instances
    bool isFastModel = false;  // no fast model for npu50xx
    const auto costModelData = getCostModelData(isFastModel, _platformOpt);
    const auto costModelCacheData = getCostModelCacheData(_platformOpt);
    const config::ArchKind archKind = config::ArchKind::NPU50XX;
    const VPUNN::VPUDevice device =
            _platformOpt.has_value() ? getVPUDeviceType(_platformOpt.value()) : getVPUDeviceType(archKind);
    _costModel = std::make_shared<VPUNN::VPUCostModel>(
            device, costModelData.data(), costModelData.size(), /*copy_model_data*/ false, /*profile*/ false,
            VPUNN_CACHE_SIZE, /*batch_size*/ 1, costModelCacheData.data(), costModelCacheData.size());
    _layerCostModel = std::make_shared<VPUNN::VPULayerCostModel>(_costModel);

    VPUX_THROW_WHEN(_costModel == nullptr || _layerCostModel == nullptr,
                    "Failed to create bundled cost models for NPU50 architecture");
}

}  // namespace arch50xx
}  // namespace VPU
}  // namespace vpux
