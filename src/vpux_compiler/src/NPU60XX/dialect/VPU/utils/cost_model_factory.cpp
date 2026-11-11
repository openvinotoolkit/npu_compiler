//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model_data.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <mutex>

namespace vpux {
namespace VPU {
namespace arch60xx {

namespace {

/**
 * @brief Get cost model data for NPU60 architecture
 *
 * @param isFastModel Whether to use the fast model
 * @return ArrayRef<char>
 */
ArrayRef<char> getCostModelData([[maybe_unused]] bool isFastModel) {
    return ArrayRef(VPU::COST_MODEL_6_1, VPU::COST_MODEL_6_1_SIZE);
}

ArrayRef<char> getCostModelCacheData() {
    return ArrayRef(VPU::COST_MODEL_CACHE_6_1, VPU::COST_MODEL_CACHE_6_1_SIZE);
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
void vpux::VPU::arch60xx::CostModelFactory::ensureBundledCostModels() const {
    if (_costModel != nullptr && _layerCostModel != nullptr) {
        // Check if already created models are also bundled
        if (_layerCostModel->get_cost_model_shared() == _costModel) {
            return;  // Already created and bundled
        }
    }

    // Track [E#70055]
    // TODO: Do not switch vpunn model to FAST temporarily, need to investigate the impact for workloads generation pass
    bool isFastModel = false;
    const auto costModelData = getCostModelData(isFastModel);
    const auto costModelCacheData = getCostModelCacheData();
    _costModel = std::make_shared<VPUNN::VPUCostModel>(
            VPUNN::VPUDevice::NPU_6_0, costModelData.data(), costModelData.size(), /*copy_model_data*/ false,
            /*profile*/ false, VPUNN_CACHE_SIZE,
            /*batch_size*/ 1, costModelCacheData.data(), costModelCacheData.size());
    _layerCostModel = std::make_shared<VPUNN::VPULayerCostModel>(_costModel);

    VPUX_THROW_WHEN(_costModel == nullptr || _layerCostModel == nullptr,
                    "Failed to create bundled cost models for NPU60 architecture");
}

}  // namespace arch60xx
}  // namespace VPU
}  // namespace vpux
