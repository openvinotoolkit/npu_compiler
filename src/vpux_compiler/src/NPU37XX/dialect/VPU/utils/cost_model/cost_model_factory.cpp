//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model_data.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

namespace vpux {
namespace VPU {
namespace arch37xx {

namespace {

/**
 * @brief Get cost model data for NPU37XX architecture
 *
 * @param isFastModel Whether to use the fast model
 * @return ArrayRef<char>
 */
ArrayRef<char> getCostModelData(bool isFastModel) {
    if (isFastModel) {
        return ArrayRef(VPU::COST_MODEL_2_7_FAST, VPU::COST_MODEL_2_7_FAST_SIZE);
    }
    return ArrayRef(VPU::COST_MODEL_2_7, VPU::COST_MODEL_2_7_SIZE);
}

ArrayRef<char> getCostModelCacheData() {
    return ArrayRef<char>();
}

}  // namespace

std::shared_ptr<VPUNN::VPUCostModel> CostModelFactory::createCostModel() const {
    // Track [E#70055]
    // TODO: Do not switch vpunn model to FAST temporarily, need to investigate the impact for workloads generation pass
    bool isFastModel = false;
    const auto costModelData = getCostModelData(isFastModel);
    const auto costModelCacheData = getCostModelCacheData();
    return std::make_shared<VPUNN::VPUCostModel>(
            costModelData.data(), costModelData.size(), /*copy_model_data*/ false, /*profile*/ false, VPUNN_CACHE_SIZE,
            /*batch_size*/ 1, costModelCacheData.data(), costModelCacheData.size());
}

std::shared_ptr<VPUNN::VPULayerCostModel> CostModelFactory::createLayerCostModel() const {
    // VPUNN provides two models - default and fast.
    // Currently use default model for workload generation. Ticket to explore moving to fast model [E#70055].
    // Currently use fast model for per layer evaluation in multi-cluster strategy selection
    bool isFastModel = true;
    const auto costModelData = getCostModelData(isFastModel);
    const auto costModelCacheData = getCostModelCacheData();
    auto layerCostModel = std::make_shared<VPUNN::VPULayerCostModel>(
            costModelData.data(), costModelData.size(), /*copy_model_data*/ false, /*profile*/ false, VPUNN_CACHE_SIZE,
            /*batch_size*/ 1, costModelCacheData.data(), costModelCacheData.size());
    // keep same per tile workload channel limit on 37XX after new vpunn software update
    layerCostModel->set_maxWorkloadsPerIntraTileSplit(50U);
    return layerCostModel;
}

std::unique_ptr<IShaveCostModelUtils> CostModelFactory::createShaveCostModelUtil() const {
    return std::make_unique<arch37xx::CostModelShaveUtil>(_isShave2ApiUsedInVPUNN);
};

}  // namespace arch37xx
}  // namespace VPU
}  // namespace vpux
