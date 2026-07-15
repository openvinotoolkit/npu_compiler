//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model_data.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"

#include <vpu_cost_model.h>

using namespace vpux::VPU;

void arch50xx::initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform> platform) {
    const bool isShave2ApiUsedInVPUNN = false;

    setCostModelFactory(context, std::make_unique<arch50xx::CostModelFactory>(platform));
    setShaveCostModelUtils(context, std::make_unique<CostModelShaveUtil>(isShave2ApiUsedInVPUNN, [context]() {
                               auto costModel = getCostModelFactory(context).createCostModel();
                               return costModel->getShaveSupportedOperations(VPUNN::VPUDevice::NPU_5_0);
                           }));
#ifdef VPUX_BUILTIN_PRECOMPUTED_STRATEGY_TABLE_5_1
    getPrecomputedStrategyTable(context).setBuiltinTable(
            reinterpret_cast<const uint8_t*>(VPU::PRECOMPUTED_STRATEGY_TABLE_5_1),
            VPU::PRECOMPUTED_STRATEGY_TABLE_5_1_SIZE);
#endif
}

void arch50xx::initializePPEVersionConfig(mlir::MLIRContext* context) {
    setPpeFactory(context, std::make_unique<VPU::arch50xx::PpeFactory>());
}
