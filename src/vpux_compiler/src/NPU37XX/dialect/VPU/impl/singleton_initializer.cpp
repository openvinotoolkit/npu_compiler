//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"

#include <vpu_cost_model.h>

using namespace vpux::VPU;

void arch37xx::initializeSingletonCache(mlir::MLIRContext* context, std::optional<config::Platform>) {
    const bool isShave2ApiUsedInVPUNN = false;

    setCostModelFactory(context, std::make_unique<arch37xx::CostModelFactory>());
    setShaveCostModelUtils(context, std::make_unique<CostModelShaveUtil>(isShave2ApiUsedInVPUNN, [context]() {
                               auto costModel = getCostModelFactory(context).createCostModel();
                               return costModel->getShaveSupportedOperations(VPUNN::VPUDevice::VPU_2_7);
                           }));
}

void arch37xx::initializePPEVersionConfig(mlir::MLIRContext* context) {
    setPpeFactory(context, std::make_unique<VPU::arch37xx::PpeFactory>());
}
