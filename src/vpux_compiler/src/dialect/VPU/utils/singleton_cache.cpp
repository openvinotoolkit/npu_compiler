//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/singleton_cache.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/utils/core/error.hpp"

#include <functional>
#include <memory>

#include <cassert>

using namespace vpux;

namespace vpux {
namespace VPU {

void setCostModelFactory(mlir::MLIRContext* context, std::unique_ptr<ICostModelFactory> costModelFactory) {
    auto& registeredInterface = getCache<VPU::SingletonCache, VPU::VPUDialect>(context);
    registeredInterface.setCostModelFactory(std::move(costModelFactory));
}

const ICostModelFactory& getCostModelFactory(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<VPU::SingletonCache, VPU::VPUDialect>(context);
    return registeredInterface.getCostModelFactory();
}

void setShaveCostModelUtils(mlir::MLIRContext* context, std::unique_ptr<CostModelShaveUtil> shaveCostModelUtils) {
    auto& registeredInterface = getCache<VPU::SingletonCache, VPU::VPUDialect>(context);
    registeredInterface.setShaveCostModelUtils(std::move(shaveCostModelUtils));
}

const CostModelShaveUtil& getShaveCostModelUtils(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<VPU::SingletonCache, VPU::VPUDialect>(context);
    return registeredInterface.getShaveCostModelUtils();
}

PrecomputedStrategyTable& getPrecomputedStrategyTable(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<VPU::SingletonCache, VPU::VPUDialect>(context);
    return registeredInterface.getPrecomputedStrategyTable();
}

}  // namespace VPU
}  // namespace vpux
