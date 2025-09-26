//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <vpu_cost_model.h>

using namespace vpux;

CycleCostInfo::CycleCostInfo(mlir::func::FuncOp func)
        : CycleCostInfo(VPU::CostModelConfig::createCostModel(config::getArch(func)), func) {
}

CycleCostInfo::CycleCostInfo(std::shared_ptr<VPUNN::VPUCostModel> costModel, mlir::func::FuncOp func)
        : _log(Logger::global().nest("cycle-cost-info", 0)) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    _archKind = config::getArch(module);
    _costModel = std::move(costModel);

    _log.trace("Analyze cycle cost for Function '@{0}'", func.getName());
    _log = _log.nest();
}

void CycleCostInfo::updateAndStoreInvalidCostCycles(size_t& cycleCost, mlir::Operation* op) {
    if (cycleCost < VPU::INVALID_COST_BASE && cycleCost != VPU::NO_COST) {
        return;
    }

    if (cycleCost >= VPU::INVALID_COST_BASE) {
        auto layerTypeStr = op->getName().getStringRef().str();
        // Store kernel name
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            layerTypeStr += "." + swKernelOp.getKernelFunction().getLeafReference().str();
        }

        _numOfTasksWithInvalidCost++;
        _layersWithInvalidCost.insert(layerTypeStr);
        _log.debug("Layer {0} has invalid cost - '{1}'. Assume cycleCost = {2} for op at '{3}'", layerTypeStr,
                   cycleCost, VPU::UNIT_COST, op->getLoc());
    }
    cycleCost = VPU::UNIT_COST;
}

void CycleCostInfo::storeCycleCost(size_t& cycleCost, mlir::Operation* op) {
    // Sanity check
    if (_cycleCosts.find(op) != _cycleCosts.end()) {
        _log.trace("Cost for mlir::Block already stored '{0}'", _cycleCosts[op]);
    }
    // Update the cost
    updateAndStoreInvalidCostCycles(cycleCost, op);
    _cycleCosts.insert({op, cycleCost});
}

size_t CycleCostInfo::getCycleCost(mlir::Operation* op) {
    if (op == nullptr) {
        return VPU::NO_COST;
    }

    auto costInterface = mlir::dyn_cast<VPUIP::CycleCostInterface>(op);
    if (costInterface == nullptr) {
        return VPU::NO_COST;
    }

    if (_cycleCosts.find(op) != _cycleCosts.end()) {
        _log.trace("Cost for mlir::Operation already stored '{0}'", _cycleCosts[op]);
        return _cycleCosts[op];
    }

    _log.trace("Cost for mlir::Operation not found in cache, querying costModel");
    size_t cycleCost = costInterface.getOperationCycleCost(_costModel);
    storeCycleCost(cycleCost, op);
    return cycleCost;
}

void CycleCostInfo::resetNNCacheCounter() {
    _costModel->getPreloadedCacheCounter().reset();
}

void CycleCostInfo::printNNCacheStatistics(Logger& log) const {
    log.info("[NN Cache statistics]  {0}", _costModel->getPreloadedCacheCounter().printString());
}
