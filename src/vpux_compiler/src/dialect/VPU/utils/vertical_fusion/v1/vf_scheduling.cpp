//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <deque>

namespace vpux::VPU::VF::v1 {
VFScheduling::VFScheduling(Logger log, bool prefetching /*true*/): _log(log), _prefetching(prefetching) {
}

const std::deque<std::shared_ptr<IVFScheduling<VFConfig>>>& VFScheduling::nextChecks() const {
    return _dependents;
}

void VFScheduling::addNext(std::shared_ptr<IVFScheduling<VFConfig>> check) {
    _dependents.emplace_back(check);
}

Byte VFScheduling::getInputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    const auto index = 0;
    auto inputSize = Byte(0);

    for (auto op : config.getInputs()) {
        auto tileInfo = tilingInfo->get(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1} {2}", index,
                        *op, config.getSubgraph());

        auto tileTypes = config.getOperationTypes(op, tileInfo.value().second, tileInfo.value().first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There are not enough types for tile of operation {0}", *op);

        // exclude output type information
        tileTypes.pop_back();
        for (auto type : tileTypes) {
            inputSize += type.getTotalAllocSize();
        }
    }

    return inputSize;
}

Byte VFScheduling::getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    auto outputSize = Byte(0);
    const auto index = 0;

    for (auto op : config.getOutputs()) {
        auto tileInfo = tilingInfo->get(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        auto tileTypes = config.getOperationTypes(op, tileInfo.value().second, tileInfo.value().first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There is no output type for tile of operation {0}", *op);

        auto type = tileTypes.back();
        outputSize += type.getTotalAllocSize();
    }

    return outputSize;
}

VPUNNCostParameters VFScheduling::fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                                  const SmallVector<TileInfo>& inputTiles) const {
    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
        mcStrategy = mcOperation.getMultiClusterStrategy().value_or(mcStrategy);
    }

    auto mode = TilingMode::ISOLATED;
    SmallVector<OutputTiling> inputAllTiles;
    if (!inputTiles.empty()) {
        inputAllTiles.push_back(inputTiles);
    }
    return VPUNNCostParameters(mcStrategy, tiling, mode, inputAllTiles, false);
}

VPUNNCostParameters VFScheduling::fillInCostParam(mlir::Operation* operation,
                                                  const TilingOperationStorage::UPtr& opStorage, size_t index) const {
    auto inputOutputTiling = opStorage->get(operation, index);

    OutputTiling outputTiling;
    SmallVector<TileInfo> inputTiling;

    if (inputOutputTiling.has_value()) {
        outputTiling = {inputOutputTiling.value().second};
        inputTiling = inputOutputTiling.value().first.tiles;
    }

    return fillInCostParam(operation, outputTiling, inputTiling);
}

bool hasPrefetchedDMA(mlir::Operation* oper, mlir::BlockArgument arg, VFConfig& config, const bool isInput) {
    if (oper->hasTrait<VPU::EltwiseOp>() && oper->getNumOperands() > 1) {
        return true;
    }

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(oper);

    if (nceOp != nullptr && (arg != nceOp.getWeightsOperand() && arg != nceOp->getOperand(0))) {
        return false;
    }

    if (isInput) {
        return true;
    }

    auto parentOp = config.getSubgraph().getOperand(arg.getArgNumber()).getDefiningOp<Const::DeclareOp>();

    if (parentOp == nullptr) {
        return false;
    }

    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        return false;
    }

    return arg == nceOp.getWeightsOperand();
}

mlir::Operation* findParent(mlir::Value operand) {
    auto parent = operand.getDefiningOp();

    while (parent != nullptr && mlir::isa<VPU::TilingViewLikeOpInterface>(parent)) {
        parent = parent->getOperand(0).getDefiningOp();
    }

    return parent;
}

StrategyCost VFScheduling::getParentCost(mlir::Operation* operation,
                                         const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost) const {
    StrategyCost parentCost = 0;
    auto* parent = findParent(operation->getOperand(0));
    if (parent == nullptr && (operation->getNumOperands() > 1 && operation->hasTrait<VPU::EltwiseOp>())) {
        parent = findParent(operation->getOperand(1));
    }
    if (parent != nullptr) {
        auto foundCost = isolatedOperCost.find(parent);
        VPUX_THROW_WHEN(foundCost == isolatedOperCost.end(), "Cannot find the cost for {0}", *parent);
        parentCost = foundCost->second;
    }

    return parentCost;
}

void VFScheduling::correctOutputSpillCost(StrategyCost& /*spillCost*/, VFConfig& /*config*/,
                                          const DenseMap<mlir::Operation*, StrategyCost>& /*isolatedOperCost*/,
                                          const int64_t /*index*/, const int64_t /*tilesNumber*/) const {
}

void VFScheduling::correctInputPrefetchingCost(StrategyCost& /*prefetchCost*/, mlir::Operation* /*operation*/,
                                               VFConfig& /*config*/,
                                               const DenseMap<mlir::Operation*, StrategyCost>& /*isolatedOperCost*/,
                                               const size_t /*index*/) const {
}

StrategyCost VFScheduling::getCost(VFConfig& config, int64_t tilesNumber,
                                   const TilingOperationStorage::UPtr& tilingInfo,
                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    return getLinearCost(config, tilesNumber, tilingInfo, costFunction);
}

StrategyCost VFScheduling::getPrefetchingCost(mlir::Operation* operation, VFConfig& config,
                                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                              const VPUNNCostParameters& parameters, const bool isInput,
                                              const TilingOperationStorage::UPtr& tilingInfo,
                                              const int64_t index) const {
    StrategyCost prefetchedCost = 0;
    auto inputTiling = tilingInfo->get(operation, index);
    if (!inputTiling.has_value()) {
        return prefetchedCost;
    }
    for (auto input : operation->getOperands() | indexed) {
        if (input.index() >= inputTiling.value().first.tiles.size()) {
            break;
        }
        auto inputOperand = input.value();
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(inputOperand)) {
            if (hasPrefetchedDMA(operation, blockArg, config, isInput)) {
                prefetchedCost += costFunction->getSpillingTypeCost(
                        config.getOperationTypes(operation, parameters._tiling[0],
                                                 parameters._operandsTiling[0])[input.index()],
                        parameters._operandsTiling[0][input.index()].axis);
            }
        }
    }

    return prefetchedCost;
}

StrategyCost VFScheduling::getLinearCost(VFConfig& config, int64_t tilesNumber,
                                         const TilingOperationStorage::UPtr& tilingInfo,
                                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    StrategyCost fullCost = 0;
    auto inputs = config.getInputs();
    DenseMap<mlir::Operation*, StrategyCost> isolatedOperCost;
    for (auto index : irange(tilesNumber)) {
        for (auto* op : config.getOperationsForTiling()) {
            auto costParameters = fillInCostParam(op, tilingInfo, index);

            // isolated operation cost
            auto isolatedCost = costFunction->getStrategyCost(op, costParameters);
            if (isolatedCost >= std::numeric_limits<StrategyCost>::max()) {
                _log.warning("Invalid VPUNN cost");
                return isolatedCost;
            }
            isolatedOperCost[op] = isolatedCost;
            fullCost += isolatedCost;

            if (llvm::find(config.getOutputs(), op) != config.getOutputs().end() && tilesNumber > 1) {
                // add the cost of output dma
                auto spillCost = costFunction->getSpillingTypeCost(
                        config.getOperationTypes(op, costParameters._tiling[0], costParameters._operandsTiling[0])
                                .back(),
                        costParameters._tiling[0].axis);
                if (spillCost >= std::numeric_limits<StrategyCost>::max()) {
                    _log.warning("Invalid VPUNN cost");
                    return spillCost;
                }
                correctOutputSpillCost(spillCost, config, isolatedOperCost, index, tilesNumber);
                fullCost += spillCost;
            }

            const bool isInput = llvm::find(inputs, op) != inputs.end();
            StrategyCost prefetchedCost =
                    getPrefetchingCost(op, config, costFunction, costParameters, isInput, tilingInfo, index);

            if (prefetchedCost >= std::numeric_limits<StrategyCost>::max()) {
                _log.warning("Invalid VPUNN cost");
                return prefetchedCost;
            }
            if (_prefetching && prefetchedCost > 0) {
                correctInputPrefetchingCost(prefetchedCost, op, config, isolatedOperCost, index);
            }

            fullCost += prefetchedCost;
        }
    }

    return fullCost;
}

}  // namespace vpux::VPU::VF::v1
