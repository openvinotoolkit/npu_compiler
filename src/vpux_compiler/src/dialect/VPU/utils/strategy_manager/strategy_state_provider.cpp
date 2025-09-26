//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_state_provider.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

#include <numeric>

using namespace vpux::VPU;

// coefficients for reannealing. How much we should pass to reanneal
const auto reannealingCoefficients = {0.25, 0.025};

std::unordered_map<MultiClusterStrategy, MultiClusterStrategy> strategyMatch = {
        {MultiClusterStrategy::SplitOverHeight, MultiClusterStrategy::HKSwitch},
        {MultiClusterStrategy::HKSwitch, MultiClusterStrategy::SplitOverKernel},
        {MultiClusterStrategy::Clustering, MultiClusterStrategy::SplitOverKernel},
        {MultiClusterStrategy::SplitOverKernel, MultiClusterStrategy::Clustering},
        {MultiClusterStrategy::SplitOverHeightOverlapped, MultiClusterStrategy::SplitOverHeight},
        {MultiClusterStrategy::SplitOverHeightOverlapped, MultiClusterStrategy::HKSwitch}};
// TODO: add VPU4 related strategies

OperationStrategy DefaultStateProvider::randomOperation(ArrayRef<mlir::Operation*> operations) {
    std::uniform_int_distribution<> opDistribution(0, operations.size() - 1);
    const auto chosenOp = operations[opDistribution(_generator)];
    auto currentStrategy = _storage->getCurrentStrategy(chosenOp);
    return std::make_pair(chosenOp, currentStrategy);
}

OperationStrategy DefaultStateProvider::getState(int temperature, double& cost, const OperationStrategy* const state) {
    if (state == nullptr) {
        initializeTemperature(temperature);
        reannealingStep(temperature, cost);
        const auto allOperations = _storage->getAllOperations();
        VPUX_THROW_WHEN(allOperations.empty(), "There are no operations added in this state");

        return randomOperation(allOperations.getArrayRef());
    }

    auto allStrategies = _storage->getAllStrategies(state->first);
    auto chosenStrategy = allStrategies[0].strategy;

    if (allStrategies.size() != 1) {
        std::uniform_int_distribution<> strategyDistribution(0, allStrategies.size() - 1);
        do {
            chosenStrategy = allStrategies[strategyDistribution(_generator)].strategy;
        } while (chosenStrategy == state->second);
    }
    return std::make_pair(state->first, chosenStrategy);
}

StrategyCost DefaultStateProvider::getCost(const OperationStrategy& state) {
    auto* operation = state.first;

    // cost of operation is calculated as isolated cost of operation in current state + transition cost
    // between operation and its parents and users

    if (_neighbours.count(operation) == 0) {
        fillInNeighbours(operation);
    }

    auto isolatedCost = _storage->getStrategyCost(state);

    auto fullTransitionCost = accumulateCost(_neighbours[operation].first, state, true);
    fullTransitionCost += accumulateCost(_neighbours[operation].second, state, false);

    return isolatedCost + fullTransitionCost;
}

StrategyCost DefaultStateProvider::accumulateCost(ArrayRef<mlir::Operation*> neighbours, const OperationStrategy& state,
                                                  bool parent) {
    if (neighbours.empty()) {
        return 0;
    }
    return std::accumulate(std::begin(neighbours), std::end(neighbours), 0, [&](StrategyCost cost, auto* neighbour) {
        StrategyCost transitionCost = 0;

        if (neighbour != nullptr) {
            if (_storage->hasAnyStrategy(neighbour)) {
                auto strategy = _storage->getCurrentStrategy(neighbour);
                const auto neighbourState = std::make_pair(neighbour, strategy);

                transitionCost =
                        parent ? getTransitionCost(neighbourState, state) : getTransitionCost(state, neighbourState);
            } else {
                transitionCost = getTransitionOutsideCost(state, neighbour, parent);
            }
        }

        return cost + transitionCost;
    });
}

void DefaultStateProvider::initializeTemperature(int temperature) {
    if (_initialTemperature.has_value()) {
        return;
    }

    _initialTemperature = temperature;
    for (auto value : reannealingCoefficients) {
        _reannealingTemperatures.push(value * _initialTemperature.value());
    }
}

void DefaultStateProvider::reannealingStep(int temperature, double& cost) {
    if (_reannealingTemperatures.empty() || _reannealingTemperatures.front() != temperature) {
        return;
    }

    _reannealingTemperatures.pop();
    // flush all best states to current state
    for (auto* operation : _storage->getAllOperations()) {
        _storage->setCurrentStrategy(std::make_pair(operation, _storage->getBestStrategy(operation)));
    }

    cost = getFullCost();
}

void DefaultStateProvider::fillInNeighbours(mlir::Operation* operation) {
    const auto transitionFilter = [&](mlir::Operation* op) -> bool {
        return op != nullptr && !mlir::isa<Const::DeclareOp>(op);
    };

    auto parents = SmallVector<mlir::Operation*>(operation->getOperands() |
                                                 transformed([this](auto operand) -> mlir::Operation* {
                                                     return getParentOp(operand);
                                                 }) |
                                                 filtered(transitionFilter));

    SmallVector<mlir::Operation*> consumers;
    getConsumersOp(consumers, operation);

    _neighbours[operation] = std::make_pair(parents, consumers);
}

void DefaultStateProvider::updateState(const OperationStrategy& state) {
    _storage->setCurrentStrategy(state);
}

void DefaultStateProvider::updateSolution(const OperationStrategy& state) {
    // flush all current states to best state
    for (auto* operation : _storage->getAllOperations()) {
        _storage->setBestStrategy(std::make_pair(operation, _storage->getCurrentStrategy(operation)));
    }
    _storage->setBestStrategy(state);
}

StrategyCost DefaultStateProvider::getFullCost() {
    auto allOperations = _storage->getAllOperations();

    StrategyCost fullCost = 0;

    if (allOperations.empty()) {
        return fullCost;
    }

    llvm::SetVector<mlir::Operation*> passedOp;

    const auto filter = [&](auto* oper) {
        return !passedOp.contains(oper);
    };

    for (auto* operation : allOperations) {
        if (operation == nullptr) {
            continue;
        }

        if (_neighbours.count(operation) == 0) {
            fillInNeighbours(operation);
        }

        const auto currentState = std::make_pair(operation, _storage->getCurrentStrategy(operation));
        fullCost += _storage->getStrategyCost(currentState);

        const auto parents = _neighbours[operation].first | filtered(filter);
        const auto users = _neighbours[operation].second | filtered(filter);
        fullCost += accumulateCost(to_small_vector(parents), currentState, true);
        fullCost += accumulateCost(to_small_vector(users), currentState, false);

        passedOp.insert(operation);
    }

    return fullCost;
}

VPUNNCostParameters DefaultStateProvider::getCostModelParameters(const OperationStrategy& state) const {
    const auto getTiling = [](const auto& strategy, auto* operation) {
        if (strategy != nullptr) {
            const Shape shape(parseIntArrayAttr<int64_t>(strategy));
            auto tiles = fillDividedTiles(operation, shape, getShape(operation->getResult(0)));

            VPUX_THROW_WHEN(mlir::failed(tiles), "Incorrect tiling {0} for operation {1}", strategy,
                            operation->getLoc());
            return tiles.value();
        }

        return OutputTiling();
    };
    return VPUNNCostParameters(state.second.getMCStrategy(), getTiling(state.second.getTilingStrategy(), state.first),
                               state.second.getTilingMode());
}

StrategyCost DefaultStateProvider::getTransitionOutsideCost(const OperationStrategy& state, mlir::Operation* operation,
                                                            const bool parent) {
    const auto tempState = std::make_pair(operation, Strategy(VPU::MultiClusterStrategy::Clustering, nullptr));

    const auto transitionCachedCost =
            parent ? _storage->getTransitionCost(tempState, state) : _storage->getTransitionCost(state, tempState);

    if (transitionCachedCost.has_value()) {
        return transitionCachedCost.value();
    }

    StrategyCost transitionCost =
            parent ? _costModel->getSpillingReadCost(state.first, getCostModelParameters(state), operation)
                   : _costModel->getSpillingWriteCost(state.first, getCostModelParameters(state));

    parent ? _storage->setTransitionCost(tempState, state, transitionCost)
           : _storage->setTransitionCost(state, tempState, transitionCost);

    return transitionCost;
}

StrategyCost DefaultStateProvider::getTransitionCost(const OperationStrategy& firstState,
                                                     const OperationStrategy& secondState) {
    const auto transitionCachedCost = _storage->getTransitionCost(firstState, secondState);

    if (transitionCachedCost.has_value()) {
        return transitionCachedCost.value();
    }

    StrategyCost transitionCost = 0;

    if (!canStayInCMX(firstState, secondState)) {
        // sum of write in DDR of each tile of parent op + sum of read from DDR of each tile of child op

        transitionCost = _costModel->getSpillingCost(firstState.first, getCostModelParameters(firstState),
                                                     secondState.first, getCostModelParameters(secondState));
    }

    _storage->setTransitionCost(firstState, secondState, transitionCost);

    return transitionCost;
}

bool DefaultStateProvider::spillAroundConcat(mlir::Operation* operation) const {
    if (!mlir::isa<VPU::ConcatOp>(operation)) {
        return false;
    }

    const auto isCMXCompatible = [&](auto* op) {
        return mlir::isa<VPU::NCEOpInterface>(op) || _storage->hasAnyStrategy(op);
    };

    return !llvm::all_of(operation->getUsers(), isCMXCompatible) ||
           !llvm::all_of(operation->getOperands() | transformed([&](auto operand) {
                             return operand.getDefiningOp();
                         }),
                         isCMXCompatible);
}

bool DefaultStateProvider::canStayInCMX(const OperationStrategy& parentState,
                                        const OperationStrategy& childState) const {
    // check if valid mapping for multi-cluster to stay in CMX
    if (!doMCStrategiesMatch(parentState.second.getMCStrategy(), childState.second.getMCStrategy())) {
        return false;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(parentState.first);
    auto siblingsAnalysis = SiblingOpsAnalysis(clusteredOp);
    if (!mlir::isa_and_nonnull<VPU::TilingBuilderOpInterface>(parentState.first) && clusteredOp != nullptr &&
        !clusteredOp.doesLayerFitIntoCMX(parentState.second.getMCStrategy(), siblingsAnalysis, Byte(0))) {
        return false;
    }

    if (spillAroundConcat(parentState.first) || spillAroundConcat(childState.first)) {
        return false;
    }

    // check if tiling strategies allow to stay in CMX
    const auto parentTilingStrategy = parentState.second.getTilingStrategy();

    if (parentTilingStrategy == nullptr) {
        return true;
    }

    const auto parentTiling = parseIntArrayAttr<int64_t>(parentTilingStrategy);

    const auto isOne = [](auto i) {
        return i == 1;
    };

    if (llvm::all_of(parentTiling, isOne)) {
        return true;
    }

    const auto isSpatialTiling = [](auto& strategy) {
        if (strategy.size() <= Dims4D::Act::numSpatialDims) {
            return false;
        }

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            if (strategy[Dims4D::Act::getSpatialDim(index).ind()] > 1) {
                return true;
            }
        }

        return false;
    };

    if (isSpatialTiling(parentTiling)) {
        return false;
    }

    // assuming, it's K tiling, check parent's memory footprint
    const auto tiles =
            fillDividedTiles(parentState.first, ShapeRef(parentTiling), getShape(parentState.first->getResult(0)));
    if (mlir::failed(tiles)) {
        return false;
    }

    if (isCMXConcatentationAvaliable(parentState.first, parentState.second.getTilingMode(), tiles.value(),
                                     parentState.second.getMCStrategy())) {
        return true;
    }

    return false;
}

bool DefaultStateProvider::doMCStrategiesMatch(const MultiClusterStrategy parentStrategy,
                                               const MultiClusterStrategy childStrategy) const {
    if ((parentStrategy == childStrategy && parentStrategy != MultiClusterStrategy::HKSwitch) ||
        strategyMatch[parentStrategy] == childStrategy) {
        return true;
    }

    return false;
}

bool DefaultStateProvider::isCMXConcatentationAvaliable(mlir::Operation* operation, const TilingMode mode,
                                                        const OutputTiling& tiles,
                                                        const MultiClusterStrategy strategy) const {
    VPUX_THROW_WHEN(tiles.empty(), "No available tiles for operation {0}", operation->getLoc());
    MultiClusterStrategySetter mcSetter(operation, strategy);

    Byte extraMemory = Byte(0);
    if (mode == TilingMode::PIPELINING) {
        extraMemory = VPU::getRequiredCMXForWeight(operation, tiles.front());
    }
    return VPU::getRequiredCMX(operation, tiles.front(), Logger::global()) + extraMemory <
           getTotalCMXFragmentationAwareSize(operation);
}

mlir::Operation* DefaultStateProvider::getParentOp(mlir::Value operand) const {
    auto parentOp = operand.getDefiningOp();
    while (parentOp != nullptr) {
        if (isPureViewOp(parentOp) && !hasTiling(parentOp)) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
            continue;
        }
        break;
    }

    return parentOp;
}

void DefaultStateProvider::getConsumersOp(SmallVector<mlir::Operation*>& users, mlir::Operation* op) const {
    for (auto user : op->getUsers()) {
        if (hasTiling(user)) {
            users.push_back(user);
        } else if (isPureViewOp(user)) {
            getConsumersOp(users, user);
        }
    }
}

bool DefaultStateProvider::hasTiling(mlir::Operation* operation) const {
    return mlir::isa_and_nonnull<VPU::ClusteredOpInterface, VPU::TilingInfoOpInterface>(operation);
}
