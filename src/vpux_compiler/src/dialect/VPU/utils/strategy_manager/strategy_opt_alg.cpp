//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_opt_alg.hpp"
#include "vpux/utils/algorithm/simulated_annealing.hpp"

using namespace vpux;

namespace vpux::VPU {

constexpr size_t SA_INIT_ITERATIONS = 200;

void SimulatedAnnealingStrategy::optimize() {
    vpux::algorithm::simulatedAnnealing<OperationStrategy>(
            _temperature, _steps,
            [this](int temperature, double& cost, const OperationStrategy* const state) {
                return _stateProvider->getState(temperature, cost, state);
            },
            [this](const OperationStrategy& state) {
                return _stateProvider->getCost(state);
            },
            [this]() {
                return _stateProvider->getFullCost();
            },
            [this](const OperationStrategy& state) {
                _stateProvider->updateState(state);
            },
            [this](const OperationStrategy& state) {
                _stateProvider->updateSolution(state);
            });
}

std::unique_ptr<IStrategyOptAlgorithm> createAlgorithm(const vpux::VPU::TilingOptions&,
                                                       const std::shared_ptr<IStateProvider>& stateProvider,
                                                       const std::shared_ptr<OperationStrategies>& strategies) {
    // number of iteration will be chosen based on compilation options
    // for long compilation iteration number for each step is equal number of operations in the storage
    return std::make_unique<SimulatedAnnealingStrategy>(
            stateProvider, getInitialTemperature(strategies),
            std::max(strategies->getAllOperations().size(), SA_INIT_ITERATIONS));
}

/*
Calculate maximum and minimum cost assigned to a layer to calculate delta.
Select the maximum delta from all layers of the IR to get the initial temperature
*/
size_t getInitialTemperature(const std::shared_ptr<OperationStrategies>& storage) {
    size_t maxDelta = 0;
    const auto allOperations = storage->getAllOperations();
    VPUX_THROW_WHEN(allOperations.empty(), "There are no operations added in this state");

    auto costComparator = [](const StrategyInfo& lhs, const StrategyInfo& rhs) {
        return lhs.strategyCost < rhs.strategyCost;
    };

    for (auto* operation : allOperations) {
        VPUX_THROW_WHEN((!storage->hasAnyStrategy(operation)), "Invalid op. There are no strategies added for this op");

        const auto& allStrategiesInfo = storage->getAllStrategies(operation);

        if (allStrategiesInfo.size() == 1) {
            continue;
        }

        auto minCostIt = std::min_element(allStrategiesInfo.begin(), allStrategiesInfo.end(), costComparator);
        auto maxCostIt = std::max_element(allStrategiesInfo.begin(), allStrategiesInfo.end(), costComparator);
        if (minCostIt != allStrategiesInfo.end() && maxCostIt != allStrategiesInfo.end()) {
            size_t delta = maxCostIt->strategyCost - minCostIt->strategyCost;
            maxDelta = std::max(maxDelta, delta);
        }
    }

    return maxDelta * 2;
}

}  // namespace vpux::VPU
