//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/state_provider_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_opt_alg_interface.hpp"

using namespace vpux::VPU;

/*
Strategy Optimization Algorithm :
Gets sets of operation and strategies ("states") from State provider. Each state is stored in "storage"
(OperationStrategies). For all states, initiates SA algorithm for optimization and get best strategy operation pair.
*/

class SimulatedAnnealingStrategy : public IStrategyOptAlgorithm {
private:
    const std::shared_ptr<IStateProvider> _stateProvider;
    const size_t _temperature;
    const size_t _steps;

public:
    /*
    Initialize the strategy optimization algorithm with State provider and temperature
    */
    SimulatedAnnealingStrategy(const std::shared_ptr<IStateProvider>& provider, const size_t temp, const size_t steps)
            : _stateProvider(provider), _temperature(temp), _steps(steps) {
    }

    /*
    Sets up and calls Simulated Annealing algorithm to find the best operation strategy based on cost
    */
    void optimize() override;
};

/*
Creates an instance of the strategy optimization algorithm
*/
std::unique_ptr<IStrategyOptAlgorithm> createAlgorithm(const vpux::VPU::TilingOptions& options,
                                                       const std::shared_ptr<IStateProvider>& stateProvider,
                                                       const std::shared_ptr<OperationStrategies>& strategies);
/*
Calculate initial temperature for Simulated Annealing
*/
size_t getInitialTemperature(const std::shared_ptr<OperationStrategies>& storage);
