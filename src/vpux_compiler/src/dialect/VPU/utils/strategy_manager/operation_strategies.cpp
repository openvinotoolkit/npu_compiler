//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include <vpux/compiler/utils/attributes.hpp>

#include <llvm/ADT/StringExtras.h>

using namespace vpux;
using namespace VPU;

void OperationStrategies::setStrategyState(const OperationStrategy& opStr, unsigned bitIndex) {
    auto& strategySet = _strategies[opStr.first];
    VPUX_THROW_WHEN(strategySet.empty(), "There are no strategies for operation {0}", opStr.first->getLoc());

    auto foundStrategy = llvm::find_if(strategySet, [&](auto& item) {
        return item.strategy == opStr.second;
    });
    VPUX_THROW_WHEN(foundStrategy == strategySet.end(), "Strategy {0}-{1} was not found for operation {2}",
                    opStr.second.getMCStrategy(), opStr.second.getTilingStrategy(), opStr.first->getLoc());

    llvm::for_each(strategySet, [&](auto& item) {
        item.strategyState.reset(bitIndex);
    });
    foundStrategy->strategyState.set(bitIndex);
}

Strategy OperationStrategies::getStrategyByState(mlir::Operation* operation, unsigned bitIndex) const {
    auto foundOperation = _strategies.find(operation);

    VPUX_THROW_WHEN(foundOperation == _strategies.end(), "Couldn't find operation {0} in the storage",
                    operation->getLoc());

    const auto& strategySet = foundOperation->second;

    VPUX_THROW_WHEN(strategySet.empty(), "Strategy list is empty for operation {0}", operation->getLoc());

    auto foundStrategy = llvm::find_if(strategySet, [&](auto& item) {
        return item.strategyState.test(bitIndex);
    });

    VPUX_THROW_WHEN(foundStrategy == strategySet.end(), "Cannot find strategy for operation {0} with index {1}",
                    operation->getLoc(), bitIndex);

    return foundStrategy->strategy;
}

CombinedTransitionKey OperationStrategies::getTransitionHash(const OperationStrategy& srcOpStr,
                                                             const OperationStrategy& dstOpStr) const {
    return CombinedTransitionKey(srcOpStr.first, srcOpStr.second, dstOpStr.first, dstOpStr.second);
}

void OperationStrategies::addStrategy(const OperationStrategy& opStr, const StrategyCost cost) {
    auto& strategySet = _strategies[opStr.first];
    auto foundStrategy = llvm::find_if(strategySet, [&](auto& item) {
        return item.strategy == opStr.second;
    });

    VPUX_THROW_WHEN(foundStrategy != strategySet.end(), "Strategy {0} - {1} was already added for operation {2}",
                    opStr.second.getMCStrategy(), opStr.second.getTilingStrategy(), opStr.first->getLoc());

    auto newInfo = StrategyInfo(opStr.second, cost, llvm::BitVector(BITS_NUMBER));
    strategySet.push_back(newInfo);

    _operationList.insert(opStr.first);
}

void OperationStrategies::setStrategy(const OperationStrategy& opStr, const StrategyCost cost) {
    auto& strategySet = _strategies[opStr.first];

    auto foundStrategy = llvm::find_if(strategySet, [&](auto& item) {
        return item.strategy == opStr.second;
    });

    VPUX_THROW_WHEN(foundStrategy == strategySet.end(), "Strategy {0} - {1} was not found for operation {2}",
                    opStr.second.getMCStrategy(), opStr.second.getTilingStrategy(), opStr.first->getLoc());

    foundStrategy->strategyCost = cost;
    foundStrategy->strategyState.reset();
}

bool OperationStrategies::hasStrategy(const OperationStrategy& opStr) const {
    auto foundOperation = _strategies.find(opStr.first);

    if (foundOperation == _strategies.end()) {
        return false;
    }

    const auto& strategySet = foundOperation->second;

    if (strategySet.empty()) {
        return false;
    }

    auto foundStrategy = llvm::find_if(strategySet, [&](auto& item) {
        return item.strategy == opStr.second;
    });

    return foundStrategy != strategySet.end();
}

bool OperationStrategies::hasAnyStrategy(mlir::Operation* op) const {
    return _strategies.count(op) != 0;
}

void OperationStrategies::setCurrentStrategy(const OperationStrategy& opStr) {
    setStrategyState(opStr, CURRENT_STATE_INDEX);
}

void OperationStrategies::setBestStrategy(const OperationStrategy& opStr) {
    setStrategyState(opStr, BEST_STATE_INDEX);
}

void OperationStrategies::setTransitionCost(const OperationStrategy& srcOpStr, const OperationStrategy& dstOpStr,
                                            const StrategyCost cost) {
    _transitionCost[getTransitionHash(srcOpStr, dstOpStr)] = cost;
}

StrategyCost OperationStrategies::getStrategyCost(const OperationStrategy& opStr) const {
    const auto& strategySet = _strategies.at(opStr.first);

    auto foundStrategy = llvm::find_if(strategySet, [&](auto& item) {
        return item.strategy == opStr.second;
    });

    VPUX_THROW_WHEN(foundStrategy == strategySet.end(), "Cannot find strategy {0} - {1} for operation {2}",
                    opStr.second.getMCStrategy(), opStr.second.getTilingStrategy(), opStr.first->getLoc());

    return foundStrategy->strategyCost;
}

std::optional<StrategyCost> OperationStrategies::getTransitionCost(const OperationStrategy& srcOpStr,
                                                                   const OperationStrategy& dstOpStr) const {
    const auto hash = getTransitionHash(srcOpStr, dstOpStr);

    if (_transitionCost.count(hash) == 0) {
        return std::nullopt;
    }

    return _transitionCost.at(hash);
}

Strategy OperationStrategies::getCurrentStrategy(mlir::Operation* operation) const {
    return getStrategyByState(operation, CURRENT_STATE_INDEX);
}

Strategy OperationStrategies::getBestStrategy(mlir::Operation* operation) const {
    return getStrategyByState(operation, BEST_STATE_INDEX);
}

llvm::SetVector<mlir::Operation*> OperationStrategies::getAllOperations() const {
    return _operationList;
}

SmallVector<StrategyInfo> OperationStrategies::getAllStrategies(mlir::Operation* operation) const {
    if (!hasAnyStrategy(operation)) {
        return {};
    }

    return _strategies.at(operation);
}
