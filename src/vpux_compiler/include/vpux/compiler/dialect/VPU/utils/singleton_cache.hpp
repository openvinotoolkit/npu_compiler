//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_factory.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"

#include <mlir/IR/DialectInterface.h>

namespace vpux {
namespace VPU {

/** @brief Singleton container for various architecture-specific factories and utilities. */
class SingletonCache final : public mlir::DialectInterface::Base<SingletonCache> {
    std::unique_ptr<ICostModelFactory> _costModelFactory;
    std::unique_ptr<CostModelShaveUtil> _shaveCostModelUtils;
    PrecomputedStrategyTable _ptcTable;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SingletonCache)

    SingletonCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const ICostModelFactory& getCostModelFactory() const {
        assert(_costModelFactory != nullptr && "Cost model factory is not set");
        return *_costModelFactory;
    }

    void setCostModelFactory(std::unique_ptr<ICostModelFactory> costModelFactory) {
        _costModelFactory = std::move(costModelFactory);
    }

    const CostModelShaveUtil& getShaveCostModelUtils() const {
        assert(_shaveCostModelUtils != nullptr && "Shave cost model utils is not set");
        return *_shaveCostModelUtils;
    }

    void setShaveCostModelUtils(std::unique_ptr<CostModelShaveUtil> shaveCostModelUtils) {
        _shaveCostModelUtils = std::move(shaveCostModelUtils);
    }

    PrecomputedStrategyTable& getPrecomputedStrategyTable() {
        return _ptcTable;
    }
};

/** @brief Sets the cost model factory in the singleton cache for the given MLIR context. */
void setCostModelFactory(mlir::MLIRContext* context, std::unique_ptr<ICostModelFactory> costModelFactory);

/**  @brief Gets the cost model factory from the singleton cache for the given MLIR context. */
const ICostModelFactory& getCostModelFactory(mlir::MLIRContext* context);

/** @brief Sets the shave cost model utilities in the singleton cache for the given MLIR context. */
void setShaveCostModelUtils(mlir::MLIRContext* context, std::unique_ptr<CostModelShaveUtil> shaveCostModelUtils);

/** @brief Gets the shave cost model utilities from the singleton cache for the given MLIR context. */
const CostModelShaveUtil& getShaveCostModelUtils(mlir::MLIRContext* context);

/** @brief Gets the per-context precomputed strategy table from the singleton cache. */
PrecomputedStrategyTable& getPrecomputedStrategyTable(mlir::MLIRContext* context);

}  // namespace VPU
}  // namespace vpux
