//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"

namespace vpux::VPU {

namespace CostModelConfig {

std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(mlir::MLIRContext* context);
std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(mlir::Operation* op);

}  // namespace CostModelConfig

/**
 * Analyzes the VPUNN Layer Cost Model with a custom `isInvalidated` function.
 * This analysis object remains preserved once constructed, until `invalidate` is called
 * or the AnalysisManager is cleared.
 */
class LayerCostModelAnalysis {
public:
    explicit LayerCostModelAnalysis(mlir::ModuleOp moduleOp);
    std::shared_ptr<VPUNN::VPULayerCostModel> getVPUNNLayerCostModel();

    // Used by AnalysisManager to check if this analysis should be preserved.
    // If return false, preserve this analysis. Otherwise release it.
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&);

    // Invalidate this analysis
    // The resource will be destroyed after pass
    void invalidate();

    // If the input analysis is empty, create a layer cost model instance and return it.
    // Otherwise, return the cached layer cost model instance.
    static std::shared_ptr<VPUNN::VPULayerCostModel> getOrCreateLayerCostModel(
            std::optional<std::reference_wrapper<LayerCostModelAnalysis>> analysis, mlir::MLIRContext* context,
            Logger log = Logger::global().nest("layer-cost-model-analysis"));

private:
    std::shared_ptr<VPUNN::VPULayerCostModel> _layerCostModel;

    // Flag indicating whether the analysis is preserved. Initialized to true.
    bool _preserved = true;
};

struct VPUNNCostParameters {
    VPUNNCostParameters(VPU::MultiClusterStrategy strategy, const OutputTiling& tiling = {},
                        TilingMode mode = TilingMode::ISOLATED,
                        const SmallVector<SmallVector<TileInfo>>& operandTiling = {}, const bool withDMAs = true)
            : _strategy(strategy), _tiling(tiling), _mode(mode), _operandsTiling(operandTiling), _withDMAs(withDMAs) {
    }

    VPU::MultiClusterStrategy _strategy;
    OutputTiling _tiling;
    TilingMode _mode;
    SmallVector<SmallVector<TileInfo>> _operandsTiling;
    bool _withDMAs = true;
};

class MultiClusterStrategySetter {
public:
    MultiClusterStrategySetter(mlir::Operation* operation, VPU::MultiClusterStrategy strategy);
    ~MultiClusterStrategySetter();

private:
    /*
     *  Set temporary strategy on operation, returns original one
     */
    void setTemporaryStrategy(VPU::MultiClusterStrategy tempStrategy);

    /*
     *  Remove temporary strategy and set original one
     */
    void removeTemporaryStrategy();

    bool _isStrategyChanged = false;
    mlir::Operation* _operation;
    std::optional<VPU::MultiClusterStrategy> _origStrategy;
};

/*
 *  Class adaptor to get cost from VPUNN
 *  for DPU, SW layers
 */

class LayerVPUNNCost final {
public:
    /*
     * Constructor with function op
     */
    LayerVPUNNCost(mlir::func::FuncOp func, Logger log = Logger::global());

    /*
     * Constructor with function op and existing layerCostModel pointer
     * to avoid creating a new layerCostModel object
     */
    LayerVPUNNCost(mlir::func::FuncOp func, std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModel,
                   Logger log = Logger::global());

    /*
     * Move constructor
     */
    LayerVPUNNCost(LayerVPUNNCost&& other) noexcept;

    /*
     *  Get the cost for operation for particular parameters
     */
    StrategyCost getStrategyCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const;

    /*
     *  Get the cost of the spill between operations
     */
    StrategyCost getSpillingCost(mlir::Operation* parentOp, const VPUNNCostParameters& parentParameters,
                                 mlir::Operation* childOp, const VPUNNCostParameters& childParameters) const;

    /*
     *  Get the cost of DMA writes in DDR
     */
    StrategyCost getSpillingWriteCost(
            mlir::Operation* operation, const VPUNNCostParameters& parameters,
            std::function<vpux::NDTypeInterface(const TileInfo&)> getOutputType = nullptr) const;

    /*
     *  Get the cost of DMA writes in DDR for all tiles
     */
    SmallVector<StrategyCost> getSpillingWriteCostsForAllTiles(
            mlir::Operation* operation, const VPUNNCostParameters& parameters,
            std::function<vpux::NDTypeInterface(const TileInfo&)> getOutputType = nullptr) const;

    /*
     *  Get the cost of DMA reads from DDR
     */
    StrategyCost getSpillingReadCost(
            mlir::Operation* operation, const VPUNNCostParameters& parameters, mlir::Operation* parentOp = nullptr,
            std::function<bool(mlir::Value value)> findOperand = nullptr,
            std::function<vpux::NDTypeInterface(const TileInfo&)> getOperandType = nullptr) const;
    StrategyCost getSpillingReadCost(
            mlir::Operation* operation, const VPUNNCostParameters& parameters, mlir::Value operand,
            std::function<vpux::NDTypeInterface(const TileInfo&)> getOperandType = nullptr) const;

    /*
     *  Get the cost of DMA reads from DDR for all tiles
     */
    SmallVector<StrategyCost> getSpillingReadCostsForAllTiles(
            mlir::Operation* operation, const VPUNNCostParameters& parameters, mlir::Operation* parentOp = nullptr,
            std::function<bool(mlir::Value value)> findOperand = nullptr,
            std::function<vpux::NDTypeInterface(const TileInfo&)> getOperandType = nullptr) const;

    StrategyCost getSpillingTypeCost(vpux::NDTypeInterface type,
                                     const std::optional<ShapeRef>& tileAxis = std::nullopt) const;
    void resetNNCacheCounter();
    void printNNCacheStatistics() const;

private:
    /*
     *  Get the cost of NCE operation.
     *   In case tiling is passed, cost is taken with tiling parameters
     */
    StrategyCost getNCELayerCost(VPU::NCEOpInterface nceOp, const VPUNNCostParameters& parameters) const;

    /*
     *  Get cost of SW kernels
     */
    StrategyCost getSWLayerCost(VPU::SWOpInterface swOp, const VPUNNCostParameters& parameters) const;

    /*
     *  Get simple cycle cost for operation which is not supported by VPUNN yet
     *  Approximate cost is size of output tensor in bytes per cluster
     */
    StrategyCost getSimpleLayerCost(vpux::NDTypeInterface outputType, const VPUNNCostParameters& parameters) const;

    /*
     *  Get divisor to get size for output tensor per cluster
     */
    size_t getNumClusterCorrectionSize(VPU::MultiClusterStrategy strategy) const;

    config::ArchKind _arch;
    int64_t _numTiles;
    int64_t _numDPUs;
    int64_t _numShaveActs;
    int64_t _numDMAPorts;
    VPUNN::VPUDevice _vpuDevice;
    std::shared_ptr<VPUNN::VPULayerCostModel> _vpunnCostModel;
    Logger _log;
};

}  // namespace vpux::VPU
