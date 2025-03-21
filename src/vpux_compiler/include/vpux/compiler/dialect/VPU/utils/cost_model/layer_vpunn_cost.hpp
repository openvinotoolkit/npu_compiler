//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"

namespace vpux::VPU {

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

    mlir::Operation* _operation;
    std::optional<VPU::MultiClusterStrategy> _origStrategy;
};

/*
 *  Class adaptor to get cost from VPUNN
 *  for DPU, SW layers
 */

class LayerVPUNNCost final {
public:
    LayerVPUNNCost(mlir::func::FuncOp func, Logger log = Logger::global()): _log(log) {
        auto module = func->getParentOfType<mlir::ModuleOp>();
        _arch = VPU::getArch(module);
        _vpunnCostModel = VPU::createLayerCostModel(_arch);

        auto tileOp = IE::getTileExecutor(module);
        auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _vpuDevice = getVPUDeviceType(_arch);
        _numShaveActs = 0;
        _numDMAPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
        if (auto shaveActExec = tileOp.getSubExecutor(ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    };

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

    StrategyCost correctStrideDMACost(vpux::NDTypeInterface type, StrategyCost cost) const;

    VPU::ArchKind _arch;
    int64_t _numTiles;
    int64_t _numDPUs;
    int64_t _numShaveActs;
    int64_t _numDMAPorts;
    VPUNN::VPUDevice _vpuDevice;
    std::shared_ptr<VPUNN::VPULayerCostModel> _vpunnCostModel;
    Logger _log;
};

}  // namespace vpux::VPU
