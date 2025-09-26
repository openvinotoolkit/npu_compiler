//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux::VPU::VF::v1 {

VFCase::VFCase(VFCase::VFConfigType& config, Dim axis): _config(config), _axis(axis) {
}

VFCase::~VFCase() {
}

VFCase::VFCase(VFCase&& vfCase): _config(vfCase._config) {
    _axis = vfCase._axis;
    _cachedCost = vfCase._cachedCost;
    _vfScheduling = std::move(vfCase._vfScheduling);
    _vfTilingStorage = std::move(vfCase._vfTilingStorage);
    _tilingNumber = vfCase._tilingNumber;

    vfCase._tilingNumber = 1;
    vfCase._vfScheduling = nullptr;
    vfCase._vfTilingStorage = nullptr;
}

VFCase& VFCase::operator=(VFCase&& other) {
    if (this == &other) {
        return *this;
    }

    std::swap(_config, other._config);
    std::swap(_vfScheduling, other._vfScheduling);
    _vfTilingStorage = std::move(other._vfTilingStorage);
    _axis = other._axis;
    _tilingNumber = other._tilingNumber;
    std::swap(_cachedCost, other._cachedCost);

    other._tilingNumber = 1;
    other._vfScheduling = nullptr;
    _vfTilingStorage = nullptr;

    return *this;
}

VFCase::VFCase(const VFCase& vfCase): _config(vfCase._config) {
    _axis = vfCase._axis;
    _cachedCost = vfCase._cachedCost;
    _vfScheduling = vfCase._vfScheduling;
    _tilingNumber = vfCase._tilingNumber;
    _vfTilingStorage = nullptr;
}

VFCase& VFCase::operator=(const VFCase& other) {
    if (this == &other) {
        return *this;
    }

    _config = other._config;
    _axis = other._axis;
    _cachedCost = other._cachedCost;
    _vfScheduling = other._vfScheduling;
    _tilingNumber = other._tilingNumber;
    _vfTilingStorage = nullptr;

    return *this;
}

void VFCase::setTilingNumber(int64_t number) {
    if (_tilingNumber != number) {
        clearCache();
    }
    _tilingNumber = number;
}

void VFCase::setScheduling(std::shared_ptr<IVFScheduling<VFCase::VFConfigType>> vfScheduling) {
    if (vfScheduling == nullptr || (_vfScheduling != nullptr && _vfScheduling->getType() != vfScheduling->getType())) {
        clearCache();
    }
    _vfScheduling = std::move(vfScheduling);
}

void VFCase::setTilingStorage(std::unique_ptr<TilingOperationStorage> vfStorage) {
    if (vfStorage == nullptr) {
        clearCache();
    }
    _vfTilingStorage = std::move(vfStorage);
}

void VFCase::clearCache() {
    _cachedCost.reset();
}

StrategyCost VFCase::getCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    VPUX_THROW_WHEN(!isInitialized(), "Cannot get cost of uninitialized VF case");

    if (!_cachedCost.has_value()) {
        if (_vfTilingStorage == nullptr) {
            _vfTilingStorage = std::make_unique<TilingOperationStorage>();
            auto tilingDims = parseIntArrayAttr<int64_t>(getTiling());
            auto tilingStorage = calculateTilingRegions(_config.getSubgraph(), tilingDims, log, _vfTilingStorage);
            VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles",
                            _config.getSubgraph(), tilingDims);
        }

        _cachedCost = _vfScheduling->getCost(_config, _tilingNumber, _vfTilingStorage, costFunction);
        log.trace("Merged VF {0} cost {1}", _config.getSubgraph().getLoc(), _cachedCost.value());
        addCMXWriteSpills(costFunction, log);
        log.trace("Merged VF {0} cost with spill write {1}", _config.getSubgraph().getLoc(), _cachedCost.value());
    }

    return _cachedCost.value();
}

VFCase::VFConfigType& VFCase::getConfig() {
    return _config;
}

mlir::ArrayAttr VFCase::getTiling() const {
    auto outType = mlir::cast<vpux::NDTypeInterface>(_config.getSubgraph()->getResult(0).getType());
    auto tilingArray = SmallVector<int64_t>(outType.getRank(), 1);
    tilingArray[_axis.ind()] = _tilingNumber;

    return getIntArrayAttr(_config.getSubgraph().getContext(), tilingArray);
}

int64_t VFCase::getTilingNumber() const {
    return _tilingNumber;
}

void VFCase::approveScheduling() {
    VPUX_THROW_WHEN(!isInitialized(), "Cannot approve uninitialized VF case");

    _config.getSubgraph().setScenario(_vfScheduling->getType());
    _config.getSubgraph().setTilingStrategyAttr(getTiling());
}

bool VFCase::isInitialized() {
    return _vfScheduling != nullptr && (_tilingNumber > 1 || _config.isPotentiallyPipelined());
}

void VFCase::addCMXWriteSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger) {
    const auto getStrategy = [](auto* operation) -> VPU::MultiClusterStrategy {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
            strategy = mcOperation.getMultiClusterStrategy().value_or(strategy);
        }
        return strategy;
    };

    StrategyCost cost = 0;

    for (auto* inputOp : _config.getVFOperations() | filtered([](mlir::Operation* op) {
                             return op->getNumOperands() > 1 && op->hasTrait<VPU::EltwiseOp>();
                         })) {
        for (auto operand : inputOp->getOperands()) {
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                auto previousOp = _config.getSubgraph().getOperand(arg.getArgNumber()).getDefiningOp();
                if (mlir::isa_and_nonnull<Const::DeclareOp>(previousOp) || !v1::isCmxOperation(previousOp, false) ||
                    isPrevOperationEarlyScheduled(previousOp, _config.getSubgraph())) {
                    continue;
                }

                if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(previousOp)) {
                    previousOp = vfOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
                }

                auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
                auto operandSize = operandType.getTotalAllocSize();
                auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(previousOp);
                if (clusteredOp != nullptr && clusteredOp->hasAttr(VPU::multiClusterStrategy)) {
                    auto numClusters =
                            VPU::getOptimalNumClusters(clusteredOp, operandType.getShape(),
                                                       mlir::cast<vpux::VPU::MultiClusterStrategyAttr>(
                                                               clusteredOp->getAttr(VPU::multiClusterStrategy))
                                                               .getValue());
                    auto clusterType = getDistributedOutputTypeFromOp(clusteredOp, operandType, numClusters);
                    operandSize = mlir::cast<vpux::NDTypeInterface>(clusterType).getTotalAllocSize();
                }

                if (!_vfScheduling->validate(_config, _vfTilingStorage, operandSize)) {
                    OutputTiling prevOpTiling;
                    if (previousOp->hasAttr(tilingStrategy)) {
                        auto prevOpStrategy = parseIntArrayAttr<int64_t>(
                                mlir::cast<mlir::ArrayAttr>(previousOp->getAttr(tilingStrategy)));
                        auto tiles = fillDividedTiles(previousOp, ShapeRef(prevOpStrategy),
                                                      getShape(previousOp->getResult(0)));
                        VPUX_THROW_WHEN(mlir::failed(tiles) || tiles.value().empty(),
                                        "Cannot get tiles {0} for the operation in VF {1}", prevOpStrategy, previousOp);
                        prevOpTiling = tiles.value();
                    }
                    const auto parentOpParams = VPUNNCostParameters(getStrategy(previousOp), prevOpTiling);
                    cost += costFunction->getSpillingWriteCost(previousOp, parentOpParams);
                }
            }
        }
    }

    _cachedCost = _cachedCost.value() + cost;
}
}  // namespace vpux::VPU::VF::v1
