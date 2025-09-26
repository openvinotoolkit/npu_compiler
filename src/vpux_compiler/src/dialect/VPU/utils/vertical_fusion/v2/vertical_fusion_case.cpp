//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace {
vpux::VPU::MultiClusterStrategy getStrategy(mlir::Operation* operation) {
    auto strategy = vpux::VPU::MultiClusterStrategy::Clustering;
    if (auto mcOperation = mlir::dyn_cast<vpux::VPU::ClusteredOpInterface>(operation)) {
        strategy = mcOperation.getMultiClusterStrategy().value_or(strategy);
    }
    return strategy;
}
}  // namespace

namespace vpux::VPU::VF::v2 {

VFCase::VFCase(VFCase::VFConfigType& config, const VFSplit& split): _config(config), _split(split) {
}

VFCase::~VFCase() {
}

VFCase::VFCase(VFCase&& vfCase): _config(vfCase._config) {
    _split = std::move(vfCase._split);
    _cachedCost = vfCase._cachedCost;
    _vfScheduling = std::move(vfCase._vfScheduling);
    _vfTilingStorage = std::move(vfCase._vfTilingStorage);

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
    _split = std::move(other._split);
    std::swap(_cachedCost, other._cachedCost);

    other._vfScheduling = nullptr;
    _vfTilingStorage = nullptr;

    return *this;
}

VFCase::VFCase(const VFCase& vfCase): _config(vfCase._config) {
    _split = vfCase._split;
    _cachedCost = vfCase._cachedCost;
    _vfScheduling = vfCase._vfScheduling;
    _vfTilingStorage = nullptr;
}

VFCase& VFCase::operator=(const VFCase& other) {
    if (this == &other) {
        return *this;
    }

    _config = other._config;
    _split = other._split;
    _cachedCost = other._cachedCost;
    _vfScheduling = other._vfScheduling;
    _vfTilingStorage = nullptr;

    return *this;
}

void VFCase::setTilingNumber(Dim dim, int64_t number) {
    if (_split[dim].value_or(1) != number) {
        clearCache();
    }
    _split[dim] = number;
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
            auto tilingStorage = calculateTilingRegions(_config, tilingDims, log, _vfTilingStorage);
            VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles",
                            _config.getSubgraph(), tilingDims);
        }
        VPUX_THROW_WHEN(llvm::any_of(_split,
                                     [](const auto& item) {
                                         return !item.second.has_value();
                                     }),
                        "Cannot get cost for VF {0} without fixed tiling number", _config.getSubgraph().getLoc());
        auto tileLen = getVFTilesLen(_split);
        _cachedCost = _vfScheduling->getCost(_config, tileLen, _vfTilingStorage, costFunction);
        log.trace("Merged VF {0} cost {1}", _config.getSubgraph().getLoc(), _cachedCost.value());
        addCMXWriteSpills(costFunction, log);
        log.trace("Merged VF {0} cost with spill write {1}", _config.getSubgraph().getLoc(), _cachedCost.value());
        addCMXReadSpills(costFunction, log);
        log.trace("Merged VF {0} cost with spill write/read {1}", _config.getSubgraph().getLoc(), _cachedCost.value());
    }

    return _cachedCost.value();
}

VFCase::VFConfigType& VFCase::getConfig() {
    return _config;
}

mlir::ArrayAttr VFCase::getTiling() {
    auto outputs = _config.getOutputs();
    VPUX_THROW_WHEN(outputs.empty(), "No outputs for VF");
    auto outType = mlir::cast<vpux::NDTypeInterface>(outputs.front()->getResult(0).getType());
    auto tilingArray = restoreTilingBySplit(outType.getRank(), _split);
    return getIntArrayAttr(outputs.front()->getContext(), tilingArray);
}

void VFCase::approveScheduling() {
    VPUX_THROW_WHEN(!isInitialized(), "Cannot approve uninitialized VF case");

    _config.getSubgraph().setScenario(_vfScheduling->getType());
    _config.getSubgraph().setTilingStrategyAttr(getTiling());
}

bool VFCase::isInitialized() {
    return _vfScheduling != nullptr &&
           llvm::all_of(_split,
                        [](const auto& item) {
                            return item.second.has_value();
                        }) &&
           getVFTilesLen(_split) > 1;
}

void VFCase::addCMXReadSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    if (_config.secondVFNeedTiling() || !_cachedCost.has_value() ||
        !outputTileAxisIsSameAsMultiClusterStrategy(_config.getSubgraph()) ||
        hasOutputSpilledForDifferentDataSizeUses(_config.getSubgraph())) {
        return;
    }

    StrategyCost cost = 0;
    auto uses = findUses(_config.getSubgraph());

    for (auto* use : uses) {
        mlir::Operation* user = use->getOwner();

        if (!v2::isCmxOperation(user, true)) {
            continue;
        }

        // if between VF and it's CMX user there are other operations which
        // might be scheduled there will be spill anyway, no need to add read cost for it
        auto* prevNode = user->getPrevNode();
        auto hasSpillReadWithoutVF = false;
        while (prevNode != nullptr && prevNode != _config.getSubgraph()) {
            if (!isPureViewOp(prevNode) && !v2::isCmxOperation(user, true)) {
                hasSpillReadWithoutVF = true;
                break;
            }
            prevNode = prevNode->getPrevNode();
        }

        if (hasSpillReadWithoutVF) {
            continue;
        }

        auto* nextOp = user;
        auto nextOperand = user->getOperand(use->getOperandNumber());

        if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(user)) {
            auto vfArgument = vfOp.getBody()->getArgument(use->getOperandNumber());
            if (to_small_vector(vfOp.getBody()->getOps<VPU::VerticalFusionOpInterface>()).size() != 1) {
                continue;
            }

            for (auto& vfUse : vfArgument.getUses()) {
                auto* vfUser = vfUse.getOwner();
                auto opNumber = vfUse.getOperandNumber();
                while (vfUser != nullptr && !mlir::isa<VPU::VerticalFusionOpInterface>(vfUser)) {
                    if (vfUser->getUses().empty()) {
                        break;
                    }
                    auto firstUse = vfUser->getUses().begin();
                    vfUser = firstUse->getOwner();
                    opNumber = firstUse->getOperandNumber();
                }

                if (vfUser != nullptr) {
                    nextOp = vfUser;
                    nextOperand = nextOp->getOperand(opNumber);
                    break;
                }
            }
        }

        OutputTiling nextOpTiling;
        if (user->hasAttr(tilingStrategy)) {
            auto nextOpStrategy =
                    parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(user->getAttr(tilingStrategy)));
            auto tiles = fillDividedTiles(nextOp, ShapeRef(nextOpStrategy), getShape(nextOp->getResult(0)));
            VPUX_THROW_WHEN(mlir::failed(tiles) || tiles.value().empty(),
                            "Cannot get tiles {0} for the operation in VF {1}", nextOpStrategy, user);
            auto tilingInfoNextOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nextOp);
            if (tiles.value().size() > 1 && tilingInfoNextOp != nullptr &&
                tilingInfoNextOp.isSupportedTiling(tiles.value(), vpux::TilingMode::PIPELINING, log)) {
                nextOpTiling.emplace_back(tiles.value().front());
            } else {
                nextOpTiling = std::move(tiles.value());
            }
        }
        const auto nextOpParams = VPUNNCostParameters(getStrategy(nextOp), nextOpTiling);
        cost += costFunction->getSpillingReadCost(nextOp, nextOpParams, nextOperand);
    }

    _cachedCost = _cachedCost.value() + cost;
}

void VFCase::addCMXWriteSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    StrategyCost cost = 0;

    for (auto* inputOp : _config.getOperationsForTiling()) {
        // check separately Eltwise operation which has different inputs
        // in order to find out if there are tensors in CMX which are in memory apart from VF
        bool eltwiseLike = (inputOp->getNumOperands() > 1 && inputOp->hasTrait<VPU::EltwiseOp>() &&
                            inputOp->getOperand(0) != inputOp->getOperand(1));
        bool isInput = llvm::find(_config.getInputs(), inputOp) != _config.getInputs().end();
        if (!(eltwiseLike && !isInput) && _config.firstVFNeedTiling()) {
            continue;
        }
        for (auto operand : inputOp->getOperands()) {
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                auto parentOperand = _config.getSubgraph().getOperand(arg.getArgNumber());
                auto previousOp = findParent(parentOperand);
                if (mlir::isa_and_nonnull<Const::DeclareOp>(previousOp) || !v2::isCmxOperation(previousOp, false) ||
                    isPrevOperationEarlyScheduled(previousOp, _config.getSubgraph()) ||
                    hasBeforeDDRUsers(previousOp, _config.getSubgraph())) {
                    continue;
                }

                bool isChecked = true;
                if (eltwiseLike) {
                    auto operandType = mlir::cast<vpux::NDTypeInterface>(previousOp->getResult(0).getType());
                    auto operandSize = operandType.getTotalAllocSize();
                    if (auto distributedOutType = VPU::getDistributedOutputType(previousOp)) {
                        operandSize = distributedOutType.getTotalAllocSize();
                    }
                    isChecked = !_vfScheduling->validate(_config, _vfTilingStorage, operandSize);
                }

                auto prevOpStrategyAttr = previousOp->getAttr(tilingStrategy);
                if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(previousOp)) {
                    if (to_small_vector(vfOp.getBody()->getOps<VPU::VerticalFusionOpInterface>()).size() != 1) {
                        continue;
                    }
                    previousOp = vfOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
                }

                if (isChecked) {
                    OutputTiling prevOpTiling;
                    if (prevOpStrategyAttr != nullptr) {
                        auto prevOpStrategy =
                                parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(prevOpStrategyAttr));
                        auto tiles = fillDividedTiles(previousOp, ShapeRef(prevOpStrategy),
                                                      getShape(previousOp->getResult(0)));
                        VPUX_THROW_WHEN(mlir::failed(tiles) || tiles.value().empty(),
                                        "Cannot get tiles {0} for the operation in VF {1}", prevOpStrategy, previousOp);
                        prevOpTiling = tiles.value();
                        auto tilingInfoNextOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(previousOp);
                        if (tiles.value().size() > 1 && tilingInfoNextOp != nullptr &&
                            tilingInfoNextOp.isSupportedTiling(tiles.value(), vpux::TilingMode::PIPELINING, log)) {
                            prevOpTiling.emplace_back(tiles.value().back());
                        } else {
                            prevOpTiling = std::move(tiles.value());
                        }
                    }
                    const auto parentOpParams = VPUNNCostParameters(getStrategy(previousOp), prevOpTiling);
                    cost += costFunction->getSpillingWriteCost(previousOp, parentOpParams);
                }
            }
        }
    }

    _cachedCost = _cachedCost.value() + cost;
}
}  // namespace vpux::VPU::VF::v2
