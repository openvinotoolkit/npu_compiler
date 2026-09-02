//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/hash_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_hash.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;
using namespace VPU;

namespace vpux::VPU::VF::v2 {

namespace {
llvm::hash_code getOutputDistributionHash(mlir::Operation* operation) {
    auto outputDistribution =
            VPU::getDistributedOutputType(operation, operation->getResult(0), getMultiClusterStrategyFromOp(operation));
    if (outputDistribution == nullptr || outputDistribution.getDistributedTypes().empty()) {
        return llvm::hash_value(0);
    }

    auto& cache = VPU::getGlobalOpTilingCache();
    auto distributedType = outputDistribution.getDistributedTypes().front();
    if (auto distributedTensorType = mlir::dyn_cast<VPU::DistributedTensorType>(distributedType)) {
        const auto distribution = VPU::DistributionInfo::getClassFromAttr(distributedTensorType.getDistribution());
        return cache.calculateShapeAndDistributionHash(distributedTensorType.getShape(), distribution);
    }
    return llvm::hash_value(0);
}
}  // namespace

VFConfig::VFConfig(VPU::VerticalFusionOp vfOp, VFCacheAnalysis& cache, bool enableVFPipelining /*true*/,
                   bool firstVFNeedsTiling /*true*/, bool secondVFNeedsTiling /*true*/)
        : _subgraph(vfOp),
          _cache(cache),
          _isPipelineEnabled(enableVFPipelining),
          _firstVFNeedsTiling(firstVFNeedsTiling),
          _secondVFNeedsTiling(secondVFNeedsTiling) {
    init();
    _isVFPipelineCandidate = _isPipelineEnabled && isVFPipelinePattern();
}

VFConfig::VFConfig(const llvm::SetVector<mlir::Operation*>& operations, VFCacheAnalysis& cache)
        : _subgraph(nullptr), _cache(cache), _isPipelineEnabled(true) {
    _vfOps = operations;
    init();
    _isVFPipelineCandidate = isVFPipelinePattern();
}

VFConfig::VFConfig(const VFConfig& other)
        : _subgraph(other._subgraph),
          _largestOp(other._largestOp),
          _inputOps(other._inputOps),
          _outputOps(other._outputOps),
          _tilingOps(other._tilingOps),
          _vfOps(other._vfOps),
          _cache(other._cache),
          _isVFPipelineCandidate(other._isVFPipelineCandidate),
          _isPipelineEnabled(other._isPipelineEnabled),
          _firstVFNeedsTiling(other._firstVFNeedsTiling),
          _secondVFNeedsTiling(other._secondVFNeedsTiling) {
}

VFConfig::VFConfig(VFConfig&& other)
        : _subgraph(other._subgraph),
          _largestOp(other._largestOp),
          _inputOps(std::move(other._inputOps)),
          _outputOps(std::move(other._outputOps)),
          _tilingOps(std::move(other._tilingOps)),
          _vfOps(std::move(other._vfOps)),
          _cache(other._cache),
          _isVFPipelineCandidate(other._isVFPipelineCandidate),
          _isPipelineEnabled(other._isPipelineEnabled),
          _firstVFNeedsTiling(other._firstVFNeedsTiling),
          _secondVFNeedsTiling(other._secondVFNeedsTiling) {
}

VFConfig& VFConfig::operator=(const VFConfig& other) {
    if (this != &other) {
        _subgraph = other._subgraph;
        _largestOp = other._largestOp;
        _inputOps = other._inputOps;
        _outputOps = other._outputOps;
        _tilingOps = other._tilingOps;
        _vfOps = other._vfOps;
        _cache = other._cache;
        _isVFPipelineCandidate = other._isVFPipelineCandidate;
        _isPipelineEnabled = other._isPipelineEnabled;
        _firstVFNeedsTiling = other._firstVFNeedsTiling;
        _secondVFNeedsTiling = other._secondVFNeedsTiling;
    }
    return *this;
}

VFConfig& VFConfig::operator=(VFConfig&& other) {
    if (this != &other) {
        _subgraph = other._subgraph;
        _largestOp = other._largestOp;
        _inputOps = std::move(other._inputOps);
        _outputOps = std::move(other._outputOps);
        _tilingOps = std::move(other._tilingOps);
        _vfOps = std::move(other._vfOps);
        _cache = other._cache;
        _isVFPipelineCandidate = other._isVFPipelineCandidate;
        _isPipelineEnabled = other._isPipelineEnabled;
        _firstVFNeedsTiling = other._firstVFNeedsTiling;
        _secondVFNeedsTiling = other._secondVFNeedsTiling;
    }
    return *this;
}

bool VFConfig::isVFPipelinePattern() const {
    // if we have operations with both executors
    const auto filterNCE = [](mlir::Operation* op) {
        return mlir::isa<VPU::NCEOpInterface>(op);
    };
    const auto filterSWKernels = [](mlir::Operation* op) {
        return mlir::isa<VPU::SWOpInterface>(op);
    };

    const auto filterDMAOps = [](mlir::Operation* op) {
        auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op);
        if (swOp == nullptr) {
            return false;
        }
        return swOp.supportLoweringAsDMA();
    };

    auto checkedOperations = getOperationsForTiling();
    VPUX_THROW_WHEN(llvm::any_of(checkedOperations,
                                 [](mlir::Operation* op) {
                                     return !mlir::isa<VPU::NCEOpInterface, VPU::SWOpInterface>(op);
                                 }),
                    "There are operations that are neither NCE nor SW in VF subgraph {0}", _subgraph);
    checkedOperations.erase(llvm::remove_if(checkedOperations, filterDMAOps), checkedOperations.end());
    return !(llvm::all_of(checkedOperations, filterNCE) || llvm::all_of(checkedOperations, filterSWKernels));
}

void VFConfig::validateConfig() const {
    VPUX_THROW_WHEN(_vfOps.empty() && _subgraph == nullptr,
                    "Vertical fusion config should be enabled by wrapped operation or list of operations");
}

void VFConfig::init() {
    validateConfig();

    if (_vfOps.empty()) {
        const auto getOpPointer = [](auto& op) -> mlir::Operation* {
            return &op;
        };
        auto operations = _subgraph.getBody()->without_terminator() | transformed(getOpPointer);
        _vfOps.insert(operations.begin(), operations.end());
    }

    if (_tilingOps.empty()) {
        _tilingOps = to_small_vector(_vfOps | filtered([](auto* operation) {
                                         return mlir::isa_and_nonnull<VPU::VerticalFusionOpInterface>(operation);
                                     }));
    }

    if (_inputOps.empty()) {
        auto operations = getVFOperations();
        const auto allOperandsInputs = [&](auto* current) -> bool {
            return llvm::all_of(current->getOperands(), [&](mlir::Value operand) {
                return mlir::dyn_cast<mlir::BlockArgument>(operand) != nullptr ||
                       !_vfOps.contains(operand.getDefiningOp());
            });
        };
        for (auto* operation : operations) {
            if (!mlir::isa<VPU::VerticalFusionOpInterface>(operation)) {
                continue;
            }

            if (!allOperandsInputs(operation)) {
                bool notInput = false;
                for (auto operand : operation->getOperands()) {
                    if (!mlir::isa<mlir::BlockArgument>(operand) &&
                        !mlir::isa<Const::DeclareOp>(operand.getDefiningOp())) {
                        auto* parent = operand.getDefiningOp();
                        if (!_vfOps.contains(parent)) {
                            continue;
                        }
                        while (parent != nullptr) {
                            if (!_vfOps.contains(parent)) {
                                break;
                            }

                            if (mlir::isa<VPU::VerticalFusionOpInterface>(parent)) {
                                notInput = true;
                                break;
                            }

                            parent = parent->getOperand(0).getDefiningOp();
                        }
                    }
                }
                if (notInput) {
                    continue;
                }
            }
            _inputOps.emplace_back(operation);
        }
    }
    if (_outputOps.empty()) {
        if (_subgraph != nullptr) {
            _outputOps = to_small_vector(_subgraph.getBody()->getTerminator()->getOperands() |
                                         transformed([](auto operand) -> mlir::Operation* {
                                             return operand.getDefiningOp();
                                         }));
        } else {
            auto operations = getVFOperations();
            const auto hasNoUserInVF = [this](auto* operation) {
                return llvm::none_of(operation->getUsers(), [&](auto* user) {
                    return _vfOps.contains(user);
                });
            };
            _outputOps = to_small_vector(operations | filtered(hasNoUserInVF));
        }
    }

    if (_largestOp == nullptr) {
        auto operations = getVFOperations();

        const auto sumTypes = [&](const Byte& sum, mlir::Value value) {
            return sum + mlir::cast<vpux::NDTypeInterface>(value.getType()).getTotalAllocSize();
        };

        const auto getAllocationSize = [&](auto valueList) -> Byte {
            return std::accumulate(valueList.begin(), valueList.end(), Byte(0), sumTypes);
        };
        const auto getTotalAllocationSize = [&](auto& operation) {
            if (operation->hasAttr(isInPlace)) {
                return getAllocationSize(operation->getOperands());
            }
            return getAllocationSize(operation->getOperands()) + getAllocationSize(operation->getResults());
        };

        auto largestOperation = std::max_element(operations.begin(), operations.end(), [&](auto& op1, auto& op2) {
            return getTotalAllocationSize(op1) < getTotalAllocationSize(op2);
        });

        if (largestOperation != operations.end()) {
            _largestOp = *largestOperation;
        }
    }
}

const llvm::SetVector<mlir::Operation*>& VFConfig::getVFOperations() const {
    validateConfig();
    return _vfOps;
}

const SmallVector<mlir::Operation*>& VFConfig::getOperationsForTiling() const {
    validateConfig();
    return _tilingOps;
}

VPU::VerticalFusionOp VFConfig::getSubgraph() const {
    return _subgraph;
}

mlir::Operation* VFConfig::getLargestOp() const {
    return _largestOp;
}

const SmallVector<mlir::Operation*>& VFConfig::getInputs() const {
    return _inputOps;
}

const SmallVector<mlir::Operation*>& VFConfig::getOutputs() const {
    return _outputOps;
}

bool VFConfig::isPipelined() const {
    return _isVFPipelineCandidate;
}

const SmallVector<NDTypeInterface>& VFConfig::getOperationTypes(mlir::Operation* operation) {
    VPUX_THROW_WHEN(!_vfOps.contains(operation), "Cannot find operation {0} in VF", *operation);

    auto origShape = Shape(getShape(operation->getResult(0)));
    const auto hash = computeOpShapeHash(operation, origShape);
    auto cachedTypes = _cache.get().getCachedOperationTypes(hash);
    if (cachedTypes != nullptr) {
        return *cachedTypes;
    }

    auto strategy = getMultiClusterStrategyFromOp(operation);
    auto tiledTypes = getTileTypes(operation, TileInfo(origShape), strategy);
    return _cache.get().cacheOperationTypes(hash, std::move(tiledTypes));
}

bool VFConfig::firstVFNeedTiling() const {
    return _firstVFNeedsTiling;
}

bool VFConfig::secondVFNeedTiling() const {
    return _secondVFNeedsTiling;
}

const SmallVector<NDTypeInterface>& VFConfig::getOperationTypes(mlir::Operation* operation, const TileInfo& outTile,
                                                                const ArrayRef<TileInfo> inputTiles) {
    const auto hash = computeRequiredCMXHash(operation, outTile, inputTiles);
    auto cachedTypes = _cache.get().getCachedOperationTypes(hash);
    if (cachedTypes != nullptr) {
        return *cachedTypes;
    }

    std::optional<InputTiling> inputTiling;
    if (!inputTiles.empty()) {
        inputTiling = InputTiling(inputTiles);
    }
    auto strategy = getMultiClusterStrategyFromOp(operation);
    auto tiledTypes = getTileTypes(operation, outTile, strategy, inputTiling);
    return _cache.get().cacheOperationTypes(hash, std::move(tiledTypes));
}

std::optional<StrategyCost> VFConfig::getCachedStrategyCost(llvm::hash_code hash) const {
    return _cache.get().getCachedStrategyCost(hash);
}

void VFConfig::cacheStrategyCost(llvm::hash_code hash, StrategyCost cost) {
    _cache.get().cacheStrategyCost(hash, cost);
}

Byte VFConfig::getOperationRequiredCMX(mlir::Operation* operation, const TileInfo& outTile,
                                       const ArrayRef<TileInfo> inputTiles) {
    const auto hash = computeRequiredCMXHash(operation, outTile, inputTiles);
    auto cachedSize = _cache.get().getCachedRequiredCMX(hash);
    if (cachedSize.has_value()) {
        return cachedSize.value();
    }

    const auto& tiledTypes = getOperationTypes(operation, outTile, inputTiles);
    const auto requiredCMX = VPU::getRequiredCMX(operation, tiledTypes);
    return _cache.get().cacheRequiredCMX(hash, requiredCMX);
}

llvm::hash_code VFConfig::computeOpShapeHash(mlir::Operation* operation, ShapeRef outShape) const {
    auto hash = llvm::hash_combine(VPU::hashOperationForTiling(operation), getOutputDistributionHash(operation),
                                   getMultiClusterStrategyFromOp(operation));
    return llvm::hash_combine(hash, llvm::hash_combine_range(outShape.begin(), outShape.end()));
}

llvm::hash_code VFConfig::computeRequiredCMXHash(mlir::Operation* operation, const TileInfo& outTile,
                                                 const ArrayRef<TileInfo> inputTiles) const {
    auto hash = llvm::hash_combine(VPU::hashOperationForTiling(operation), getOutputDistributionHash(operation),
                                   getMultiClusterStrategyFromOp(operation));
    hash = llvm::hash_combine(hash, getTileHash(outTile));
    for (const auto& inputTile : inputTiles) {
        hash = llvm::hash_combine(hash, getTileHash(inputTile));
    }
    return hash;
}
}  // namespace vpux::VPU::VF::v2
