//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_cost_strategy.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <deque>

namespace vpux::VPU::VF::v2 {
VFScheduling::VFScheduling(Logger log, bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy)
        : _log(log), _prefetching(prefetching), _costStrategy(std::move(costStrategy)) {
    VPUX_THROW_WHEN(_costStrategy == nullptr, "VF cost strategy is not initialized");
}

VFScheduling::~VFScheduling() = default;

bool VFScheduling::validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                            const Byte reservedMemory) const {
    return validateScenario(config, tilingInfo, reservedMemory) &&
           _costStrategy->validate(*this, config, tilingInfo, reservedMemory);
}

const std::deque<std::shared_ptr<IVFScheduling<VFConfig>>>& VFScheduling::nextChecks() const {
    return _dependents;
}

void VFScheduling::addNext(std::shared_ptr<IVFScheduling<VFConfig>> check) {
    _dependents.emplace_back(check);
}

Byte VFScheduling::getInputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    const auto index = 0;
    auto inputSize = Byte(0);

    auto inputOperands = llvm::SmallPtrSet<mlir::Value, 4>();
    for (auto op : config.getInputs()) {
        auto tileInfo = tilingInfo->getRef(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        const auto& tileInfoValue = tileInfo.value().get();
        const auto& tileTypes = config.getOperationTypes(op, tileInfoValue.second, tileInfoValue.first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There are not enough types for tile of operation {0}", *op);

        // avoid repeatedly counting if inputOps share the same operand
        const auto inputTypesCount = tileTypes.size() - op->getNumResults();
        VPUX_THROW_UNLESS(inputTypesCount <= op->getNumOperands(), "Mismatch between tile types and operands");
        for (size_t i = 0; i < inputTypesCount; ++i) {
            auto operand = op->getOperand(i);
            if (inputOperands.contains(operand)) {
                continue;
            }
            inputOperands.insert(operand);
            inputSize += tileTypes[i].getTotalAllocSize();
        }
    }

    return inputSize;
}

Byte VFScheduling::getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const {
    auto outputSize = Byte(0);
    const auto index = 0;

    for (auto op : config.getOutputs()) {
        auto tileInfo = tilingInfo->getRef(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        const auto& tileInfoValue = tileInfo.value().get();
        const auto& tileTypes = config.getOperationTypes(op, tileInfoValue.second, tileInfoValue.first.tiles);
        VPUX_THROW_WHEN(tileTypes.empty(), "There is no output type for tile of operation {0}", *op);

        auto type = tileTypes.back();
        outputSize += type.getTotalAllocSize();
    }

    return outputSize;
}

Byte VFScheduling::getResidentOutputSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                         mlir::Operation* excludedOperation) const {
    const auto index = 0;
    const auto& operations = config.getOperationsForTiling();
    const auto noInternalUser = operations.size();

    llvm::DenseMap<mlir::Operation*, size_t> operationIndices;
    for (const auto indexedOperation : llvm::enumerate(operations)) {
        operationIndices[indexedOperation.value()] = indexedOperation.index();
    }

    SmallVector<size_t> lastInternalUser(operations.size(), noInternalUser);
    for (const auto indexedOperation : llvm::enumerate(operations)) {
        for (auto operand : indexedOperation.value()->getOperands()) {
            auto parent = findParent(operand);
            if (parent == nullptr || !llvm::is_contained(operations, parent)) {
                continue;
            }
            const auto parentIndex = operationIndices.find(parent);
            VPUX_THROW_UNLESS(parentIndex != operationIndices.end(),
                              "Parent operation {0} is not in the VF operations list", parent->getLoc());
            auto& lastUser = lastInternalUser[parentIndex->second];
            lastUser = lastUser == noInternalUser ? indexedOperation.index()
                                                  : std::max(lastUser, indexedOperation.index());
        }
    }

    SmallVector<Byte> releasedOutputSize(operations.size() + 1, Byte(0));
    Byte activeOutputSize(0);
    Byte residentOutputSize(0);
    for (const auto indexedOperation : llvm::enumerate(operations)) {
        const auto operationIndex = indexedOperation.index();
        activeOutputSize -= releasedOutputSize[operationIndex];
        residentOutputSize = std::max(residentOutputSize, activeOutputSize);

        if (lastInternalUser[operationIndex] == noInternalUser || indexedOperation.value() == excludedOperation) {
            continue;
        }

        auto opTiling = tilingInfo->getRef(indexedOperation.value(), index);
        VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index,
                        *indexedOperation.value());
        const auto& opTilingValue = opTiling.value().get();
        const auto& operationTypes =
                config.getOperationTypes(indexedOperation.value(), opTilingValue.second, opTilingValue.first.tiles);
        VPUX_THROW_WHEN(operationTypes.empty(), "There is no output type for tile of operation {0}",
                        *indexedOperation.value());

        const auto outputSize = operationTypes.back().getTotalAllocSize();
        activeOutputSize += outputSize;
        releasedOutputSize[lastInternalUser[operationIndex] + 1] += outputSize;
        residentOutputSize = std::max(residentOutputSize, activeOutputSize);
    }

    return residentOutputSize;
}

bool VFScheduling::isSharedWeightsSupported(VFConfig& config) const {
    // OptimizeParallelCopies may keep parallel copies when inplace eltwise are involved. Scenario becomes more complex
    // when there are multi eltwise ops inside. Skip this for now.
    auto hasMultiInPlaceEltwise = llvm::count_if(config.getVFOperations(), [](mlir::Operation* op) {
                                      return op->hasTrait<VPU::EltwiseOp>() && op->hasAttr(isInPlace);
                                  }) > 1;
    return !hasMultiInPlaceEltwise;
}

Byte VFScheduling::calculateSharedSize(VFConfig& config, mlir::Operation* operation,
                                       const vpux::VPU::VFOperationTiling& inputOutputTiling) const {
    const auto& inTile = inputOutputTiling.first;
    const auto& outTile = inputOutputTiling.second;
    Byte size(0);
    const auto& tiledTypes = config.getOperationTypes(operation, outTile, inTile.tiles);
    for (auto operandIdx : irange(tiledTypes.size() - operation->getNumResults())) {
        auto operand = operation->getOperand(operandIdx);
        if (operandIdx >= inTile.tiles.size()) {
            break;
        }
        if (mlir::isa<mlir::BlockArgument>(operand)) {
            auto canBeShared = isOperandSharedWeightsForTiling(operation, operand, inTile.tiles[operandIdx]);
            if (canBeShared) {
                size += tiledTypes[operandIdx].getTotalAllocSize();
            }
        }
    }
    return size;
};

Byte VFScheduling::getSharedSizeByAllTiles(ArrayRef<mlir::Operation*> operations, VFConfig& config,
                                           const TilingOperationStorage::UPtr& tilingInfo) const {
    Byte reservedSize(0);
    if (!isSharedWeightsSupported(config)) {
        return reservedSize;
    }

    const auto index = 0;
    for (auto* operation : operations) {
        auto opTiling = tilingInfo->getRef(operation, index);
        if (!opTiling.has_value()) {
            continue;
        }
        reservedSize += calculateSharedSize(config, operation, opTiling.value().get());
    }
    return reservedSize;
}

mlir::Operation* findParent(mlir::Value operand) {
    auto parent = operand.getDefiningOp();

    while (parent != nullptr && mlir::isa<VPU::TilingViewLikeOpInterface>(parent)) {
        parent = parent->getOperand(0).getDefiningOp();
    }

    return parent;
}

StrategyCost VFScheduling::getCost(VFConfig& config, int64_t tilesNumber,
                                   const TilingOperationStorage::UPtr& tilingInfo,
                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    VPUX_THROW_WHEN(_costStrategy == nullptr, "VF cost strategy is not initialized");
    return _costStrategy->calculate(*this, config, tilesNumber, tilingInfo, costFunction);
}
}  // namespace vpux::VPU::VF::v2
