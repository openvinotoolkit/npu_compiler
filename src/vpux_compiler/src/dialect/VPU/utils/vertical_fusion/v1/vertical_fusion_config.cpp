//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

// the length of VF pipelining pattern, should match the pattern DPU-SW-DPU for now
constexpr int64_t VF_PIPELINE_LENGTH = 3;
constexpr int64_t VF_POTENTIAL_PIPELINE_LENGTH = 2;

namespace vpux::VPU::VF::v1 {

VFConfig::VFConfig(VPU::VerticalFusionOp vfOp, bool enableVFPipelining /*true*/, bool firstVFNeedsTiling /*true*/,
                   bool secondVFNeedsTiling /*true*/)
        : _subgraph(vfOp),
          _isPipelineEnabled(enableVFPipelining),
          _firstVFNeedsTiling(firstVFNeedsTiling),
          _secondVFNeedsTiling(secondVFNeedsTiling) {
    _isVFPipelineCandidate = _isPipelineEnabled && isVFPipelinePattern();
}

const SmallVector<mlir::Operation*>& VFConfig::getVFOperations() {
    if (_vfOps.empty()) {
        const auto getOpPointer = [](auto& op) -> mlir::Operation* {
            return &op;
        };
        llvm::copy(_subgraph.getBody()->without_terminator() | transformed(getOpPointer), std::back_inserter(_vfOps));
    }

    return _vfOps;
}

SmallVector<mlir::Operation*> VFConfig::getOperationsForTiling() {
    return to_small_vector(getVFOperations() | filtered([](auto* operation) {
                               return mlir::isa_and_nonnull<VPU::VerticalFusionOpInterface>(operation);
                           }));
}

void VFConfig::invalidatePointers() {
    _vfOps.clear();
    _largestOp = nullptr;
    _inputOps.clear();
    _outputOps.clear();
    _tilesCache.clear();
}

VPU::VerticalFusionOp VFConfig::getSubgraph() const {
    return _subgraph;
}

mlir::Operation* VFConfig::getLargestOp() {
    if (_largestOp == nullptr) {
        auto operations = _subgraph.getBody()->without_terminator();

        const auto sumTypes = [&](const Byte& sum, mlir::Value value) {
            return sum + mlir::cast<vpux::NDTypeInterface>(value.getType()).getTotalAllocSize();
        };

        const auto getAllocationSize = [&](auto valueList) -> Byte {
            return std::accumulate(valueList.begin(), valueList.end(), Byte(0), sumTypes);
        };

        auto largestOperation = std::max_element(operations.begin(), operations.end(), [&](auto& op1, auto& op2) {
            return getAllocationSize(op1.getOperands()) + getAllocationSize(op1.getResults()) <
                   getAllocationSize(op2.getOperands()) + getAllocationSize(op2.getResults());
        });

        if (largestOperation == operations.end()) {
            return nullptr;
        }

        _largestOp = &(*largestOperation);
    }
    return _largestOp;
}

const SmallVector<mlir::Operation*>& VFConfig::getInputs() {
    if (_inputOps.empty()) {
        const auto allOperandsInputs = [](auto* current) -> bool {
            return llvm::all_of(current->getOperands(), [](mlir::Value operand) {
                return mlir::dyn_cast<mlir::BlockArgument>(operand) != nullptr;
            });
        };
        auto operations = getVFOperations();
        for (auto* operation : operations) {
            if (!mlir::isa<VPU::VerticalFusionOpInterface>(operation)) {
                continue;
            }

            if (!allOperandsInputs(operation)) {
                bool notInput = false;
                for (auto operand : operation->getOperands()) {
                    if (!mlir::isa<mlir::BlockArgument>(operand)) {
                        auto* parent = operand.getDefiningOp();
                        while (parent != nullptr) {
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
    return _inputOps;
}

const SmallVector<mlir::Operation*>& VFConfig::getOutputs() {
    if (_outputOps.empty()) {
        _outputOps = to_small_vector(_subgraph.getBody()->getTerminator()->getOperands() |
                                     transformed([](auto operand) -> mlir::Operation* {
                                         return operand.getDefiningOp();
                                     }));
    }
    return _outputOps;
}

bool VFConfig::isPipelined() const {
    return _isVFPipelineCandidate;
}

SmallVector<NDTypeInterface> VFConfig::getOperationTypes(mlir::Operation* operation) {
    VPUX_THROW_WHEN(llvm::find(getVFOperations(), operation) == _vfOps.end(), "Cannot find operation {0} in VF {1}",
                    *operation, _subgraph);

    auto origShape = Shape(getShape(operation->getResult(0)));
    if (_tilesCache.find(operation) == _tilesCache.end()) {
        _tilesCache[operation][origShape] = getTileTypes(operation, TileInfo(origShape));
    }

    return _tilesCache[operation][origShape];
}

bool VFConfig::firstVFNeedTiling() const {
    return _firstVFNeedsTiling;
}

bool VFConfig::secondVFNeedTiling() const {
    return _secondVFNeedsTiling;
}

SmallVector<NDTypeInterface> VFConfig::getOperationTypes(mlir::Operation* operation, const TileInfo& outTile,
                                                         const ArrayRef<TileInfo> inputTiles) {
    auto cachedTypes = _tilesCache.find(operation);
    if (cachedTypes == _tilesCache.end() || cachedTypes->second.find(outTile.shape) == cachedTypes->second.end()) {
        std::optional<InputTiling> inputTiling = std::nullopt;
        if (!inputTiles.empty()) {
            inputTiling = InputTiling(inputTiles);
        }
        _tilesCache[operation][outTile.shape] = getTileTypes(operation, outTile, inputTiling);
    }

    return _tilesCache[operation][outTile.shape];
}

bool VFConfig::isVFPipelinePattern() {
    auto& operations = getVFOperations();
    if (operations.size() != VF_PIPELINE_LENGTH) {
        return false;
    }
    return mlir::isa<VPU::NCEOpInterface>(operations[0]) && mlir::isa<VPU::SWOpInterface>(operations[1]) &&
           mlir::isa<VPU::NCEOpInterface>(operations[2]);
}

bool VFConfig::isPotentiallyPipelined() {
    if (!_isPipelineEnabled || isPipelined()) {
        return false;
    }

    // WA trying to predict pipelined case
    if (getVFOperations().size() != VF_POTENTIAL_PIPELINE_LENGTH) {
        return false;
    }

    if (!mlir::isa<VPU::SWOpInterface>(_vfOps[0]) || !mlir::isa<VPU::NCEOpInterface>(_vfOps[1])) {
        return false;
    }

    auto parentVF = _subgraph->getOperand(0).getDefiningOp<VPU::VerticalFusionOp>();

    if (parentVF == nullptr) {
        return false;
    }

    auto parentConfig = VFConfig(parentVF);

    auto parentOps = parentConfig.getVFOperations();
    if (parentOps.size() != 1) {
        return false;
    }

    return mlir::isa<VPU::NCEOpInterface>(parentOps.front());
}
}  // namespace vpux::VPU::VF::v1
