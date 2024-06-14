//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_config.hpp"

using namespace vpux;
using namespace VPU;

// the length of VF pipelining pattern
// should match the pattern DPU-SW-DPU for now
constexpr int64_t VF_PIPELINE_LENGTH = 3;
constexpr int64_t VF_POTENTIAL_PIPELINE_LENGTH = 2;

VFConfig::VFConfig(VPU::VerticalFusionOp vfOp, bool enableVFPipelining /*true*/)
        : _subgraph(vfOp), _isPipelineEnabled(enableVFPipelining) {
    _isVFPipelineCandidate = _isPipelineEnabled && isVFPipelinePattern();
}

bool VFConfig::isVFPipelinePattern() {
    // Only support VF Pipeline when the VF subgraph contains DPU->SW->DPU tasks
    // More generic cases will be supported in the future
    // Track [E#95184]
    auto& operations = getVFOperations();
    if (operations.size() != VF_PIPELINE_LENGTH) {
        return false;
    }
    return mlir::isa<VPU::NCEOpInterface>(operations[0]) && mlir::isa<VPU::SWOpInterface>(operations[1]) &&
           mlir::isa<VPU::NCEOpInterface>(operations[2]);
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

VPU::VerticalFusionOp VFConfig::getSubgraph() const {
    return _subgraph;
}

mlir::Operation* VFConfig::getLargestOp() {
    if (_largestOp == nullptr) {
        auto operations = _subgraph.getBody()->without_terminator();

        const auto sumTypes = [&](const Byte& sum, mlir::Value value) {
            return sum + value.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
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
        _inputOps = to_small_vector(_subgraph.getBody()->without_terminator() | filtered([](auto& current) -> bool {
                                        return llvm::all_of(current.getOperands(), [](mlir::Value operand) {
                                            return operand.dyn_cast<mlir::BlockArgument>() != nullptr;
                                        });
                                    }) |
                                    transformed([](auto& current) -> mlir::Operation* {
                                        return &current;
                                    }));
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

void VFConfig::disableVFPipeline() {
    _isVFPipelineCandidate = false;
}

void VFConfig::restoreVFPipeline() {
    _isVFPipelineCandidate = _isPipelineEnabled && isVFPipelinePattern();
}

bool VFConfig::isPotentiallyPipelined() const {
    if (!_isPipelineEnabled || isPipelined()) {
        return false;
    }

    // WA trying to predict pipelined case
    if (_vfOps.size() != VF_POTENTIAL_PIPELINE_LENGTH) {
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
