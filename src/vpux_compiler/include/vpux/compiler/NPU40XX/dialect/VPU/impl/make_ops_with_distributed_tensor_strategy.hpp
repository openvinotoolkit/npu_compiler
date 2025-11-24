//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::VPU::arch40xx {

/*
   Class for getting MakeOpsWithDistributedTensorStrategy patterns for NPU40XX
*/
class MakeOpsWithDistributedTensorStrategy : public IGreedilyPassStrategy {
public:
    MakeOpsWithDistributedTensorStrategy(
            const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup)
            : _typeLookup(typeLookup), _inputTypeLookup(inputTypeLookup) {
    }
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& _typeLookup;
    const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& _inputTypeLookup;
};

}  // namespace vpux::VPU::arch40xx
