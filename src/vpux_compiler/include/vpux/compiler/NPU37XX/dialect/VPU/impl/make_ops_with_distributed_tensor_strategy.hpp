//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux::VPU::arch37xx {

/*
   Class for getting MakeOpsWithDistributedTensorStrategy patterns for NPU37XX
*/
class MakeOpsWithDistributedTensorStrategy : public IGreedilyPassStrategy {
public:
    MakeOpsWithDistributedTensorStrategy(
            const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
            bool enableExplicitDistributionInfoAttr)
            : _typeLookup(typeLookup),
              _inputTypeLookup(inputTypeLookup),
              _enableExplicitDistributionInfoAttr(enableExplicitDistributionInfoAttr) {
    }
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& _typeLookup;
    const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& _inputTypeLookup;
    bool _enableExplicitDistributionInfoAttr = false;
};

}  // namespace vpux::VPU::arch37xx
