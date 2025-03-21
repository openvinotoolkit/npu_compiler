//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<IGreedilyPassStrategy> createMakeOpsWithDistributedTensorStrategy(
        mlir::func::FuncOp funcOp, const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
        const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
        bool enableExplicitDistributionInfoAttr);

}  // namespace vpux::VPU
