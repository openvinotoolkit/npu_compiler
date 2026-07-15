//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"

#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/TilingInterface.h>

namespace vpux::VPU {

struct TileDimensionInfo {
    vpux::Dim dimension;
    int64_t numBlocks;
    bool isUnrolled;
    mlir::scf::ForOp forOp;
    int64_t id;  // identifier to sort the loop from outermost to innermost
};

struct UnrollConfig {
    llvm::SmallVector<int64_t> unrollFactors;
    SmallVector<vpux::Dim> accessOrder;
    size_t totalBlocks;
    llvm::SmallVector<mlir::scf::ForOp> forOps;
};

mlir::LogicalResult mergeUnrollOperationsInBlock(mlir::Block* block, const UnrollConfig& config, int64_t numOriginalOps,
                                                 bool cleanupAfterMerge = true);
mlir::LogicalResult mergeUnrolledOperations(mlir::scf::ForOp forOp, SmallVector<TileDimensionInfo>& tileDimInfoVec);

mlir::scf::ForOp fuseSiblingForLoops(mlir::scf::ForOp target, mlir::scf::ForOp source, mlir::RewriterBase& rewriter,
                                     bool residualLoops = false);

unsigned getNestingDepth(mlir::Operation* op);
void collectLoops(mlir::Operation* rootOp, SmallVector<mlir::scf::ForOp>& loops);

}  // namespace vpux::VPU
