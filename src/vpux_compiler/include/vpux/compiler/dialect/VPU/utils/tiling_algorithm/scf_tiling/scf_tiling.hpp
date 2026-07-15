//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::VPU {

// Apply tiling using SCF dialect
mlir::LogicalResult applySCFTiling(mlir::Operation* operation, mlir::RewriterBase& builder);

// Apply VF tiling using SCF dialect
SmallVector<mlir::Operation*> applySCFVerticalFusion(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                     const MergeConfiguration& mergeConfig, Logger log);

SmallVector<mlir::OpFoldResult> staticTileSizeComputation(
        mlir::OpBuilder& builder, ArrayRef<mlir::Operation*> operations, mlir::Operation* lastOperation,
        ShapeRef strategy, ShapeRef outputShape, std::unordered_map<Dim, std::pair<int64_t, int64_t>>& remainders);

SmallVector<mlir::OpFoldResult> dynamicTileSizeComputation(mlir::OpBuilder& builder,
                                                           ArrayRef<mlir::Operation*> operations,
                                                           mlir::Operation* lastOperation, ShapeRef strategy,
                                                           bool useBoundedType = true);

}  // namespace vpux::VPU
