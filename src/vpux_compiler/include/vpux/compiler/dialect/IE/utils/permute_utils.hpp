//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {

std::optional<IE::PermuteCastOp> tryToFindPermuteCastOp(mlir::Location loc, mlir::Value input,
                                                        const DimsOrder& outOrder, ShapeRef outShape,
                                                        mlir::PatternRewriter& rewriter);

IE::LayerWithPermuteInterface getFusableLayerWithPermuteInterface(mlir::Operation* op);

bool isTrivialReorder(IE::ReorderOp origOp);
bool isTrivialTranspose(IE::TransposeOp origOp);
bool isTrivialMemPermute(IE::MemPermuteOp origOp);

}  // namespace vpux::IE
