//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Rewrite/PatternApplicator.h>

namespace vpux {

// Traverse patterns first and collect which ops/interfaces need to be collected.
// Then iterate over IR using func.walk and collect all ops related for patterns
std::vector<mlir::Operation*> collectOpsForPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet& patterns);

using OrderedPatternSet = SmallVector<mlir::RewritePatternSet>;

// Apply patterns to provided inputs using mlir pattern applicator
void applyPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet&& patterns, ArrayRef<mlir::Operation*> ops);

// Iterate over IR to collect ops once and then apply patterns to already collected ops
// This is very fast, downside is it only works on frozen view of the IR before any pattern is
// applied. Unlike ApplyPatternsAndFoldGreedily this does not do any folding or dead code (ops) elimination
void collectOpsAndApplyPatterns(mlir::func::FuncOp func, mlir::RewritePatternSet&& patterns);

// Go over whole IR using func.walk and remove all dead code(ops)
void runLocalDCE(mlir::func::FuncOp func);

}  // namespace vpux
