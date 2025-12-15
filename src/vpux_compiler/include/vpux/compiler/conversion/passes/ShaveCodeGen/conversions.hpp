//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class RewritePatternSet;
}  // namespace mlir

namespace vpux::ShaveCodeGen {
void populateIEReduceToLinalgPatterns(mlir::RewritePatternSet& patternSet);
void populateIEDataMovementToTensorPatterns(mlir::RewritePatternSet& patternSet);
void populateIEShapeManipulationToTensorPatterns(mlir::RewritePatternSet& patternSet);
}  // namespace vpux::ShaveCodeGen
