//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class RewritePatternSet;
}

namespace vpux::ShaveCodeGen {
void populateIEReduceToLinalgPatterns(mlir::RewritePatternSet& patternSet);
}
