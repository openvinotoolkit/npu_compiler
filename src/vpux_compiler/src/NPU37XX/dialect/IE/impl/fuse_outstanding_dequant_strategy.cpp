//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_outstanding_dequant_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_dequant.hpp"

namespace vpux::IE::arch37xx {

//
// FuseOutstandingDequantStrategy
//

void FuseOutstandingDequantStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::IE::DequantizeWithNCEReLUxRewriter>(ctx, log);
}

}  // namespace vpux::IE::arch37xx
