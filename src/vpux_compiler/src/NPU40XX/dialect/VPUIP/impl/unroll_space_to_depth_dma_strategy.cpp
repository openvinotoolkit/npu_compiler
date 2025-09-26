//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_space_to_depth_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_space_to_depth_dma.hpp"

namespace vpux::VPUIP::arch40xx {

UnrollSpaceToDepthDMAStrategy::UnrollSpaceToDepthDMAStrategy(int64_t dmaPortCount): _dmaPortCount(dmaPortCount) {
}

void UnrollSpaceToDepthDMAStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::VPUIP::SingleClusterSpaceToDepthDMARewriter>(ctx, _dmaPortCount, log);
    patterns.add<vpux::VPUIP::MultiClusterSpaceToDepthDMARewriter>(ctx, _dmaPortCount, log);
}

}  // namespace vpux::VPUIP::arch40xx
