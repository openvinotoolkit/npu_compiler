//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_per_axis_tile_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_per_axis_tile_dma.hpp"

namespace vpux::VPUIP::arch40xx {

UnrollPerAxisTileDMAStrategy::UnrollPerAxisTileDMAStrategy(int64_t dmaPortCount): _dmaPortCount(dmaPortCount) {
}

void UnrollPerAxisTileDMAStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::VPUIP::SingleClusterPerAxisTileDMARewriter>(ctx, _dmaPortCount, log);
    patterns.add<vpux::VPUIP::MultiClusterPerAxisTileDMARewriter>(ctx, _dmaPortCount, false, log);
}

}  // namespace vpux::VPUIP::arch40xx
