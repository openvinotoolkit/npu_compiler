//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_permute_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_permute_dma.hpp"

namespace vpux::VPUIP::arch40xx {

UnrollPermuteDMAStrategy::UnrollPermuteDMAStrategy(int64_t dmaPortCount): _dmaPortCount(dmaPortCount) {
}

void UnrollPermuteDMAStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::VPUIP::SingleClusterPermuteDMARewriter>(ctx, _dmaPortCount, log);
    patterns.add<vpux::VPUIP::MultiClusterPermuteDMARewriter>(ctx, _dmaPortCount, log);
}

}  // namespace vpux::VPUIP::arch40xx
