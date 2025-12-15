//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_permute_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_permute_dma.hpp"

namespace vpux::VPUIP::arch40xx {

UnrollPermuteDMAStrategy::UnrollPermuteDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount)
        : _ctx(ctx), _dmaPortCount(dmaPortCount) {
}

void UnrollPermuteDMAStrategy::addPatterns(llvm::SmallVector<mlir::RewritePatternSet>& patterns, Logger& log) const {
    mlir::RewritePatternSet patternSet1(_ctx);
    patternSet1.add<vpux::VPUIP::MultiClusterPermuteDMARewriter>(_ctx, _dmaPortCount, log);
    mlir::RewritePatternSet patternSet2(_ctx);
    patternSet2.add<vpux::VPUIP::SingleClusterPermuteDMARewriter>(_ctx, _dmaPortCount, log);

    patterns.push_back(std::move(patternSet1));
    patterns.push_back(std::move(patternSet2));
}

}  // namespace vpux::VPUIP::arch40xx
