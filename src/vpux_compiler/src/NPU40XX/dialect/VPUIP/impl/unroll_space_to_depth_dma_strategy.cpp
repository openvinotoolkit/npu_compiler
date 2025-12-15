//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_space_to_depth_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_space_to_depth_dma.hpp"

namespace vpux::VPUIP::arch40xx {

UnrollSpaceToDepthDMAStrategy::UnrollSpaceToDepthDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount)
        : _ctx(ctx), _dmaPortCount(dmaPortCount) {
}

void UnrollSpaceToDepthDMAStrategy::addPatterns(llvm::SmallVector<mlir::RewritePatternSet>& patterns,
                                                Logger& log) const {
    mlir::RewritePatternSet patternSet1(_ctx);
    patternSet1.add<vpux::VPUIP::MultiClusterSpaceToDepthDMARewriter>(_ctx, _dmaPortCount, log);
    mlir::RewritePatternSet patternSet2(_ctx);
    patternSet2.add<vpux::VPUIP::SingleClusterSpaceToDepthDMARewriter>(_ctx, _dmaPortCount, log);

    patterns.push_back(std::move(patternSet1));
    patterns.push_back(std::move(patternSet2));
}

}  // namespace vpux::VPUIP::arch40xx
