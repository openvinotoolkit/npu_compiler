//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_depth_to_space_dma_strategy.hpp"
#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_depth_to_space_dma.hpp"

namespace vpux::VPUIP::arch40xx {

UnrollDepthToSpaceDMAStrategy::UnrollDepthToSpaceDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount)
        : _ctx(ctx), _dmaPortCount(dmaPortCount) {
}

void UnrollDepthToSpaceDMAStrategy::addPatterns(llvm::SmallVector<mlir::RewritePatternSet>& patterns,
                                                Logger& log) const {
    mlir::RewritePatternSet patternSet1(_ctx);
    patternSet1.add<vpux::VPUIP::MultiClusterDepthToSpaceDMARewriter>(_ctx, _dmaPortCount, log);
    mlir::RewritePatternSet patternSet2(_ctx);
    patternSet2.add<vpux::VPUIP::SingleClusterDepthToSpaceDMARewriter>(_ctx, _dmaPortCount, log);

    patterns.push_back(std::move(patternSet1));
    patterns.push_back(std::move(patternSet2));
}

}  // namespace vpux::VPUIP::arch40xx
