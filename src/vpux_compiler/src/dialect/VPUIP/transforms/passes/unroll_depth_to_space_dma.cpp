//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_depth_to_space_dma_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLDEPTHTOSPACEDMA
#define GEN_PASS_DEF_UNROLLDEPTHTOSPACEDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollDepthToSpaceDMAPass
//

class UnrollDepthToSpaceDMAPass final : public VPUIP::impl::UnrollDepthToSpaceDMABase<UnrollDepthToSpaceDMAPass> {
public:
    explicit UnrollDepthToSpaceDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollDepthToSpaceDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollDepthToSpaceDMAPass)) {
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    auto unrollStrategy = VPUIP::createUnrollDepthToSpaceDMAStrategy(func);
    unrollStrategy->addPatterns(patterns, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollDepthToSpaceDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollDepthToSpaceDMAPass(Logger log) {
    return std::make_unique<UnrollDepthToSpaceDMAPass>(log);
}
