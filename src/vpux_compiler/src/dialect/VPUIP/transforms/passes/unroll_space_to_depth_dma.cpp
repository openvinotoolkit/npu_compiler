//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_space_to_depth_dma_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLSPACETODEPTHDMA
#define GEN_PASS_DEF_UNROLLSPACETODEPTHDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollSpaceToDepthDMAPass
//

class UnrollSpaceToDepthDMAPass final : public VPUIP::impl::UnrollSpaceToDepthDMABase<UnrollSpaceToDepthDMAPass> {
public:
    explicit UnrollSpaceToDepthDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSpaceToDepthDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollSpaceToDepthDMAPass)) {
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    auto unrollStrategy = VPUIP::createUnrollSpaceToDepthDMAStrategy(func);
    unrollStrategy->addPatterns(patterns, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollSpaceToDepthDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollSpaceToDepthDMAPass(Logger log) {
    return std::make_unique<UnrollSpaceToDepthDMAPass>(log);
}
