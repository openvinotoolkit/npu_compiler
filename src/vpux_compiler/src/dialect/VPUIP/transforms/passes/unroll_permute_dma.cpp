//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_permute_dma_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLPERMUTEDMA
#define GEN_PASS_DEF_UNROLLPERMUTEDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollPermuteDMAPass
//

class UnrollPermuteDMAPass final : public VPUIP::impl::UnrollPermuteDMABase<UnrollPermuteDMAPass> {
public:
    explicit UnrollPermuteDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPermuteDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollPermuteDMAPass)) {
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    auto unrollStrategy = VPUIP::createUnrollPermuteDMAStrategy(func);
    unrollStrategy->addPatterns(patterns, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPermuteDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollPermuteDMAPass(Logger log) {
    return std::make_unique<UnrollPermuteDMAPass>(log);
}
