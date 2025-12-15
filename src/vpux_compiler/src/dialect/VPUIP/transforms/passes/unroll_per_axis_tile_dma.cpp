//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_per_axis_tile_dma_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLPERAXISTILEDMA
#define GEN_PASS_DEF_UNROLLPERAXISTILEDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollPerAxisTileDMAPass
//

class UnrollPerAxisTileDMAPass final : public VPUIP::impl::UnrollPerAxisTileDMABase<UnrollPerAxisTileDMAPass> {
public:
    explicit UnrollPerAxisTileDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollPerAxisTileDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollPerAxisTileDMAPass)) {
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    auto unrollStrategy = VPUIP::createUnrollPerAxisTileDMAStrategy(func);
    unrollStrategy->addPatterns(patterns, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollPerAxisTileDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollPerAxisTileDMAPass(Logger log) {
    return std::make_unique<UnrollPerAxisTileDMAPass>(log);
}
