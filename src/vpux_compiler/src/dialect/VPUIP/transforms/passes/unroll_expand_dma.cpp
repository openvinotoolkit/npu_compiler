//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_expand_dma.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_expand_dma_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLEXPANDDMA
#define GEN_PASS_DEF_UNROLLEXPANDDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

//
// UnrollExpandDMAPass
//

class UnrollExpandDMAPass final : public VPUIP::impl::UnrollExpandDMABase<UnrollExpandDMAPass> {
public:
    explicit UnrollExpandDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollExpandDMAPass::safeRunOnFunc() {
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollExpandDMAPass)) {
        return;
    }
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    auto unrollStrategy = VPUIP::createUnrollExpandDMAStrategy(func);

    unrollStrategy->addPatterns(patterns, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

//
// createUnrollExpandDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollExpandDMAPass(Logger log) {
    return std::make_unique<UnrollExpandDMAPass>(log);
}
