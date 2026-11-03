//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_planner.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_collector.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edit_plan.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZEDMACOPIES
#define GEN_PASS_DEF_OPTIMIZEDMACOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// OptimizeDMACopiesPass
//

class OptimizeDMACopiesPass final : public VPUIP::impl::OptimizeDMACopiesBase<OptimizeDMACopiesPass> {
public:
    using Base = VPUIP::impl::OptimizeDMACopiesBase<OptimizeDMACopiesPass>;

    explicit OptimizeDMACopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeDMACopiesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto maxRounds = getDefaultGreedyRewriteConfig().getMaxIterations();
    for (int64_t round = 0; round < maxRounds; ++round) {
        // ---- Step 1: Collect edges
        VPUIP::DmaOptimization::EdgeCollector collector(_log);
        auto edgeCollection = collector.collect(func);
        _log.trace("OptimizeDMACopiesPass: round {0} Stage 1 collected {1} edges", round, edgeCollection.edges.size());
        if (edgeCollection.edges.empty()) {
            break;
        }

        // ---- Step 2: plan copy reuse and generate the edit plan
        VPUIP::DmaOptimization::EditPlan editPlan;
        auto& aliasesInfo = getAnalysis<AliasesInfo>();
        VPUIP::DmaOptimization::CopyReusePlanner copyReusePlanner(_log);
        const auto plannedReuseCount = copyReusePlanner.planCopyReuse(edgeCollection, aliasesInfo, editPlan);
        _log.trace("OptimizeDMACopiesPass: round {0} Stage 2 planned {1} copy reuses", round, plannedReuseCount);

        if (editPlan.empty()) {
            _log.trace("OptimizeDMACopiesPass: round {0} produced an empty plan; fixpoint reached", round);
            break;
        }

        // ---- Step 3: apply IR transformation collected by the EditPlan
        mlir::IRRewriter rewriter(&ctx);
        VPUIP::DmaOptimization::EditPlanApplier applier(_log);
        if (mlir::failed(applier.run(rewriter, editPlan))) {
            _log.error("OptimizeDMACopiesPass: round {0} EditPlan verification failed; aborting apply", round);
            break;
        }
        _log.trace("OptimizeDMACopiesPass: round {0} Stage 3 applied {1} edits", round, editPlan.size());

        // Next round must re-derive AliasesInfo against the mutated IR.
        getAnalysisManager().invalidate({});

        if (round + 1 == maxRounds) {
            _log.warning("OptimizeDMACopiesPass reached the {0}-round recollection limit", maxRounds);
        }
    }
}

}  // namespace

//
// createOptimizeDMACopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeDMACopiesPass(Logger log) {
    return std::make_unique<OptimizeDMACopiesPass>(log);
}
