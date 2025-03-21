//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/AnalysisManager.h>
#include "vpux/utils/core/logger.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLDMAANALYSIS
#define GEN_PASS_DEF_UNROLLDMAANALYSIS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollDMAAnalysis
//

class UnrollDMAAnalysisPass final : public VPUIP::impl::UnrollDMAAnalysisBase<UnrollDMAAnalysisPass> {
public:
    UnrollDMAAnalysisPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollDMAAnalysisPass::safeRunOnFunc() {
    [[maybe_unused]] auto resultAnalysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollDMAAnalysisPass(Logger log) {
    return std::make_unique<UnrollDMAAnalysisPass>(log);
}
