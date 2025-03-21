//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_INVALIDATEUNROLLDMAANALYSIS
#define GEN_PASS_DEF_INVALIDATEUNROLLDMAANALYSIS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UnrollDMAAnalysis
//

class InvalidateUnrollDMAAnalysisPass final :
        public VPUIP::impl::InvalidateUnrollDMAAnalysisBase<InvalidateUnrollDMAAnalysisPass> {
public:
    InvalidateUnrollDMAAnalysisPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void InvalidateUnrollDMAAnalysisPass::safeRunOnFunc() {
    // Invalidates UnrollDMAAnalysis by not calling markAnalysesPreserved
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createInvalidateUnrollDMAAnalysisPass(Logger log) {
    return std::make_unique<InvalidateUnrollDMAAnalysisPass>(log);
}
