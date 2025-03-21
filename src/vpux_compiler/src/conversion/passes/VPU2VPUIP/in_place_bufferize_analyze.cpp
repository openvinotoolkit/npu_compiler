//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

namespace vpux {
#define GEN_PASS_DECL_INPLACEBUFFERIZATIONANALYZE
#define GEN_PASS_DEF_INPLACEBUFFERIZATIONANALYZE
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// InPlaceBufferizationAnalyzePass
//

class InPlaceBufferizationAnalyzePass final :
        public impl::InPlaceBufferizationAnalyzeBase<InPlaceBufferizationAnalyzePass> {
private:
    void safeRunOnModule() final;
};

void InPlaceBufferizationAnalyzePass::safeRunOnModule() {
    mlir::bufferization::OneShotBufferizationOptions options = vpux::getOneShotBufferizationOptions();
    mlir::ModuleOp moduleOp = getOperation();

    mlir::bufferization::OneShotAnalysisState state(moduleOp, options);
    if (mlir::failed(mlir::bufferization::analyzeOp(moduleOp, state, /*statistics=*/nullptr))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createInPlaceBufferizationAnalyzePass
//

std::unique_ptr<mlir::Pass> vpux::createInPlaceBufferizationAnalyzePass() {
    return std::make_unique<InPlaceBufferizationAnalyzePass>();
}
