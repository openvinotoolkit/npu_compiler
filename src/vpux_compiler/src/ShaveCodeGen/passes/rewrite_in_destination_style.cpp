//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_REWRITEINDESTINATIONSTYLE
#define GEN_PASS_DEF_REWRITEINDESTINATIONSTYLE
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// RewriteInDestinationStylePass
//

class RewriteInDestinationStylePass final :
        public vpux::ShaveCodeGen::impl::RewriteInDestinationStyleBase<RewriteInDestinationStylePass> {
public:
    explicit RewriteInDestinationStylePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void RewriteInDestinationStylePass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });
    target.addIllegalOp<mlir::tensor::FromElementsOp, mlir::tensor::GenerateOp, mlir::tensor::PadOp>();

    mlir::RewritePatternSet patterns(&ctx);
    mlir::linalg::populateConvertToDestinationStylePatterns(patterns);

    if (mlir::failed(mlir::applyPartialConversion(swModule, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createRewriteInDestinationStylePass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createRewriteInDestinationStylePass(Logger log) {
    return std::make_unique<RewriteInDestinationStylePass>(log);
}
