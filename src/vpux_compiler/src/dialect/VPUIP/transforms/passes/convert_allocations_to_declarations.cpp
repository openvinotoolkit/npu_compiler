//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTALLOCATIONSTODECLARATIONS
#define GEN_PASS_DEF_CONVERTALLOCATIONSTODECLARATIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/convert_allocations_to_declarations.hpp.inc>

//
// ConvertAllocationsToDeclarationsPass
//

class ConvertAllocationsToDeclarationsPass final :
        public VPUIP::impl::ConvertAllocationsToDeclarationsBase<ConvertAllocationsToDeclarationsPass> {
public:
    explicit ConvertAllocationsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertAllocationsToDeclarationsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addIllegalOp<VPUIP::StaticAllocOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAllocationsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertAllocationsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertAllocationsToDeclarationsPass>(log);
}
