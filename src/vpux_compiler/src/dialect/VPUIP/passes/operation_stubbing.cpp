//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// StubConversion
//

class StubConversion final : public mlir::ConversionPattern {
public:
    StubConversion(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}, benefitLow, ctx), _log(log) {
        this->setDebugName("StubConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StubConversion::matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation '{1}'", origOp->getName(), origOp->getLoc());
    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(), "Got wrong newOperands size : '{0}'",
                      newOperands.size());

    if (origOp->getName().getDialectNamespace() == IE::IEDialect::getDialectNamespace()) {
        rewriter.replaceOpWithNewOp<IE::StubOp>(origOp, origOp->getResults().getTypes(), origOp->getOperands());
    } else if (origOp->getName().getDialectNamespace() == VPU::VPUDialect::getDialectNamespace()) {
        rewriter.replaceOpWithNewOp<VPU::StubOp>(origOp, origOp->getResults().getTypes(), origOp->getOperands());
    } else if (origOp->getName().getDialectNamespace() == VPUIP::VPUIPDialect::getDialectNamespace()) {
        rewriter.replaceOpWithNewOp<VPUIP::StubOp>(origOp, origOp->getResults().getTypes(), origOp->getOperands());
    } else {
        return mlir::failure();
    }

    _log.trace("Replaced with 'Stub' Operation");

    return mlir::success();
}

//
// OperationStubbing
//

class OperationStubbing final : public VPUIP::OperationStubbingBase<OperationStubbing> {
public:
    explicit OperationStubbing(std::function<bool(mlir::Operation*)> condition, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        _condition = condition;
    }

private:
    void safeRunOnFunc() final;
    std::function<bool(mlir::Operation*)> _condition;
};

void OperationStubbing::safeRunOnFunc() {
    auto& ctx = getContext();

    vpux::BufferizeTypeConverter typeConverter;
    mlir::ConversionTarget target(ctx);

    // Legal Ops and Dialects
    target.addLegalOp<IE::StubOp>();
    target.addLegalOp<VPU::StubOp>();
    target.addLegalOp<VPUIP::StubOp>();

    // Dynamically Legal Ops
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        return !(_condition(op));
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<StubConversion>(typeConverter, &ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOperationStubbingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOperationStubbingPass(std::function<bool(mlir::Operation*)> condition,
                                                                     Logger log) {
    return std::make_unique<OperationStubbing>(condition, log);
}
