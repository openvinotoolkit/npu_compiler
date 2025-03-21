//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTASYNCOPSTOTASKS
#define GEN_PASS_DEF_CONVERTASYNCOPSTOTASKS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ConvertAsyncOpsToTasksPass
//

class ConvertAsyncOpsToTasksPass final : public VPUIP::impl::ConvertAsyncOpsToTasksBase<ConvertAsyncOpsToTasksPass> {
public:
    explicit ConvertAsyncOpsToTasksPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// InlineAsyncRegion
//

class InlineAsyncRegion final : public mlir::OpConversionPattern<mlir::async::ExecuteOp> {
public:
    InlineAsyncRegion(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::async::ExecuteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InlineAsyncRegion::matchAndRewrite(mlir::async::ExecuteOp execOp, OpAdaptor newArgs,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found 'async.execute' operation at '{0}'", execOp->getLoc());

    auto yieldOp = mlir::dyn_cast_or_null<mlir::async::YieldOp>(execOp.getBody()->getTerminator());
    VPUX_THROW_UNLESS(yieldOp != nullptr, "'async.execute' body doesn't have corresponding terminator");

    auto barrierOp = rewriter.create<VPURT::DeclareVirtualBarrierOp>(execOp.getLoc());

    _log.nest(1).trace("Traverse 'async.execute' body");

    auto blockArgs = execOp.getBody()->getArguments();

    const auto reassignBlockArgs = [&](mlir::Operation* op) {
        auto opInputs = op->getOperands();
        for (unsigned int idx = 0; idx < opInputs.size(); idx++) {
            auto input = opInputs[idx];
            for (auto arg : blockArgs) {
                if (input == arg) {
                    op->setOperand(idx, newArgs.getBodyOperands()[arg.getArgNumber()]);
                    break;
                }
            }
        }
    };

    for (auto& op : llvm::make_early_inc_range(execOp.getBody()->without_terminator())) {
        if (VPUIP::isPureViewOp(&op)) {
            reassignBlockArgs(&op);
            op.moveBefore(execOp);
            continue;
        }

        if (mlir::isa<VPURT::DeclareBufferOp>(op)) {
            op.moveBefore(execOp);
            continue;
        }

        mlir::SmallVector<mlir::Value> waitBarriers, updateBarriers;
        if (!newArgs.getDependencies().empty()) {
            _log.nest(3).trace("Append it's wait barrier list");
            waitBarriers.append(newArgs.getDependencies().begin(), newArgs.getDependencies().end());
        }
        if (!execOp.getToken().use_empty()) {
            _log.nest(3).trace("Append it's update barrier list");
            updateBarriers.push_back(barrierOp.getBarrier());
        }

        auto taskOp = rewriter.create<VPURT::TaskOp>(execOp.getLoc(), waitBarriers, updateBarriers);
        auto& block = taskOp.getBody().emplaceBlock();
        // If the operation has operand from its block. Need to reset it to Execute operand.
        reassignBlockArgs(&op);
        op.moveBefore(&block, block.end());
    }

    // Same goes to 'yieldOp', need to reset respective 'blockArg' to Execute operand.
    auto opInputs = yieldOp->getOperands();
    for (unsigned int idx = 0; idx < opInputs.size(); idx++) {
        auto input = opInputs[idx];
        for (auto arg : blockArgs) {
            if (input == arg) {
                yieldOp->setOperand(idx, newArgs.getBodyOperands()[arg.getArgNumber()]);
                break;
            }
        }
    }

    SmallVector<mlir::Value> newResults;
    newResults.reserve(execOp->getNumResults());
    newResults.push_back(barrierOp.getBarrier());
    newResults.append(yieldOp.getOperands().begin(), yieldOp.getOperands().end());

    rewriter.replaceOp(execOp, newResults);

    return mlir::success();
}

//
// RemoveWait
//

class RemoveWait final : public mlir::OpConversionPattern<mlir::async::AwaitOp> {
public:
    RemoveWait(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::async::AwaitOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::AwaitOp waitOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveWait::matchAndRewrite(mlir::async::AwaitOp waitOp, OpAdaptor newArgs,
                                                mlir::ConversionPatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(waitOp.getResult() != nullptr, "'async.await' Operation without result is not supported");

    // Pure view like operation were replaced with DeclareTensorOp
    // so the remaining Await ops have no users
    if (waitOp.getResult().use_empty()) {
        rewriter.eraseOp(waitOp);
        return mlir::success();
    }

    // If you faced with "'async.await' has more than one consumer" make sure you add your layer
    // into src/vpux_compiler/src/dialect/VPUIP/ops.cpp `redirectOpInterfacesForIE(...)`
    // and `redirectOpInterfacesForIERT(...)`
    VPUX_THROW_UNLESS(waitOp.getResult().hasOneUse(), "'async.await' doesn't have only one consumer");
    VPUX_THROW_UNLESS(mlir::isa<mlir::func::ReturnOp>(*waitOp.getResult().user_begin()),
                      "'async.await' has non 'return' consumer");

    rewriter.replaceOp(waitOp, newArgs.getOperand());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertAsyncOpsToTasksPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto func = getOperation();

    mlir::TypeConverter typeConverter;

    typeConverter.addConversion([](mlir::async::TokenType token) {
        return VPURT::BarrierType::get(token.getContext());
    });
    typeConverter.addTargetMaterialization(dummyConverter<VPURT::BarrierType>);
    typeConverter.addArgumentMaterialization(dummyConverter<VPURT::BarrierType>);
    typeConverter.addSourceMaterialization(dummyConverter<mlir::async::TokenType>);

    typeConverter.addConversion([](mlir::async::ValueType future) {
        return future.getValueType();
    });
    typeConverter.addTargetMaterialization(dummyConverter<mlir::MemRefType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::MemRefType>);
    typeConverter.addSourceMaterialization(dummyConverter<mlir::async::ValueType>);

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InlineAsyncRegion>(typeConverter, &ctx, _log);
    patterns.add<RemoveWait>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAsyncOpsToTasksPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertAsyncOpsToTasksPass(Logger log) {
    return std::make_unique<ConvertAsyncOpsToTasksPass>(log);
}
