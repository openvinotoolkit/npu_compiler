//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPCONVERTWITHRESHAPEKINDOPS
#define GEN_PASS_DEF_SWAPCONVERTWITHRESHAPEKINDOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapConvertWithReshapeKindOps
//

class SwapConvertWithReshapeKindOps final :
        public IE::impl::SwapConvertWithReshapeKindOpsBase<SwapConvertWithReshapeKindOps> {
public:
    explicit SwapConvertWithReshapeKindOps(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class OpSwapConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

bool isReshapeKindOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return mlir::isa<IE::AffineReshapeOp, IE::DepthToSpaceOp, IE::ReshapeOp, IE::SqueezeOp, IE::TransposeOp,
                     IE::UnsqueezeOp>(op);
}

// For OV 2.0 API U8 we can have:
// NetworkInput (NCHW) -> Convert -> Transpose-> FQ . Because of this lately after propagate quantize
// dequantize pass and fuse convert with quantize pass, will be needed to propagate the quantizeCast
// quantParams to Transpose. We want to avoid this. Also in the end this Transpose will be done as
// PermuteCast.

// Output Case:
// Convert -> N reshapeKindOps -> return => N reshapeKindOps -> Convert -> return

bool canBeSwapped(IE::ConvertOp origOp, mlir::Operation* swapOp) {
    return (origOp.getDstElemType().isUnsignedInteger(8) &&
            mlir::isa<mlir::func::ReturnOp>(*swapOp->getResult(0).getUsers().begin())) ||
           mlir::isa<mlir::BlockArgument>(origOp.getInput());
}

void swapOps(IE::ConvertOp origOp, mlir::Operation* swapOp, mlir::PatternRewriter& rewriter) {
    const auto origDataType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto swapDataType = mlir::cast<vpux::NDTypeInterface>(swapOp->getResult(0).getType());
    const auto newDataType = swapDataType.changeElemType(origDataType.getElementType());

    rewriter.setInsertionPointAfter(swapOp);
    auto newConvert = rewriter.create<IE::ConvertOp>(origOp->getLoc(), swapOp->getResult(0), origOp.getDstElemType());
    swapOp->getResult(0).replaceAllUsesExcept(newConvert.getOutput(),
                                              llvm::SmallPtrSet<mlir::Operation*, 1>{newConvert});
    origOp->replaceAllUsesWith(mlir::ValueRange(origOp.getInput()));
    swapOp->getResult(0).setType(newDataType);
    rewriter.eraseOp(origOp);
}

mlir::Operation* findLastReshapeKindOp(mlir::Operation* swapOp) {
    while (isReshapeKindOp(*swapOp->getResult(0).getUsers().begin())) {
        swapOp = *swapOp->getResult(0).getUsers().begin();
    }
    return swapOp;
}

//
// OpSwapConverter
//

class SwapConvertWithReshapeKindOps::OpSwapConverter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    OpSwapConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapConvertWithReshapeKindOps::OpSwapConverter::matchAndRewrite(
        IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!origOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto swapOp = *origOp.getOutput().getUsers().begin();
    auto swapOpLoop = swapOp;

    if (isReshapeKindOp(swapOp)) {
        if (canBeSwapped(origOp, swapOp)) {
            swapOps(origOp, swapOp, rewriter);
            return mlir::success();
        }

        // Handle intermediate reshape kind ops
        swapOp = findLastReshapeKindOp(swapOp);

        if (!mlir::isa<mlir::func::ReturnOp>(*swapOp->getResult(0).getUsers().begin())) {
            return mlir::failure();
        }

        // Process swap Convert loop for reshape kind ops
        while (isReshapeKindOp(*swapOpLoop->getResult(0).getUsers().begin())) {
            swapOps(origOp, swapOpLoop, rewriter);
            return mlir::success();
        }
    }

    return mlir::failure();
}

void SwapConvertWithReshapeKindOps::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapConvertWithReshapeKindOps::OpSwapConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapConvertWithReshapeKindOpsPass(Logger log) {
    return std::make_unique<SwapConvertWithReshapeKindOps>(log);
}
