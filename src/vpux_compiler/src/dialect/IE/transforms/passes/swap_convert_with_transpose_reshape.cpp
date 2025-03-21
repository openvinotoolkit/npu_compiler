//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPCONVERTWITHTRANSPOSERESHAPE
#define GEN_PASS_DEF_SWAPCONVERTWITHTRANSPOSERESHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapConvertWithTransposeReshape
//

class SwapConvertWithTransposeReshape final :
        public IE::impl::SwapConvertWithTransposeReshapeBase<SwapConvertWithTransposeReshape> {
public:
    explicit SwapConvertWithTransposeReshape(Logger log): _log(log) {
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
    return mlir::isa<IE::TransposeOp, IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(op);
}

//
// OpSwapConverter
//

class SwapConvertWithTransposeReshape::OpSwapConverter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    OpSwapConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapConvertWithTransposeReshape::OpSwapConverter::matchAndRewrite(
        IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const {
    auto swapOp = *origOp.getOutput().getUsers().begin();
    if (isReshapeKindOp(swapOp)) {
        const auto origDataType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
        auto swapDataType = swapOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const auto newDataType = swapDataType.changeElemType(origDataType.getElementType());

        rewriter.setInsertionPointAfter(swapOp);
        auto newConvert =
                rewriter.create<IE::ConvertOp>(origOp->getLoc(), swapOp->getResult(0), origOp.getDstElemType());
        swapOp->getResult(0).replaceAllUsesExcept(newConvert.getOutput(),
                                                  llvm::SmallPtrSet<mlir::Operation*, 1>{newConvert});
        origOp->replaceAllUsesWith(mlir::ValueRange(origOp.getInput()));
        swapOp->getResult(0).setType(newDataType);
        rewriter.eraseOp(origOp);
    }

    return mlir::success();
}

void SwapConvertWithTransposeReshape::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    const auto isLegalOp = [](IE::ConvertOp op) -> bool {
        if (!op.getOutput().hasOneUse()) {
            return true;
        }
        auto childOp = *op.getOutput().getUsers().begin();
        if (isReshapeKindOp(childOp)) {
            // For OV 2.0 API U8 we can have:
            // NetworkInput (NCHW) -> Convert -> Transpose-> FQ . Because of this lately after propagate quantize
            // dequantize pass and fuse convert with quantize pass, will be needed to propagate the quantizeCast
            // quantParams to Transpose. We want to avoid this. Also in the end this Transpose will be done as
            // PermuteCast.
            return !op.getInput().isa<mlir::BlockArgument>();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvertOp>(isLegalOp);
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::AffineReshapeOp>();
    target.addLegalOp<IE::SqueezeOp>();
    target.addLegalOp<IE::UnsqueezeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapConvertWithTransposeReshape::OpSwapConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapConvertWithTransposeReshapePass(Logger log) {
    return std::make_unique<SwapConvertWithTransposeReshape>(log);
}
