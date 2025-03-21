//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLREDUCEMINALLAXES
#define GEN_PASS_DEF_UNROLLREDUCEMINALLAXES
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ReduceMinRewriter
//

class ReduceMinRewriter final : public mlir::OpRewritePattern<IE::ReduceMinOp> {
public:
    ReduceMinRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMinOp>(ctx), _log(log) {
        setDebugName("ReduceMinRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMinRewriter::matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Try to unroll ReduceMin at '{1}'", getDebugName(), origOp->getLoc(), origOp);
    mlir::MLIRContext* ctx = origOp->getContext();
    const auto origInput = origOp->getOperand(0);
    const auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());

    // Unroll ReduceMin on all axes to multiple ReduceMin ops on single axis
    auto prevInput = origInput;
    for (size_t idx : irange(axes.size())) {
        auto inputShape = getShape(prevInput);
        _log.trace("[{0}] Search for the {1}-th axis to unroll based on input shape {2} and input {3}", getDebugName(),
                   idx, inputShape, prevInput);
        auto maxNonOneAxis = getMaxNonOneDim(inputShape);
        // create single-axis ReduceMin on the largest non-one axis
        auto maxAxis = maxNonOneAxis.value_or(Dim(0)).ind();
        SmallVector<int64_t> unrolledAxes = {maxAxis};
        auto axesAttr = getIntArrayAttr(ctx, ArrayRef(unrolledAxes));
        auto newOp = rewriter.create<IE::ReduceMinOp>(takeOpLoc(origOp, StringLiteral("dim_{0}"), idx), prevInput,
                                                      nullptr, axesAttr, false);
        prevInput = newOp->getResult(0);
        _log.trace("[{0}] create newOp {1} with size {2}", getDebugName(), newOp,
                   parseIntArrayAttr<int64_t>(newOp.getAxesValue().value()).size());
    }
    rewriter.replaceOp(origOp, prevInput);
    return mlir::success();
}

//
// UnrollReduceMinAllAxesPass
//

class UnrollReduceMinAllAxesPass final : public IE::impl::UnrollReduceMinAllAxesBase<UnrollReduceMinAllAxesPass> {
public:
    explicit UnrollReduceMinAllAxesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollReduceMinAllAxesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegalReduceMin = [&](IE::ReduceMinOp op) {
        // Skip if axes is empty or has only one element
        if (parseIntArrayAttr<int64_t>(op.getAxesValue().value()).size() <= 1) {
            return true;
        }

        // For ReduceMin, in case all axes are reduced, conversion to MaxPool on NCE shows suboptimal
        // performance than SHAVE when the kernel size exceeds maxKernelSize.
        // Instead, unroll it to multiple ReduceMin ops on single axis can avoid this issue
        // when totalsize is over REDUCEMIN_DPU_THRESHOLD as profiled in E#-126141.
        auto inputTotalSize = getShape(op->getOperand(0)).totalSize();
        if (inputTotalSize < REDUCEMIN_DPU_THRESHOLD) {
            return true;
        }

        const auto maxKernelSize = VPU::getMaxKernelSize(op);
        if ((getShape(op->getResult(0)).totalSize() == 1) && (inputTotalSize > std::pow(maxKernelSize, 2))) {
            return false;
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ReduceMinOp>(isLegalReduceMin);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReduceMinRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollReduceMinAllAxesPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollReduceMinAllAxesPass(Logger log) {
    return std::make_unique<UnrollReduceMinAllAxesPass>(log);
}
