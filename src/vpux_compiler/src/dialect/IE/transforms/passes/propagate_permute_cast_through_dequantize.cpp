//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/SmallVector.h>
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEPERMUTECASTTHROUGHDEQUANTIZE
#define GEN_PASS_DEF_PROPAGATEPERMUTECASTTHROUGHDEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// PermuteCastRewriter
//

class PermuteCastRewriter final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    PermuteCastRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PermuteCastOp>(ctx), _log(log) {
        this->setDebugName("PermuteCastRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isSupportedDequantizeLayout(IE::PermuteCastOp permuteCastOp, IE::DequantizeOp dequantizeOp) {
    // Doing similar check to verifyDequantizeLayoutInfo with same rules
    const auto permutedCastOutput = permuteCastOp.getOutput();
    const auto permuteCastOutputOrder = DimsOrder::fromValue(permutedCastOutput);
    const auto inputType = mlir::cast<NDTypeInterface>(dequantizeOp.getInput().getType());
    const auto quantizedType = mlir::cast<mlir::quant::QuantizedType>(inputType.getElementType());
    SmallVector<DimsOrder> supportedLayouts;
    if (quantizedType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto numDims = inputType.getRank();
        if (numDims == 3) {
            supportedLayouts = {DimsOrder::HWC};
        } else if (numDims == 4) {
            supportedLayouts = {DimsOrder::NHWC};
        } else {
            VPUX_THROW("Unsupported rank '{0}'", numDims);
        }
    } else {
        supportedLayouts = {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC};
    }
    return std::find(supportedLayouts.begin(), supportedLayouts.end(), permuteCastOutputOrder) !=
           supportedLayouts.end();
}

mlir::LogicalResult PermuteCastRewriter::matchAndRewrite(IE::PermuteCastOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto parentDequantOp = origOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (!parentDequantOp) {
        return mlir::failure();
    }

    if (!isSupportedDequantizeLayout(origOp, parentDequantOp)) {
        return mlir::failure();
    }
    auto newPermuteCastOp = rewriter.create<IE::PermuteCastOp>(origOp.getLoc(), parentDequantOp.getInput(),
                                                               origOp.getDstOrder(), origOp.getMemPerm());
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, newPermuteCastOp.getResult(),
                                                  parentDequantOp.getDstElemType());
    return mlir::success();
}

//
// PropagatePermuteCastThroughDequantizePass
//

class PropagatePermuteCastThroughDequantizePass final :
        public vpux::IE::impl::PropagatePermuteCastThroughDequantizeBase<PropagatePermuteCastThroughDequantizePass> {
public:
    explicit PropagatePermuteCastThroughDequantizePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagatePermuteCastThroughDequantizePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteCastRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagatePermuteCastThroughDequantizePass(Logger log) {
    return std::make_unique<PropagatePermuteCastThroughDequantizePass>(log);
}
