//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/SmallVector.h>

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
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(quantizedType)) {
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
    auto newPermuteCastOp = rewriter.createOrFold<IE::PermuteCastOp>(origOp.getLoc(), parentDequantOp.getInput(),
                                                                     origOp.getDstOrder(), origOp.getMemPerm());
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, newPermuteCastOp, parentDequantOp.getDstElemType());
    return mlir::success();
}

//
// PropagatePermuteCastThroughDequantizePass
//

class PropagatePermuteCastThroughDequantizePass final :
        public vpux::IE::impl::PropagatePermuteCastThroughDequantizeBase<PropagatePermuteCastThroughDequantizePass> {
public:
    explicit PropagatePermuteCastThroughDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagatePermuteCastThroughDequantizePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteCastRewriter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagatePermuteCastThroughDequantizePass(Logger log) {
    return std::make_unique<PropagatePermuteCastThroughDequantizePass>(log);
}
