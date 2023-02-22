//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/enums.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

class FusePermuteRewrite final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    FusePermuteRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        setDebugName("FusePermuteRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePermuteRewrite::matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto outOrder = DimsOrder::fromValue(origOp.output());

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    SmallVector<int64_t> noPadBeginEnd(inOrder.numDims(), 0);
    const auto& ctx = origOp.getContext();
    const auto permQuantOutType = origOp.output().getType();
    const auto permQuantElemType = permQuantOutType.cast<vpux::NDTypeInterface>().getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = appendLoc(origOp->getLoc(), "PermuteQuantize");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, origOp.input(), origOp.dstOrderAttr(), memPermAttr, dstElemTypeAttr,
            getIntArrayAttr(ctx, noPadBeginEnd), getIntArrayAttr(ctx, noPadBeginEnd));

    rewriter.replaceOp(origOp, permuteQuantizeOp.output());

    return mlir::success();
}

//
// ConvertReorderToPermuteQuantizePass
//

class ConvertReorderToPermuteQuantizePass final :
        public IE::ConvertReorderToPermuteQuantizeBase<ConvertReorderToPermuteQuantizePass> {
public:
    explicit ConvertReorderToPermuteQuantizePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool isSupportedReorder(IE::ReorderOp reorder, Logger log) const;

private:
    Logger _log;
};

bool ConvertReorderToPermuteQuantizePass::isSupportedReorder(IE::ReorderOp reorder, Logger log) const {
    const auto inType = reorder.input().getType().dyn_cast<vpux::NDTypeInterface>();
    if (inType == nullptr) {
        log.trace("Input type does not implement NDTypeInterface");
        return false;
    }
    const auto outType = reorder.output().getType().dyn_cast<vpux::NDTypeInterface>();
    if (outType == nullptr) {
        log.trace("Output type does not implement NDTypeInterface");
        return false;
    }
    const auto inOrder = inType.getDimsOrder();
    const auto expectedInOrder = DimsOrder::NCHW;
    if (inOrder != expectedInOrder) {
        log.trace("Unsupported input layout. Expected: '{0}', got: '{1}'", expectedInOrder, inOrder);
        return false;
    }
    const auto outOrder = outType.getDimsOrder();
    const auto expectedOutOrder = DimsOrder::NHWC;
    if (outOrder != expectedOutOrder) {
        log.trace("Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, outOrder);
        return false;
    }
    const auto inElemType = inType.getElementType();
    if (!inElemType.isF16()) {
        log.trace("Unsupported input element type. Expected: f16, got: '{0}'", inElemType);
        return false;
    }
    const auto outElemType = outType.getElementType();
    if (!outElemType.isF16()) {
        log.trace("Unsupported output element type. Expected: f16, got: '{0}'", outElemType);
        return false;
    }
    const ShapeRef inShape = inType.getShape();
    const auto inputAlignment = VPU::NCEInvariant::getAlignment(inElemType);
    if (!vpux::IE::isShapeCompatibleWithODUPermute(inShape, inputAlignment)) {
        log.trace("Cannot cast input shape {0} to 1xWxCxH", inShape);
        return false;
    }
    const ShapeRef outShape = outType.getShape();
    const auto outputAlignment = VPU::NCEInvariant::getAlignment(outElemType);
    if (!vpux::IE::isShapeCompatibleWithODUPermute(outShape, outputAlignment)) {
        log.trace("Cannot cast output shape {0} to 1xWxCxH", outShape);
        return false;
    }

    return true;
}

void ConvertReorderToPermuteQuantizePass::safeRunOnFunc() {
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("ConvertReorderToPermuteQuantizePass enabled only for VPUX37XX device. Got: {0}", arch);
        return;
    }

    const auto isLegalReorder = [&](IE::ReorderOp reorder) -> bool {
        return !isSupportedReorder(reorder, _log);
    };
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ReorderOp>(isLegalReorder);
    target.addLegalOp<IE::PermuteQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusePermuteRewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFusePermuteQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertReorderToPermuteQuantizePass(Logger log) {
    return std::make_unique<ConvertReorderToPermuteQuantizePass>(log);
}
