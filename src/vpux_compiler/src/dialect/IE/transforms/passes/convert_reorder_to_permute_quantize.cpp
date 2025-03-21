//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTREORDERTOPERMUTEQUANTIZE
#define GEN_PASS_DEF_CONVERTREORDERTOPERMUTEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

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

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    SmallVector<int64_t> noPadBeginEnd(inOrder.numDims(), 0);
    const auto& ctx = origOp.getContext();
    const auto permQuantOutType = origOp.getOutput().getType();
    const auto permQuantElemType = permQuantOutType.cast<vpux::NDTypeInterface>().getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = appendLoc(origOp->getLoc(), "PermuteQuantize");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, origOp.getInput(), origOp.getDstOrderAttr(), memPermAttr, dstElemTypeAttr,
            getIntArrayAttr(ctx, noPadBeginEnd), getIntArrayAttr(ctx, noPadBeginEnd));

    rewriter.replaceOp(origOp, permuteQuantizeOp.getOutput());

    return mlir::success();
}

//
// ConvertReorderToPermuteQuantizePass
//

class ConvertReorderToPermuteQuantizePass final :
        public IE::impl::ConvertReorderToPermuteQuantizeBase<ConvertReorderToPermuteQuantizePass> {
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
    const auto inType = reorder.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    if (inType == nullptr) {
        log.trace("Input type does not implement NDTypeInterface");
        return false;
    }
    const auto outType = reorder.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
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
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inElemType);
    if (!IE::isODUPermuteEffectiveForShape(inShape, inAlignment)) {
        log.trace("ODU permute is not effective for input shape {0}", inShape);
        return false;
    }
    const ShapeRef outShape = outType.getShape();
    const auto outAlignment = VPU::NCEInvariant::getAlignment(outElemType);
    if (!IE::isODUPermuteEffectiveForShape(outShape, outAlignment)) {
        log.trace("ODU permute is not effective for output shape {0}", outShape);
        return false;
    }

    // For pattern Reorder -> Avgpool(quantize-dequantize or just quantize) -> Eltwise
    // if the Avgpool is quantized, the reorder can't be propagated through eltwise
    // after converting to PermuteQuantize because of the quantized type
    // Don't convert to PermuteQuantize to enable the propagation and fusion
    // Reorder -> Avgpool -> Eltwise will eventually become
    // PermuteCast -> Avgpool -> Eltwise -> PermuteCast with propagation and fusion
    // But if converted to PermuteQuantize
    // PermuteQuantize (f16 to f16) -> Avgpool (f16 to quant)
    // cannot be transformed into
    // Avgpool (f16 to quant) -> PermuteQuantize (quant to quant)
    // because PermuteQuantize expects f16 input
    auto hasQuantizedAvgPoolUserToPropagate = [](IE::ReorderOp reorder) -> bool {
        if (!reorder->hasOneUse()) {
            return false;
        }
        // condition 1, quantize avgpool user
        auto avgpoolOp = mlir::dyn_cast_or_null<IE::AvgPoolOp>(*reorder.getOutput().getUsers().begin());
        if (avgpoolOp == nullptr || !avgpoolOp->hasOneUse() || !isQuantizedPurposeAvgPool(avgpoolOp)) {
            return false;
        }
        // condition 2, optional dequantize avgpool and eltwise after the avgpool(s)
        if (auto nextOp = mlir::dyn_cast_or_null<IE::AvgPoolOp>(*avgpoolOp.getOutput().getUsers().begin())) {
            if (!avgpoolOp->hasOneUse() || !isQuantizedPurposeAvgPool(avgpoolOp)) {
                return false;
            }
            avgpoolOp = nextOp;
        }
        return (*avgpoolOp.getOutput().getUsers().begin())->hasTrait<IE::EltwiseOp>();
    };
    if (hasQuantizedAvgPoolUserToPropagate(reorder)) {
        log.trace("PermuteQuantize can not be propagated through avgpool");
        return false;
    }

    return true;
}

void ConvertReorderToPermuteQuantizePass::safeRunOnFunc() {
    auto func = getOperation();

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
// createConvertReorderToPermuteQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertReorderToPermuteQuantizePass(Logger log) {
    return std::make_unique<ConvertReorderToPermuteQuantizePass>(log);
}
