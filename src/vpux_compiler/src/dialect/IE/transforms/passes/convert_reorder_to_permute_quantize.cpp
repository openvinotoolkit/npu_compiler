//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

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

    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    auto curInput = origOp.getInput();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto origMemPerm = vpux::getPermutationFromOrders(inOrder, outOrder, origOp->getContext());
    if (IE::canConvertToNCHWInOrderWithPermuteCast(inType, origMemPerm) && outOrder == DimsOrder::NHWC) {
        // There is a chance to convert reorderOp to permuteQuantizeOp after inserting a permuteCastOp for input
        const auto inMemPerm = vpux::getPermutationFromOrders(inOrder, DimsOrder::NCHW, origOp->getContext());
        auto inPermuteCastOp =
                rewriter.create<IE::PermuteCastOp>(appendLoc(origOp->getLoc(), "PermuteCast"), origOp.getInput(),
                                                   DimsOrder::NCHW.toAffineMap(origOp->getContext()), inMemPerm);
        curInput = inPermuteCastOp.getResult();
        inOrder = DimsOrder::NCHW;
    }

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    SmallVector<int64_t> noPadBeginEnd(inOrder.numDims(), 0);
    const auto& ctx = origOp.getContext();
    const auto permQuantOutType = origOp.getOutput().getType();
    const auto permQuantElemType = mlir::cast<vpux::NDTypeInterface>(permQuantOutType).getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = appendLoc(origOp->getLoc(), "PermuteQuantize");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, curInput, origOp.getDstOrderAttr(), memPermAttr, dstElemTypeAttr,
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
    explicit ConvertReorderToPermuteQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool isSupportedReorder(IE::ReorderOp reorder, Logger log) const;
};

bool hasQuantizedAvgPoolUserToPropagate(IE::ReorderOp reorder) {
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
        avgpoolOp = nextOp;
    }
    return (*avgpoolOp.getOutput().getUsers().begin())->hasTrait<IE::EltwiseOp>();
}

bool ConvertReorderToPermuteQuantizePass::isSupportedReorder(IE::ReorderOp reorder, Logger log) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(reorder.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(reorder.getOutput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto origMemPerm = vpux::getPermutationFromOrders(inOrder, outOrder, reorder->getContext());
    if (IE::canConvertToNCHWInOrderWithPermuteCast(inType, origMemPerm) && outOrder == DimsOrder::NHWC) {
        // There is a chance to convert reorderOp to permuteQuantizeOp after inserting a permuteCastOp for input
        inType = inType.changeDimsOrder(DimsOrder::NCHW);
    }

    if (!IE::isLegalReorderLikeToPermuteQuantize(inType, outType, log)) {
        log.trace("Can not convert to PermuteQuantize");
        return false;
    }
    // Add quant type
    // TODO: If consumer is convolution op we can support quantized input.
    // E#-183528 is to address some regressions caused by pass MovePermutePostEltwise after dropping the WA.
    auto consumer = *reorder.getResult().getUsers().begin();
    auto inElemType = inType.getElementType();
    if (mlir::isa<mlir::quant::QuantizedType>(inElemType) && !mlir::isa<IE::ConvolutionOp>(consumer)) {
        _log.trace("Cannot convert MemPermute with quantized input to PermuteQuantize when consumer is not a conv");
        return false;
    }
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
    target.addLegalOp<IE::PermuteCastOp>();

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
