//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_HANDLELARGEPADS
#define GEN_PASS_DEF_HANDLELARGEPADS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// The function will move the padding from layer parameter to its input
// for example, top pad is 5, but what HW can support is 2, we will update the layer's
// top pad to 2, and concat the input with 3 lines zero constant.
std::tuple<mlir::Value, Shape, Shape> getInputConcatAndPadding(mlir::PatternRewriter& rewriter, mlir::Operation* origOp,
                                                               ShapeRef padStart, ShapeRef padEnd,
                                                               ArrayRef<int64_t> kernelShape) {
    const auto KY = kernelShape[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelShape[Dims4D::Kernel::X.ind()];
    const auto padTop = padStart[Dims4D::PadsBegin::Top];
    const auto padLeft = padStart[Dims4D::PadsBegin::Left];
    const auto padBottom = padEnd[Dims4D::PadsEnd::Bottom];
    const auto padRight = padEnd[Dims4D::PadsEnd::Right];
    auto newPadStart = Shape(padStart.raw());
    auto newPadEnd = Shape(padEnd.raw());
    auto input = origOp->getOperand(0);
    const auto inputShape = getShape(input);
    Shape zeroConstShape(inputShape.size());
    auto inputConcat = input;

    zeroConstShape[Dims4D::Act::N] = inputShape[Dims4D::Act::N];
    zeroConstShape[Dims4D::Act::C] = inputShape[Dims4D::Act::C];
    SmallVector<mlir::Value> concats;

    auto createConst = [&](int64_t pad, int64_t kernel, bool isDimX) {
        zeroConstShape[Dims4D::Act::W] = isDimX ? (pad - kernel / 2) : getShape(inputConcat)[Dims4D::Act::W];
        zeroConstShape[Dims4D::Act::H] = isDimX ? getShape(inputConcat)[Dims4D::Act::H] : (pad - kernel / 2);
        const auto zeroType = mlir::RankedTensorType::get(
                zeroConstShape.raw(), mlir::cast<NDTypeInterface>(input.getType()).getElementType());
        auto zeroConst = Const::createZerosConst(rewriter, takeOpLoc(origOp, "_pad_zero"), zeroType);
        const auto dataOrder = mlir::cast<NDTypeInterface>(input.getType()).getDimsOrder();
        const auto orderMap = dataOrder.toAffineMap(rewriter.getContext());
        return rewriter.createOrFold<IE::ReorderOp>(takeOpLoc(origOp, "_reorder"), zeroConst, orderMap);
    };

    if (padLeft > KX / 2) {
        concats.push_back(createConst(padLeft, KX, true));
        newPadStart[Dims4D::PadsBegin::Left] = KX / 2;
    }
    concats.push_back(inputConcat);
    if (padRight > KX / 2) {
        concats.push_back(createConst(padRight, KX, true));
        newPadEnd[Dims4D::PadsEnd::Right] = KX / 2;
    }

    inputConcat = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "_input_concat0"), mlir::ValueRange(concats),
                                                Dims4D::Act::W)
                          ->getResult(0);

    concats.clear();

    if (padTop > KY / 2) {
        concats.push_back(createConst(padTop, KY, false));
        newPadStart[Dims4D::PadsBegin::Top] = KY / 2;
    }
    concats.push_back(inputConcat);
    if (padBottom > KY / 2) {
        concats.push_back(createConst(padBottom, KY, false));
        newPadEnd[Dims4D::PadsEnd::Bottom] = KY / 2;
    }

    inputConcat = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "_input_concat1"), mlir::ValueRange(concats),
                                                Dims4D::Act::H)
                          ->getResult(0);

    return std::make_tuple(inputConcat, newPadStart, newPadEnd);
}

mlir::LogicalResult createNewOpAndReplaceOldOne(mlir::PatternRewriter& rewriter, mlir::Operation* origOp,
                                                mlir::Value input, ShapeRef padStart, ShapeRef padEnd) {
    mlir::IRMapping mapper;
    mlir::Builder builder(origOp->getContext());

    SmallVector<mlir::Value> operands = origOp->getOperands();
    operands[0] = input;
    mapper.map(origOp->getOperands(), operands);
    auto* newOp = rewriter.clone(*origOp, mapper);

    if (newOp->hasAttr("pads_begin") && newOp->hasAttr("pads_end")) {
        auto padBeginAttr =
                builder.getI64ArrayAttr({padStart[Dims4D::PadsBegin::Top], padStart[Dims4D::PadsBegin::Left]});
        auto padEndAttr = builder.getI64ArrayAttr({padEnd[Dims4D::PadsEnd::Bottom], padEnd[Dims4D::PadsEnd::Right]});
        newOp->setAttr("pads_begin", padBeginAttr);
        newOp->setAttr("pads_end", padEndAttr);
    } else {
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, newOp->getResult(0));
    return mlir::success();
}

//
// ConvGeneralRewriter
//
template <class ConcreteOp>
class ConvGeneralRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvGeneralRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvGeneralRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Handle larger padding for '{0} 'layer at '{1}'", origOp->getName(), origOp->getLoc());

    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto kernelShape = getShape(origOp.getFilter());

    mlir::Value inputConcat;
    Shape newPadStart, newPadEnd;
    std::tie(inputConcat, newPadStart, newPadEnd) = getInputConcatAndPadding(
            rewriter, origOp, padStart, padEnd, {kernelShape[Dims4D::Filter::KY], kernelShape[Dims4D::Filter::KX]});

    return createNewOpAndReplaceOldOne(rewriter, origOp, inputConcat, newPadStart, newPadEnd);
}

//
// PoolingGeneralRewriter
//
template <class ConcreteOp>
class PoolingGeneralRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    PoolingGeneralRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult PoolingGeneralRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Handle larger padding for '{0} 'layer at '{1}'", origOp->getName(), origOp->getLoc());

    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto kernelShape = parseIntArrayAttr<int64_t>(origOp.getKernelSize());

    mlir::Value inputConcat;
    Shape newPadStart, newPadEnd;
    std::tie(inputConcat, newPadStart, newPadEnd) =
            getInputConcatAndPadding(rewriter, origOp, padStart, padEnd, kernelShape);

    return createNewOpAndReplaceOldOne(rewriter, origOp, inputConcat, newPadStart, newPadEnd);
}

//
// HandleLargePadsPass
//

class HandleLargePadsPass final : public IE::impl::HandleLargePadsBase<HandleLargePadsPass> {
public:
    explicit HandleLargePadsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isPaddingSupported(ShapeRef padStart, ShapeRef padEnd, ArrayRef<int64_t> kernelShape) {
    const auto KY = kernelShape[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelShape[Dims4D::Kernel::X.ind()];
    const auto padTop = padStart[Dims4D::PadsBegin::Top];
    const auto padLeft = padStart[Dims4D::PadsBegin::Left];
    const auto padBottom = padEnd[Dims4D::PadsEnd::Bottom];
    const auto padRight = padEnd[Dims4D::PadsEnd::Right];

    if (padTop > KY / 2 || padBottom > KY / 2 || padLeft > KX / 2 || padRight > KX / 2) {
        return false;
    }

    return true;
}

template <class ConcreteOp>
bool isLegalConv(ConcreteOp op) {
    auto padStart = Shape(parseIntArrayAttr<int64_t>(op.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(op.getPadsEnd()));
    auto kernelShape = getShape(op.getFilter());
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    // Required for SEP Dilated Group Convolution case , for all other case it does not change padding
    padStart[Dims4D::PadsBegin::Top] -= dilations[Dims4D::Dilation::Y.ind()] - 1;
    padStart[Dims4D::PadsBegin::Left] -= dilations[Dims4D::Dilation::X.ind()] - 1;
    padEnd[Dims4D::PadsEnd::Bottom] -= dilations[Dims4D::Dilation::Y.ind()] - 1;
    padEnd[Dims4D::PadsEnd::Right] -= dilations[Dims4D::Dilation::X.ind()] - 1;
    return isPaddingSupported(padStart, padEnd, {kernelShape[Dims4D::Filter::KY], kernelShape[Dims4D::Filter::KX]});
}

template <class ConcreteOp>
bool isLegalPooling(ConcreteOp op) {
    auto padStart = Shape(parseIntArrayAttr<int64_t>(op.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(op.getPadsEnd()));
    auto kernelShape = parseIntArrayAttr<int64_t>(op.getKernelSize());

    return isPaddingSupported(padStart, padEnd, kernelShape);
}

void HandleLargePadsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(&isLegalConv<IE::GroupConvolutionOp>);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(&isLegalConv<IE::ConvolutionOp>);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(&isLegalPooling<IE::MaxPoolOp>);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(&isLegalPooling<IE::AvgPoolOp>);

    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ConcatOp>();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvGeneralRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralRewriter<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<PoolingGeneralRewriter<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<PoolingGeneralRewriter<IE::AvgPoolOp>>(&ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleLargePadsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleLargePadsPass(Logger log) {
    return std::make_unique<HandleLargePadsPass>(log);
}
