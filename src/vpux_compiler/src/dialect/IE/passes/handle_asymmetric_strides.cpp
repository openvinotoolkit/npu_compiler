//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/loop.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <tuple>

using namespace vpux;

namespace {

struct OperationPart {
    mlir::ArrayAttr strides;
    mlir::ArrayAttr padBegin;
    mlir::ArrayAttr padEnd;
};

mlir::Operation* createFQ(mlir::PatternRewriter& rewriter, mlir::Operation* inputOp, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.output().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(inputOp->getResult(0)));
    return rewriter.create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, inputOp->getResult(0), fq.input_low(),
                                               fq.input_high(), fq.output_low(), fq.output_high(), fq.levels(),
                                               fq.auto_broadcast());
}

mlir::LogicalResult generalSplitter(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                    mlir::ArrayAttr stridesAttr, ArrayRef<int64_t> kernelSize, mlir::ArrayAttr padBegin,
                                    mlir::ArrayAttr padEnd,
                                    FuncRef<mlir::Operation*(mlir::Location, mlir::Value, OperationPart)> makeOperation,
                                    Logger log) {
    mlir::MLIRContext* ctx = origOp->getContext();

    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr);

    const auto KX = kernelSize[0];
    const auto KY = kernelSize[1];

    const auto inputShape = getShape(origOp->getOperand(0));
    const auto H = inputShape[Dims4D::Act::H];
    const auto W = inputShape[Dims4D::Act::W];

    const auto minStride = std::min(strides[0], strides[1]);
    const auto maxStride = std::max(strides[0], strides[1]);

    const auto newStrides = getIntArrayAttr(ctx, makeArrayRef({maxStride, maxStride}));

    log.trace("New strides for {0} are {1}", origOp->getLoc(), newStrides);

    if (W == KX || H == KY) {
        auto* newOp = makeOperation(origOp->getLoc(), origOp->getOperand(0), {newStrides, padBegin, padEnd});
        rewriter.replaceOp(origOp, newOp->getResult(0));

        return mlir::success();
    }

    const auto inputFQ = origOp->getOperand(0).getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp->getResult(0).user_begin()));

    int64_t steps_h = (strides[1] + strides[0] - 1) / strides[0];
    int64_t steps_w = (strides[0] + strides[1] - 1) / strides[1];

    auto paddingEnd = parseIntArrayAttr<int64_t>(padEnd);

    mlir::SmallVector<mlir::Value> hSliced;
    for (auto i : irange(steps_w)) {
        mlir::SmallVector<mlir::Value> wSliced;
        for (auto j : irange(steps_h)) {
            Shape offsets(inputShape.size());

            offsets[Dims4D::Act::H] = j * minStride;
            offsets[Dims4D::Act::W] = i * minStride;

            SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                            inputShape[Dims4D::Act::H] - offsets[Dims4D::Act::H],
                                            inputShape[Dims4D::Act::W] - offsets[Dims4D::Act::W]};

            const auto loc = appendLoc(origOp->getLoc(), "slice {0}, {1}", i, j);

            mlir::Operation* slicedInput = rewriter.create<IE::SliceOp>(
                    loc, origOp->getOperand(0), getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            // TODO: temporary FQ propagation
            if (inputFQ != nullptr && outputFQ != nullptr && (i != 0 || j != 0)) {
                slicedInput->setOperand(0, inputFQ->getOperand(0));
                slicedInput = createFQ(rewriter, slicedInput, inputFQ);
            }

            auto newPaddingEnd = paddingEnd;

            newPaddingEnd[0] += offsets[Dims4D::Act::H];
            newPaddingEnd[1] += offsets[Dims4D::Act::W];

            VPUX_THROW_WHEN(newPaddingEnd[0] > (KY + 1) / 2 || newPaddingEnd[1] > (KX + 1) / 2,
                            "Paddings are out of range");

            auto* newOp = makeOperation(loc, slicedInput->getResult(0),
                                        {newStrides, padBegin, getIntArrayAttr(ctx, newPaddingEnd)});

            // TODO: temporary FQ propagation
            if (inputFQ != nullptr && outputFQ != nullptr) {
                newOp = createFQ(rewriter, newOp, outputFQ);
            }

            wSliced.push_back(newOp->getResult(0));
        }

        if (!wSliced.empty()) {
            hSliced.push_back(wSliced.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), wSliced,
                                                                                  Dims4D::Act::H, minStride, maxStride)
                                                  : wSliced.front());
        }
    }

    if (hSliced.empty()) {
        return mlir::failure();
    }

    const auto concatOp = hSliced.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), hSliced, Dims4D::Act::W,
                                                                              minStride, maxStride)
                                              : hSliced.front();

    rewriter.replaceOp(origOp, concatOp);

    return mlir::success();
}

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto filterShape = getShape(origOp.filter());

    return generalSplitter(
            origOp, rewriter, origOp.strides(), {filterShape[Dims4D::Act::W], filterShape[Dims4D::Act::H]},
            origOp.pads_begin(), origOp.pads_end(),
            [&](mlir::Location loc, mlir::Value input, OperationPart part) -> mlir::Operation* {
                return rewriter.create<IE::ConvolutionOp>(loc, input, origOp.filter(), origOp.bias(), part.strides,
                                                          part.padBegin, part.padEnd, origOp.dilations(),
                                                          origOp.post_opAttr());
            },
            _log.nest());
}

//
// HandleAsymmetricStridesPass
//

class HandleAsymmetricStridesPass final : public IE::HandleAsymmetricStridesBase<HandleAsymmetricStridesPass> {
public:
    explicit HandleAsymmetricStridesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleAsymmetricStridesPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto checkStridesRelation = [](const int64_t strideRight, const int64_t strideLeft,
                                         const int64_t inputSize) -> bool {
        return strideRight > strideLeft && strideLeft > 0 && strideRight % strideLeft == 0 &&
               inputSize % (strideRight / strideLeft) == 0;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(op.strides());
        const auto SY = kernelStrides[0];
        const auto SX = kernelStrides[1];

        const auto inputShape = getShape(op.input());
        const auto H = inputShape[Dims4D::Act::H];
        const auto W = inputShape[Dims4D::Act::W];

        // for now there is implementation for the case
        // this way produces less slices and operations
        // in case of more generic option, new implementation needed.
        return SY == SX || !(checkStridesRelation(SX, SY, W) || checkStridesRelation(SY, SX, H));
    });

    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createHandleAsymmetricStridesPass(Logger log) {
    return std::make_unique<HandleAsymmetricStridesPass>(log);
}
