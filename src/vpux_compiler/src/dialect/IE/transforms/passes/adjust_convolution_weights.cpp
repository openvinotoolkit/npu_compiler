//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTCONVOLUTIONWEIGHTS
#define GEN_PASS_DEF_ADJUSTCONVOLUTIONWEIGHTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// AdjustConvWeights
//

class AdjustConvWeights final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    AdjustConvWeights(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value createNewWeights(IE::ConvolutionOp convOp, mlir::Value weights, int64_t realInputChannel,
                             IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) {
    auto weightShape = getShape(weights);
    SmallVector<int64_t> realWeightsSizes(weightShape.begin(), weightShape.end());
    realWeightsSizes[Dims4D::Filter::IC.ind()] = realInputChannel;

    auto realWeightsOffsets = SmallVector<int64_t>(weightShape.size(), 0);

    auto realWeights = rewriter.createOrFold<IE::SliceOp>(convOp->getLoc(), weights,
                                                          getIntArrayAttr(convOp.getContext(), realWeightsOffsets),
                                                          getIntArrayAttr(convOp.getContext(), realWeightsSizes));

    SmallVector<int64_t> realChannelOfConcatInputs, junkChannelOfConcatInputs;

    for (auto input : concatOp.getInputs()) {
        auto sliceOp = input.getDefiningOp<IE::SliceOp>();

        if (!sliceOp) {
            auto inputShape = getShape(input);
            realChannelOfConcatInputs.push_back(inputShape[Dims4D::Act::C]);
            junkChannelOfConcatInputs.push_back(0);
        } else {
            auto sliceInputShape = getShape(sliceOp.getInputs().front());
            auto sliceOutputShape = getShape(input);
            realChannelOfConcatInputs.push_back(sliceOutputShape[Dims4D::Act::C]);
            junkChannelOfConcatInputs.push_back(sliceInputShape[Dims4D::Act::C] - sliceOutputShape[Dims4D::Act::C]);
        }
    }

    SmallVector<mlir::Value> newConcatInputs;

    auto subWeightsOffsets = SmallVector<int64_t>(weightShape.size(), 0);
    SmallVector<int64_t> subWeightsSizes(weightShape.begin(), weightShape.end());
    auto padBegin = mlir::SmallVector<int64_t>(weightShape.size(), 0);
    auto padEnd = mlir::SmallVector<int64_t>(weightShape.size(), 0);

    for (size_t i = 0; i < realChannelOfConcatInputs.size(); ++i) {
        realWeightsSizes[Dims4D::Filter::IC.ind()] = realChannelOfConcatInputs[i];

        auto subWeightsSlice = rewriter.createOrFold<IE::SliceOp>(
                convOp->getLoc(), realWeights, getIntArrayAttr(convOp.getContext(), subWeightsOffsets),
                getIntArrayAttr(convOp.getContext(), realWeightsSizes));
        subWeightsOffsets[Dims4D::Filter::IC.ind()] += realChannelOfConcatInputs[i];

        padEnd[Dims4D::Filter::IC.ind()] = junkChannelOfConcatInputs[i];
        auto subWeightsExpand = rewriter.createOrFold<IE::ExpandOp>(convOp->getLoc(), subWeightsSlice,
                                                                    getIntArrayAttr(rewriter, ArrayRef(padBegin)),
                                                                    getIntArrayAttr(rewriter, ArrayRef(padEnd)));

        newConcatInputs.push_back(subWeightsExpand);
    }

    return rewriter.createOrFold<IE::ConcatOp>(convOp->getLoc(), newConcatInputs, getConcatAxis(concatOp).value());
}

/*
    This pass handle this case:
          conv1             conv2
           |                 |
    output 1x16x40x40   output 1x16x40x40
           |                 |
     slice 1x12x40x40      /
           |            /
            \        /
             \    /
        Concat 1x28x40x40
              |
        Expand 1x32x40x40
              |
            Conv3
    This pass will adjust the weights of Conv3 and convert this structure to
          conv1             conv2
           |                 |
    output 1x16x40x40   output 1x16x40x40
           |                 |
             \             /
                \       /
                  \  /
            Concat 1x32x40x40
                    |
                  Conv3

The origin input channel of Conv3's weights will be:

 n n n n n n n n n n n n n n n n n n n n n n n n n n n n 0 0 0 0
 | <----------------------28---------------------------> |

And it will be convert to:

 n n n n n n n n n n n n 0 0 0 0 n n n n n n n n n n n n n n n n
 | <--------12-------> |        | <-------------16-----------> |
*/

mlir::LogicalResult AdjustConvWeights::matchAndRewrite(IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(expandOp.getInput().getType());
    const auto inputElemType = inputType.getElementType();
    if (mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
        innerLog.trace("'Expand' at '{0}' had quantized type, we can't handle it.", expandOp->getLoc());
        return mlir::failure();
    }

    auto concatOp = expandOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || concatOp->getNumResults() != 1 || !concatOp->hasOneUse()) {
        innerLog.trace("'Expand' at '{0}' input is not 'Concat' or 'Concat' has more than one users",
                       expandOp->getLoc());
        return mlir::failure();
    }

    auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(*(expandOp.getOutput().user_begin()));

    if (!convOp || !expandOp->hasOneUse()) {
        innerLog.trace("'Expand' at '{0}' output is not 'Convolution' or 'Expand' has more than one users",
                       expandOp->getLoc());
        return mlir::failure();
    }

    auto filterCst = convOp.getFilter().getDefiningOp<Const::DeclareOp>();
    if (filterCst == nullptr) {
        innerLog.trace("'Convolution' at '{0}' don't have const filter", convOp->getLoc());
        return mlir::failure();
    }

    SmallVector<mlir::Value> concatInputs;
    bool hasSlice = false;
    for (auto input : concatOp.getInputs()) {
        auto sliceOp = input.getDefiningOp<IE::SliceOp>();

        if (!sliceOp) {
            concatInputs.push_back(input);
        } else {
            auto sliceInShape = getShape(sliceOp.getSource());
            auto sliceOutShape = getShape(sliceOp.getResult());
            auto sliceAxis = vpux::IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
            if (!sliceAxis.has_value()) {
                return mlir::failure();
            }
            // This optimization is achieved by adjusting the weights channel of the convolution to avoid certain DMA
            // operations. It can only handle cases where slice on C.
            if (sliceAxis.value() != Dims4D::Act::C) {
                innerLog.trace("'Slice' at '{0}' don't slice on C", sliceOp);
                return mlir::failure();
            }

            hasSlice = true;
            concatInputs.push_back(sliceOp.getInputs().front());
        }
    }

    if (!hasSlice) {
        innerLog.trace("All inputs of 'Concat' at '{0}' output is not 'Slice'.", concatOp->getLoc());
        return mlir::failure();
    }
    auto totalInput = 0;

    for (auto concatInput : concatInputs) {
        auto inputShape = getShape(concatInput);
        totalInput += inputShape[Dims4D::Act::C];
    }

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedChannel = iface.getInputChannelAlignment();
    if (totalInput % alignedChannel != 0) {
        innerLog.trace("Total input channels {0} is not align to {1}.", totalInput, alignedChannel);
        return mlir::failure();
    }

    auto convInput = getShape(expandOp.getOutput())[Dims4D::Act::C];
    if (convInput > totalInput) {
        return mlir::failure();
    }

    // Calculate expand shape
    const auto concatAxis = getConcatAxis(concatOp);
    const auto expandAxis = getExpandAxis(expandOp);
    if (!concatAxis.has_value() || !expandAxis.has_value()) {
        return mlir::failure();
    }

    const auto concatAxisVal = concatAxis.value();
    const auto expandAxisVal = expandAxis.value();

    if (concatAxisVal != expandAxisVal || expandAxisVal != Dims4D::Act::C) {
        innerLog.trace("'Concat' axis '{0}' not same with 'Expand' axis '{1}', or not on channel.", concatAxisVal,
                       expandAxisVal);
        return mlir::failure();
    }

    auto newConcatOp = rewriter.create<IE::ConcatOp>(concatOp->getLoc(), concatInputs, concatAxisVal);

    auto paddedFilter = createNewWeights(convOp, convOp.getFilter(), getShape(expandOp.getInput())[Dims4D::Act::C],
                                         concatOp, rewriter);

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            convOp, convOp.getOutput().getType(), newConcatOp.getOutput(), paddedFilter, convOp.getBias(),
            convOp.getStridesAttr(), convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(), convOp.getDilationsAttr(),
            convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());

    return mlir::success();
}

//
// AdjustConvolutionWeightsPass
//

class AdjustConvolutionWeightsPass final : public IE::impl::AdjustConvolutionWeightsBase<AdjustConvolutionWeightsPass> {
public:
    explicit AdjustConvolutionWeightsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionWeightsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<AdjustConvWeights>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAdjustConvolutionWeightsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionWeightsPass(Logger log) {
    return std::make_unique<AdjustConvolutionWeightsPass>(log);
}
