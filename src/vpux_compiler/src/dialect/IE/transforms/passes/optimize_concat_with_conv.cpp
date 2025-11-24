//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Support/LLVM.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZECONCATWITHCONV
#define GEN_PASS_DEF_OPTIMIZECONCATWITHCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isElementTypeSupported(mlir::Type elementType) {
    // Conv in NCE only works on some float and quantized types.
    return mlir::isa<mlir::FloatType>(elementType) || mlir::quant::QuantizedType::castToStorageType(elementType);
}

//
// OptimizeConcat
//

class OptimizeConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    OptimizeConcat(mlir::MLIRContext* ctx, int64_t dpuCount, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _dpuCount(dpuCount), _log(log) {
        this->setDebugName("OptimizeConcat");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _dpuCount;
    Logger _log;
};

/*
For subgraph:

        Input0                Input1
    [1, HWC, 1, 1]#NCHW    [1, HWC, 1, 1]#NCHW
            \              /
                 Concat
             [1, H*W*C, 2, 1]#NCHW

Converts to:

        Input0                Input1
    [1, HWC, 1, 1]#NCHW    [1, HWC, 1, 1]#NCHW
          |                      |
        Reshape               Reshape
    [1, C, H, W]#NCHW      [1, C, H, W]#NCHW
          |                     |
       LayoutCast            LayoutCast
    [1, C, H, W]#NHWC    [1, C, H, W]#NHWC
             \                 /
                    Concat
               [1, C, 2H, W]#NHWC
                     |
                    Conv with Kernel[2C, C, H+1, 1]
               [1, 2C, H, W]#NHWC
                     |
                 LayoutCast
               [1, 2C, H, W] #NCHW
                     |
                  Reshape
               [1, H*W*C, 2, 1] #NCHW

*/

mlir::LogicalResult OptimizeConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto elementType = origOp.getOutput().getType().getElementType();
    if (!isElementTypeSupported(elementType)) {
        return matchFailed(_log, rewriter, origOp, "Can not apply optimization with element type {0} at '{1}'",
                           elementType, origOp->getLoc());
    }

    auto dimOrder = DimsOrder::fromValue(origOp);
    if (dimOrder != DimsOrder::NCHW) {
        return matchFailed(_log, rewriter, origOp, "Can not apply optimization with layout {0} at '{1}'", dimOrder,
                           origOp->getLoc());
    }
    auto concatAxis = IE::getConcatAxis(origOp);
    if (!concatAxis.has_value() || concatAxis.value().ind() != Dims4D::Act::H.ind()) {
        return matchFailed(_log, rewriter, origOp,
                           "Can not apply optimization with concat axis other than DimH at '{0}'", origOp->getLoc());
    }

    auto concatInputs = origOp.getInputs();
    if (concatInputs.size() != 2) {
        return matchFailed(_log, rewriter, origOp, "Can not apply optimization with concat inputs num {0} at '{1}'",
                           concatInputs.size(), origOp->getLoc());
    }

    auto concatInShape = getShape(concatInputs.front());
    if (concatInShape[Dims4D::Act::N] != 1 || concatInShape[Dims4D::Act::H] != 1 ||
        concatInShape[Dims4D::Act::W] != 1 || concatInShape[Dims4D::Act::C] == 1) {
        return matchFailed(_log, rewriter, origOp, "Cannot apply optimization with concat input shape {0} at '{1}'",
                           concatInShape, origOp->getLoc());
    }

    const auto channelAlignment = VPU::NCEInvariant::getAlignment(origOp.getType());
    const auto H = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    // Find suitable C which meet SOK requirement, here we limit the search range from [_dpuCount,  _dpuCount*2) to
    // avoid introducing too much workloads
    std::optional<int64_t> suitableC = std::nullopt;
    for (int64_t tileCount = _dpuCount; tileCount < _dpuCount * 2; tileCount++) {
        if (concatInShape.totalSize() % (H * channelAlignment * tileCount) == 0) {
            suitableC = channelAlignment * tileCount;
            break;
        }
    }
    if (!suitableC.has_value()) {
        return matchFailed(
                _log, rewriter, origOp,
                "Can not find suitable DimC size with concat input shape {0} at '{1}', the conversion is skipped",
                concatInShape, origOp->getLoc());
    }
    const auto C = suitableC.value();

    auto areInputsHaveSameShape = llvm::all_of(concatInputs, [&](auto input) {
        return getShape(input) == concatInShape;
    });

    if (!areInputsHaveSameShape) {
        return matchFailed(_log, rewriter, origOp, "concat inputs have different shapes at '{0}'", origOp->getLoc());
    }

    const auto ctx = rewriter.getContext();
    _log.trace("process concat op at {0}", origOp->getLoc());

    // create new concat on outer most dimension
    SmallVector<mlir::Value> inputs;
    Shape newConcatInShape = {1, C, H, concatInShape.totalSize() / (H * C)};
    for (auto input : concatInputs) {
        auto newInReshape = rewriter.create<IE::ReshapeOp>(appendLoc(input.getLoc(), "_reshape"), input, nullptr, false,
                                                           getIntArrayAttr(ctx, newConcatInShape.raw()));
        auto layoutCastOp = rewriter.create<IE::LayoutCastOp>(appendLoc(newInReshape.getLoc(), "_layout_cast"),
                                                              newInReshape, DimsOrder::NHWC.toAffineMap(ctx));
        inputs.push_back(layoutCastOp);
    }
    auto newConcat =
            rewriter.create<IE::ConcatOp>(appendLoc(origOp.getLoc(), "_input_concat"), inputs, concatAxis.value());

    // create conv to do the permute
    SmallVector<int64_t> padBegin(2, 0);
    SmallVector<int64_t> padEnd(2, 0);
    SmallVector<int64_t> strides(2, 1);
    SmallVector<int64_t> dilations(2, 1);

    /*
     create weights with shape[2C, C, H+1, 1] for the new concat conv
     The weights values are filled as follows:
                     0                 1          ...        C-1
     Filter0: [1, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
     Filter1: [0, 0,..., 0, 1], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
     Filter2: [0, 0,..., 0, 0], [1, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
     Filter3: [0, 0,..., 0, 0], [0, 0,..., 0, 1], ...,  [0, 0, ..., 0, 0]
     ...
     Filter(2C-2): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [1, 0, ..., 0, 0]
     Filter(2C-1): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 1]
    */
    const auto perFilterSize = H + 1;
    const auto perOutChannelFilterSize = C * perFilterSize;
    const auto OC = 2 * C;
    std::vector<vpux::type::float16> weightsVals(OC * perOutChannelFilterSize, checked_cast<vpux::type::float16>(0.f));
    for (int64_t i = 0; i < C; i++) {
        // for 2i-th filter, set the i-th array first element to 1
        weightsVals[2 * i * perOutChannelFilterSize + i * perFilterSize] = checked_cast<vpux::type::float16>(1.f);
        // for 2i+1-th filter, set the i-th array last element to 1
        weightsVals[(2 * i + 1) * perOutChannelFilterSize + i * perFilterSize + H] =
                checked_cast<vpux::type::float16>(1.f);
    }

    Shape weightsShape = {2 * C, C, H + 1, 1};
    const auto weightStorageType = mlir::RankedTensorType::get(weightsShape.raw(), mlir::Float16Type::get(ctx));
    const auto weightStorageAttr = Const::createConstContent(weightStorageType, ArrayRef(weightsVals));
    const auto weightContentAttr = Const::ContentAttr::get(weightStorageAttr);
    const auto declLoc = appendLoc(origOp.getLoc(), "weights_for_concat");

    const auto weightExpressedElemType =
            IE::composeWeightsExpressedType(newConcat.getResult().getType().getElementType());
    const auto weightExpressedType = mlir::RankedTensorType::get(weightsShape.raw(), weightExpressedElemType);
    auto targetContentAttr = weightContentAttr.transform().castElemType(weightExpressedElemType).get();
    auto weightsConst = rewriter.create<Const::DeclareOp>(declLoc, weightExpressedType, std::move(targetContentAttr));
    const auto reorderLoc = appendLoc(weightsConst.getLoc(), "reorder_weights_for_DPU_concat");
    const auto weightTypeNCHW = mlir::cast<vpux::NDTypeInterface>(weightsConst.getOutput().getType());
    const auto reorderType = weightTypeNCHW.changeDimsOrder(DimsOrder::NHWC);
    const auto orderMap = DimsOrder::NHWC.toAffineMap(ctx);
    auto weightsReorder =
            rewriter.createOrFold<IE::ReorderOp>(reorderLoc, reorderType, weightsConst.getOutput(), orderMap);

    auto newConv = rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), newConcat, weightsReorder,
                                                      /*bias=*/nullptr, getIntArrayAttr(ctx, strides),
                                                      getIntArrayAttr(ctx, padBegin), getIntArrayAttr(ctx, padEnd),
                                                      getIntArrayAttr(ctx, dilations),
                                                      /*postOp=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                                      /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
    _log.trace("create new conv {0}", newConv);
    auto newOutLayoutCast = rewriter.create<IE::LayoutCastOp>(appendLoc(newConv.getLoc(), "_layout_cast_"),
                                                              newConv.getOutput(), DimsOrder::NCHW.toAffineMap(ctx));

    auto concatOutShape = getShape(origOp);
    auto newOutReshape =
            rewriter.create<IE::ReshapeOp>(appendLoc(newOutLayoutCast.getLoc(), "_reshape_"), newOutLayoutCast, nullptr,
                                           false, getIntArrayAttr(ctx, concatOutShape.raw()));
    rewriter.replaceAllUsesWith(origOp, newOutReshape);
    return mlir::success();
}

//
// OptimizeConcatWithConvAndAdd
//

class OptimizeConcatWithConvAndAdd final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    OptimizeConcatWithConvAndAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        this->setDebugName("OptimizeConcatWithConvAndAdd");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final;
    bool isBeneficialToConvert(IE::ConcatOp concatOp) const;

private:
    Logger _log;
};

/*
For example:
        Input0                Input1
    [1, C, H, W]#NHWC    [1, C, H, W]#NHWC
            \              /
                 Concat
             [1, 2C, H, W]#NHWC
Converts to:
   Filter1          Input0                Input1         Filter2
[2C, C, 1, 1]    [1, C, H, W]#NHWC    [1, C, H, W]#NHWC [2C, C, 1, 1]
        \              |                      |        /
                Convolution1           Convolution2
                [1, 2C, H, W]#NHWC      [1, 2C, H, W]#NHWC
                       |                     |
                        \                   /
                                Add
                            [1, 2C, H, W]#NHWC
Filter1 : [1], [0], ..., [0]
Filter2 : [0], [1], ..., [0]
...
Filter2C: [0], [0], ..., [1]
*/
mlir::LogicalResult OptimizeConcatWithConvAndAdd::matchAndRewrite(IE::ConcatOp concatOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!isBeneficialToConvert(concatOp)) {
        return matchFailed(_log, rewriter, concatOp, "Not beneficial to convert concat at '{0}'", concatOp->getLoc());
    }

    // Compose weights for the convolution, the new weights output channels are the same as the concat output channels
    // and those channels value are 1. And other channels are 0. And than those convolution ops' outputs can be added
    // together to get the final concat results.
    auto composeWeights = [](int inChannels, int outChannels, int startChannel) {
        SmallVector<float> weightsVals(inChannels * outChannels, checked_cast<vpux::type::float16>(0.f));
        for (int i = 0; i < inChannels; ++i) {
            auto targetChannel = startChannel + i;
            auto weightIndex = targetChannel * inChannels + i;
            weightsVals[weightIndex] = 1.0f;
        }
        return weightsVals;
    };

    auto ctx = rewriter.getContext();

    auto concatInputs = concatOp.getInputs();
    auto concatOutput = concatOp.getOutput();
    auto concatOutShape = getShape(concatOutput);
    auto totalOutChannels = concatOutShape[Dims4D::Act::C];
    auto accumulateChannels = 0;

    SmallVector<mlir::Value> convResults;

    for (size_t i = 0; i < concatInputs.size(); ++i) {
        auto currentInput = concatInputs[i];
        auto currentType = mlir::cast<vpux::NDTypeInterface>(currentInput.getType());
        const int64_t currentInChannels = currentType.getShape()[Dims4D::Act::C];

        auto weightsVals = composeWeights(currentInChannels, totalOutChannels, accumulateChannels);
        accumulateChannels += currentInChannels;

        const Shape weightsShape = {totalOutChannels, currentInChannels, 1, 1};
        const DimsOrder weightOrder = DimsOrder::OYXI;

        VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(currentInput.getType()),
                          "{0} doesn't support dynamic shapes", IE::ConvolutionOp::getOperationName());
        const auto weightType = mlir::RankedTensorType::get(
                weightsShape.raw(), mlir::cast<NDTypeInterface>(currentInput.getType()).getElementType(),
                getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
        auto weight = Const::buildWeightsConst(rewriter, currentInput.getLoc(), weightType, ArrayRef(weightsVals));

        const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

        const auto convLoc = takeOpLoc(concatOp, llvm::formatv("convolution_for_concat_{0}", i).str());
        auto newConv =
                rewriter.create<IE::ConvolutionOp>(convLoc, currentInput, weight,
                                                   /*bias=*/nullptr, strides, kernelPadsBegin, kernelPadsEnd, dilations,
                                                   /*postOp=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                                   /*outputChannels=*/nullptr, /*inputChannels=*/nullptr);

        convResults.push_back(newConv.getOutput());
    }

    IE::AddOp addOp;
    for (size_t i = 1; i < convResults.size(); ++i) {
        const auto declLoc = takeOpLoc(concatOp, llvm::formatv("add_for_concat_{0}", i).str());
        addOp = rewriter.create<IE::AddOp>(declLoc, convResults[0], convResults[i],
                                           IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY),
                                           nullptr, nullptr, nullptr, nullptr);
        convResults[0] = addOp.getOutput();
    }

    rewriter.replaceAllUsesWith(concatOp, addOp);
    return mlir::success();
}

bool OptimizeConcatWithConvAndAdd::isBeneficialToConvert(IE::ConcatOp concatOp) const {
    const auto elementType = concatOp.getOutput().getType().getElementType();
    if (!isElementTypeSupported(elementType)) {
        _log.trace("Concat at {0} with element type {1} is not supported", concatOp.getLoc(), elementType);
        return false;
    }

    auto dimOrder = DimsOrder::fromValue(concatOp);
    if (dimOrder != DimsOrder::NHWC) {
        _log.trace("Concat input at {0} layout not support", concatOp.getLoc());
        return false;
    }

    auto concatAxis = IE::getConcatAxis(concatOp);
    if (!concatAxis.has_value() || concatAxis.value().ind() != Dims4D::Act::C.ind()) {
        _log.trace("Concat input at {0} concat axis is not channel", concatOp.getLoc());
        return false;
    }

    // Experimental number to determine if the concat input is efficient to convert. Details data: E161332.
    constexpr int CONVERT_CONCAT_RATIO = 8192;
    auto concatOutput = concatOp.getOutput();
    auto concatOutShape = getShape(concatOutput);
    if ((concatOutShape[Dims4D::Act::H] * concatOutShape[Dims4D::Act::W]) / concatOutShape[Dims4D::Act::C] <
        CONVERT_CONCAT_RATIO) {
        _log.trace("Concat input at {0} not efficient to convert", concatOp.getLoc());
        return false;
    }

    // Following check the new Convolution ops can be optimized by AdjustConvShape pass.
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(
            mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType()).getElementType());

    auto isQuantizedType = [](NDTypeInterface ndType) {
        const auto elementType = ndType.getElementType();
        return mlir::isa<mlir::quant::QuantizedType>(elementType);
    };
    const auto concatOutType = mlir::cast<vpux::NDTypeInterface>(concatOutput.getType());
    if (isQuantizedType(concatOutType)) {
        _log.trace("Concat input at {0} with unsupported Quantized Type", concatOp.getLoc());
        return false;
    }
    const auto strideX = 1;

    auto concatInputs = concatOp.getInputs();
    for (size_t i = 0; i < concatInputs.size(); ++i) {
        const auto currentInput = concatInputs[i];
        const auto currentType = mlir::cast<vpux::NDTypeInterface>(currentInput.getType());

        Shape filterShape = {concatOutShape[Dims4D::Act::C], currentType.getShape()[Dims4D::Act::C], 1, 1};

        int64_t alignedInputChannel = channelAlignment;
        int64_t alignedOutputChannel = channelAlignment;
        auto caculateExpandShapeSize = [](ShapeRef shape, int64_t alignedChannel) {
            auto expandShape = shape.toValues();
            expandShape[Dims4D::Act::C] = alignValUp(shape[Dims4D::Act::C], alignedChannel);
            return expandShape.totalSize();
        };

        const auto inputShape = currentType.getShape();
        const auto outputShape = concatOutShape;
        Shape maybePaddedInputShape(inputShape.raw());
        auto padNum = 0;
        const auto wcInDimSize = inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::W];
        if (wcInDimSize % alignedInputChannel) {
            padNum = (alignValUp(wcInDimSize, alignedInputChannel) - wcInDimSize) / inputShape[Dims4D::Act::C];
            maybePaddedInputShape[Dims4D::Act::W] = inputShape[Dims4D::Act::W] + padNum;
        }

        const auto wcOutDimSize = outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::W];
        if (wcOutDimSize % alignedOutputChannel) {
            if ((wcOutDimSize % alignedInputChannel) || (alignedOutputChannel % alignedInputChannel)) {
                _log.trace("The output channel*width ({0}) can't get align factor {1}", wcOutDimSize,
                           alignedOutputChannel);
                return false;
            }
            alignedOutputChannel = alignedInputChannel;
        }

        auto calcBorrowFactor = [](int64_t channel, int64_t alignedChannel) {
            auto leastAlignedChannel = std::lcm(channel, alignedChannel);
            return (leastAlignedChannel / channel);
        };

        const auto borrowIn = calcBorrowFactor(maybePaddedInputShape[Dims4D::Act::C], alignedInputChannel);
        const auto borrowOut = calcBorrowFactor(outputShape[Dims4D::Act::C], alignedOutputChannel);

        auto realInFactor = std::lcm(strideX, borrowIn);
        if (realInFactor == 0 || maybePaddedInputShape[Dims4D::Act::W] % realInFactor != 0) {
            _log.trace("Don't have factor {0} in input DimW", realInFactor);
            return false;
        }

        if (outputShape[Dims4D::Act::W] % borrowOut) {
            _log.trace("Don't have factor {0} in output DimW", borrowOut);
            return false;
        }

        auto newInputDimW = maybePaddedInputShape[Dims4D::Act::W] / realInFactor;
        while (realInFactor < filterShape[Dims4D::Filter::KX] && newInputDimW > 1) {
            auto divisor = vpux::smallestDivisor(newInputDimW);
            realInFactor *= divisor;
            newInputDimW /= divisor;
        }

        Shape newFilterShape(filterShape.raw());
        const auto borrowFactor = std::max(borrowIn, borrowOut);
        newFilterShape[Dims4D::Filter::IC] *= borrowFactor;
        newFilterShape[Dims4D::Filter::OC] *= borrowFactor;

        const auto newFilterSize = newFilterShape.totalSize();
        const auto expandedInputSize = caculateExpandShapeSize(maybePaddedInputShape, alignedInputChannel);
        const auto expandedOutputSize = caculateExpandShapeSize(outputShape, alignedOutputChannel);
        const auto expandedTotalSize = expandedInputSize + expandedOutputSize;

        Byte elemSizeBytes = currentType.getElemTypeSize().to<Byte>();
        Byte cmxMemSize = (currentType.getTotalAllocSize() + concatOutType.getTotalAllocSize()) / elemSizeBytes.count();
        const auto elementSize = currentType.getCompactAllocSize().count() / maybePaddedInputShape.totalSize();

        if (expandedTotalSize * elementSize < cmxMemSize.count() || newFilterSize > Byte(1_MB).count()) {
            _log.trace("CMX size not meet perfromance requirement");
            return false;
        }

        constexpr int EFFICIENT_CHANNEL_SIZE = 4;
        if (inputShape[Dims4D::Act::C] == EFFICIENT_CHANNEL_SIZE) {
            // For better performance, check the input channel.
            alignedInputChannel = EFFICIENT_CHANNEL_SIZE;
        }

        auto kernelScaled =
                static_cast<float>(newFilterShape.totalSize()) / static_cast<float>(filterShape.totalSize());
        auto outputTensorScaled = static_cast<float>(expandedOutputSize) / static_cast<float>(outputShape.totalSize());
        if ((filterShape[Dims4D::Filter::IC] % alignedInputChannel) == 0 &&
            (kernelScaled / outputTensorScaled) > alignedOutputChannel) {
            _log.trace("The shape adjust cost greater than expand when input channel already aligned");
            return false;
        }
    }

    return true;
}

//
// OptimizeConcatWithConvPass
//

class OptimizeConcatWithConvPass final : public IE::impl::OptimizeConcatWithConvBase<OptimizeConcatWithConvPass> {
public:
    explicit OptimizeConcatWithConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeConcatWithConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeConcat>(&ctx, tileOp.getCount(), _log);
    patterns.add<OptimizeConcatWithConvAndAdd>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeConcatWithConvPass(Logger log) {
    return std::make_unique<OptimizeConcatWithConvPass>(log);
}
