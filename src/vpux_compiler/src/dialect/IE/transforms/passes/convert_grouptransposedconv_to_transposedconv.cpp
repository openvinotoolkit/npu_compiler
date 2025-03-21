//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGROUPTRANSPOSEDCONVTOTRANSPOSEDCONV
#define GEN_PASS_DEF_CONVERTGROUPTRANSPOSEDCONVTOTRANSPOSEDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// GroupTransposedConvConverter
//

class GroupTransposedConvConverter final : public mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp> {
public:
    GroupTransposedConvConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Converts IE.GroupTransposedConvolution to IE.TransposedConvolution(s) for non-depthwise cases.
// IE.GroupTransposedConvolution: input   [N, GROUPS * IC, H, W]
//                                weights [GROUPS, OC, IC, KH, KW]
// to GROUPS IE.TransposedConvolutions: input   [N, IC, H, W]
//                                      weights [OC, IC, KH, KW]
//
// For example: IE.GroupTransposedConvolution: input   [1, 64, 10, 10]
//                                             weights [2, 32, 32, 3, 1]
//              to 2x IE.TransposedConvolution: input   [1, 32, 10, 10]
//                                              weights [32, 32, 3, 1]
mlir::LogicalResult GroupTransposedConvConverter::matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got GroupConvolutionOp at '{0}'", origOp->getLoc());

    const auto input = origOp.getInput();
    const auto weights = origOp.getFilter();
    const auto inputShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    if (inputShape.size() != 4 || weightsShape.size() != 5) {
        return matchFailed(rewriter, origOp,
                           "Only 4D inputs and 5D weights are supported, got {0}D inputs and {1}D weights",
                           inputShape.size(), weightsShape.size());
    }

    const auto groups = weightsShape.front();
    if (groups == inputShape[Dims4D::Act::C]) {
        return matchFailed(rewriter, origOp, "Depthwise operation skipped");
    }

    Shape newInShape(inputShape.raw());
    newInShape[Dims4D::Act::C] /= groups;
    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    Shape newWeightsShape(weightsShape.raw());
    newWeightsShape[Dim(IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX)] = 1;
    const auto weightsShapeAttr = getIntArrayAttr(getContext(), newWeightsShape);
    const auto newWeightsShapeSqueezed = Shape(weightsShape.begin() + 1, weightsShape.end());
    const auto newWeightsShapeSqueezedAttr = getIntArrayAttr(getContext(), newWeightsShapeSqueezed);

    SmallVector<mlir::Value> slices;
    for (const auto sliceIdx : irange(groups)) {
        // Slice input
        Shape inputOffsets(inputShape.size(), 0);
        inputOffsets[Dims4D::Act::C] = checked_cast<int64_t>(inputShape[Dims4D::Act::C] / groups * sliceIdx);
        const auto inputOffsetsAttr = getIntArrayAttr(getContext(), inputOffsets);
        const auto inputSlice = rewriter.createOrFold<IE::SliceOp>(
                takeOpLoc(origOp, StringLiteral("slice_{0}"), sliceIdx), input, inputOffsetsAttr, inputShapeAttr);

        // Slice weights
        mlir::Value weightsSlice;
        Shape weightsOffsets(weightsShape.size(), 0);
        weightsOffsets[Dim(IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX)] = sliceIdx;
        const auto weightsOffsetsAttr = getIntArrayAttr(getContext(), weightsOffsets);
        if (auto fqOp = weights.getDefiningOp<IE::FakeQuantizeOp>()) {
            const auto sliceFqConstInput = [&](mlir::Value fqInput, StringRef locSuffix) {
                auto fqInputType = fqInput.getType().cast<NDTypeInterface>();
                const auto fqInputShape = fqInputType.getShape();
                Shape newFqInputShape(fqInputShape.raw());
                newFqInputShape[Dim(IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX)] = 1;

                const auto numElems = fqInputType.getNumElements();
                const auto numElemsGroupDim = fqInputShape[Dim(IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX)];
                VPUX_THROW_UNLESS(numElems == numElemsGroupDim,
                                  "Per-axis quantization for GroupTransposedConvolution only works on the group "
                                  "dimension, got dimensions {0}",
                                  fqInputShape);
                if (numElems != 1) {
                    const auto newFqInputShapeAttr = getIntArrayAttr(getContext(), newFqInputShape);
                    fqInput = rewriter.createOrFold<IE::SliceOp>(
                            takeOpLoc(fqOp, StringLiteral("fq_weights_{0}_{1}"), locSuffix, sliceIdx), fqInput,
                            weightsOffsetsAttr, newFqInputShapeAttr);
                }

                const auto newFqInputShapeSqueezed = Shape(newFqInputShape.begin() + 1, newFqInputShape.end());
                const auto newFqInputShapeSqueezedAttr = getIntArrayAttr(getContext(), newFqInputShapeSqueezed);
                return rewriter.createOrFold<IE::ReshapeOp>(
                        takeOpLoc(fqOp, StringLiteral("reshape_weights_{0}_{1}"), locSuffix, sliceIdx), fqInput,
                        nullptr, false, newFqInputShapeSqueezedAttr);
            };

            auto newInput = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fqOp, StringLiteral("slice_in_{0}"), sliceIdx),
                                                               fqOp.getInput(), weightsOffsetsAttr, weightsShapeAttr);
            newInput = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(fqOp, StringLiteral("reshape_in_{0}"), sliceIdx),
                                                            newInput, nullptr, false, newWeightsShapeSqueezedAttr);
            auto inputLow = sliceFqConstInput(fqOp.getInputLow(), "in_low");
            auto inputHigh = sliceFqConstInput(fqOp.getInputHigh(), "in_high");
            auto outputLow = sliceFqConstInput(fqOp.getOutputLow(), "out_low");
            auto outputHigh = sliceFqConstInput(fqOp.getOutputHigh(), "out_high");
            weightsSlice = rewriter.create<IE::FakeQuantizeOp>(
                    takeOpLoc(fqOp, StringLiteral("fq_in_{0}"), sliceIdx), newInput, inputLow, inputHigh, outputLow,
                    outputHigh, fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
        } else {
            weightsSlice = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("weights_{0}"), sliceIdx),
                                                              weights, weightsOffsetsAttr, weightsShapeAttr);
            weightsSlice =
                    rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("slice_in_{0}"), sliceIdx),
                                                       weights, weightsOffsetsAttr, weightsShapeAttr);
            weightsSlice =
                    rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("reshape_in_{0}"), sliceIdx),
                                                         weightsSlice, nullptr, false, newWeightsShapeSqueezedAttr);
        }

        _log.nest().trace("Creating TransposedConvolution op for group {0} with channels [{1}-{2})", sliceIdx,
                          inputOffsets[Dims4D::Act::C], inputOffsets[Dims4D::Act::C] + newInShape[Dims4D::Act::C]);

        auto newTransposedConvLoc = appendLoc(origOp->getLoc(), "as_gconv_{0}", sliceIdx);
        auto transposedConvOp = rewriter.create<IE::TransposedConvolutionOp>(
                newTransposedConvLoc, inputSlice, weightsSlice, origOp.getOutputShape(), /*bias*/ nullptr,
                origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());

        slices.push_back(transposedConvOp);
    }

    auto concatOp = rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slices, Dims4D::Act::C.ind());
    takeOpLoc(concatOp, "concat_out");

    return mlir::success();
}

//
// DepthwiseGroupTransposedConvConverter
//

class DepthwiseGroupTransposedConvConverter final : public mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp> {
public:
    DepthwiseGroupTransposedConvConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Const::DeclareOp findWeightsConstant(mlir::Value weightsOperand) const;
    mlir::Value createNewWeightsConst(mlir::PatternRewriter& rewriter, Const::DeclareOp weightsOp) const;

    Logger _log;
};

Const::DeclareOp DepthwiseGroupTransposedConvConverter::findWeightsConstant(mlir::Value weightsOperand) const {
    auto constOp = weightsOperand.getDefiningOp<Const::DeclareOp>();
    if (auto fqOp = weightsOperand.getDefiningOp<IE::FakeQuantizeOp>()) {
        constOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
    }
    return constOp;
}

mlir::Value DepthwiseGroupTransposedConvConverter::createNewWeightsConst(mlir::PatternRewriter& rewriter,
                                                                         Const::DeclareOp weightsOp) const {
    const auto weightsType = weightsOp.getType().cast<NDTypeInterface>();
    const auto weightsShape = weightsType.getShape().raw();
    const int64_t newIC = weightsShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    const int64_t newOC = weightsShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    const int64_t newKY = weightsShape[IE::GROUP_TRANSPOSED_CONV_KY_DIM_INDEX];
    const int64_t newKX = weightsShape[IE::GROUP_TRANSPOSED_CONV_KX_DIM_INDEX];
    Shape newWeightsShape({newIC, newOC, newKY, newKX});
    const auto newWeightsSize = std::accumulate(newWeightsShape.begin(), newWeightsShape.end(), static_cast<int64_t>(1),
                                                std::multiplies<int64_t>());
    std::vector<vpux::type::float16> newValues(newWeightsSize, 0.0f);

    const auto groups = weightsShape.front();
    const auto groupSize = std::accumulate(weightsShape.begin() + 1, weightsShape.end(), static_cast<int64_t>(1),
                                           std::multiplies<int64_t>());
    auto content = weightsOp.getContent();
    auto values = to_small_vector(content.getValues<vpux::type::float16>());
    for (auto group : irange(groups)) {
        auto inputStart = group * groupSize;
        auto outputStart = group * newOC * newKY * newKX + group * newKY * newKX;
        std::copy_n(values.begin() + inputStart, groupSize, newValues.begin() + outputStart);
    }

    const auto baseType =
            mlir::RankedTensorType::get(newWeightsShape.raw(), mlir::Float16Type::get(weightsOp.getContext()));
    return Const::createConst(rewriter, weightsOp.getLoc(), baseType, ArrayRef(newValues));
}

// Converts IE.GroupTransposedConvolution to IE.TransposedConvolution for depthwise cases.
// IE.GroupTransposedConvolution: input   [N, GROUPS, H, W]
//                                weights [GROUPS, 1, 1, KH, KW]
// IE.TransposedConvolution: input   [N, GROUPS, H, W]
//                           weights [GROUPS, GROUPS, KH, KW]
//
// For example: IE.GroupTransposedConvolution: input   [1, 64, 10, 10]
//                                             weights [64, 1, 1, 3, 1]
//              to IE.TransposedConvolution: input   [1, 64, 10, 10]
//                                           weights [64, 64, 3, 1]
mlir::LogicalResult DepthwiseGroupTransposedConvConverter::matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got depthwise GroupConvolutionOp at '{0}'", origOp->getLoc());

    const auto input = origOp.getInput();
    const auto weights = origOp.getFilter();
    const auto inputShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    if (inputShape.size() != 4 || weightsShape.size() != 5) {
        return matchFailed(rewriter, origOp,
                           "Only 4D inputs and 5D weights are supported, got {0}D inputs and {1}D weights",
                           inputShape.size(), weightsShape.size());
    }

    const auto groups = weightsShape.front();
    if (groups != inputShape[Dims4D::Act::C]) {
        return matchFailed(rewriter, origOp, "Non-Depthwise operation skipped");
    }

    const auto weightsOp = findWeightsConstant(weights);
    if (weightsOp == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to find weights constant");
    }

    _log.nest().trace("Converting to TransposedConvolution");

    auto newWeights = createNewWeightsConst(rewriter, weightsOp);
    if (auto fqOp = weights.getDefiningOp<IE::FakeQuantizeOp>()) {
        auto inputLow = fqOp.getInputLow();
        auto inputHigh = fqOp.getInputHigh();
        auto outputLow = fqOp.getOutputLow();
        auto outputHigh = fqOp.getOutputHigh();

        const auto fqParamShape = inputLow.getType().cast<NDTypeInterface>().getShape().raw();
        const Shape newFQParamShapeSqueezed = Shape({fqParamShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX],
                                                     fqParamShape[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                                                     fqParamShape[IE::GROUP_TRANSPOSED_CONV_KY_DIM_INDEX],
                                                     fqParamShape[IE::GROUP_TRANSPOSED_CONV_KX_DIM_INDEX]});
        const auto newFQParamShapeSqueezedAttr = getIntArrayAttr(getContext(), newFQParamShapeSqueezed);
        inputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_in_low_resh"), inputLow, nullptr, false, newFQParamShapeSqueezedAttr));
        outputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_in_high_resh"), outputLow, nullptr, false, newFQParamShapeSqueezedAttr));
        inputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_out_low_resh"), inputHigh, nullptr, false, newFQParamShapeSqueezedAttr));
        outputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_out_high_resh"), outputHigh, nullptr, false, newFQParamShapeSqueezedAttr));

        newWeights = rewriter.create<IE::FakeQuantizeOp>(takeOpLoc(origOp, "weights_fq"), newWeights, inputLow,
                                                         inputHigh, outputLow, outputHigh, fqOp.getLevelsAttr(),
                                                         fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
    }

    rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
            origOp, input, newWeights, origOp.getOutputShape(), nullptr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());

    return mlir::success();
}

//
// ConvertGroupTransposedConvToTransposedConvPass
//

class ConvertGroupTransposedConvToTransposedConvPass final :
        public IE::impl::ConvertGroupTransposedConvToTransposedConvBase<
                ConvertGroupTransposedConvToTransposedConvPass> {
public:
    explicit ConvertGroupTransposedConvToTransposedConvPass(const bool enableSEPTransposedConv, Logger log)
            : _enableSEPTransposedConv(enableSEPTransposedConv) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableSEPTransposedConv;
};

mlir::LogicalResult ConvertGroupTransposedConvToTransposedConvPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (enableSEPTransposedConv.hasValue()) {
        _enableSEPTransposedConv = enableSEPTransposedConv.getValue();
    }

    return mlir::success();
}

void ConvertGroupTransposedConvToTransposedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    const auto isLegalGroupTransposedConv = [&](IE::GroupTransposedConvolutionOp groupTransposedConv) {
        _log.trace("Got '{0}' at '{1}'", groupTransposedConv->getName(), groupTransposedConv->getLoc());
        if (!_enableSEPTransposedConv) {
            _log.nest().trace("SEP disabled for TransposedConvolutions");
            return true;
        }
        if (!VPU::isSupportedSEPTransposedConv(groupTransposedConv, logCb, /*checkLayout=*/false,
                                               /*checkChannelAlignment=*/false)) {
            _log.nest().trace("GroupTransposedConvolutionOp cannot be executed using SEP");
            return true;
        }
        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::GroupTransposedConvolutionOp>(isLegalGroupTransposedConv);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::TransposedConvolutionOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupTransposedConvConverter>(&ctx, _log);
    patterns.add<DepthwiseGroupTransposedConvConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGroupTransposedConvToTransposedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupTransposedConvToTransposedConvPass(
        const bool enableSEPTransposedConv, Logger log) {
    return std::make_unique<ConvertGroupTransposedConvToTransposedConvPass>(enableSEPTransposedConv, log);
}
