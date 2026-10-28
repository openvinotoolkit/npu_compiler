//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGROUPTRANSPOSEDCONVTOTRANSPOSEDCONV
#define GEN_PASS_DEF_CONVERTGROUPTRANSPOSEDCONVTOTRANSPOSEDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertGroupTransposedConvToTransposedConv(IE::GroupTransposedConvolutionOp groupTransposedConv,
                                                      bool enableSEPTransposedConv, Logger log) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    log.trace("Got '{0}' at '{1}'", groupTransposedConv->getName(), groupTransposedConv->getLoc());
    if (!enableSEPTransposedConv) {
        log.nest().trace("SEP disabled for TransposedConvolutions");
        return false;
    }
    auto seOp = mlir::dyn_cast<IE::SEOpInterface>(groupTransposedConv.getOperation());
    if (!seOp || !seOp.isSupported(logCb)) {
        log.nest().trace("GroupTransposedConvolutionOp cannot be executed using SEP");
        return false;
    }

    return true;
}

//
// GroupTransposedConvConverter
//

class GroupTransposedConvConverter final : public mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp> {
public:
    GroupTransposedConvConverter(mlir::MLIRContext* ctx, bool enableSEPTransposedConv, Logger log)
            : mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp>(ctx),
              _enableSEPTransposedConv(enableSEPTransposedConv),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableSEPTransposedConv;
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
    if (!shouldConvertGroupTransposedConvToTransposedConv(origOp, _enableSEPTransposedConv, _log)) {
        return mlir::failure();
    }

    _log.trace("Got GroupConvolutionOp at '{0}'", origOp->getLoc());

    const auto input = origOp.getInput();
    const auto weights = origOp.getFilter();
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getShape();
    if (inputShape.size() != 4 || weightsShape.size() != 5) {
        return matchFailed(_log, rewriter, origOp,
                           "Only 4D inputs and 5D weights are supported, got {0}D inputs and {1}D weights",
                           inputShape.size(), weightsShape.size());
    }

    const auto groups = weightsShape.front();
    if (groups == inputShape[Dims4D::Act::C]) {
        return matchFailed(_log, rewriter, origOp, "Depthwise operation skipped");
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
        const auto inputSlice = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, "slice_{0}", sliceIdx), input,
                                                                   inputOffsetsAttr, inputShapeAttr);

        // Slice weights
        mlir::Value weightsSlice;
        Shape weightsOffsets(weightsShape.size(), 0);
        weightsOffsets[Dim(IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX)] = sliceIdx;
        const auto weightsOffsetsAttr = getIntArrayAttr(getContext(), weightsOffsets);
        if (auto fqOp = weights.getDefiningOp<IE::FakeQuantizeOp>()) {
            const auto sliceFqConstInput = [&](mlir::Value fqInput, StringRef locSuffix) {
                auto fqInputType = mlir::cast<vpux::NDTypeInterface>(fqInput.getType());
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
                            takeOpLoc(fqOp, "fq_weights_{0}_{1}", locSuffix, sliceIdx), fqInput, weightsOffsetsAttr,
                            newFqInputShapeAttr);
                }

                const auto newFqInputShapeSqueezed = Shape(newFqInputShape.begin() + 1, newFqInputShape.end());
                const auto newFqInputShapeSqueezedAttr = getIntArrayAttr(getContext(), newFqInputShapeSqueezed);
                return rewriter.createOrFold<IE::ReshapeOp>(
                        takeOpLoc(fqOp, "reshape_weights_{0}_{1}", locSuffix, sliceIdx), fqInput,
                        newFqInputShapeSqueezedAttr);
            };

            auto newInput = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fqOp, "slice_in_{0}", sliceIdx),
                                                               fqOp.getInput(), weightsOffsetsAttr, weightsShapeAttr);
            newInput = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(fqOp, "reshape_in_{0}", sliceIdx), newInput,
                                                            newWeightsShapeSqueezedAttr);
            auto inputLow = sliceFqConstInput(fqOp.getInputLow(), "in_low");
            auto inputHigh = sliceFqConstInput(fqOp.getInputHigh(), "in_high");
            auto outputLow = sliceFqConstInput(fqOp.getOutputLow(), "out_low");
            auto outputHigh = sliceFqConstInput(fqOp.getOutputHigh(), "out_high");
            weightsSlice = IE::FakeQuantizeOp::create(rewriter, takeOpLoc(fqOp, "fq_in_{0}", sliceIdx), newInput,
                                                      inputLow, inputHigh, outputLow, outputHigh, fqOp.getLevelsAttr(),
                                                      fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
        } else {
            weightsSlice = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, "weights_{0}", sliceIdx), weights,
                                                              weightsOffsetsAttr, weightsShapeAttr);
            weightsSlice = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, "slice_in_{0}", sliceIdx), weights,
                                                              weightsOffsetsAttr, weightsShapeAttr);
            weightsSlice = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in_{0}", sliceIdx),
                                                                weightsSlice, newWeightsShapeSqueezedAttr);
        }

        _log.nest().trace("Creating TransposedConvolution op for group {0} with channels [{1}-{2})", sliceIdx,
                          inputOffsets[Dims4D::Act::C], inputOffsets[Dims4D::Act::C] + newInShape[Dims4D::Act::C]);

        auto newTransposedConvLoc = appendLoc(origOp->getLoc(), "as_gconv_{0}", sliceIdx);
        auto transposedConvOp = IE::TransposedConvolutionOp::create(
                rewriter, newTransposedConvLoc, inputSlice, weightsSlice, origOp.getOutputShape(), /*bias*/ nullptr,
                origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getSpatialOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

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
    DepthwiseGroupTransposedConvConverter(mlir::MLIRContext* ctx, bool enableSEPTransposedConv, Logger log)
            : mlir::OpRewritePattern<IE::GroupTransposedConvolutionOp>(ctx),
              _enableSEPTransposedConv(enableSEPTransposedConv),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Const::DeclareOp findWeightsConstant(mlir::Value weightsOperand) const;
    mlir::Value createNewWeightsConst(mlir::PatternRewriter& rewriter, Const::DeclareOp weightsOp) const;

    bool _enableSEPTransposedConv;
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
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(weightsOp.getType());
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
    if (!shouldConvertGroupTransposedConvToTransposedConv(origOp, _enableSEPTransposedConv, _log)) {
        return mlir::failure();
    }

    _log.trace("Got depthwise GroupConvolutionOp at '{0}'", origOp->getLoc());

    const auto input = origOp.getInput();
    const auto weights = origOp.getFilter();
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getShape();
    if (inputShape.size() != 4 || weightsShape.size() != 5) {
        return matchFailed(_log, rewriter, origOp,
                           "Only 4D inputs and 5D weights are supported, got {0}D inputs and {1}D weights",
                           inputShape.size(), weightsShape.size());
    }

    const auto groups = weightsShape.front();
    if (groups != inputShape[Dims4D::Act::C]) {
        return matchFailed(_log, rewriter, origOp, "Non-Depthwise operation skipped");
    }

    const auto weightsOp = findWeightsConstant(weights);
    if (weightsOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Unable to find weights constant");
    }

    _log.nest().trace("Converting to TransposedConvolution");

    auto newWeights = createNewWeightsConst(rewriter, weightsOp);
    if (auto fqOp = weights.getDefiningOp<IE::FakeQuantizeOp>()) {
        auto inputLow = fqOp.getInputLow();
        auto inputHigh = fqOp.getInputHigh();
        auto outputLow = fqOp.getOutputLow();
        auto outputHigh = fqOp.getOutputHigh();

        const auto fqParamShape = mlir::cast<vpux::NDTypeInterface>(inputLow.getType()).getShape().raw();
        const Shape newFQParamShapeSqueezed = Shape({fqParamShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX],
                                                     fqParamShape[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                                                     fqParamShape[IE::GROUP_TRANSPOSED_CONV_KY_DIM_INDEX],
                                                     fqParamShape[IE::GROUP_TRANSPOSED_CONV_KX_DIM_INDEX]});
        const auto newFQParamShapeSqueezedAttr = getIntArrayAttr(getContext(), newFQParamShapeSqueezed);
        inputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_in_low_resh"), inputLow, newFQParamShapeSqueezedAttr));
        outputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_in_high_resh"), outputLow, newFQParamShapeSqueezedAttr));
        inputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_out_low_resh"), inputHigh, newFQParamShapeSqueezedAttr));
        outputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::ReshapeOp>(
                takeOpLoc(origOp, "weights_out_high_resh"), outputHigh, newFQParamShapeSqueezedAttr));

        newWeights = IE::FakeQuantizeOp::create(rewriter, takeOpLoc(origOp, "weights_fq"), newWeights, inputLow,
                                                inputHigh, outputLow, outputHigh, fqOp.getLevelsAttr(),
                                                fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
    }

    rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
            origOp, input, newWeights, origOp.getOutputShape(), nullptr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getSpatialOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    return mlir::success();
}

//
// ConvertGroupTransposedConvToTransposedConvPass
//

class ConvertGroupTransposedConvToTransposedConvPass final :
        public IE::impl::ConvertGroupTransposedConvToTransposedConvBase<
                ConvertGroupTransposedConvToTransposedConvPass> {
public:
    explicit ConvertGroupTransposedConvToTransposedConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertGroupTransposedConvToTransposedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto func = getOperation();
    const auto moduleOp = getModuleOp(func);
    const auto enableSEPtrsOps = config::hasEnableSEPtrsOperations(moduleOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupTransposedConvConverter>(&ctx, enableSEPtrsOps, _log);
    patterns.add<DepthwiseGroupTransposedConvConverter>(&ctx, enableSEPtrsOps, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createConvertGroupTransposedConvToTransposedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupTransposedConvToTransposedConvPass(Logger log) {
    return std::make_unique<ConvertGroupTransposedConvToTransposedConvPass>(log);
}
