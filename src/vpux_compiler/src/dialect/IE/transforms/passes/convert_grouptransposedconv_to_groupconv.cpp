//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGROUPTRANSPOSEDCONVTOGROUPCONV
#define GEN_PASS_DEF_CONVERTGROUPTRANSPOSEDCONVTOGROUPCONV
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

mlir::LogicalResult GroupTransposedConvConverter::matchAndRewrite(IE::GroupTransposedConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got GroupTransposedConvolutionOp layer at '{0}'", origOp->getLoc());

    auto padsOutput = Shape(parseIntArrayAttr<int64_t>(origOp.getOutputPadding()));

    const auto featureShape = getShape(origOp.getInput());
    if (featureShape.size() != 4) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupTransposedConvolutionOp is supported, expected 4D feature but got {0}",
                           featureShape.size());
    }

    const auto outputShape = getShape(origOp.getOutput());
    if (outputShape.size() != 4) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupTransposedConvolutionOp is supported, expected 4D output shape but got {0}",
                           outputShape.size());
    }

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto origFilterShape = to_small_vector(filterType.getShape());
    if (origFilterShape.size() != 5) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupTransposedConvolutionOp is supported, expected 5D filter but got {0}",
                           origFilterShape.size());
    }

    auto dwConvFilter = origOp.getFilter().getDefiningOp();
    if (dwConvFilter == nullptr) {
        return matchFailed(rewriter, origOp, "GroupTransposedConvolutionOp has no filter Op");
    }

    auto featureUpScale = IE::createUpsampling(rewriter, takeOpLoc(origOp, "upscale_in"), origOp, padsOutput, true);
    if (mlir::failed(featureUpScale)) {
        _log.nest().trace("Failed to create Upsampling for {0}", origOp->getLoc());
        return mlir::failure();
    }
    auto paddingOutput = featureUpScale.value();

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    // convert filter shape from 5D to 4D
    auto groups = origFilterShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    origFilterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX] *= groups;
    origFilterShape.erase(origFilterShape.begin());

    const auto filter4DShapeAttr = getIntArrayAttr(rewriter.getContext(), origFilterShape);
    auto reshaped4DFilter = rewriter.createOrFold<IE::ReshapeOp>(
            takeOpLoc(origOp, "reshape_filter"), dwConvFilter->getResult(0), nullptr, false, filter4DShapeAttr);

    const auto postOp = origOp.getPostOpAttr();
    const auto clampOp = origOp.getClampAttr();
    const auto outputChannels = origOp.getOutputChannelsAttr();
    const auto inputChannels = origOp.getInputChannelsAttr();

    if (padsOutput[Dims4D::PadsOutput::Y] > 0) {
        paddingOutput = IE::createPadding(rewriter, takeOpLoc(origOp, "height"), paddingOutput, Dims4D::Act::H,
                                          padsOutput[Dims4D::PadsOutput::Y], nullptr);
    }
    if (padsOutput[Dims4D::PadsOutput::X] > 0) {
        paddingOutput = IE::createPadding(rewriter, takeOpLoc(origOp, "width"), paddingOutput, Dims4D::Act::W,
                                          padsOutput[Dims4D::PadsOutput::X], nullptr);
    }

    auto resultOP =
            rewriter.create<IE::GroupConvolutionOp>(origOp->getLoc(), paddingOutput, reshaped4DFilter, nullptr, strides,
                                                    padsBegin, padsEnd, dilations, getIntAttr(rewriter, groups), postOp,
                                                    clampOp, outputChannels, inputChannels)
                    .getOutput();

    rewriter.replaceOp(origOp, resultOP);

    _log.trace("Replaced GroupTransposedConvolutionOp at '{0}' with 'IE::GroupConvolutionOp' (2D)", origOp.getLoc());

    return mlir::success();
}

//
// ConvertGroupTransposedConvToGroupConvPass
//

class ConvertGroupTransposedConvToGroupConvPass final :
        public IE::impl::ConvertGroupTransposedConvToGroupConvBase<ConvertGroupTransposedConvToGroupConvPass> {
public:
    explicit ConvertGroupTransposedConvToGroupConvPass(const bool enableSEPTransposedConv, Logger log)
            : _enableSEPTransposedConv(enableSEPTransposedConv) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableSEPTransposedConv;
};

mlir::LogicalResult ConvertGroupTransposedConvToGroupConvPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableSEPTransposedConv.hasValue()) {
        _enableSEPTransposedConv = enableSEPTransposedConv.getValue();
    }

    return mlir::success();
}

void ConvertGroupTransposedConvToGroupConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    const auto isLegalGroupTransposedConv = [&](IE::GroupTransposedConvolutionOp groupTransposedConv) {
        _log.trace("Got '{0}' at '{1}'", groupTransposedConv->getName(), groupTransposedConv->getLoc());
        if (_enableSEPTransposedConv &&
            VPU::isSupportedSEPTransposedConv(groupTransposedConv, logCb, /*checkLayout=*/false,
                                              /*checkChannelAlignment=*/false)) {
            _log.nest(1).trace("GroupTransposedConvolutionOp can be executed using SEP");
            return true;
        }
        if (mlir::failed(IE::canConvertGroupTransposedConvToGroupConv(groupTransposedConv))) {
            _log.nest(1).trace("GroupTransposedConvolutionOp cannot be converted.");
            return true;
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::GroupTransposedConvolutionOp>(isLegalGroupTransposedConv);
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupTransposedConvConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGroupTransposedConvToGroupConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupTransposedConvToGroupConvPass(
        const bool enableSEPTransposedConv, Logger log) {
    return std::make_unique<ConvertGroupTransposedConvToGroupConvPass>(enableSEPTransposedConv, log);
}
