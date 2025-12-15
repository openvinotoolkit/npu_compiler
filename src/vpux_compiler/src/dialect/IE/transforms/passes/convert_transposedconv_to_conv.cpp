//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTRANSPOSEDCONV2DTOCONV2D
#define GEN_PASS_DEF_CONVERTTRANSPOSEDCONV2DTOCONV2D
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// TransposedConvolutionConversion
//

class TransposedConvolutionConversion final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    TransposedConvolutionConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx), _log(log) {
        setDebugName("TransposedConvolutionConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TransposedConvolutionConversion::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::TransposedConvolution Operation '{0}'", origOp->getLoc());

    auto padsOutput = Shape(parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPadding()));

    const auto featureShape = getShape(origOp.getInput());
    VPUX_THROW_UNLESS(featureShape.size() == 4, "Only 2D transposed convolution is supported");

    const auto outputShape = getShape(origOp.getOutput());
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Only 2D transposed convolution is supported");

    auto filterShape = getShape(origOp.getFilter()).toValues();
    VPUX_THROW_UNLESS(filterShape.size() == 4, "Only 2D transposed convolution is supported");

    auto featureUpScale = IE::createUpsampling(rewriter, takeOpLoc(origOp, "upscale_in"), origOp, padsOutput, false);
    if (mlir::failed(featureUpScale)) {
        _log.nest().trace("Failed to create Upsampling for {0}", origOp->getLoc());
        return mlir::failure();
    }
    auto paddingOutput = featureUpScale.value();

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    auto resultOP = rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), paddingOutput, origOp.getFilter(),
                                                       origOp.getBias(), strides, padsBegin, padsEnd, dilations,
                                                       origOp.getPostOpAttr(), origOp.getClampAttr(),
                                                       /*staticScale=*/nullptr, origOp.getOutputPaddingAttr(),
                                                       origOp.getInputPaddingAttr())
                            .getOutput();

    const auto nceOutputShape = mlir::cast<vpux::NDTypeInterface>(resultOP.getType()).getShape();
    if (origOp.getOutputShape() != nullptr && nceOutputShape != outputShape) {
        // In case the outputShape is specified, create sliceOp for crop
        auto upsamplingOp = paddingOutput.getDefiningOp<IE::UpsamplingOp>();
        const auto padHeightVector = parseIntArrayAttr<int64_t>(upsamplingOp.getPadAttr().getPadsHeight());
        const auto padWidthVector = parseIntArrayAttr<int64_t>(upsamplingOp.getPadAttr().getPadsWidth());
        const auto origPadLeft = filterShape[Dims4D::Filter::KX] - 1;
        const auto origPadTop = filterShape[Dims4D::Filter::KY] - 1;
        const auto reducedPadLeft = origPadLeft - padWidthVector[0];
        const auto reducedPadTop = origPadTop - padHeightVector[0];
        const auto padsBeginVector = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
        auto offsets = SmallVector<int64_t>(outputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(outputShape.begin(), outputShape.end());
        offsets[Dims4D::Act::H.ind()] = padsBeginVector[Dims4D::PadsBegin::Top] - reducedPadTop;
        offsets[Dims4D::Act::W.ind()] = padsBeginVector[Dims4D::PadsBegin::Left] - reducedPadLeft;

        resultOP = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "slice_out"), resultOP,
                                                getIntArrayAttr(getContext(), offsets),
                                                getIntArrayAttr(getContext(), sizes))
                           .getResult();

        const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp.getOutput().user_begin()));
        if (outputFQ != nullptr) {
            resultOP = vpux::IE::createFQ(rewriter, resultOP, outputFQ, takeOpLoc(outputFQ, "fq_out")).getOutput();
        }
    }

    rewriter.replaceOp(origOp, resultOP);

    _log.trace("Replaced TransposedConvolution at '{0}' with 'IE::Convolution' (2D)", origOp.getLoc());

    return mlir::success();
}

//
// ConvertTransposedConv2DToConv2DPass
//

class ConvertTransposedConv2DToConv2DPass final :
        public IE::impl::ConvertTransposedConv2DToConv2DBase<ConvertTransposedConv2DToConv2DPass> {
public:
    explicit ConvertTransposedConv2DToConv2DPass(const bool enableSEPTransposedConv, Logger log)
            : _enableSEPTransposedConv(enableSEPTransposedConv) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    bool _enableSEPTransposedConv;
};

mlir::LogicalResult ConvertTransposedConv2DToConv2DPass::initialize(mlir::MLIRContext* ctx) {
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

void ConvertTransposedConv2DToConv2DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    const auto isLegalTransposedConvOp = [&](IE::TransposedConvolutionOp transposedConv) {
        _log.trace("Got '{0}' at '{1}'", transposedConv->getName(), transposedConv->getLoc());
        if (_enableSEPTransposedConv && VPU::isSupportedSEPTransposedConv(transposedConv, logCb, /*checkLayout=*/false,
                                                                          /*checkChannelAlignment=*/false)) {
            _log.nest(1).trace("TransposedConv can be executed using SEP");
            return true;
        }
        if (mlir::failed(IE::canConvertTransposedConvToConv(transposedConv))) {
            _log.nest(1).trace("TransposedConv cannot be converted. Filter must be constant");
            return true;
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::TransposedConvolutionOp>(isLegalTransposedConvOp);
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TransposedConvolutionConversion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertTransposedConv2DToConv2DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertTransposedConv2DToConv2DPass(const bool enableSEPTransposedConv,
                                                                                Logger log) {
    return std::make_unique<ConvertTransposedConv2DToConv2DPass>(enableSEPTransposedConv, log);
}
