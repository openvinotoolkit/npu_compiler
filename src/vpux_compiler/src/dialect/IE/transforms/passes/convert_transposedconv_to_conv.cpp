//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTRANSPOSEDCONV2DTOCONV2D
#define GEN_PASS_DEF_CONVERTTRANSPOSEDCONV2DTOCONV2D
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertTransposedConvOp(IE::TransposedConvolutionOp transposedConv, bool enableSEPTransposedConv,
                                   Logger log) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    log.trace("Got '{0}' at '{1}'", transposedConv->getName(), transposedConv->getLoc());
    if (enableSEPTransposedConv) {
        auto seOp = mlir::dyn_cast<IE::SEOpInterface>(transposedConv.getOperation());
        if (seOp && seOp.isSupported(logCb)) {
            log.nest(1).trace("TransposedConv can be executed using SEP");
            return false;
        }
    }
    if (mlir::failed(IE::canConvertTransposedConvToConv(transposedConv))) {
        log.nest(1).trace("TransposedConv cannot be converted. Filter must be constant");
        return false;
    }

    return true;
}

//
// TransposedConvolutionConversion
//

class TransposedConvolutionConversion final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    TransposedConvolutionConversion(mlir::MLIRContext* ctx, bool enableSEPTransposedConv, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx),
              _enableSEPTransposedConv(enableSEPTransposedConv),
              _log(log) {
        setDebugName("TransposedConvolutionConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableSEPTransposedConv;
    Logger _log;
};

mlir::LogicalResult TransposedConvolutionConversion::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertTransposedConvOp(origOp, _enableSEPTransposedConv, _log)) {
        return mlir::failure();
    }

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
    // E#222712: Preserve original dilations; hard-coding [1,1] dropped dilation and caused shape mismatches.
    auto dilations = origOp.getDilationsAttr();
    const auto dilationsVec = parseIntArrayAttr<int64_t>(dilations);
    const auto dilY = dilationsVec[Dims4D::Dilation::Y.ind()];
    const auto dilX = dilationsVec[Dims4D::Dilation::X.ind()];

    auto resultOP =
            rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), paddingOutput, origOp.getFilter(), origOp.getBias(),
                                               /*scale*/ nullptr, /*zero_points*/ nullptr, strides, padsBegin, padsEnd,
                                               dilations, origOp.getPostOpAttr(), origOp.getClampAttr(),
                                               /*staticScale=*/nullptr, origOp.getOutputPaddingAttr(),
                                               origOp.getInputPaddingAttr())
                    .getOutput();

    const auto nceOutputShape = mlir::cast<vpux::NDTypeInterface>(resultOP.getType()).getShape();
    if (origOp.getOutputShape() != nullptr && nceOutputShape != outputShape) {
        // In case the outputShape is specified, create sliceOp for crop
        auto upsamplingOp = paddingOutput.getDefiningOp<IE::UpsamplingOp>();
        const auto padHeightVector = parseIntArrayAttr<int64_t>(upsamplingOp.getPadAttr().getPadsHeight());
        const auto padWidthVector = parseIntArrayAttr<int64_t>(upsamplingOp.getPadAttr().getPadsWidth());
        // Mirror createUpsampling: (K_eff - 1) = (K - 1) * dilation.
        const auto origPadLeft = (filterShape[Dims4D::Filter::KX] - 1) * dilX;
        const auto origPadTop = (filterShape[Dims4D::Filter::KY] - 1) * dilY;
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

    const auto origLoc = origOp.getLoc();

    rewriter.replaceOp(origOp, resultOP);

    _log.trace("Replaced TransposedConvolution at '{0}' with 'IE::Convolution' (2D)", origLoc);

    return mlir::success();
}

//
// ConvertTransposedConv2DToConv2DPass
//

class ConvertTransposedConv2DToConv2DPass final :
        public IE::impl::ConvertTransposedConv2DToConv2DBase<ConvertTransposedConv2DToConv2DPass> {
public:
    explicit ConvertTransposedConv2DToConv2DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertTransposedConv2DToConv2DPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto func = getOperation();
    const auto moduleOp = getModuleOp(func);
    const auto enableSEPtrsOps = config::hasEnableSEPtrsOperations(moduleOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TransposedConvolutionConversion>(&ctx, enableSEPtrsOps, _log);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertTransposedConv2DToConv2DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertTransposedConv2DToConv2DPass(Logger log) {
    return std::make_unique<ConvertTransposedConv2DToConv2DPass>(log);
}
