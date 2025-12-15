//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTNEARESTTOSTRIDEDCONCAT
#define GEN_PASS_DEF_CONVERTNEARESTTOSTRIDEDCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertNearestToBroadcastOrStridedConcatPass
//

class ConvertNearestToBroadcastOrStridedConcatPass final :
        public IE::impl::ConvertNearestToStridedConcatBase<ConvertNearestToBroadcastOrStridedConcatPass> {
public:
    explicit ConvertNearestToBroadcastOrStridedConcatPass(const bool interpolateAsSEOp, Logger log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class NearestToBroadcastConverter;
    class NearestToStridedConcatConverter;

private:
    void safeRunOnFunc() final;

private:
    bool _interpolateAsSEOp;
};

mlir::LogicalResult ConvertNearestToBroadcastOrStridedConcatPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (interpolateAsSEOp.hasValue()) {
        _interpolateAsSEOp = interpolateAsSEOp.getValue();
    }

    return mlir::success();
}

// NearestToBroadcastConverter

class ConvertNearestToBroadcastOrStridedConcatPass::NearestToBroadcastConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    NearestToBroadcastConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertNearestToBroadcastOrStridedConcatPass::NearestToBroadcastConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto& ctx = origOp.getContext();
    const auto outShape = getShape(origOp.getOutput());

    if (!IE::isBroadCastInterpolate(origOp)) {
        return mlir::failure();
    }

    const auto shapeStorageType =
            mlir::RankedTensorType::get({static_cast<int64_t>(outShape.size())}, getSInt64Type(ctx));
    const auto shapeConst = Const::createConst(rewriter, takeOpLoc(origOp, "out_shape"), shapeStorageType,
                                               outShape.raw(), [&](Const::ContentSetup& setup) {
                                                   return setup.castElemType(getSInt32Type(rewriter.getContext()));
                                               });
    rewriter.replaceOpWithNewOp<IE::BroadcastOp>(origOp, origOp.getInput(), shapeConst, nullptr,
                                                 IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));

    return mlir::success();
}

// NearestToStridedConcatConverter

class ConvertNearestToBroadcastOrStridedConcatPass::NearestToStridedConcatConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    NearestToStridedConcatConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertNearestToBroadcastOrStridedConcatPass::NearestToStridedConcatConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = getShape(origOp.getInput());
    const auto outShape = getShape(origOp.getOutput());
    const auto attrs = origOp.getAttr();
    const auto nearestMode = attrs.getNearestMode().getValue();

    int64_t outputW = 0;
    int64_t outputH = 0;

    const auto outShapeSize = outShape.size();
    if (outShapeSize == 2) {
        outputW = outShape[Dim(1)];
        outputH = outShape[Dim(0)];
    } else if (outShapeSize == 4) {
        outputW = outShape[Dims4D::Act::W];
        outputH = outShape[Dims4D::Act::H];
    } else {
        VPUX_THROW("Wrong number of spatial dims: {0}", outShapeSize);
    }

    // TODO: add support for cases where output dimension is not divisible by input dimension
    VPUX_THROW_UNLESS(outputW % inputShape[Dims4D::Act::W] == 0 && outputH % inputShape[Dims4D::Act::H] == 0,
                      "Only N times upsampling is supported");

    const auto scaleX = outputW / inputShape[Dims4D::Act::W];
    const auto scaleY = outputH / inputShape[Dims4D::Act::H];

    const auto inputFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = !(origOp->getResult(0).use_empty())
                                  ? mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp->getResult(0).user_begin()))
                                  : nullptr;

    SmallVector<mlir::Value> widthSlices;
    SmallVector<mlir::Value> heightSlices;
    mlir::Value widthConcatOp;
    // Here is an assumption : scaleX !=0 AND scaleY !=0 as output shape is non-zero

    for (int j = 0; j < scaleX; ++j) {
        widthSlices.push_back(origOp.getInput());
    }

    widthConcatOp = widthSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "wslices_concat"),
                                                                            widthSlices, Dims4D::Act::W, 1, scaleX)
                                                      .getOutput()
                                            : widthSlices.front();

    // TODO remove this propagation after moving such functionality to Q-D propagation pass
    if (inputFQ != nullptr && outputFQ != nullptr && widthSlices.size() != 0) {
        widthConcatOp =
                vpux::IE::createFQ(rewriter, widthConcatOp, outputFQ, takeOpLoc(outputFQ, "fq_out")).getOutput();
    }
    for (int i = 0; i < scaleY; ++i) {
        heightSlices.push_back(widthConcatOp);
    }
    const auto resultConcat = heightSlices.size() != 1
                                      ? rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "hslices_concat"), heightSlices,
                                                                      Dims4D::Act::H, 1, scaleY)
                                      : heightSlices.front();

    int64_t padW = 0;
    int64_t padH = 0;

    if (nearestMode == IE::InterpolateNearestMode::CEIL) {
        padW = scaleX - 1;
        padH = scaleY - 1;
    }
    if (nearestMode == IE::InterpolateNearestMode::ROUND_PREFER_CEIL ||
        nearestMode == IE::InterpolateNearestMode::ROUND_PREFER_FLOOR) {
        if (scaleX % 2 == 0 && nearestMode == IE::InterpolateNearestMode::ROUND_PREFER_FLOOR) {
            padW = scaleX / 2 - 1;
        } else {
            padW = scaleX / 2;
        }
        if (scaleY % 2 == 0 && nearestMode == IE::InterpolateNearestMode::ROUND_PREFER_FLOOR) {
            padH = scaleY / 2 - 1;
        } else {
            padH = scaleY / 2;
        }
    }
    const bool needPaddingW = padW != 0;
    const bool needPaddingH = padH != 0;

    auto tensorPaddedWidth =
            (needPaddingW) ? IE::createPadding(rewriter, origOp, resultConcat, Dims4D::Act::W, -padW, padW, "width")
                           : resultConcat;
    auto tensorPadded = (needPaddingH) ? IE::createPadding(rewriter, origOp, tensorPaddedWidth, Dims4D::Act::H, -padH,
                                                           padH, "height")
                                       : tensorPaddedWidth;

    rewriter.replaceOp(origOp, tensorPadded);

    return mlir::success();
}  // namespace

//
// safeRunOnFunc
//

void ConvertNearestToBroadcastOrStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalConvertToStrideConcat = [&](IE::InterpolateOp op) {
        const auto attrs = op.getAttr();
        const bool validAxesAttrSize = (op.getAxesAttrAttr().size() == 2 || op.getAxesAttrAttr().size() == 4);
        const auto inputShape = getShape(op.getInput());
        const auto outShape = getShape(op.getOutput());

        return attrs.getMode().getValue() == IE::InterpolateMode::NEAREST && !attrs.getAntialias().getValue() &&
               attrs.getCoordMode().getValue() == IE::InterpolateCoordMode::ASYMMETRIC && validAxesAttrSize &&
               (outShape[Dims4D::Act::W] % inputShape[Dims4D::Act::W] == 0) &&
               (outShape[Dims4D::Act::H] % inputShape[Dims4D::Act::H] == 0);
    };

    const auto isLegalConvertToBroadCast = [&](IE::InterpolateOp op) {
        return IE::isBroadCastInterpolate(op);
    };

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        if (_interpolateAsSEOp) {
            if (VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false,
                                                   /*checkBatch=*/false)) {
                return true;
            }
        }

        return !(isLegalConvertToStrideConcat(op) || isLegalConvertToBroadCast(op));
    });
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::BroadcastOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(2);
    patterns.add<NearestToBroadcastConverter>(&ctx, benefitLevels[0], _log);
    patterns.add<NearestToStridedConcatConverter>(&ctx, benefitLevels[1], _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNearestToBroadCastOrStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNearestToBroadCastOrStridedConcatPass(const bool interpolateAsSEOp,
                                                                                         Logger log) {
    return std::make_unique<ConvertNearestToBroadcastOrStridedConcatPass>(interpolateAsSEOp, log);
}
