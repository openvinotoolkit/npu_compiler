//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SPLITBILINERINTOHANDW
#define GEN_PASS_DEF_SPLITBILINERINTOHANDW
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SplitBilinerIntoHAndWPass
//

class SplitBilinerIntoHAndWPass final : public IE::impl::SplitBilinerIntoHAndWBase<SplitBilinerIntoHAndWPass> {
public:
    explicit SplitBilinerIntoHAndWPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class BilinearInterpolateOpConverter;

private:
    void safeRunOnFunc() final;
};

// BilinearInterpolateOpConverter
class SplitBilinerIntoHAndWPass::BilinearInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    BilinearInterpolateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isLegalOpToConvertToSliceAndConv(IE::InterpolateOp op, LogCb logCb) {
    const auto attrs = op.getAttr();
    const auto interpMode = attrs.getMode().getValue();
    const auto antiAlias = attrs.getAntialias().getValue();
    const auto coordMode = attrs.getCoordMode().getValue();
    const auto inputShape = getShape(op.getInput());
    const auto outShape = getShape(op.getOutput());
    const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();

    if (!VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false,
                                            /*checkBatch=*/false)) {
        return false;
    }

    if ((interpMode != IE::InterpolateMode::LINEAR_ONNX && interpMode != IE::InterpolateMode::LINEAR) || antiAlias) {
        return false;
    }

    const auto inputElemType = inputType.getElementType();
    if (inputElemType.isa<mlir::quant::QuantizedType>()) {
        // Support of quantized case will be open after E#104698 fix AC issue.
        return false;
    }

    // Only support 4D Input shape.
    if (inputShape.size() != 4) {
        return false;
    }

    if ((inputShape[Dims4D::Act::N] != outShape[Dims4D::Act::N]) ||
        (inputShape[Dims4D::Act::C] != outShape[Dims4D::Act::C])) {
        return false;
    }

    if (auto alignInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op.getOperation())) {
        const auto alignment = alignInterface.getInputChannelAlignment();
        // The created convolution's output channel will be inputShape[Dims4D::Act::C]*2, inputShape[Dims4D::Act::C]
        // align to alignment/2 will also get some performance improvement.
        if ((inputShape[Dims4D::Act::C] * 2) % alignment != 0) {
            return false;
        }
    }

    // Runtime already has a efficient implementation for this case
    // And also current solution for this case will produce lots of DMAs, which is not efficient.
    if (inputShape[Dims4D::Act::H] == 1 && inputShape[Dims4D::Act::W] == 1) {
        return false;
    }

    if (coordMode != vpux::IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) {
        return false;
    }
    // This conversion method can only be effective with 2x scaling.
    if (inputShape[Dims4D::Act::W] * vpux::IE::CONVERT_BILINEAR_TO_STRIDED_CONCAT_CONVOLUTION_V2_SUPPORTED_SCALE !=
                outShape[Dims4D::Act::W] ||
        inputShape[Dims4D::Act::H] * vpux::IE::CONVERT_BILINEAR_TO_STRIDED_CONCAT_CONVOLUTION_V2_SUPPORTED_SCALE !=
                outShape[Dims4D::Act::H]) {
        return false;
    }

    return true;
}

/*
This pass split the bilinear interpolate into interpolate on H and interpolate on W,
the interpolate on H will convert to sep interpolate, the interpolate on W will convert to convolution.

For the interpolate on W, there was a example:

Input: NHWC 1x2x3x3         Kernel:NHWC 4x2x1x2
    C1:        C2:
    1 2 3      10 20 30     C1: 0.75 0.25
    4 5 6      40 50 60     C2: 0    0
    7 8 9      70 80 90     C3: 0    0
            |               C4: 0.75 0.25
            |               C5: 0.25 0.75
           Conv ----------  C6: 0    0
            |               C7: 0    0
            |               C8: 0.25 0.75
            V
    Output: NHWC 1x4x3x2
    C1: 1.25    2.25
        4.25    5.25
        7.25    8.25

    C2: 12.5    22.5
        42.5    52.5
        72.5    82.5

    C3: 1.75    2.75
        4.75    5.75
        7.75    8.75

    C4: 17.5    27.5
        47.5    57.5
        77.5    87.5
         |
    AffineReshape
         |
         V
    NHWC 1x2x3x4                        Slice1 from input:  Slice2 from input:
C1: 1.25    1.75    2.25    2.75        C1: 1               C1: 3
    4.25    4.75    5.25    5.75            4                   6
    7.25    7.75    8.25    8.75            7                   9

C2: 12.5    17.5    22.5    27.5        C2  10              C2  30
    42.5    47.5    52.5    57.5            40                  60
    72.5    77.5    82.5    87.5            70                  90
                |                           |                   |
                |                           |                   |
                +------------------------concat-----------------+
                                            |
                                            V
                                Result: NHWC 1x2x3x6
                    C1: 1   1.25    1.75    2.25    2.75    3
                        4   4.25    4.75    5.25    5.75    6
                        7   7.25    7.75    8.25    8.75    9

                    C2: 10  12.5    17.5    22.5    27.5    30
                        40  42.5    47.5    52.5    57.5    60
                        70  72.5    77.5    82.5    87.5    90
*/

mlir::LogicalResult SplitBilinerIntoHAndWPass::BilinearInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (!isLegalOpToConvertToSliceAndConv(origOp, logCb)) {
        return mlir::failure();
    }
    _log.trace("Get bilinear Interpolate Op {0}", origOp);

    const auto loc = origOp.getLoc();
    auto ctx = origOp->getContext();

    auto shapeCalcMode = origOp.getAttr().getShapeCalcMode().getValue();
    auto newSizesAttrAttr = origOp.getSizesAttrAttr();
    auto newScalesAttrAttr = origOp.getScalesAttrAttr();
    auto inputShape = getShape(origOp.getInput());
    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto axes = IE::getInterpAxesVal(origOp.getLoc(), origOp.getAxes(), origOp.getAxesAttrAttr(), inputType);

    auto dimPtr = std::find(axes.begin(), axes.end(), Dims4D::Act::H.ind());
    auto dimIdx = std::distance(axes.begin(), dimPtr);

    if (shapeCalcMode == IE::InterpolateCalcMode::SIZES) {
        const auto sizes = parseIntArrayAttr<int64_t>(newSizesAttrAttr);
        auto newSizes = SmallVector<double>(sizes.size(), 1.0);
        for (auto axis : axes | indexed) {
            newSizes[axis.index()] =
                    (Dim(axis.value()) == Dims4D::Act::H) ? sizes[axis.index()] : inputShape[Dim(axis.value())];
        }
        newSizesAttrAttr = getIntArrayAttr(getContext(), newSizes);
    }

    if (shapeCalcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales = parseFPArrayAttr<double>(newScalesAttrAttr);
        auto newScales = SmallVector<double>(scales.size(), 1.0);
        newScales[dimIdx] = scales[dimIdx];
        newScalesAttrAttr = getFPArrayAttr(getContext(), newScales);
    }

    const auto interpolateHShape = Shape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                         inputShape[Dims4D::Act::H] * 2, inputShape[Dims4D::Act::W]};

    auto interpolateHOutputType = outputType.changeShape(interpolateHShape);

    auto interpolateOnH = rewriter.create<IE::InterpolateOp>(
            appendLoc(loc, "_interpolateOnH"), interpolateHOutputType, origOp.getInput(), origOp.getSizes(),
            origOp.getScales(), origOp.getAxes(), newSizesAttrAttr, newScalesAttrAttr, origOp.getAxesAttrAttr(),
            origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(),
            origOp.getAttr(), origOp.getOutputChannelsAttr());

    auto interpOnWLoc = appendLoc(loc, "_interpolateOnW");

    auto interpolateOnHOutputShape = getShape(interpolateOnH.getOutput());

    // This pass use convolution instead of linear interpolation on one axis. The output of linear interpolation is
    // the result of linear fitting for two input datas, the 2x scaled pytorch_half_pixel linear interpolation
    // results in only two types of linear fitting:
    // Output = 0.25 * Input1 + 0.75 * Input2
    // Output = 0.75 * Input1 + 0.25 * Input2
    // Therefore, the kernel will be set to [0.25, 0.75] [0.75, 0.25] values, allowing convolution to
    // simulate this linear interpolation operation. So here KY and KX are set to 1 and 2, kernel values set to 0.25
    // and 0.75.

    const auto OC = interpolateOnHOutputShape[Dims4D::Act::C] * 2;
    const auto IC = interpolateOnHOutputShape[Dims4D::Act::C];
    const auto KY = 1;
    const auto KX = 2;

    const Shape weightShape = {OC, IC, KY, KX};
    SmallVector<float> weights(weightShape.totalSize(), .0f);

    for (auto i = 0; i < OC; ++i) {
        auto beginIndex = i * KY * KX + i * IC * KY * KX;
        if (i < OC / 2) {
            weights[beginIndex] = 0.75;
            weights[beginIndex + 1] = 0.25;
        } else {
            beginIndex = beginIndex - IC * KY * KX;
            weights[beginIndex] = 0.25;
            weights[beginIndex + 1] = 0.75;
        }
    }

    const DimsOrder weightOrder = DimsOrder::OYXI;
    const auto weightType = mlir::RankedTensorType::get(
            weightShape.raw(), mlir::cast<NDTypeInterface>(interpolateOnH.getOutput().getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightOrder, nullptr, nullptr));
    auto weight =
            Const::buildWeightsConst(rewriter, interpolateOnH.getOutput().getLoc(), weightType, ArrayRef(weights));

    const auto convOutputShape = Shape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C] * 2,
                                       inputShape[Dims4D::Act::H] * 2, inputShape[Dims4D::Act::W] - 1};

    auto convOutputType = outputType.changeShape(convOutputShape);

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(interpOnWLoc, convOutputType, interpolateOnH.getOutput(), weight,
                                                     nullptr, strides, padsBegin, padsEnd, dilations, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr);

    const SmallVector<int64_t> reshapeOutputShape = {
            convOutputShape[Dims4D::Act::N], convOutputShape[Dims4D::Act::C] / 2, convOutputShape[Dims4D::Act::H],
            convOutputShape[Dims4D::Act::W] * 2};
    const auto reshapeOutputShapeAttr = getIntArrayAttr(ctx, reshapeOutputShape);
    const SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind()},
                                                         {Dims4D::Act::C.ind()},
                                                         {Dims4D::Act::H.ind()},
                                                         {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    auto reshapeOp = rewriter.create<IE::AffineReshapeOp>(
            interpOnWLoc, convOp.getOutput(), getIntArrayOfArray(ctx, inDimMapping), reshapeOutputShapeAttr);

    auto offsets = SmallVector<int64_t>(interpolateOnHOutputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(interpolateOnHOutputShape.begin(), interpolateOnHOutputShape.end());

    sizes[Dims4D::Act::W.ind()] = 1;
    auto leftSlice = rewriter.create<IE::SliceOp>(interpOnWLoc, interpolateOnH.getOutput(),
                                                  getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));

    offsets[Dims4D::Act::W.ind()] = interpolateOnHOutputShape[Dims4D::Act::W] - 1;
    auto rightSlice = rewriter.create<IE::SliceOp>(interpOnWLoc, interpolateOnH.getOutput(),
                                                   getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));

    SmallVector<mlir::Value> concatVector;
    concatVector.push_back(leftSlice.getResult());
    concatVector.push_back(reshapeOp.getOutput());
    concatVector.push_back(rightSlice.getResult());

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, concatVector, Dims4D::Act::W);

    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitBilinerIntoHAndWPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<BilinearInterpolateOpConverter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitBilinerIntoHAndWPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitBilinerIntoHAndWPass(Logger log) {
    return std::make_unique<SplitBilinerIntoHAndWPass>(log);
}
