//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertNearestToStridedConcatPass
//

class ConvertNearestToStridedConcatPass final :
        public IE::ConvertNearestToStridedConcatBase<ConvertNearestToStridedConcatPass> {
public:
    explicit ConvertNearestToStridedConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class NearestInterpolateOpConverter;

private:
    void safeRunOnFunc() final;
};

class ConvertNearestToStridedConcatPass::NearestInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    NearestInterpolateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertNearestToStridedConcatPass::NearestInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = getShape(origOp.input());
    const auto outShape = parseIntArrayAttr<int64_t>(origOp.sizes_attrAttr());
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Input shape must be 4d");
    VPUX_THROW_UNLESS(outShape.size() == 2, "Wrong number of spatial dims");
    // TODO: if out shape is not N times of in shape
    VPUX_THROW_UNLESS(outShape[1] % inputShape[Dims4D::Act::W] == 0 && outShape[0] % inputShape[Dims4D::Act::H] == 0,
                      "Only support N times upsampling");
    const auto scaleX = outShape[1] / inputShape[Dims4D::Act::W];
    const auto scaleY = outShape[0] / inputShape[Dims4D::Act::H];

    SmallVector<mlir::Value> widthSlices;
    SmallVector<mlir::Value> heightSlices;
    // Here is an assumption : scaleX !=0 AND scaleY !=0 as output shape is non-zero
    for (int i = 0; i < scaleY; ++i) {
        for (int j = 0; j < scaleX; ++j) {
            widthSlices.push_back(origOp.input());
        }
        heightSlices.push_back(widthSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), widthSlices,
                                                                                       Dims4D::Act::W, 1, scaleX)
                                                       : widthSlices.front());
        widthSlices.clear();
    }
    const auto resultConcat = heightSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), heightSlices,
                                                                                       Dims4D::Act::H, 1, scaleY)
                                                       : heightSlices.front();
    rewriter.replaceOp(origOp, resultConcat);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertNearestToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        const auto attrs = op.attr();
        return !(attrs.mode().getValue() == IE::InterpolateMode::nearest && !attrs.antialias().getValue() &&
                 attrs.coord_mode().getValue() == IE::InterpolateCoordMode::asymmetric &&
                 attrs.shape_calc_mode().getValue() == IE::InterpolateCalcMode::sizes &&
                 op.axes_attrAttr().size() == 2);
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<NearestInterpolateOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNearestToStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNearestToStridedConcatPass(Logger log) {
    return std::make_unique<ConvertNearestToStridedConcatPass>(log);
}
