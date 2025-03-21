//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <numeric>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTUPSAMPLINGTOSTRIDEDCONCAT
#define GEN_PASS_DEF_CONVERTUPSAMPLINGTOSTRIDEDCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

auto getConcatResult(mlir::PatternRewriter& rewriter, vpux::Dim axis, int64_t factor, mlir::Value input,
                     ShapeRef constShape, IE::UpsamplingOp origOp) {
    const auto zeroType = mlir::RankedTensorType::get(
            constShape, mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getElementType());
    auto constZeros = Const::createZerosConst(rewriter, origOp->getLoc(), zeroType);

    SmallVector<mlir::Value> concatInputs;
    concatInputs.push_back(input);
    for (int i = 0; i < factor - 1; i++) {
        concatInputs.push_back(constZeros);
    }

    return rewriter
            .create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("concat_cst_d{0}"), axis.ind()),
                                  mlir::ValueRange(concatInputs), axis, 1, factor)
            .getOutput();
}

//
// ConvertUpsamplingToStridedConcatPass
//

class ConvertUpsamplingToStridedConcatPass final :
        public IE::impl::ConvertUpsamplingToStridedConcatBase<ConvertUpsamplingToStridedConcatPass> {
public:
    explicit ConvertUpsamplingToStridedConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class UpsamplingOpConverter;

private:
    void safeRunOnFunc() final;
};

// UpsamplingOpConverter
class ConvertUpsamplingToStridedConcatPass::UpsamplingOpConverter final :
        public mlir::OpRewritePattern<IE::UpsamplingOp> {
public:
    UpsamplingOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::UpsamplingOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::UpsamplingOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertUpsamplingToStridedConcatPass::UpsamplingOpConverter::matchAndRewrite(
        IE::UpsamplingOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Upsampling Op {0}", origOp->getLoc());

    const auto inputShape = getShape(origOp.getInput());

    const auto upsamplingFactorVectorTmp = parseIntArrayAttr<int64_t>(origOp.getUpsamplingFactor());
    auto padChannel = parseIntArrayAttr<int64_t>(origOp.getPadAttr().getPadsChannel());
    auto padHeight = parseIntArrayAttr<int64_t>(origOp.getPadAttr().getPadsHeight());
    auto padWidth = parseIntArrayAttr<int64_t>(origOp.getPadAttr().getPadsWidth());
    SmallVector<int64_t> upsamplingFactorVector = {1, upsamplingFactorVectorTmp[2], upsamplingFactorVectorTmp[1],
                                                   upsamplingFactorVectorTmp[0]};
    SmallVector<int64_t> padingLAtt = {0, padChannel[0], padHeight[0], padWidth[0]};
    SmallVector<int64_t> padingRAtt = {0, padChannel[1], padHeight[1], padWidth[1]};
    bool convertNow = (upsamplingFactorVectorTmp[2] != 1);
    bool needSlicing = false;
    Shape sliceShape(inputShape.raw());

    // The DMA only handle this scenario
    //   outputShape = inShape*upsamplingFactor
    // Or it can't use a DMA to describe it.
    // We can get below formula based on the upsampingOP's defination
    //   outputShape =  inShape + (inShape - 1) * (upsamplingFactorVector - 1) + padLVector + padRVector
    // It can be simplified as below when padL = 0
    //    outputShape=inShape*upsamplingFactorVector - (upsamplingFactorVector - 1) + padRVector
    auto calculatePaddingEndFun = [&](size_t ch, int64_t factor) {
        auto delta = factor - 1;
        if (padingRAtt[ch] < delta) {
            // This upsamplingOP can't be converted to DMA, and run the stride concat logic.
            convertNow = true;
            needSlicing = true;
            sliceShape[Dim(ch)] = (sliceShape[Dim(ch)] * factor) - (delta - padingRAtt[ch]);
            padingRAtt[ch] = 0;
        } else {
            padingRAtt[ch] -= delta;
            sliceShape[Dim(ch)] = sliceShape[Dim(ch)] * factor;
        }
    };

    auto upsamplingResult = origOp.getInput();
    for (size_t i = 0; i < upsamplingFactorVector.size(); i++) {
        if (upsamplingFactorVector[i] > 1) {
            calculatePaddingEndFun(i, upsamplingFactorVector[i]);
        }
    }
    if (convertNow) {
        for (size_t i = 0; i < upsamplingFactorVector.size(); i++) {
            if (upsamplingFactorVector[i] > 1 && inputShape[Dim(i)] > 1) {
                upsamplingResult = getConcatResult(rewriter, Dim(i), upsamplingFactorVector[i], upsamplingResult,
                                                   getShape(upsamplingResult), origOp);
            }
        }
        if (needSlicing && (sliceShape != inputShape)) {
            const Shape sliceOffsets{0, 0, 0, 0};
            auto slice = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "slice_in"), upsamplingResult,
                                                      getIntArrayAttr(rewriter, sliceOffsets.raw()),
                                                      getIntArrayAttr(rewriter, sliceShape.raw()));
            upsamplingResult = slice.getResult();
        }
    } else {
        auto padChannelAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, upsamplingFactorVectorTmp[2] - 1});
        auto padHeightAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, upsamplingFactorVectorTmp[1] - 1});
        auto padWidthAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, upsamplingFactorVectorTmp[0] - 1});
        auto padAttr = IE::UpsamplingPadAttr::get(rewriter.getContext(), padChannelAttr, padHeightAttr, padWidthAttr);

        auto newUpsample =
                rewriter.create<IE::UpsamplingOp>(origOp->getLoc(), upsamplingResult, origOp.getUpsamplingFactor(),
                                                  padAttr, origOp.getOutputChannelsAttr());
        upsamplingResult = newUpsample.getOutput();
    }
    auto isZero = [](auto val) {
        return val == 0;
    };
    if ((!llvm::all_of(padingLAtt, isZero)) || (!llvm::all_of(padingRAtt, isZero))) {
        auto padBeginAttr = getIntArrayAttr(rewriter, padingLAtt);
        auto padEndAttr = getIntArrayAttr(rewriter, padingRAtt);
        auto zeroFpAttr = getFPAttr(rewriter, 0.0f);
        auto padingOp = rewriter.create<IE::PadOp>(takeOpLoc(origOp, "pad_out"), upsamplingResult, nullptr, nullptr,
                                                   nullptr, padBeginAttr, padEndAttr, zeroFpAttr, IE::PadMode::CONSTANT,
                                                   origOp.getOutputChannelsAttr());
        upsamplingResult = padingOp.getOutput();
    }

    rewriter.replaceOp(origOp, upsamplingResult);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertUpsamplingToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::UpsamplingOp>([&](IE::UpsamplingOp op) {
        const auto inputShape = getShape(op.getInput());
        SmallVector<int64_t> padLVector = {
                checked_cast<int64_t>(op.getPadAttr().getPadsChannel()[0].cast<mlir::IntegerAttr>().getInt()),
                checked_cast<int64_t>(op.getPadAttr().getPadsHeight()[0].cast<mlir::IntegerAttr>().getInt()),
                checked_cast<int64_t>(op.getPadAttr().getPadsWidth()[0].cast<mlir::IntegerAttr>().getInt())};

        SmallVector<int64_t> padRVector = {
                checked_cast<int64_t>(op.getPadAttr().getPadsChannel()[1].cast<mlir::IntegerAttr>().getInt()),
                checked_cast<int64_t>(op.getPadAttr().getPadsHeight()[1].cast<mlir::IntegerAttr>().getInt()),
                checked_cast<int64_t>(op.getPadAttr().getPadsWidth()[1].cast<mlir::IntegerAttr>().getInt())};
        const auto upsamplingFactorVector = parseIntArrayAttr<int64_t>(op.getUpsamplingFactor());

        // Upsampling only supports 4D Input shape
        // Upsampling supports pads only for 3 axes
        // Upsampling supports factors only for 3 axes
        if (inputShape.size() != 4 || padLVector.size() != 3 || padRVector.size() != 3 ||
            upsamplingFactorVector.size() != 3) {
            return true;
        }

        // Based on upsamplingOP's defination, the input/output's shape has below relation
        //      outputShape=inShape*upsamplingFactorVector - (upsamplingFactorVector - 1) + padLVector + padRVector
        // To make the upsampingOP can be converted to DMA. It should promise
        //      outputShape = inShape*upsamplingFactor
        // So padL = 0 and padRVector = upsamplingFactorVector - 1
        int64_t initValue = 0;
        int64_t sumPadL = std::accumulate(padLVector.begin(), padLVector.end(), initValue);
        if (sumPadL) {
            return false;
        }

        if (1 != upsamplingFactorVector[0] - padRVector[2] || 1 != upsamplingFactorVector[1] - padRVector[1] ||
            1 != upsamplingFactorVector[2] - padRVector[0]) {
            return false;
        }
        return true;
    });

    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::PadOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UpsamplingOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertUpsamplingToStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertUpsamplingToStridedConcatPass(Logger log) {
    return std::make_unique<ConvertUpsamplingToStridedConcatPass>(log);
}
