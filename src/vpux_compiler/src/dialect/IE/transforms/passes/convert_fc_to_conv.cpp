//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTFCTOCONV
#define GEN_PASS_DEF_CONVERTFCTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertFCToConvPass
//

class ConvertFCToConvPass final : public IE::impl::ConvertFCToConvBase<ConvertFCToConvPass> {
public:
    explicit ConvertFCToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class FullyConnectedOpConverter;
    class DynamicFullyConnectedRewriter;

private:
    void safeRunOnFunc() final;
};

//
// DynamicFullyConnectedRewriter
//
// Normalizes a dynamically-shaped FullyConnected into a statically-shaped one, so that
// FullyConnectedOpConverter below only ever deals with static shapes.
//
//
class ConvertFCToConvPass::DynamicFullyConnectedRewriter final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    DynamicFullyConnectedRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx, benefit), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertFCToConvPass::DynamicFullyConnectedRewriter::matchAndRewrite(
        IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto input = origOp.getInput();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    if (!mlir::isa<Core::BoundedTensorType>(inType)) {
        return matchFailed(_log, rewriter, origOp, "FullyConnected input is already static");
    }

    const auto inShape = inType.getShape();
    if (inShape.size() != 2 || inShape[Dim(0)] != mlir::ShapedType::kDynamic ||
        inShape[Dim(1)] == mlir::ShapedType::kDynamic) {
        return matchFailed(_log, rewriter, origOp, "Only dynamic batch with static channels is supported");
    }

    _log.trace("Got dynamic '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* ctx = getContext();

    // Pad the dynamic rows up to their bound.
    const auto staticInType =
            mlir::RankedTensorType::get(getBoundedShape(input).raw(), inType.getElementType(),
                                        vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace()));
    auto expanded =
            IE::DynamicExpandOp::create(rewriter, takeOpLoc(origOp, "input_expand"), staticInType, input).getOutput();

    auto staticFC =
            IE::FullyConnectedOp::create(rewriter, origOp->getLoc(), expanded, origOp.getWeights(), origOp.getBias());

    // the static column count.
    const auto outShape = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getShape().raw();
    const auto si32 = getSInt32Type(ctx);
    const auto dimType = mlir::RankedTensorType::get({1}, si32);

    auto shapeOf = IE::ShapeOfOp::create(rewriter, takeOpLoc(origOp, "shape_of"),
                                         mlir::RankedTensorType::get({2}, si32), input, si32);
    auto rowDim =
            IE::SliceOp::create(rewriter, takeOpLoc(origOp, "row_dim"), dimType, shapeOf.getOutput(),
                                getIntArrayAttr(ctx, ArrayRef<int64_t>{0}), getIntArrayAttr(ctx, ArrayRef<int64_t>{1}));
    auto colDim = Const::createDenseConst(rewriter, takeOpLoc(origOp, "col_dim"), dimType,
                                          checked_cast<int32_t>(outShape[1]));
    auto targetShape = IE::ConcatOp::create(
            rewriter, takeOpLoc(origOp, "target_shape"), mlir::RankedTensorType::get({2}, si32),
            mlir::ValueRange{rowDim.getResult(), colDim},
            /*per_axis=*/nullptr, getIntArrayOfArray(ctx, SmallVector<SmallVector<int64_t>>{{0}, {1}}));

    rewriter.replaceOpWithNewOp<IE::DynamicReshapeOp>(
            origOp, origOp.getOutput().getType(), staticFC.getOutput(), targetShape.getOutput(),
            getIntArrayAttr(ctx, outShape), getIntArrayAttr(ctx, getBoundedShape(origOp.getOutput()).raw()), nullptr);

    return mlir::success();
}

//
// FullyConnectedOpConverter
//

class ConvertFCToConvPass::FullyConnectedOpConverter final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    FullyConnectedOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertFCToConvPass::FullyConnectedOpConverter::matchAndRewrite(
        IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getShape().raw();
    const std::array<int64_t, 4> newInShape = {inputShape[0], inputShape[1], 1, 1};
    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    auto newInput =
            IE::ReshapeOp::create(rewriter, takeOpLoc(origOp, "input_reshape"), origOp.getInput(), inputShapeAttr);

    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType()).getShape().raw();
    const std::array<int64_t, 4> newWeightsShape = {weightsShape[0], weightsShape[1], 1, 1};
    const auto filterShapeAttr = getIntArrayAttr(getContext(), newWeightsShape);
    auto newFilter =
            IE::ReshapeOp::create(rewriter, takeOpLoc(origOp, "filter_reshape"), origOp.getWeights(), filterShapeAttr);

    mlir::Value newBias;
    if (origOp.getBias() != nullptr) {
        const auto biasShape = mlir::cast<vpux::NDTypeInterface>(origOp.getBias().getType()).getShape().raw();
        const std::array<int64_t, 4> newBiasShape = {biasShape[0], biasShape[1], 1, 1};
        const auto biasShapeAttr = getIntArrayAttr(getContext(), newBiasShape);
        newBias = IE::ReshapeOp::create(rewriter, takeOpLoc(origOp, "bias_reshape"), origOp.getBias(), biasShapeAttr);
    }

    auto newStrides = getIntArrayAttr(getContext(), ov::Strides{1, 1});
    auto newPadsBegin = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newPadsEnd = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newDilations = getIntArrayAttr(getContext(), ov::Strides{1, 1});

    auto convOp = IE::ConvolutionOp::create(rewriter, takeOpLoc(origOp, "as_convolution"), newInput, newFilter, newBias,
                                            nullptr, nullptr, newStrides, newPadsBegin, newPadsEnd, newDilations);
    copyLoopAttributes(origOp, convOp.getOperation());

    const auto convShape = mlir::cast<vpux::NDTypeInterface>(convOp.getOutput().getType()).getShape().raw();
    const std::array<int64_t, 2> outputShape = {convShape[0], convShape[1]};
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    auto newOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, convOp.getOutput(), outputShapeAttr);
    extendOpLoc(newOp, "output_reshape");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertFCToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::FullyConnectedOp>();
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::ReshapeOp>();
    // Ops emitted by DynamicFullyConnectedRewriter to pad the rows and restore the dynamic result type.
    target.addLegalOp<IE::DynamicExpandOp>();
    target.addLegalOp<IE::DynamicReshapeOp>();
    target.addLegalOp<IE::ShapeOfOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    // Higher benefit: a dynamic FullyConnected must be made static before it can be lowered to a
    // convolution, so this rewriter runs first and hands a static FullyConnected to the converter.
    patterns.add<DynamicFullyConnectedRewriter>(&ctx, _log, /*benefit=*/2);
    patterns.add<FullyConnectedOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertFCToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertFCToConvPass(Logger log) {
    return std::make_unique<ConvertFCToConvPass>(log);
}
