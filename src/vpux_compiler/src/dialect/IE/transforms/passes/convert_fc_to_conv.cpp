//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>

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

private:
    void safeRunOnFunc() final;
};

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
    const auto inputShape = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 4> newInShape = {inputShape[0], inputShape[1], 1, 1};
    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    auto newInput = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "input_reshape"), origOp.getInput(), nullptr,
                                                   false, inputShapeAttr);

    const auto weightsShape = origOp.getWeights().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 4> newWeightsShape = {weightsShape[0], weightsShape[1], 1, 1};
    const auto filterShapeAttr = getIntArrayAttr(getContext(), newWeightsShape);
    auto newFilter = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "filter_reshape"), origOp.getWeights(), nullptr,
                                                    false, filterShapeAttr);

    mlir::Value newBias;
    if (origOp.getBias() != nullptr) {
        const auto biasShape = origOp.getBias().getType().cast<vpux::NDTypeInterface>().getShape().raw();
        const std::array<int64_t, 4> newBiasShape = {biasShape[0], biasShape[1], 1, 1};
        const auto biasShapeAttr = getIntArrayAttr(getContext(), newBiasShape);
        newBias = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "bias_reshape"), origOp.getBias(), nullptr, false,
                                                 biasShapeAttr);
    }

    auto newStrides = getIntArrayAttr(getContext(), ov::Strides{1, 1});
    auto newPadsBegin = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newPadsEnd = getIntArrayAttr(getContext(), ov::CoordinateDiff{0, 0});
    auto newDilations = getIntArrayAttr(getContext(), ov::Strides{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(takeOpLoc(origOp, "as_convolution"), newInput, newFilter, newBias,
                                                     newStrides, newPadsBegin, newPadsEnd, newDilations, nullptr,
                                                     nullptr, nullptr, nullptr, nullptr);

    const auto convShape = convOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 2> outputShape = {convShape[0], convShape[1]};
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    auto newOp =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, convOp.getOutput(), nullptr, false, outputShapeAttr);
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

    mlir::RewritePatternSet patterns(&ctx);
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
