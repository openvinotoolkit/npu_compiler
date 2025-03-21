//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/scale_shift_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSCALESHIFTTODW
#define GEN_PASS_DEF_CONVERTSCALESHIFTTODW
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertScaleShiftToDepthwisePass
//

class ConvertScaleShiftToDWPass final : public IE::impl::ConvertScaleShiftToDWBase<ConvertScaleShiftToDWPass> {
public:
    explicit ConvertScaleShiftToDWPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ScaleShiftOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ScaleShiftOpConverter
//

class ConvertScaleShiftToDWPass::ScaleShiftOpConverter final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    ScaleShiftOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScaleShiftOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScaleShiftToDWPass::ScaleShiftOpConverter::matchAndRewrite(
        IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const {
    if (mlir::failed(vpux::IE::isBeneficialConvertScaleShiftToDW(origOp, _log))) {
        return mlir::failure();
    }

    const SmallVector<int32_t> strides = {1, 1};
    const SmallVector<int32_t> padBegin = {0, 0};
    const SmallVector<int32_t> padEnd = {0, 0};
    const SmallVector<int32_t> dilations = {1, 1};

    const int64_t kernelSize = 1;

    auto dilationsAttr = getIntArrayAttr(origOp.getContext(), dilations);
    auto stridesAttr = getIntArrayAttr(origOp.getContext(), strides);
    auto padBeginAttr = getIntArrayAttr(origOp.getContext(), padBegin);
    auto padEndAttr = getIntArrayAttr(origOp.getContext(), padEnd);

    auto outShape = getShape(origOp.getOutput()).toValues();
    auto groupAttr = getIntAttr(origOp.getContext(), outShape[Dims4D::Act::C]);
    const SmallVector<int64_t> weightShape = {outShape[Dims4D::Act::C], 1, kernelSize, kernelSize};
    mlir::Value weights;

    auto createConstOp = [&](ArrayRef<int64_t> shape, vpux::type::float16 value) -> mlir::Value {
        const auto elemType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
        auto shape1x1x1x1 = SmallVector<int64_t>(shape.size(), 1);
        const auto dataStorageType = mlir::RankedTensorType::get(shape1x1x1x1, elemType);

        return Const::createConst(rewriter, origOp.getLoc(), dataStorageType, ArrayRef(value),
                                  [&](Const::ContentSetup& contentAttrSetup) {
                                      for (auto dim : enumerate(shape)) {
                                          if (dim.value() > 1) {
                                              contentAttrSetup =
                                                      contentAttrSetup.broadcast(Dim(dim.index()), dim.value());
                                          }
                                      }
                                      return std::move(contentAttrSetup);
                                  });
    };

    if (origOp.getWeights() != nullptr) {
        const auto multiply = origOp.getWeights();
        const auto weightShapeAttr = getIntArrayAttr(origOp.getContext(), weightShape);
        auto dwConvFilter = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), multiply, nullptr, false, weightShapeAttr);
        weights = dwConvFilter.getOutput();
    } else {
        weights = createConstOp(weightShape, 1.0f);
        auto inputFq = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
        if (inputFq != nullptr) {
            // the created FQ is artificial, levels are set to 255 in correlation with FQ range from 0.0f to 254.0f (255
            // integer values)
            const auto lowFpTypeAttr = inputFq.getLowFpTypeAttr();
            const auto levelsAttr = lowFpTypeAttr == nullptr ? getIntAttr(rewriter, 255) : nullptr;
            auto newFqOp =
                    rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), weights, weights, weights, weights, weights,
                                                        levelsAttr, lowFpTypeAttr, inputFq.getAutoBroadcastAttr());
            weights = newFqOp.getOutput();
        }
    }

    // if ScaleShift has biases with a shape 1x1x1x1 it should be expanded to output channels
    mlir::Value biases = origOp.getBiases();
    if (biases != nullptr) {
        auto origBiasShape = getShape(biases).toValues();
        if (origBiasShape[Dims4D::Act::C] == 1) {
            const SmallVector<int64_t> biasShape = {1, outShape[Dims4D::Act::C], kernelSize, kernelSize};
            const auto biaseShapeAttr = getIntArrayAttr(origOp.getContext(), biasShape);
            auto dwConvBias = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), biases, nullptr, false, biaseShapeAttr);
            biases = dwConvBias.getOutput();
        }
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, origOp.getInput(), weights, biases, stridesAttr,
                                                        padBeginAttr, padEndAttr, dilationsAttr, groupAttr, nullptr,
                                                        nullptr, nullptr, nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScaleShiftToDWPass::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ScaleShiftOpConverter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScaleShiftToDWPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScaleShiftToDWPass(Logger log) {
    return std::make_unique<ConvertScaleShiftToDWPass>(log);
}
