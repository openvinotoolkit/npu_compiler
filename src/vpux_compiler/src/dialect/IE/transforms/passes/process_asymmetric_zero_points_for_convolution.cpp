//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <cmath>
#include <functional>

namespace vpux::IE {
#define GEN_PASS_DECL_PROCESSASYMMETRICZEROPOINTSFORCONVOLUTION
#define GEN_PASS_DEF_PROCESSASYMMETRICZEROPOINTSFORCONVOLUTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <typename InOp_t, typename OutOp_t>
OutOp_t checkOp(InOp_t inOp, const std::function<mlir::Value(InOp_t)>& valGetter,
                const std::function<bool(OutOp_t)>& opChecker) {
    if (inOp == nullptr) {
        return nullptr;
    }
    mlir::Value val = valGetter(inOp);
    if (val == nullptr) {
        return nullptr;
    }
    auto op = val.getDefiningOp<OutOp_t>();
    if (op == nullptr) {
        return nullptr;
    }
    if (!opChecker(op)) {
        return nullptr;
    }
    return op;
}

mlir::Value getConvWeights(IE::ConvolutionOp convOp) {
    return convOp.getFilter();
}

bool checkFQ(IE::FakeQuantizeOp fqOp) {
    if (fqOp.getLevels() != 256 || !fqOp.getOutputLow().hasOneUse() || !fqOp.getOutputHigh().hasOneUse() ||
        !fqOp.getOutput().hasOneUse()) {
        return false;
    }
    const auto iLoShape = getShape(fqOp.getInputLow());
    const auto iHiShape = getShape(fqOp.getInputHigh());
    const auto oLoShape = getShape(fqOp.getOutputLow());
    const auto oHiShape = getShape(fqOp.getOutputHigh());
    const auto expectedShape = Shape{1, 1, 1, 1};
    if (iLoShape != expectedShape || iHiShape != expectedShape || oLoShape != expectedShape ||
        oHiShape != expectedShape) {
        return false;
    }
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return false;
    }
    const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
    const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());
    const auto outQuantizeElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(),
                             fqOp.getLowFpType(), realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast());
    if (outQuantizeElemType == nullptr) {
        return false;
    }
    const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outQuantizeElemType);
    if (uniformQuantType == nullptr) {
        return false;
    }
    return !uniformQuantType.isSigned() && uniformQuantType.getStorageTypeIntegralWidth() == 8 &&
           uniformQuantType.getZeroPoint() != 128;
}

mlir::quant::QuantizedType getQuantizedElementTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp) {
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
    const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());
    return getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(),
                            fqOp.getLowFpType(), realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast());
}

class ZeroPointWithConvolution final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ZeroPointWithConvolution(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ZeroPointWithConvolution");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ZeroPointWithConvolution::matchAndRewrite(IE::ConvolutionOp convOp,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto convFilterShape = convOp.getFilter().getType().getShape();
    // Convolution with 1x1 kernel will be handled by more optimized approach.
    if (convFilterShape[Dims4D::Act::H.ind()] == 1 && convFilterShape[Dims4D::Act::W.ind()] == 1) {
        return mlir::failure();
    }

    // Input should not be quantized, quantized input will cause normal f16 execution (not mixed execution)
    // Due to unnecessary cast to i/u8 and back to f16 may lose precision, so to prevent this
    // We don't decompose when input is FQ, or output is going to FQ, though this check needs to be improved
    auto maybeFQInput = convOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
    if (maybeFQInput != nullptr) {
        return mlir::failure();
    }
    auto users = convOp.getOutput().getUsers();
    const auto hasAnyFQUser{std::any_of(users.begin(), users.end(), [](const auto& user) {
        return mlir::isa<IE::FakeQuantizeOp>(user);
    })};
    if (hasAnyFQUser) {
        return mlir::failure();
    }
    // At this step Matmuls are converted to Convolutions, ops which were originally convolutions match FQ ->
    // Convolution pattern
    auto maybeFQFilter = checkOp<IE::ConvolutionOp, IE::FakeQuantizeOp>(convOp, getConvWeights, checkFQ);

    if (maybeFQFilter == nullptr) {
        return mlir::failure();
    }
    auto oLoConst = maybeFQFilter.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto oHiConst = maybeFQFilter.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    const auto quantizedElemType = getQuantizedElementTypeFromFakeQuantize(maybeFQFilter);
    const auto [scales, zeroPoints] = extractScalesAndZeroPoints(quantizedElemType);
    auto context = convOp->getContext();
    auto [int8Min, int8Max, int8Type] = getStorageParams(context, 256, true);
    const double diff = (zeroPoints.front() + int8Min) * scales.front();
    auto insertionPointBackup = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(oHiConst);
    auto newOutLoContent = oLoConst.transformContentAttr().add(diff).get();
    auto newLoInput =
            rewriter.create<Const::DeclareOp>(oLoConst.getLoc(), oLoConst.getType(), std::move(newOutLoContent));
    auto newOutHiContent = oHiConst.transformContentAttr().add(diff).get();
    auto newHiInput =
            rewriter.create<Const::DeclareOp>(oHiConst.getLoc(), oHiConst.getType(), std::move(newOutHiContent));

    rewriter.replaceOp(oLoConst, newLoInput);
    rewriter.replaceOp(oHiConst, newHiInput);
    rewriter.restoreInsertionPoint(insertionPointBackup);
    // upper part is right conv weights
    const auto convInput = convOp.getInput();

    const auto origConvClone = rewriter.clone(*convOp);
    auto fakeQuantizeOutput = convOp.getFilter();
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(fakeQuantizeOutput.getType());
    auto originalWeightValue = -diff;
    // Scales cannot be negative, therefore the weight values must be set according to the sign of the original value.
    auto weightValue = originalWeightValue < 0 ? -1.0f : 1.0f;
    SmallVector<float> weightsForNewConvolution(filterType.getNumElements(), weightValue);
    const auto filterShape =
            mlir::RankedTensorType::get(getShape(fakeQuantizeOutput), mlir::Float16Type::get(rewriter.getContext()));

    // Quantization below is necessary to reduce footprint of weights by half.
    auto scale = std::fabs(originalWeightValue);
    auto quantType = mlir::quant::UniformQuantizedType::get(
            mlir::quant::QuantizationFlags::Signed, getSInt8Type(context), mlir::Float16Type::get(context), scale,
            /*zp=*/0, /*min=*/static_cast<int64_t>(int8Min),
            /*max=*/static_cast<int64_t>(int8Max));
    const auto denseElementVal = Const::createConstContent<float>(filterShape, weightsForNewConvolution);
    const auto contentAttr = Const::ContentAttr::get(denseElementVal);
    auto quantWeightConstAttr = contentAttr.transform().castElemType(quantType).get();
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(contentAttr.getType()).changeElemType(quantType);
    auto quantizedFilter = rewriter.create<Const::DeclareOp>(convOp.getLoc(), weightsType, quantWeightConstAttr);
    const auto newConv = rewriter.create<IE::ConvolutionOp>(
            appendLoc(convOp.getLoc(), "new_conv"), convInput, quantizedFilter.getOutput(),
            /*bias=*/nullptr, convOp.getStridesAttr(), convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(),
            convOp.getDilationsAttr(),
            /*postOp=*/nullptr,
            /*clamp=*/nullptr,
            /*static_scale=*/nullptr,
            /*outputPadding=*/nullptr,
            /*inputPadding=*/nullptr);
    auto sub = rewriter.create<IE::AddOp>(
            appendLoc(convOp.getLoc(), "new_add"), origConvClone->getResult(0), newConv->getResult(0),
            IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT), nullptr,
            nullptr, nullptr, nullptr);
    rewriter.replaceOp(convOp, sub.getOutput());
    return mlir::success();
}

class ProcessAsymmetricZeroPointsForConvolutionPass final :
        public IE::impl::ProcessAsymmetricZeroPointsForConvolutionBase<ProcessAsymmetricZeroPointsForConvolutionPass> {
public:
    explicit ProcessAsymmetricZeroPointsForConvolutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ProcessAsymmetricZeroPointsForConvolutionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ZeroPointWithConvolution>(&ctx, _log);
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createProcessAsymmetricZeroPointsForConvolutionPass(Logger log) {
    return std::make_unique<ProcessAsymmetricZeroPointsForConvolutionPass>(log);
}
