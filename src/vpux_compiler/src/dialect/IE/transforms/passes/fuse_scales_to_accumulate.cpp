//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESCALESTOACCUMULATE
#define GEN_PASS_DEF_FUSESCALESTOACCUMULATE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class FuseScalesToAccumulate final : public mlir::OpRewritePattern<IE::AccumulateOp> {
public:
    FuseScalesToAccumulate(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AccumulateOp>(ctx), _log(log) {
        setDebugName("FuseScalesToAccumulate");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AccumulateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkFullyConnected(IE::FullyConnectedOp fcOp) const;
    bool checkTranspose(IE::TransposeOp transpose) const;
    bool checkReshape(IE::ReshapeOp reshape) const;
    bool checkFakeQuantize(IE::FakeQuantizeOp fqOp) const;
    IE::FakeQuantizeOp findFakeQuantize(mlir::Value input) const;
    std::optional<SmallVector<float>> fetchScales(IE::FakeQuantizeOp fqOp, size_t numElements,
                                                  mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

bool FuseScalesToAccumulate::checkFullyConnected(IE::FullyConnectedOp fcOp) const {
    if (fcOp == nullptr) {
        return false;
    }
    const auto lhsShape = getShape(fcOp.getInput());
    const auto rhsShape = getShape(fcOp.getWeights());
    return lhsShape.size() == 2 && rhsShape.size() == 2;
}

bool FuseScalesToAccumulate::checkTranspose(IE::TransposeOp transpose) const {
    if (transpose == nullptr) {
        return false;
    }
    const auto inShape = getShape(transpose.getInput());
    const auto outShape = getShape(transpose.getOutput());
    const auto expectedShape = Shape{inShape[Dim(1)], inShape[Dim(0)]};
    return outShape == expectedShape;
}

bool FuseScalesToAccumulate::checkReshape(IE::ReshapeOp reshape) const {
    if (reshape == nullptr) {
        return false;
    }
    const auto inShape = getShape(reshape.getInput());
    const auto outShape = getShape(reshape.getOutput());
    return inShape.size() == 3 && outShape.size() == 2 &&
           outShape == Shape{inShape[Dim(0)] * inShape[Dim(1)], inShape[Dim(2)]};
}

bool FuseScalesToAccumulate::checkFakeQuantize(IE::FakeQuantizeOp fqOp) const {
    if (fqOp == nullptr) {
        return false;
    }
    const auto inLowShape = getShape(fqOp.getInputLow());
    if (inLowShape != Shape{1, 1, 1}) {
        return false;
    }
    const auto inHighShape = getShape(fqOp.getInputHigh());
    if (inHighShape != Shape{1, 1, 1}) {
        return false;
    }

    const auto outLow = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    if (outLow == nullptr) {
        return false;
    }
    const auto outHigh = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (outHigh == nullptr) {
        return false;
    }

    const auto inShape = getShape(fqOp.getInput());
    const auto outLowShape = getShape(fqOp.getOutputLow());
    const auto outHighShape = getShape(fqOp.getOutputHigh());
    const auto expectedShape1x1xC = Shape{1, 1, inShape[Dim(2)]};
    const auto expectedShapeCx1x1 = Shape{inShape[Dim(0)], 1, 1};
    return (outLowShape == expectedShape1x1xC || outLowShape == expectedShapeCx1x1) && outHighShape == outLowShape;
}

IE::FakeQuantizeOp FuseScalesToAccumulate::findFakeQuantize(mlir::Value input) const {
    // IE.FakeQuantize -> IE.Reshape -> IE.Transpose -> [IE.FullyConnected] -> IE.Accumulate
    if (input == nullptr) {
        return nullptr;
    }
    auto fcOp = input.getDefiningOp<IE::FullyConnectedOp>();
    if (!checkFullyConnected(fcOp)) {
        return nullptr;
    }
    auto rhsFCInput = fcOp.getWeights();
    if (rhsFCInput == nullptr) {
        return nullptr;
    }
    // IE.FakeQuantize -> IE.Reshape -> [IE.Transpose] -> IE.FullyConnected -> IE.Accumulate
    auto transposeOp = rhsFCInput.getDefiningOp<IE::TransposeOp>();
    IE::ReshapeOp reshapeOp{nullptr};
    if (!checkTranspose(transposeOp)) {
        reshapeOp = rhsFCInput.getDefiningOp<IE::ReshapeOp>();
    } else {
        auto transposeInput = transposeOp.getInput();
        if (transposeInput == nullptr) {
            return nullptr;
        }
        reshapeOp = transposeInput.getDefiningOp<IE::ReshapeOp>();
    }
    // IE.FakeQuantize -> [IE.Reshape] -> IE.Transpose -> IE.FullyConnected -> IE.Accumulate
    // IE.Transpose may be optional:
    // IE.FakeQuantize -> [IE.Reshape] -> IE.FullyConnected -> IE.Accumulate
    if (!checkReshape(reshapeOp)) {
        return nullptr;
    }
    auto reshapeInput = reshapeOp.getInput();
    if (reshapeInput == nullptr) {
        return nullptr;
    }

    // [IE.FakeQuantize] -> IE.Reshape -> IE.Transpose -> IE.FullyConnected -> IE.Accumulate
    auto fqOp = reshapeInput.getDefiningOp<IE::FakeQuantizeOp>();
    if (!checkFakeQuantize(fqOp)) {
        return nullptr;
    }
    return fqOp;
}

std::optional<SmallVector<float>> FuseScalesToAccumulate::fetchScales(IE::FakeQuantizeOp fqOp, size_t numElements,
                                                                      mlir::PatternRewriter& rewriter) const {
    if (fqOp == nullptr) {
        return SmallVector<float>(numElements, 1.f);
    }

    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    const auto realType = fqOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto levels = fqOp.getLevels();
    if (!levels.has_value()) {
        // Currently there is no need for this pass to treat non-integer quantization cases.
        return std::nullopt;
    }

    const auto outQuantizeElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), levels,
                             /*lowFpType=*/std::nullopt, realElemType,
                             /*isSigned=*/false, fqOp.getLoc(), fqOp.getAutoBroadcast());

    if (outQuantizeElemType == nullptr) {
        return std::nullopt;
    }

    auto perAxis = outQuantizeElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    if (perAxis == nullptr) {
        return std::nullopt;
    }
    const auto f64Scales = perAxis.getScales();
    SmallVector<float> f32Scales;
    const auto toF32 = [](const double val) -> float {
        return val;
    };
    std::transform(f64Scales.begin(), f64Scales.end(), std::back_inserter(f32Scales), toF32);

    const auto lowContent = outLowConst.getContentAttr().fold();
    SmallVector<float> lowVals = lowContent.getValues<float>();
    const auto highContent = outHighConst.getContentAttr().fold();
    SmallVector<float> highVals = highContent.getValues<float>();
    VPUX_THROW_UNLESS(lowVals.size() == highVals.size(), "Sizes of low values and high values must match.");
    for (const auto& idx : irange(f32Scales.size())) {
        const auto scale = isFloatEqual(f32Scales[idx], 0.f) ? 1.f : f32Scales[idx];
        lowVals[idx] = lowVals[idx] / scale;
        highVals[idx] = highVals[idx] / scale;
    }

    const bool usesFP16 = mlir::isa<mlir::Float16Type>(mlir::cast<NDTypeInterface>(realType).getElementType());
    auto adjustDType = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
        if (usesFP16) {
            return setup.castElemType(mlir::Float16Type::get(rewriter.getContext()));
        }

        return std::move(setup);
    };

    const auto scaleShape =
            mlir::RankedTensorType::get(getShape(fqOp.getOutputLow()), mlir::Float32Type::get(rewriter.getContext()));
    auto newLowVal = Const::createConst(rewriter, outLowConst.getLoc(), scaleShape, ArrayRef(lowVals), adjustDType);
    auto newHighVal = Const::createConst(rewriter, outHighConst.getLoc(), scaleShape, ArrayRef(highVals), adjustDType);

    outLowConst.getOutput().replaceAllUsesWith(newLowVal);
    outHighConst.getOutput().replaceAllUsesWith(newHighVal);

    return f32Scales;
}

mlir::LogicalResult FuseScalesToAccumulate::matchAndRewrite(IE::AccumulateOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (origOp.getLhsScale() != nullptr || origOp.getRhsScale() != nullptr) {
        return matchFailed(rewriter, origOp, "IE.Accumulate at {0} already has scales.", origOp->getLoc());
    }
    const auto lhsFq = findFakeQuantize(origOp.getLhs());
    const auto rhsFq = findFakeQuantize(origOp.getRhs());
    if (lhsFq == nullptr && rhsFq == nullptr) {
        return matchFailed(rewriter, origOp, "IE.Accumulate at {0} is not supported.", origOp->getLoc());
    }
    const auto shape = getShape(origOp.getOutput());
    const auto lhsScales = fetchScales(lhsFq, checked_cast<size_t>(shape.back()), rewriter);
    if (!lhsScales.has_value()) {
        return matchFailed(rewriter, origOp, "Failed to fetch scales for the first input.", origOp->getLoc());
    }
    const auto rhsScales = fetchScales(rhsFq, checked_cast<size_t>(shape.back()), rewriter);
    if (!rhsScales.has_value()) {
        return matchFailed(rewriter, origOp, "Failed to fetch scales for the second input.", origOp->getLoc());
    }

    const bool usesFP16 =
            mlir::isa<mlir::Float16Type>(mlir::cast<NDTypeInterface>(origOp.getLhs().getType()).getElementType());
    auto adjustDType = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
        if (usesFP16) {
            return setup.castElemType(mlir::Float16Type::get(rewriter.getContext()));
        }

        return std::move(setup);
    };

    const auto scaleType = mlir::RankedTensorType::get({shape.back()}, mlir::Float32Type::get(rewriter.getContext()));
    const auto lhsScaleVal =
            Const::createConst(rewriter, origOp.getLoc(), scaleType, ArrayRef(lhsScales.value()), adjustDType);
    const auto rhsScaleVal =
            Const::createConst(rewriter, origOp.getLoc(), scaleType, ArrayRef(rhsScales.value()), adjustDType);

    rewriter.replaceOpWithNewOp<IE::AccumulateOp>(origOp, origOp.getLhs(), origOp.getRhs(), lhsScaleVal, rhsScaleVal);

    return mlir::success();
}

class FuseScalesToAccumulatePass final : public IE::impl::FuseScalesToAccumulateBase<FuseScalesToAccumulatePass> {
public:
    explicit FuseScalesToAccumulatePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseScalesToAccumulatePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseScalesToAccumulate>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseScalesToAccumulatePass(Logger log) {
    return std::make_unique<FuseScalesToAccumulatePass>(log);
}
