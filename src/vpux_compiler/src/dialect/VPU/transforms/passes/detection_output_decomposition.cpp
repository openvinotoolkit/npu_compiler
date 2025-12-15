//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>
#include <vpux/utils/core/error.hpp>

namespace vpux::VPU {
#define GEN_PASS_DECL_DETECTIONOUTPUTDECOMPOSITION
#define GEN_PASS_DEF_DETECTIONOUTPUTDECOMPOSITION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// Helper functions
//

int64_t getNumPriors(VPU::DetectionOutputOp origOp) {
    const auto priorBoxSize = origOp.getAttr().getNormalized().getValue() ? 4 : 5;
    const auto numPriors = getShape(origOp.getInProposals())[Dim(2)] / priorBoxSize;
    return static_cast<int64_t>(numPriors);
}

mlir::Value createReshape(mlir::PatternRewriter& rewriter, mlir::Value tensor, ArrayRef<int64_t> newShape) {
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    auto reshape = rewriter.create<VPU::ReshapeOp>(tensor.getLoc(), tensor, nullptr, false, newShapeAttr);
    return reshape.getOutput();
}

mlir::Value reshapeLogitsTo4D(mlir::PatternRewriter& rewriter, mlir::Value input, int64_t numPriors) {
    const auto shape = getShape(input);
    VPUX_THROW_UNLESS(shape.size() == 2, "BoxLogits shape must be 2D");

    const auto batch = shape[Dim(0)];
    const auto width = shape[Dim(1)];
    const auto boxSize = 4;

    VPUX_THROW_UNLESS((width % (boxSize * numPriors)) == 0, "DetectionOutput BoxLogits tensor shape {0} is incorrect",
                      shape);

    const auto numLocClasses = width / (boxSize * numPriors);
    const auto newShape = SmallVector<int64_t>{batch, numPriors, numLocClasses, boxSize};
    return createReshape(rewriter, input, newShape);
}

mlir::Value reshapeClassPredictionsTo4D(mlir::PatternRewriter& rewriter, mlir::Value input, int64_t numPriors) {
    const auto shape = getShape(input);
    VPUX_THROW_UNLESS(shape.size() == 2, "ClassPredictions shape must be 2D");

    const auto batch = shape[Dim(0)];
    const auto width = shape[Dim(1)];

    VPUX_THROW_UNLESS(width % numPriors == 0, "DetectionOutput ClassPredictions tensor shape {0} is incorrect", shape);

    const auto numClasses = width / numPriors;
    const auto newShape = SmallVector<int64_t>{batch, 1, numPriors, numClasses};
    return createReshape(rewriter, input, newShape);
}

mlir::Value reshapePriorBoxesTo4D(mlir::PatternRewriter& rewriter, mlir::Value input, int64_t numPriors) {
    const auto shape = getShape(input);
    VPUX_THROW_UNLESS(shape.size() == 3, "PriorBoxes shape must be 3D");

    const auto batch = shape[Dim(0)];
    const auto height = shape[Dim(1)];
    const auto width = shape[Dim(2)];

    VPUX_THROW_UNLESS(width % numPriors == 0, "DetectionOutput PriorBoxes tensor shape {0} is incorrect", shape);
    const auto boxSize = width / numPriors;

    const auto newShape = SmallVector<int64_t>{batch, height, numPriors, boxSize};

    return createReshape(rewriter, input, newShape);
}

mlir::LogicalResult checkSupportedArguments(VPU::DetectionOutputOp origOp) {
    const auto attrs = origOp.getAttr();

    const auto normalized = origOp.getAttr().getNormalized().getValue();
    const auto varianceEncodedInTarget = origOp.getAttr().getVarianceEncodedInTarget().getValue();
    if (!normalized && !varianceEncodedInTarget) {
        return errorAt(origOp,
                       "DetectionOutput: undefined case - normalized == false && varianceEncodedInTarget == false");
    }

    const auto keepTopKArray = origOp.getAttr().getKeepTopK().getValue();
    if (keepTopKArray.size() != 1) {
        return errorAt(origOp, "DetectionOutput: keepTopK array size={0} is not supported", keepTopKArray.size());
    }

    const auto hasAdditionalInputs = (origOp->getNumOperands() > 3);
    if (hasAdditionalInputs) {
        return errorAt(origOp, "DetectionOutput: only 3 inputs is supported");
    }

    const auto codeType = attrs.getCodeType().getValue();
    if (codeType != IE::DetectionOutputCodeType::CENTER_SIZE) {
        return errorAt(origOp, "DetectionOutput: only CodeType::CENTER_SIZE is supported");
    }

    const auto mxNetNms = (attrs.getDecreaseLabelId().getValue() == true);
    if (mxNetNms) {
        return errorAt(origOp, "DetectionOutput: decreaseLabelId == true is not supported");
    }

    return mlir::success();
}

mlir::Value transposeClassPredictions(mlir::PatternRewriter& rewriter, mlir::Value tensor) {
    const auto ctx = rewriter.getContext();

    // keep default 4D order because this is the order that will be used throughout.
    // pretend that the data was never reordered in the first place.
    const auto defaultOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));  //              \/--swap--\/
    const auto permutationMap =
            mlir::AffineMapAttr::get(DimsOrder::NCWH.toAffineMap(ctx));  // [Batch, 1, Priors, Classes]

    return rewriter.create<VPU::MemPermuteOp>(tensor.getLoc(), tensor, defaultOrder, permutationMap);
}

mlir::Value transposeBoxes(mlir::PatternRewriter& rewriter, mlir::Value boxes4D) {
    const auto ctx = rewriter.getContext();

    // keep default 4D order because this is the order that will be used throughout
    // pretend that the data was never reordered in the first place
    const auto defaultOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));  //           \/--swap--\/
    const auto permutationMap =
            mlir::AffineMapAttr::get(DimsOrder::NHCW.toAffineMap(ctx));  // [Batch, Priors, Classes, 4 or 5]

    return rewriter.create<VPU::MemPermuteOp>(boxes4D.getLoc(), boxes4D, defaultOrder, permutationMap);
}

mlir::Value cutOnWidth(mlir::PatternRewriter& rewriter, mlir::Value tensor, int64_t width) {
    auto slicedShape = Shape(getShape(tensor));
    slicedShape.back() = width;
    const auto offsets = Shape(slicedShape.size());

    auto sliceOp = rewriter.create<VPU::SliceOp>(tensor.getLoc(), tensor, offsets, slicedShape);
    return sliceOp.getResult();
}

//
// DetectionOutputDecompositionPass
//

class DetectionOutputDecomposition final : public mlir::OpRewritePattern<VPU::DetectionOutputOp> {
public:
    DetectionOutputDecomposition(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::DetectionOutputOp>(ctx), _log(std::move(log)) {
        setDebugName("DetectionOutputDecomposition");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DetectionOutputOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DetectionOutputDecomposition::matchAndRewrite(VPU::DetectionOutputOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto loc = origOp->getLoc();
    const auto supported = checkSupportedArguments(origOp);
    if (!mlir::succeeded(supported)) {
        return supported;
    }

    const auto numPriors = getNumPriors(origOp);
    auto priors4D = reshapePriorBoxesTo4D(rewriter, origOp.getInProposals(), numPriors);

    const auto normalized = origOp.getAttr().getNormalized().getValue();
    if (!normalized) {
        const auto inputWidth = origOp.getAttr().getInputWidth().getInt();
        const auto inputHeight = origOp.getAttr().getInputHeight().getInt();
        priors4D =
                rewriter.create<VPU::DetectionOutputNormalizeOp>(origOp->getLoc(), priors4D, inputWidth, inputHeight);
    }

    const auto boxLogits = reshapeLogitsTo4D(rewriter, origOp.getInBoxLogits(), numPriors);
    const auto boxLogitsTransposed = transposeBoxes(rewriter, boxLogits);

    const auto codeType = origOp.getAttr().getCodeType().getValue();
    const auto clipBeforeNms = origOp.getAttr().getClipBeforeNms().getValue();
    auto decodeBoxes = rewriter.create<VPU::DetectionOutputDecodeBoxesOp>(
            appendLoc(loc, "decodeBoxes"), boxLogitsTransposed, priors4D, codeType, clipBeforeNms);

    const auto classPredictions = reshapeClassPredictionsTo4D(rewriter, origOp.getInClassPreds(), numPriors);
    const auto transposedClassPredictions = transposeClassPredictions(rewriter, classPredictions);
    const auto confidenceThreshold = origOp.getAttr().getConfidenceThreshold();

    // In rare cases the number of available detections per class is smaller than TopK attribute
    // Choose the smallest value between TopK and NumPriors
    const auto detectionsPerClass = [&]() -> int64_t {
        const auto topK = origOp.getAttr().getTopK();
        const auto topKValue = checked_cast<int64_t>(topK.getValue().getSExtValue());
        return std::min(topKValue, numPriors);
    }();
    const auto detectionsPerClassAttr = getIntAttr(rewriter, detectionsPerClass);

    auto sort = rewriter.create<VPU::DetectionOutputSortOp>(appendLoc(loc, "sort"), transposedClassPredictions,
                                                            confidenceThreshold, detectionsPerClassAttr);

    auto confidenceSlice = cutOnWidth(rewriter, sort.getOutConfidence(), detectionsPerClass);
    auto indicesSlice = cutOnWidth(rewriter, sort.getOutIndices(), detectionsPerClass);

    const auto sizesShape = getShape(sort.getOutSizes());
    const auto newSizesShape = SmallVector<int64_t>{sizesShape[Dims4D::Act::N], sizesShape[Dims4D::Act::C],
                                                    sizesShape[Dims4D::Act::W], sizesShape[Dims4D::Act::H]};
    const auto reshapedSizes = createReshape(rewriter, sort.getOutSizes(), newSizesShape);

    const auto nmsThreshold = origOp.getAttr().getNmsThreshold();
    const auto backgroundId = origOp.getAttr().getBackgroundLabelId();
    auto nmsCaffe = rewriter.create<VPU::DetectionOutputNmsCaffeOp>(
            appendLoc(loc, "nmsCaffe"), confidenceSlice, decodeBoxes.getOutDecodedBoxes(), indicesSlice, reshapedSizes,
            detectionsPerClassAttr, nmsThreshold, backgroundId);

    const auto keepTopKValue = mlir::cast<mlir::IntegerAttr>(origOp.getAttr().getKeepTopK()[0]).getInt();
    const auto keepTopK = getIntAttr(rewriter.getContext(), keepTopKValue);
    const auto clipAfterNms = origOp.getAttr().getClipAfterNms();
    rewriter.replaceOpWithNewOp<VPU::DetectionOutputCollectResultsOp>(origOp, nmsCaffe.getOutConfidence(),
                                                                      nmsCaffe.getOutBoxes(), nmsCaffe.getOutSizes(),
                                                                      keepTopK, clipAfterNms);

    return mlir::success();
}

class DetectionOutputDecompositionPass final :
        public VPU::impl::DetectionOutputDecompositionBase<DetectionOutputDecompositionPass> {
public:
    explicit DetectionOutputDecompositionPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DetectionOutputDecompositionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::DetectionOutputOp>();
    target.addLegalOp<VPU::DetectionOutputNormalizeOp>();
    target.addLegalOp<VPU::DetectionOutputDecodeBoxesOp>();
    target.addLegalOp<VPU::DetectionOutputSortOp>();
    target.addLegalOp<VPU::DetectionOutputNmsCaffeOp>();
    target.addLegalOp<VPU::DetectionOutputCollectResultsOp>();

    target.addLegalOp<VPU::ReshapeOp>();
    target.addLegalOp<VPU::MemPermuteOp>();
    target.addLegalOp<VPU::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();  // for auxiliary buffer

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DetectionOutputDecomposition>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDetectionOutputDecompositionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDetectionOutputDecompositionPass(Logger log) {
    return std::make_unique<DetectionOutputDecompositionPass>(log);
}
