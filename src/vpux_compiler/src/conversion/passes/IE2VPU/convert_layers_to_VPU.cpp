//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_layers_to_VPU.hpp"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/type/float16.hpp"

// Generated
#include <type_traits>
#include <vpux/compiler/conversion/convert_layers_to_VPU.hpp.inc>

namespace vpux {
#define GEN_PASS_DECL_CONVERTLAYERS2VPU
#define GEN_PASS_DEF_CONVERTLAYERS2VPU
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

//
// IfRewrite
//

mlir::LogicalResult IfRewrite::matchAndRewrite(IE::IfOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Found If Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::IRMapping mapper;
    auto thenBlock = &origOp.getThenBranch().getBlocks().front();
    auto elseBlock = &origOp.getElseBranch().getBlocks().front();

    for (auto valueIt : llvm::enumerate(thenBlock->getArguments())) {
        auto blockArg = origOp.getInputs()[valueIt.index()];
        mapper.map(valueIt.value(), blockArg);
    }
    for (auto valueIt : llvm::enumerate(elseBlock->getArguments())) {
        auto blockArg = origOp.getInputs()[valueIt.index()];
        mapper.map(valueIt.value(), blockArg);
    }

    // Then branch construct
    SmallVector<mlir::Value> thenBranchResults;
    SmallVector<mlir::Type> outTypes;
    for (auto& op : origOp.getThenBranch().getOps()) {
        mlir::Operation* newOp = rewriter.clone(op, mapper);
        if (mlir::isa<IE::YieldOp>(op)) {
            for (mlir::Value operand : newOp->getOperands()) {
                thenBranchResults.push_back(operand);
                outTypes.push_back(operand.getType());
            }
            rewriter.eraseOp(newOp);
            continue;
        }
        for (const auto& [result, newResult] : zip(op.getResults(), newOp->getResults())) {
            mapper.map(result, newResult);
        }
    }

    // Else branch construct
    SmallVector<mlir::Value> elseBranchResults;
    for (auto& op : origOp.getElseBranch().getOps()) {
        mlir::Operation* newOp = rewriter.clone(op, mapper);
        if (mlir::isa<IE::YieldOp>(op)) {
            for (mlir::Value operand : newOp->getOperands()) {
                elseBranchResults.push_back(operand);
            }
            rewriter.eraseOp(newOp);
            continue;
        }
        for (const auto& [result, newResult] : zip(op.getResults(), newOp->getResults())) {
            mapper.map(result, newResult);
        }
    }

    auto cond = origOp.getCond();
    SmallVector<mlir::Value> branchResults;
    int64_t numInputs = thenBranchResults.size();
    for (auto i = 0; i < numInputs; i++) {
        auto result = rewriter.create<VPU::ConditionalCopyOp>(origOp.getLoc(), outTypes[i], cond, thenBranchResults[i],
                                                              elseBranchResults[i]);
        branchResults.push_back(result);
    }
    rewriter.replaceOp(origOp, branchResults);

    return mlir::success();
}

//
// CTCGreedyDecoderSeqLenRewrite
//

mlir::LogicalResult CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoderSeqLen Operation '{0}'", origOp->getLoc());

    mlir::Value blankIndexValue = origOp.getBlankIndex();
    if (blankIndexValue == nullptr) {
        // Default value is C-1
        auto* ctx = origOp->getContext();
        const auto inShape = getShape(origOp.getInput()).raw();

        if (inShape.size() != 3) {
            return errorAt(origOp.getLoc(), "ConvertLayers2VPU::CTCGreedyDecoderSeqLenRewrite: First input tensor "
                                            "should have 3 dimensions: [N, T, C]");
        }
        auto blankIndxDefValue = checked_cast<int32_t>(inShape.back() - 1);
        auto blankIndxShape = mlir::RankedTensorType::get(
                {1}, mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signed));
        blankIndexValue = Const::createConst(rewriter, origOp.getLoc(), blankIndxShape, ArrayRef(blankIndxDefValue));
    }
    rewriter.replaceOpWithNewOp<VPU::CTCGreedyDecoderSeqLenOp>(origOp, origOp.getInput(), origOp.getSequenceLength(),
                                                               blankIndexValue, origOp.getMergeRepeatedAttr());
    return mlir::success();
}

//
// ProposalRewrite
//

mlir::LogicalResult ProposalRewrite::matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Proposal Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::ProposalOp>(origOp, origOp.getClassProbs(), origOp.getBboxDeltas(),
                                                 origOp.getImageShape(), origOp.getProposalAttrsAttr());
    _log.trace("Replaced with 'VPU.ProposalOp'");

    return mlir::success();
}

//
// SplitRewrite
//

mlir::LogicalResult SplitRewrite::matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::SplitOp>(origOp, origOp.getInput(), origOp.getAxis(), origOp.getNumSplitsAttr(),
                                              origOp.getAxisValueAttr());

    return mlir::success();
}

mlir::LogicalResult StubRewrite::matchAndRewrite(IE::StubOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Found Stub Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::StubOp>(origOp, origOp.getOutputs().getTypes(), origOp.getInputs());

    return mlir::success();
}

//
// NonMaxSuppressionRewrite
//

mlir::LogicalResult NonMaxSuppressionRewrite::matchAndRewrite(IE::NonMaxSuppressionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found NonMaxSuppression Operation '{0}'", origOp->getLoc());

    // The NMS act-shave kernel reads the iou/score thresholds from scalar input tensors. Always
    // provide both as operands: pass through the runtime operand when it is present, otherwise
    // materialize a single-element constant tensor from the corresponding threshold attribute.
    const auto scoresElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInBoxScores().getType()).getElementType();
    const auto thresholdType = mlir::RankedTensorType::get({1}, scoresElemType);

    const auto materializeThreshold = [&](mlir::Value runtimeOperand, mlir::FloatAttr valueAttr,
                                          StringRef name) -> mlir::Value {
        if (runtimeOperand != nullptr) {
            return runtimeOperand;
        }
        const auto value = static_cast<float>(valueAttr != nullptr ? valueAttr.getValueAsDouble() : 0.0);
        return Const::createFloatConst(rewriter, appendLoc(origOp->getLoc(), name), thresholdType,
                                       ArrayRef<float>{value});
    };

    auto iouThreshold =
            materializeThreshold(origOp.getIouThreshold(), origOp.getIouThresholdValueAttr(), "nms_iou_threshold");
    auto scoreThreshold = materializeThreshold(origOp.getScoreThreshold(), origOp.getScoreThresholdValueAttr(),
                                               "nms_score_threshold");

    rewriter.replaceOpWithNewOp<VPU::NonMaxSuppressionOp>(
            origOp, origOp.getInBoxCoords(), origOp.getInBoxScores(), iouThreshold, scoreThreshold,
            origOp.getBoxEncodingAttr(), origOp.getSortResultDescendingAttr(),
            origOp.getMaxOutputBoxesPerClassValueAttr(), origOp.getIouThresholdValueAttr(),
            origOp.getScoreThresholdValueAttr(), origOp.getSoftNmsSigmaValueAttr());

    _log.trace("Replaced with 'VPU.NonMaxSuppressionOp'");

    return mlir::success();
}

//
// ExperimentalDetectronROIFeatureExtractorRewrite
//

mlir::LogicalResult ExperimentalDetectronROIFeatureExtractorRewrite::matchAndRewrite(
        IE::ExperimentalDetectronROIFeatureExtractorOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found ExperimentalDetectronROIFeatureExtractor Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::ExperimentalDetectronROIFeatureExtractorOp>(origOp, origOp.getInputs(),
                                                                                 origOp.getAttrAttr());

    _log.trace("Replaced with 'VPU.ExperimentalDetectronROIFeatureExtractorOp'");

    return mlir::success();
}

//
// GRUCellRewrite
//

mlir::LogicalResult GRUCellRewrite::matchAndRewrite(IE::GRUCellOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found GRUCell Operation '{0}'", origOp->getLoc());

    auto* ctx = origOp->getContext();
    const auto inputShape = getShape(origOp.getInputData()).raw();
    const auto batchSize = inputShape[0];
    const auto inputSize = inputShape[1];
    SmallVector<int64_t> newInputShape = {batchSize, 1, inputSize};
    const auto newInputShapeAttr = getIntArrayAttr(ctx, newInputShape);
    auto newInput = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getInputData(), newInputShapeAttr);

    const auto initialStateShape = getShape(origOp.getInitialHiddenState()).raw();
    const auto hiddenSize = initialStateShape[1];
    SmallVector<int64_t> newInitialStateShape = {batchSize, 1, hiddenSize};
    const auto newInitialStateShapeAttr = getIntArrayAttr(ctx, newInitialStateShape);
    auto newInitialState =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getInitialHiddenState(), newInitialStateShapeAttr);

    SmallVector<int64_t> newWeightsShape = {1, 3 * hiddenSize, inputSize};
    const auto newWeightsShapeAttr = getIntArrayAttr(ctx, newWeightsShape);
    auto newWeights = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getWeights(), newWeightsShapeAttr);

    SmallVector<int64_t> newReWeightsShape = {1, 3 * hiddenSize, hiddenSize};
    const auto newReWeightsShapeAttr = getIntArrayAttr(ctx, newReWeightsShape);
    auto newReWeights =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getRecurrenceWeights(), newReWeightsShapeAttr);

    const auto biasesShape = getShape(origOp.getBiases()).raw();
    SmallVector<int64_t> newBiasesShape = {1, biasesShape[0]};
    const auto newBiasesShapeAttr = getIntArrayAttr(ctx, newBiasesShape);
    auto newBiases = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getBiases(), newBiasesShapeAttr);

    const auto seqLenAttr = getIntAttr(ctx, 1);
    const auto directionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD);

    auto gruSeq =
            rewriter.create<VPU::GRUSequenceOp>(origOp->getLoc(), newInput, newInitialState, newWeights, newReWeights,
                                                newBiases, origOp.getHiddenSizeAttr(), seqLenAttr, directionAttr,
                                                origOp.getShouldLinearBeforeResetAttr(), origOp.getClipAttr());
    SmallVector<int64_t> newOutputShape = {batchSize, hiddenSize};
    const auto newOutputShapeAttr = getIntArrayAttr(ctx, newOutputShape);
    rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(origOp, gruSeq.getOutputHiddenState(), newOutputShapeAttr);

    return mlir::success();
}

//
// InterpolateRewrite
//

mlir::LogicalResult InterpolateRewrite::matchAndRewrite(IE::InterpolateOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    // Scale-as-parameter path: lower to VPU::InterpolateDMAOp.
    // InterpolateDMA has no non-DMA fallback (shave-writes-to-DDR is disabled), so any arch that
    // accepts a scales-as-parameter Interpolate must register IE::LayerWithDmaInterface for it.
    if (IE::isScalesAsParameter(origOp.getScales(), origOp.getScalesAttr())) {
        _log.trace("Found Interpolate with scales as parameter '{0}'", origOp->getLoc());

        auto opWithDma = mlir::dyn_cast<IE::LayerWithDmaInterface>(origOp.getOperation());
        VPUX_THROW_UNLESS(opWithDma, "Interpolate with scales-as-parameter requires IE::LayerWithDmaInterface");
        VPUX_THROW_UNLESS(opWithDma.isSupported(),
                          "Interpolate with scales-as-parameter is unsupported for the current arch/mode "
                          "combination: InterpolateDMA is not supported");

        const auto outputType = origOp.getOutput().getType();
        auto module = origOp->getParentOfType<mlir::ModuleOp>();

        // Default kernel CMX workspace; clamp to fragmentation-aware CMX so it never exceeds what
        // the scheduler can safely give on the current arch.
        constexpr int64_t kerWszBytes = (1024 + 256) * 1024;
        const int64_t fragAwareBytes = VPU::getTotalCMXFragmentationAwareSize(module).count();
        const int64_t auxBytes = std::min(kerWszBytes, fragAwareBytes);
        _log.info("InterpolateDMA aux buffer: fragAware={0}B, kerWsz={1}B, aux={2}B at '{3}'", fragAwareBytes,
                  kerWszBytes, auxBytes, origOp->getLoc());
        auto auxType = mlir::RankedTensorType::get({1, 1, 1, auxBytes}, getUInt8Type(rewriter.getContext()));
        mlir::Value auxBuffer = VPU::createEmptyAuxiliaryBuffer(rewriter, origOp->getLoc(), auxType);

        rewriter.replaceOpWithNewOp<VPU::InterpolateDMAOp>(origOp, outputType, origOp.getInput(), origOp.getScales(),
                                                           /*aux_buffer=*/auxBuffer, origOp.getAxesAttrAttr(),
                                                           origOp.getAttrAttr(),
                                                           /*multiClusterStrategy=*/nullptr);

        return mlir::success();
    }

    rewriter.replaceOpWithNewOp<VPU::InterpolateOp>(
            origOp, origOp.getType(), origOp.getInput(), origOp.getSizes(), origOp.getScales(), origOp.getAxes(),
            /*coordinates*/ nullptr, /* lambdas */ nullptr, origOp.getSizesAttrAttr(), origOp.getScalesAttrAttr(),
            origOp.getAxesAttrAttr(), origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(),
            origOp.getInitialOutputDimsAttrAttr(),
            /*initial_input_offset_attr=*/nullptr, /*initial_output_offset_attr=*/nullptr,
            /*multiClusterStrategy=*/nullptr, origOp.getAttrAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());
    return mlir::success();
}

//
// TopKRewrite
//

mlir::LogicalResult TopKRewrite::matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::TopKOp>(origOp, origOp.getInput(), origOp.getK(), origOp.getKValueAttr(),
                                             origOp.getAxisAttr(), origOp.getModeAttr(), origOp.getSortAttr(),
                                             origOp.getElementTypeAttr(), /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// AtanRewrite
//

mlir::LogicalResult AtanRewrite::matchAndRewrite(IE::AtanOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Atan Operation '{0}'", origOp->getLoc());
    auto input = origOp.getInput();

    auto opWithDma = mlir::dyn_cast<IE::LayerWithDmaInterface>(origOp.getOperation());
    if (opWithDma && opWithDma.isSupported()) {
        rewriter.replaceOpWithNewOp<VPU::AtanDmaOp>(origOp, input);
        return mlir::success();
    }

    rewriter.replaceOpWithNewOp<VPU::AtanOp>(origOp, input);
    return mlir::success();
}

//
// ScatterUpdateRewrite
//

mlir::LogicalResult ScatterUpdateRewrite::matchAndRewrite(IE::ScatterUpdateOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found ScatterUpdate Operation '{0}'", origOp->getLoc());

    auto opWithDma = mlir::dyn_cast<IE::LayerWithDmaInterface>(origOp.getOperation());
    if (opWithDma && opWithDma.isSupported()) {
        rewriter.replaceOpWithNewOp<VPU::ScatterUpdateSwDmaOp>(origOp, origOp.getInput(), origOp.getIndices(),
                                                               origOp.getUpdates(), origOp.getAxisValueAttr());
        return mlir::success();
    }

    rewriter.replaceOpWithNewOp<VPU::ScatterUpdateOp>(origOp, origOp.getInput(), origOp.getIndices(),
                                                      origOp.getUpdates(), origOp.getAxisValueAttr());
    return mlir::success();
}

//
// MaxPool8Rewrite
//

mlir::LogicalResult MaxPool8Rewrite::matchAndRewrite(IE::MaxPool8Op origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found MaxPool8 Operation '{0}'", origOp->getLoc());

    auto* ctx = origOp->getContext();
    auto iShape = getShape(origOp.getInput());
    auto oShape = getShape(origOp.getOutput());
    auto initialInputDimsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>(iShape.begin(), iShape.end()));
    auto initialOutputDimsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>(oShape.begin(), oShape.end()));

    rewriter.replaceOpWithNewOp<VPU::MaxPool8Op>(
            origOp, origOp.getInput(), origOp.getKernelSizeAttr(), origOp.getStridesAttr(), origOp.getDilationsAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getRoundingTypeAttr(),
            origOp.getIndexElementTypeAttr(), origOp.getAxisAttr(), initialInputDimsAttr, initialOutputDimsAttr,
            /*initial_input_offset_attr=*/nullptr, /*initial_output_offset_attr=*/nullptr,
            /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// TransposedConvRewrite
//

mlir::LogicalResult TransposedConvRewrite::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found TransposedConvolution Operation '{0}'", origOp->getLoc());

    auto outType = origOp.getOutput().getType();

    rewriter.replaceOpWithNewOp<VPU::TransposedConvolutionOp>(
            origOp, outType, origOp.getInput(), origOp.getFilter(), origOp.getOutputShape(), origOp.getBias(),
            origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getSpatialOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    return mlir::success();
}

//
// NormalizeL2Rewrite
//

mlir::LogicalResult NormalizeL2Rewrite::matchAndRewrite(IE::NormalizeL2Op origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found NormalizeL2 Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::NormalizeL2Op>(origOp, origOp.getData(), origOp.getAxesValueAttr(),
                                                    origOp.getEpsAttr(), origOp.getEpsModeAttr(),
                                                    /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// LSTMCellRewrite
//

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(IE::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto weights = origOp.getWeights();
    const auto biases = origOp.getBiases();
    if (!weights || !biases) {
        return matchFailed(rewriter, origOp,
                           "VPU::LSTMCell does not support missing weights or biases; it should have been decomposed "
                           "by the DecomposeLSTMCellPass.");
    }

    rewriter.replaceOpWithNewOp<VPU::LSTMCellOp>(origOp, origOp.getInputData(), origOp.getInitialHiddenState(),
                                                 origOp.getInitialCellState(), weights, origOp.getRecurrenceWeights(),
                                                 biases, origOp.getHiddenSizeAttr());
    return mlir::success();
}

//
// LSTMSequenceRewrite
//

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(IE::LSTMSequenceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    const auto weights = origOp.getWeights();
    if (weights) {
        return matchFailed(rewriter, origOp,
                           "VPU::LSTMSequence does not support weights; it should have been decomposed by "
                           "the DecomposeLSTMSequencePass.");
    }
    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::LSTMSequenceOp>(
            origOp, origOp.getInputData(), origOp.getInitialHiddenState(), origOp.getInitialCellState(),
            origOp.getSequenceLengthData(), origOp.getRecurrenceWeights(), origOp.getBiases(),
            origOp.getSequenceLengthAttr(), origOp.getDirectionAttr(), /*initial_output_offset_attr=*/nullptr,
            /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// LSTMGatesRewrite
//

mlir::LogicalResult LSTMGatesRewrite::matchAndRewrite(IE::LSTMGatesOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LSTMGates Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::LSTMGatesOp>(origOp, origOp.getGatesInput(), origOp.getInitialCellState(),
                                                  /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// GroupConvolutionRewrite
//

mlir::LogicalResult GroupConvolutionRewrite::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found GroupConvolutionRewrite Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::GroupConvolutionOp>(
            origOp, origOp.getOutput().getType(), origOp.getInput(), origOp.getFilter(), origOp.getBias(),
            origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
            origOp.getGroupsAttr(), origOp.getPostOpAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());

    return mlir::success();
}

//
// EmbeddingSegmentsSumRewriter
//

mlir::LogicalResult EmbeddingSegmentsSumRewriter::matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getSegmentIds(), origOp.getPerSampleWeights(),
            /*indices_value=*/nullptr, /*segment_ids_value=*/nullptr, origOp.getNumSegmentsValueAttr(),
            origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
    return mlir::success();
}

//
// EmbeddingBagOffsetsSumRewriter
//

mlir::LogicalResult EmbeddingBagOffsetsSumRewriter::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Found EmbeddingBagOffsetsSumOp Operation '{0}'", origOp->getLoc());
    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
            origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getOffsets(), origOp.getPerSampleWeights(),
            /*indices_value=*/nullptr,
            /*offsets_value=*/nullptr, origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
    return mlir::success();
}

//
// RandomUniformRewrite
//

mlir::LogicalResult RandomUniformRewrite::matchAndRewrite(IE::RandomUniformOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found RandomUniform Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::RandomUniformOp>(origOp, origOp.getMin(), origOp.getMax(),
                                                      origOp.getOutputShapeAttr(), origOp.getOutputTypeAttr(),
                                                      origOp.getGlobalSeedAttr(), origOp.getOpSeedAttr(),
                                                      /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// DynamicReshapeRewrite
//

mlir::LogicalResult DynamicReshapeRewrite::matchAndRewrite(IE::DynamicReshapeOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DynamicReshape Operation '{0}'", origOp->getLoc());

    const auto outputType = origOp.getOutput().getType();
    rewriter.replaceOpWithNewOp<VPU::DynamicReshapeOp>(origOp, outputType, origOp.getInput(), origOp.getShape(),
                                                       origOp.getOutputShapeAttr(), origOp.getOutputBoundsAttr(),
                                                       origOp.getOnlySetShapeAttr());

    return mlir::success();
}

//
// DynamicTileRewrite
//

mlir::LogicalResult DynamicTileRewrite::matchAndRewrite(IE::DynamicTileOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DynamicTileOp Operation '{0}'", origOp->getLoc());

    const auto outputType = origOp.getOutput().getType();
    rewriter.replaceOpWithNewOp<VPU::DynamicTileOp>(origOp, outputType, origOp.getInput(), origOp.getTargetShape(),
                                                    origOp.getRepeats(), origOp.getRepeatsValuesAttr(),
                                                    origOp.getOutputShapeAttr(), origOp.getOutputBoundsAttr());

    return mlir::success();
}

//
// DynamicQuantizeRewrite
//

mlir::LogicalResult DynamicQuantizeRewrite::matchAndRewrite(IE::DynamicQuantizeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DynamicQuantizeOp Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::DynamicQuantizeOp>(
            origOp,
            mlir::TypeRange{origOp.getOutput().getType(), origOp.getScale().getType(), origOp.getZeroPoint().getType()},
            origOp.getInput(), origOp.getMin(), origOp.getMax(), origOp.getDstElemTypeAttr(), nullptr);
    return mlir::success();
}

//
// AddRewrite
//

mlir::LogicalResult AddRewrite::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Add Operation '{0}'", origOp->getLoc());

    if (origOp.getScales() != nullptr) {
        return matchFailed(rewriter, origOp, "SW eltwise Add does not support scales-as-input");
    }

    constexpr int64_t RANK_4D = 4;

    mlir::Value input1 = origOp.getInput1();
    mlir::Value input2 = origOp.getInput2();
    const auto input1Type = mlir::cast<NDTypeInterface>(input1.getType());
    const auto input2Type = mlir::cast<NDTypeInterface>(input2.getType());

    if (input1Type.getRank() != input2Type.getRank()) {
        return matchFailed(rewriter, origOp, "The input ranks are not the same: {0} and {1}", input1Type.getRank(),
                           input2Type.getRank());
    }

    auto inputRank = input1Type.getRank();
    if (inputRank < RANK_4D) {
        auto reshapeTo4D = [&origOp, &rewriter](auto input) {
            const auto inputShape = getShape(input).raw();
            auto fillOneCnt = RANK_4D - mlir::cast<vpux::NDTypeInterface>(input.getType()).getRank();
            auto newInputShape = SmallVector<int64_t>(fillOneCnt, 1);
            newInputShape.append(inputShape.begin(), inputShape.end());
            return rewriter.createOrFold<VPU::ReshapeOp>(origOp.getLoc(), input,
                                                         getIntArrayAttr(origOp.getContext(), newInputShape));
        };
        input1 = reshapeTo4D(input1);
        input2 = reshapeTo4D(input2);
    }
    // Rewrite
    auto newAddOp = rewriter.create<VPU::AddOp>(origOp.getLoc(), input1, input2, origOp.getAutoBroadcastAttr(),
                                                origOp.getPostOpAttr());
    if (inputRank < RANK_4D) {
        rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(
                origOp, newAddOp, getIntArrayAttr(origOp.getContext(), getShape(origOp.getOutput()).raw()));
    } else {
        rewriter.replaceOp(origOp, newAddOp);
    }
    return mlir::success();
}

//
// FlashSDPARewrite
//

mlir::LogicalResult FlashSDPARewrite::matchAndRewrite(IE::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    enum struct Rotate { Left, Right };

    // B - batch, H - numHeads, L - targetSequenceLength
    // AffineReshape
    // Rotate left:  [1, B, H, L] -> [B, H, L, 1]
    // Rotate right: [B, H, L, 1] -> [1, B, H, L]
    const auto rotateShape = [&](mlir::Value tensor, Rotate direction) -> mlir::Value {
        auto shape = getShape(tensor).toValues();

        SmallVector<SmallVector<int64_t>> inDimMapping;

        if (direction == Rotate::Left) {
            _log.trace("Rotate shape left  <-");
            _log.trace("From: {0}", shape);
            std::rotate(shape.begin(), std::next(shape.begin()), shape.end());
            _log.trace("To:   {0}", shape);

            inDimMapping = SmallVector<SmallVector<int64_t>>{{0}, {0}, {1}, {2, 3}};
        } else {
            _log.trace("Rotate shape right ->");
            _log.trace("From: {0}", shape);
            std::rotate(shape.rbegin(), std::next(shape.rbegin()), shape.rend());
            _log.trace("To:   {0}", shape);

            inDimMapping = SmallVector<SmallVector<int64_t>>{{0, 1}, {2}, {3}, {3}};
        }

        auto newShapeAttr = getIntArrayAttr(getContext(), shape);
        auto inDimMappingAttr = getIntArrayOfArray(getContext(), inDimMapping);

        auto loc = appendLoc(tensor.getLoc(), "reshaped");
        auto affineReshape = rewriter.create<VPU::AffineReshapeOp>(loc, tensor, inDimMappingAttr, newShapeAttr);

        return affineReshape.getOutput();
    };

    if (origOp.getInputRunningMax().getType().getRank() != 4 || origOp.getInputRunningSum().getType().getRank() != 4) {
        return errorAt(origOp, "'{0}' must have 4D input shapes", origOp->getName());
    }

    // Shift 4D max and sum tensor shapes to have 1 at the end of the shape
    // to align dimensions with the first output to satisfy MC tiling limitation
    // because we can't specify MC tiling dimensions for multiple outputs, they must align
    auto inputRunningMax = rotateShape(origOp.getInputRunningMax(), Rotate::Left);
    auto inputRunningSum = rotateShape(origOp.getInputRunningSum(), Rotate::Left);

    auto newOp = rewriter.create<VPU::FlashSDPAOp>(
            origOp->getLoc(), origOp.getQuery(), origOp.getKey(), origOp.getValue(), origOp.getInputRunningOutput(),
            inputRunningMax, inputRunningSum, origOp.getAttentionMask(), origOp.getSourceSeqLenPadSizeAttr(),
            origOp.getIsHeadAttr(), origOp.getIsTailAttr());

    auto resultRunningMax = rotateShape(newOp.getResultRunningMax(), Rotate::Right);
    auto resultRunningSum = rotateShape(newOp.getResultRunningSum(), Rotate::Right);

    rewriter.replaceOp(origOp, mlir::ValueRange{newOp.getResultRunningOutput(), resultRunningMax, resultRunningSum});

    return mlir::success();
}

//
// LogSoftmaxTopKRewrite
//

mlir::LogicalResult LogSoftmaxTopKRewrite::matchAndRewrite(IE::LogSoftmaxTopKOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LogSoftmaxTopK Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::LogSoftmaxTopKOp>(origOp, origOp.getInput(), origOp.getAxisIndAttr(),
                                                       origOp.getPadSizeAttr(), origOp.getDstElemTypeAttr());

    return mlir::success();
}

//
// LogSoftmaxPeakRewrite
//

mlir::LogicalResult LogSoftmaxPeakRewrite::matchAndRewrite(IE::LogSoftmaxPeakOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LogSoftmaxPeak Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::LogSoftmaxPeakOp>(origOp, origOp.getInput(), origOp.getAxisIndAttr(),
                                                       origOp.getPadSizeAttr(), origOp.getDstElemTypeAttr());

    return mlir::success();
}
namespace {

//
// ConvertLayers2VPUPass
//

class ConvertLayers2VPUPass final : public impl::ConvertLayers2VPUBase<ConvertLayers2VPUPass> {
public:
    explicit ConvertLayers2VPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<Shave::ShaveDialect>();

    if (config::isPureHostCompileFunc(func)) {
        // host pipeline related
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();
        target.addLegalDialect<mlir::tensor::TensorDialect>();
    }

    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();
    target.addLegalOp<Core::ReinterpretCastOp>();
    target.addLegalOp<VPU::AffineReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<IfRewrite>(&ctx, _log);
    patterns.add<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.add<ProposalRewrite>(&ctx, _log);
    patterns.add<SplitRewrite>(&ctx, _log);
    patterns.add<StubRewrite>(&ctx, _log);
    patterns.add<NonMaxSuppressionRewrite>(&ctx, _log);
    patterns.add<InterpolateRewrite>(&ctx, _log);
    patterns.add<GRUCellRewrite>(&ctx, _log);
    patterns.add<ExperimentalDetectronROIFeatureExtractorRewrite>(&ctx, _log);
    patterns.add<TopKRewrite>(&ctx, _log);
    patterns.add<AtanRewrite>(&ctx, _log);
    patterns.add<ScatterUpdateRewrite>(&ctx, _log);
    patterns.add<MaxPool8Rewrite>(&ctx, _log);
    patterns.add<TransposedConvRewrite>(&ctx, _log);
    patterns.add<NormalizeL2Rewrite>(&ctx, _log);
    patterns.add<LSTMCellRewrite>(&ctx, _log);
    patterns.add<LSTMSequenceRewrite>(&ctx, _log);
    patterns.add<LSTMGatesRewrite>(&ctx, _log);
    patterns.add<GroupConvolutionRewrite>(&ctx, _log);
    patterns.add<EmbeddingSegmentsSumRewriter>(&ctx, _log);
    patterns.add<EmbeddingBagOffsetsSumRewriter>(&ctx, _log);
    patterns.add<RandomUniformRewrite>(&ctx, _log);
    patterns.add<DynamicReshapeRewrite>(&ctx, _log);
    patterns.add<DynamicTileRewrite>(&ctx, _log);
    patterns.add<DynamicQuantizeRewrite>(&ctx, _log);
    patterns.add<AddRewrite>(&ctx, _log);
    patterns.add<FlashSDPARewrite>(&ctx, _log);
    patterns.add<LogSoftmaxTopKRewrite>(&ctx, _log);
    patterns.add<LogSoftmaxPeakRewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUPass>(log);
}
