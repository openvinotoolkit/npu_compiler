//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSELSTMSEQUENCE
#define GEN_PASS_DEF_DECOMPOSELSTMSEQUENCE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ExtractWeightsAndBiasesFromLSTMSequenceRewriter
//

// The matrix multiplication of inputData and weights, and the addition of biases, can be computed once for the entire
// sequence, as they are not calculated recursively. This rewriter extracts these operations from LSTMSequence to allow
// them to run on the DPU.

class ExtractWeightsAndBiasesFromLSTMSequenceRewriter final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    ExtractWeightsAndBiasesFromLSTMSequenceRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx, benefit), _log(std::move(log)) {
        this->setDebugName("ExtractWeightsAndBiasesFromLSTMSequenceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Properly adjust dynamic shapes to enable their use in StridedSlice/DynamicReshape operations.
// If we need to alter the dynamic shape while preserving the dynamic dimension, we should utilize
// Slice and Concat operations.
std::pair<mlir::Value, mlir::Value> createShapeSliceConcat(
        mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, mlir::TypeAttr dataTypeShapeOf,
        mlir::ArrayRef<int64_t> sliceBegins, mlir::ArrayRef<int64_t> sliceSizes, mlir::ArrayRef<int64_t> constValues,
        mlir::ArrayRef<int64_t> secondSliceBegins, mlir::ArrayRef<int64_t> secondSliceSizes, int64_t concatAxis,
        mlir::MLIRContext* ctx, const std::string& suffix) {
    // suffix is required to avoid duplicated names.
    auto shapeOf = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOf" + suffix), input, dataTypeShapeOf);
    auto dim0 =
            rewriter.create<IE::SliceOp>(appendLoc(loc, "_dim0" + suffix), shapeOf,
                                         rewriter.getI64ArrayAttr(sliceBegins), rewriter.getI64ArrayAttr(sliceSizes));

    auto constDataType = mlir::RankedTensorType::get({1}, getSInt64Type(ctx));
    auto const1 = Const::createConst(rewriter, appendLoc(loc, "_const" + suffix), constDataType, ArrayRef(constValues));
    auto dim1 = rewriter.create<IE::SliceOp>(appendLoc(loc, "_dim1 " + suffix), shapeOf,
                                             rewriter.getI64ArrayAttr(secondSliceBegins),
                                             rewriter.getI64ArrayAttr(secondSliceSizes));
    const auto axisAttr = getIntAttr(ctx, concatAxis);
    SmallVector<mlir::Value> concatInputs{dim0, const1, dim1};

    return {shapeOf.getOutput(),
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_concat" + suffix), concatInputs, axisAttr).getOutput()};
}

// The matrix multiplication of inputData and weights, and the addition of biases, can be computed once for the entire
// sequence, as they are not calculated recursively.
// However, when dealing with dynamic input for LSTMSequence, this approach cannot be applied directly because Add and
// MatMul operations do not yet support dynamic shapes fully. Instead, we can execute these operations using an
// upper-bounded input (via the DynamicExtend operation). Subsequently, we can use Slice to remove any unnecessary data
// from the results.
mlir::Value padWeightsAndBiases(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value inputData,
                                mlir::Value newInputData, mlir::Value newWeights, mlir::MLIRContext* ctx) {
    auto padForMatMul = rewriter.create<IE::DynamicExpandOp>(appendLoc(loc, "_padForMatMul"), newInputData);

    auto matMulInputOp =
            rewriter.create<IE::MatMulOp>(appendLoc(loc, "_matMul"), padForMatMul, newWeights, false, true, nullptr);

    auto addShape = to_small_vector(getShape(matMulInputOp.getOutput()));
    auto inputDataShape = to_small_vector(getShape(inputData));
    auto addBounds = addShape;

    const auto addShapeRank = checked_cast<int64_t>(addShape.size());
    const auto inputDataShapeRank = checked_cast<int64_t>(inputDataShape.size());

    const auto seqLengthAddIndex = addShapeRank - 2;
    const auto seqLengthDataIndex = inputDataShapeRank - 2;

    addShape[seqLengthAddIndex] = to_small_vector(inputDataShape)[seqLengthDataIndex];

    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(inputData.getType());
    VPUX_THROW_UNLESS(boundedType != nullptr, "Expected to get BoundedTensorType at {0}", inputData.getLoc());
    addBounds[seqLengthAddIndex] = to_small_vector(boundedType.getBounds())[seqLengthDataIndex];

    const auto shapeAttr = getIntArrayAttr(ctx, addShape);
    const auto boundedShapeAttr = getIntArrayAttr(ctx, addBounds);
    const auto dataTypeShapeOf = mlir::TypeAttr::get(getSInt64Type(ctx));

    // In order to apply StridedSlice (to the operations that were performed with upper bounds), we need to determine
    // the correct shape for the output tensor. To achieve this, we will obtain the actual dynamic shape using ShapeOf
    // and use it as the ends input for StridedSlice.

    // %shape_of_cst = IE.ShapeOf(%cst_0) -> 1x2x512x512
    auto shapeOfCst = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOfCst"), newWeights, dataTypeShapeOf);

    // %slice_cst = IE.Slice %shape_of_cst [0] [1] -> 1
    auto sliceBatchCst = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceCst"), shapeOfCst,
                                                      /*begins=*/rewriter.getI64ArrayAttr({0}),
                                                      /*sizes=*/rewriter.getI64ArrayAttr({1}));
    // TODO: here we use a knowledge that LSTMSequence will be multiclustered for the dynamic shape case
    // by the number of directions, so we set the channel number to 1
    auto constDataType = mlir::RankedTensorType::get({1}, getSInt64Type(ctx));
    auto channelsCst =
            Const::createConst(rewriter, appendLoc(loc, "_channelsNum"), constDataType, ArrayRef<int64_t>{1});

    // %shape_of_0 = IE.ShapeOf(%0) -> 1x1x?x512
    auto shapeOfInput = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOfInput"), newInputData, dataTypeShapeOf);

    // %slice_0 = IE.Slice %shape_of_0 [2] [1] -> ?
    auto sliceInput = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceInput"), shapeOfInput,
                                                   /*begins=*/rewriter.getI64ArrayAttr({2}),
                                                   /*sizes=*/rewriter.getI64ArrayAttr({1}));

    // %shape_of_3 = IE.ShapeOf(%3) -> 1x2x35x512
    auto shapeOfAdd =
            rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOfAdd"), matMulInputOp.getOutput(), dataTypeShapeOf);

    // %slice_3 = IE.Slice %shape_of_3 [3] [1] -> 512
    auto sliceAdd = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceAdd"), shapeOfAdd,
                                                 /*begins=*/rewriter.getI64ArrayAttr({3}),
                                                 /*sizes=*/rewriter.getI64ArrayAttr({1}));

    // %concat = IE.Concat(%slice_cst, %slice_0, %slice_3) -> 1x2x?x512
    const auto axisAttr = getIntAttr(ctx, 0);
    SmallVector<mlir::Value> concatInputs;
    concatInputs.push_back(sliceBatchCst);
    concatInputs.push_back(channelsCst);
    concatInputs.push_back(sliceInput);
    concatInputs.push_back(sliceAdd);

    auto newShapeOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_newShape"), concatInputs, axisAttr);
    auto reshapedAddOp = rewriter.create<IE::DynamicReshapeOp>(appendLoc(loc, "_reshapedAdd"),
                                                               /*data=*/matMulInputOp.getOutput(),
                                                               /*shape=*/newShapeOp.getOutput(),
                                                               /*output_shape=*/shapeAttr,
                                                               /*output_bounds=*/boundedShapeAttr);

    return reshapedAddOp;
}

mlir::LogicalResult ExtractWeightsAndBiasesFromLSTMSequenceRewriter::matchAndRewrite(
        IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const {
    auto inputData = op.getInputData();

    const auto weights = op.getWeights();
    const auto biases = op.getBiases();
    if (!weights || !biases) {
        return mlir::failure();
    }

    const auto initialHiddenStateShape = getShape(op.getInitialHiddenState());
    const auto batchSize = initialHiddenStateShape[Dim(0)];
    const auto numDirections = initialHiddenStateShape[Dim(1)];

    const auto ctx = rewriter.getContext();
    const auto loc = op.getLoc();
    const auto axisOne = 1;
    const auto axisZeroArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{axisOne});

    mlir::Value newInputData;
    if (auto boundedInputData = mlir::dyn_cast<Core::BoundedTensorType>(inputData.getType())) {
        auto newInputDataShape = to_small_vector(getShape(inputData));
        auto newInputDataBounds = to_small_vector(boundedInputData.getBounds());

        newInputDataShape.insert(newInputDataShape.begin() + axisOne, 1);
        newInputDataBounds.insert(newInputDataBounds.begin() + axisOne, 1);

        const auto newInputDataShapeAttr = getIntArrayAttr(ctx, newInputDataShape);
        const auto newInputDataBoundsAttr = getIntArrayAttr(ctx, newInputDataBounds);

        mlir::Value concat;
        auto dataTypeShapeOf = mlir::TypeAttr::get(getSInt64Type(ctx));
        std::tie(std::ignore, concat) = createShapeSliceConcat(rewriter, loc, inputData, dataTypeShapeOf, {0}, {1}, {1},
                                                               {1}, {2}, 0, ctx, "_pass1_inputDataUnsqueeze");

        newInputData = rewriter.create<IE::DynamicReshapeOp>(appendLoc(loc, "_inputDataUnsqueeze"), inputData, concat,
                                                             newInputDataShapeAttr, newInputDataBoundsAttr);
    } else {
        newInputData = rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_inputDataUnsqueeze"), inputData, nullptr,
                                                        axisOneArrayAttr);
    }

    mlir::Value newWeights =
            rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_weightsUnsqueeze"), weights, nullptr, axisZeroArrayAttr);

    if (numDirections > 1) {
        auto newInputDataShape = Shape(getShape(newInputData));
        newInputDataShape[Dim(1)] = numDirections;
        auto newWeightsShape = Shape(getShape(newWeights));
        newWeightsShape[Dim(0)] = batchSize;

        if (auto boundedNewInputData = mlir::dyn_cast<Core::BoundedTensorType>(newInputData.getType())) {
            const auto shapeAttr = getIntArrayAttr(ctx, to_small_vector(newInputDataShape));
            auto newInputDataBounds = to_small_vector(boundedNewInputData.getBounds());
            newInputDataBounds[1] = numDirections;
            const auto boundedShapeAttr = getIntArrayAttr(ctx, newInputDataBounds);

            mlir::Value concat;
            auto dataTypeShapeOf = mlir::TypeAttr::get(getSInt64Type(ctx));
            std::tie(std::ignore, concat) =
                    createShapeSliceConcat(rewriter, loc, newInputData, dataTypeShapeOf, {0}, {1}, {numDirections}, {2},
                                           {2}, 0, ctx, "_pass1_inputDataBroadcast");

            newInputData = rewriter.create<IE::DynamicBroadcastOp>(
                    appendLoc(loc, "_inputDataBroadcast"), newInputData, concat, nullptr,
                    IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY), shapeAttr, boundedShapeAttr);
        } else {
            newInputData = IE::createBroadcast(rewriter, appendLoc(loc, "_inputDataBroadcast"), newInputData,
                                               newInputDataShape);
        }
        newWeights = IE::createBroadcast(rewriter, appendLoc(loc, "_weightsBroadcast"), newWeights, newWeightsShape);
    }
    auto newBiasesOp =
            rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_biasesUnsqueeze"), biases, nullptr, axisOneArrayAttr);

    if (mlir::isa<Core::BoundedTensorType>(inputData.getType())) {
        auto reshapedAddOp = padWeightsAndBiases(rewriter, loc, inputData, newInputData, newWeights, ctx);

        auto newLSTMSequenceOp = rewriter.create<IE::LSTMSequenceOp>(
                loc, reshapedAddOp, op.getInitialHiddenState(), op.getInitialCellState(), nullptr,
                op.getReccurenceWeights(), newBiasesOp, op.getSequenceLengthAttr(), op.getDirectionAttr());

        mlir::Value newOutputHiddenValues = newLSTMSequenceOp.getOutputHiddenValues();
        // if user of newOutputHiddenValues is DynamicReshape, it means that output shape is propagated
        // by DynamicReshape
        if (auto dynReshape = mlir::dyn_cast<IE::DynamicReshapeOp>(*op.getOutputHiddenValues().getUsers().begin())) {
            rewriter.setInsertionPoint(dynReshape);

            // while LSTMSequence can work with strided data, it is not guaranteed following operations will do the
            // same, so we insert StridedSlice and DynamicReshape to pack data without strides
            auto reshapedLSTMSequenceOp = rewriter.create<IE::DynamicReshapeOp>(
                    appendLoc(loc, "_reshapedLSTMSequence"), newLSTMSequenceOp.getOutputHiddenValues(),
                    dynReshape.getShape(), dynReshape.getOutputShapeAttr(), dynReshape.getOutputBoundsAttr(),
                    /*only_set_shape*/ true);

            auto rank = mlir::cast<NDTypeInterface>(newLSTMSequenceOp.getOutputHiddenValues().getType()).getRank();
            const SmallVector<int64_t> begins(rank, 0);
            const SmallVector<int64_t> strides(rank, 1);

            const auto beginsAttr = getIntArrayAttr(ctx, begins);
            const auto stridesAttr = getIntArrayAttr(ctx, strides);

            const SmallVector<int64_t> empty(rank, 0);
            const auto emptyAttr = getIntArrayAttr(ctx, empty);
            auto sliceOp = rewriter.create<IE::StridedSliceOp>(appendLoc(loc, "_denseDataLSTMSequence"),
                                                               /*data=*/reshapedLSTMSequenceOp.getOutput(),
                                                               /*begins=*/nullptr,
                                                               /*ends=*/dynReshape.getShape(),
                                                               /*strides=*/nullptr,
                                                               /*beginsAttr=*/beginsAttr,
                                                               /*endsAttr=*/nullptr,
                                                               /*stridesAttr=*/stridesAttr,
                                                               /*beginMask=*/emptyAttr,
                                                               /*endMask=*/emptyAttr,
                                                               /*newAxisMask=*/emptyAttr,
                                                               /*shrinkAxisMask=*/emptyAttr,
                                                               /*ellipsisMask=*/emptyAttr);
            newOutputHiddenValues = sliceOp.getOutput();
        }

        rewriter.replaceAllUsesWith(op.getResults(),
                                    mlir::ValueRange{newOutputHiddenValues, newLSTMSequenceOp.getOutputHiddenState(),
                                                     newLSTMSequenceOp.getOutputCellState()});
    } else {
        auto matMulInputOp = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_matMul"), newInputData, newWeights, false,
                                                           true, nullptr);
        if (VPU::LSTMSequenceOp::isSupported(op)) {
            auto newLSTMSequenceOp = rewriter.create<IE::LSTMSequenceOp>(
                    loc, matMulInputOp, op.getInitialHiddenState(), op.getInitialCellState(), nullptr,
                    op.getReccurenceWeights(), newBiasesOp, op.getSequenceLengthAttr(), op.getDirectionAttr());

            rewriter.replaceOp(op, newLSTMSequenceOp);
        } else {
            auto addOp = rewriter.create<IE::AddOp>(
                    appendLoc(loc, "_add"), matMulInputOp, newBiasesOp,
                    IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY), nullptr, nullptr,
                    nullptr, nullptr);

            auto newLSTMSequenceOp = rewriter.create<IE::LSTMSequenceOp>(
                    loc, addOp, op.getInitialHiddenState(), op.getInitialCellState(), nullptr,
                    op.getReccurenceWeights(), nullptr, op.getSequenceLengthAttr(), op.getDirectionAttr());

            rewriter.replaceOp(op, newLSTMSequenceOp);
        }
    }
    return mlir::success();
}

std::pair<mlir::Value, mlir::Value> splitOnDim(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                               int64_t dim, int& splitIdx) {
    if (!input) {
        return {nullptr, nullptr};
    }

    auto inputShape = Shape(getShape(input));
    const auto inputShapeVec = to_small_vector(inputShape);
    const auto ctx = rewriter.getContext();

    VPUX_THROW_UNLESS(dim < static_cast<int64_t>(inputShapeVec.size()), "Dim {0} is out of expected range [0, {1}]",
                      dim, inputShapeVec.size() - 1);

    if (inputShape.isDynamic() && inputShapeVec[dim] == 1) {
        // In the dynamic case, the decomposition of the LSTMSequence operation is performed before the
        // ExtractWeightsAndBiasesFromLSTMSequenceRewriter optimization. This means that the inputData
        // tensor has not yet been broadcasted to match the required shape for matrix multiplication
        // and addition operations. As a result, the inputData tensor retains its original shape,
        // which includes a dimension of size 1.

        // Since the inputData tensor has a dimension of size 1, it does not need to be split further
        // along this dimension. Therefore, we can directly use the inputData tensor as both the forward
        // and reverse inputs for the LSTMSequence operation without any additional splitting or processing.
        return {input, input};
    }

    VPUX_THROW_UNLESS(!inputShape.isDynamic() && inputShapeVec[dim] == 2, "Expected inputShape[{0}] to be 2, got {1}",
                      dim, inputShapeVec[dim]);

    SmallVector<int64_t> sliceSizes(inputShapeVec);
    sliceSizes[dim] = 1;
    const auto sliceSizesArrayAttr = getIntArrayAttr(ctx, sliceSizes);
    SmallVector<int64_t> sliceOffsets(inputShapeVec.size(), 0);

    mlir::Value sliceForward = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceForward_{0}", splitIdx), input,
                                                            getIntArrayAttr(ctx, sliceOffsets), sliceSizesArrayAttr);
    sliceOffsets[dim] = 1;
    mlir::Value sliceReverse = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceReverse_{0}", splitIdx), input,
                                                            getIntArrayAttr(ctx, sliceOffsets), sliceSizesArrayAttr);

    splitIdx++;
    return {sliceForward, sliceReverse};
}

//
// Base class for DecomposeLSTMSequenceBidirectionalRewriter
//

class BaseDecomposeLSTMSequenceBidirectionalRewriter : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    BaseDecomposeLSTMSequenceBidirectionalRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx, benefit), _log(std::move(log)) {
    }

protected:
    mlir::LogicalResult decompose(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter, bool isDynamic) const;

private:
    Logger _log;
};

mlir::LogicalResult BaseDecomposeLSTMSequenceBidirectionalRewriter::decompose(IE::LSTMSequenceOp op,
                                                                              mlir::PatternRewriter& rewriter,
                                                                              bool isDynamic) const {
    const auto direction = op.getDirection();
    if (direction != IE::RNNSequenceDirection::BIDIRECTIONAL) {
        return mlir::failure();
    }

    const auto loc = op.getLoc();
    const auto ctx = rewriter.getContext();

    int splitIdx = 0;

    const auto [inputDataForward, inputDataReverse] =
            splitOnDim(rewriter, loc, op.getInputData(), isDynamic ? 0 : 1, splitIdx);
    const auto [initialHiddenStateForward, initialHiddenStateReverse] =
            splitOnDim(rewriter, loc, op.getInitialHiddenState(), 1, splitIdx);
    const auto [initialCellStateForward, initialCellStateReverse] =
            splitOnDim(rewriter, loc, op.getInitialCellState(), 1, splitIdx);
    const auto [weightsForward, weightsReverse] = splitOnDim(rewriter, loc, op.getWeights(), 0, splitIdx);
    const auto [recurrenceWeightsForward, recurrenceWeightsReverse] =
            splitOnDim(rewriter, loc, op.getReccurenceWeights(), 0, splitIdx);
    const auto [biasesForward, biasesReverse] = splitOnDim(rewriter, loc, op.getBiases(), 0, splitIdx);

    auto lstmSequenceForwardOp = rewriter.create<IE::LSTMSequenceOp>(
            appendLoc(loc, "_forward"), inputDataForward, initialHiddenStateForward, initialCellStateForward,
            weightsForward, recurrenceWeightsForward, biasesForward, op.getSequenceLengthAttr(),
            IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD));

    auto lstmSequenceReverseOp = rewriter.create<IE::LSTMSequenceOp>(
            appendLoc(loc, "_reverse"), inputDataReverse, initialHiddenStateReverse, initialCellStateReverse,
            weightsReverse, recurrenceWeightsReverse, biasesReverse, op.getSequenceLengthAttr(),
            IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::REVERSE));

    auto outputHiddenValuesConcatOp =
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_hiddenValuesConcat"),
                                          SmallVector<mlir::Value>{lstmSequenceForwardOp.getOutputHiddenValues(),
                                                                   lstmSequenceReverseOp.getOutputHiddenValues()},
                                          Dim(1));
    auto outputHiddenStateConcatOp =
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_hiddenStateConcat"),
                                          SmallVector<mlir::Value>{lstmSequenceForwardOp.getOutputHiddenState(),
                                                                   lstmSequenceReverseOp.getOutputHiddenState()},
                                          Dim(1));
    auto outputCellStateConcatOp =
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_cellStateConcat"),
                                          SmallVector<mlir::Value>{lstmSequenceForwardOp.getOutputCellState(),
                                                                   lstmSequenceReverseOp.getOutputCellState()},
                                          Dim(1));

    const SmallVector<mlir::Value> newResults{outputHiddenValuesConcatOp, outputHiddenStateConcatOp,
                                              outputCellStateConcatOp};
    rewriter.replaceOp(op, newResults);

    return mlir::success();
}

//
// DecomposeDynamicLSTMSequenceBidirectionalRewriter
//

// Decompose a bidirectional dynamic LSTMSequence into one forward and one reverse operator.
// This optimization allows us to skip the extra StridedSlice operations that are added as
// part of the ExtractWeightsAndBiasesFromLSTMSequenceRewriter pass.

class DecomposeDynamicLSTMSequenceBidirectionalRewriter final : public BaseDecomposeLSTMSequenceBidirectionalRewriter {
public:
    DecomposeDynamicLSTMSequenceBidirectionalRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : BaseDecomposeLSTMSequenceBidirectionalRewriter(ctx, benefit, std::move(log)) {
        this->setDebugName("DecomposeDynamicLSTMSequenceBidirectionalRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final {
        if (!IE::hasDynamicTensors(op)) {
            return mlir::failure();
        }
        return decompose(op, rewriter, true);
    }
};

//
// DecomposeLSTMSequenceBidirectionalRewriter
//

// Decompose a bidirectional LSTMSequence into one forward and one reverse operator. It is a preparation step for
// unrolling an LSTMSequence operator to LSTMCell operators and is executed if the operation configuration is not
// supported by the VPU::LSTMSequenceOp.

class DecomposeLSTMSequenceBidirectionalRewriter final : public BaseDecomposeLSTMSequenceBidirectionalRewriter {
public:
    DecomposeLSTMSequenceBidirectionalRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : BaseDecomposeLSTMSequenceBidirectionalRewriter(ctx, benefit, std::move(log)) {
        this->setDebugName("DecomposeLSTMSequenceBidirectionalRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final {
        // At this stage this optimization will not be needed in case of dynamic shapes.
        if (VPU::LSTMSequenceOp::isSupported(op) || IE::hasDynamicTensors(op)) {
            return mlir::failure();
        }
        return decompose(op, rewriter, false);
    }
};

//
// UnrollLSTMSequenceToLSTMCellsRewriter
//

// Convert an LSTMSequence operator to LSTMCell operators if the operation configuration is unsupported by the
// VPU::LSTMSequenceOp.

class UnrollLSTMSequenceToLSTMCellsRewriter final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    UnrollLSTMSequenceToLSTMCellsRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx, benefit), _log(std::move(log)) {
        this->setDebugName("UnrollLSTMSequenceToLSTMCellsRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnrollLSTMSequenceToLSTMCellsRewriter::matchAndRewrite(IE::LSTMSequenceOp op,
                                                                           mlir::PatternRewriter& rewriter) const {
    if (VPU::LSTMSequenceOp::isSupported(op) || IE::hasDynamicTensors(op)) {
        return mlir::failure();
    }

    const auto direction = op.getDirection();
    VPUX_THROW_WHEN(direction != IE::RNNSequenceDirection::FORWARD && direction != IE::RNNSequenceDirection::REVERSE,
                    "Expected direction to be FORWARD or REVERSE, got {0}", direction);
    const auto isReverseDirection = direction == IE::RNNSequenceDirection::REVERSE;

    const auto ctx = rewriter.getContext();
    const auto loc = op.getLoc();

    const auto axisZeroArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1});

    int squeezeIdx = 0;
    const auto squeezeOnDim = [&](mlir::Value input, const mlir::ArrayAttr& axis) -> mlir::Value {
        if (!input) {
            return nullptr;
        }
        return rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_sqeenze_{0}", squeezeIdx++), input, nullptr, axis);
    };

    const mlir::Value inputData = squeezeOnDim(op.getInputData(), axisOneArrayAttr);
    mlir::Value hiddenState = squeezeOnDim(op.getInitialHiddenState(), axisOneArrayAttr);
    mlir::Value cellState = squeezeOnDim(op.getInitialCellState(), axisOneArrayAttr);
    const mlir::Value weights = squeezeOnDim(op.getWeights(), axisZeroArrayAttr);
    const mlir::Value reccurenceWeights = squeezeOnDim(op.getReccurenceWeights(), axisZeroArrayAttr);
    const mlir::Value biases = squeezeOnDim(op.getBiases(), axisZeroArrayAttr);

    const auto inputDataShape = getShape(inputData).raw();
    VPUX_THROW_UNLESS(inputDataShape.size() == 3, "inputData expected to be of rank 3, got {0}", inputDataShape.size());
    const auto sequenceLenght = op.getSequenceLength().has_value() ? op.getSequenceLength().value() : 1;
    const auto hiddenSizeAttr = getIntAttr(ctx, getShape(hiddenState).back());

    SmallVector<int64_t> sliceOffsets(inputDataShape.size(), 0);
    SmallVector<int64_t> sliceSizes(inputDataShape);
    sliceSizes[1] = 1;
    const auto sliceSizesAttr = getIntArrayAttr(ctx, sliceSizes);

    SmallVector<mlir::Value> lstmCellResults;

    for (int i = 0; i < sequenceLenght; i++) {
        sliceOffsets[1] = isReverseDirection ? sequenceLenght - 1 - i : i;
        auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_slice_{0}", i), inputData,
                                                    getIntArrayAttr(ctx, sliceOffsets), sliceSizesAttr);
        auto sqeezeOp =
                rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_squeeze_{0}", i), sliceOp, nullptr, axisOneArrayAttr);
        auto lstmCellOp =
                rewriter.create<IE::LSTMCellOp>(appendLoc(loc, "_lstmCell_{0}", i), sqeezeOp, hiddenState, cellState,
                                                weights, reccurenceWeights, biases, hiddenSizeAttr);
        auto unsqueezeOp = rewriter.create<IE::UnsqueezeOp>(
                appendLoc(loc, "_unsqueeze_{0}", i), lstmCellOp.getOutputHiddenState(), nullptr, axisOneArrayAttr);

        lstmCellResults.push_back(unsqueezeOp.getOutput());
        hiddenState = lstmCellOp.getOutputHiddenState();
        cellState = lstmCellOp.getOutputCellState();
    }

    if (isReverseDirection) {
        std::reverse(lstmCellResults.begin(), lstmCellResults.end());
    }

    mlir::Value newOutputHiddenValues =
            rewriter.create<IE::ConcatOp>(takeOpLoc(op, "_concat"), lstmCellResults, Dim(1));
    newOutputHiddenValues = rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), newOutputHiddenValues,
                                                             nullptr, axisOneArrayAttr);
    const mlir::Value newHiddenState =
            rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), hiddenState, nullptr, axisOneArrayAttr);
    const mlir::Value newCellState =
            rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), cellState, nullptr, axisOneArrayAttr);

    const SmallVector<mlir::Value> newResults{newOutputHiddenValues, newHiddenState, newCellState};
    rewriter.replaceOp(op, newResults);

    return mlir::success();
}

//
// UnrollLSTMSequenceToLSTMCellsPass
//

class DecomposeLSTMSequencePass final : public IE::impl::DecomposeLSTMSequenceBase<DecomposeLSTMSequencePass> {
public:
    explicit DecomposeLSTMSequencePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeLSTMSequencePass::safeRunOnFunc() {
    auto& ctx = getContext();

    // To explicitly control the patterns exec order to assure dependency
    // benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run

    const uint32_t levelCount = 4;

    const auto benefitLevels = getBenefitLevels(levelCount);

    mlir::RewritePatternSet patterns(&ctx);
    // In the dynamic case, decompose bidirectional LSTMSequence first to simplify handling of dynamic shapes
    // and avoid complex slicing operations. This makes subsequent optimizations easier.
    patterns.add<DecomposeDynamicLSTMSequenceBidirectionalRewriter>(&ctx, benefitLevels[0], _log);
    patterns.add<ExtractWeightsAndBiasesFromLSTMSequenceRewriter>(&ctx, benefitLevels[1], _log);
    patterns.add<DecomposeLSTMSequenceBidirectionalRewriter>(&ctx, benefitLevels[2], _log);
    patterns.add<UnrollLSTMSequenceToLSTMCellsRewriter>(&ctx, benefitLevels[3], _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeLSTMSequencePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeLSTMSequencePass(Logger log) {
    return std::make_unique<DecomposeLSTMSequencePass>(std::move(log));
}
