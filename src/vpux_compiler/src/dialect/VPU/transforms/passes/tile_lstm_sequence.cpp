//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/utils/core/error.hpp>
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>

namespace vpux::VPU {
#define GEN_PASS_DECL_TILELSTMSEQUENCE
#define GEN_PASS_DEF_TILELSTMSEQUENCE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class TileLSTMSequence final : public mlir::OpRewritePattern<VPU::LSTMSequenceOp> {
public:
    TileLSTMSequence(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::LSTMSequenceOp>(ctx), _log(std::move(log)) {
        setDebugName("TileLSTMSequence");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LSTMSequenceOp lstmSequenceOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

private:
    bool fitIntoCMX(VPU::LSTMSequenceOp op) const;
    bool fitIntoCMX(SmallVector<Byte>& bufferSizes, int64_t totalAvailableCMXSize, VPU::ArchKind archKind) const;

    mlir::FailureOr<int> getNumSplits(VPU::LSTMSequenceOp op) const;
    void tileLSTMSequence(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter, int numSplits) const;
    void tileBidirectionalLSTMSequence(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter, int numSplits) const;
    void tileForwardOrReverseLSTMSequence(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter, int numSplits) const;
};

bool TileLSTMSequence::fitIntoCMX(VPU::LSTMSequenceOp op) const {
    SmallVector<Byte> bufferSizes;

    for (const auto& operand : op.getOperands()) {
        const auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        bufferSizes.push_back(operandType.getTotalAllocSize());
    }

    for (const auto& result : op.getResults()) {
        const auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
        bufferSizes.push_back(resultType.getTotalAllocSize());
    }

    const auto totalAvailableCMXSize = getTotalCMXSize(op).count();
    return fitIntoCMX(bufferSizes, totalAvailableCMXSize, getArch(op));
}

bool TileLSTMSequence::fitIntoCMX(SmallVector<Byte>& bufferSizes, int64_t totalAvailableCMXSize,
                                  VPU::ArchKind archKind) const {
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(archKind, bufferSizes).count() <= totalAvailableCMXSize;
}

mlir::FailureOr<int> TileLSTMSequence::getNumSplits(VPU::LSTMSequenceOp op) const {
    auto inputData = mlir::cast<vpux::NDTypeInterface>(op.getInputData().getType());
    auto outputHiddenValues = mlir::cast<vpux::NDTypeInterface>(op.getOutputHiddenValues().getType());

    auto inputDataShape = Shape(inputData.getShape());
    auto outputHiddenValuesShape = Shape(outputHiddenValues.getShape());
    const auto seqLenght = op.getSequenceLength().has_value() ? op.getSequenceLength().value() : 1;

    if (inputDataShape.size() != 4 || outputHiddenValuesShape.size() != 4) {
        _log.trace("LSTMSequenceOp cannot be tiled due to unsupported shape sizes.");
        return mlir::failure();
    }

    SmallVector<Byte> bufferSizes;
    const auto numBuffers = op.getNumOperands() + op.getNumResults();
    bufferSizes.reserve(numBuffers);

    const auto operands = op.getOperands();
    const auto results = op.getResults();

    const auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto numClustersAvailableForCompilation = IE::getTileExecutor(module).getCount();
    const auto multiClusterStrategy = op.getMultiClusterStrategy();

    const auto applyMultiClusterTiling = [&](NDTypeInterface type,
                                             VPU::MultiClusterStrategy strategy) -> NDTypeInterface {
        auto typeShape = to_small_vector(type.getShape());
        const auto tilingScheme = VPU::getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, type);
        VPUX_THROW_UNLESS(typeShape.size() == tilingScheme.size(), "Type shape and tiling scheme are not compatible.",
                          typeShape.size(), tilingScheme.size());
        std::transform(typeShape.begin(), typeShape.end(), tilingScheme.begin(), typeShape.begin(),
                       [](double shapeVal, double tileVal) {
                           return std::ceil(shapeVal / tileVal);
                       });
        return type.changeShape(Shape(typeShape));
    };

    for (const auto& operand : operands) {
        auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        if (multiClusterStrategy.has_value()) {
            operandType = applyMultiClusterTiling(operandType, multiClusterStrategy.value());
        }
        bufferSizes.push_back(operandType.getTotalAllocSize());
    }

    for (const auto& result : results) {
        auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
        if (multiClusterStrategy.has_value()) {
            resultType = applyMultiClusterTiling(resultType, multiClusterStrategy.value());
        }
        bufferSizes.push_back(resultType.getTotalAllocSize());
    }

    const auto totalAvailableCMXSize = getTotalCMXSize(op).count();
    const auto archKind = getArch(op);

    if (fitIntoCMX(bufferSizes, totalAvailableCMXSize, archKind)) {
        return 1;  // numSplits
    }

    const auto inputDataOperandIdx = std::find(operands.begin(), operands.end(), op.getInputData()) - operands.begin();
    const auto outputHiddenValuesResultIdx =
            std::find(results.begin(), results.end(), op.getOutputHiddenValues()) - results.begin();

    const auto inputDataBufferIdx = inputDataOperandIdx;
    const auto outputHiddenValuesBufferIdx = op.getNumOperands() + outputHiddenValuesResultIdx;

    if (multiClusterStrategy.has_value()) {
        inputData = applyMultiClusterTiling(inputData, multiClusterStrategy.value());
        inputDataShape = Shape(inputData.getShape());

        outputHiddenValues = applyMultiClusterTiling(outputHiddenValues, multiClusterStrategy.value());
        outputHiddenValuesShape = Shape(outputHiddenValues.getShape());
    }

    for (int numSplits = 2; numSplits <= seqLenght; numSplits++) {
        const int64_t newLargestSeqLenght =
                std::ceil(checked_cast<double>(seqLenght) / checked_cast<double>(numSplits));

        auto newInputDataShape = inputDataShape;
        newInputDataShape[Dims4D::Act::H] = newLargestSeqLenght;
        bufferSizes[inputDataBufferIdx] = inputData.changeShape(newInputDataShape).getTotalAllocSize();

        auto newOutputHiddenValuesShape = outputHiddenValuesShape;
        newOutputHiddenValuesShape[Dims4D::Act::H] = newLargestSeqLenght;
        bufferSizes[outputHiddenValuesBufferIdx] =
                outputHiddenValues.changeShape(newOutputHiddenValuesShape).getTotalAllocSize();

        if (fitIntoCMX(bufferSizes, totalAvailableCMXSize, archKind)) {
            return numSplits;
        }
    }

    return mlir::failure();
}

void TileLSTMSequence::tileBidirectionalLSTMSequence(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter,
                                                     const int numSplits) const {
    const auto direction = op.getDirection();
    VPUX_THROW_WHEN(direction != IE::RNNSequenceDirection::BIDIRECTIONAL, "Expected BIDIRECTIONAL direction, got {0}",
                    direction);

    const auto loc = op.getLoc();
    const auto ctx = rewriter.getContext();

    int splitIdx = 0;
    const auto splitOnDim = [&](mlir::Value input, int64_t dim) -> std::pair<mlir::Value, mlir::Value> {
        const auto inputShape = getShape(input).raw();

        VPUX_THROW_UNLESS(dim < static_cast<int64_t>(inputShape.size()), "Dim {0} is out of expected range [0, {1}]",
                          dim, inputShape.size() - 1);
        VPUX_THROW_UNLESS(inputShape[dim] == 2, "Expected inputShape[{0}] to be 2, got {1}", dim, inputShape[dim]);

        SmallVector<int64_t> sliceSizes(inputShape);
        sliceSizes[dim] = 1;
        const auto sliceSizesArrayAttr = getIntArrayAttr(ctx, sliceSizes);
        SmallVector<int64_t> sliceOffsets(inputShape.size(), 0);

        const mlir::Value sliceForward =
                rewriter.create<VPU::SliceOp>(appendLoc(loc, "_sliceForward_{0}", splitIdx), input,
                                              getIntArrayAttr(ctx, sliceOffsets), sliceSizesArrayAttr);
        sliceOffsets[dim] = 1;
        const mlir::Value sliceReverse =
                rewriter.create<VPU::SliceOp>(appendLoc(loc, "_sliceReverse_{0}", splitIdx), input,
                                              getIntArrayAttr(ctx, sliceOffsets), sliceSizesArrayAttr);
        splitIdx++;
        return {sliceForward, sliceReverse};
    };

    const auto inputData = op.getInputData();
    const auto [inputDataForward, inputDataReverse] = splitOnDim(inputData, 1);

    const auto inputDataShape = Shape(getShape(inputData));
    const auto inputDataForwardShape = Shape(getShape(inputDataForward));
    const auto outputHiddenValuesShape = Shape(getShape(op.getOutputHiddenValues()));

    const auto seqLenght = op.getSequenceLength().has_value() ? op.getSequenceLength().value() : 1;
    mlir::Value hiddenState = op.getInitialHiddenState();
    mlir::Value cellState = op.getInitialCellState();
    SmallVector<mlir::Value> outputHiddenValuesVecForward;
    SmallVector<mlir::Value> outputHiddenValuesVecReverse;

    int64_t seqLenghtOffset = 0;
    const int64_t splitSize = seqLenght / numSplits;
    const int64_t reminder = seqLenght % numSplits;

    for (int i = 0; i < numSplits; i++) {
        const auto newSeqLenght = i < reminder ? splitSize + 1 : splitSize;

        auto sliceSizes = inputDataForwardShape;
        sliceSizes[Dims4D::Act::H] = newSeqLenght;

        SmallVector<int64_t> sliceOffsetsForward(inputDataShape.size(), 0);
        sliceOffsetsForward[2] = seqLenghtOffset;

        SmallVector<int64_t> sliceOffsetsReverse(inputDataShape.size(), 0);
        sliceOffsetsReverse[2] = seqLenght - seqLenghtOffset - newSeqLenght;

        const mlir::Value sliceForward = rewriter.create<VPU::SliceOp>(
                appendLoc(loc, "_sliceForward_{0}", i), inputDataForward, getIntArrayAttr(ctx, sliceOffsetsForward),
                getIntArrayAttr(ctx, sliceSizes));
        const mlir::Value sliceReverse = rewriter.create<VPU::SliceOp>(
                appendLoc(loc, "_sliceReverse_{0}", i), inputDataReverse, getIntArrayAttr(ctx, sliceOffsetsReverse),
                getIntArrayAttr(ctx, sliceSizes));
        const SmallVector<mlir::Value> sliceOps{sliceForward, sliceReverse};

        const mlir::Value newLstmSequenceInput =
                rewriter.create<VPU::ConcatOp>(appendLoc(loc, "_concat_{0}", i), sliceOps, 1);

        auto newLSTMSequenceOp = rewriter.create<VPU::LSTMSequenceOp>(
                appendLoc(loc, "_tile_{0}", 1), newLstmSequenceInput, hiddenState, cellState, op.getReccurenceWeights(),
                getIntAttr(ctx, newSeqLenght), op.getDirectionAttr(), op.getMultiClusterStrategyAttr());

        const auto [newOutputHiddenValuesForward, newOutputHiddenValuesReverse] =
                splitOnDim(newLSTMSequenceOp.getOutputHiddenValues(), 1);
        outputHiddenValuesVecForward.push_back(newOutputHiddenValuesForward);
        outputHiddenValuesVecReverse.push_back(newOutputHiddenValuesReverse);

        hiddenState = newLSTMSequenceOp.getOutputHiddenState();
        cellState = newLSTMSequenceOp.getOutputCellState();

        seqLenghtOffset += newSeqLenght;
    }

    std::reverse(outputHiddenValuesVecReverse.begin(), outputHiddenValuesVecReverse.end());

    const mlir::Value outputHiddenValuesForward =
            rewriter.create<VPU::ConcatOp>(appendLoc(loc, "_concatForward"), outputHiddenValuesVecForward, 2);
    const mlir::Value outputHiddenValuesReverse =
            rewriter.create<VPU::ConcatOp>(appendLoc(loc, "_concatReverse"), outputHiddenValuesVecReverse, 2);

    const SmallVector<mlir::Value> outputHiddenValuesVec{outputHiddenValuesForward, outputHiddenValuesReverse};
    const mlir::Value newOutputHiddenValues =
            rewriter.create<VPU::ConcatOp>(appendLoc(loc, "_concat"), outputHiddenValuesVec, 1);

    const SmallVector<mlir::Value> newResults{newOutputHiddenValues, hiddenState, cellState};
    rewriter.replaceOp(op, newResults);
}

void TileLSTMSequence::tileForwardOrReverseLSTMSequence(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter,
                                                        const int numSplits) const {
    const auto direction = op.getDirection();
    VPUX_THROW_WHEN(direction != IE::RNNSequenceDirection::FORWARD && direction != IE::RNNSequenceDirection::REVERSE,
                    "Expected FORWARD or REVERSE direction, got {0}", direction);

    const auto loc = op.getLoc();
    const auto ctx = rewriter.getContext();

    const auto inputData = op.getInputData();
    const auto inputDataShape = Shape(getShape(inputData));
    const auto outputHiddenValuesShape = Shape(getShape(op.getOutputHiddenValues()));

    const auto seqLenght = op.getSequenceLength().has_value() ? op.getSequenceLength().value() : 1;
    mlir::Value newHiddenState = op.getInitialHiddenState();
    mlir::Value newCellState = op.getInitialCellState();
    SmallVector<mlir::Value> outputHiddenValuesVec;

    int64_t seqLenghtOffset = 0;
    const int64_t splitSize = seqLenght / numSplits;
    const int64_t reminder = seqLenght % numSplits;

    for (int i = 0; i < numSplits; i++) {
        const auto newSeqLenght = i < reminder ? splitSize + 1 : splitSize;

        auto sliceSizes = inputDataShape;
        sliceSizes[Dims4D::Act::H] = newSeqLenght;

        Shape sliceOffsets(inputDataShape.size(), 0);
        sliceOffsets[Dims4D::Act::H] = direction == vpux::IE::RNNSequenceDirection::FORWARD
                                               ? seqLenghtOffset
                                               : seqLenght - seqLenghtOffset - newSeqLenght;

        const mlir::Value newLstmSequenceInput =
                rewriter.create<VPU::SliceOp>(appendLoc(loc, "_slice_{0}", i), inputData,
                                              getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, sliceSizes));

        auto newLSTMSequenceOp = rewriter.create<VPU::LSTMSequenceOp>(
                appendLoc(loc, "_tile_{0}", 1), newLstmSequenceInput, newHiddenState, newCellState,
                op.getReccurenceWeights(), getIntAttr(ctx, newSeqLenght), op.getDirectionAttr(),
                op.getMultiClusterStrategyAttr());

        outputHiddenValuesVec.push_back(newLSTMSequenceOp.getOutputHiddenValues());
        newHiddenState = newLSTMSequenceOp.getOutputHiddenState();
        newCellState = newLSTMSequenceOp.getOutputCellState();

        seqLenghtOffset += newSeqLenght;
    }

    if (direction == vpux::IE::RNNSequenceDirection::REVERSE) {
        std::reverse(outputHiddenValuesVec.begin(), outputHiddenValuesVec.end());
    }

    const mlir::Value newOutputHiddenValues =
            rewriter.create<VPU::ConcatOp>(appendLoc(loc, "_concat"), outputHiddenValuesVec, 2);

    const SmallVector<mlir::Value> newResults{newOutputHiddenValues, newHiddenState, newCellState};
    rewriter.replaceOp(op, newResults);
}

void TileLSTMSequence::tileLSTMSequence(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter, int numSplits) const {
    if (op.getDirection() == vpux::IE::RNNSequenceDirection::BIDIRECTIONAL) {
        tileBidirectionalLSTMSequence(op, rewriter, numSplits);
    } else {
        tileForwardOrReverseLSTMSequence(op, rewriter, numSplits);
    }
}

mlir::LogicalResult TileLSTMSequence::matchAndRewrite(VPU::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const {
    if (fitIntoCMX(op)) {
        return mlir::failure();
    }

    // Tile the LSTMSequenceOp sequentially, creating a chain of LSTMSequenceOps split on the seq_length dimension.
    // This cannot be done in the case of dynamic seq_length.
    if (IE::hasDynamicTensors(op)) {
        return mlir::failure();
    }

    const auto numSplits = getNumSplits(op);
    if (mlir::failed(numSplits) || numSplits == 1) {
        return mlir::failure();
    }

    tileLSTMSequence(op, rewriter, numSplits.value());

    return mlir::success();
};

class TileLSTMSequencePass final : public VPU::impl::TileLSTMSequenceBase<TileLSTMSequencePass> {
public:
    explicit TileLSTMSequencePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void TileLSTMSequencePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<TileLSTMSequence>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(greedyPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createTileLSTMSequencePass(Logger log) {
    return std::make_unique<TileLSTMSequencePass>(std::move(log));
}
