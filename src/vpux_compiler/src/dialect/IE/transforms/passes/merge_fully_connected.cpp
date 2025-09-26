//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MERGEFULLYCONNECTED
#define GEN_PASS_DEF_MERGEFULLYCONNECTED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Experimental value to ensure the split of matmul is beneficial.
constexpr int64_t DEFAULT_GROUP_SIZE = 8;

struct UnrolledMatMulBranch {
    mlir::Operation* input;
    mlir::Operation* weights;
    IE::FullyConnectedOp matMul;
};

template <class ConcreteOp>
std::optional<ConcreteOp> getSingleUseParent(mlir::Value value) {
    auto parent = value.getDefiningOp<ConcreteOp>();
    if (parent == nullptr || !parent->hasOneUse()) {
        return std::nullopt;
    }
    return parent;
}

template <class ConcreteOp>
std::optional<ConcreteOp> getSingleUser(mlir::Value value, bool checkOneUse = true) {
    if (!value.hasOneUse()) {
        return std::nullopt;
    }
    auto user = mlir::dyn_cast<ConcreteOp>(*value.getUsers().begin());
    if (user == nullptr) {
        return std::nullopt;
    }
    if (checkOneUse && !user->hasOneUse()) {
        return std::nullopt;
    }

    return user;
}
bool shapeEqualsToOne(ShapeRef shape) {
    return llvm::all_of(shape, [](const auto& dimSize) {
        return dimSize == 1;
    });
}

/*
Convert subgraph:

  FakeQuantize             FakeQuantize
    [1, C, H]               [1, C, H]
       |                        |
    Reshape    Slice         Reshape    Slice
    [C, H]     [1, C]        [C, H]     [1, C]
       |         |              |         |
   Transpose     |          Transpose     |
    [H, C]       |              [H, C]    |
       \        /                \        /
         FC(0)     ...             FC(N-1)
        [1, H]                    [1, H]
          |                         |
        Reshape                  Reshape
      [1, 1, 1, H]            [1, 1, 1, H]
               \             /
                   Concat
                 [1, N, 1, H]

    To:

  WeightsConcat                  WeightsConcat
   [subN, C, H]                     [subN, C, H]
      |                               |
   Transpose                      Transpose
   [subN, H, C]                     [subN, H, C]
      |                               |
    Reshape      InputConcat     Reshape      InputConcat
 [subN*H, C]      [1, subN*C]   [subN*H, C]      [1, subN*C]
      |            |               |            |
   FakeQuantize   Reshape        FakeQuantize   Reshape
 [subN*H, C]      [subN, C]        [subN*H, C]      [subN, C]
       \        /                     \        /
          FC              ...             FC
    [subN, subN*H]                     [subN, subN*H]
        /       \                      /       \
      Slice0   Slice(subN-1)       Slice0   Slice(subN-1)
      [1, H]    [1, H]            [1, H]    [1, H]
       |          |                 |          |
     Reshape    Reshape           Reshape    Reshape
  [1, 1, 1, H]  [1, 1, 1, H]     [1, 1, 1, H]  [1, 1, 1, H]
       \           \               /           /
                         Concat
                      [1, N, 1, H]

*/
class MergeFullyConnectedWithWeightsAsConstant : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    MergeFullyConnectedWithWeightsAsConstant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
        setDebugName("MergeFullyConnectedWithWeightsAsConstant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const override;

protected:
    virtual std::optional<SmallVector<UnrolledMatMulBranch>> getUnrolledMatMulBranch(IE::FullyConnectedOp origOp) const;
    virtual mlir::Value getMatMulInputSource(IE::FullyConnectedOp origOp) const;
    virtual IE::ConcatOp getMatMulOutputConcat(IE::FullyConnectedOp origOp) const;
    virtual std::optional<mlir::Operation*> getMatMulWeights(IE::FullyConnectedOp origOp, ShapeRef expectedOutShape,
                                                             IE::ConcatOp concatOutput) const;
    virtual bool validateUnrolledMatMulBranch(SmallVector<UnrolledMatMulBranch>& matMulBranches) const;
    virtual mlir::Value buildNewMatMulInput(ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx,
                                            size_t batchOffset, size_t batchSize,
                                            mlir::PatternRewriter& rewriter) const;
    virtual mlir::Value buildNewMatMulWeights(ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx,
                                              size_t batchOffset, size_t batchSize,
                                              mlir::PatternRewriter& rewriter) const;
    virtual SmallVector<mlir::Value> sliceNewMatMulOutput(mlir::Value newMatMulOutput, size_t batchIdx,
                                                          int64_t sliceSize, mlir::PatternRewriter& rewriter) const;
    virtual mlir::Value reshapeOutputSlice(mlir::Value sliceOut, mlir::PatternRewriter& rewriter) const;
    virtual void cleanUpMatMulBranches(ArrayRef<UnrolledMatMulBranch> branches, mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

mlir::LogicalResult MergeFullyConnectedWithWeightsAsConstant::matchAndRewrite(IE::FullyConnectedOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    auto unrolledMatMulBranch = getUnrolledMatMulBranch(origOp);
    if (!unrolledMatMulBranch.has_value()) {
        _log.trace("Can not find unrolled matmul pattern");
        return mlir::failure();
    }

    // Get the input source
    auto source = getMatMulInputSource(origOp);
    const auto matMulOutShape = getShape(origOp.getResult());
    const auto OC = matMulOutShape.back();
    if (OC >= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }

    auto unrolledMatMulBranchValue = unrolledMatMulBranch.value();
    if (!validateUnrolledMatMulBranch(unrolledMatMulBranchValue)) {
        return mlir::failure();
    }

    const auto groupSize = unrolledMatMulBranchValue.size();
    const auto& subBatchSize = DEFAULT_GROUP_SIZE;
    const auto batchNum = divUp(groupSize, static_cast<size_t>(subBatchSize));
    const auto remainderBatchSize = groupSize - (batchNum - 1) * subBatchSize;
    const auto& outputSliceSize = OC;

    SmallVector<mlir::Value> slices;
    rewriter.setInsertionPointAfterValue(source);
    size_t batchOffset = 0;
    for (auto batchId : irange(batchNum)) {
        auto currentBatchSize = batchId == batchNum - 1 ? remainderBatchSize : subBatchSize;
        _log.trace("Create new matmul for batch {0}, batch size {1}", batchId, currentBatchSize);

        // create new input
        auto newInput =
                buildNewMatMulInput(unrolledMatMulBranchValue, batchId, batchOffset, currentBatchSize, rewriter);
        // create new weights
        auto newWeights =
                buildNewMatMulWeights(unrolledMatMulBranchValue, batchId, batchOffset, currentBatchSize, rewriter);
        // create new MatMul
        auto insertPoint =
                newInput.getDefiningOp()->isBeforeInBlock(newWeights.getDefiningOp()) ? newWeights : newInput;
        rewriter.setInsertionPointAfterValue(insertPoint);
        auto newMatMulOp = rewriter.create<IE::FullyConnectedOp>(
                appendLoc(origOp.getLoc(), "_matmul_batch_{0}", batchId), newInput, newWeights, origOp.getBias());
        // Slice out expected result [1, OC] from output [batchSize, batchSize*OC]
        auto output = sliceNewMatMulOutput(newMatMulOp.getResult(), batchId, outputSliceSize, rewriter);

        slices.append(output);
        batchOffset += currentBatchSize;
    }
    auto outConcat = getMatMulOutputConcat(origOp);
    _log.trace("Replace concat at {0}", outConcat->getLoc());
    auto concatAxis = getConcatAxis(outConcat);
    VPUX_THROW_UNLESS(concatAxis.has_value(), "Can not get concat axis at '{0}'", outConcat.getLoc());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(outConcat, slices, concatAxis.value());
    cleanUpMatMulBranches(unrolledMatMulBranchValue, rewriter);
    return mlir::success();
}

std::optional<SmallVector<UnrolledMatMulBranch>> MergeFullyConnectedWithWeightsAsConstant::getUnrolledMatMulBranch(
        IE::FullyConnectedOp origOp) const {
    auto inSlice = getSingleUseParent<IE::SliceOp>(origOp.getInput());
    if (!inSlice.has_value()) {
        _log.trace("SliceOp is not found on input");
        return std::nullopt;
    }
    auto outReshape = getSingleUser<IE::ReshapeOp>(origOp.getResult());
    if (!outReshape.has_value()) {
        _log.trace("ReshapeOp is not found on output");
        return std::nullopt;
    }
    auto maybeOutConcat = getSingleUser<IE::ConcatOp>(outReshape.value().getResult(), false);
    if (!maybeOutConcat.has_value()) {
        _log.trace("ConcatOp is not found on output");
        return std::nullopt;
    }

    auto outConcat = maybeOutConcat.value();
    auto concatAxis = getConcatAxis(outConcat);
    if (!concatAxis.has_value()) {
        _log.trace("Concat is not on single axis");
        return std::nullopt;
    }
    const auto matMulOutShape = getShape(origOp.getResult());
    if (matMulOutShape[Dim(0)] != 1) {
        return std::nullopt;
    }

    SmallVector<UnrolledMatMulBranch> matMulBranch;
    // Check if the input source is split into multi branches with same matmul pattern:
    auto source = inSlice.value().getSource();
    for (auto user : source.getUsers()) {
        auto maybeSlice = mlir::dyn_cast<IE::SliceOp>(user);
        if (maybeSlice == nullptr) {
            return std::nullopt;
        }
        auto maybeMatMul = getSingleUser<IE::FullyConnectedOp>(maybeSlice);
        if (!maybeMatMul.has_value()) {
            return std::nullopt;
        }
        auto matMul = maybeMatMul.value();
        auto fakeQuantize = getMatMulWeights(matMul, matMulOutShape, outConcat);
        if (!fakeQuantize.has_value()) {
            return std::nullopt;
        }
        matMulBranch.push_back({maybeSlice, fakeQuantize.value(), matMul});
    }
    return matMulBranch;
}

mlir::Value MergeFullyConnectedWithWeightsAsConstant::getMatMulInputSource(IE::FullyConnectedOp origOp) const {
    auto sliceOp = origOp.getInput().getDefiningOp<IE::SliceOp>();
    VPUX_THROW_WHEN(sliceOp == nullptr, "Unexpected input op type at {0}", origOp->getLoc());
    return sliceOp.getSource();
}

IE::ConcatOp MergeFullyConnectedWithWeightsAsConstant::getMatMulOutputConcat(IE::FullyConnectedOp origOp) const {
    auto outReshape = getSingleUser<IE::ReshapeOp>(origOp.getResult()).value();
    return getSingleUser<IE::ConcatOp>(outReshape.getResult(), false).value();
}

std::optional<mlir::Operation*> MergeFullyConnectedWithWeightsAsConstant::getMatMulWeights(
        IE::FullyConnectedOp origOp, ShapeRef expectedOutShape, IE::ConcatOp concatOutput) const {
    // Matmul op is expected to have 2D input and output. so need check the rank of input and output
    // Left-hand matrix must have exactly two dimensions.
    const auto lhs = origOp.getInput();
    const auto lhsType = mlir::dyn_cast<vpux::NDTypeInterface>(lhs.getType());
    if (lhsType.getRank() != 2) {
        return std::nullopt;
    }
    // Right-hand matrix must have exactly two dimensions.
    auto rhs = origOp.getWeights();
    const auto rhsType = mlir::dyn_cast<vpux::NDTypeInterface>(rhs.getType());
    if (rhsType.getRank() != 2) {
        return std::nullopt;
    }
    auto outShape = getShape(origOp.getOutput());
    if (outShape != expectedOutShape) {
        return std::nullopt;
    }
    // Check that the producer of the left-hand matrix is IE.Slice
    auto inSlice = getSingleUseParent<IE::SliceOp>(origOp.getInput());
    if (!inSlice.has_value()) {
        return std::nullopt;
    }
    // Check that Slice offset, which should only slice on the second dim
    auto inSliceShape = getShape(inSlice.value().getSource());
    auto offsets = parseIntArrayAttr<int64_t>(inSlice.value().getStaticOffsetsAttr());
    auto sizes = parseIntArrayAttr<int64_t>(inSlice.value().getStaticSizesAttr());
    if (offsets.size() != 2 || sizes[0] != inSliceShape[Dim(0)]) {
        return std::nullopt;
    }

    // Check that the producer of the right-hand matrix is IE.Transpose
    auto weightsTranspose = getSingleUseParent<IE::TransposeOp>(origOp.getWeights());
    if (!weightsTranspose.has_value()) {
        return std::nullopt;
    }
    // Check that transpose transforms [d0, d1] shape into [d1, d0]
    auto orderValue = weightsTranspose.value().getOrderValue();
    if (!orderValue.has_value() || DimsOrder::fromAffineMap(orderValue.value()) != DimsOrder::CN) {
        return std::nullopt;
    }
    // Check that the producer of the Transpose is IE.ReshapeOp
    auto weightsReshape = getSingleUseParent<IE::ReshapeOp>(weightsTranspose.value().getInput());
    if (!weightsReshape.has_value()) {
        return std::nullopt;
    }
    // Check that reshape collapses [1,  d1, d2] shape into [d1, d2]
    const auto reshapeInputDims = getShape(weightsReshape.value().getInput());
    if (reshapeInputDims.size() != 3 || reshapeInputDims[Dim(0)] != 1) {
        return std::nullopt;
    }
    const Shape expectedShape = {reshapeInputDims[Dim(0)] * reshapeInputDims[Dim(1)], reshapeInputDims[Dim(2)]};
    const auto reshapeOutputDims = getShape(weightsReshape.value().getOutput());
    if (reshapeOutputDims != expectedShape) {
        return std::nullopt;
    }
    auto fq = getSingleUseParent<IE::FakeQuantizeOp>(weightsReshape.value().getInput());
    if (!fq.has_value()) {
        return std::nullopt;
    }

    auto reshapeOutput = mlir::dyn_cast<IE::ReshapeOp>(*origOp->getUsers().begin());
    if (reshapeOutput == nullptr || !reshapeOutput->hasOneUse()) {
        return std::nullopt;
    }

    auto concat = mlir::dyn_cast<IE::ConcatOp>(*reshapeOutput->getUsers().begin());
    if (concat.getOperation() != concatOutput.getOperation()) {
        return std::nullopt;
    }
    return fq.value().getOperation();
}

bool MergeFullyConnectedWithWeightsAsConstant::validateUnrolledMatMulBranch(
        SmallVector<UnrolledMatMulBranch>& matMulBranches) const {
    if (matMulBranches.empty()) {
        return false;
    }

    llvm::sort(matMulBranches, [&](UnrolledMatMulBranch& a, UnrolledMatMulBranch& b) {
        auto inputSliceA = mlir::dyn_cast<IE::SliceOp>(a.input);
        auto inputSliceB = mlir::dyn_cast<IE::SliceOp>(b.input);
        auto offsetsA = parseIntArrayAttr<int64_t>(inputSliceA.getStaticOffsetsAttr());
        auto offsetsB = parseIntArrayAttr<int64_t>(inputSliceB.getStaticOffsetsAttr());
        return offsetsA[1] < offsetsB[1];
    });

    // Check the input source is split into slice users
    for (auto idx : irange(matMulBranches.size() - 1)) {
        auto inputSlice = mlir::dyn_cast<IE::SliceOp>(matMulBranches[idx].input);
        auto nextInputSlice = mlir::dyn_cast<IE::SliceOp>(matMulBranches[idx + 1].input);
        auto offsets = parseIntArrayAttr<int64_t>(inputSlice.getStaticOffsetsAttr());
        auto size = parseIntArrayAttr<int64_t>(inputSlice.getStaticSizesAttr());
        auto nextOffsets = parseIntArrayAttr<int64_t>(nextInputSlice.getStaticOffsetsAttr());
        if (size[1] + offsets[1] != nextOffsets[1]) {
            return false;
        }
    }

    // Check FakeQuantize operands
    auto checkWeightConstant = [&](auto idx) {
        auto operand = matMulBranches[0].weights->getOperand(idx);
        auto shape = getShape(operand);
        if (shapeEqualsToOne(shape)) {
            auto hasSameConst = llvm::all_of(matMulBranches, [&](auto& branch) {
                auto curConstInput = branch.weights->getOperand(idx);
                return curConstInput == operand;
            });
            return hasSameConst;
        }
        return true;
    };

    auto getSplatZp = [&](IE::FakeQuantizeOp fq) -> mlir::FailureOr<int64_t> {
        auto inLowConst = fq.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHighConst = fq.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLowConst = fq.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHighConst = fq.getOutputHigh().getDefiningOp<Const::DeclareOp>();

        if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
            return mlir::failure();
        }

        auto outLowConstAttr = outLowConst.getContentAttr();
        auto outHighConstAttr = outHighConst.getContentAttr();
        auto scalesAndZeroPoints = getScalesAndZeroPointsFromContentAttr(
                outLowConstAttr, outHighConstAttr, fq.getAutoBroadcast(), fq.getLevels(), nullptr,
                Const::hasNegativeValues(inLowConst.getContent()), _log);
        if (mlir::failed(scalesAndZeroPoints)) {
            return mlir::failure();
        }

        const auto& zeroPoints = std::get<1>(scalesAndZeroPoints.value());
        const auto zpFront = zeroPoints.front();
        const bool allZpEqual = std::all_of(zeroPoints.begin(), zeroPoints.end(), [zpFront](auto zp) {
            return zp == zpFront;
        });
        if (allZpEqual) {
            return zpFront;
        }

        return mlir::failure();
    };

    auto firstFq = mlir::dyn_cast<IE::FakeQuantizeOp>(matMulBranches[0].weights);
    if (firstFq == nullptr) {
        return false;
    }

    auto firstSplatZp = getSplatZp(firstFq);
    if (mlir::failed(firstSplatZp)) {
        return false;
    }

    // All merged zero-points should be the same since HW doesn't support multi-zp case
    auto allZpIsSplatAndEqual =
            std::all_of(matMulBranches.begin(), matMulBranches.end(), [&](const UnrolledMatMulBranch& branch) -> bool {
                auto fq = mlir::dyn_cast<IE::FakeQuantizeOp>(branch.weights);
                if (fq == nullptr) {
                    return false;
                }

                auto splatZp = getSplatZp(fq);
                if (mlir::failed(splatZp)) {
                    return false;
                }
                return splatZp.value() == firstSplatZp.value();
            });
    if (!allZpIsSplatAndEqual) {
        return false;
    }

    const auto operandSize = matMulBranches[0].weights->getNumOperands();
    for (auto operandIdx : irange(operandSize)) {
        if (!checkWeightConstant(operandIdx)) {
            return false;
        }
    }
    return true;
}

mlir::Value MergeFullyConnectedWithWeightsAsConstant::buildNewMatMulInput(ArrayRef<UnrolledMatMulBranch> branches,
                                                                          size_t batchIdx, size_t batchOffset,
                                                                          size_t batchSize,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto matMul = branches[0].matMul;
    auto source = getMatMulInputSource(matMul);
    const auto sourceShape = getShape(source);
    const auto IC = getShape(matMul.getInput())[Dim(1)];

    SmallVector<int64_t> sliceOffsets(sourceShape.size(), 0);
    sliceOffsets[1] = batchOffset * IC;
    SmallVector<int64_t> sliceSizes = to_small_vector(getShape(source));
    sliceSizes[1] = batchSize * IC;

    auto newSlice = rewriter.create<IE::SliceOp>(appendLoc(source.getLoc(), "_slice_{0}", batchIdx), source,
                                                 getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, sliceSizes));

    SmallVector<int64_t> newInputShape{checked_cast<int64_t>(batchSize), IC};
    return rewriter.create<IE::ReshapeOp>(appendLoc(newSlice->getLoc(), "_reshape_{0}", batchIdx), newSlice, nullptr,
                                          false, getIntArrayAttr(ctx, newInputShape));
}

mlir::Value MergeFullyConnectedWithWeightsAsConstant::buildNewMatMulWeights(ArrayRef<UnrolledMatMulBranch> branches,
                                                                            size_t batchIdx, size_t batchOffset,
                                                                            size_t batchSize,
                                                                            mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto insertTransposeReshapeBeforeFQ = [&](ArrayRef<mlir::Value> values, const int64_t subBatchId) -> mlir::Value {
        auto origInShape = getShape(values.front());
        VPUX_THROW_WHEN(origInShape.size() != 3, "Input shape must have 3 dimensions");
        mlir::Value newInput = values.front();

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        if (!shapeEqualsToOne(origInShape)) {
            // Create new input
            rewriter.setInsertionPointAfterValue(values.back());
            newInput = rewriter.create<IE::ConcatOp>(appendLoc(values.front().getLoc(), "_concat_{0}", subBatchId),
                                                     values, Dim(0))
                               .getResult();
        }

        // Create transpose
        rewriter.setInsertionPointAfterValue(newInput);
        auto perm = SmallVector<uint32_t>{0, 2, 1};
        const auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, ctx));
        auto transpose = rewriter.createOrFold<IE::TransposeOp>(
                appendLoc(newInput.getLoc(), "transpose_{0}", subBatchId), newInput, nullptr, orderAttr);

        // create affineReshape
        auto inShape = getShape(transpose).raw();
        auto reshapeOutShape = SmallVector<int64_t>{inShape[0] * inShape[1], inShape[2]};
        const auto reshapeOutShapeAttr = getIntArrayAttr(ctx, reshapeOutShape);
        SmallVector<SmallVector<int64_t>> inDimMapping{{0}, {0}, {1}};
        return rewriter.createOrFold<IE::AffineReshapeOp>(appendLoc(newInput.getLoc(), "reshape_{0}", subBatchId),
                                                          transpose, getIntArrayOfArray(ctx, inDimMapping),
                                                          reshapeOutShapeAttr);
    };

    SmallVector<mlir::Value> fqInputs, inLows, inHighs, outLows, outHighs;
    for (auto i : irange(batchSize)) {
        auto fq = mlir::dyn_cast<IE::FakeQuantizeOp>(branches[batchOffset + i].weights);
        fqInputs.push_back(fq.getInput());
        inLows.push_back(fq.getInputLow());
        inHighs.push_back(fq.getInputHigh());
        outLows.push_back(fq.getOutputLow());
        outHighs.push_back(fq.getOutputHigh());
    }

    auto newFQInput = insertTransposeReshapeBeforeFQ(fqInputs, batchIdx);
    auto newInLow = insertTransposeReshapeBeforeFQ(inLows, batchIdx);
    auto newInHigh = insertTransposeReshapeBeforeFQ(inHighs, batchIdx);
    auto newOutLow = insertTransposeReshapeBeforeFQ(outLows, batchIdx);
    auto newOutHigh = insertTransposeReshapeBeforeFQ(outHighs, batchIdx);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto firstFq = mlir::dyn_cast<IE::FakeQuantizeOp>(branches[batchIdx].weights);
    rewriter.setInsertionPointAfterValue(firstFq);
    auto newFakeQuantizedLoc = appendLoc(firstFq.getLoc(), "_fused_{0}", batchIdx);
    auto newFakeQuantizedOp = rewriter.create<IE::FakeQuantizeOp>(
            newFakeQuantizedLoc, newFQInput, newInLow, newInHigh, newOutLow, newOutHigh, firstFq.getLevelsAttr(),
            firstFq.getLowFpTypeAttr(), firstFq.getAutoBroadcastAttr());
    return newFakeQuantizedOp.getResult();
}

SmallVector<mlir::Value> MergeFullyConnectedWithWeightsAsConstant::sliceNewMatMulOutput(
        mlir::Value newMatMulOutput, size_t batchIdx, int64_t sliceSize, mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    SmallVector<mlir::Value> outputs;
    const auto sliceNum = getShape(newMatMulOutput)[Dim(0)];
    for (auto idx : irange(sliceNum)) {
        const auto sliceLoc = appendLoc(newMatMulOutput.getLoc(), "_slice_{0}_{1}", batchIdx, idx);
        SmallVector<int64_t> sliceOffsets = {idx, idx * sliceSize};
        SmallVector<int64_t> sliceSizes = {1, sliceSize};
        auto slice = rewriter.create<IE::SliceOp>(sliceLoc, newMatMulOutput, getIntArrayAttr(ctx, sliceOffsets),
                                                  getIntArrayAttr(ctx, sliceSizes));

        _log.trace("create new slice {0}", slice->getLoc());
        auto reshape = reshapeOutputSlice(slice, rewriter);
        outputs.push_back(reshape);
    }
    return outputs;
}

mlir::Value MergeFullyConnectedWithWeightsAsConstant::reshapeOutputSlice(mlir::Value sliceOut,
                                                                         mlir::PatternRewriter& rewriter) const {
    // new output shape size is always 2, reshape to 4
    auto shape = getShape(sliceOut);
    SmallVector<int64_t> newShape{1, 1, shape[Dim(0)], shape[Dim(1)]};
    auto ctx = rewriter.getContext();
    const auto newLoc = appendLoc(sliceOut.getLoc(), "_reshape");
    return rewriter.create<IE::ReshapeOp>(newLoc, sliceOut, nullptr, false, getIntArrayAttr(ctx, newShape));
}

void MergeFullyConnectedWithWeightsAsConstant::cleanUpMatMulBranches(ArrayRef<UnrolledMatMulBranch> branches,
                                                                     mlir::PatternRewriter& rewriter) const {
    for (auto& branch : branches) {
        auto user = *branch.matMul->getUsers().begin();
        rewriter.eraseOp(user);
        rewriter.eraseOp(branch.matMul);
    }
}

/*
Convert subgraph:

 Weights      Input       Weights      Input
[N, C, H]    [N, 1, H]  [N, C, H]    [N, 1, H]
     |          |            |          |
  Split       Split        Split       Split
[1, C, H]    [1, 1, H]  [1, C, H]    [1, 1, H]
    |           |            |          |
 Convert        |        Convert        |
[1, C, H]       |       [1, C, H]       |
    |           |            |          |
 Reshape      Reshape     Reshape     Reshape
 [C, H]      [1, H]        [C, H]      [1, H]
    \        /               \        /
        FC(0)    ...           FC(N-1)
        [1, C]                [1, C]
          |                     |
       Reshape               Reshape
      [1, 1, C]             [1, 1, C]
           \                /
                 Concat
                [N, 1, C]
                   |

To:

 Weights      Input       Weights      Input
[N, C, H]    [N, 1, H]  [N, C, H]    [N, 1, H]
     |          |            |          |
  Slice        Slice        Slice      Slice
[subN, C, H][subN, 1, H] [subN, C, H] [subN, 1, H]
    |           |            |          |
 Convert        |        Convert        |
[subN, C, H]    |       [subN, C, H]    |
    |           |            |          |
 Reshape      Reshape     Reshape     Reshape
[subN*C, H]  [subN, H]    [subN*C, H]  [subN, H]
    \        /               \        /
        FC           ...        FC
    [subN, subN*C]          [subN,subN*C]
      /       \               /       \
  Slice0   Slice(subN-1)  Slice0   Slice(subN-1)
   [1, C]    [1, C]        [1, C]    [1, C]
    |          |             |          |
  Reshape    Reshape      Reshape    Reshape
 [1, 1, C]  [1, 1, C]    [1, 1, C]  [1, 1, H]
       \         \         /           /
                 Concat
                [N, 1, C]
                   |
*/

class MergeFullyConnectedForDQPatternWithConvert final : public MergeFullyConnectedWithWeightsAsConstant {
public:
    MergeFullyConnectedForDQPatternWithConvert(mlir::MLIRContext* ctx, Logger log)
            : MergeFullyConnectedWithWeightsAsConstant(ctx, log) {
        setDebugName("MergeFullyConnectedForDQPatternWithConvert");
    }

protected:
    std::optional<SmallVector<UnrolledMatMulBranch>> getUnrolledMatMulBranch(
            IE::FullyConnectedOp origOp) const override;
    mlir::Value getMatMulInputSource(IE::FullyConnectedOp origOp) const override;
    IE::ConcatOp getMatMulOutputConcat(IE::FullyConnectedOp origOp) const override;
    std::optional<mlir::Operation*> getMatMulWeights(IE::FullyConnectedOp origOp, ShapeRef expectedOutShape,
                                                     IE::ConcatOp concatOutput) const override;
    bool validateUnrolledMatMulBranch(SmallVector<UnrolledMatMulBranch>& matMulBranches) const override;
    mlir::Value buildNewMatMulInput(ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx, size_t batchOffset,
                                    size_t batchSize, mlir::PatternRewriter& rewriter) const override;
    mlir::Value buildNewMatMulWeights(ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx, size_t batchOffset,
                                      size_t batchSize, mlir::PatternRewriter& rewriter) const override;
    mlir::Value reshapeOutputSlice(mlir::Value sliceOut, mlir::PatternRewriter& rewriter) const override;
    void cleanUpMatMulBranches(ArrayRef<UnrolledMatMulBranch> branches, mlir::PatternRewriter& rewriter) const override;
};

std::optional<SmallVector<UnrolledMatMulBranch>> MergeFullyConnectedForDQPatternWithConvert::getUnrolledMatMulBranch(
        IE::FullyConnectedOp origOp) const {
    auto inReshape = getSingleUseParent<IE::AffineReshapeOp>(origOp.getInput());
    if (!inReshape.has_value()) {
        _log.trace("AffineReshapeOp is not found on input");
        return std::nullopt;
    }
    auto inSplit = inReshape.value().getInput().getDefiningOp<IE::SplitOp>();
    if (inSplit == nullptr) {
        _log.trace("SplitOp is not found on input");
        return std::nullopt;
    }
    auto weightsReshape = getSingleUseParent<IE::AffineReshapeOp>(origOp.getWeights());
    if (!weightsReshape.has_value()) {
        _log.trace("AffineReshapeOp is not found on weights");
        return std::nullopt;
    }

    auto convert = getSingleUseParent<IE::ConvertOp>(weightsReshape.value().getInput());
    if (!convert.has_value()) {
        _log.trace("convertOp is not found on weights");
        return std::nullopt;
    }
    auto weightsSplit = convert.value().getInput().getDefiningOp<IE::SplitOp>();
    if (weightsSplit == nullptr) {
        _log.trace("SplitOp is not found on weights");
        return std::nullopt;
    }

    auto reshapeUser = getSingleUser<IE::AffineReshapeOp>(origOp);
    if (!reshapeUser.has_value()) {
        _log.trace("AffineReshapeOp is not found on output");
        return std::nullopt;
    }

    auto outConcat = getSingleUser<IE::ConcatOp>(reshapeUser.value(), false);
    if (!outConcat.has_value()) {
        _log.trace("concat is not found on output");
        return std::nullopt;
    }

    auto concatAxis = getConcatAxis(outConcat.value());
    if (!concatAxis.has_value()) {
        _log.trace("Concat is not on single axis");
        return std::nullopt;
    }

    const auto matMulOutShape = getShape(origOp.getResult());
    if (matMulOutShape[Dim(0)] != 1) {
        _log.trace("output shape {0} is not match", matMulOutShape);
        return std::nullopt;
    }

    // Check if the source is split into multi branches with same matmul pattern
    SmallVector<UnrolledMatMulBranch> unrolledMatMulBranch;
    for (auto user : inSplit->getUsers()) {
        auto maybeReshape = mlir::dyn_cast<IE::AffineReshapeOp>(user);
        if (maybeReshape == nullptr || !maybeReshape->hasOneUse()) {
            _log.trace("reshape is not found");
            return std::nullopt;
        }
        auto maybeMatMul = getSingleUser<IE::FullyConnectedOp>(maybeReshape);
        if (!maybeMatMul.has_value()) {
            return std::nullopt;
        }

        auto matMul = maybeMatMul.value();
        auto split = getMatMulWeights(matMul, matMulOutShape, outConcat.value());
        if (!split.has_value() || split != weightsSplit.getOperation()) {
            return std::nullopt;
        }
        unrolledMatMulBranch.push_back({inSplit, split.value(), matMul});
    }
    auto weightsSplitUserSize = std::distance(weightsSplit->getUsers().begin(), weightsSplit->getUsers().end());
    if (unrolledMatMulBranch.size() != static_cast<size_t>(weightsSplitUserSize)) {
        _log.trace("Input split user and weight split user size are not matched");
        return std::nullopt;
    }
    return unrolledMatMulBranch;
}

mlir::Value MergeFullyConnectedForDQPatternWithConvert::getMatMulInputSource(IE::FullyConnectedOp origOp) const {
    auto reshapeOp = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    auto splitOp = reshapeOp.getInput().getDefiningOp<IE::SplitOp>();
    return splitOp.getInput();
}

IE::ConcatOp MergeFullyConnectedForDQPatternWithConvert::getMatMulOutputConcat(IE::FullyConnectedOp origOp) const {
    auto outReshape = getSingleUser<IE::AffineReshapeOp>(origOp.getResult()).value();
    return getSingleUser<IE::ConcatOp>(outReshape.getResult(), false).value();
}

std::optional<mlir::Operation*> MergeFullyConnectedForDQPatternWithConvert::getMatMulWeights(
        IE::FullyConnectedOp origOp, ShapeRef expectedOutShape, IE::ConcatOp concatOutput) const {
    // Matmul op is expected to have 2D input and output. so need check the rank of input and output
    // Left-hand matrix must have exactly two dimensions.
    const auto lhs = origOp.getInput();
    const auto lhsType = mlir::dyn_cast<vpux::NDTypeInterface>(lhs.getType());
    if (lhsType.getRank() != 2) {
        _log.trace("lhsType is not suitable", lhsType);
        return std::nullopt;
    }
    // Right-hand matrix must have exactly two dimensions.
    auto rhs = origOp.getWeights();
    const auto rhsType = mlir::dyn_cast<vpux::NDTypeInterface>(rhs.getType());
    if (rhsType.getRank() != 2) {
        _log.trace("rhsType is not suitable", rhsType);
        return std::nullopt;
    }
    auto outShape = getShape(origOp.getOutput());
    if (outShape != expectedOutShape) {
        _log.trace("outShape is not suitable", outShape);
        return std::nullopt;
    }

    auto inReshape = getSingleUseParent<IE::AffineReshapeOp>(origOp.getInput());
    if (!inReshape.has_value()) {
        _log.trace("AffineReshapeOp is not found on input");
        return std::nullopt;
    }
    auto inSplit = inReshape.value().getInput().getDefiningOp<IE::SplitOp>();
    if (inSplit == nullptr) {
        _log.trace("SplitOp is not found on input");
        return std::nullopt;
    }

    auto weightsReshape = getSingleUseParent<IE::AffineReshapeOp>(origOp.getWeights());
    if (!weightsReshape.has_value()) {
        _log.trace("AffineReshapeOp is not found on weights");
        return std::nullopt;
    }
    // Check that reshape collapses [1,  d1, d2] shape into [d1, d2]
    const auto reshapeInputDims = getShape(weightsReshape.value().getInput());
    if (reshapeInputDims.size() != 3 || reshapeInputDims[Dim(0)] != 1) {
        _log.trace("reshapeInputDims is not suitable {0}", reshapeInputDims);
        return std::nullopt;
    }
    const Shape expectedShape = {reshapeInputDims[Dim(0)] * reshapeInputDims[Dim(1)], reshapeInputDims[Dim(2)]};
    const auto reshapeOutputDims = getShape(weightsReshape.value().getOutput());
    if (reshapeOutputDims != expectedShape) {
        _log.trace("reshapeOutputDims is not suitable {0}", reshapeOutputDims);
        return std::nullopt;
    }

    auto convert = getSingleUseParent<IE::ConvertOp>(weightsReshape.value().getInput());
    if (!convert.has_value()) {
        _log.trace("convertOp is not found on weights");
        return std::nullopt;
    }
    auto dstType = convert.value().getDstElemType();
    if (!dstType.isF16()) {
        _log.trace("convertOp is not convert to fp16 on weights");
        return std::nullopt;
    }

    auto weightsSplit = convert.value().getInput().getDefiningOp<IE::SplitOp>();
    if (weightsSplit == nullptr) {
        _log.trace("SplitOp is not found on weights");
        return std::nullopt;
    }

    auto reshapeUser = mlir::dyn_cast<IE::AffineReshapeOp>(*origOp->getUsers().begin());
    if (reshapeUser == nullptr || !reshapeUser->hasOneUse()) {
        _log.trace("AffineReshapeOp is not found on output");
        return std::nullopt;
    }
    auto concat = mlir::dyn_cast<IE::ConcatOp>(*reshapeUser->getUsers().begin());
    if (concat != concatOutput.getOperation()) {
        return std::nullopt;
    }
    return weightsSplit;
}

bool MergeFullyConnectedForDQPatternWithConvert::validateUnrolledMatMulBranch(
        SmallVector<UnrolledMatMulBranch>&) const {
    // The input and weights split op are already checked in getUnrolledMatMulBranch, so don't need any other checks
    // here
    return true;
}

mlir::Value MergeFullyConnectedForDQPatternWithConvert::buildNewMatMulInput(ArrayRef<UnrolledMatMulBranch> branches,
                                                                            size_t batchIdx, size_t batchOffset,
                                                                            size_t batchSize,
                                                                            mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto matMul = branches[0].matMul;
    auto source = getMatMulInputSource(matMul);

    SmallVector<int64_t> inSliceOffsets(getShape(source).size(), 0);
    inSliceOffsets[0] = batchOffset;
    SmallVector<int64_t> inSliceSizes = to_small_vector(getShape(source));
    inSliceSizes[0] = batchSize;
    auto slice = rewriter.create<IE::SliceOp>(appendLoc(source.getLoc(), "_slice_{0}", batchIdx), source,
                                              getIntArrayAttr(ctx, inSliceOffsets), getIntArrayAttr(ctx, inSliceSizes));

    auto sliceOutShape = to_small_vector(getShape(slice.getResult()));
    SmallVector<int64_t> newInputShape{sliceOutShape[0] * sliceOutShape[1], sliceOutShape[2]};
    const auto reshapeOutShapeAttr = getIntArrayAttr(ctx, newInputShape);
    SmallVector<SmallVector<int64_t>> inDimMapping{{0}, {0}, {1}};
    return rewriter.createOrFold<IE::AffineReshapeOp>(appendLoc(slice.getLoc(), "_reshape"), slice,
                                                      getIntArrayOfArray(ctx, inDimMapping), reshapeOutShapeAttr);
}

mlir::Value MergeFullyConnectedForDQPatternWithConvert::buildNewMatMulWeights(ArrayRef<UnrolledMatMulBranch> branches,
                                                                              size_t batchIdx, size_t batchOffset,
                                                                              size_t batchSize,
                                                                              mlir::PatternRewriter& rewriter) const {
    auto split = mlir::dyn_cast<IE::SplitOp>(branches[0].weights);
    auto source = split.getInput();
    auto sourceShape = getShape(source);
    auto ctx = rewriter.getContext();

    SmallVector<int64_t> sliceOffsets(sourceShape.size(), 0);
    sliceOffsets[0] = batchOffset;
    SmallVector<int64_t> sliceSizes = to_small_vector(sourceShape);
    sliceSizes[0] = batchSize;
    auto slice = rewriter.create<IE::SliceOp>(appendLoc(source.getLoc(), "_slice_{0}", batchIdx), source,
                                              getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, sliceSizes));

    auto convert =
            rewriter.create<IE::ConvertOp>(appendLoc(slice.getLoc(), "_convert"), slice, mlir::Float16Type::get(ctx))
                    .getOutput();

    auto outShape = getShape(convert);
    SmallVector<int64_t> newWeightsShape{outShape[Dim(0)] * outShape[Dim(1)], outShape[Dim(2)]};
    SmallVector<SmallVector<int64_t>> inDimMapping{{0}, {0}, {1}};
    return rewriter.createOrFold<IE::AffineReshapeOp>(appendLoc(convert.getLoc(), "_reshape"), convert,
                                                      getIntArrayOfArray(ctx, inDimMapping),
                                                      getIntArrayAttr(ctx, newWeightsShape));
}

mlir::Value MergeFullyConnectedForDQPatternWithConvert::reshapeOutputSlice(mlir::Value sliceOut,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto shape = getShape(sliceOut);
    SmallVector<int64_t> newShape{1, shape[Dim(0)], shape[Dim(1)]};
    const auto reshapeLoc = appendLoc(sliceOut.getLoc(), "_reshape");
    return rewriter.create<IE::ReshapeOp>(reshapeLoc, sliceOut, nullptr, false, getIntArrayAttr(ctx, newShape));
}

void MergeFullyConnectedForDQPatternWithConvert::cleanUpMatMulBranches(ArrayRef<UnrolledMatMulBranch> branches,
                                                                       mlir::PatternRewriter& rewriter) const {
    auto inputSource = branches[0].input;
    auto weightSource = branches[0].weights;
    for (auto& branch : branches) {
        auto matMul = branch.matMul;
        auto inReshape = matMul.getInput().getDefiningOp<IE::AffineReshapeOp>();
        auto weightsReshape = matMul.getWeights().getDefiningOp<IE::AffineReshapeOp>();
        auto weightsConvert = weightsReshape.getInput().getDefiningOp<IE::ConvertOp>();
        auto outReshape = mlir::dyn_cast<IE::AffineReshapeOp>(*matMul->getUsers().begin());

        rewriter.eraseOp(outReshape);
        rewriter.eraseOp(matMul);
        rewriter.eraseOp(inReshape);
        rewriter.eraseOp(weightsReshape);
        rewriter.eraseOp(weightsConvert);
    }
    rewriter.eraseOp(inputSource);
    rewriter.eraseOp(weightSource);
}

class MergeFullyConnectedForDQPatternWithDequantize final : public MergeFullyConnectedWithWeightsAsConstant {
public:
    MergeFullyConnectedForDQPatternWithDequantize(mlir::MLIRContext* ctx, Logger log)
            : MergeFullyConnectedWithWeightsAsConstant(ctx, log) {
        setDebugName("MergeFullyConnectedForDQPatternWithDequantize");
    }

protected:
    std::optional<SmallVector<UnrolledMatMulBranch>> getUnrolledMatMulBranch(
            IE::FullyConnectedOp origOp) const override;
    mlir::Value getMatMulInputSource(IE::FullyConnectedOp origOp) const override;
    IE::ConcatOp getMatMulOutputConcat(IE::FullyConnectedOp origOp) const override;
    std::optional<mlir::Operation*> getMatMulWeights(IE::FullyConnectedOp origOp, ShapeRef expectedOutShape,
                                                     IE::ConcatOp concatOutput) const override;
    bool validateUnrolledMatMulBranch(SmallVector<UnrolledMatMulBranch>& matMulBranches) const override;
    mlir::Value buildNewMatMulInput(ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx, size_t batchOffset,
                                    size_t batchSize, mlir::PatternRewriter& rewriter) const override;
    mlir::Value buildNewMatMulWeights(ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx, size_t batchOffset,
                                      size_t batchSize, mlir::PatternRewriter& rewriter) const override;
    mlir::Value reshapeOutputSlice(mlir::Value sliceOut, mlir::PatternRewriter& rewriter) const override;
    void cleanUpMatMulBranches(ArrayRef<UnrolledMatMulBranch> branches, mlir::PatternRewriter& rewriter) const override;

private:
    mlir::FailureOr<IE::ConcatOp> getConcatThroughReshapeUser(IE::FullyConnectedOp origOp) const;
    std::optional<IE::SliceOp> getWeightsSourceSliceOp(IE::FullyConnectedOp origOp) const;
};

std::optional<IE::SliceOp> MergeFullyConnectedForDQPatternWithDequantize::getWeightsSourceSliceOp(
        IE::FullyConnectedOp origOp) const {
    auto validateReshape = [this](IE::ReshapeOp reshapeOp) -> bool {
        // Check that reshape collapses [1,  d1, d2] shape into [d1, d2]
        const auto reshapeInputDims = getShape(reshapeOp.getInput());
        if (reshapeInputDims.size() != 3 || reshapeInputDims[Dim(0)] != 1) {
            _log.trace("reshapeInputDims is not suitable {0}", reshapeInputDims);
            return false;
        }

        const Shape expectedShape = {reshapeInputDims[Dim(0)] * reshapeInputDims[Dim(1)], reshapeInputDims[Dim(2)]};
        const auto reshapeOutputDims = getShape(reshapeOp.getOutput());
        if (reshapeOutputDims != expectedShape) {
            _log.trace("reshapeOutputDims is not suitable {0}", reshapeOutputDims);
            return false;
        }

        return true;
    };
    auto weightsReshape = getSingleUseParent<IE::ReshapeOp>(origOp.getWeights());
    if (!weightsReshape.has_value() || !validateReshape(weightsReshape.value())) {
        _log.trace("Can't find valid ReshapeOp on weights");
        return std::nullopt;
    }

    auto dequantize = getSingleUseParent<IE::DequantizeOp>(weightsReshape.value().getInput());
    if (!dequantize.has_value()) {
        _log.trace("DequantizeOp is not found on weights");
        return std::nullopt;
    }

    auto weightsSlice = getSingleUseParent<IE::SliceOp>(dequantize.value().getInput());
    if (!weightsSlice.has_value()) {
        _log.trace("SliceOp is not found on weights");
        return std::nullopt;
    }

    return weightsSlice;
}

mlir::FailureOr<IE::ConcatOp> MergeFullyConnectedForDQPatternWithDequantize::getConcatThroughReshapeUser(
        IE::FullyConnectedOp origOp) const {
    auto reshapeUser = getSingleUser<IE::ReshapeOp>(origOp);
    if (!reshapeUser.has_value()) {
        _log.trace("ReshapeOp is not found on output");
        return mlir::failure();
    }

    auto outConcat = getSingleUser<IE::ConcatOp>(reshapeUser.value(), false);
    if (!outConcat.has_value()) {
        _log.trace("ConcatOp is not found on output");
        return mlir::failure();
    }

    return outConcat.value();
}

std::optional<SmallVector<UnrolledMatMulBranch>> MergeFullyConnectedForDQPatternWithDequantize::getUnrolledMatMulBranch(
        IE::FullyConnectedOp origOp) const {
    // Check FC input for current FC, pattern source - SliceOp is expected
    auto inSlice = getSingleUseParent<IE::SliceOp>(origOp.getInput());
    if (!inSlice.has_value()) {
        _log.trace("SliceOp is not found on input");
        return std::nullopt;
    }
    auto inSource = inSlice.value().getSource();

    // Check FC weights for current FC, pattern source - SliceOp - DequantizeOp - ReshapeOp is expected
    auto getWeightsSource = [this](IE::FullyConnectedOp origOp) -> std::optional<mlir::Value> {
        auto slice = getWeightsSourceSliceOp(origOp);
        if (!slice.has_value()) {
            return std::nullopt;
        }

        return slice.value().getSource();
    };
    auto weightsSource = getWeightsSource(origOp);
    if (!weightsSource.has_value()) {
        return std::nullopt;
    }

    // Check users for current FC, pattern FullyConnectedOp - ReshapeOp - ConcatOp is expected
    auto validateOutputPattern = [this](IE::FullyConnectedOp origOp) -> mlir::FailureOr<IE::ConcatOp> {
        auto concat = getConcatThroughReshapeUser(origOp);
        if (mlir::failed(concat)) {
            return mlir::failure();
        }

        auto concatAxis = getConcatAxis(concat.value());
        if (!concatAxis.has_value()) {
            _log.trace("Concat is not on a single axis");
            return mlir::failure();
        }

        return concat;
    };
    auto validateUsers = validateOutputPattern(origOp);
    if (mlir::failed(validateUsers)) {
        return std::nullopt;
    }
    auto outConcat = validateUsers.value();

    const auto matMulOutShape = getShape(origOp.getResult());
    if (matMulOutShape[Dim(0)] != 1) {
        _log.trace("Output shape {0} does not match", matMulOutShape);
        return std::nullopt;
    }

    // Check if the source is split into multiple branches with the same matmul pattern
    auto validateBranches = [this](mlir::Value inSource, mlir::Value weightsSource, ShapeRef matMulOutShape,
                                   SmallVector<UnrolledMatMulBranch>& unrolledMatMulBranch,
                                   IE::ConcatOp expectedOutConcat) {
        for (auto input : expectedOutConcat.getInputs()) {
            auto reshapeOp = input.getDefiningOp<IE::ReshapeOp>();
            if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
                _log.trace("The ReshapeOp is not found");
                return false;
            }

            auto matmulOp = reshapeOp.getInput().getDefiningOp<IE::FullyConnectedOp>();
            if (matmulOp == nullptr || !matmulOp->hasOneUse()) {
                _log.trace("The FullyConnectedOp is not found");
                return false;
            }

            auto sliceOp = matmulOp.getInput().getDefiningOp<IE::SliceOp>();
            if (sliceOp == nullptr || !sliceOp->hasOneUse()) {
                _log.trace("The sliceOp is not found");
                return false;
            }

            if (sliceOp.getSource() != inSource) {
                _log.trace("The input source is not the same");
                return false;
            }

            auto weights = getMatMulWeights(matmulOp, matMulOutShape, expectedOutConcat);
            if (!weights.has_value() || weights.value()->getOperand(0) != weightsSource) {
                _log.trace("The weights source is not the same");
                return false;
            }
            unrolledMatMulBranch.push_back({sliceOp, weights.value(), matmulOp});
        }

        auto weightsSplitUserSize = std::distance(weightsSource.getUsers().begin(), weightsSource.getUsers().end());
        if (unrolledMatMulBranch.size() != static_cast<size_t>(weightsSplitUserSize)) {
            _log.trace("Input split user and weight split user size are not matched");
            return false;
        }

        return true;
    };
    SmallVector<UnrolledMatMulBranch> unrolledMatMulBranch;
    if (!validateBranches(inSource, weightsSource.value(), matMulOutShape, unrolledMatMulBranch, outConcat)) {
        return std::nullopt;
    }

    return unrolledMatMulBranch;
}

mlir::Value MergeFullyConnectedForDQPatternWithDequantize::getMatMulInputSource(IE::FullyConnectedOp origOp) const {
    auto sliceOp = origOp.getInput().getDefiningOp<IE::SliceOp>();
    return sliceOp.getSource();
}

IE::ConcatOp MergeFullyConnectedForDQPatternWithDequantize::getMatMulOutputConcat(IE::FullyConnectedOp origOp) const {
    auto outReshape = getSingleUser<IE::ReshapeOp>(origOp.getResult()).value();
    return getSingleUser<IE::ConcatOp>(outReshape.getResult(), false).value();
}

std::optional<mlir::Operation*> MergeFullyConnectedForDQPatternWithDequantize::getMatMulWeights(
        IE::FullyConnectedOp origOp, ShapeRef expectedOutShape, IE::ConcatOp concatOutput) const {
    // Matmul op is expected to have 2D input and output. so need check the rank of input and output
    // Left-hand matrix must have exactly two dimensions.
    const auto lhs = origOp.getInput();
    const auto lhsType = mlir::dyn_cast<vpux::NDTypeInterface>(lhs.getType());
    if (lhsType.getRank() != 2) {
        _log.trace("lhsType is not suitable", lhsType);
        return std::nullopt;
    }
    // Right-hand matrix must have exactly two dimensions.
    auto rhs = origOp.getWeights();
    const auto rhsType = mlir::dyn_cast<vpux::NDTypeInterface>(rhs.getType());
    if (rhsType.getRank() != 2) {
        _log.trace("rhsType is not suitable", rhsType);
        return std::nullopt;
    }

    // Validate output shape
    auto outShape = getShape(origOp.getOutput());
    if (outShape != expectedOutShape) {
        _log.trace("outShape is not suitable", outShape);
        return std::nullopt;
    }

    // Validate reshape user and concat operation: all branches should have the same output Concat
    auto validateReshapeUserAndConcat = [this](IE::FullyConnectedOp origOp, IE::ConcatOp concatOutput) {
        auto concat = getConcatThroughReshapeUser(origOp);
        if (mlir::failed(concat)) {
            return false;
        }

        return concat.value() == concatOutput.getOperation();
    };
    if (!validateReshapeUserAndConcat(origOp, concatOutput)) {
        return std::nullopt;
    }

    return getWeightsSourceSliceOp(origOp);
}

bool MergeFullyConnectedForDQPatternWithDequantize::validateUnrolledMatMulBranch(
        SmallVector<UnrolledMatMulBranch>& branches) const {
    auto inputSource = branches[0].input->getOperand(0);
    auto weightsSource = branches[0].weights->getOperand(0);

    // 1.Validate input source shape
    // Input souce should be with shape [1, (N x IC)]
    // Input SliceOps should split source on d1 into shape [1, IC]
    auto validateInputSourceShape = [this](mlir::Value inputSource, ArrayRef<UnrolledMatMulBranch> branches) {
        auto numBranches = checked_cast<int64_t>(branches.size());
        auto inputSourceShape = getShape(inputSource);
        auto refSliceShape = getShape(branches[0].input->getResult(0));
        auto ic = refSliceShape[Dim(1)];

        if (inputSourceShape.size() != 2 || inputSourceShape[Dim(0)] != 1 ||
            inputSourceShape[Dim(1)] != numBranches * ic) {
            _log.trace("Input source shape {0} and Slice shape {1} don't match", inputSourceShape, refSliceShape);
            return false;
        }

        return true;
    };
    if (!validateInputSourceShape(inputSource, branches)) {
        return false;
    }

    // 2.Validate weights source shape
    // Weights souce should be with shape [N, OC, IC]
    // Weights SliceOps should split weights on d0 into shape [1, OC, IC]
    auto validateWeightsSourceShape = [this](mlir::Value weightsSource, ArrayRef<UnrolledMatMulBranch> branches) {
        auto numBranches = checked_cast<int64_t>(branches.size());
        auto weightsSourceShape = getShape(weightsSource);
        auto refWeightsSliceShape = getShape(branches[0].weights->getResult(0));

        if (weightsSourceShape.size() != 3 || weightsSourceShape[Dim(0)] != numBranches ||
            refWeightsSliceShape[Dim(0)] != 1 || weightsSourceShape[Dim(1)] != refWeightsSliceShape[Dim(1)] ||
            weightsSourceShape[Dim(2)] != refWeightsSliceShape[Dim(2)]) {
            _log.trace("Weights source shape {0} and weights slice shape {1} don't match", weightsSourceShape,
                       refWeightsSliceShape);
            return false;
        }

        return true;
    };
    if (!validateWeightsSourceShape(weightsSource, branches)) {
        return false;
    }

    // 3.Validate slice operations in branches
    auto validateSliceOperations = [this](ArrayRef<UnrolledMatMulBranch> branches) {
        std::set<int64_t> uniqueInputSliceOffsets;
        std::set<int64_t> uniqueWeightsSliceOffsets;
        auto refSliceShape = getShape(branches[0].input->getResult(0));
        auto refWeightsSliceShape = getShape(branches[0].weights->getResult(0));
        auto ic = refSliceShape[Dim(1)];

        for (auto branch : branches | indexed) {
            auto currBranch = branch.value();
            auto currindex = checked_cast<int64_t>(branch.index());
            auto inputSliceOp = mlir::dyn_cast<IE::SliceOp>(currBranch.input);
            VPUX_THROW_WHEN(inputSliceOp == nullptr, "Can't find input SliceOp");

            // All input SliceOps should have the same shape
            auto shape = getShape(inputSliceOp.getResult());
            if (shape != refSliceShape) {
                _log.trace("Mismatched input slice shape {0} with reference shape {1}", shape, refSliceShape);
                return false;
            }

            // Input slice offsets should be on d1
            // Every SliceOp should have unique offsets and offset on d1 should can be divided by IC
            auto inputSliceDimIndex = 1;
            auto offsets = parseIntArrayAttr<int64_t>(inputSliceOp.getStaticOffsets());
            if (offsets[inputSliceDimIndex] % ic != 0) {
                _log.trace("Mismatched input slice offsets {0}, expect {1}", offsets, currindex * ic);
                return false;
            }
            if (uniqueInputSliceOffsets.find(offsets[inputSliceDimIndex]) != uniqueInputSliceOffsets.end()) {
                _log.trace("Duplicated input slice offsets {0}", offsets);
                return false;
            }
            uniqueInputSliceOffsets.insert(offsets[inputSliceDimIndex]);

            auto weightsSliceOp = mlir::dyn_cast<IE::SliceOp>(currBranch.weights);
            VPUX_THROW_WHEN(weightsSliceOp == nullptr, "Can't find weights SliceOp");

            // All weights SliceOps should have the same shape
            shape = getShape(weightsSliceOp.getResult());
            if (shape != refWeightsSliceShape) {
                _log.trace("Mismatched weights slice shape {0} with reference shape {1}", shape, refWeightsSliceShape);
                return false;
            }

            // Weights slice offsets should be on d0
            // Every SliceOp should have unique offsets
            auto weightsSliceDimIndex = 0;
            offsets = parseIntArrayAttr<int64_t>(weightsSliceOp.getStaticOffsets());
            if (uniqueWeightsSliceOffsets.find(offsets[weightsSliceDimIndex]) != uniqueWeightsSliceOffsets.end()) {
                _log.trace("Duplicated weights slice offsets {0}", offsets);
                return false;
            }
            uniqueWeightsSliceOffsets.insert(offsets[weightsSliceDimIndex]);
        }

        return true;
    };
    return validateSliceOperations(branches);
}

mlir::Value MergeFullyConnectedForDQPatternWithDequantize::buildNewMatMulInput(ArrayRef<UnrolledMatMulBranch> branches,
                                                                               size_t batchIdx, size_t batchOffset,
                                                                               size_t batchSize,
                                                                               mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto matMul = branches[0].matMul;
    auto matMulInShape = getShape(matMul.getInput());
    auto source = getMatMulInputSource(matMul);

    SmallVector<int64_t> inSliceOffsets(getShape(source).size(), 0);
    inSliceOffsets[1] = checked_cast<int64_t>(batchOffset) * matMulInShape[Dim(1)];
    SmallVector<int64_t> inSliceSizes = to_small_vector(getShape(source));
    inSliceSizes[1] = checked_cast<int64_t>(batchSize) * matMulInShape[Dim(1)];
    auto slice = rewriter.create<IE::SliceOp>(appendLoc(matMul.getLoc(), "_slice_{0}", batchIdx), source,
                                              getIntArrayAttr(ctx, inSliceOffsets), getIntArrayAttr(ctx, inSliceSizes));

    SmallVector<int64_t> newInputShape{checked_cast<int64_t>(batchSize), matMulInShape[Dim(1)]};
    const auto reshapeOutShapeAttr = getIntArrayAttr(ctx, newInputShape);
    return rewriter.createOrFold<IE::ReshapeOp>(appendLoc(matMul.getLoc(), "_reshape"), slice, nullptr, false,
                                                reshapeOutShapeAttr);
}

mlir::Value MergeFullyConnectedForDQPatternWithDequantize::buildNewMatMulWeights(
        ArrayRef<UnrolledMatMulBranch> branches, size_t batchIdx, size_t batchOffset, size_t batchSize,
        mlir::PatternRewriter& rewriter) const {
    auto source = branches[0].weights->getOperand(0);
    auto sourceShape = getShape(source);
    auto ctx = rewriter.getContext();

    rewriter.setInsertionPointAfterValue(source);

    SmallVector<int64_t> sliceOffsets(sourceShape.size(), 0);
    sliceOffsets[0] = checked_cast<int64_t>(batchOffset);
    SmallVector<int64_t> sliceSizes = to_small_vector(sourceShape);
    sliceSizes[0] = checked_cast<int64_t>(batchSize);
    auto slice = rewriter.create<IE::SliceOp>(appendLoc(source.getLoc(), "_slice_{0}", batchIdx), source,
                                              getIntArrayAttr(ctx, sliceOffsets), getIntArrayAttr(ctx, sliceSizes));

    auto dequantizeOp = rewriter.create<IE::DequantizeOp>(appendLoc(slice.getLoc(), "_dequantize"), slice,
                                                          mlir::Float16Type::get(ctx))
                                .getOutput();

    auto outShape = getShape(dequantizeOp);
    SmallVector<int64_t> newWeightsShape{outShape[Dim(0)] * outShape[Dim(1)], outShape[Dim(2)]};
    SmallVector<SmallVector<int64_t>> inDimMapping{{0}, {0}, {1}};
    return rewriter.createOrFold<IE::AffineReshapeOp>(appendLoc(dequantizeOp.getLoc(), "_reshape"), dequantizeOp,
                                                      getIntArrayOfArray(ctx, inDimMapping),
                                                      getIntArrayAttr(ctx, newWeightsShape));
}

mlir::Value MergeFullyConnectedForDQPatternWithDequantize::reshapeOutputSlice(mlir::Value sliceOut,
                                                                              mlir::PatternRewriter& rewriter) const {
    // new output shape size is always 2, reshape to 4
    auto ctx = rewriter.getContext();
    auto shape = getShape(sliceOut);
    SmallVector<int64_t> newShape{1, 1, shape[Dim(0)], shape[Dim(1)]};
    const auto reshapeLoc = appendLoc(sliceOut.getLoc(), "_reshape");
    return rewriter.create<IE::ReshapeOp>(reshapeLoc, sliceOut, nullptr, false, getIntArrayAttr(ctx, newShape));
}

void MergeFullyConnectedForDQPatternWithDequantize::cleanUpMatMulBranches(ArrayRef<UnrolledMatMulBranch> branches,
                                                                          mlir::PatternRewriter& rewriter) const {
    for (auto& branch : branches) {
        auto matMul = branch.matMul;
        auto inSlice = matMul.getInput().getDefiningOp<IE::SliceOp>();
        auto weightsReshape = matMul.getWeights().getDefiningOp<IE::ReshapeOp>();
        auto weightsDequantize = weightsReshape.getInput().getDefiningOp<IE::DequantizeOp>();
        auto weightsSlice = weightsDequantize.getInput().getDefiningOp<IE::SliceOp>();

        auto outReshape = mlir::dyn_cast<IE::ReshapeOp>(*matMul->getUsers().begin());

        rewriter.eraseOp(outReshape);
        rewriter.eraseOp(matMul);
        rewriter.eraseOp(inSlice);
        rewriter.eraseOp(weightsReshape);
        rewriter.eraseOp(weightsDequantize);
        rewriter.eraseOp(weightsSlice);
    }
}

class MergeFullyConnectedPass final : public IE::impl::MergeFullyConnectedBase<MergeFullyConnectedPass> {
public:
    explicit MergeFullyConnectedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MergeFullyConnectedPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeFullyConnectedWithWeightsAsConstant>(&ctx, _log);
    patterns.add<MergeFullyConnectedForDQPatternWithConvert>(&ctx, _log);
    patterns.add<MergeFullyConnectedForDQPatternWithDequantize>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createMergeFullyConnectedPass(Logger log) {
    return std::make_unique<MergeFullyConnectedPass>(log);
}
