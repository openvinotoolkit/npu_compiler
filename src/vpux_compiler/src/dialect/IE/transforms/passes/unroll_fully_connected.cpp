//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/ValueRange.h>
#include <cstdint>

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLFULLYCONNECTED
#define GEN_PASS_DEF_UNROLLFULLYCONNECTED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class UnrollFullyConnected final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    UnrollFullyConnected(mlir::MLIRContext* ctx, bool accumulateMatmulWithDPU, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx),
              _log(log),
              _accumulateMatmulWithDPU(accumulateMatmulWithDPU) {
        setDebugName("UnrollFullyConnected");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkConcat(IE::ConcatOp concat) const;
    bool isSupportedPattern(mlir::Operation* op) const;
    bool isTransposeReshape(mlir::Operation* op) const;
    bool isReshapeTranspose(mlir::Operation* op) const;
    bool isReshapeOnly(mlir::Operation* op) const;
    SmallVector<mlir::Value> findMatMulInputs(IE::FullyConnectedOp matMulOp) const;
    SmallVector<mlir::Value> splitLeftInput(const mlir::Value lhs, const int64_t groups, mlir::Location origLoc,
                                            mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> buildMatMuls(IE::FullyConnectedOp matMulOp, const mlir::ValueRange lhsInputs,
                                          const mlir::ValueRange rhsInputs, bool isReduceSumForAccumulate,
                                          mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> accumulateMatMuls(const mlir::ValueRange matMuls, mlir::PatternRewriter& rewriter) const;
    mlir::Value reduceSumForAccumulateMatMuls(const mlir::ValueRange matMuls, mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> reshapeTo2d(mlir::ValueRange values, mlir::PatternRewriter& rewriter) const;
    bool isUnrollingBeneficial(IE::FullyConnectedOp origOp, mlir::ValueRange inputs) const;

private:
    Logger _log;
    bool _accumulateMatmulWithDPU = false;
};

bool UnrollFullyConnected::checkConcat(IE::ConcatOp concat) const {
    auto nestedLog = _log.nest();
    const auto concatInputs = concat.getInputs();
    const auto concatOutput = concat.getOutput();
    const auto outputShape = getShape(concatOutput);
    const auto isConcatInputCompatible = [&](const mlir::Value in) -> bool {
        if (in.getDefiningOp<IE::FakeQuantizeOp>() == nullptr &&
            in.getDefiningOp<IE::DynamicDequantizeOp>() == nullptr && in.getDefiningOp<IE::DequantizeOp>() == nullptr) {
            nestedLog.trace("Concat input does not have FakeQuantize producer: input = {1}.", in);
            return false;
        }

        const auto inputShape = getShape(in);
        const Shape expectedInputShape1xGxW = {1, outputShape[Dim(1)], outputShape[Dim(2)]};
        const Shape expectedInputShapeGx1xW = {outputShape[Dim(0)], 1, outputShape[Dim(2)]};
        if (inputShape != expectedInputShape1xGxW && inputShape != expectedInputShapeGx1xW) {
            nestedLog.trace("Concat input has incompatible shape: expected shape = {0} or {1}, input shape = {2}.",
                            expectedInputShape1xGxW, expectedInputShapeGx1xW, inputShape);
            return false;
        }

        return true;
    };
    return std::all_of(concatInputs.begin(), concatInputs.end(), isConcatInputCompatible);
}

bool UnrollFullyConnected::isTransposeReshape(mlir::Operation* op) const {
    // Check that the producer of the right-hand matrix is IE.Transpose
    if (!mlir::isa_and_nonnull<IE::TransposeOp>(op)) {
        return false;
    }
    auto transpose = mlir::cast<IE::TransposeOp>(op);
    // Check that transpose transforms [d0, d1] shape into [d1, d0]
    const auto transposeInShape = getShape(transpose.getInput());
    if (transposeInShape.size() < 2) {
        return false;
    }
    const auto transposeOutShape = getShape(transpose.getOutput());
    const auto expectedTransposeOutShape = Shape{transposeInShape[Dim(1)], transposeInShape[Dim(0)]};
    if (expectedTransposeOutShape != transposeOutShape) {
        return false;
    }
    // IE.Transpose must have IE.AffineReshape producer
    auto transposeProducer = transpose.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::AffineReshapeOp>(transposeProducer)) {
        return false;
    }
    auto reshape = mlir::cast<IE::AffineReshapeOp>(transposeProducer);
    // Check that reshape collapses [d0, d1, d2] shape into [d0 * d1, d2]
    const auto reshapeInputDims = getShape(reshape.getInput());
    if (reshapeInputDims.size() < 3) {
        return false;
    }
    const Shape expectedOutputShape = {reshapeInputDims[Dim(0)] * reshapeInputDims[Dim(1)], reshapeInputDims[Dim(2)]};
    const auto reshapeOutputDims = getShape(reshape.getOutput());
    return reshapeOutputDims == expectedOutputShape;
}

bool UnrollFullyConnected::isReshapeTranspose(mlir::Operation* op) const {
    // Check that the producer of the right-hand matrix is IE.AffineReshape
    if (!mlir::isa_and_nonnull<IE::AffineReshapeOp>(op)) {
        return false;
    }
    auto reshape = mlir::cast<IE::AffineReshapeOp>(op);
    // Check that reshape collapses [d0, d1, d2] shape into [d0, d1 * d2]
    const auto reshapeInputDims = getShape(reshape.getInput());
    if (reshapeInputDims.size() < 3) {
        return false;
    }
    const Shape expectedOutputShape = {reshapeInputDims[Dim(0)], reshapeInputDims[Dim(1)] * reshapeInputDims[Dim(2)]};
    const auto reshapeOutputDims = getShape(reshape.getOutput());
    if (expectedOutputShape != reshapeOutputDims) {
        return false;
    }
    // IE.AffineReshape must have IE.Transpose producer
    auto reshapeProducer = reshape.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::TransposeOp>(reshapeProducer)) {
        return false;
    }
    auto transpose = mlir::cast<IE::TransposeOp>(reshapeProducer);
    // Check that transpose transforms [d0, d1, d2] shape into [d2, d0, d1]
    const auto maybeMap = transpose.getOrderValue();
    if (!maybeMap.has_value()) {
        return false;
    }
    const auto permutation =
            to_small_vector(DimsOrder::fromAffineMap(maybeMap.value()).toPermutation() | transformed([](Dim d) {
                                return static_cast<unsigned>(d.ind());
                            }));
    const SmallVector<SmallVector<unsigned>> expectedMap = {{2, 0, 1}, {1, 0, 2}};
    return std::find(expectedMap.begin(), expectedMap.end(), permutation) != expectedMap.end();
}

bool UnrollFullyConnected::isReshapeOnly(mlir::Operation* op) const {
    // Check that the producer of the right-hand matrix is IE.AffineReshape
    if (!mlir::isa_and_nonnull<IE::AffineReshapeOp>(op)) {
        return false;
    }
    auto reshape = mlir::cast<IE::AffineReshapeOp>(op);
    // Check that reshape collapses [d0, d1, d2] shape into [d0, d1 * d2]
    const auto reshapeInputDims = getShape(reshape.getInput());
    if (reshapeInputDims.size() < 3) {
        return false;
    }
    const Shape expectedOutputShape = {reshapeInputDims[Dim(0)], reshapeInputDims[Dim(1)] * reshapeInputDims[Dim(2)]};
    const auto reshapeOutputDims = getShape(reshape.getOutput());
    if (expectedOutputShape != reshapeOutputDims) {
        return false;
    }
    // IE.AffineReshape must have IE.Transpose producer
    auto reshapeProducer = reshape.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ConcatOp>(reshapeProducer)) {
        return false;
    }

    return true;
}

bool UnrollFullyConnected::isSupportedPattern(mlir::Operation* op) const {
    return isTransposeReshape(op) || isReshapeTranspose(op) || isReshapeOnly(op);
}

SmallVector<mlir::Value> UnrollFullyConnected::findMatMulInputs(IE::FullyConnectedOp matMulOp) const {
    auto nestedLog = _log.nest();
    // Left-hand matrix must have exactly two dimensions.
    const auto lhs = matMulOp.getInput();
    const auto lhsType = mlir::cast<vpux::NDTypeInterface>(lhs.getType());
    if (lhsType.getRank() != 2) {
        nestedLog.debug("Matmul has non-2D lhs input.");
        return {};
    }
    // Right-hand matrix must have exactly two dimensions.
    const auto rhs = matMulOp.getWeights();
    const auto rhsType = mlir::cast<vpux::NDTypeInterface>(rhs.getType());
    if (rhsType.getRank() != 2) {
        nestedLog.debug("Matmul has non-2D rhs input.");
        return {};
    }

    // Right-hand matrix must have either IE.Transpose or IE.AffineReshape producer
    mlir::Operation* lastOp = rhs.getDefiningOp();

    if (!isSupportedPattern(lastOp)) {
        nestedLog.debug("No Transpose-Reshape or Reshape-Transpose pattern on rhs input.");
        return {};
    }
    // If the producer of rhs is IE.Transpose, next operation must be IE.AffineReshape
    // If the producer of rhs is IE.AffineReshape, next operation must be IE.Transpose

    if (!isReshapeOnly(lastOp)) {
        lastOp = lastOp->getOperand(0).getDefiningOp();
    }
    // Either way, the pass expects a concatenation to be at the root of the pattern.
    auto maybeConcat = lastOp->getOperand(0).getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ConcatOp>(maybeConcat)) {
        nestedLog.debug("No concat at rhs pattern root.");
        return {};
    }
    // The concat must concatenate [1xHxW, ..., 1xHxW] inputs into CxHxW shape.
    auto concat = mlir::cast<IE::ConcatOp>(maybeConcat);
    if (!checkConcat(concat)) {
        nestedLog.debug("Concat at rhs pattern root is unsupported.");
        return {};
    }
    return concat.getInputs();
}

SmallVector<mlir::Value> UnrollFullyConnected::splitLeftInput(const mlir::Value lhs, const int64_t groups,
                                                              mlir::Location origLoc,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto lhsShape = getShape(lhs);
    const auto blockSize = lhsShape[Dim(1)] / groups;
    const SmallVector<int64_t> staticSizes = {lhsShape[Dim(0)], blockSize};
    const auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), staticSizes);
    SmallVector<mlir::Value> inputChunks;
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(origLoc, "slice_{0}", idx);
        const SmallVector<int64_t> offsets = {0, idx * blockSize};
        const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);
        auto slice = rewriter.create<IE::SliceOp>(loc, lhs, offsetsAttr, staticSizesAttr);
        inputChunks.push_back(slice.getResult());
    }
    return inputChunks;
}

SmallVector<mlir::Value> UnrollFullyConnected::buildMatMuls(IE::FullyConnectedOp origOp,
                                                            const mlir::ValueRange lhsInputs,
                                                            const mlir::ValueRange rhsInputs,
                                                            bool isReduceSumForAccumulate,
                                                            mlir::PatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(lhsInputs.size() == rhsInputs.size(),
                      "The number of left-hand matrices does not match the number of right-hand matrices");
    auto loc = origOp->getLoc();
    auto ctx = origOp->getContext();
    SmallVector<mlir::Value> matMuls;
    for (const auto& idx : irange(lhsInputs.size())) {
        mlir::Value lastOpOutput = rhsInputs[idx];
        const auto rhsShape = getShape(lastOpOutput);
        const auto lhsShape = getShape(lhsInputs[idx]);
        if (lhsShape[Dim(1)] != rhsShape[Dim(1)]) {
            const auto transposeLoc = appendLoc(loc, "transpose_{0}", idx);
            SmallVector<unsigned> transPerm = {1, 0};
            const auto orderAttr =
                    mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
            lastOpOutput = rewriter.create<IE::TransposeOp>(transposeLoc, rhsInputs[idx],
                                                            /*order=*/nullptr, orderAttr)
                                   .getOutput();
        }
        const auto matMulLoc = appendLoc(loc, "matmul_{0}", idx);
        auto newMatMul = rewriter.create<IE::FullyConnectedOp>(matMulLoc, lhsInputs[idx], lastOpOutput,
                                                               /*bias=*/nullptr);
        if (!isReduceSumForAccumulate) {
            matMuls.push_back(newMatMul.getOutput());
        } else {
            auto matMulShape = getShape(newMatMul.getOutput());
            // fullyconnectOp output shape size is always 2, reshape to 4
            SmallVector<int64_t> newShapeOut{1, 1, matMulShape[Dim(0)], matMulShape[Dim(1)]};
            const auto reshapeLoc = appendLoc(loc, "reshape_{0}", idx);
            auto reshape = rewriter.create<IE::ReshapeOp>(reshapeLoc, newMatMul.getOutput(), nullptr, false,
                                                          getIntArrayAttr(ctx, newShapeOut));
            matMuls.push_back(reshape.getOutput());
        }
    }
    return matMuls;
}

SmallVector<mlir::Value> UnrollFullyConnected::accumulateMatMuls(const mlir::ValueRange matMuls,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto numGroups = matMuls.size();
    VPUX_THROW_UNLESS(numGroups >= 2, "The group must contain at least two IE.MatMul operations, got {0}", numGroups);
    SmallVector<mlir::Value> addOps;
    // Add up the first two matrix multiplications.
    // Next iterations will add each MatMul to the previous result.
    const auto addLoc = appendLoc(matMuls[0].getLoc(), "add");

    mlir::Value addMatMuls;
    auto ctx = rewriter.getContext();
    if (_accumulateMatmulWithDPU) {
        addMatMuls =
                rewriter.create<IE::AddOp>(addLoc, matMuls[0], matMuls[1],
                                           IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT),
                                           nullptr, nullptr, nullptr, nullptr)
                        .getOutput();
    } else {
        addMatMuls = rewriter.create<IE::AccumulateOp>(addLoc, matMuls[0], matMuls[1],
                                                       /*lhsScale=*/nullptr,
                                                       /*rhsScale=*/nullptr)
                             .getOutput();
    }
    addOps.push_back(addMatMuls);
    for (const auto& idx : irange(numGroups - 2)) {
        // idx + 2 because the first two MatMul operations have already been summed up.
        const auto& matMul = matMuls[idx + 2];
        const auto loc = appendLoc(matMul.getLoc(), "add_{0}", idx + 2);
        mlir::Value accumulateValue;
        if (_accumulateMatmulWithDPU) {
            accumulateValue = rewriter.create<IE::AddOp>(loc, addOps.back(), matMuls[idx + 2],
                                                         IE::AutoBroadcastTypeAttr::get(
                                                                 ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT),
                                                         nullptr, nullptr, nullptr, nullptr)
                                      .getOutput();
        } else {
            accumulateValue = rewriter.create<IE::AccumulateOp>(loc, addOps.back(), matMuls[idx + 2],
                                                                /*lhsScale=*/nullptr,
                                                                /*rhsScale=*/nullptr)
                                      .getOutput();
        }
        addOps.push_back(accumulateValue);
    }
    return addOps;
}

mlir::Value UnrollFullyConnected::reduceSumForAccumulateMatMuls(const mlir::ValueRange matMuls,
                                                                mlir::PatternRewriter& rewriter) const {
    const auto numGroups = matMuls.size();
    VPUX_THROW_UNLESS(numGroups >= 2, "The group must contain at least two IE.MatMul operations, got {0}", numGroups);

    auto ctx = matMuls.front().getContext();
    const auto concatLoc = appendLoc(matMuls.front().getLoc(), "concat");
    auto concat = rewriter.create<IE::ConcatOp>(concatLoc, matMuls, Dims4D::Act::C);

    auto axesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{vpux::Dims4D::Act::C.ind()});
    const auto reduceSumLoc = appendLoc(matMuls.front().getLoc(), "_reduceSum_for_accumulate");
    auto newReduceSumOp = rewriter.create<IE::ReduceSumOp>(reduceSumLoc, concat.getOutput(), nullptr, axesAttr, false,
                                                           nullptr, nullptr);

    auto newMatMulShape = getShape(newReduceSumOp.getOutput());
    SmallVector<int64_t> newShapeOut{newMatMulShape[Dim(1)], newMatMulShape[Dim(2)]};

    const auto reshapeLoc = appendLoc(matMuls.front().getLoc(), "reshape_out");
    auto reshape = rewriter.create<IE::ReshapeOp>(reshapeLoc, newReduceSumOp.getOutput(), nullptr, false,
                                                  getIntArrayAttr(ctx, newShapeOut));
    return reshape.getOutput();
}

SmallVector<mlir::Value> UnrollFullyConnected::reshapeTo2d(mlir::ValueRange values,
                                                           mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> reshapedValues;
    size_t counter = 0;
    const auto to2d = [&rewriter, &counter](const mlir::Value val) -> mlir::Value {
        const ShapeRef sliceShape = getShape(val);
        const SmallVector<int64_t> target2dShape = {sliceShape[Dim(0)] * sliceShape[Dim(1)], sliceShape[Dim(2)]};
        const auto target2dShapeAttr = getIntArrayAttr(rewriter.getContext(), target2dShape);
        const auto reshapeLoc = appendLoc(IE::getValueLocation(val), "reshape_{0}", counter++);
        auto reshape = rewriter.create<IE::ReshapeOp>(reshapeLoc, val, nullptr, false, target2dShapeAttr);
        return reshape.getOutput();
    };
    std::transform(values.begin(), values.end(), std::back_inserter(reshapedValues), to2d);
    return reshapedValues;
}

bool isBeneficialToUseReduceSumForAccumulate(IE::FullyConnectedOp origOp) {
    // Benefit when: KV cache model, the first dim is one
    auto outShape = getShape(origOp.getOutput());
    return outShape[Dim(0)] == 1;
}

bool UnrollFullyConnected::isUnrollingBeneficial(IE::FullyConnectedOp origOp, mlir::ValueRange concatInputs) const {
    auto nestedLog = _log.nest();
    if (auto fqParent = concatInputs.front().getDefiningOp<IE::FakeQuantizeOp>()) {
        const auto levels = fqParent.getLevels();

        if (!levels.has_value()) {
            nestedLog.trace("FakeQuantize parent at loc {0} does not have 'levels' attribute.", fqParent->getLoc());
            return false;
        }

        // Worse performance observed for unrolled MatMul when weights elem type is larger than 4 bits
        constexpr int64_t MAX_QUANT_LEVELS = 16;
        if (levels.value() > MAX_QUANT_LEVELS) {
            nestedLog.trace("Quantized element type has more than 4bits.");
            return false;
        }
    }

    if (auto dynamicDequant = concatInputs.front().getDefiningOp<IE::DynamicDequantizeOp>()) {
        auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(dynamicDequant.getInput().getType());
        if (auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputType.getElementType())) {
            if (uniformType.getStorageTypeIntegralWidth() != 4 && uniformType.getStorageTypeIntegralWidth() != 2) {
                return false;
            }
        }
    }

    if (auto dequant = concatInputs.front().getDefiningOp<IE::DequantizeOp>()) {
        auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(dequant.getInput().getType());
        if (auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inputType.getElementType())) {
            if (!uniformType.isSigned() || uniformType.getStorageTypeIntegralWidth() != 4) {
                return false;
            }
        }
    }

    const auto fcInputShape = getShape(origOp.getInput());
    const auto inputChannels = fcInputShape[Dim(1)];
    // Input channels are over supported DPU dim size, favor unrolling the op as input channels will need to be
    // splitted in fp16 case as well anyway
    if (inputChannels > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return true;
    }

    const auto fcWeightsShape = getShape(origOp.getWeights());

    const auto numGroups = static_cast<double>(concatInputs.size());
    const auto groupSize = static_cast<double>(inputChannels) / numGroups;
    const auto outputChannels = static_cast<double>(fcWeightsShape[Dim(1)]);
    const auto spatialSz = static_cast<double>(fcInputShape[Dim(0)]);

    // Depending on the group-quantized Matmul, it might more performant to run it as 1 Conv operation with fp16
    // weights, rather than n smaller Convolutions with quantized weights.
    // Ideally, there would be a cost model that looks at the configuration of the MatMul and decide whether it is
    // better to run with fp16 weights rather than the quant ones.
    // Barring that, we profiled a set of MatMuls with different in/out channel sized, num groups and spatial size and
    // compared performance with i4 weights vs. fp16 weights. It was observed that i4 outperformed fp16 when output
    // channels and group size were larger, while fp16 was better when the number of groups and the spatial size was
    // high.
    // The coefficients for the formula below were obtained by taking the collected measurements and solving the
    // equation:
    //         A * group_sz + B * out_ch + C * num_groups + D * spatial_sz + CST = (i4_fps / fp16_fps)
    // As such unrolling is beneficial when i4_fps > fp16_fps => (i4_fps / fp16_fps) > 1 (= EXPERIMENTAL_THRESHOLD)

    // IMPORTANT: Profiling was done for the set of MatMul configurations that are likely to currently appear in LLMs,
    //            while also being dependent on the state of compiler optimizations implemented at the time. It is,
    //            therefore, an approximation that is subject to change. Overtime we might need to re-evaluate the
    //            group-quantized vs. fp16 analysis, unless a more permanent, cost-based solution is found.
    constexpr double GRP_SZ_COEFF = 0.00588;
    constexpr double OUT_CH_COEFF = 0.0000675;
    constexpr double NUM_GRP_COEFF = -13.9;
    constexpr double SPATIAL_SZ_COEFF = 0.462;
    constexpr double CST_COEFF = 0.858;
    const double unrollPerfMetric = GRP_SZ_COEFF * groupSize + OUT_CH_COEFF * outputChannels +
                                    NUM_GRP_COEFF / numGroups + SPATIAL_SZ_COEFF / spatialSz + CST_COEFF;

    constexpr double EXPERIMENTAL_THRESHOLD = 1;
    nestedLog.trace("Unroll perf metric {0} must be larger than threshold {1} to unroll the Matmul", unrollPerfMetric,
                    EXPERIMENTAL_THRESHOLD);
    return unrollPerfMetric > EXPERIMENTAL_THRESHOLD;
}

mlir::LogicalResult UnrollFullyConnected::matchAndRewrite(IE::FullyConnectedOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto opLoc = origOp->getLoc();
    _log.debug("Found FullyConnectedOp at loc: {0}", opLoc);
    const auto matMulInputs = findMatMulInputs(origOp);
    if (matMulInputs.empty()) {
        return matchFailed(rewriter, origOp, "IE.MatMul at {0} is not supported", opLoc);
    }

    if (!isUnrollingBeneficial(origOp, matMulInputs)) {
        _log.debug("Performance-wise it is better not to unroll FullyConnectedOp at loc: {0}; will be run in float16 "
                   "precision",
                   opLoc);
        return mlir::failure();
    }

    auto nestedLog = _log.nest();
    nestedLog.debug("Unroll GPTQ FullyConnected.");
    const auto rhsChunks = reshapeTo2d(matMulInputs, rewriter);
    const auto numChunks = checked_cast<int64_t>(rhsChunks.size());
    // Split left input into the number of chunks:
    const auto lhsChunks = splitLeftInput(origOp.getInput(), numChunks, opLoc, rewriter);
    // Multiply lhs by rhs in pairs

    if (isBeneficialToUseReduceSumForAccumulate(origOp)) {
        const auto matMuls = buildMatMuls(origOp, lhsChunks, rhsChunks, true, rewriter);
        // Use ReduceSum to sum up MatMul results
        const auto reduceSum = reduceSumForAccumulateMatMuls(matMuls, rewriter);
        rewriter.replaceOp(origOp, reduceSum);
        nestedLog.debug("Accumulate using ReduceSum op.");
    } else {
        const auto matMuls = buildMatMuls(origOp, lhsChunks, rhsChunks, false, rewriter);
        // Sum up MatMul results
        const auto addOps = accumulateMatMuls(matMuls, rewriter);
        VPUX_THROW_WHEN(addOps.empty(), "The group must contain at least one IE.Accumulate operation, got 0.");
        // The last IE.Accumulate operation in the list will contain the total sum.
        rewriter.replaceOp(origOp, addOps.back());
        nestedLog.debug("Accumulate using IE.Accumulate op.");
    }

    _log.debug("Successfully unrolled FullyConnectedOp at loc: {0}", opLoc);
    return mlir::success();
}

class UnrollFullyConnectedPass final : public IE::impl::UnrollFullyConnectedBase<UnrollFullyConnectedPass> {
public:
    explicit UnrollFullyConnectedPass(Logger log, bool accumulateMatmulWithDPU)
            : _accumulateMatmulWithDPU(accumulateMatmulWithDPU) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _accumulateMatmulWithDPU = false;
};

mlir::LogicalResult UnrollFullyConnectedPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (accumulateMatmulWithDPU.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_accumulateMatmulWithDPU' field to the value {1} of the "
                   "pass option "
                   "'accumulateMatmulWithDPU' generated by MLIR",
                   _accumulateMatmulWithDPU, accumulateMatmulWithDPU);
        _accumulateMatmulWithDPU = accumulateMatmulWithDPU;
    }

    return mlir::success();
}

void UnrollFullyConnectedPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrollFullyConnected>(&ctx, _accumulateMatmulWithDPU, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollFullyConnectedPass(Logger log, bool accumulateMatmulWithDPU) {
    return std::make_unique<UnrollFullyConnectedPass>(log, accumulateMatmulWithDPU);
}
