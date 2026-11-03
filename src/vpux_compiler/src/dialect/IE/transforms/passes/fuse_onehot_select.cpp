//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEONEHOTSELECT
#define GEN_PASS_DEF_FUSEONEHOTSELECT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Skip rank/precision-only ops (Reshape/AffineReshape/Convert/Squeeze/Unsqueeze) to reach the
// producer that actually performs the computation.
mlir::Operation* traceThroughViewLikeOps(mlir::Value value) {
    auto op = value.getDefiningOp();
    while (mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::ConvertOp, IE::SqueezeOp, IE::UnsqueezeOp>(
            op)) {
        op = op->getOperand(0).getDefiningOp();
    }
    return op;
}

// Reads a splat scalar value, given either as a constant operand or as an attribute, as type T.
template <typename T>
mlir::FailureOr<T> getScalarConstOrAttr(mlir::Value operand, mlir::Attribute attr) {
    if (operand) {
        auto cst = operand.getDefiningOp<Const::DeclareOp>();
        if (cst == nullptr || !cst.getContentAttr().isSplat() || cst.getContent().getType().getNumElements() != 1) {
            return mlir::failure();
        }
        const auto content = cst.getContent();
        const auto elemType = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType();
        if (mlir::isa_and_present<mlir::IntegerType>(elemType)) {
            return checked_cast<T>(content.getSplatValue<int64_t>());
        }
        return checked_cast<T>(content.getSplatValue<double>());
    }
    if (auto intAttr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(attr)) {
        return checked_cast<T>(intAttr.getValue().getSExtValue());
    }
    if (auto fpAttr = mlir::dyn_cast_if_present<mlir::FloatAttr>(attr)) {
        return checked_cast<T>(fpAttr.getValueAsDouble());
    }
    return mlir::failure();
}

// FuseOneHotFullyConnectedToGather
//
// Replaces the embedding lookup pattern
//
//   FC(Select(mask, NaN, OneHot(index, depth, 1, 0)), W)
//
// with a Gather from a transposed weight table. For a bias-free FullyConnected with weights
// W[M, K], a one-hot input selects one row of the equivalent table T[K, M], where
// T[k, m] = W[m, k]. This avoids materializing the K-element one-hot tensor and the FC operation.
//
// Because Gather has no default value for an invalid index, the rewrite replaces indices selected
// by the original mask with zero before Gather and restores NaN afterward. It separately maps
// negative indices to zero for Gather and restores their all-zero result. Under OneHot's
// IGNORE_NEGATIVE mode, every negative index produces an all-zero one-hot vector and therefore an
// all-zero output for a bias-free FC.
// Supported weights are direct f16 or f32 constants.
class FuseOneHotFullyConnectedToGather final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    FuseOneHotFullyConnectedToGather(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
        setDebugName("FuseOneHotFullyConnectedToGather");
    }

    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp fcOp, mlir::PatternRewriter& rewriter) const final {
        return fuseOneHotFullyConnected(fcOp, rewriter, _log, getDebugName());
    }

private:
    struct WeightTable final {
        Const::DeclareOp source;
        int64_t outputChannels = 0;
        int64_t inputChannels = 0;
    };

    struct OneHotSelectMatch final {
        IE::SelectOp select;
        IE::OneHotOp oneHot;
    };

    static mlir::FailureOr<WeightTable> matchWeightChain(mlir::Value weightValue);
    static mlir::FailureOr<mlir::Value> createOneHotTable(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                          WeightTable weights);
    static mlir::Value createIndexConst(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Type elemType,
                                        int64_t value);
    static bool isSupportedFlattenReshape(IE::AffineReshapeOp reshapeOp);
    static mlir::FailureOr<OneHotSelectMatch> matchOneHotSelectInput(mlir::Value linearInput);
    static mlir::LogicalResult fuseOneHotFullyConnected(IE::FullyConnectedOp fcOp, mlir::PatternRewriter& rewriter,
                                                        Logger log, StringRef debugName);

    Logger _log;
};

mlir::FailureOr<FuseOneHotFullyConnectedToGather::WeightTable> FuseOneHotFullyConnectedToGather::matchWeightChain(
        mlir::Value weightValue) {
    auto source = weightValue.getDefiningOp<Const::DeclareOp>();
    if (source == nullptr) {
        return mlir::failure();
    }

    const auto sourceType = mlir::cast<NDTypeInterface>(source.getOutput().getType());
    const auto weightShape = sourceType.getShape();
    if (weightShape.size() != 2 || (!sourceType.getElementType().isF16() && !sourceType.getElementType().isF32())) {
        return mlir::failure();
    }

    return WeightTable{source, weightShape[Dim(0)], weightShape[Dim(1)]};
}

mlir::FailureOr<mlir::Value> FuseOneHotFullyConnectedToGather::createOneHotTable(mlir::PatternRewriter& rewriter,
                                                                                 mlir::Location loc,
                                                                                 WeightTable weights) {
    const int64_t M = weights.outputChannels;
    const int64_t K = weights.inputChannels;
    const auto weightContent = weights.source.getContent();
    const auto sourceElemType = mlir::cast<NDTypeInterface>(weights.source.getOutput().getType()).getElementType();
    const auto tableType = mlir::RankedTensorType::get({K, M}, sourceElemType);

    const auto createTable = [&](auto inputValues, auto elemTag) -> mlir::FailureOr<mlir::Value> {
        using ElemT = std::decay_t<decltype(elemTag)>;
        const auto expectedSize = checked_cast<size_t>(M * K);
        if (inputValues.size() != 1 && inputValues.size() != expectedSize) {
            return mlir::failure();
        }

        SmallVector<ElemT> tableValues(checked_cast<size_t>(K * M));
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                const auto inputIndex = inputValues.size() == 1 ? 0 : checked_cast<size_t>(m * K + k);
                tableValues[checked_cast<size_t>(k * M + m)] = inputValues[inputIndex];
            }
        }

        return Const::createConst<ElemT>(rewriter, appendLoc(loc, "onehot_table"), tableType, ArrayRef(tableValues));
    };

    if (sourceElemType.isF16()) {
        return createTable(weightContent.getValues<vpux::type::float16>(), vpux::type::float16{});
    }
    return createTable(weightContent.getValues<float>(), float{});
}

mlir::Value FuseOneHotFullyConnectedToGather::createIndexConst(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                               mlir::Type elemType, int64_t value) {
    const auto type = mlir::RankedTensorType::get({1}, elemType);
    if (elemType.isInteger(64)) {
        return Const::createConst<int64_t>(rewriter, loc, type, ArrayRef<int64_t>{value});
    }
    return Const::createConst<int32_t>(rewriter, loc, type, ArrayRef<int32_t>{checked_cast<int32_t>(value)});
}

bool FuseOneHotFullyConnectedToGather::isSupportedFlattenReshape(IE::AffineReshapeOp reshapeOp) {
    const auto inputShape = getShape(reshapeOp.getInput());
    const auto outputShape = getShape(reshapeOp.getOutput());
    if (inputShape.isDynamic() || outputShape.isDynamic() || inputShape.size() != 3 || outputShape.size() != 2) {
        return false;
    }

    // The OneHot output is [N, M, K]. The only supported reshape flattens the first two
    // dimensions into the FC row dimension and preserves the one-hot depth as the last dimension.
    const SmallVector<SmallVector<int64_t>> expectedMapping{{0}, {0}, {1}};
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMappingAttr());
    return dimMapping == expectedMapping && inputShape[Dim(2)] == outputShape[Dim(1)] &&
           inputShape.totalSize() == outputShape.totalSize();
}

mlir::FailureOr<FuseOneHotFullyConnectedToGather::OneHotSelectMatch>
FuseOneHotFullyConnectedToGather::matchOneHotSelectInput(mlir::Value linearInput) {
    auto value = linearInput;
    auto* producer = value.getDefiningOp();
    if (auto affineReshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(producer)) {
        if (!isSupportedFlattenReshape(affineReshape)) {
            return mlir::failure();
        }
        value = producer->getOperand(0);
    }

    auto selectOp = value.getDefiningOp<IE::SelectOp>();
    if (selectOp == nullptr || selectOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        return mlir::failure();
    }
    auto invalidValue = getScalarConstOrAttr<double>(selectOp.getInput2(), nullptr);
    if (mlir::failed(invalidValue) || !std::isnan(invalidValue.value())) {
        return mlir::failure();
    }

    auto oneHotOp = selectOp.getInput3().getDefiningOp<IE::OneHotOp>();
    if (oneHotOp == nullptr || oneHotOp.getAxisAttr() != 2 || !oneHotOp.getMode().has_value() ||
        oneHotOp.getMode().value() != IE::OneHotMode::IGNORE_NEGATIVE) {
        return mlir::failure();
    }

    auto oneHotOnValue = getScalarConstOrAttr<double>(oneHotOp.getOnValue(), oneHotOp.getOnValueAttrAttr());
    auto oneHotOffValue = getScalarConstOrAttr<double>(oneHotOp.getOffValue(), oneHotOp.getOffValueAttrAttr());
    if (mlir::failed(oneHotOnValue) || mlir::failed(oneHotOffValue) || oneHotOnValue.value() != 1.0 ||
        oneHotOffValue.value() != 0.0) {
        return mlir::failure();
    }

    return OneHotSelectMatch{selectOp, oneHotOp};
}

mlir::LogicalResult FuseOneHotFullyConnectedToGather::fuseOneHotFullyConnected(IE::FullyConnectedOp fcOp,
                                                                               mlir::PatternRewriter& rewriter,
                                                                               Logger log, StringRef debugName) {
    if (fcOp.getBias() != nullptr) {
        return mlir::failure();
    }

    auto selector = matchOneHotSelectInput(fcOp.getInput());
    if (mlir::failed(selector)) {
        return mlir::failure();
    }
    auto selectOp = selector.value().select;
    auto oneHotOp = selector.value().oneHot;
    const auto index = oneHotOp.getInput();
    const auto indexShape = getShape(index);
    if (indexShape.isDynamic()) {
        return mlir::failure();
    }
    const bool rankExpanded = oneHotOp.getRankExpanded().value_or(false);

    Shape logicalIndexShape;
    if (rankExpanded) {
        if (indexShape.size() != 3 || indexShape[Dim(2)] != 1) {
            return mlir::failure();
        }
        logicalIndexShape = Shape{indexShape[Dim(0)], indexShape[Dim(1)]};
    } else {
        if (indexShape.size() != 2) {
            return mlir::failure();
        }
        logicalIndexShape = Shape{indexShape[Dim(0)], indexShape[Dim(1)]};
    }

    auto depth = getScalarConstOrAttr<int64_t>(oneHotOp.getDepth(), oneHotOp.getDepthAttrAttr());
    if (mlir::failed(depth) || depth.value() <= 0) {
        return mlir::failure();
    }

    const auto inputShape = getShape(fcOp.getInput());
    const auto outputShape = getShape(fcOp.getResult());
    if (inputShape.isDynamic() || outputShape.isDynamic()) {
        return mlir::failure();
    }
    const int64_t logicalRows = logicalIndexShape.totalSize();
    if (inputShape.back() != depth.value() || inputShape.totalSize() != logicalRows * depth.value()) {
        return mlir::failure();
    }
    if (inputShape.size() != 2 || outputShape.size() != 2 || outputShape[Dim(0)] != logicalRows) {
        return mlir::failure();
    }

    auto weights = matchWeightChain(fcOp.getWeights());
    if (mlir::failed(weights) || weights.value().inputChannels != depth.value() ||
        outputShape.back() != weights.value().outputChannels) {
        return mlir::failure();
    }

    const auto outputElemType = mlir::cast<NDTypeInterface>(fcOp.getResult().getType()).getElementType();
    const auto weightElemType = mlir::cast<NDTypeInterface>(fcOp.getWeights().getType()).getElementType();
    if (outputElemType != weightElemType) {
        return mlir::failure();
    }

    const int64_t M = weights.value().outputChannels;
    const int64_t K = weights.value().inputChannels;

    const Shape conditionShape{logicalIndexShape[Dim(0)], logicalIndexShape[Dim(1)], 1};
    if (getShape(selectOp.getInput1()) != conditionShape) {
        return mlir::failure();
    }

    auto* ctx = rewriter.getContext();
    const auto loc = fcOp.getLoc();
    auto table = createOneHotTable(rewriter, loc, weights.value());
    if (mlir::failed(table)) {
        return mlir::failure();
    }

    // Make indices safe for Gather. For positions selected by the original mask, zero is a harmless
    // fallback because the result is replaced by NaN below. Negative and out-of-range indices
    // represent all-off one-hot vectors and must therefore produce a zero vector after Gather.
    mlir::Value logicalIndex = index;
    if (rankExpanded) {
        logicalIndex = IE::ReshapeOp::create(rewriter, appendLoc(loc, "onehot_index_2d"), index,
                                             getIntArrayAttr(ctx, to_small_vector(logicalIndexShape)))
                               .getOutput();
    }
    const auto indexElemType = mlir::cast<NDTypeInterface>(logicalIndex.getType()).getElementType();
    auto zeroIndex = createIndexConst(rewriter, appendLoc(loc, "zero_index"), indexElemType, 0);
    auto depthIndex = createIndexConst(rewriter, appendLoc(loc, "onehot_depth"), indexElemType, depth.value());
    auto negativeIndex = IE::LessOp::create(rewriter, appendLoc(loc, "onehot_is_negative"), logicalIndex, zeroIndex,
                                            IE::AutoBroadcastType::NUMPY);
    auto outOfRangeIndex = IE::GreaterEqualOp::create(rewriter, appendLoc(loc, "onehot_is_out_of_range"), logicalIndex,
                                                      depthIndex, IE::AutoBroadcastType::NUMPY);
    auto invalidIndex =
            IE::LogicalOrOp::create(rewriter, appendLoc(loc, "onehot_is_invalid"), negativeIndex.getOutput(),
                                    outOfRangeIndex.getOutput(), IE::AutoBroadcastType::NUMPY);
    auto invalidIndex3D =
            IE::ReshapeOp::create(rewriter, appendLoc(loc, "onehot_is_invalid_3d"), invalidIndex.getOutput(),
                                  getIntArrayAttr(ctx, to_small_vector(conditionShape)));
    auto invalid2D = IE::ReshapeOp::create(rewriter, appendLoc(loc, "onehot_invalid_2d"), selectOp.getInput1(),
                                           getIntArrayAttr(ctx, to_small_vector(logicalIndexShape)));
    auto safeIndexForInvalid =
            IE::SelectOp::create(rewriter, appendLoc(loc, "onehot_safe_invalid"), invalidIndex.getOutput(), zeroIndex,
                                 logicalIndex, IE::AutoBroadcastType::NUMPY);
    auto safeIndex = IE::SelectOp::create(rewriter, appendLoc(loc, "onehot_safe_index"), invalid2D.getOutput(),
                                          zeroIndex, safeIndexForInvalid.getOutput(), IE::AutoBroadcastType::NUMPY);
    auto gather = IE::GatherOp::create(rewriter, appendLoc(loc, "onehot_gather"), table.value(), safeIndex.getOutput(),
                                       0, 0, nullptr);

    const auto zeroVectorType = mlir::RankedTensorType::get({1, 1, M}, weightElemType);
    auto zeroVector =
            Const::createDenseConst<float>(rewriter, appendLoc(loc, "onehot_zero_vector"), zeroVectorType, 0.0f);
    auto safeGather = IE::SelectOp::create(rewriter, appendLoc(loc, "onehot_zero_invalid"), invalidIndex3D.getOutput(),
                                           zeroVector, gather.getOutput(), IE::AutoBroadcastType::NUMPY);

    mlir::Value computed = safeGather.getOutput();

    mlir::Value result = IE::SelectOp::create(rewriter, appendLoc(loc, "onehot_invalid_result"), selectOp.getInput1(),
                                              selectOp.getInput2(), computed, IE::AutoBroadcastType::NUMPY)
                                 .getOutput();
    if (getShape(result) != outputShape) {
        result = IE::ReshapeOp::create(rewriter, appendLoc(loc, "onehot_linear_output_reshape"), result,
                                       getIntArrayAttr(ctx, to_small_vector(outputShape)))
                         .getOutput();
    }
    rewriter.replaceOp(fcOp, result);
    log.trace("[{0}] Replaced one-hot FullyConnected with Gather table [{1}, {2}]", debugName, K, M);
    return mlir::success();
}

// Returns the one-hot depth N when `oneHotOp` is a plain index selector along axis 0 with
// on_value == 1 and off_value == 0 (so that one-hot multiply + reduce is an exact index selection);
mlir::FailureOr<int64_t> getPlainOneHotDepth(IE::OneHotOp oneHotOp) {
    if (oneHotOp.getAxisAttr() != 0) {
        return mlir::failure();
    }
    const auto depth = getScalarConstOrAttr<int64_t>(oneHotOp.getDepth(), oneHotOp.getDepthAttrAttr());
    const auto onValue = getScalarConstOrAttr<double>(oneHotOp.getOnValue(), oneHotOp.getOnValueAttrAttr());
    const auto offValue = getScalarConstOrAttr<double>(oneHotOp.getOffValue(), oneHotOp.getOffValueAttrAttr());
    if (mlir::failed(depth) || mlir::failed(onValue) || mlir::failed(offValue)) {
        return mlir::failure();
    }
    if (onValue.value() != 1.0 || offValue.value() != 0.0) {
        return mlir::failure();
    }
    return depth.value();
}

struct OneHotReduceSelect {
    IE::ConcatOp concatOp;
    IE::OneHotOp oneHotOp;
    int64_t depth;
};

mlir::FailureOr<OneHotReduceSelect> matchOneHotReduceSelect(IE::ReduceSumOp reduceOp) {
    // The selection must sum over the stack axis (axis 0) and drop it.
    if (reduceOp.getKeepDims()) {
        return mlir::failure();
    }
    const auto axesVals = parseIntArrayAttr<int64_t>(reduceOp.getAxesValue());
    if (axesVals.size() != 1 || axesVals.front() != 0) {
        return mlir::failure();
    }

    auto mulOp = reduceOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (mulOp == nullptr) {
        return mlir::failure();
    }
    // The one-hot selector [N, 1, 1] broadcasts against the branch concat [N, 1, X]; require NUMPY broadcast.
    if (mulOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        return mlir::failure();
    }

    // Identify which Multiply operand is the concat of branches and which is the one-hot selector.
    IE::ConcatOp concatOp = nullptr;
    mlir::Value selectorVal = nullptr;
    for (auto operand : {mulOp.getInput1(), mulOp.getInput2()}) {
        if (auto candidate = operand.getDefiningOp<IE::ConcatOp>()) {
            concatOp = candidate;
        } else {
            selectorVal = operand;
        }
    }
    if (concatOp == nullptr || selectorVal == nullptr) {
        return mlir::failure();
    }
    // Concat must stack the branches along axis 0 (plain per-axis concat) so that ReduceSum over axis 0
    // implements branch selection.
    const auto concatAxis = IE::getConcatAxis(concatOp);
    if (!concatAxis.has_value() || concatAxis.value().ind() != 0) {
        return mlir::failure();
    }

    // Selector must be a genuine one-hot index selector (on==1, off==0, axis==0).
    auto oneHotOp = mlir::dyn_cast_if_present<IE::OneHotOp>(traceThroughViewLikeOps(selectorVal));
    if (oneHotOp == nullptr) {
        return mlir::failure();
    }
    const auto depth = getPlainOneHotDepth(oneHotOp);
    if (mlir::failed(depth)) {
        return mlir::failure();
    }
    const int64_t N = depth.value();

    // Concat must stack exactly N branches along axis 0.
    if (checked_cast<int64_t>(concatOp.getInputs().size()) != N) {
        return mlir::failure();
    }

    return OneHotReduceSelect{concatOp, oneHotOp, N};
}

//
// FuseGroupedMatMulSelect
//
// Recognizes the "compute all N codebooks then one-hot select one" pattern and rewrites it into a
// single weight gather + matmul, so that only the selected codebook weight is loaded at runtime.
//
//   logits = ReduceSum_axis0( onehot(group_id) * Concat_axis0( hidden @ W_i^T  for i in 0..N-1 ) )
//          = hidden @ W_group_id^T
//
// After:
//   stackedW = const [N, M, K]
//   selW     = Reshape( Gather(stackedW, group_id, axis=0) )   -> [M, K]
//   logits   = Convert( hidden @ selW^T )
//
class FuseGroupedMatMulSelect final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    FuseGroupedMatMulSelect(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("FuseGroupedMatMulSelect");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp reduceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseGroupedMatMulSelect::matchAndRewrite(IE::ReduceSumOp reduceOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), reduceOp->getName(), reduceOp->getLoc());
    auto* ctx = rewriter.getContext();

    const auto matched = matchOneHotReduceSelect(reduceOp);
    if (mlir::failed(matched)) {
        return mlir::failure();
    }

    auto matchedValue = matched.value();
    auto oneHotOp = matchedValue.oneHotOp;
    auto concatOp = matchedValue.concatOp;
    const int64_t N = matchedValue.depth;
    const auto concatInputs = concatOp.getInputs();

    // Every branch must compute sharedAct @ W_i^T via either MatMul(transpose_b) or FullyConnected
    // (both equivalent), with a constant weight. The fused op is always built as MatMul(transpose_b).
    mlir::Value sharedAct = nullptr;
    SmallVector<Const::DeclareOp> weights;
    weights.reserve(N);
    for (auto branch : concatInputs) {
        mlir::Operation* branchOp = traceThroughViewLikeOps(branch);
        mlir::Value act = nullptr;
        mlir::Value weight = nullptr;
        if (auto matMulOp = mlir::dyn_cast_if_present<IE::MatMulOp>(branchOp)) {
            if (!matMulOp.getTransposeB() || matMulOp.getTransposeA()) {
                return mlir::failure();
            }
            act = matMulOp.getInput1();
            weight = matMulOp.getInput2();
        } else if (auto fcOp = mlir::dyn_cast_if_present<IE::FullyConnectedOp>(branchOp)) {
            if (fcOp.getBias() != nullptr) {
                return mlir::failure();
            }
            act = fcOp.getInput();
            weight = fcOp.getWeights();
        } else {
            return mlir::failure();
        }
        if (sharedAct == nullptr) {
            sharedAct = act;
        } else if (act != sharedAct) {
            return mlir::failure();
        }
        auto weightConst = weight.getDefiningOp<Const::DeclareOp>();
        if (weightConst == nullptr) {
            return mlir::failure();
        }
        weights.push_back(weightConst);
    }

    // All weights must share the same [M, K] f16 shape so they can be stacked into [N, M, K].
    const auto firstWType = mlir::cast<vpux::NDTypeInterface>(weights.front().getOutput().getType());
    const auto firstWShape = firstWType.getShape();
    if (firstWShape.size() != 2 || !firstWType.getElementType().isF16()) {
        return mlir::failure();
    }
    const int64_t M = firstWShape[Dim(0)];
    const int64_t K = firstWShape[Dim(1)];
    for (auto weightConst : weights) {
        const auto wType = mlir::cast<vpux::NDTypeInterface>(weightConst.getOutput().getType());
        if (wType.getShape() != firstWShape || !wType.getElementType().isF16()) {
            return mlir::failure();
        }
    }

    // The selector index feeding the one-hot is the group id; reuse it directly to gather the weight.
    // It must select a single codebook (one index), otherwise the gathered weight is not [M, K].
    auto groupId = oneHotOp.getInput();
    if (mlir::cast<vpux::NDTypeInterface>(groupId.getType()).getShape().totalSize() != 1) {
        return mlir::failure();
    }

    // The fused MatMul (sharedAct[A0, K] @ W[M, K]^T -> [A0, M]) must reproduce the ReduceSum output shape
    const auto actShape = getShape(sharedAct);
    const auto reduceOutShape = getShape(reduceOp.getOutput());
    if (actShape.size() != 2 || actShape[Dim(1)] != K) {
        return mlir::failure();
    }
    if (reduceOutShape.size() != 2 || reduceOutShape[Dim(0)] != actShape[Dim(0)] || reduceOutShape[Dim(1)] != M) {
        return mlir::failure();
    }

    const auto loc = reduceOp.getLoc();

    // Stack the N weight constants into a single [N, M, K] f16 constant.
    SmallVector<vpux::type::float16> stackedData;
    stackedData.reserve(checked_cast<size_t>(N * M * K));
    for (auto weightConst : weights) {
        const auto wContent = weightConst.getContent();
        const auto vals = to_small_vector(wContent.getValues<vpux::type::float16>());
        stackedData.append(vals.begin(), vals.end());
    }
    const auto stackedType = mlir::RankedTensorType::get({N, M, K}, mlir::Float16Type::get(ctx));
    auto stackedWeight = Const::createConst<vpux::type::float16>(rewriter, appendLoc(loc, "stacked_w"), stackedType,
                                                                 ArrayRef(stackedData));

    // Gather the selected weight: [N, M, K] -> [1, M, K] -> [M, K].
    auto gatherOp = IE::GatherOp::create(rewriter, appendLoc(loc, "gather_w"), stackedWeight, groupId,
                                         /*axis_value=*/0, /*batch_dims=*/0,
                                         /*indices_rank=*/nullptr);
    auto selectedWeight = IE::ReshapeOp::create(rewriter, appendLoc(loc, "reshape_w"), gatherOp.getOutput(),
                                                getIntArrayAttr(ctx, SmallVector<int64_t>{M, K}));

    auto newMatMul = IE::MatMulOp::create(rewriter, appendLoc(loc, "select_matmul"), sharedAct,
                                          selectedWeight.getOutput(), /*transpose_a=*/false,
                                          /*transpose_b=*/true);
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(reduceOp.getOutput().getType()).getElementType();
    mlir::Value result = newMatMul.getOutput();
    if (mlir::cast<vpux::NDTypeInterface>(result.getType()).getElementType() != outElemType) {
        result = IE::ConvertOp::create(rewriter, appendLoc(loc, "out_cvt"), result, mlir::TypeAttr::get(outElemType))
                         .getOutput();
    }

    rewriter.replaceOp(reduceOp, result);
    _log.trace("[{0}] Fused {1}-way one-hot MatMul select into gather+matmul", this->getDebugName(), N);
    return mlir::success();
}

//
// FuseGroupedGatherSelect
//
// Detects the "gather all N codebooks' row then one-hot select one" pattern and rewrites it into a
// single flat-index gather, so that only the selected row is loaded at runtime. Every branch gathers
// one row of a constant codebook E_i[V, D] by a shared token index, and the branches are one-hot
// selected by group_id.
//
//   embs = ReduceSum_axis0( onehot(group_id) * Concat_axis0( Gather(E_i, token_id)  for i in 0..N-1 ) )
//        = E_group_id[token_id]
//
// After:
//   stackedE = const [N*V, D]
//   flatIdx  = group_id * V + token_id
//   embs     = Convert( Gather(stackedE, flatIdx, axis=0) )   -> [1, D]
//
class FuseGroupedGatherSelect final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    FuseGroupedGatherSelect(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("FuseGroupedGatherSelect");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp reduceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseGroupedGatherSelect::matchAndRewrite(IE::ReduceSumOp reduceOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), reduceOp->getName(), reduceOp->getLoc());
    auto* ctx = rewriter.getContext();

    const auto matched = matchOneHotReduceSelect(reduceOp);
    if (mlir::failed(matched)) {
        return mlir::failure();
    }
    auto matchedValue = matched.value();
    auto oneHotOp = matchedValue.oneHotOp;
    auto concatOp = matchedValue.concatOp;
    const int64_t N = matchedValue.depth;
    const auto concatInputs = concatOp.getInputs();

    // Every branch must gather one row of a constant codebook E_i along axis 0, sharing the same
    // token index across all branches.
    mlir::Value sharedIdx = nullptr;
    SmallVector<Const::DeclareOp> tables;
    tables.reserve(N);
    for (auto branch : concatInputs) {
        auto gatherOp = mlir::dyn_cast_if_present<IE::GatherOp>(traceThroughViewLikeOps(branch));
        if (gatherOp == nullptr || gatherOp.getBatchDims() != 0) {
            return mlir::failure();
        }
        // Gather must be along axis 0.
        const auto axis = gatherOp.getAxisValue();
        if (axis != 0) {
            return mlir::failure();
        }
        if (sharedIdx == nullptr) {
            sharedIdx = gatherOp.getIndices();
        } else if (gatherOp.getIndices() != sharedIdx) {
            return mlir::failure();
        }
        auto tableConst = gatherOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (tableConst == nullptr) {
            return mlir::failure();
        }
        tables.push_back(tableConst);
    }

    // All codebooks must share the same [V, D] f16 shape so they can be stacked into [N*V, D].
    const auto firstType = mlir::cast<vpux::NDTypeInterface>(tables.front().getOutput().getType());
    const auto firstShape = firstType.getShape();
    if (firstShape.size() != 2 || !firstType.getElementType().isF16()) {
        return mlir::failure();
    }
    const int64_t V = firstShape[Dim(0)];
    const int64_t D = firstShape[Dim(1)];
    for (auto tableConst : tables) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(tableConst.getOutput().getType());
        if (type.getShape() != firstShape || !type.getElementType().isF16()) {
            return mlir::failure();
        }
    }

    // group_id (one-hot input) and token_id (shared gather index) must each select a single element,
    // so the fused gather returns exactly one [1, D] row.
    auto groupId = oneHotOp.getInput();
    const auto groupIdType = mlir::cast<vpux::NDTypeInterface>(groupId.getType());
    const auto tokenType = mlir::cast<vpux::NDTypeInterface>(sharedIdx.getType());
    if (groupIdType.getShape().totalSize() != 1 || tokenType.getShape().totalSize() != 1) {
        return mlir::failure();
    }

    // The fused gather output [1, D] must reproduce the ReduceSum output shape.
    const auto reduceOutShape = getShape(reduceOp.getOutput());
    if (reduceOutShape.size() != 2 || reduceOutShape[Dim(0)] != 1 || reduceOutShape[Dim(1)] != D) {
        return mlir::failure();
    }

    const auto loc = reduceOp.getLoc();

    // Stack the N codebooks into one [N*V, D] f16 constant (row-major concat along axis 0).
    SmallVector<vpux::type::float16> stackedData;
    stackedData.reserve(checked_cast<size_t>(N * V * D));
    for (auto tableConst : tables) {
        const auto content = tableConst.getContent();
        const auto vals = to_small_vector(content.getValues<vpux::type::float16>());
        stackedData.append(vals.begin(), vals.end());
    }
    const auto stackedType = mlir::RankedTensorType::get({N * V, D}, mlir::Float16Type::get(ctx));
    auto stackedTable = Const::createConst<vpux::type::float16>(rewriter, appendLoc(loc, "stacked_e"), stackedType,
                                                                ArrayRef(stackedData));

    // flatIdx = group_id * V + token_id, computed in the shared index element type.
    mlir::Value groupIdIdx = groupId;
    const auto idxElemType = tokenType.getElementType();
    if (groupIdType.getElementType() != idxElemType) {
        groupIdIdx =
                IE::ConvertOp::create(rewriter, appendLoc(loc, "gid_cvt"), groupId, mlir::TypeAttr::get(idxElemType))
                        .getOutput();
    }
    const auto makeIdxConst = [&](int64_t value) -> mlir::Value {
        const auto constType = mlir::RankedTensorType::get({1}, idxElemType);
        if (idxElemType.getIntOrFloatBitWidth() == 64) {
            return Const::createConst(rewriter, loc, constType, ArrayRef<int64_t>({value}));
        }
        return Const::createConst(rewriter, loc, constType, ArrayRef<int32_t>({checked_cast<int32_t>(value)}));
    };
    auto vConst = makeIdxConst(V);
    auto scaled = IE::MultiplyOp::create(rewriter, appendLoc(loc, "idx_scale"), groupIdIdx, vConst,
                                         IE::AutoBroadcastType::NUMPY, /*post_op=*/nullptr, /*clamp=*/nullptr,
                                         /*output_padding=*/nullptr, /*input_padding=*/nullptr);
    auto flatIdx = IE::AddOp::create(rewriter, appendLoc(loc, "idx_offset"), scaled.getOutput(), sharedIdx,
                                     IE::AutoBroadcastType::NUMPY, /*post_op=*/nullptr, /*clamp=*/nullptr,
                                     /*output_padding=*/nullptr, /*input_padding=*/nullptr);

    // Gather the selected row: [N*V, D] -> [*idx, D].
    auto gatherOp = IE::GatherOp::create(rewriter, appendLoc(loc, "gather_e"), stackedTable, flatIdx.getOutput(),
                                         /*axis_value=*/0, /*batch_dims=*/0,
                                         /*indices_rank=*/nullptr);

    mlir::Value result = gatherOp.getOutput();
    if (getShape(result) != reduceOutShape) {
        result = IE::ReshapeOp::create(rewriter, appendLoc(loc, "reshape_out"), result,
                                       getIntArrayAttr(ctx, to_small_vector(reduceOutShape)))
                         .getOutput();
    }
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(reduceOp.getOutput().getType()).getElementType();
    if (mlir::cast<vpux::NDTypeInterface>(result.getType()).getElementType() != outElemType) {
        result = IE::ConvertOp::create(rewriter, appendLoc(loc, "out_cvt"), result, mlir::TypeAttr::get(outElemType))
                         .getOutput();
    }

    rewriter.replaceOp(reduceOp, result);
    _log.trace("[{0}] Fused {1}-way one-hot Gather select into flat-index gather", this->getDebugName(), N);
    return mlir::success();
}

//
// FuseOneHotSelectPass
//

class FuseOneHotSelectPass final : public IE::impl::FuseOneHotSelectBase<FuseOneHotSelectPass> {
public:
    explicit FuseOneHotSelectPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseOneHotSelectPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseOneHotFullyConnectedToGather>(&ctx, _log);
    patterns.add<FuseGroupedMatMulSelect>(&ctx, _log);
    patterns.add<FuseGroupedGatherSelect>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseOneHotSelectPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOneHotSelectPass(Logger log) {
    return std::make_unique<FuseOneHotSelectPass>(log);
}
