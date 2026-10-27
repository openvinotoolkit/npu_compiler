//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPOPERATIONSWITHGATHERANDSLICE
#define GEN_PASS_DEF_SWAPOPERATIONSWITHGATHERANDSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

static bool isPerTensorShape(ShapeRef shape) {
    return llvm::all_of(shape, [](int64_t d) {
        return d == 1;
    });
}

//
// MoveTwoInputsEltwiseOpAfterGather
//

template <class ConcreteOp>
class MoveTwoInputsEltwiseOpAfterGather final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    MoveTwoInputsEltwiseOpAfterGather(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("MoveTwoInputsEltwiseOpAfterGather");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) const;
    std::optional<ConcreteOp> getSupportedOp(IE::GatherOp gatherOp) const;
    mlir::Value createGatherOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                               IE::GatherOp gatherOp) const;
    const Dim SUPPORTED_GATHER_AXIS = Dim(0);

    Logger _log;
};

template <class ConcreteOp>
bool MoveTwoInputsEltwiseOpAfterGather<ConcreteOp>::isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) const {
    return inShape.totalSize() > outShape.totalSize();
}

template <class ConcreteOp>
std::optional<ConcreteOp> MoveTwoInputsEltwiseOpAfterGather<ConcreteOp>::getSupportedOp(IE::GatherOp gatherOp) const {
    if (gatherOp.getAxisValue() != SUPPORTED_GATHER_AXIS.ind()) {
        _log.trace("Only support GatherOp with axis on the first dim");
        return std::nullopt;
    }

    auto op = gatherOp.getInput().getDefiningOp<ConcreteOp>();
    if (op == nullptr || !op->hasOneUse()) {
        return std::nullopt;
    }

    if constexpr (std::is_same_v<ConcreteOp, IE::DynamicDequantizeOp>) {
        if (op.getZp() != nullptr) {
            return std::nullopt;
        }
        if (op->hasAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR)) {
            return std::nullopt;
        }
    } else {
        if (op.getPostOpAttr() != nullptr || op.getClampAttr() != nullptr || op.getOutputPaddingAttr() != nullptr ||
            op.getInputPaddingAttr() != nullptr) {
            return std::nullopt;
        }
    }

    auto outputShape = getShape(op->getResult(0));
    // Per-tensor operands (all dimensions equal to 1) are invariant to which rows are selected
    // and can be reused directly without gathering. All other operands must have a gather-axis
    // size that matches the output; otherwise the transformation is invalid.
    auto outputGatherAxisSize = outputShape[SUPPORTED_GATHER_AXIS];
    auto hasIncompatibleGatherAxis = [this, outputGatherAxisSize](mlir::Value operand) {
        auto shape = getShape(operand);
        if (isPerTensorShape(shape)) {
            return false;
        }
        return shape[SUPPORTED_GATHER_AXIS] != outputGatherAxisSize;
    };
    if (llvm::any_of(op->getOperands(), hasIncompatibleGatherAxis)) {
        return std::nullopt;
    }

    return op;
}

template <class ConcreteOp>
mlir::Value MoveTwoInputsEltwiseOpAfterGather<ConcreteOp>::createGatherOp(mlir::PatternRewriter& rewriter,
                                                                          mlir::Location loc, mlir::Value input,
                                                                          IE::GatherOp gatherOp) const {
    return IE::GatherOp::create(rewriter, appendLoc(loc, "gather"), input, gatherOp.getIndices(),
                                gatherOp.getAxisValue(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
}

template <class ConcreteOp>
mlir::LogicalResult MoveTwoInputsEltwiseOpAfterGather<ConcreteOp>::matchAndRewrite(
        IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", gatherOp->getName(), gatherOp->getLoc());

    // Conversion is benificial when GatherOp is reducing tensor size.
    auto inputShapeSize = getShape(gatherOp.getInput()).toValues();
    auto outputShapeSize = getShape(gatherOp.getOutput()).toValues();
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(gatherOp.getInput().getType())) {
        inputShapeSize = Shape(boundedType.getBounds().raw());
    }
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(gatherOp.getOutput().getType())) {
        outputShapeSize = Shape(boundedType.getBounds().raw());
    }
    if (!isBeneficialToConvert(inputShapeSize, outputShapeSize)) {
        return matchFailed(_log.nest(), rewriter, gatherOp, "Not beneficial to move operation after GatherOp");
    }

    auto getOp = getSupportedOp(gatherOp);
    if (!getOp.has_value()) {
        return mlir::failure();
    }
    auto op = getOp.value();

    auto gatherLoc = gatherOp->getLoc();
    // Per-tensor operands (all dimensions equal to 1) are row-selection-invariant and reused
    // directly; all other operands are gathered along the gather axis.
    auto maybeGatherOperand = [&](mlir::Value operand, mlir::StringRef suffix) -> mlir::Value {
        auto shape = getShape(operand);
        if (isPerTensorShape(shape)) {
            return operand;
        }
        return createGatherOp(rewriter, appendLoc(gatherLoc, suffix), operand, gatherOp);
    };
    auto newOperand0 = maybeGatherOperand(op->getOperand(0), "new_lhs");
    auto newOperand1 = maybeGatherOperand(op->getOperand(1), "new_rhs");

    mlir::IRMapping opMapper;
    opMapper.map(op->getOperand(0), newOperand0);
    opMapper.map(op->getOperand(1), newOperand1);
    auto newOp = rewriter.clone(*op, opMapper);

    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    _log.trace("Successfully replaced '{0}' at '{1}'", gatherOp->getName(), gatherLoc);

    rewriter.replaceOp(gatherOp, newOp->getResult(0));

    return mlir::success();
}

//
// MoveConvertAfterGather
//

class MoveConvertAfterGather final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    MoveConvertAfterGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("MoveConvertAfterGather");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isBeneficialToConvert(IE::ConvertOp convertOp, IE::GatherOp gatherOp) const;
    Logger _log;
};

// Conversion is beneficial when ConvertOp increases tensor size and GatherOp reduces tensor size:
// This is a definite positive optimization for this case because the costs of both GatherOp and ConvertOp are
// decreased after the transformation.
// TODO: Develop a cost model to determine if conversion is beneficial in other cases, such as when both ConvertOp
// and GatherOp are reducing tensor size.
bool MoveConvertAfterGather::isBeneficialToConvert(IE::ConvertOp convertOp, IE::GatherOp gatherOp) const {
    auto getIORatio = [](NDTypeInterface inType, NDTypeInterface outType) {
        return checked_cast<double>(inType.getTotalAllocSize().count()) /
               checked_cast<double>(outType.getTotalAllocSize().count());
    };

    auto convertIORatio = getIORatio(convertOp.getInput().getType(), convertOp.getOutput().getType());
    auto gatherIORatio = getIORatio(gatherOp.getInput().getType(), gatherOp.getOutput().getType());

    // Definite positive optimization: ConvertOp increases tensor size and GatherOp reduces tensor size.
    const bool convertIncreaseGatherReduces = convertIORatio < 1.0 && gatherIORatio > 1.0;

    // Heuristic win: the pre-gather data volume is large enough relative to the post-gather
    // volume that swapping is beneficial regardless of the Convert direction.
    static constexpr double GATHERINOUTRATIO = 10;
    auto gatherInShape = getShape(gatherOp.getInput());
    auto gatherOutShape = getShape(gatherOp.getOutput());
    if (gatherInShape.isDynamic() || gatherOutShape.isDynamic()) {
        return false;
    }
    const auto gatherInSize = gatherInShape.totalSize();
    const auto gatherOutSize = gatherOutShape.totalSize();
    const Bit inTypeSize = getElemTypeSize(convertOp.getInput().getType());
    const Bit outTypeSize = getElemTypeSize(convertOp.getOutput().getType());
    const bool gatherDataReductionExceedsThreshold =
            gatherInSize * inTypeSize.count() + gatherOutSize * outTypeSize.count() >
            GATHERINOUTRATIO * (gatherOutSize * inTypeSize.count() + gatherOutSize * outTypeSize.count());

    return convertIncreaseGatherReduces || gatherDataReductionExceedsThreshold;
}

mlir::LogicalResult MoveConvertAfterGather::matchAndRewrite(IE::GatherOp gatherOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto gatherOpName = gatherOp->getName();
    const auto gatherOpLoc = gatherOp->getLoc();
    _log.trace("Got '{0}' at '{1}'", gatherOpName, gatherOpLoc);

    auto convertOp = gatherOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr || !convertOp->hasOneUse()) {
        return mlir::failure();
    }

    if (!isBeneficialToConvert(convertOp, gatherOp)) {
        return matchFailed(_log.nest(), rewriter, gatherOp, "Not beneficial to move operation after GatherOp");
    }

    auto newGather =
            IE::GatherOp::create(rewriter, gatherOpLoc, convertOp.getInput(), gatherOp.getIndices(),
                                 gatherOp.getAxisValue(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
    auto newConvert =
            IE::ConvertOp::create(rewriter, convertOp->getLoc(), newGather.getOutput(), convertOp.getDstElemType());

    rewriter.replaceOp(gatherOp, newConvert.getOutput());

    _log.trace("Successfully replaced '{0}' at '{1}'", gatherOpName, gatherOpLoc);

    return mlir::success();
}

//
// MoveConvertBeforeGather
//
// Hoist IE.Convert before IE.GatherOp (optionally through IE.AffineReshapeOp) to
// reduce the data volume that GatherOp must process.
//
// Pattern:
//   Gather(input, indices) -> [AffineReshape ->] Convert(dstElemType)
// becomes:
//   Convert(input, dstElemType) -> Gather(converted_input, indices) -> [AffineReshape]
//
// Applied when:
//   1. Convert reduces element byte size — so Gather reads and writes fewer bytes.
//   2. GatherOp does not increase element count — so the hoisted Convert does not
//      operate on more elements than the original Convert did.

class MoveConvertBeforeGather final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    MoveConvertBeforeGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("MoveConvertBeforeGather");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveConvertBeforeGather::matchAndRewrite(IE::ConvertOp convertOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());

    // Match: GatherOp -> [AffineReshapeOp ->] ConvertOp.
    IE::GatherOp gatherOp;
    IE::AffineReshapeOp reshapeOp;

    if (auto reshape = convertOp.getInput().getDefiningOp<IE::AffineReshapeOp>()) {
        if (!reshape.getOutput().hasOneUse()) {
            return mlir::failure();
        }
        reshapeOp = reshape;
        gatherOp = reshape.getInput().getDefiningOp<IE::GatherOp>();
    } else {
        gatherOp = convertOp.getInput().getDefiningOp<IE::GatherOp>();
    }

    if (!gatherOp || !gatherOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    // Condition 1: Convert must reduce element byte size to benefit Gather data movement.
    const Bit inTypeSize = getElemTypeSize(convertOp.getInput().getType());
    const Bit outTypeSize = getElemTypeSize(convertOp.getOutput().getType());
    if (outTypeSize >= inTypeSize) {
        return matchFailed(_log.nest(), rewriter, convertOp,
                           "Not beneficial: Convert does not reduce element byte size");
    }

    // Condition 2: GatherOp must not increase element count, so the hoisted Convert
    // does not process more elements than the original Convert.
    auto gatherInputShape = getShape(gatherOp.getInput());
    auto convertInputShape = getShape(convertOp.getInput());
    if (gatherInputShape.isDynamic() || convertInputShape.isDynamic()) {
        return mlir::failure();
    }
    if (gatherInputShape.totalSize() > convertInputShape.totalSize()) {
        return matchFailed(_log.nest(), rewriter, convertOp,
                           "Not beneficial: hoisted Convert would process more elements than original");
    }

    const auto convertLoc = convertOp->getLoc();
    const auto gatherLoc = gatherOp->getLoc();

    // Hoist Convert before Gather.
    auto newConvert = IE::ConvertOp::create(rewriter, appendLoc(convertLoc, "pre_gather"), gatherOp.getInput(),
                                            convertOp.getDstElemType());

    // Re-create Gather on the converted input.
    auto newGather =
            IE::GatherOp::create(rewriter, gatherLoc, newConvert.getOutput(), gatherOp.getIndices(),
                                 gatherOp.getAxisValue(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());

    mlir::Value result = newGather.getOutput();

    if (reshapeOp) {
        result = IE::AffineReshapeOp::create(rewriter, reshapeOp->getLoc(), newGather.getOutput(),
                                             reshapeOp.getDimMappingAttr(), reshapeOp.getShapeValueAttr());
    }

    rewriter.replaceOp(convertOp, result);
    if (reshapeOp) {
        rewriter.eraseOp(reshapeOp);
    }
    rewriter.eraseOp(gatherOp);

    _log.trace("Successfully moved Convert before GatherOp at '{0}'", gatherLoc);
    return mlir::success();
}

bool isBeneficialForPropagation(ShapeRef sourceShape, ShapeRef outputShape) {
    return outputShape.totalSize() * 2 < sourceShape.totalSize();
}

//
// MoveSliceBeforeEltwise (generic two-input eltwise)
//
// If a slice only trims the result of a two-input elementwise op (Multiply, Add, ...),
// hoist that slice to the eltwise inputs so the operation is performed on smaller
// operands and less computation is required.
//
// Pattern (conceptual):
//   Eltwise(InputA, InputB) -> Slice(out)
// becomes
//   Slice(InputA) -> SliceA
//   Slice(InputB) -> SliceB
//   Eltwise(SliceA, SliceB) -> out
//
// Each operand is sliced per-dimension: dimensions that match the output are sliced
// with the same offset/size, while broadcast dimensions (operand extent 1) are left
// untouched. An operand that requires no slicing on any dimension (single element, or
// broadcast along the sliced axis) is reused directly.
//
// Apply this rewrite only when:
// - the eltwise output feeds a single Slice user,
// - the output slice actually reduces total element count, and
// - every operand is broadcast-compatible with the eltwise output.

template <typename EltwiseOpType>
class MoveSliceBeforeEltwise final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx),
              _log(log),
              _debugNameStorage(formatv("MoveSliceBefore{0}", EltwiseOpType::getOperationName())) {
        setDebugName(_debugNameStorage);
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    std::string _debugNameStorage;
};

template <typename EltwiseOpType>
mlir::LogicalResult MoveSliceBeforeEltwise<EltwiseOpType>::matchAndRewrite(IE::SliceOp sliceOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto eltwiseOp = sliceOp.getSource().template getDefiningOp<EltwiseOpType>();
    if (eltwiseOp == nullptr) {
        return mlir::failure();
    }

    if (!eltwiseOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    if (eltwiseOp.getInputPaddingAttr() != nullptr || eltwiseOp.getOutputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape != getShape(eltwiseOp.getOutput()) || sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    // Keep only profitable rewrites
    if (!isBeneficialForPropagation(sourceShape, outputShape)) {
        return mlir::failure();
    }

    const auto sliceOffsets = parseIntArrayAttr<int64_t>(staticOffsets);
    const auto sliceSizes = parseIntArrayAttr<int64_t>(staticSizes);

    const auto eltwiseOperands = eltwiseOp.getOperands();

    // Validate every operand and pre-compute its slice parameters before mutating the IR.
    // Creating a slice for an earlier operand and then bailing out on a later operand would
    // leave a dangling slice behind on each match attempt, which keeps the greedy driver
    // from ever reaching a fixpoint and makes the pass fail.
    struct OperandSlicePlan {
        bool needSlice = false;
        SmallVector<int64_t> offsets;
        SmallVector<int64_t> sizes;
    };
    SmallVector<OperandSlicePlan> operandPlans;
    operandPlans.reserve(eltwiseOperands.size());

    for (auto operand : eltwiseOperands) {
        const auto opShape = getShape(operand);
        if (opShape.size() != sourceShape.size()) {
            return mlir::failure();
        }

        OperandSlicePlan plan;
        plan.offsets.assign(opShape.size(), 0);
        plan.sizes.assign(opShape.raw().begin(), opShape.raw().end());
        for (size_t d = 0; d < opShape.size(); ++d) {
            if (opShape[Dim(d)] == sourceShape[Dim(d)]) {
                plan.offsets[d] = sliceOffsets[d];
                plan.sizes[d] = sliceSizes[d];
                if (sliceOffsets[d] != 0 || sliceSizes[d] != opShape[Dim(d)]) {
                    plan.needSlice = true;
                }
            } else if (opShape[Dim(d)] != 1) {
                // Operand is neither full-size nor broadcast on this dimension: not supported.
                return mlir::failure();
            }
        }

        operandPlans.push_back(std::move(plan));
    }

    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(eltwiseOperands.size());

    for (auto [operandIdx, operand] : llvm::enumerate(eltwiseOperands)) {
        const auto& plan = operandPlans[operandIdx];
        if (!plan.needSlice) {
            newInputs.push_back(operand);
            continue;
        }

        auto newSlice = IE::SliceOp::create(rewriter, takeOpLoc(sliceOp, "slice_{0}", operandIdx), operand,
                                            getIntArrayAttr(rewriter.getContext(), plan.offsets),
                                            getIntArrayAttr(rewriter.getContext(), plan.sizes));
        newInputs.push_back(newSlice.getResult());
    }

    mlir::IRMapping mapper;
    mapper.map(eltwiseOp.getOperands(), newInputs);
    auto* newEltwise = rewriter.clone(*eltwiseOp.getOperation(), mapper);
    newEltwise->setLoc(takeOpLoc(sliceOp, "as_eltwise"));
    newEltwise->getResult(0).setType(sliceOp.getResult().getType());

    rewriter.replaceOp(sliceOp, newEltwise->getResults());
    return mlir::success();
}

//
// MoveSliceBeforeActivation (generic)
//
// A generic pattern that hoists a Slice operation before an element-wise
// activation function, reducing the amount of data the activation must
// process. The activation type is given as a template parameter, so the
// same logic works for Swish, ReLU, Sigmoid, Tanh, etc.
//
// Pattern:
//   Activation(input) -> Slice(out)
// becomes
//   Slice(input) -> input_sliced
//   Activation(input_sliced) -> out

mlir::Value createSlicedActivation(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Operation* oldOp,
                                   mlir::Value slicedInput) {
    mlir::IRMapping mapper;
    mapper.map(oldOp->getOperand(0), slicedInput);

    auto* newOp = rewriter.clone(*oldOp, mapper);
    newOp->setLoc(loc);
    newOp->getResult(0).setType(slicedInput.getType());

    return newOp->getResult(0);
}

template <typename ActivationOpType>
class MoveSliceBeforeActivation final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeActivation(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx),
              _log(log),
              _debugNameStorage(formatv("MoveSliceBefore{0}", ActivationOpType::getOperationName())) {
        setDebugName(_debugNameStorage);
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    // debug name's lifetime must be managed by the rewriter
    std::string _debugNameStorage;
};

template <typename ActivationOpType>
mlir::LogicalResult MoveSliceBeforeActivation<ActivationOpType>::matchAndRewrite(
        IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto actOp = sliceOp.getSource().template getDefiningOp<ActivationOpType>();
    if (!actOp) {
        return mlir::failure();
    }

    if (!actOp->getResult(0).hasOneUse()) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    if (sourceShape != getShape(actOp->getResult(0))) {
        return mlir::failure();
    }
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    // Keep only profitable rewrites
    if (!isBeneficialForPropagation(sourceShape, outputShape)) {
        return mlir::failure();
    }

    auto newSlice = IE::SliceOp::create(rewriter, takeOpLoc(sliceOp, "slice_input"),
                                        actOp->getOperand(0),  // first operand is the data input
                                        staticOffsets, staticSizes);

    auto newResult = createSlicedActivation(rewriter, takeOpLoc(sliceOp, "act_sliced"), actOp.getOperation(),
                                            newSlice.getResult());

    rewriter.replaceOp(sliceOp, newResult);
    return mlir::success();
}

//
// MoveSliceBeforeFakeQuantize
//
// Hoist a Slice before a FakeQuantize so the (fake) quantization is applied to the
// smaller, sliced tensor. Only the data input is sliced; the four range operands
// (input_low/input_high/output_low/output_high) are passed through unchanged, which is
// valid when they are per-tensor scalars broadcast across the whole tensor. Cloning the
// FakeQuantize preserves the levels, low_fp_type and auto_broadcast attributes.
//
// Pattern:
//   FakeQuantize(data, inLow, inHigh, outLow, outHigh) -> Slice(out)
// becomes
//   Slice(data) -> dataSliced
//   FakeQuantize(dataSliced, inLow, inHigh, outLow, outHigh) -> out

class MoveSliceBeforeFakeQuantize final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeFakeQuantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("MoveSliceBeforeFakeQuantize");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveSliceBeforeFakeQuantize::matchAndRewrite(IE::SliceOp sliceOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto fqOp = sliceOp.getSource().getDefiningOp<IE::FakeQuantizeOp>();
    if (fqOp == nullptr) {
        return mlir::failure();
    }

    if (!fqOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape != getShape(fqOp.getOutput()) || sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    // Keep only profitable rewrites
    if (!isBeneficialForPropagation(sourceShape, outputShape)) {
        return mlir::failure();
    }

    // Only the data input is sliced; the range operands must be per-tensor scalars so
    // they stay valid after the data tensor is reduced.
    for (const auto range : {fqOp.getInputLow(), fqOp.getInputHigh(), fqOp.getOutputLow(), fqOp.getOutputHigh()}) {
        if (!isPerTensorShape(getShape(range))) {
            return mlir::failure();
        }
    }

    auto newSlice = IE::SliceOp::create(rewriter, takeOpLoc(sliceOp, "slice_input"), fqOp.getInput(), staticOffsets,
                                        staticSizes);

    mlir::IRMapping mapper;
    mapper.map(fqOp.getInput(), newSlice.getResult());
    auto* newFq = rewriter.clone(*fqOp.getOperation(), mapper);
    newFq->setLoc(takeOpLoc(sliceOp, "as_fq"));
    newFq->getResult(0).setType(sliceOp.getResult().getType());

    rewriter.replaceOp(sliceOp, newFq->getResults());
    return mlir::success();
}

// Re-enqueue the FullyConnected that produces `value` (if any) so a Slice that just landed on its
// (reshaped) output is propagated within the same greedy iteration instead of the next full walk.
// Returns true when a producer FullyConnected was found and re-enqueued.
bool reenqueueProducerFullyConnected(mlir::PatternRewriter& rewriter, mlir::Value value) {
    auto fcOp = value.getDefiningOp<IE::FullyConnectedOp>();
    if (fcOp == nullptr) {
        return false;
    }
    rewriter.modifyOpInPlace(fcOp, [] {});
    return true;
}

//
// PropagateSliceThroughFullyConnected
//
// Anchored on FullyConnected, propagate a downstream Slice through the op. These are FC
// tiling rules expressed on the IE dialect, dispatched by the sliced axis:
//
//   1) Batch-dim slice (single one-use consumer chain), optionally through one AffineReshape:
//        FullyConnected -> [AffineReshape ->] Slice
//      The Slice is hoisted onto the FC activations (input rows). This tiles the FC on its
//      "batch"/height dim, which becomes a Conv height once FC is lowered.
//
//   2) Output-channel-dim slice(s) (one or more consumers), through a Reshape/AffineReshape:
//        FullyConnected -> Reshape/AffineReshape -> Slice (one per branch)
//      The split is pushed onto the weights/bias so each branch computes only its channels.
//
class PropagateSliceThroughFullyConnected final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    PropagateSliceThroughFullyConnected(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
        setDebugName("PropagateSliceThroughFullyConnected");
    }

    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp fcOp, mlir::PatternRewriter& rewriter) const final;

private:
    // Case 1: hoist a batch-dim Slice onto the FC activations.
    mlir::LogicalResult moveSliceToActivations(IE::FullyConnectedOp fcOp, mlir::PatternRewriter& rewriter) const;
    // Case 2: split output-channel-dim Slice consumers onto the FC weights/bias.
    mlir::LogicalResult splitToWeights(IE::FullyConnectedOp fcOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

std::optional<size_t> getSingleDifferentDim(mlir::ArrayRef<int64_t> lhs, mlir::ArrayRef<int64_t> rhs) {
    if (lhs.size() != rhs.size()) {
        return std::nullopt;
    }

    std::optional<size_t> differentDim;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] == rhs[i]) {
            continue;
        }

        if (differentDim.has_value()) {
            return std::nullopt;
        }
        differentDim = i;
    }

    return differentDim;
}

// Maps a Slice acting on the output of a reshape (Reshape/AffineReshape) back onto the reshape's
// input coordinates. `sourceShape`/`outputShape` are the Slice source (reshape output) and result
// shapes; their single differing axis is the sliced axis. When that axis is the result of flattening
// several reshape-input dims (e.g. [B, C] -> [B, numHeads, headDim]), the offset/size are scaled onto
// the flat input axis, which requires every group dim preceding the sliced axis to be size 1 (otherwise
// the flattened slice is non-contiguous). Returns the offsets/sizes expressed on the reshape input shape.
mlir::FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> mapSliceToReshapeInput(
        ShapeRef inputShape, ShapeRef sourceShape, ShapeRef outputShape, mlir::ArrayRef<int64_t> sliceOffsets,
        mlir::ArrayRef<int64_t> sliceSizes) {
    if (sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    const auto sliceAxis = getSingleDifferentDim(sourceShape.raw(), outputShape.raw());
    if (!sliceAxis.has_value()) {
        return mlir::failure();
    }
    const auto axis = sliceAxis.value();

    std::optional<size_t> inputSliceAxis;
    int64_t innerSize = 1;

    // A leading unit dim prepended by the reshape (input [..] -> output [1, ..], identical tail) keeps a
    // 1:1 dim mapping with no flattening, so the sliced axis maps straight to `axis - 1`.
    const bool prependsUnitDim =
            sourceShape.size() == inputShape.size() + 1 && sourceShape.front() == 1 && axis > 0 &&
            std::equal(inputShape.raw().begin(), inputShape.raw().end(), std::next(sourceShape.raw().begin()));
    if (prependsUnitDim) {
        inputSliceAxis = axis - 1;
    } else {
        // reassociationMap[i] lists the input dims that reshape-output dim i maps to; a flattened
        // output dim maps to exactly one input dim.
        auto reassociationMap = IE::getReassociationMapExtension(sourceShape.raw(), inputShape.raw());
        if (mlir::failed(reassociationMap)) {
            return mlir::failure();
        }
        const auto& map = reassociationMap.value();
        if (map[axis].size() != 1) {
            return mlir::failure();
        }
        const auto inAxis = map[axis].front();
        inputSliceAxis = static_cast<size_t>(inAxis);

        // Walk the group of output dims flattened into the same input dim: dims before the sliced axis
        // must be size 1 for contiguity, dims after it scale the flat offset/size.
        for (size_t dim = 0; dim < sourceShape.size(); ++dim) {
            if (map[dim].size() != 1 || map[dim].front() != inAxis) {
                continue;
            }
            if (dim < axis) {
                if (sourceShape[Dim(static_cast<int64_t>(dim))] != 1) {
                    return mlir::failure();
                }
            } else if (dim > axis) {
                innerSize *= sourceShape[Dim(static_cast<int64_t>(dim))];
            }
        }
    }

    SmallVector<int64_t> newOffsets(inputShape.size(), 0);
    SmallVector<int64_t> newSizes(inputShape.raw());
    newOffsets[inputSliceAxis.value()] = sliceOffsets[axis] * innerSize;
    newSizes[inputSliceAxis.value()] = sliceSizes[axis] * innerSize;

    return std::make_pair(newOffsets, newSizes);
}

// Shared skeleton match for both FC slice-propagation cases:
//   FullyConnected -> [AffineReshape | Reshape ->] Slice(s)
// Returns the optional intermediate reshape op (nullptr when Slices consume the FC output
// directly) and all Slice consumers. The caller dispatches on the sliced axis: the batch
// dim (B) is handled by moveSliceToActivations, the output-channel dim (C) by splitToWeights.
struct FcReshapeSlicesMatch {
    mlir::Operation* reshapeOp = nullptr;
    SmallVector<IE::SliceOp> sliceConsumers;
};

std::optional<FcReshapeSlicesMatch> matchFcReshapeSlices(IE::FullyConnectedOp fcOp) {
    if (!fcOp.getOutput().hasOneUse()) {
        return std::nullopt;
    }

    mlir::Operation* user = *fcOp.getOutput().getUsers().begin();

    FcReshapeSlicesMatch match;
    mlir::Value sliceSource;
    if (mlir::isa<IE::SliceOp>(user)) {
        sliceSource = fcOp.getOutput();
    } else if (mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp>(user)) {
        match.reshapeOp = user;
        sliceSource = user->getResult(0);
    } else {
        return std::nullopt;
    }

    for (auto* sliceUser : sliceSource.getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(sliceUser);
        if (sliceOp == nullptr) {
            return std::nullopt;
        }
        match.sliceConsumers.push_back(sliceOp);
    }
    if (match.sliceConsumers.empty()) {
        return std::nullopt;
    }

    return match;
}

// Batch-dim propagation onto the FC activations is beneficial only when the FullyConnected feeds a
// single one-use chain FullyConnected -> [AffineReshape ->] Slice (the multi-Slice channel fan-out
// is handled by splitToWeights) and that terminal Slice actually shrinks the tensor. On success the
// matched Slice and optional intermediate AffineReshape are returned via the out-params.
bool isBeneficialForPropagation(IE::FullyConnectedOp fcOp, IE::SliceOp& sliceOp, IE::AffineReshapeOp& reshapeOp) {
    auto match = matchFcReshapeSlices(fcOp);
    if (!match.has_value()) {
        return false;
    }

    // Activations propagation only handles a single-Slice chain, optionally through an
    // AffineReshape (a plain Reshape is only supported by the weights split).
    if (match->sliceConsumers.size() != 1) {
        return false;
    }
    if (match->reshapeOp != nullptr) {
        reshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(match->reshapeOp);
        if (reshapeOp == nullptr) {
            return false;
        }
    }
    sliceOp = match->sliceConsumers.front();

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape.size() != outputShape.size() || sourceShape.isDynamic() || outputShape.isDynamic()) {
        return false;
    }

    return isBeneficialForPropagation(sourceShape, outputShape);
}

mlir::LogicalResult PropagateSliceThroughFullyConnected::matchAndRewrite(IE::FullyConnectedOp fcOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), fcOp->getName(), fcOp->getLoc());

    // Captured before this FC is erased; on success its producer FC (a stacked FullyConnected) is
    // re-enqueued so the chain propagates within the same greedy iteration.
    const auto fcInput = fcOp.getInput();

    if (mlir::succeeded(moveSliceToActivations(fcOp, rewriter)) || mlir::succeeded(splitToWeights(fcOp, rewriter))) {
        reenqueueProducerFullyConnected(rewriter, fcInput);
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult PropagateSliceThroughFullyConnected::moveSliceToActivations(IE::FullyConnectedOp fcOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    const auto fcOutShape = getShape(fcOp.getOutput());
    if (fcOutShape.size() != 2) {
        return mlir::failure();
    }

    // Batch-dim slice only applies to a single one-use chain FullyConnected -> [AffineReshape ->]
    // Slice; the multi-Slice channel fan-out is handled by splitToWeights instead.
    IE::SliceOp sliceOp;
    IE::AffineReshapeOp reshapeOp;
    if (!isBeneficialForPropagation(fcOp, sliceOp, reshapeOp)) {
        return mlir::failure();
    }

    auto staticOffsetsAttr = sliceOp.getStaticOffsetsAttr();
    auto staticSizesAttr = sliceOp.getStaticSizesAttr();
    if (!staticOffsetsAttr || !staticSizesAttr) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());

    auto finalOffsets = parseIntArrayAttr<int64_t>(staticOffsetsAttr);
    auto finalSizes = parseIntArrayAttr<int64_t>(staticSizesAttr);

    if (reshapeOp) {
        auto mapped = mapSliceToReshapeInput(getShape(reshapeOp.getInput()), sourceShape, outputShape, finalOffsets,
                                             finalSizes);
        if (mlir::failed(mapped)) {
            return mlir::failure();
        }
        finalOffsets = std::move(mapped->first);
        finalSizes = std::move(mapped->second);
    }

    if (finalOffsets.size() != 2 || finalSizes.size() != 2) {
        return mlir::failure();
    }
    // Only the batch dim (FC output dim 0) may be sliced here; the output-channel dim must be full.
    if (finalOffsets[1] != 0 || finalSizes[1] != fcOutShape[Dim(1)]) {
        return mlir::failure();
    }

    auto inputShape = getShape(fcOp.getInput());
    SmallVector<int64_t> inputOffsets(inputShape.size(), 0);
    SmallVector<int64_t> inputSizes(inputShape.raw());
    inputOffsets[0] = finalOffsets[0];
    inputSizes[0] = finalSizes[0];

    // Materialize the hoisted ops at the consumer Slice, so IR order matches slicing the
    // FC output in place (this pattern is anchored on the FC, not the Slice).
    rewriter.setInsertionPoint(sliceOp);

    auto inputSlice = IE::SliceOp::create(rewriter, takeOpLoc(sliceOp, "slice_input"), fcOp.getInput(),
                                          getIntArrayAttr(rewriter.getContext(), inputOffsets),
                                          getIntArrayAttr(rewriter.getContext(), inputSizes));

    auto newFC = IE::FullyConnectedOp::create(rewriter, takeOpLoc(sliceOp, "fc_sliced"), inputSlice.getResult(),
                                              fcOp.getWeights(), fcOp.getBias());

    mlir::Value result = newFC.getOutput();
    if (reshapeOp) {
        result = IE::AffineReshapeOp::create(rewriter, takeOpLoc(sliceOp, "reshape_sliced"), newFC.getOutput(),
                                             reshapeOp.getDimMappingAttr(),
                                             getIntArrayAttr(rewriter.getContext(), outputShape.raw()));
    }

    const auto sliceOpLoc = sliceOp->getLoc();

    rewriter.replaceOp(sliceOp, result);
    if (reshapeOp) {
        rewriter.eraseOp(reshapeOp);
    }
    rewriter.eraseOp(fcOp);

    _log.trace("[{0}] Moved Slice onto FC activations for '{1}'", getDebugName(), sliceOpLoc);
    return mlir::success();
}

//
// splitToWeights
//
// When FullyConnected's output is reshaped and then split by several IE.Slice consumers along the
// reshaped output-channel axis (e.g. splitting a fused QKV projection into separate Q/K/V branches),
// push the split to the weights/bias instead of the (wider) output. Each branch gets its own
// FullyConnected computing only the channels it needs, so the wide intermediate result never needs
// to be materialized.
//
// Pattern:
//   FullyConnected(input, weights, bias) -> Reshape/AffineReshape -> Slice (one per branch, along
//   the reshaped output-channel axis)
// becomes, per Slice:
//   Slice(weights) -> Slice(bias) -> FullyConnected(input, slicedWeights, slicedBias) ->
//   Reshape/AffineReshape
//
// Requires a bias operand; FullyConnected ops without bias are not matched.
//
mlir::LogicalResult PropagateSliceThroughFullyConnected::splitToWeights(IE::FullyConnectedOp fcOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), fcOp->getName(), fcOp->getLoc());

    // Channel-dim split requires the FC output to be reshaped before the Slices; the direct
    // FullyConnected -> Slice chain is the batch-dim case handled by moveSliceToActivations.
    auto match = matchFcReshapeSlices(fcOp);
    if (!match.has_value() || match->reshapeOp == nullptr) {
        return mlir::failure();
    }

    auto weights = fcOp.getWeights();
    auto bias = fcOp.getBias();
    // This rewrite currently matches only FullyConnected ops that have a bias operand.
    if (bias == nullptr) {
        return mlir::failure();
    }

    const auto weightsShape = getShape(weights);
    if (weightsShape.size() != 2 || weightsShape.isDynamic()) {
        return mlir::failure();
    }
    const auto biasShape = getShape(bias);
    if (biasShape.isDynamic() || biasShape.empty()) {
        return mlir::failure();
    }
    const int64_t biasAxis = biasShape.size() - 1;
    if (biasShape[Dim(biasAxis)] != weightsShape[Dim(0)]) {
        return mlir::failure();
    }

    // Bridges the flat 2D FullyConnected output to the multi-dim shape the Slices operate on.
    auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(match->reshapeOp);
    auto plainReshapeOp = mlir::dyn_cast<IE::ReshapeOp>(match->reshapeOp);
    const mlir::Value reshapeOutput = affineReshapeOp ? affineReshapeOp.getOutput() : plainReshapeOp.getOutput();
    mlir::Operation* const reshapeOpErase = match->reshapeOp;

    SmallVector<IE::SliceOp> sliceOps = std::move(match->sliceConsumers);
    for (auto sliceOp : sliceOps) {
        if (!sliceOp.getStaticOffsetsAttr() || !sliceOp.getStaticSizesAttr()) {
            return mlir::failure();
        }
    }

    const auto fcOutShape = getShape(fcOp.getOutput());
    if (fcOutShape.size() != 2 || fcOutShape.isDynamic()) {
        return mlir::failure();
    }

    const auto reshapeOutShape = getShape(reshapeOutput);
    if (reshapeOutShape.isDynamic()) {
        return mlir::failure();
    }

    struct SplitInfo {
        IE::SliceOp sliceOp;
        int64_t offset;
        int64_t size;
    };

    const auto rank = reshapeOutShape.size();
    std::optional<size_t> commonDiffAxis;

    // First pass: every slice must restrict the exact same axis, like a single VariadicSplit would.
    for (auto sliceOp : sliceOps) {
        const auto sliceOutShape = getShape(sliceOp.getResult());
        if (sliceOutShape.size() != rank) {
            return mlir::failure();
        }

        const auto diffAxis = getSingleDifferentDim(reshapeOutShape.raw(), sliceOutShape.raw());
        if (!diffAxis.has_value()) {
            return mlir::failure();
        }

        // Axis 0 is never the output-channel dim (splitting it is handled by
        // moveSliceToActivations instead); any other axis (interior or innermost) is a
        // candidate here, subject to the contiguity check in mapSliceToReshapeInput.
        if (diffAxis.value() == 0) {
            return mlir::failure();
        }
        if (!commonDiffAxis.has_value()) {
            commonDiffAxis = diffAxis.value();
        } else if (*commonDiffAxis != diffAxis.value()) {
            return mlir::failure();
        }
    }

    // Second pass: reuse the reshape/affine-reshape reconstruction to translate each slice from the
    // reshaped output-channel axis onto the flat FullyConnected output axis (dim 1).
    SmallVector<SplitInfo> splits;
    splits.reserve(sliceOps.size());
    std::optional<int64_t> expectedSplitLen;
    for (auto sliceOp : sliceOps) {
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
        const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());
        const auto sliceOutShape = getShape(sliceOp.getResult());

        auto mapped = mapSliceToReshapeInput(fcOutShape, reshapeOutShape, sliceOutShape, sliceOffsets, sliceSizes);
        if (mlir::failed(mapped)) {
            return mlir::failure();
        }
        const auto& flatOffsets = mapped->first;
        const auto& flatSizes = mapped->second;

        // The split must leave the batch dim (FC output dim 0) untouched; only the output-channel
        // dim may be restricted here (a batch-dim slice is handled by moveSliceToActivations).
        if (flatOffsets[0] != 0 || flatSizes[0] != fcOutShape[Dim(0)]) {
            return mlir::failure();
        }

        // Mirror VariadicSplit semantics: every branch must cover an equal-length span of the axis.
        const auto splitLen = flatSizes[1];
        if (!expectedSplitLen.has_value()) {
            expectedSplitLen = splitLen;
        } else if (*expectedSplitLen != splitLen) {
            return mlir::failure();
        }

        splits.push_back({sliceOp, flatOffsets[1], flatSizes[1]});
    }

    const auto fcLoc = fcOp->getLoc();
    // Tag with the enclosing function name so locations can't collide even if two matched
    // FullyConnectedOps in sibling outlined functions share the same original location.
    // Deterministic (unlike a process-wide counter), so IR output doesn't depend on pass scheduling.
    auto parentFunc = fcOp->getParentOfType<mlir::func::FuncOp>();
    const auto splitId = parentFunc ? parentFunc.getSymName() : mlir::StringRef("");
    for (auto [idx, split] : llvm::enumerate(splits)) {
        const SmallVector<int64_t> weightsOffsets{split.offset, 0};
        const SmallVector<int64_t> weightsSizes{split.size, weightsShape[Dim(1)]};
        auto newWeights = IE::SliceOp::create(rewriter, takeOpLoc(fcOp, "split{0}_W_{1}", splitId, idx), weights,
                                              getIntArrayAttr(rewriter.getContext(), weightsOffsets),
                                              getIntArrayAttr(rewriter.getContext(), weightsSizes));

        SmallVector<int64_t> biasOffsets(biasShape.size(), 0);
        SmallVector<int64_t> biasSizes(biasShape.raw().begin(), biasShape.raw().end());
        biasOffsets[biasAxis] = split.offset;
        biasSizes[biasAxis] = split.size;
        auto newBias = IE::SliceOp::create(rewriter, takeOpLoc(fcOp, "split{0}_bias_{1}", splitId, idx), bias,
                                           getIntArrayAttr(rewriter.getContext(), biasOffsets),
                                           getIntArrayAttr(rewriter.getContext(), biasSizes));

        auto newFc = IE::FullyConnectedOp::create(rewriter, appendLoc(fcLoc, "split{0}_qkv_{1}", splitId, idx),
                                                  fcOp.getInput(), newWeights.getResult(), newBias.getResult());

        const auto sliceOutShapeAttr =
                getIntArrayAttr(rewriter.getContext(), getShape(split.sliceOp.getResult()).raw());
        mlir::Value newReshapeOut;
        if (affineReshapeOp) {
            newReshapeOut = IE::AffineReshapeOp::create(rewriter, split.sliceOp->getLoc(), newFc.getOutput(),
                                                        affineReshapeOp.getDimMappingAttr(), sliceOutShapeAttr)
                                    .getOutput();
        } else {
            newReshapeOut =
                    IE::ReshapeOp::create(rewriter, split.sliceOp->getLoc(), newFc.getOutput(), sliceOutShapeAttr)
                            .getOutput();
        }

        rewriter.replaceOp(split.sliceOp, newReshapeOut);
    }

    rewriter.eraseOp(reshapeOpErase);
    rewriter.eraseOp(fcOp);

    _log.trace("[{0}] Rewrite succeeded for '{1}'", getDebugName(), fcLoc);
    return mlir::success();
}

//
// MoveSliceBeforeRMS
//
// Hoist a Slice before an RMS normalization when the slice does not touch the
// normalization (last) dimension. RMS normalizes each row across the last axis, so
// slicing any other axis commutes with the normalization.
//
// Pattern:
//   RMS(input, gamma) -> Slice(out)
// becomes
//   Slice(input) -> input_sliced
//   RMS(input_sliced, gamma) -> out

class MoveSliceBeforeRMS final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeRMS(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("MoveSliceBeforeRMS");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveSliceBeforeRMS::matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto rmsOp = sliceOp.getSource().getDefiningOp<IE::RMSOp>();
    if (rmsOp == nullptr || !rmsOp->hasOneUse()) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    // Keep only profitable rewrites
    if (!isBeneficialForPropagation(sourceShape, outputShape)) {
        return mlir::failure();
    }

    // The normalization dimension (last dim) must not be sliced.
    const auto rmsInputShape = getShape(rmsOp.getInput());
    if (rmsInputShape.size() == 0) {
        return mlir::failure();
    }
    const auto normDim = rmsInputShape.size() - 1;
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(staticOffsets);
    const auto sliceSizes = parseIntArrayAttr<int64_t>(staticSizes);
    if (sliceOffsets[normDim] != 0 || sliceSizes[normDim] != rmsInputShape[Dim(normDim)]) {
        return mlir::failure();
    }

    auto inputSlice = IE::SliceOp::create(rewriter, takeOpLoc(sliceOp, "slice_input"), rmsOp.getInput(), staticOffsets,
                                          staticSizes);
    auto newRMS = IE::RMSOp::create(rewriter, takeOpLoc(sliceOp, "rms_sliced"), inputSlice.getResult(),
                                    rmsOp.getGamma(), rmsOp.getEpsAttr(), rmsOp.getEpsModeAttr());

    const auto sliceLoc = sliceOp->getLoc();
    rewriter.replaceOp(sliceOp, newRMS.getResult());
    _log.trace("[{0}] Rewrite succeeded for '{1}'", getDebugName(), sliceLoc);
    return mlir::success();
}

//
// MoveSliceBeforeAffineReshape
//
// Hoist a Slice before a standalone AffineReshape (one whose input is not a
// FullyConnected, which is already handled by PropagateSliceThroughFullyConnected). The
// slice is remapped from the reshape output space to the reshape input space, and is
// rejected when the sliced axis is split or merged by the reshape.
//
// Pattern:
//   AffineReshape(input) -> Slice(out)
// becomes
//   Slice(input) -> input_sliced
//   AffineReshape(input_sliced) -> out

class MoveSliceBeforeAffineReshape final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeAffineReshape(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("MoveSliceBeforeAffineReshape");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveSliceBeforeAffineReshape::matchAndRewrite(IE::SliceOp sliceOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto reshapeOp = sliceOp.getSource().getDefiningOp<IE::AffineReshapeOp>();
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
        return mlir::failure();
    }

    // FullyConnected -> AffineReshape -> Slice is handled by PropagateSliceThroughFullyConnected.
    // Re-enqueue that FC so it is revisited this iteration instead of the next full walk.
    if (reenqueueProducerFullyConnected(rewriter, reshapeOp.getInput())) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    // Keep only profitable rewrites
    if (!isBeneficialForPropagation(sourceShape, outputShape)) {
        return mlir::failure();
    }

    const auto sliceOffsets = parseIntArrayAttr<int64_t>(staticOffsets);
    const auto sliceSizes = parseIntArrayAttr<int64_t>(staticSizes);
    const auto reshapeInShape = getShape(reshapeOp.getInput());
    const auto reshapeOutShape = getShape(reshapeOp.getResult());
    const auto dimMappingList = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());

    // Collect reshape output dims where slicing is non-trivial.
    mlir::DenseSet<int64_t> modifiedAxes;
    for (size_t i = 0; i < sliceOffsets.size(); ++i) {
        if (sliceOffsets[i] != 0 || sliceSizes[i] != reshapeOutShape[Dim(i)]) {
            modifiedAxes.insert(static_cast<int64_t>(i));
        }
    }
    if (modifiedAxes.empty()) {
        return mlir::failure();
    }

    // Reject if any sliced axis is split or merged by the AffineReshape.
    if (IE::areModifiedAxesSplitOrMerged(dimMappingList, reshapeInShape, reshapeOutShape, modifiedAxes,
                                         /*swapOrder=*/false, _log.nest())) {
        return mlir::failure();
    }

    const auto invertedDimMapping =
            IE::invertDimMappingWithAxesNotSplitOrMerged(dimMappingList, reshapeInShape, reshapeOutShape);
    SmallVector<int64_t> inOffsets(reshapeInShape.size(), 0);
    SmallVector<int64_t> inSizes(reshapeInShape.raw().begin(), reshapeInShape.raw().end());
    for (auto outDim : modifiedAxes) {
        const auto inDim = invertedDimMapping[outDim];
        inOffsets[inDim] = sliceOffsets[outDim];
        inSizes[inDim] = sliceSizes[outDim];
    }

    const auto ctx = rewriter.getContext();
    const auto sliceOpLoc = sliceOp->getLoc();
    auto inputSlice = IE::SliceOp::create(rewriter, takeOpLoc(sliceOp, "slice_input"), reshapeOp.getInput(),
                                          getIntArrayAttr(ctx, inOffsets), getIntArrayAttr(ctx, inSizes));
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(sliceOp, inputSlice.getResult(), reshapeOp.getDimMappingAttr(),
                                                     getIntArrayAttr(ctx, outputShape.raw()));

    _log.trace("[{0}] Rewrite succeeded for '{1}'", getDebugName(), sliceOpLoc);
    return mlir::success();
}

//
// SwapSlices
//
// Commute two nested slices that restrict orthogonal (disjoint) sets of axes, pushing
// the lower-indexed axis toward the producer. This is what allows a sequence-dim slice
// to move past a channel-dim split so that sibling channel splits can be deduplicated.
//
// Pattern:
//   Slice_inner(x)   restricts axis set I
//   Slice_outer(...) restricts axis set O   (O and I disjoint)
// becomes
//   Slice_low(x)     restricts min-axis slice on x
//   Slice_high(...)  restricts the remaining slice
//
// The rewrite fires only when the outer slice restricts a strictly lower minimum axis
// than the inner slice, which keeps the canonical form stable and guarantees termination.

SmallVector<int64_t> getRestrictedAxes(mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
                                       ShapeRef inputShape) {
    SmallVector<int64_t> axes;
    for (size_t d = 0; d < offsets.size(); ++d) {
        if (offsets[d] != 0 || sizes[d] != inputShape[Dim(d)]) {
            axes.push_back(static_cast<int64_t>(d));
        }
    }
    return axes;
}

class SwapSlices final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    SwapSlices(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("SwapSlices");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapSlices::matchAndRewrite(IE::SliceOp outerSlice, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), outerSlice->getLoc());

    auto innerSlice = outerSlice.getSource().getDefiningOp<IE::SliceOp>();
    if (innerSlice == nullptr || !innerSlice->hasOneUse()) {
        return mlir::failure();
    }

    const auto rootValue = innerSlice.getSource();
    const auto rootShape = getShape(rootValue);
    const auto innerShape = getShape(innerSlice.getResult());

    auto innerOffsetsAttr = innerSlice.getStaticOffsetsAttr();
    auto innerSizesAttr = innerSlice.getStaticSizesAttr();
    auto outerOffsetsAttr = outerSlice.getStaticOffsetsAttr();
    auto outerSizesAttr = outerSlice.getStaticSizesAttr();
    if (!innerOffsetsAttr || !innerSizesAttr || !outerOffsetsAttr || !outerSizesAttr) {
        return mlir::failure();
    }

    const auto innerOffsets = parseIntArrayAttr<int64_t>(innerOffsetsAttr);
    const auto innerSizes = parseIntArrayAttr<int64_t>(innerSizesAttr);
    const auto outerOffsets = parseIntArrayAttr<int64_t>(outerOffsetsAttr);
    const auto outerSizes = parseIntArrayAttr<int64_t>(outerSizesAttr);

    if (rootShape.size() != innerShape.size()) {
        return mlir::failure();
    }

    const auto innerAxes = getRestrictedAxes(innerOffsets, innerSizes, rootShape);
    const auto outerAxes = getRestrictedAxes(outerOffsets, outerSizes, innerShape);
    if (innerAxes.empty() || outerAxes.empty()) {
        return mlir::failure();
    }

    // The two slices must restrict disjoint axis sets to be able to commute.
    for (auto axis : outerAxes) {
        if (llvm::is_contained(innerAxes, axis)) {
            return mlir::failure();
        }
    }

    // Canonical direction: only push the lower-indexed axis toward the producer.
    if (outerAxes.front() >= innerAxes.front()) {
        return mlir::failure();
    }

    // Low slice: restrict the outer slice's axes directly on the producer value.
    SmallVector<int64_t> lowOffsets(rootShape.size(), 0);
    SmallVector<int64_t> lowSizes(rootShape.raw().begin(), rootShape.raw().end());
    for (auto axis : outerAxes) {
        lowOffsets[axis] = outerOffsets[axis];
        lowSizes[axis] = outerSizes[axis];
    }

    // High slice: apply the inner slice's axes on top of the low slice.
    SmallVector<int64_t> highOffsets(rootShape.size(), 0);
    SmallVector<int64_t> highSizes(lowSizes);
    for (auto axis : innerAxes) {
        highOffsets[axis] = innerOffsets[axis];
        highSizes[axis] = innerSizes[axis];
    }

    const auto ctx = rewriter.getContext();
    const auto outerSliceLoc = outerSlice->getLoc();
    auto lowSlice = IE::SliceOp::create(rewriter, takeOpLoc(outerSlice, "slice_low"), rootValue,
                                        getIntArrayAttr(ctx, lowOffsets), getIntArrayAttr(ctx, lowSizes));
    rewriter.replaceOpWithNewOp<IE::SliceOp>(outerSlice, lowSlice.getResult(), getIntArrayAttr(ctx, highOffsets),
                                             getIntArrayAttr(ctx, highSizes));

    _log.trace("[{0}] Rewrite succeeded for '{1}'", getDebugName(), outerSliceLoc);
    return mlir::success();
}

//
// MergeHorizontalSlices
//
// Deduplicate sibling slices that read the same value with identical offsets and sizes.
// This lets slices produced independently along different branches (for example the two
// halves of a gate/up split, or the two consumers of a residual add) collapse into a
// single slice so downstream hoisting patterns can proceed.
//
// Restricted to a producer whose only two users are both slices, matching the LLM
// prefill-tail graph. Any other fan-out (more than two users, or a non-slice user, for
// example grouped-convolution groups feeding QKV projections) is left untouched to avoid
// downstream-pass regressions on the merged form.

class MergeHorizontalSlices final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MergeHorizontalSlices(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("MergeHorizontalSlices");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeHorizontalSlices::matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    SmallVector<mlir::Operation*> users(sliceOp.getSource().getUsers().begin(), sliceOp.getSource().getUsers().end());

    if (users.size() != 2 || !llvm::all_of(users, [](mlir::Operation* user) {
            return mlir::isa<IE::SliceOp>(user);
        })) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    for (auto* user : users) {
        auto sibling = mlir::dyn_cast<IE::SliceOp>(user);
        if (sibling == nullptr || sibling == sliceOp) {
            continue;
        }
        if (sibling.getStaticOffsetsAttr() != staticOffsets || sibling.getStaticSizesAttr() != staticSizes) {
            continue;
        }

        if (sibling->getBlock() != sliceOp->getBlock() || sibling->isBeforeInBlock(sliceOp)) {
            continue;
        }
        const auto siblingLoc = sibling->getLoc();
        rewriter.replaceOp(sibling, sliceOp.getResult());
        _log.trace("[{0}] Merged duplicate slice at '{1}'", getDebugName(), siblingLoc);
        return mlir::success();
    }

    return mlir::failure();
}

//
// SwapOperationsWithGatherAndSlicePass
//

class SwapOperationsWithGatherAndSlicePass final :
        public IE::impl::SwapOperationsWithGatherAndSliceBase<SwapOperationsWithGatherAndSlicePass> {
public:
    explicit SwapOperationsWithGatherAndSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SwapOperationsWithGatherAndSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveTwoInputsEltwiseOpAfterGather<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveTwoInputsEltwiseOpAfterGather<IE::SubtractOp>>(&ctx, _log);
    patterns.add<MoveTwoInputsEltwiseOpAfterGather<IE::DynamicDequantizeOp>>(&ctx, _log);
    patterns.add<MoveConvertAfterGather>(&ctx, _log);
    patterns.add<MoveConvertBeforeGather>(&ctx, _log);
    patterns.add<MoveSliceBeforeEltwise<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveSliceBeforeEltwise<IE::AddOp>>(&ctx, _log);
    patterns.add<MoveSliceBeforeActivation<IE::SwishOp>>(&ctx, _log);
    patterns.add<MoveSliceBeforeActivation<IE::GeluOp>>(&ctx, _log);
    patterns.add<MoveSliceBeforeFakeQuantize>(&ctx, _log);
    patterns.add<PropagateSliceThroughFullyConnected>(&ctx, _log);
    patterns.add<MoveSliceBeforeRMS>(&ctx, _log);
    patterns.add<MoveSliceBeforeAffineReshape>(&ctx, _log);
    patterns.add<SwapSlices>(&ctx, _log);
    patterns.add<MergeHorizontalSlices>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapOperationsWithGatherAndSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapOperationsWithGatherAndSlicePass(Logger log) {
    return std::make_unique<SwapOperationsWithGatherAndSlicePass>(log);
}
