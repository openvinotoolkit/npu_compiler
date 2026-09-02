//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
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
    return rewriter.create<IE::GatherOp>(appendLoc(loc, "gather"), input, gatherOp.getIndices(),
                                         gatherOp.getAxisValue(), gatherOp.getBatchDims(),
                                         gatherOp.getIndicesRankAttr());
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

    auto newGather = rewriter.create<IE::GatherOp>(gatherOpLoc, convertOp.getInput(), gatherOp.getIndices(),
                                                   gatherOp.getAxisValue(), gatherOp.getBatchDims(),
                                                   gatherOp.getIndicesRankAttr());
    auto newConvert =
            rewriter.create<IE::ConvertOp>(convertOp->getLoc(), newGather.getOutput(), convertOp.getDstElemType());

    rewriter.replaceOp(gatherOp, newConvert.getOutput());

    _log.trace("Successfully replaced '{0}' at '{1}'", gatherOpName, gatherOpLoc);

    return mlir::success();
}

bool isBeneficialForPropagation(ShapeRef sourceShape, ShapeRef outputShape) {
    return outputShape.totalSize() * 2 < sourceShape.totalSize();
}

bool isSingleElement(mlir::Value value) {
    return getShape(value).totalSize() == 1;
}

//
// MoveSliceBeforeMultiply
//

// If a slice only trims the result of a Multiply, attempt to hoist that slice to the
// Multiply's inputs so the multiplication is performed on smaller operands and less
// computation is required.
//
// Pattern (conceptual):
//   Multiply(InputA, InputB) -> Slice(out)
// becomes
//   Slice(InputA) -> SliceA
//   Slice(InputB) -> SliceB
//   Multiply(SliceA, SliceB) -> out
//
// Apply this rewrite only when:
// - the output slice actually reduces total element count, and
// - each operand can be safely sliced in the same way.

class MoveSliceBeforeMultiply final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeMultiply(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("MoveSliceBeforeMultiply");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveSliceBeforeMultiply::matchAndRewrite(IE::SliceOp sliceOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto multiplyOp = sliceOp.getSource().getDefiningOp<IE::MultiplyOp>();
    if (multiplyOp == nullptr) {
        return mlir::failure();
    }

    if (!multiplyOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    if (multiplyOp.getInputPaddingAttr() != nullptr || multiplyOp.getOutputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    auto staticOffsets = sliceOp.getStaticOffsetsAttr();
    auto staticSizes = sliceOp.getStaticSizesAttr();
    if (!staticOffsets || !staticSizes) {
        return mlir::failure();
    }

    const auto sourceShape = getShape(sliceOp.getSource());
    const auto outputShape = getShape(sliceOp.getResult());
    if (sourceShape != getShape(multiplyOp.getOutput()) || sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    // Keep only profitable rewrites
    if (!isBeneficialForPropagation(sourceShape, outputShape)) {
        return mlir::failure();
    }

    const auto mulOperands = multiplyOp.getOperands();
    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(mulOperands.size());
    for (auto operand : mulOperands) {
        if (isSingleElement(operand)) {
            newInputs.push_back(operand);
            continue;
        }

        if (getShape(operand) != sourceShape) {
            return mlir::failure();
        }

        newInputs.push_back(nullptr);
    }

    for (auto [operandIdx, operand] : llvm::enumerate(mulOperands)) {
        if (newInputs[operandIdx] != nullptr) {
            continue;
        }

        auto newSlice = rewriter.create<IE::SliceOp>(takeOpLoc(sliceOp, "slice_{0}", operandIdx), operand,
                                                     staticOffsets, staticSizes);
        newInputs[operandIdx] = newSlice.getResult();
    }

    mlir::IRMapping mapper;
    mapper.map(multiplyOp.getOperands(), newInputs);
    auto* newMultiply = rewriter.clone(*multiplyOp.getOperation(), mapper);
    newMultiply->setLoc(takeOpLoc(sliceOp, "as_multiply"));
    newMultiply->getResult(0).setType(sliceOp.getResult().getType());

    rewriter.replaceOp(sliceOp, newMultiply->getResults());
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

    auto newSlice = rewriter.create<IE::SliceOp>(takeOpLoc(sliceOp, "slice_input"),
                                                 actOp->getOperand(0),  // first operand is the data input
                                                 staticOffsets, staticSizes);

    auto newResult = createSlicedActivation(rewriter, takeOpLoc(sliceOp, "act_sliced"), actOp.getOperation(),
                                            newSlice.getResult());

    rewriter.replaceOp(sliceOp, newResult);
    return mlir::success();
}

//
// MoveSliceBeforeFullyConnected
//
// Hoist Slice before FullyConnected, optionally through one AffineReshape.
// This reduces the amount of computation performed by the matrix multiply.
//
// Supported patterns:
//   1) FullyConnected -> Slice
//   2) FullyConnected -> AffineReshape -> Slice
//
// In both cases the Slice is pushed to the FC input (batch dim only).
//

class MoveSliceBeforeFullyConnected final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveSliceBeforeFullyConnected(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("MoveSliceBeforeFullyConnected");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
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

mlir::FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> mapSliceToAffineReshapeInput(
        IE::AffineReshapeOp affineReshapeOp, ShapeRef sourceShape, ShapeRef outputShape,
        mlir::ArrayRef<int64_t> sliceOffsets, mlir::ArrayRef<int64_t> sliceSizes) {
    if (sourceShape.size() != outputShape.size()) {
        return mlir::failure();
    }

    const auto sliceAxis = getSingleDifferentDim(sourceShape.raw(), outputShape.raw());
    if (!sliceAxis.has_value()) {
        return mlir::failure();
    }

    const auto inputShape = getShape(affineReshapeOp.getInput());
    std::optional<size_t> inputSliceAxis;

    if (sourceShape.size() == inputShape.size() + 1 && sourceShape.front() == 1 && sliceAxis.value() > 0) {
        inputSliceAxis = sliceAxis.value() - 1;
    } else {
        auto reassociationMap = IE::getReassociationMapExtension(sourceShape.raw(), inputShape.raw());
        if (mlir::succeeded(reassociationMap)) {
            auto mappedDims = reassociationMap.value()[sliceAxis.value()];
            if (mappedDims.size() == 1) {
                inputSliceAxis = mappedDims.front();
            }
        }
    }

    if (!inputSliceAxis.has_value()) {
        return mlir::failure();
    }

    SmallVector<int64_t> newOffsets(inputShape.size(), 0);
    SmallVector<int64_t> newSizes(inputShape.raw());
    newOffsets[inputSliceAxis.value()] = sliceOffsets[sliceAxis.value()];
    newSizes[inputSliceAxis.value()] = sliceSizes[sliceAxis.value()];

    return std::make_pair(newOffsets, newSizes);
}

mlir::LogicalResult MoveSliceBeforeFullyConnected::matchAndRewrite(IE::SliceOp sliceOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got slice at '{1}'", getDebugName(), sliceOp->getLoc());

    auto staticOffsetsAttr = sliceOp.getStaticOffsetsAttr();
    auto staticSizesAttr = sliceOp.getStaticSizesAttr();
    if (!staticOffsetsAttr || !staticSizesAttr) {
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

    auto finalOffsets = parseIntArrayAttr<int64_t>(staticOffsetsAttr);
    auto finalSizes = parseIntArrayAttr<int64_t>(staticSizesAttr);

    IE::FullyConnectedOp fcOp = nullptr;
    IE::AffineReshapeOp reshapeOp = nullptr;

    auto srcOp = sliceOp.getSource().getDefiningOp();

    if (auto directFC = mlir::dyn_cast_if_present<IE::FullyConnectedOp>(srcOp)) {
        fcOp = directFC;
    } else if (auto resh = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(srcOp)) {
        if (!resh.getOutput().hasOneUse()) {
            return mlir::failure();
        }

        auto mapped = mapSliceToAffineReshapeInput(resh, sourceShape, outputShape, finalOffsets, finalSizes);
        if (mlir::failed(mapped)) {
            return mlir::failure();
        }
        finalOffsets = std::move(mapped->first);
        finalSizes = std::move(mapped->second);

        auto reshInput = resh.getInput();
        fcOp = reshInput.getDefiningOp<IE::FullyConnectedOp>();
        if (!fcOp) {
            return mlir::failure();
        }
        reshapeOp = resh;
    } else {
        return mlir::failure();
    }

    if (!fcOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    const auto fcOutShape = getShape(fcOp.getOutput());
    if (fcOutShape.size() != 2) {
        return mlir::failure();
    }

    if (sourceShape.isDynamic() || outputShape.isDynamic()) {
        return mlir::failure();
    }

    if (finalOffsets.size() != 2 || finalSizes.size() != 2) {
        return mlir::failure();
    }
    if (finalOffsets[1] != 0 || finalSizes[1] != fcOutShape[Dim(1)]) {
        return mlir::failure();
    }

    auto inputShape = getShape(fcOp.getInput());
    SmallVector<int64_t> inputOffsets(inputShape.size(), 0);
    SmallVector<int64_t> inputSizes(inputShape.raw());
    inputOffsets[0] = finalOffsets[0];
    inputSizes[0] = finalSizes[0];

    auto inputSlice = rewriter.create<IE::SliceOp>(takeOpLoc(sliceOp, "slice_input"), fcOp.getInput(),
                                                   getIntArrayAttr(rewriter.getContext(), inputOffsets),
                                                   getIntArrayAttr(rewriter.getContext(), inputSizes));

    auto newFC = rewriter.create<IE::FullyConnectedOp>(takeOpLoc(sliceOp, "fc_sliced"), inputSlice.getResult(),
                                                       fcOp.getWeights(), fcOp.getBias());

    mlir::Value result = newFC.getOutput();
    if (reshapeOp) {
        result = rewriter.create<IE::AffineReshapeOp>(takeOpLoc(sliceOp, "reshape_sliced"), newFC.getOutput(),
                                                      reshapeOp.getDimMappingAttr(),
                                                      getIntArrayAttr(rewriter.getContext(), outputShape.raw()));
    }

    rewriter.replaceOp(sliceOp, result);
    if (reshapeOp) {
        rewriter.eraseOp(reshapeOp);
    }
    rewriter.eraseOp(fcOp);

    _log.trace("[{0}] Rewrite succeeded for '{1}'", getDebugName(), sliceOp->getLoc());
    return mlir::success();
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
    patterns.add<MoveSliceBeforeMultiply>(&ctx, _log);
    patterns.add<MoveSliceBeforeActivation<IE::SwishOp>>(&ctx, _log);
    patterns.add<MoveSliceBeforeFullyConnected>(&ctx, _log);

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
