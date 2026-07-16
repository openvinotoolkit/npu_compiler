//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPOPERATIONWITHGATHER
#define GEN_PASS_DEF_SWAPOPERATIONWITHGATHER
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
    if (gatherOp.getAxis() != nullptr) {
        _log.trace("Does not support the case where GatherOp axis constant has not been converted into an attribute");
        return std::nullopt;
    }

    if (gatherOp.getAxisValueAttr() != nullptr && gatherOp.getAxisValue().value() != SUPPORTED_GATHER_AXIS.ind()) {
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
    return rewriter.create<IE::GatherOp>(appendLoc(loc, "gather"), input, gatherOp.getIndices(), gatherOp.getAxis(),
                                         gatherOp.getAxisValueAttr(), gatherOp.getBatchDims(),
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

    return convertIORatio < 1.0f && gatherIORatio > 1.0f;
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
                                                   gatherOp.getAxis(), gatherOp.getAxisValueAttr(),
                                                   gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
    auto newConvert =
            rewriter.create<IE::ConvertOp>(convertOp->getLoc(), newGather.getOutput(), convertOp.getDstElemType());

    rewriter.replaceOp(gatherOp, newConvert.getOutput());

    _log.trace("Successfully replaced '{0}' at '{1}'", gatherOpName, gatherOpLoc);

    return mlir::success();
}

//
// SwapOperationWithGatherPass
//

class SwapOperationWithGatherPass final : public IE::impl::SwapOperationWithGatherBase<SwapOperationWithGatherPass> {
public:
    explicit SwapOperationWithGatherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SwapOperationWithGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveTwoInputsEltwiseOpAfterGather<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveTwoInputsEltwiseOpAfterGather<IE::SubtractOp>>(&ctx, _log);
    patterns.add<MoveTwoInputsEltwiseOpAfterGather<IE::DynamicDequantizeOp>>(&ctx, _log);
    patterns.add<MoveConvertAfterGather>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapOperationWithGatherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapOperationWithGatherPass(Logger log) {
    return std::make_unique<SwapOperationWithGatherPass>(log);
}
