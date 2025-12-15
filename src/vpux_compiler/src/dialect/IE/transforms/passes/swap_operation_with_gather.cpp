//
// Copyright (C) 2024-2025 Intel Corporation.
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
    auto isGatherAxisBroadcasted = [outputShape, this](mlir::Value operand) {
        auto inputShape = getShape(operand);
        auto broadCastAxes = IE::getDiffInOutSizeDims(inputShape, outputShape);
        for (auto axis : broadCastAxes) {
            if (axis == SUPPORTED_GATHER_AXIS) {
                return true;
            }
        }
        return false;
    };
    if (llvm::any_of(op->getOperands(), isGatherAxisBroadcasted)) {
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
    auto newLoc = appendLoc(gatherLoc, "new_lhs");
    auto newGather1 = createGatherOp(rewriter, newLoc, op->getOperand(0), gatherOp);
    newLoc = appendLoc(gatherLoc, "new_rhs");
    auto newGather2 = createGatherOp(rewriter, gatherLoc, op->getOperand(1), gatherOp);

    mlir::IRMapping opMapper;
    opMapper.map(op->getOperand(0), newGather1);
    opMapper.map(op->getOperand(1), newGather2);
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
    _log.trace("Got '{0}' at '{1}'", gatherOp->getName(), gatherOp->getLoc());

    auto convertOp = gatherOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr || !convertOp->hasOneUse()) {
        return mlir::failure();
    }

    if (!isBeneficialToConvert(convertOp, gatherOp)) {
        return matchFailed(_log.nest(), rewriter, gatherOp, "Not beneficial to move operation after GatherOp");
    }

    auto newGather = rewriter.create<IE::GatherOp>(gatherOp->getLoc(), convertOp.getInput(), gatherOp.getIndices(),
                                                   gatherOp.getAxis(), gatherOp.getAxisValueAttr(),
                                                   gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
    auto newConvert =
            rewriter.create<IE::ConvertOp>(convertOp->getLoc(), newGather.getOutput(), convertOp.getDstElemType());

    rewriter.replaceOp(gatherOp, newConvert.getOutput());

    _log.trace("Successfully replaced '{0}' at '{1}'", gatherOp->getName(), gatherOp->getLoc());

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
