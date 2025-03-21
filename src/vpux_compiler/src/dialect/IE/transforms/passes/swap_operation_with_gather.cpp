//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
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
// MoveEltwiseAfterGather
//

template <class ConcreteOp>
class MoveEltwiseAfterGather final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    MoveEltwiseAfterGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("MoveEltwiseAfterGather");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) const;
    std::optional<ConcreteOp> getSupportedEltwiseOp(IE::GatherOp gatherOp) const;
    const Dim SUPPORTED_GATHER_AXIS = Dim(0);

    Logger _log;
};

template <class ConcreteOp>
bool MoveEltwiseAfterGather<ConcreteOp>::isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) const {
    return inShape.totalSize() > outShape.totalSize();
}

template <class ConcreteOp>
std::optional<ConcreteOp> MoveEltwiseAfterGather<ConcreteOp>::getSupportedEltwiseOp(IE::GatherOp gatherOp) const {
    if (gatherOp.getAxis() != nullptr) {
        _log.trace("Does not support the case where GatherOp axis constant has not been converted into an attribute");
        return std::nullopt;
    }

    if (gatherOp.getAxisValueAttr() != nullptr && gatherOp.getAxisValue().value() != SUPPORTED_GATHER_AXIS.ind()) {
        _log.trace("Only support GatherOp with axis on the first dim");
        return std::nullopt;
    }

    auto eltwiseOp = gatherOp.getInput().getDefiningOp<ConcreteOp>();
    if (eltwiseOp == nullptr || !eltwiseOp->hasOneUse()) {
        return std::nullopt;
    }

    if (eltwiseOp.getPostOpAttr() != nullptr || eltwiseOp.getClampAttr() != nullptr ||
        eltwiseOp.getOutputChannelsAttr() != nullptr || eltwiseOp.getInputChannelsAttr() != nullptr) {
        _log.trace("Eltwise operation is not supported");
        return std::nullopt;
    }

    auto outputShape = getShape(eltwiseOp->getResult(0));
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
    if (llvm::any_of(eltwiseOp->getOperands(), isGatherAxisBroadcasted)) {
        _log.trace(
                "Cannot swap Eltwise operation with GatherOp due to a conflict between broadcast axis and gather axis");
        return std::nullopt;
    }

    return eltwiseOp;
}

template <class ConcreteOp>
mlir::LogicalResult MoveEltwiseAfterGather<ConcreteOp>::matchAndRewrite(IE::GatherOp gatherOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", gatherOp->getName(), gatherOp->getLoc());

    // Conversion is benificial when GatherOp is reducing tensor size.
    if (!isBeneficialToConvert(getShape(gatherOp.getInput()), getShape(gatherOp.getOutput()))) {
        return matchFailed(_log.nest(), rewriter, gatherOp, "Not beneficial to move operation after GatherOp");
    }

    auto getEltwiseOp = getSupportedEltwiseOp(gatherOp);
    if (!getEltwiseOp.has_value()) {
        return mlir::failure();
    }
    auto eltwiseOp = getEltwiseOp.value();

    auto gatherLoc = gatherOp->getLoc();

    auto newGather1 = rewriter.create<IE::GatherOp>(
            appendLoc(gatherLoc, "gather_1"), eltwiseOp.getInput1(), gatherOp.getIndices(), gatherOp.getAxis(),
            gatherOp.getAxisValueAttr(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());

    auto newGather2 = rewriter.create<IE::GatherOp>(
            appendLoc(gatherLoc, "gather_2"), eltwiseOp.getInput2(), gatherOp.getIndices(), gatherOp.getAxis(),
            gatherOp.getAxisValueAttr(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());

    mlir::IRMapping eltwiseMapper;
    eltwiseMapper.map(eltwiseOp->getOperand(0), newGather1.getOutput());
    eltwiseMapper.map(eltwiseOp->getOperand(1), newGather2.getOutput());
    auto newEltwiseOp = rewriter.clone(*eltwiseOp, eltwiseMapper);

    vpux::inferReturnTypes(newEltwiseOp, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(gatherOp, newEltwiseOp->getResult(0));

    _log.trace("Successfully replaced '{0}' at '{1}'", gatherOp->getName(), gatherLoc);

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
    patterns.add<MoveEltwiseAfterGather<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveEltwiseAfterGather<IE::SubtractOp>>(&ctx, _log);
    patterns.add<MoveConvertAfterGather>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
