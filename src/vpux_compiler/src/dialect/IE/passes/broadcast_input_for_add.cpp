//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vpux/compiler/utils/rewriter.hpp>

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/strings.hpp"

using namespace vpux;

namespace {

Const::DeclareOp createShapeConstForBroadCast(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                              mlir::Location loc, ShapeRef shape) {
    SmallVector<int64_t> constShape = {4};
    auto intType = getSInt64Type(ctx);
    const auto shapeStorageType = mlir::RankedTensorType::get(constShape, intType);
    const auto shapeDenseAttr = mlir::DenseElementsAttr::get(shapeStorageType, shape.raw());
    auto newContentAttr = Const::ContentAttr::get(shapeDenseAttr).convertElemType(getSInt32Type(ctx));
    return rewriter.create<Const::DeclareOp>(loc, shapeStorageType, newContentAttr);
}

//
// BroadcastInputRewriter
//

class BroadcastInputRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    BroadcastInputRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("BroadcastInputRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isAddQuantized(mlir::Value input) const;
    void createBroadcastBeforeAddOp(IE::AddOp origOp, mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                    mlir::Value broadcastInput, mlir::Value origInput, ShapeRef target_shape) const;
    Logger _log;
};

bool BroadcastInputRewriter::isAddQuantized(mlir::Value input) const {
    const mlir::Operation* inputOp = input.getDefiningOp();
    if (inputOp == nullptr) {
        _log.trace("AvgPool's input is the region argument. Assuming it is not quantized.");
        return false;
    }
    return mlir::isa<IE::FakeQuantizeOp>(inputOp);
}

void BroadcastInputRewriter::createBroadcastBeforeAddOp(IE::AddOp origOp, mlir::PatternRewriter& rewriter,
                                                        mlir::MLIRContext* ctx, mlir::Value broadcastInput,
                                                        mlir::Value origInput, ShapeRef target_shape) const {
    const auto origOpLoc = vpux::stringifyLocation(origOp.getLoc());
    const auto broadcastedName = mlir::StringAttr::get(ctx, origOpLoc + "_broadcasted");
    const auto broadcastedLoc = mlir::NameLoc::get(broadcastedName);
    const auto fusedName = mlir::StringAttr::get(ctx, origOpLoc + "_fused");
    const auto fusedLoc = mlir::NameLoc::get(fusedName);
    auto broadcastedOp = rewriter.create<IE::BroadcastOp>(
            broadcastedLoc, broadcastInput, createShapeConstForBroadCast(rewriter, ctx, broadcastedLoc, target_shape),
            nullptr, IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));

    auto newAddOp = rewriter.create<IE::AddOp>(fusedLoc, origInput, broadcastedOp, origOp.auto_broadcast(),
                                               origOp.post_opAttr());
    rewriter.replaceOp(origOp, newAddOp.output());
}

mlir::LogicalResult BroadcastInputRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::AddOp Operation '{0}'", origOp->getLoc());

    const auto ctx = origOp->getContext();
    const auto lhsShape = origOp.input1().getType().template cast<vpux::NDTypeInterface>().getShape();
    const auto rhsShape = origOp.input2().getType().template cast<vpux::NDTypeInterface>().getShape();

    const auto nonTrivialDimPredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    const auto nonTrivialInputDims =
            std::count_if(lhsShape.raw().begin(), lhsShape.raw().end(), nonTrivialDimPredicate);
    const auto nonTrivialWeightDims =
            std::count_if(rhsShape.raw().begin(), rhsShape.raw().end(), nonTrivialDimPredicate);

    // Pattern like:
    // IE.Add: tensor<1xMxNxOxf16>, tensor<1xMxNx1xf16>, IE.Add: tensor<PxMxNxOxf16>, tensor<PxMxNx1xf16> (Input order
    // and vice versa), those cannot convert to Scaleshift, broadcast the input and lowering to eltwise add.
    if (lhsShape == rhsShape || !(nonTrivialInputDims > 1 && nonTrivialWeightDims > 1)) {
        return mlir::failure();
    }

    // If weights is constant, will be convert to ScaleShift in AdaptShapeForScaleShift Pass.
    if ((isAddQuantized(origOp.input1())
                 ? mlir::isa_and_nonnull<Const::DeclareOp>(
                           mlir::dyn_cast<IE::FakeQuantizeOp>(origOp.input1().getDefiningOp()).input().getDefiningOp())
                 : mlir::isa_and_nonnull<Const::DeclareOp>(origOp.input1().getDefiningOp())) ||
        (isAddQuantized(origOp.input2())
                 ? mlir::isa_and_nonnull<Const::DeclareOp>(
                           mlir::dyn_cast<IE::FakeQuantizeOp>(origOp.input2().getDefiningOp()).input().getDefiningOp())
                 : mlir::isa_and_nonnull<Const::DeclareOp>(origOp.input2().getDefiningOp()))) {
        return mlir::failure();
    }

    if (nonTrivialInputDims > nonTrivialWeightDims) {
        createBroadcastBeforeAddOp(origOp, rewriter, ctx, origOp.input2(), origOp.input1(),
                                   ShapeRef(getShape(origOp.input1())));
    } else {
        createBroadcastBeforeAddOp(origOp, rewriter, ctx, origOp.input1(), origOp.input2(),
                                   ShapeRef(getShape(origOp.input2())));
    }

    return mlir::success();
}

//
// BroadcastInputForAddPass
//
class BroadcastInputForAddPass final : public IE::BroadcastInputForAddBase<BroadcastInputForAddPass> {
public:
    explicit BroadcastInputForAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BroadcastInputForAddPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BroadcastInputRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createBroadcastInputForAddPass(Logger log) {
    return std::make_unique<BroadcastInputForAddPass>(log);
}
