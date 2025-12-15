//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DYNAMICCONCATTOSCATTERNDUPDATE
#define GEN_PASS_DEF_DYNAMICCONCATTOSCATTERNDUPDATE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class DynamicConcatToScatterNDUpdate final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    DynamicConcatToScatterNDUpdate(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DynamicConcatToScatterNDUpdate::matchAndRewrite(IE::ConcatOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto loc = origOp->getLoc();

    const auto firstInputType = mlir::cast<mlir::RankedTensorType>(origOp.getInputs().front().getType());
    const auto secondInputType = mlir::cast<mlir::RankedTensorType>(origOp.getInputs().back().getType());

    auto firstInput = origOp.getInputs().front();
    auto secondInput = origOp.getInputs().back();

    bool isFirstStatic = firstInputType.hasStaticShape();
    bool isSecondStatic = secondInputType.hasStaticShape();

    if (isFirstStatic == isSecondStatic) {
        return mlir::failure();
    }

    auto dynamicInput = isFirstStatic ? secondInput : firstInput;
    auto staticInput = isFirstStatic ? firstInput : secondInput;

    auto origResultType = origOp.getType();
    auto boundedOrigResultType = mlir::dyn_cast<Core::BoundedTensorType>(origResultType);
    const auto boundsShape = boundedOrigResultType.getBounds();
    SmallVector<int64_t> boundsVec(boundsShape.begin(), boundsShape.end());
    BoundsRef boundsRef(boundsVec);

    auto boundedDynamicInputType = mlir::dyn_cast<Core::BoundedTensorType>(dynamicInput.getType());
    const auto updatedDynamicInputType = boundedDynamicInputType.changeBounds(boundsRef);
    dynamicInput.setType(updatedDynamicInputType);

    const auto dataTypeShapeOf = mlir::TypeAttr::get(getSInt64Type(ctx));

    auto shapeOfOp =
            rewriter.create<IE::ShapeOfOp>(appendLoc(origOp->getLoc(), "_shapeOf"), dynamicInput, dataTypeShapeOf);

    auto channelDimOp = rewriter.create<IE::SliceOp>(appendLoc(origOp->getLoc(), "_sliceOf"), shapeOfOp,
                                                     rewriter.getI64ArrayAttr({1}), rewriter.getI64ArrayAttr({1}));

    auto constDataType = mlir::RankedTensorType::get({1}, getSInt64Type(ctx));
    auto addendCstOp =
            Const::createConst(rewriter, appendLoc(loc, "_channelsNum"), constDataType, ArrayRef<int64_t>{1});
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto newChannelDimOp = rewriter.create<IE::AddOp>(
            appendLoc(origOp->getLoc(), "_add"), channelDimOp, addendCstOp, broadcastType,
            /*post_op=*/nullptr, /*clamp=*/nullptr, /*outputPadding*/ nullptr, /*inputPadding*/ nullptr);

    auto batchDimOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceCst"), shapeOfOp,
                                                   /*begins=*/rewriter.getI64ArrayAttr({0}),
                                                   /*sizes=*/rewriter.getI64ArrayAttr({1}));

    auto widthAndHeightDimOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceCst2"), shapeOfOp,
                                                            /*begins=*/rewriter.getI64ArrayAttr({2}),
                                                            /*sizes=*/rewriter.getI64ArrayAttr({2}));

    const auto axisAttr = getIntAttr(ctx, 0);
    SmallVector<mlir::Value> concatenatedDims;
    concatenatedDims.push_back(batchDimOp);
    concatenatedDims.push_back(newChannelDimOp);
    concatenatedDims.push_back(widthAndHeightDimOp);

    auto newShapeOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_newShape"), concatenatedDims, axisAttr);

    auto newInputDataShape = to_small_vector(getShape(dynamicInput));
    const auto newInputDataShapeAttr = getIntArrayAttr(ctx, newInputDataShape);
    const auto newInputDataBoundsAttr = getIntArrayAttr(ctx, boundsVec);
    auto dynamicInputReshaped = rewriter.create<IE::DynamicReshapeOp>(
            appendLoc(loc, "_reshaped"), dynamicInput, newShapeOp, newInputDataShapeAttr, newInputDataBoundsAttr,
            /*onlySetShape*/ true);

    auto funcOp = mlir::cast<mlir::func::FuncOp>(channelDimOp.getOperation()->getParentOp());

    auto funcType = funcOp.getFunctionType();
    auto updatedInputTypes = funcType.getInputs();

    auto dynamicResultType = mlir::RankedTensorType::get(origResultType.getShape(), origResultType.getElementType(),
                                                         origResultType.getEncoding());

    SmallVector<mlir::Type, 4> finalInputTypes;
    finalInputTypes.push_back(updatedDynamicInputType);
    finalInputTypes.append(updatedInputTypes.begin() + 1, updatedInputTypes.end());

    auto zeroCstOp = Const::createConst(rewriter, appendLoc(loc, "_channelsNum"), constDataType, ArrayRef<int64_t>{0});
    SmallVector<mlir::Value> indicesValues;
    indicesValues.push_back(zeroCstOp);
    indicesValues.push_back(channelDimOp);
    indicesValues.push_back(zeroCstOp);
    auto indicesOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_indices"), indicesValues, axisAttr);

    auto finalFunctionType = mlir::FunctionType::get(ctx, finalInputTypes, {dynamicResultType});
    funcOp.setType(finalFunctionType);
    auto scatterOp = rewriter.create<IE::ScatterNDUpdateOp>(appendLoc(origOp->getLoc(), "_scatterNDUpdate"),
                                                            dynamicInputReshaped, indicesOp, staticInput);
    rewriter.replaceOp(origOp, scatterOp.getResult());

    return mlir::success();
}

class DynamicConcatToScatterNDUpdatePass final :
        public IE::impl::DynamicConcatToScatterNDUpdateBase<DynamicConcatToScatterNDUpdatePass> {
public:
    explicit DynamicConcatToScatterNDUpdatePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DynamicConcatToScatterNDUpdatePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<DynamicConcatToScatterNDUpdate>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createDynamicConcatToScatterNDUpdatePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDynamicConcatToScatterNDUpdatePass(Logger log) {
    return std::make_unique<DynamicConcatToScatterNDUpdatePass>(log);
}
