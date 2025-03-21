//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
namespace IE {

enum class FuseMode { CONVERT_TO_SLICE, CONVERT_TO_EXPAND };

FuseMode getFuseMode(ShapeRef patternInShape, ShapeRef patternOutShape);

// If FuseMode is 'CONVERT_TO_SLICE',  the 'firstShape' is 'newSliceOffsets'; the 'secondShape' is 'newSliceStaticSizes'
// If FuseMode is 'CONVERT_TO_EXPAND', the 'firstShape' is 'newPadsBegin'; the 'secondShape' is 'newPadsEnd'
mlir::FailureOr<std::tuple<Shape, Shape, FuseMode>> getSliceExpandFusedParameters(IE::SliceOp sliceOp,
                                                                                  IE::ExpandOp expandOp);
mlir::FailureOr<std::tuple<Shape, Shape, FuseMode>> getExpandSliceFusedParameters(IE::ExpandOp expandOp,
                                                                                  IE::SliceOp sliceOp);

mlir::LogicalResult genericOptimizeSliceImplicitExpand(IE::ExpandOp layerOp, mlir::Operation* implicitOp,
                                                       bool hasCalculationCost, mlir::PatternRewriter& rewriter,
                                                       Logger innerLog);

mlir::LogicalResult genericOptimizeSliceImplicitShapeCastExpand(IE::ExpandOp layerOp, IE::ShapeCastOp shapeCastOp,
                                                                mlir::Operation* implicitOp,
                                                                mlir::PatternRewriter& rewriter, Logger innerLog);

bool isMiddleOpLegal(IE::SliceOp sliceOp, mlir::Operation* op, IE::ExpandOp expandOp);
bool isPReluLegal(IE::SliceOp sliceOp, IE::PReluOp preluOp, IE::ExpandOp expandOp);
bool isConcatLegal(IE::SliceOp sliceOp, IE::ConcatOp concatOp, IE::ExpandOp expandOp);

//
// OptimizeSliceImplicitExpand
//

template <class ImplicitLayer>
class OptimizeSliceImplicitExpand : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceImplicitExpand(mlir::MLIRContext* ctx, Logger log, bool hasCalculationCost)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log), _hasCalculationCost(hasCalculationCost) {
        setDebugName("OptimizeSliceImplicitExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        const auto innerLog = _log.nest();

        auto implicitOp = origOp.getInput().getDefiningOp<ImplicitLayer>();
        if (implicitOp == nullptr) {
            return mlir::failure();
        }
        return genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(), _hasCalculationCost, rewriter,
                                                  innerLog);
    }

private:
    Logger _log;
    bool _hasCalculationCost;
};

//
// OptimizeSliceExpand
//

class OptimizeSliceExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeExpandSlice
//

class OptimizeExpandSlice final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeExpandSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeExpandSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeSliceShapeCastExpand
//

template <class EltwiseLikeSwOp>
class OptimizeSliceShapeCastExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceShapeCastExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceShapeCastExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        const auto innerLog = _log.nest();

        auto shapeCastOp = origOp.getInput().getDefiningOp<IE::ShapeCastOp>();
        if (shapeCastOp == nullptr) {
            innerLog.trace("Expand '{0}' input is not 'shapeCastOp'", origOp->getLoc());
            return mlir::failure();
        }
        if (!shapeCastOp->hasOneUse()) {
            return mlir::failure();
        }

        auto swOp = shapeCastOp.getSource().getDefiningOp<EltwiseLikeSwOp>();
        if (swOp == nullptr) {
            innerLog.trace("ShapeCastOp '{0}' input is not SW op", shapeCastOp->getLoc());
            return mlir::failure();
        }
        if (!swOp->hasOneUse()) {
            return mlir::failure();
        }

        return genericOptimizeSliceImplicitShapeCastExpand(origOp, shapeCastOp, swOp.getOperation(), rewriter,
                                                           innerLog);
    }

private:
    Logger _log;
};

//
// OptimizeSliceMultiInputsExpand
//

template <class ConcreteOp>
class OptimizeSliceMultiInputsExpand : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceMultiInputsExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceMultiInputsExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());

        auto implicitOp = expandOp.getInput().getDefiningOp<ConcreteOp>();
        if (implicitOp == nullptr) {
            return mlir::failure();
        }

        if (!isOpLegal(implicitOp, expandOp)) {
            _log.trace("MiddleOp '{0}' is illegal", implicitOp->getName());
            return mlir::failure();
        }

        auto newInputValues = updateInputsForOp(rewriter, implicitOp, expandOp);

        mlir::IRMapping mapper;
        mapper.map(implicitOp.getOperands(), newInputValues);
        auto newOp = rewriter.clone(*implicitOp, mapper);
        vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

        _log.trace("Optimization completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOp(expandOp, newOp->getResults());

        return mlir::success();
    }

protected:
    virtual SmallVector<mlir::Value> updateInputsForOp(mlir::PatternRewriter& rewriter, ConcreteOp origOp,
                                                       IE::ExpandOp expandOp) const = 0;
    virtual bool isOpLegal(ConcreteOp origOp, IE::ExpandOp expandOp) const = 0;

private:
    Logger _log;
};

//
// OptimizeSlicePReluExpand
//

class OptimizeSlicePReluExpand final : public OptimizeSliceMultiInputsExpand<IE::PReluOp> {
public:
    OptimizeSlicePReluExpand(mlir::MLIRContext* ctx, Logger log)
            : OptimizeSliceMultiInputsExpand<IE::PReluOp>(ctx, log) {
        setDebugName("OptimizeSlicePReluExpand");
    }

private:
    SmallVector<mlir::Value> updateInputsForOp(mlir::PatternRewriter& rewriter, IE::PReluOp origOp,
                                               IE::ExpandOp expandOp) const override;
    bool isOpLegal(IE::PReluOp origOp, IE::ExpandOp expandOp) const override {
        auto sliceOp = origOp.getInput().getDefiningOp<IE::SliceOp>();
        return sliceOp != nullptr && isPReluLegal(sliceOp, origOp, expandOp);
    }
};

//
// OptimizeSliceConcatExpand
//

class OptimizeSliceConcatExpand final : public OptimizeSliceMultiInputsExpand<IE::ConcatOp> {
public:
    OptimizeSliceConcatExpand(mlir::MLIRContext* ctx, Logger log)
            : OptimizeSliceMultiInputsExpand<IE::ConcatOp>(ctx, log) {
        setDebugName("OptimizeSliceConcatExpand");
    }

public:
    SmallVector<mlir::Value> updateInputsForOp(mlir::PatternRewriter& rewriter, IE::ConcatOp origOp,
                                               IE::ExpandOp expandOp) const override;
    bool isOpLegal(IE::ConcatOp origOp, IE::ExpandOp expandOp) const override {
        return isConcatLegal(nullptr, origOp, expandOp);
    }
};

//
// OptimizeSliceOpsExpand
//

class OptimizeSliceOpsExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceOpsExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceOpsExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeSliceEltwiseExpand
//

template <class EltwiseLikeOp>
class OptimizeSliceEltwiseExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceEltwiseExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceEltwiseExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

        auto eltwiseOp = origOp.getInput().getDefiningOp<EltwiseLikeOp>();
        if (eltwiseOp == nullptr || !eltwiseOp->hasOneUse()) {
            _log.trace("Cannot get 'Eltwise' before '{0}' or Eltwise has multi uses", origOp->getName());
            return mlir::failure();
        }

        SmallVector<mlir::Value> inputs;
        auto outputType = origOp.getOutput().getType();
        for (auto input : eltwiseOp->getOperands()) {
            auto sliceOp = input.getDefiningOp();
            if (!mlir::isa<IE::SliceOp>(sliceOp) || !sliceOp->hasOneUse()) {
                _log.trace("Cannot get 'Slice' before '{0} or Slice has multi uses'", eltwiseOp->getName());
                return mlir::failure();
            }
            auto sliceInput = sliceOp->getOperand(0);
            if (sliceInput.getType() != outputType) {
                return mlir::failure();
            }
            inputs.push_back(sliceInput);
        }

        mlir::IRMapping mapper;
        mapper.map(eltwiseOp.getOperands(), inputs);
        auto newOp = rewriter.clone(*eltwiseOp, mapper);
        vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

        _log.trace("Optimization completed successfully at '{0}'", origOp->getLoc());
        rewriter.replaceOp(origOp, newOp->getResults());

        return mlir::success();
    }

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
