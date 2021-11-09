//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph_ops/convolution_ie.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <ngraph/slice_plan.hpp>

using namespace vpux;

namespace {

//
// UpstreamSlicePass
//

class UpstreamSlicePass final : public IE::UpstreamSliceBase<UpstreamSlicePass> {
public:
    explicit UpstreamSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GenericSliceUpstreaming;

private:
    void safeRunOnFunc() final;
};

//
// GenericSliceUpstreaming
//

class UpstreamSlicePass::GenericSliceUpstreaming final : public mlir::OpInterfaceRewritePattern<IE::LayerOpInterface> {
public:
    GenericSliceUpstreaming(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isUpstreamPossible(IE::LayerOpInterface sliceOp, mlir::Value tensor) {
    if (tensor.isa<mlir::BlockArgument>())
        return false;
    mlir::Operation* parentOp = tensor.getDefiningOp();
    // Unary and eltwise ops are primary candidates for upstreaming slice ops.
    // Later on, implementation could handle also Conv, Pool upstreaming
    if (!parentOp->hasTrait<IE::EltwiseOp>())
        return false;

    if (parentOp->getNumResults() > 1)
        return false;

    // Strided slice is known to be done as UPA task.
    // It does not support datatypes generically
    // so we can't afford changing datatype of this operation.
    if (mlir::isa<IE::StridedSliceOp>(sliceOp)) {
        auto sliceOpElementType = tensor.getType().cast<mlir::RankedTensorType>().getElementType();
        auto parentOpElementType = parentOp->getOperand(0).getType().cast<mlir::RankedTensorType>().getElementType();
        if (sliceOpElementType != parentOpElementType)
            return false;
    }

    // Another restriction due to limited implementation.
    // Upstreaming through the graph using op interfaces it's hard
    // enough to reason about which operands are path of the activation
    // path and which are parameters.
    // An interface that makes this dinstinction easy would be of help.
    const auto operands = parentOp->getOperands();
    if (operands.size() > 1 &&
        std::adjacent_find(operands.begin(), operands.end(), [](mlir::Value val1, mlir::Value val2) {
            return getShape(val1) != getShape(val2);
        }) != operands.end())
        return false;

    const auto inputShape = getShape(sliceOp.getInputs()[0]);
    const auto outputShape = getShape(sliceOp.getOutputs()[0]);

    // Can't reason yet with generic shape dimensions
    if (inputShape.size() != 4 || outputShape.size() != 4)
        return false;

    // Can't handle yet upstreaming and adapting channelwise parameters
    if (const auto quantAxis = IE::getQuantAxisIndex(parentOp)) {
        if (inputShape[Dim(quantAxis.getValue())] != outputShape[Dim(quantAxis.getValue())]) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult UpstreamSlicePass::GenericSliceUpstreaming::matchAndRewrite(IE::LayerOpInterface origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<IE::SliceOp, IE::StridedSliceOp>(origOp)) {
        return mlir::failure();
    }

    auto origInput = origOp.getInputs()[0];
    if (!origInput.hasOneUse()) {
        return mlir::failure();
    }

    if (!isUpstreamPossible(origOp, origInput)) {
        return mlir::failure();
    }

    const auto reInferReturnTypes = [](mlir::InferTypeOpInterface& op, bool limitToShape = false) {
        mlir::SmallVector<mlir::Type> inferredTypes;
        VPUX_THROW_UNLESS(op.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                                              op->getAttrDictionary(), op->getRegions(), inferredTypes)
                                  .succeeded(),
                          "New type inference failed for '{0}'", op);
        if (limitToShape) {
            for (auto p : zip(op->getResults(), inferredTypes)) {
                auto resVal = std::get<0>(p);
                const auto origType = resVal.getType().cast<mlir::ShapedType>();
                const auto inferredType = std::get<1>(p).cast<mlir::ShapedType>();
                const auto newType = changeShape(origType, getShape(inferredType));
                resVal.setType(newType);
            }
        } else {
            for (auto p : zip(op->getResults(), inferredTypes)) {
                std::get<0>(p).setType(std::get<1>(p));
            }
        }
    };

    mlir::InferTypeOpInterface parentOp = origInput.getDefiningOp();
    rewriter.setInsertionPoint(parentOp);
    auto opOperands = parentOp->getOpOperands();
    if (std::adjacent_find(opOperands.begin(), opOperands.end(), [](mlir::OpOperand& val1, mlir::OpOperand& val2) {
            return val1.get() != val2.get();
        }) == opOperands.end()) {
        auto newSlice = mlir::cast<mlir::InferTypeOpInterface>(rewriter.clone(*origOp));
        newSlice->setOperand(0, opOperands[0].get());
        reInferReturnTypes(newSlice);
        for (auto& operand : opOperands) {
            operand.set(newSlice->getResult(0));
        }
    } else {
        for (auto& operand : opOperands) {
            auto newSlice = mlir::cast<mlir::InferTypeOpInterface>(rewriter.clone(*origOp));
            newSlice->setOperand(0, operand.get());
            reInferReturnTypes(newSlice);
            operand.set(newSlice->getResult(0));
        }
    }

    VPUX_THROW_UNLESS(parentOp->getResults().size() == 1, "Don't support backprop for multiple outputs yet '{0}'",
                      parentOp);
    reInferReturnTypes(parentOp, true);
    rewriter.replaceOp(origOp, parentOp->getResults());

    _log.trace("Upstreamed slice.");

    return mlir::success();
}

//
// safeRunOnFunc
//

void UpstreamSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GenericSliceUpstreaming>(&ctx, _log);

    IE::SliceOp::getCanonicalizationPatterns(patterns, &ctx);
    IE::StridedSliceOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUpstreamSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUpstreamSlicePass(Logger log) {
    return std::make_unique<UpstreamSlicePass>(log);
}
