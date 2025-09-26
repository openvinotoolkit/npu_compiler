//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MVNFUSION
#define GEN_PASS_DEF_MVNFUSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr double EPS_THRESHOLD = 1e-4;

//
// Helpers
//

std::optional<Shape> canConvertToMVN1(ShapeRef inputShapeRef, ArrayRef<int64_t> axes, bool& isAcrossChannel) {
    const auto inputRank = inputShapeRef.size();
    const auto inputShape = inputShapeRef.raw();

    isAcrossChannel = false;
    if (inputRank == 2 && axes.size() == 1 && axes[0] == 1) {
        // HxW -> 1xHxWx1
        return Shape{1, inputShape[0], inputShape[1], 1};
    } else if (inputRank == 3) {
        if (axes.size() == 1 && axes[0] == 2) {
            // CxHxW -> CxHxWx1
            return Shape{inputShape[0], inputShape[1], inputShape[2], 1};
        } else if (axes.size() == 2 && axes[0] == 1 && axes[1] == 2) {
            // CxHxW -> 1xCxHxW
            return Shape{1, inputShape[0], inputShape[1], inputShape[2]};
        }
    } else if (inputRank == 4) {
        if (axes.size() == 3 && axes[0] == 1 && axes[1] == 2 && axes[2] == 3) {
            isAcrossChannel = true;
            return inputShape;
        } else if (axes.size() == 2 && axes[0] == 2 && axes[1] == 3) {
            return inputShape;
        }
    }
    return std::nullopt;
}

std::optional<double> getEpsValue(mlir::Value epsInput, bool isOutsideEps) {
    auto convertOp = epsInput.getDefiningOp<IE::ConvertOp>();
    auto constOp = epsInput.getDefiningOp<Const::DeclareOp>();
    if (convertOp) {
        constOp = convertOp.getInput().getDefiningOp<Const::DeclareOp>();
    }
    if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
        return std::nullopt;
    }
    const auto epsContent = constOp.getContent();
    const auto epsValue = epsContent.getSplatValue<double>();
    if (isOutsideEps && epsValue > EPS_THRESHOLD) {
        return std::nullopt;
    }
    return epsValue;
}

void normalizeAndSortAxes(SmallVector<int64_t>& axes, int64_t rank) {
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0) {
            axes[i] += rank;
        }
    }
    std::sort(axes.begin(), axes.end());
}

//
// MVNFusion
//

class MVNFusion final : public mlir::OpRewritePattern<IE::DivideOp> {
public:
    MVNFusion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DivideOp>(ctx), _log(log) {
        setDebugName("MVNFusion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// This pass convert this subgraph
//    (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean(x^2, axes) - (ReduceMean(x, axes) ^ 2)) + eps)
// or
//    (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean(x^2, axes) - (ReduceMean(x, axes) ^ 2) + eps))
// to a single MVN1
//

mlir::LogicalResult MVNFusion::matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Divide '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto meanSubOp = origOp.getInput1().getDefiningOp<IE::SubtractOp>();
    if (meanSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "No x SubtractOp found");
    }

    auto inputMeanOp = meanSubOp.getInput2().getDefiningOp<IE::ReduceMeanOp>();
    if (inputMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No x ReduceMeanOp found");
    }
    auto inputMeanAxesValue = IE::extractAxes(origOp.getLoc(), inputMeanOp);

    if (inputMeanOp.getInput() != meanSubOp.getInput1()) {
        return matchFailed(rewriter, origOp, "Not the same input");
    }
    const auto mvnInput = inputMeanOp.getInput();
    const auto inputShape = getShape(mvnInput);
    const auto inputRank = inputShape.size();
    if (inputRank < 2 || inputRank > 4) {
        return matchFailed(rewriter, origOp, "Invalid input shape rank");
    }

    // inside-sqrt or outside-sqrt
    auto insideEpsSqrtOp = origOp.getInput2().getDefiningOp<IE::SqrtOp>();
    auto outsideEpsAddOp = origOp.getInput2().getDefiningOp<IE::AddOp>();

    IE::SubtractOp squareSubOp = nullptr;
    mlir::Value epsInput;
    bool isOutsideEps;
    if (insideEpsSqrtOp) {
        auto insideEpsAddOp = insideEpsSqrtOp.getInput().getDefiningOp<IE::AddOp>();
        if (insideEpsAddOp == nullptr) {
            return matchFailed(rewriter, origOp, "No inside-eps AddOp found");
        }
        squareSubOp = insideEpsAddOp.getInput1().getDefiningOp<IE::SubtractOp>();
        if (squareSubOp == nullptr) {
            return matchFailed(rewriter, origOp, "No inside-eps SubtractOp found");
        }

        epsInput = insideEpsAddOp.getInput2();
        isOutsideEps = false;
    } else if (outsideEpsAddOp) {
        auto outsideEpsSqrtOp = outsideEpsAddOp.getInput1().getDefiningOp<IE::SqrtOp>();
        if (outsideEpsSqrtOp == nullptr) {
            return matchFailed(rewriter, origOp, "No outside-eps SqrtOp found");
        }
        squareSubOp = outsideEpsSqrtOp.getInput().getDefiningOp<IE::SubtractOp>();
        if (squareSubOp == nullptr) {
            return matchFailed(rewriter, origOp, "No outside-eps SubtractOp found");
        }

        epsInput = outsideEpsAddOp.getInput2();
        isOutsideEps = true;
    } else {
        return matchFailed(rewriter, origOp, "No inside-eps or outside-eps mode found");
    }

    if (squareSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "No square SubtractOp found");
    }

    auto epsValueOpt = getEpsValue(epsInput, isOutsideEps);
    if (!epsValueOpt.has_value()) {
        return matchFailed(rewriter, origOp, "No valid eps found");
    }
    const auto epsValue = epsValueOpt.value();

    auto isMultiplySquare = [](IE::MultiplyOp op) {
        return op.getInput1() == op.getInput2();
    };

    auto squareMeanOp = squareSubOp.getInput1().getDefiningOp<IE::ReduceMeanOp>();
    if (squareMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No square ReduceMeanOp found");
    }
    auto squareMeanAxesValue = IE::extractAxes(origOp.getLoc(), squareMeanOp);

    if (inputMeanAxesValue != squareMeanAxesValue) {
        return matchFailed(rewriter, origOp, "ReduceMean ops have different axes");
    }

    auto squareOp = squareMeanOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (squareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No x MultiplyOp found");
    }
    if (!isMultiplySquare(squareOp)) {
        return matchFailed(rewriter, origOp, "x MultiplyOp is not square");
    }
    if (squareOp.getInput1() != mvnInput) {
        return matchFailed(rewriter, origOp, "Not the same input");
    }

    auto meanSquareOp = squareSubOp.getInput2().getDefiningOp<IE::MultiplyOp>();
    if (meanSquareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No MultiplyOp for ReduceMean found");
    }
    if (!isMultiplySquare(meanSquareOp)) {
        return matchFailed(rewriter, origOp, "MultiplyOp for ReduceMean is not square");
    }
    auto meanToSquareOp = meanSquareOp.getInput1().getDefiningOp<IE::ReduceMeanOp>();
    if (meanToSquareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No ReduceMeanOp for square found");
    }
    if (meanToSquareOp != inputMeanOp) {
        return matchFailed(rewriter, origOp, "Not the same ReduceMean input");
    }

    normalizeAndSortAxes(inputMeanAxesValue, inputRank);

    bool isAcrossChannel;
    const auto newShapeOpt = canConvertToMVN1(inputShape, inputMeanAxesValue, isAcrossChannel);
    if (!newShapeOpt.has_value()) {
        return matchFailed(rewriter, origOp, "Cannot convert to mvn");
    }

    const auto ctx = origOp.getContext();
    auto preReshapeOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "in_reshape"), inputMeanOp.getInput(), nullptr,
                                                       false, getIntArrayAttr(ctx, newShapeOpt.value()));

    const auto normVarianceAttr = mlir::BoolAttr::get(ctx, true);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(ctx, isAcrossChannel);
    const auto epsAttr = getFPAttr(ctx, epsValue);

    auto mvnOp = rewriter.create<IE::MVNOp>(origOp.getLoc(), preReshapeOp.getOutput(), acrossChannelsAttr,
                                            normVarianceAttr, epsAttr);

    _log.trace("Replace '{0}' with new op '{1}'", origOp.getLoc(), mvnOp);
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, mvnOp.getOutput(), nullptr, false,
                                                                 getIntArrayAttr(ctx, getShape(origOp.getOutput())));
    extendOpLoc(outReshape, "out_reshape");
    return mlir::success();
}

//
// MVNFusionOvRef
//

class MVNFusionOvRef final : public mlir::OpRewritePattern<IE::DivideOp> {
public:
    MVNFusionOvRef(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DivideOp>(ctx), _log(log) {
        setDebugName("MVNFusionOvRef");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// Converts the typical OV MVN decomposed-subgraph back to MVN1
//   D = x - ReduceMean(x, axes)
// out = D /  Sqrt(ReduceMean(D ^ 2) + eps), or
//     = D / (Sqrt(ReduceMean(D ^ 2)      ) + eps)
//

mlir::LogicalResult MVNFusionOvRef::matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Divide '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto meanSubOp = origOp.getInput1().getDefiningOp<IE::SubtractOp>();
    if (meanSubOp == nullptr) {
        return matchFailed(rewriter, origOp, "No SubtractOp found");
    }

    auto inputMeanOp = meanSubOp.getInput2().getDefiningOp<IE::ReduceMeanOp>();
    if (inputMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No ReduceMeanOp(x) found");
    }
    auto inputMeanAxesValue = IE::extractAxes(origOp.getLoc(), inputMeanOp);

    if (inputMeanOp.getInput() != meanSubOp.getInput1()) {
        return matchFailed(rewriter, origOp, "No (x - ReduceMeanOp(x)) found");
    }
    const auto mvnInput = inputMeanOp.getInput();
    const auto inputShape = getShape(mvnInput);
    const auto inputRank = inputShape.size();
    if (inputRank < 2 || inputRank > 4) {
        return matchFailed(rewriter, origOp, "Invalid input shape rank");
    }

    // inside-sqrt or outside-sqrt
    auto insideEpsSqrtOp = origOp.getInput2().getDefiningOp<IE::SqrtOp>();
    auto outsideEpsAddOp = origOp.getInput2().getDefiningOp<IE::AddOp>();

    mlir::Value epsInput;
    mlir::Operation* reduceMean2 = nullptr;
    bool isOutsideEps;
    if (insideEpsSqrtOp) {
        auto insideEpsAddOp = insideEpsSqrtOp.getInput().getDefiningOp<IE::AddOp>();
        if (insideEpsAddOp == nullptr) {
            return matchFailed(rewriter, origOp, "No inside-eps AddOp found");
        }
        epsInput = insideEpsAddOp.getInput2();
        reduceMean2 = insideEpsAddOp.getInput1().getDefiningOp();
        isOutsideEps = false;
    } else if (outsideEpsAddOp) {
        auto outsideEpsSqrtOp = outsideEpsAddOp.getInput1().getDefiningOp<IE::SqrtOp>();
        if (outsideEpsSqrtOp == nullptr) {
            return matchFailed(rewriter, origOp, "No outside-eps SqrtOp found");
        }
        epsInput = outsideEpsAddOp.getInput2();
        reduceMean2 = outsideEpsSqrtOp.getInput().getDefiningOp();
        isOutsideEps = true;
    } else {
        return matchFailed(rewriter, origOp, "No inside-eps or outside-eps mode found");
    }

    auto epsValueOpt = getEpsValue(epsInput, isOutsideEps);
    if (!epsValueOpt.has_value()) {
        return matchFailed(rewriter, origOp, "No valid eps found");
    }
    const auto epsValue = epsValueOpt.value();

    VPUX_THROW_UNLESS(reduceMean2, "Checked Add/Sqrt op (mandatory inputs) has nullptr input");

    auto squareMeanOp = mlir::dyn_cast<IE::ReduceMeanOp>(*reduceMean2);
    if (squareMeanOp == nullptr) {
        return matchFailed(rewriter, origOp, "No square ReduceMeanOp found");
    }
    auto squareMeanAxesValue = IE::extractAxes(origOp.getLoc(), squareMeanOp);

    if (inputMeanAxesValue != squareMeanAxesValue) {
        return matchFailed(rewriter, origOp, "ReduceMean ops have different axes");
    }

    // detect Pow(x,2) or Multiply(x,x)
    auto getSquareOp = [](mlir::Operation* op) -> mlir::Operation* {
        if (auto power = mlir::dyn_cast<IE::PowerOp>(*op)) {
            auto constOp = power.getInput2().getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr || !constOp.getContentAttr().isSplat()) {
                return nullptr;
            }
            const auto coefContent = constOp.getContent();
            const auto coefValue = coefContent.getSplatValue<double>();
            return (coefValue == 2.0) ? op : nullptr;
        } else if (auto mul = mlir::dyn_cast<IE::MultiplyOp>(op)) {
            return mul.getInput1() == mul.getInput2() ? op : nullptr;
        }
        return nullptr;
    };

    mlir::Operation* squareOp = getSquareOp(squareMeanOp.getInput().getDefiningOp());
    if (squareOp == nullptr) {
        return matchFailed(rewriter, origOp, "No Squaring op found");
    }
    if (squareOp->getOperand(0).getDefiningOp() != meanSubOp) {
        return matchFailed(rewriter, origOp, "Subtract->SquareOp link not found");
    }

    normalizeAndSortAxes(inputMeanAxesValue, inputRank);

    bool isAcrossChannel;
    const auto newShapeOpt = canConvertToMVN1(inputShape, inputMeanAxesValue, isAcrossChannel);
    if (!newShapeOpt.has_value()) {
        return matchFailed(rewriter, origOp, "Cannot convert to mvn");
    }

    const auto ctx = origOp.getContext();
    auto preReshapeOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "in_reshape"), inputMeanOp.getInput(), nullptr,
                                                       false, getIntArrayAttr(ctx, newShapeOpt.value()));

    const auto normVarianceAttr = mlir::BoolAttr::get(ctx, true);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(ctx, isAcrossChannel);
    const auto epsAttr = getFPAttr(ctx, epsValue);

    auto mvnOp = rewriter.create<IE::MVNOp>(origOp.getLoc(), preReshapeOp.getOutput(), acrossChannelsAttr,
                                            normVarianceAttr, epsAttr);

    _log.trace("Replace '{0}' with new op '{1}'", origOp.getLoc(), mvnOp);
    auto outReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, mvnOp.getOutput(), nullptr, false,
                                                                 getIntArrayAttr(ctx, getShape(origOp.getOutput())));
    extendOpLoc(outReshape, "out_reshape");
    return mlir::success();
}

//
// MVNFusionPass
//

class MVNFusionPass final : public IE::impl::MVNFusionBase<MVNFusionPass> {
public:
    explicit MVNFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MVNFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MVNFusion>(&ctx, _log);
    patterns.add<MVNFusionOvRef>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMVNFusionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMVNFusionPass(Logger log) {
    return std::make_unique<MVNFusionPass>(log);
}
