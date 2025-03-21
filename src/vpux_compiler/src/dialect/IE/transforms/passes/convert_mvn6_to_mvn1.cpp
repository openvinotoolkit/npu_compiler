//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTMVN6TOMVN1
#define GEN_PASS_DEF_CONVERTMVN6TOMVN1
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertMVN6ToMVN1
//

class ConvertMVN6ToMVN1 final : public mlir::OpRewritePattern<IE::MVN6Op> {
public:
    ConvertMVN6ToMVN1(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVN6Op>(ctx), _log(log) {
        setDebugName("ConvertMVN6ToMVN1");
    }

    mlir::LogicalResult matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertMVN6ToMVN1::matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    auto inputShapeVal = inputShape.raw();
    const auto inputShapeSize = inputShape.size();
    const auto inRank = inputType.getRank();

    if (origOp.getScale() || origOp.getBias()) {
        _log.nest().trace("MVN6 got scale/bias, cannot convert to MVN1.");
        return mlir::failure();
    }

    if (inputShapeSize < 2 || inputShapeSize > 5) {
        _log.nest().trace("MVN6 -> MVN1 conversion pass supports only 2D, 3D, 4D or 5D cases. Got {0}D input shape",
                          inputShapeSize);
        return mlir::failure();
    }

    const auto epsMode = origOp.getEpsMode();
    const auto eps = origOp.getEpsAttr().getValueAsDouble();
    const bool normalizeVariance = origOp.getNormalizeVariance();
    bool acrossChannels = false;

    if (epsMode != IE::MvnEpsMode::INSIDE_SQRT) {
        _log.nest().trace("MVN-1 does not support OUTSIDE_SQRT eps mode, unless small enough 'eps' values. If "
                          "OUTSIDE_SQRT is not supported, we should do MVNFusion pass");

        const double epsThreshold = 1e-3;
        if (eps > epsThreshold) {
            _log.nest().trace("For small enough 'eps' values, can treat OUTSIDE_SQRT mode as INSIDE_SQRT. Can not "
                              "convert because of large epsilon value: {0} vs {1}",
                              eps, epsThreshold);
            return mlir::failure();
        }
    }

    SmallVector<int64_t> axesAttr;

    if (origOp.getAxes() != nullptr && !origOp.getAxesValue().has_value()) {
        auto axesConst = origOp.getAxes().getDefiningOp<Const::DeclareOp>();
        if (axesConst == nullptr) {
            return mlir::failure();
        }

        const auto axesContent = axesConst.getContent();
        axesAttr = to_small_vector(axesContent.getValues<int64_t>());

        for (auto& axis : axesAttr) {
            if (axis < 0) {
                axis += inRank;
            }
        }
        std::sort(axesAttr.begin(), axesAttr.end());
    } else if (origOp.getAxes() == nullptr && origOp.getAxesValue().has_value()) {
        axesAttr = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    } else {
        return mlir::failure();
    }

    // 4D input and axis is 1, we need a Transpose to transpose dim C
    // to spatial dim
    IE::TransposeOp transposeIn;
    if (inputShapeSize == 4 && axesAttr.size() == 1 && axesAttr[0] == 1) {
        _log.trace("Transpose dim C to spatial dim");
        const auto transposeOrder = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 2, 3, 1}, getContext()));
        const auto transposeLoc = appendLoc(origOp->getLoc(), "transpose_mvn_in");
        rewriter.setInsertionPoint(origOp);
        transposeIn = rewriter.create<IE::TransposeOp>(transposeLoc, origOp.getInput(), nullptr, transposeOrder);
        origOp.setOperand(0, transposeIn);
        vpux::inferReturnTypes(origOp, vpux::InferShapedTypeMode::SHAPE);

        inputShapeVal = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getShape().raw();
        axesAttr.clear();
        axesAttr.push_back(inputShapeSize - 1);
    }

    SmallVector<int64_t> newInShape;

    if ((inputShapeSize == 2 || inputShapeSize == 3 || inputShapeSize == 4) && axesAttr.size() == 1 &&
        static_cast<uint32_t>(axesAttr[0]) == (inputShapeSize - 1)) {
        acrossChannels = false;
        if (inputShape.size() == 4) {
            newInShape = {inputShapeVal[0], inputShapeVal[1] * inputShapeVal[2], 1, inputShapeVal[3]};
        } else if (inputShape.size() == 3) {
            newInShape = {inputShapeVal[0], inputShapeVal[1], inputShapeVal[2], 1};
        } else if (inputShape.size() == 2) {
            newInShape = {1, inputShapeVal[0], inputShapeVal[1], 1};
        } else {
            return mlir::failure();
        }
    } else if (inputShapeSize == 3) {
        if (axesAttr.size() == 2 && axesAttr[0] == 1 && axesAttr[1] == 2) {
            newInShape = {1, inputShapeVal[0], inputShapeVal[1], inputShapeVal[2]};
        }
    } else if (inputShapeSize == 4) {
        newInShape = decltype(newInShape){inputShapeVal.begin(), inputShapeVal.end()};

        if (axesAttr.size() == 3 && axesAttr[0] == 1 && axesAttr[1] == 2 && axesAttr[2] == 3) {
            acrossChannels = true;
        } else if (axesAttr.size() == 2 && axesAttr[0] == 2 && axesAttr[1] == 3) {
            acrossChannels = false;
        } else {
            _log.nest().trace("MVN-1 layer supports only normalization across channel or spatial dimension, in this "
                              "case we should do MVNFusion pass");
            return mlir::failure();
        }
    } else if (inputShapeSize == 5 && axesAttr.size() == 3 && axesAttr[0] == 2 && axesAttr[1] == 3 &&
               axesAttr[2] == 4) {
        if (inputShape.size() == 5) {
            newInShape = {inputShapeVal[0], inputShapeVal[1], inputShapeVal[2], inputShapeVal[3] * inputShapeVal[4]};

        } else {
            _log.nest().trace("Unexpected input shape");
            return mlir::failure();
        }
    }

    const auto normVarianceAttr = mlir::BoolAttr::get(getContext(), normalizeVariance);
    const auto acrossChannelsAttr = mlir::BoolAttr::get(getContext(), acrossChannels);
    const auto epsAttr = getFPAttr(getContext(), eps);

    if (newInShape.size() == 4) {
        const auto origLoc = origOp->getLoc();
        auto reshapeInput = rewriter.create<IE::ReshapeOp>(origLoc, origOp.getInput(), nullptr, false,
                                                           getIntArrayAttr(getContext(), newInShape));

        auto mvnOp = rewriter.create<IE::MVNOp>(origLoc, reshapeInput.getOutput(), acrossChannelsAttr, normVarianceAttr,
                                                epsAttr);

        auto reshapeOut = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, mvnOp.getOutput(), nullptr, false,
                                                                     getIntArrayAttr(getContext(), inputShapeVal));

        if (transposeIn) {
            const auto transposeOrder = mlir::AffineMapAttr::get(
                    mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 3, 1, 2}, getContext()));
            const auto transposeLoc = appendLoc(origLoc, "transpose_mvn_out");
            rewriter.setInsertionPointAfter(reshapeOut);
            auto transposeOut =
                    rewriter.create<IE::TransposeOp>(transposeLoc, reshapeOut.getOutput(), nullptr, transposeOrder);
            reshapeOut.getOutput().replaceAllUsesExcept(transposeOut.getOutput(), transposeOut);
        }
        return mlir::success();
    } else {
        _log.nest().trace("MVN6 -> MVN1 conversion pass not applied");
        return mlir::failure();
    }
}

//
// ConvertMVN6ToMVN1Pass
//

class ConvertMVN6ToMVN1Pass final : public IE::impl::ConvertMVN6ToMVN1Base<ConvertMVN6ToMVN1Pass> {
public:
    explicit ConvertMVN6ToMVN1Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertMVN6ToMVN1Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertMVN6ToMVN1>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertMVN6ToMVN1Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMVN6ToMVN1Pass(Logger log) {
    return std::make_unique<ConvertMVN6ToMVN1Pass>(log);
}
