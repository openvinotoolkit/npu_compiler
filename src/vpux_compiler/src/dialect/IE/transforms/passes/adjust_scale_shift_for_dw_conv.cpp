//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/scale_shift_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTSCALESHIFTFORDWCONV
#define GEN_PASS_DEF_ADJUSTSCALESHIFTFORDWCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Approximate of N from experiment when need to make adjustments to scaleShift
static const int32_t MAX_SCALE_SHIFT_N = 16;

mlir::LogicalResult mergeNCAndRewrite(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc,
                                      IE::ScaleShiftOp origScaleShiftOp) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    auto activation = origScaleShiftOp.getInput();
    auto origOutShape = getShape(origScaleShiftOp->getResult(0));

    auto getNewValue = [&](mlir::Value origValue) {
        if (origValue == nullptr) {
            return origValue;
        }

        auto broadcastShape = Shape({origOutShape[N], origOutShape[C], 1, 1});
        auto reshapeShape = Shape({1, origOutShape[N] * origOutShape[C], 1, 1});

        auto broadcastOp = IE::createBroadcast(rewriter, appendLoc(loc, "_broadcast"), origValue, broadcastShape);

        return rewriter.createOrFold<IE::ReshapeOp>(loc, broadcastOp, nullptr, false,
                                                    getIntArrayAttr(ctx, ShapeRef(reshapeShape)));
    };

    auto activationReshapeShape = Shape({1, origOutShape[N] * origOutShape[C], origOutShape[H], origOutShape[W]});
    auto activationReshapeOp = rewriter.createOrFold<IE::ReshapeOp>(loc, activation, nullptr, false,
                                                                    getIntArrayAttr(ctx, activationReshapeShape));
    auto scaleShiftOp =
            rewriter.create<IE::ScaleShiftOp>(loc, activationReshapeOp, getNewValue(origScaleShiftOp.getWeights()),
                                              getNewValue(origScaleShiftOp.getBiases()));

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origScaleShiftOp, scaleShiftOp.getOutput(), nullptr, false,
                                               getIntArrayAttr(ctx, origOutShape));

    return mlir::success();
}

//
// AdjustScaleShiftForDWConvPass
//

class AdjustScaleShiftForDWConvPass final :
        public IE::impl::AdjustScaleShiftForDWConvBase<AdjustScaleShiftForDWConvPass> {
public:
    explicit AdjustScaleShiftForDWConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ScaleShiftOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ScaleShiftOpConverter
//

class AdjustScaleShiftForDWConvPass::ScaleShiftOpConverter final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    ScaleShiftOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScaleShiftOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustScaleShiftForDWConvPass::ScaleShiftOpConverter::matchAndRewrite(
        IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto inputShape = getShape(origOp.getInput());
    if (inputShape.size() != 4 || inputShape[Dims4D::Act::N] == 1) {
        return mlir::failure();
    }

    if (mlir::failed(vpux::IE::isBeneficialConvertScaleShiftToDW(origOp, _log))) {
        return mlir::failure();
    }

    // If the Weights/Biases of ScaleShiftOp is Constant and with single dense value
    // BroadCast weight can be finished at compile stage. No additional time overhead.
    // Otherwise, It will introduce a Tile Op.
    // The experiment show there is no benefits when batch size < MAX_SCALE_SHIFT_N
    if (!VPU::isNullOrConstWithSingleValue(origOp.getWeights()) && inputShape[Dims4D::Act::N] < MAX_SCALE_SHIFT_N) {
        _log.trace("No benefit due to Weights is not splat constant and N smaller than '{0}'", MAX_SCALE_SHIFT_N);
        return mlir::failure();
    }

    if (!VPU::isNullOrConstWithSingleValue(origOp.getBiases()) && inputShape[Dims4D::Act::N] < MAX_SCALE_SHIFT_N) {
        _log.trace("No benefit due to Biases is not splat constant and N smaller than '{0}'", MAX_SCALE_SHIFT_N);
        return mlir::failure();
    }

    // For avoiding vast Convolution pieces, merge input's N and C
    if (mlir::failed(mergeNCAndRewrite(rewriter, origOp.getContext(), origOp.getLoc(), origOp))) {
        return mlir::failure();
    }
    _log.nest().trace("Adjust input and weights shape of curOp for converting DW Convolution.");

    return mlir::success();
}

//
// safeRunOnFunc
//

void AdjustScaleShiftForDWConvPass::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ScaleShiftOpConverter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createAdjustScaleShiftForDWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustScaleShiftForDWConvPass(Logger log) {
    return std::make_unique<AdjustScaleShiftForDWConvPass>(log);
}
