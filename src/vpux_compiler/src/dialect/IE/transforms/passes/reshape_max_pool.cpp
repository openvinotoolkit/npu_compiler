//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_RESHAPEMAXPOOL
#define GEN_PASS_DEF_RESHAPEMAXPOOL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ReshapeMaxPoolPass
//

class ReshapeMaxPoolPass final : public IE::impl::ReshapeMaxPoolBase<ReshapeMaxPoolPass> {
public:
    explicit ReshapeMaxPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// MaxPoolConverter
//

class MaxPoolConverter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        this->setDebugName("MaxPoolConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolConverter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto inShape = getShape(origOp.getInput());
    if (inShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }
    if (inShape.size() != 4) {
        return mlir::failure();
    }
    if (inShape[Dims4D::Act::W] != 1) {
        return mlir::failure();
    }
    const auto outShape = getShape(origOp.getOutput());
    if (outShape.size() != 4) {
        return mlir::failure();
    }
    const auto kernel = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    if (kernel.back() != 1) {
        return mlir::failure();
    }
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    if (strides.back() != 1) {
        return mlir::failure();
    }

    int64_t divisor = 1;
    if (inShape[Dims4D::Act::C] % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0) {
        for (int64_t i = inShape[Dims4D::Act::C] / 2; i > 2; i--) {
            if ((i < VPU::NCEInvariant::VPU_DIMENSION_LIMIT) && (inShape[Dims4D::Act::C] % i == 0) &&
                ((inShape[Dims4D::Act::C] / i) % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0)) {
                divisor = i;
                break;
            }
        }
    } else {
        for (int64_t i = inShape[Dims4D::Act::C] / 2; i > 2; i--) {
            if ((i < VPU::NCEInvariant::VPU_DIMENSION_LIMIT) && (inShape[Dims4D::Act::C] % i == 0)) {
                divisor = i;
                break;
            }
        }
    }
    if (divisor == 1 || inShape[Dims4D::Act::C] % divisor != 0) {
        return mlir::failure();
    }

    auto newInputShape = {
            inShape[Dims4D::Act::N],
            inShape[Dims4D::Act::C] / divisor,
            inShape[Dims4D::Act::W] * divisor,
            inShape[Dims4D::Act::H],
    };

    auto ctx = origOp.getContext();
    const auto inputShapeAttr = getIntArrayAttr(ctx, newInputShape);
    const SmallVector<unsigned> order = {0, 1, 3, 2};
    auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(order, ctx));
    auto transposeIn = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "transpose_in"), origOp.getInput(),
                                                        nullptr, orderAttr);
    auto reshapeIn = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "reshape_in"), transposeIn.getOutput(),
                                                    nullptr, false, inputShapeAttr);

    const auto newKernel = getIntArrayAttr(ctx, SmallVector<int64_t>{kernel[1], kernel[0]});
    const auto newStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{strides[1], strides[0]});
    auto maxpool = rewriter.create<IE::MaxPoolOp>(
            origOp.getLoc(), reshapeIn.getOutput(), newKernel, newStrides, origOp.getPadsBeginAttr(),
            origOp.getPadsEndAttr(), origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    const SmallVector<int64_t> newOutputShape = {
            outShape[Dims4D::Act::N],
            outShape[Dims4D::Act::C],
            outShape[Dims4D::Act::W],
            outShape[Dims4D::Act::H],
    };
    const auto outputShapeAttr = getIntArrayAttr(ctx, newOutputShape);
    auto reshapeOut = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "reshape_out"), maxpool->getResult(0),
                                                     nullptr, false, outputShapeAttr);

    auto transposeOut = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "transpose_out"),
                                                         reshapeOut.getOutput(), nullptr, orderAttr);
    origOp.getOutput().replaceAllUsesWith(transposeOut.getOutput());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ReshapeMaxPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MaxPoolConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createReshapeMaxPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReshapeMaxPoolPass(Logger log) {
    return std::make_unique<ReshapeMaxPoolPass>(log);
}
