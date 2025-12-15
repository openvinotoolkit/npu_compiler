//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZETILEOP
#define GEN_PASS_DEF_OPTIMIZETILEOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

IE::AutoBroadcastType getBroadCastType(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, IE::AutoBroadcastType>(op)
            .Case<IE::MultiplyOp>([&](auto multiply) {
                return multiply.getAutoBroadcast();
            })

            .Case<IE::AddOp>([&](auto add) {
                return add.getAutoBroadcast();
            })
            .Default([&](auto) -> IE::AutoBroadcastType {
                VPUX_THROW("Unexpected operation type at '{0}'", op);
            });
}

class FoldTileOpRewriter final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    FoldTileOpRewriter(mlir::MLIRContext* ctx, const Logger& log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("FoldTileOpRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult FoldTileOpRewriter::matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto origInputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto origInputShape = origInputType.getShape();

    // If the tile op is used as input for op like eltwise or multiply, and its size is too big to fit into CMX, which
    // means that the op will be tiled into multiple small ones. And it will cost lots of time before executing the
    // eltwise/multiply op. So it will be performant if it can be fused into the post op.
    auto hasLargeSingleChannelInput = origInputType.getTotalAllocSize() > vpux::VPU::getTotalCMXSize(origOp) &&
                                      origInputShape.size() == 4 && origInputShape[Dims4D::Act::C] == 1;

    if (origInputShape.totalSize() != 1 && !hasLargeSingleChannelInput) {
        return mlir::failure();
    }

    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto isFoldableViewOp = [](mlir::Operation* viewOp) {
        if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp, IE::ShapeCastOp>(viewOp)) {
            return false;
        }
        if (!viewOp->hasOneUse()) {
            return false;
        }
        return true;
    };

    auto outputValue = mlir::cast<mlir::Value>(origOp.getOutput());
    auto outputUserOp = *(outputValue.getUsers().begin());
    while (isFoldableViewOp(outputUserOp)) {
        outputValue = outputUserOp->getResult(0);
        outputUserOp = *(outputValue.getUsers().begin());
    }

    // For the large single channel input, don't fold TileOp if the output is used by FoldableViewOp, since the compiler
    // may not be able to back infer the new output shape
    auto hasFoldableUser = isFoldableViewOp(*origOp->getUsers().begin());
    if (hasLargeSingleChannelInput && hasFoldableUser) {
        return mlir::failure();
    }

    // More ops which support auto broadcast may also apply here!
    if (!mlir::isa_and_nonnull<IE::MultiplyOp, IE::AddOp>(outputUserOp)) {
        return mlir::failure();
    }

    // Can't fold TileOp if the layer has precision convert like from fp16 to fp32
    auto userInType = mlir::cast<vpux::NDTypeInterface>(outputUserOp->getOperand(0).getType());
    auto userOutType = mlir::cast<vpux::NDTypeInterface>(outputUserOp->getResult(0).getType());
    if (userInType.getElementType() != userOutType.getElementType()) {
        return mlir::failure();
    }

    // Can't fold TileOp if the layer has post operation
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(outputUserOp)) {
        if (layerWithPostOp.getPostOp() != nullptr) {
            return mlir::failure();
        }
    }

    _log.trace("Folding TileOp at '{0}'", origOp.getLoc());

    if (hasLargeSingleChannelInput) {
        auto tileOutShape = getShape(origOp.getOutput());

        auto lhsIsTileOp = outputUserOp->getOperand(0).getDefiningOp() == origOp;
        auto lhsShape = lhsIsTileOp ? tileOutShape : getShape(outputUserOp->getOperand(0));
        auto rhsShape = lhsIsTileOp ? getShape(outputUserOp->getOperand(1)) : tileOutShape;
        auto broadCastType = getBroadCastType(outputUserOp);
        const auto outShape = IE::broadcastEltwiseShape(lhsShape, rhsShape, broadCastType, outputUserOp->getLoc());
        if (mlir::failed(outShape)) {
            return mlir::failure();
        }

        rewriter.replaceAllUsesWith(origOp, origOp.getInput());
        return mlir::success();
    }

    auto newShape = SmallVector<int64_t>(mlir::cast<vpux::NDTypeInterface>(outputValue.getType()).getRank(), 1);
    auto newReshapeOp = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origOp.getInput(), nullptr, false,
                                                             getIntArrayAttr(ctx, newShape));

    outputValue.replaceAllUsesWith(newReshapeOp);

    return mlir::success();
}

//
// OptimizeTileOpPass
//

class OptimizeTileOpPass final : public IE::impl::OptimizeTileOpBase<OptimizeTileOpPass> {
public:
    explicit OptimizeTileOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeTileOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldTileOpRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeTileOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeTileOpPass(Logger log) {
    return std::make_unique<OptimizeTileOpPass>(log);
}
