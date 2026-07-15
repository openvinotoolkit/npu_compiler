//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

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
        if (layerWithPostOp.hasPPE()) {
            return mlir::failure();
        }
    }

    // Determine the shape that will replace the TileOp's output in the eltwise after fold:
    //   - large single channel: replace with the original TileOp input (e.g. 1x1xHxW)
    //   - scalar: replace with an all-ones shape of the same rank as outputValue
    const auto replacementShape =
            hasLargeSingleChannelInput
                    ? to_small_vector(getShape(origOp.getInput()))
                    : SmallVector<int64_t>(mlir::cast<vpux::NDTypeInterface>(outputValue.getType()).getRank(), 1);

    // Verify that substituting replacementShape preserves the eltwise output shape.
    const auto existingOutShape = to_small_vector(getShape(outputUserOp->getResult(0)));
    const auto lhsIsReplaced = outputUserOp->getOperand(0) == outputValue;
    const auto newLhsShape = lhsIsReplaced ? replacementShape : to_small_vector(getShape(outputUserOp->getOperand(0)));
    const auto newRhsShape = lhsIsReplaced ? to_small_vector(getShape(outputUserOp->getOperand(1))) : replacementShape;
    const auto broadCastType = getBroadCastType(outputUserOp);
    const auto newOutShape = IE::broadcastEltwiseShape(ArrayRef<int64_t>(newLhsShape), ArrayRef<int64_t>(newRhsShape),
                                                       broadCastType, outputUserOp->getLoc());
    if (mlir::failed(newOutShape) || newOutShape.value() != existingOutShape) {
        return mlir::failure();
    }

    _log.trace("Folding TileOp at '{0}'", origOp.getLoc());

    if (hasLargeSingleChannelInput) {
        rewriter.replaceOp(origOp, origOp.getInput());
        return mlir::success();
    }

    auto newReshapeOp = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origOp.getInput(),
                                                             getIntArrayAttr(ctx, replacementShape));
    rewriter.replaceAllUsesWith(outputValue, newReshapeOp);
    return mlir::success();
}

//
// FuseTileConvertRewrite
//
// Pattern: Convert -> Tile -> Convert
//
// Benefits:
// 1. Reduces data size for Tile DMA
// 2. Convert operates on smaller tensor before Tile expansion
//

class FuseTileConvertRewrite final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    FuseTileConvertRewrite(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("FuseTileConvertRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult FuseTileConvertRewrite::matchAndRewrite(IE::ConvertOp convertOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseTileConvertRewrite: Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    // Check if input is from TileOp
    auto tileOp = convertOp.getInput().getDefiningOp<IE::TileOp>();
    if (tileOp == nullptr) {
        nestedLogger.trace("ConvertOp does not have TileOp input");
        return mlir::failure();
    }

    if (!tileOp.getResult().hasOneUse()) {
        nestedLogger.trace("TileOp has multiple users");
        return mlir::failure();
    }

    // Check if TileOp's input is from another ConvertOp
    auto prevConvertOp = tileOp.getInput().getDefiningOp<IE::ConvertOp>();
    if (prevConvertOp == nullptr) {
        nestedLogger.trace("TileOp does not have ConvertOp input");
        return mlir::failure();
    }

    if (!prevConvertOp.getResult().hasOneUse()) {
        nestedLogger.trace("Previous ConvertOp has multiple users");
        return mlir::failure();
    }

    // Get the final destination element type from the second ConvertOp
    const auto finalDstElemType = convertOp.getDstElemType();

    // Check if the optimization reduces data size for Tile operation
    // Original: Convert(A->B) -> Tile(B) -> Convert(B->C)
    // Optimized: Convert(A->C) -> Tile(C)
    // Only beneficial when sizeof(C) <= sizeof(B), i.e., Tile operates on smaller or equal data
    const auto tileInputType = mlir::cast<vpux::NDTypeInterface>(tileOp.getInput().getType());
    const auto intermediateElemBitWidth = tileInputType.getElemTypeSize().count();
    const auto finalElemBitWidth = vpux::getElemTypeSize(finalDstElemType).count();

    if (finalElemBitWidth > intermediateElemBitWidth) {
        nestedLogger.trace("Optimization would increase Tile data size: intermediate {0} bits -> final {1} bits",
                           intermediateElemBitWidth, finalElemBitWidth);
        return mlir::failure();
    }

    // Get the original input to the first ConvertOp
    auto originalInput = prevConvertOp.getInput();

    auto newConvertOp = rewriter.create<IE::ConvertOp>(prevConvertOp.getLoc(), originalInput,
                                                       mlir::TypeAttr::get(finalDstElemType));

    auto tileOutType = mlir::cast<vpux::NDTypeInterface>(tileOp.getOutput().getType());
    auto newTileOutType = mlir::cast<mlir::RankedTensorType>(tileOutType.changeElemType(finalDstElemType));
    auto newTileOp = rewriter.create<IE::TileOp>(tileOp.getLoc(), newTileOutType, newConvertOp.getOutput(), nullptr,
                                                 tileOp.getRepeatsValuesAttr());

    rewriter.replaceOp(convertOp, newTileOp.getOutput());

    nestedLogger.trace("Successfully fused Convert -> Tile -> Convert pattern");
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
    patterns.add<FuseTileConvertRewrite>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createOptimizeTileOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeTileOpPass(Logger log) {
    return std::make_unique<OptimizeTileOpPass>(log);
}
