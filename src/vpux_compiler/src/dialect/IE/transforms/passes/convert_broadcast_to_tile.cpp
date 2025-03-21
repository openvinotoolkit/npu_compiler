//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTBROADCASTTOTILE
#define GEN_PASS_DEF_CONVERTBROADCASTTOTILE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertBroadcastToTile
//

template <class ConcreteOp>
class ConvertBroadcastToTile final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvertBroadcastToTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ConvertBroadcastToTile");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvertBroadcastToTile<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    const auto inputShape = to_small_vector(getShape(origOp.getInput()));
    const auto outputShape = to_small_vector(getShape(origOp.getOutput()));
    const auto broadcastType = origOp.getMode().value_or(IE::BroadcastType::NUMPY);
    SmallVector<int64_t> broadcastAxes;

    VPUX_THROW_UNLESS(inputShape.size() <= outputShape.size(), "Broadcast input rank {0} exceeds output rank {1}",
                      inputShape.size(), outputShape.size());

    const auto inTy = origOp.getInput().getType().template cast<vpux::NDTypeInterface>();
    const auto outTy = origOp.getOutput().getType().template cast<vpux::NDTypeInterface>();

    auto inputShapeBounded = inputShape;
    auto outputShapeBounded = outputShape;

    auto updateBoundedShapeIfDynamic = [](const auto& type, auto& boundedShape, const auto& errorMsg) {
        if (type.getShape().isDynamic()) {
            const auto boundedType = type.template cast<vpux::BoundedTypeInterface>();
            const auto bounds = boundedType.getBounds();

            VPUX_THROW_WHEN(bounds == nullptr, errorMsg);

            boundedShape = parseIntArrayAttr<int64_t>(bounds);
        }
    };

    updateBoundedShapeIfDynamic(inTy, inputShapeBounded,
                                "ConvertBroadcastToTile: Missed bounds for input with dynamic dims");
    updateBoundedShapeIfDynamic(outTy, outputShapeBounded,
                                "ConvertBroadcastToTile: Missed bounds for output with dynamic dims");

    // Finds the axes over which the broadcasting rules apply. For example:
    // NUMPY and BIDIRECTIONAL: input 16x1x1, output 1x16x50x50 will return the axes [0, 2, 3]
    // EXPLICIT:                input 16, output 1x16x50x50, axesMapping 1 will return the axes [0, 2, 3]
    if (broadcastType == IE::BroadcastType::BIDIRECTIONAL || broadcastType == IE::BroadcastType::NUMPY) {
        broadcastAxes = vpux::IE::getBroadcastAxesNumpyBidirectional(inputShapeBounded, outputShapeBounded);
    } else if (broadcastType == IE::BroadcastType::EXPLICIT) {
        auto axesMapping = IE::constInputToData(origOp.getLoc(), origOp.getAxesMapping()).value();
        broadcastAxes = vpux::IE::getBroadcastAxesExplicit(axesMapping, outputShape);
    }

    auto adjustedInputShape = inputShape;
    auto adjustedInputBoundedShape = std::move(inputShapeBounded);
    for (const auto& axis : broadcastAxes) {
        if (adjustedInputShape.size() < outputShape.size()) {
            adjustedInputShape.insert(adjustedInputShape.begin() + axis, 1);
            adjustedInputBoundedShape.insert(adjustedInputBoundedShape.begin() + axis, 1);
        }
    }

    const auto adjustedInputShapeAttr = getIntArrayAttr(origOp->getContext(), adjustedInputShape);
    if (mlir::isa_and_nonnull<IE::BroadcastOp>(origOp)) {
        SmallVector<int32_t> repeats(outputShape.size());
        for (size_t i = 0; i < repeats.size(); ++i) {
            repeats[i] = checked_cast<int32_t>(outputShapeBounded[i] / adjustedInputBoundedShape[i]);
        }

        const auto dataType = mlir::RankedTensorType::get({checked_cast<int64_t>(repeats.size())},
                                                          getSInt32Type(origOp->getContext()));
        const auto repeatsConstOp =
                Const::createConst(rewriter, takeOpLoc(origOp, "n_repeats"), dataType, ArrayRef(repeats));

        auto reshapeInputOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, adjustedInputShapeAttr);

        rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), reshapeInputOp.getOutput(), repeatsConstOp,
                                                nullptr /*repeats_value*/);
    } else {
        const auto adjustedInputShapeRank = checked_cast<int64_t>(adjustedInputShape.size());
        const auto dataType =
                mlir::RankedTensorType::get({adjustedInputShapeRank}, getSInt32Type(origOp->getContext()));
        auto inputShapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(adjustedInputShape), -1);

        const auto shapeTensor = Const::createConst(rewriter, origOp->getLoc(), dataType, ArrayRef(inputShapeValues));

        const auto outputShapeAttr = getIntArrayAttr(origOp->getContext(), outputShape);
        const auto outputBoundedShapeAttr = mlir::cast<IE::DynamicBroadcastOp>(*origOp).getOutputBoundsAttr();

        // In case of dynamic shapes the repeats can't be calculated properly at this stage, since all we currently
        // have are upper bounds, which do not give the full picture. We will be able to get real shapes later, in
        // the SW kernel.
        // But we can try to save the useful information we have - target_shape - and distribute it further.
        if (getShape(origOp.getInput()).isDynamic() || Shape(adjustedInputShape).isDynamic()) {
            const auto adjustedInputBoundedShapeAttr = getIntArrayAttr(origOp->getContext(), adjustedInputBoundedShape);

            auto reshapeInputOp = rewriter.create<IE::DynamicReshapeOp>(
                    takeOpLoc(origOp, "dreshape_in"), origOp.getInput(), shapeTensor, adjustedInputShapeAttr,
                    adjustedInputBoundedShapeAttr);
            rewriter.replaceOpWithNewOp<IE::DynamicTileOp>(
                    origOp, origOp.getType(),
                    adjustedInputShape != inputShape ? reshapeInputOp.getOutput() : origOp.getInput(),
                    origOp.getTargetShape(), nullptr /*repeats*/, nullptr /*repeats_value*/, outputShapeAttr,
                    outputBoundedShapeAttr);
        } else {
            auto reshapeInputOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                                 nullptr, false, adjustedInputShapeAttr);
            rewriter.replaceOpWithNewOp<IE::DynamicTileOp>(
                    origOp, origOp.getType(), reshapeInputOp.getOutput(), origOp.getTargetShape(), nullptr /*repeats*/,
                    nullptr /*repeats_value*/, outputShapeAttr, outputBoundedShapeAttr);
        }
    }

    return mlir::success();
}

//
// ConvertBroadcastToTilePass
//

class ConvertBroadcastToTilePass final : public IE::impl::ConvertBroadcastToTileBase<ConvertBroadcastToTilePass> {
public:
    explicit ConvertBroadcastToTilePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertBroadcastToTilePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    target.addIllegalOp<IE::BroadcastOp>();
    target.addIllegalOp<IE::DynamicBroadcastOp>();

    target.addLegalOp<Const::DeclareOp>();

    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::DynamicReshapeOp>();

    target.addLegalOp<IE::TileOp>();
    target.addLegalOp<IE::DynamicTileOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertBroadcastToTile<IE::BroadcastOp>>(&ctx, _log);
    patterns.add<ConvertBroadcastToTile<IE::DynamicBroadcastOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet tilePatterns(&ctx);
    IE::TileOp::getCanonicalizationPatterns(tilePatterns, &ctx);
    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(tilePatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBroadcastToTilePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBroadcastToTilePass(Logger log) {
    return std::make_unique<ConvertBroadcastToTilePass>(log);
}
