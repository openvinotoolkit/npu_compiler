//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>

using namespace vpux;

namespace {

//
// ConvertBroadcastToTile
//

class ConvertBroadcastToTile final : public mlir::OpRewritePattern<IE::BroadcastOp> {
public:
    ConvertBroadcastToTile(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::BroadcastOp>(ctx), _log(log) {
        setDebugName("ConvertBroadcastToTile");
    }

    mlir::LogicalResult matchAndRewrite(IE::BroadcastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

SmallVector<int64_t> getBroadcastAxesNumpyBidirectional(const SmallVector<int64_t>& inputShape,
                                                        const SmallVector<int64_t>& outputShape) {
    SmallVector<int64_t> broadcastAxes;
    const auto startAxis = static_cast<int64_t>(outputShape.size()) - static_cast<int64_t>(inputShape.size());
    VPUX_THROW_UNLESS(startAxis >= 0, "Broadcast axes not known deterministically");
    for (int64_t i = 0; i < static_cast<int64_t>(outputShape.size()); i++) {
        if (i < startAxis || outputShape[i] != inputShape[i - startAxis]) {
            broadcastAxes.push_back(i);
        }
    }
    return broadcastAxes;
}

SmallVector<int64_t> getBroadcastAxesExplicit(const SmallVector<int64_t>& axesMapping,
                                              const SmallVector<int64_t>& outputShape) {
    SmallVector<int64_t> broadcastAxes(outputShape.size());
    std::iota(broadcastAxes.begin(), broadcastAxes.end(), 0);
    for (auto i = axesMapping.rbegin(); i != axesMapping.rend(); ++i) {
        broadcastAxes.erase(broadcastAxes.begin() + *i);
    }
    return broadcastAxes;
}

mlir::LogicalResult ConvertBroadcastToTile::matchAndRewrite(IE::BroadcastOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto inputShape = to_small_vector(getShape(origOp.input()));
    const auto outputShape = to_small_vector(getShape(origOp.output()));
    const auto broadcastType = origOp.mode().getValueOr(IE::BroadcastType::NUMPY);
    SmallVector<int64_t> broadcastAxes;

    VPUX_THROW_UNLESS(inputShape.size() <= outputShape.size(), "Broadcast input rank {0} exceeds output rank {1}",
                      inputShape.size(), outputShape.size());

    // Finds the axes over which the broadcasting rules apply. For example:
    // NUMPY and BIDIRECTIONAL: input 16x1x1, output 1x16x50x50 will return the axes [0, 2, 3]
    // EXPLICIT:                input 16, output 1x16x50x50, axesMapping 1 will return the axes [0, 2, 3]
    if (broadcastType == IE::BroadcastType::BIDIRECTIONAL || broadcastType == IE::BroadcastType::NUMPY) {
        broadcastAxes = getBroadcastAxesNumpyBidirectional(inputShape, outputShape);
    } else if (broadcastType == IE::BroadcastType::EXPLICIT) {
        auto axesMapping = IE::constInputToData(origOp.getLoc(), origOp.axes_mapping()).getValue();
        broadcastAxes = getBroadcastAxesExplicit(axesMapping, outputShape);
    }

    auto adjustedInputShape = inputShape;
    for (const auto& axis : broadcastAxes) {
        if (adjustedInputShape.size() < outputShape.size()) {
            adjustedInputShape.insert(adjustedInputShape.begin() + axis, 1);
        }
    }

    SmallVector<int32_t> repeats(outputShape.size());
    for (size_t i = 0; i < repeats.size(); ++i) {
        repeats[i] = static_cast<int32_t>(outputShape[i] / adjustedInputShape[i]);
    }

    const auto dataType =
            mlir::RankedTensorType::get({static_cast<int64_t>(repeats.size())}, getSInt32Type(getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataType, makeArrayRef(repeats));
    auto repeatsConstOp =
            rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataType, Const::ContentAttr::get(dataAttr));

    const auto adjustedInputShapeAttr = getIntArrayAttr(getContext(), adjustedInputShape);
    auto reshapeInputOp =
            rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, adjustedInputShapeAttr);

    rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), reshapeInputOp.output(), repeatsConstOp);

    return mlir::success();
}

//
// ConvertBroadcastToTilePass
//

class ConvertBroadcastToTilePass final : public IE::ConvertBroadcastToTileBase<ConvertBroadcastToTilePass> {
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
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::TileOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertBroadcastToTile>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
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
