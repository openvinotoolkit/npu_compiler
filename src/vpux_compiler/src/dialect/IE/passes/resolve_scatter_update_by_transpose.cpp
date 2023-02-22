//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {
//
// Resolve scatter_update by transpose pass
//
class ResolveScatterUpdateByTransposePass final :
        public IE::ResolveScatterUpdateByTransposeBase<ResolveScatterUpdateByTransposePass> {
public:
    explicit ResolveScatterUpdateByTransposePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class TransposePlanning;

private:
    void safeRunOnFunc() final;
};

//
// TransposePlanning
//

class ResolveScatterUpdateByTransposePass::TransposePlanning final :
        public mlir::OpRewritePattern<IE::ScatterUpdateOp> {
public:
    TransposePlanning(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ScatterUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ResolveScatterUpdateByTransposePass::TransposePlanning::matchAndRewrite(
        IE::ScatterUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::ScatterUpdate Operation '{0}'", origOp->getLoc());
    IE::LayerOpInterface newOp;

    const auto inType = origOp.input().getType().cast<NDTypeInterface>();
    const auto inputShape = inType.getShape();

    const auto axis = origOp.axis_value().getValue();
    const unsigned int adjustedAxisValue = static_cast<unsigned int>((axis + inputShape.size()) % inputShape.size());

    if (adjustedAxisValue == 0) {
        const auto axisAttr = getIntAttr(rewriter.getContext(), 0);
        rewriter.replaceOpWithNewOp<IE::ScatterUpdateOp>(origOp, origOp.input(), origOp.indices(), origOp.updates(),
                                                         nullptr, axisAttr);

        return mlir::success();
    } else {
        auto transposeInputOrder = SmallVector<unsigned int>(inputShape.size(), 0);
        unsigned int vectorSize = static_cast<unsigned int>(inputShape.size());
        unsigned int refVal = 0;
        transposeInputOrder[0] = adjustedAxisValue;
        for (unsigned int i = 1; i < vectorSize; i++) {
            if (refVal == adjustedAxisValue) {
                refVal++;
            }
            transposeInputOrder[i] = refVal;
            refVal++;
        }

        const auto updatesType = origOp.updates().getType().cast<NDTypeInterface>();
        const auto updatesShape = updatesType.getShape();
        const auto indicesType = origOp.indices().getType().cast<NDTypeInterface>();

        const auto indicesSize = static_cast<unsigned int>(indicesType.getShape().size());
        const auto updatesSize = static_cast<unsigned int>(updatesShape.size());

        auto transposeUpdatesOrder = SmallVector<unsigned int>(updatesSize, 0);
        transposeUpdatesOrder[0] = adjustedAxisValue;
        for (unsigned int i = 1; i < indicesSize; i++) {
            transposeUpdatesOrder[i] = transposeUpdatesOrder[i - 1] + 1;
        }
        transposeUpdatesOrder[indicesSize] = 0;
        for (unsigned int i = indicesSize + 1; i < indicesSize + adjustedAxisValue; i++) {
            transposeUpdatesOrder[i] = transposeUpdatesOrder[i - 1] + 1;
        }
        for (unsigned int i = indicesSize + adjustedAxisValue; i < updatesSize; i++) {
            transposeUpdatesOrder[i] = i;
        }

        const auto orderInputAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transposeInputOrder, origOp->getContext()));
        const auto orderUpdatesAttr = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(transposeUpdatesOrder, origOp->getContext()));

        auto transposedUpdates =
                rewriter.create<IE::TransposeOp>(origOp->getLoc(), origOp.updates(), nullptr, orderUpdatesAttr);
        auto transposedInput =
                rewriter.create<IE::TransposeOp>(origOp->getLoc(), origOp.input(), nullptr, orderInputAttr);

        const auto axisAttr = getIntAttr(rewriter.getContext(), 0);
        auto outputOrig =
                rewriter.create<IE::ScatterUpdateOp>(origOp->getLoc(), transposedInput.output(), origOp.indices(),
                                                     transposedUpdates.output(), nullptr, axisAttr);

        auto transposeOutputOrder = SmallVector<unsigned int>(inputShape.size(), 0);
        unsigned int refVal2 = 1;
        transposeOutputOrder[adjustedAxisValue] = 0;
        for (int64_t i = 0; i < vectorSize; i++) {
            if (i != adjustedAxisValue) {
                transposeOutputOrder[i] = refVal2;
                refVal2++;
            }
        }

        const auto orderOutputAttr = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(transposeOutputOrder, origOp->getContext()));
        rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, outputOrig.output(), nullptr, orderOutputAttr);

        return mlir::success();
    }
}

//
// safeRunOnFunc
//

void ResolveScatterUpdateByTransposePass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalOp = [&](IE::ScatterUpdateOp scatterOp) {
        VPUX_THROW_UNLESS(scatterOp.axis_value().hasValue(), "axis_value is null");
        const auto axis = scatterOp.axis_value().getValue();
        return axis == 0;
    };

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::ScatterUpdateOp>(isLegalOp);
    target.addLegalOp<IE::TransposeOp>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<ResolveScatterUpdateByTransposePass::TransposePlanning>(&ctx, _log);

    auto func = getFunction();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createResolveScatterUpdateByTransposePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createResolveScatterUpdateByTransposePass(Logger log) {
    return std::make_unique<ResolveScatterUpdateByTransposePass>(log);
}
