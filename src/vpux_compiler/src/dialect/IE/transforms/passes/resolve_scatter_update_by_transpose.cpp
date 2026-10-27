//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_RESOLVESCATTERUPDATEBYTRANSPOSE
#define GEN_PASS_DEF_RESOLVESCATTERUPDATEBYTRANSPOSE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertScatterUpdateOp(IE::ScatterUpdateOp scatterOp) {
    VPUX_THROW_UNLESS(scatterOp.getAxisValue().has_value(), "axis_value is null");
    const auto axis = scatterOp.getAxisValue().value();
    return axis != 0;
}

//
// Resolve scatter_update by transpose pass
//
class ResolveScatterUpdateByTransposePass final :
        public IE::impl::ResolveScatterUpdateByTransposeBase<ResolveScatterUpdateByTransposePass> {
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
    if (!shouldConvertScatterUpdateOp(origOp)) {
        return mlir::failure();
    }

    _log.trace("Found IE::ScatterUpdate Operation '{0}'", origOp->getLoc());
    IE::LayerOpInterface newOp;

    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inType.getShape();

    const auto axis = origOp.getAxisValue().value();
    const unsigned int adjustedAxisValue = static_cast<unsigned int>((axis + inputShape.size()) % inputShape.size());

    if (adjustedAxisValue == 0) {
        const auto axisAttr = getIntAttr(rewriter.getContext(), 0);
        rewriter.replaceOpWithNewOp<IE::ScatterUpdateOp>(origOp, origOp.getInput(), origOp.getIndices(),
                                                         origOp.getUpdates(), nullptr, axisAttr);

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

        const auto updatesType = mlir::cast<vpux::NDTypeInterface>(origOp.getUpdates().getType());
        const auto updatesShape = updatesType.getShape();
        const auto indicesType = mlir::cast<vpux::NDTypeInterface>(origOp.getIndices().getType());

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

        auto transposedUpdates = IE::TransposeOp::create(rewriter, takeOpLoc(origOp, "transposed_updates"),
                                                         origOp.getUpdates(), nullptr, orderUpdatesAttr);
        auto transposedInput = IE::TransposeOp::create(rewriter, takeOpLoc(origOp, "transposed_input"),
                                                       origOp.getInput(), nullptr, orderInputAttr);

        const auto axisAttr = getIntAttr(rewriter.getContext(), 0);
        auto outputOrig =
                IE::ScatterUpdateOp::create(rewriter, takeOpLoc(origOp, "scatter_out"), transposedInput.getOutput(),
                                            origOp.getIndices(), transposedUpdates.getOutput(), nullptr, axisAttr);

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
        auto transposeOp =
                rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, outputOrig.getOutput(), nullptr, orderOutputAttr);
        extendOpLoc(transposeOp, "transpose_out");

        return mlir::success();
    }
}

//
// safeRunOnFunc
//

void ResolveScatterUpdateByTransposePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<ResolveScatterUpdateByTransposePass::TransposePlanning>(&ctx, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}
}  // namespace

//
// createResolveScatterUpdateByTransposePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createResolveScatterUpdateByTransposePass(Logger log) {
    return std::make_unique<ResolveScatterUpdateByTransposePass>(log);
}
