//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDEPTH2SPACELAYER
#define GEN_PASS_DEF_CONVERTDEPTH2SPACELAYER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertDepth2SpaceLayerPass
//

class ConvertDepth2SpaceLayerPass final : public IE::impl::ConvertDepth2SpaceLayerBase<ConvertDepth2SpaceLayerPass> {
public:
    explicit ConvertDepth2SpaceLayerPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class Depth2SpaceLayerConverter;

private:
    void safeRunOnFunc() final;
};

class ConvertDepth2SpaceLayerPass::Depth2SpaceLayerConverter final : public mlir::OpRewritePattern<IE::DepthToSpaceOp> {
public:
    Depth2SpaceLayerConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DepthToSpaceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DepthToSpaceOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

/*
 * In this transformation we decompose DepthToSpace operation to the next sequence of ops:
 * Reshape(shapeBegin)->Transpose(order)->Reshape(shapeEnd)
 *
 * if mode equal to blocks_first
 * shapeBegin = [N, blockSize, blockSize, ..., blockSize, C / (blockSize ^ K), D1, D2, ..., DK]
 *
 * if mode equal to depth_first
 * shapeBegin = [N, C / (blockSize ^ K), blockSize, blockSize, ..., blockSize, D1, D2, ..., DK]
 *
 */
mlir::LogicalResult ConvertDepth2SpaceLayerPass::Depth2SpaceLayerConverter::matchAndRewrite(
        IE::DepthToSpaceOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();

    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

    // Check input shape must be 4d out of caution
    // TODO: Can I make sure INPUT format is [N, C, D1, D2, ..., DK] for me ?
    // Actual the below code is relatively general following above format not only for 4d
    VPUX_THROW_UNLESS(inputShape.size() == 4,
                      "Input shape must be 4d (Maybe it's the chance to unlock the restriction)");

    // Calculate Reshape shapeBegin
    const auto spatialDims = inputShape.size() - 2;  // Exclude two dims: N & C
    mlir::SmallVector<int64_t> shapeBegin{inputShape[Dims4D::Act::N]};
    auto C = inputShape[Dims4D::Act::C];
    for (size_t i = 0; i < spatialDims; ++i) {
        shapeBegin.push_back(blockSize);
        C /= blockSize;
    }
    switch (mode) {
    case IE::DepthToSpaceMode::BLOCKS_FIRST:
        shapeBegin.push_back(C);
        break;
    case IE::DepthToSpaceMode::DEPTH_FIRST:
        shapeBegin.insert(shapeBegin.begin() + 1, C);
        break;
    }
    for (size_t i = 0; i < spatialDims; ++i) {
        shapeBegin.push_back(inputShape[Dim(2 + i)]);
    }

    // Calculate Transpose order
    mlir::SmallVector<uint32_t> order{checked_cast<uint32_t>(Dims4D::Act::N.ind())};
    switch (mode) {
    case IE::DepthToSpaceMode::BLOCKS_FIRST:
        order.push_back(checked_cast<uint32_t>(spatialDims + 1));
        for (size_t i = 1; i <= spatialDims; ++i) {
            order.push_back(checked_cast<uint32_t>(spatialDims + 1 + i));
            order.push_back(checked_cast<uint32_t>(i));
        }
        break;
    case IE::DepthToSpaceMode::DEPTH_FIRST:
        order.push_back(checked_cast<uint32_t>(Dims4D::Act::C.ind()));
        for (size_t i = 1; i <= spatialDims; ++i) {
            order.push_back(checked_cast<uint32_t>(spatialDims + 1 + i));
            order.push_back(checked_cast<uint32_t>(i) + 1);
        }
        break;
    }

    // Calculate Reshape shapeEnd
    std::vector<int64_t> shapeEnd{inputShape[Dims4D::Act::N], C};
    for (size_t i = 0; i < spatialDims; ++i) {
        shapeEnd.push_back(blockSize * inputShape[Dim(2 + i)]);
    }
    // Check output shape
    const auto outShape = to_small_vector(getShape(origOp.getOutput()));
    VPUX_THROW_UNLESS(
            outShape.size() == shapeEnd.size() && std::equal(shapeEnd.begin(), shapeEnd.end(), outShape.begin()),
            "Replacing failed: output shape mismatched");

    auto reshapeBegin = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(), nullptr,
                                                       false, getIntArrayAttr(ctx, shapeBegin));
    auto transpose =
            rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), reshapeBegin.getOutput(), nullptr,
                                             mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(order, ctx)));
    auto reshapeEnd = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out"), transpose.getOutput(), nullptr,
                                                     false, getIntArrayAttr(ctx, shapeEnd));
    rewriter.replaceOp(origOp, reshapeEnd.getOutput());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertDepth2SpaceLayerPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::DepthToSpaceOp>([&](IE::DepthToSpaceOp depthToSpaceOp) {
        return VPUIP::isLegalAndBeneficialConvertToDMA(depthToSpaceOp.getOperation(), _log);
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::TransposeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<Depth2SpaceLayerConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDepth2SpaceLayerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDepth2SpaceLayerPass(Logger log) {
    return std::make_unique<ConvertDepth2SpaceLayerPass>(log);
}
