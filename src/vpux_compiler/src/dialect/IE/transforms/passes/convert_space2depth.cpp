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
#define GEN_PASS_DECL_CONVERTSPACE2DEPTHLAYER
#define GEN_PASS_DEF_CONVERTSPACE2DEPTHLAYER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

constexpr size_t spatialDimIndex = 2;
namespace {

//
// ConvertSpace2DepthLayerPass
//

class ConvertSpace2DepthLayerPass final : public IE::impl::ConvertSpace2DepthLayerBase<ConvertSpace2DepthLayerPass> {
public:
    explicit ConvertSpace2DepthLayerPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class Space2DepthLayerConverter;

private:
    void safeRunOnFunc() final;
};

class ConvertSpace2DepthLayerPass::Space2DepthLayerConverter final : public mlir::OpRewritePattern<IE::SpaceToDepthOp> {
public:
    Space2DepthLayerConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SpaceToDepthOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SpaceToDepthOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSpace2DepthLayerPass::Space2DepthLayerConverter::matchAndRewrite(
        IE::SpaceToDepthOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();

    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

    // Should fail in frontend if input rank < 3
    auto spatialDims = checked_cast<size_t>(inputShape.size() - spatialDimIndex);
    auto C = inputShape[Dims4D::Act::C];

    SmallVector<int64_t> shapeBegin{inputShape[Dims4D::Act::N], C};
    for (size_t i = 0; i < spatialDims; ++i) {
        shapeBegin.push_back(inputShape[Dim(spatialDimIndex + i)] / blockSize);
        shapeBegin.push_back(blockSize);
    }

    SmallVector<uint32_t> order{checked_cast<uint32_t>(Dims4D::Act::N.ind())};

    auto fillBlockOrder = [spatialDims](SmallVector<uint32_t>& orderVec) -> void {
        for (size_t i = 0; i < spatialDims; ++i) {
            orderVec.push_back(checked_cast<uint32_t>(spatialDimIndex + 2 * i + 1));
        }
    };

    auto fillSpatialOrder = [spatialDims](SmallVector<uint32_t>& orderVec) -> void {
        for (size_t i = 0; i < spatialDims; ++i) {
            orderVec.push_back(checked_cast<uint32_t>(spatialDimIndex + 2 * i));
        }
    };

    switch (mode) {
    case IE::SpaceToDepthMode::BLOCKS_FIRST:
        fillBlockOrder(order);
        order.push_back(checked_cast<uint32_t>(Dims4D::Act::C.ind()));
        fillSpatialOrder(order);
        break;
    case IE::SpaceToDepthMode::DEPTH_FIRST:
        order.push_back(checked_cast<uint32_t>(Dims4D::Act::C.ind()));
        fillBlockOrder(order);
        fillSpatialOrder(order);
    }

    SmallVector<int64_t> shapeEnd{inputShape[Dims4D::Act::N]};
    shapeEnd.push_back(C * checked_cast<int64_t>(std::pow(blockSize, spatialDims)));
    for (size_t i = 0; i < spatialDims; ++i) {
        shapeEnd.push_back(inputShape[Dim(spatialDimIndex + i)] / blockSize);
    }

    // Check output shape
    const auto outShape = to_small_vector(getShape(origOp.getOutput()));
    VPUX_THROW_UNLESS(
            outShape.size() == shapeEnd.size() && std::equal(shapeEnd.begin(), shapeEnd.end(), outShape.begin()),
            "Replacing failed due to output shape mismatch: original '{0}', new '{1}'", outShape, shapeEnd);

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

void ConvertSpace2DepthLayerPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::SpaceToDepthOp>([&](IE::SpaceToDepthOp spaceToDepthOp) {
        return VPUIP::isLegalAndBeneficialConvertToDMA(spaceToDepthOp.getOperation(), _log);
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::TransposeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<Space2DepthLayerConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSpace2DepthLayerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSpace2DepthLayerPass(Logger log) {
    return std::make_unique<ConvertSpace2DepthLayerPass>(log);
}
