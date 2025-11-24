//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDEFORMABLECONVTOCONV
#define GEN_PASS_DEF_CONVERTDEFORMABLECONVTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

std::vector<float> generateIndices(int N, int H, int W, int KH = 1, int KW = 1) {
    std::vector<float> indices(N * KH * H * KW * W * 2);

    // Precompute center offsets
    const float center_w = static_cast<float>(W - 1) / 2.0f;
    const float center_h = static_cast<float>(H - 1) / 2.0f;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int offset_h = 0; offset_h < KH; ++offset_h) {
                    for (int offset_w = 0; offset_w < KW; ++offset_w) {
                        // Calculate the index in the flattened array
                        int idx = (n * H * W + h * W + w) * 2 + (offset_h * KW + offset_w);
                        // Set the indices for the current position
                        indices[idx] = static_cast<float>(w + offset_w) - center_w;
                        indices[idx + 1] = static_cast<float>(h + offset_h) - center_h;
                    }
                }
            }
        }
    }

    return indices;
}

//
// ConvertDeformableConvToConv
//

class ConvertDeformableConvToConv final : public mlir::OpRewritePattern<IE::DeformableConvolutionOp> {
public:
    ConvertDeformableConvToConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DeformableConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvertDeformableConvToConv");
    }

    mlir::LogicalResult matchAndRewrite(IE::DeformableConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// matchAndRewrite
//

mlir::LogicalResult ConvertDeformableConvToConv::matchAndRewrite(IE::DeformableConvolutionOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto kernelShape = getShape(origOp.getKernel());
    const auto maxKernelSize = std::min(config::getMaxKernelSize(origOp), VPU::NCEInvariant::MAX_STRIDE);
    if (kernelShape[Dims4D::Act::H] > maxKernelSize || kernelShape[Dims4D::Act::W] > maxKernelSize) {
        return matchFailed(rewriter, origOp, "DeformableConvolutionOp with kernel shape {0} is not supported.",
                           kernelShape);
    }
    auto inputShape = getShape(origOp.getInput());
    if (inputShape.size() != 4) {
        return matchFailed(rewriter, origOp, "DeformableConvolutionOp with input shape {0} is not supported.",
                           inputShape);
    }
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    if (strides[Dims4D::Strides::X.ind()] != 1 || strides[Dims4D::Strides::Y.ind()] != 1) {
        return matchFailed(rewriter, origOp, "DeformableConvolutionOp with input stride > 1 is not supported.");
    }

    if (origOp.getDeformableGroup() != 1 || origOp.getGroup() != 1) {
        return matchFailed(rewriter, origOp,
                           "DeformableConvolutionOp with DeformableGroup or Group > 1 is not supported.");
    }

    _log.nest().trace("Convert DeformableConv = '{0}'", origOp);
    auto ctx = rewriter.getContext();
    const auto N = inputShape[Dims4D::Act::N];
    const auto H = inputShape[Dims4D::Act::H];
    const auto W = inputShape[Dims4D::Act::W];
    const auto KH = kernelShape[Dims4D::Act::H];
    const auto KW = kernelShape[Dims4D::Act::W];

    // construct new input with gridsample from the original input based offset
    // Example: offset (N,2*KH*KW,H,W) -> grid (N,H*KH,W*KW,2) with input shape (N,C,H,W):
    //
    // offset (N,2*KH*KW,H,W) -> shapeCast to (N,2,KH,KW,H,W) -> transpose to (N,KH,H,KW,W,2)
    // -> shapeCast to (N, H*KH,W*KW,2) -> add spatial indices -> divide by scale (norm to -1~1) = grid

    const SmallVector<int64_t> newShapeBeforePerm = {N, 2, KH, KW, H, W};
    auto shapeCastBeforePerm = rewriter.create<IE::ReshapeOp>(
            appendLoc(origOp.getLoc(), "reshape_before_offset_perm"), origOp.getOffset(), /*shape=*/nullptr,
            /*special_zero=*/false, getIntArrayAttr(ctx, newShapeBeforePerm));

    const SmallVector<int64_t> newPermVec{0, 2, 4, 3, 5, 1};
    auto newPerm = mlir::AffineMap::getPermutationMap(ArrayRef(newPermVec), ctx);
    const auto dimOrder6D =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 1, 2, 3, 4, 5}, ctx));
    auto offsetPermOp = rewriter.create<IE::MemPermuteOp>(appendLoc(origOp.getLoc(), "offset_perm"),
                                                          shapeCastBeforePerm.getResult(), dimOrder6D,
                                                          mlir::AffineMapAttr::get(newPerm));

    const SmallVector<int64_t> newShapeAfterPerm = {N, KH * H, KW * W, 2};
    auto shapeCastAfterPerm = rewriter.create<IE::ReshapeOp>(
            appendLoc(origOp.getLoc(), "reshape_after_offset_perm"), offsetPermOp.getOutput(), /*shape=*/nullptr,
            /*special_zero=*/false, getIntArrayAttr(ctx, newShapeAfterPerm));

    // Create indices constant with shape (N, KH*H, KW*W, 2)
    const auto indicesData = generateIndices(N, H, W, KH, KW);

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(shapeCastAfterPerm.getResult().getType()).getElementType();
    auto indicesType = mlir::RankedTensorType::get(newShapeAfterPerm, elemType);
    auto offsetIndicesConst = Const::createConst(rewriter, origOp.getLoc(), indicesType, ArrayRef(indicesData));
    auto offsetIndicesConvertOp =
            rewriter.create<IE::ConvertOp>(appendLoc(origOp.getLoc(), "offset"), offsetIndicesConst, elemType);

    auto addOp = rewriter.create<IE::AddOp>(appendLoc(origOp.getLoc(), "offset_add"), shapeCastAfterPerm.getResult(),
                                            offsetIndicesConvertOp.getOutput(), IE::AutoBroadcastType::NUMPY, nullptr,
                                            nullptr, nullptr, nullptr);

    // normalize the offsets to [-1, 1] range to get the grid input for GridSampleOp
    const float scaleH = 2.0f / static_cast<float>(H - 1);
    const float scaleW = 2.0f / static_cast<float>(W - 1);
    auto scaleType = mlir::RankedTensorType::get({1, 1, 1, 2}, elemType);
    auto scaleConst = Const::createConst(rewriter, origOp.getLoc(), scaleType, ArrayRef({scaleW, scaleH}));
    auto scaleInput = rewriter.create<IE::ConvertOp>(appendLoc(origOp.getLoc(), "offset_scale"), scaleConst, elemType)
                              .getOutput();
    auto divideOp =
            rewriter.create<IE::MultiplyOp>(appendLoc(origOp.getLoc(), "offset_norm"), addOp.getOutput(), scaleInput,
                                            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);

    // use corresponding padding mode according to BilinearInterpolatePad attr
    const auto gridSampleMode = IE::GridSampleModeAttr::get(ctx, IE::GridSampleMode::BILINEAR);
    const auto gridSamplePaddingMode =
            origOp.getBilinearInterpolatePad()
                    ? IE::GridSamplePaddingModeAttr::get(ctx, IE::GridSamplePaddingMode::ZEROS)
                    : IE::GridSamplePaddingModeAttr::get(ctx, IE::GridSamplePaddingMode::BORDER);
    auto gridSampleOp = rewriter.create<IE::GridSampleOp>(
            appendLoc(origOp.getLoc(), "gridsample"), origOp.getInput(), divideOp.getOutput(),
            /*alignCorners=*/false, gridSampleMode, gridSamplePaddingMode);

    // create multiply for mask input
    // for 1x1 DefConv, the mask shape (N,1,H,W) matches gridSample outout (N,C,H,W) with broadcast on Dim C
    // for KHxKW DefConv, the mask shape (N,KH*KW,H,W) need shapecast+tranpose to match gridSample output (N,C,KH*H,
    // KW*W) (N,KH*KW,H,W) -> (N,1,KH,KW,H,W) -> (N,1,KH,H,KW,W) -> (N,1,KH*H,KW*W)
    const SmallVector<int64_t> newShapeBeforeMaskPerm = {N, 1, KH, KW, H, W};
    auto shapeCastBeforeMaskPerm = rewriter.create<IE::ReshapeOp>(
            appendLoc(origOp.getLoc(), "reshape_before_mask_perm"), origOp.getMask(), /*shape=*/nullptr,
            /*special_zero=*/false, getIntArrayAttr(ctx, newShapeBeforeMaskPerm));

    const SmallVector<int64_t> newPermVecMaskMul{0, 1, 2, 4, 3, 5};
    auto newPermMaskMul = mlir::AffineMap::getPermutationMap(ArrayRef(newPermVecMaskMul), ctx);
    auto maskPermOp = rewriter.create<IE::MemPermuteOp>(appendLoc(origOp.getLoc(), "mask_perm"),
                                                        shapeCastBeforeMaskPerm.getResult(), dimOrder6D,
                                                        mlir::AffineMapAttr::get(newPermMaskMul));

    const SmallVector<int64_t> newShapeAfterMaskPerm = {N, 1, KH * H, KW * W};
    auto shapeCastAfterMaskPerm = rewriter.create<IE::ReshapeOp>(
            appendLoc(origOp.getLoc(), "reshape_after_mask_perm"), maskPermOp.getOutput(), /*shape=*/nullptr,
            /*special_zero=*/false, getIntArrayAttr(ctx, newShapeAfterMaskPerm));

    auto maskMultiplyOp = rewriter.create<IE::MultiplyOp>(
            appendLoc(origOp.getLoc(), "mask_mul"), shapeCastAfterMaskPerm.getResult(), gridSampleOp.getOutput(),
            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);

    auto newConvInput = maskMultiplyOp.getOutput();
    // create new conv with the same parameters but different stride as DeformableConvolutionOp
    // for KH x KW DefConv, the stride is (KH, KW) and pad is always (0, 0)
    auto newConvStride = parseIntArrayAttr<int64_t>(origOp.getStrides());
    newConvStride[0] = KW;
    newConvStride[1] = KH;
    const auto allZeroPads = SmallVector<int64_t>(2, 0);
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            origOp.getLoc(), newConvInput, origOp.getKernel(), /*bias=*/nullptr, getIntArrayAttr(ctx, newConvStride),
            getIntArrayAttr(ctx, allZeroPads), getIntArrayAttr(ctx, allZeroPads), origOp.getDilations(), nullptr,
            nullptr, nullptr, nullptr, nullptr);
    rewriter.replaceOp(origOp, newConvOp);

    return mlir::success();
}

//
// ConvertDeformableConvToConvPass
//

class ConvertDeformableConvToConvPass final :
        public IE::impl::ConvertDeformableConvToConvBase<ConvertDeformableConvToConvPass> {
public:
    explicit ConvertDeformableConvToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;
};

//
// safeRunOnFunc
//

void ConvertDeformableConvToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertDeformableConvToConv>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDeformableConvToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDeformableConvToConvPass(Logger log) {
    return std::make_unique<ConvertDeformableConvToConvPass>(log);
}
