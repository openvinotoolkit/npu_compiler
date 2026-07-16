//
// Copyright (C) 2024-2026 Intel Corporation
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
#include "vpux/compiler/utils/walk_utils.hpp"

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

    using namespace Dims4D::Act;

    int64_t paddedC = inShape[C];
    int64_t divisor = 1;

    if (inShape[C] % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0) {
        // For 16-aligned C: find a divisor (with optional padding) where both new_C and new_W are 16-aligned.
        // This avoids producing non-aligned W dimensions that cause costly scatter DMA downstream.
        const auto maxChannelAlignment = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT *
                                         VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;  // 256 when alignment is 16
        const auto maxPadC =
                inShape[C] +
                maxChannelAlignment;  //  originC + 256 (when vpu channel alignment is 16) which is the max value for
                                      //  which we need to find a divisor that satisfies the alignment for both C and W.
                                      //  Beyond that, we would be padding too much without finding a suitable divisor.
        for (int64_t tryC = inShape[C]; tryC <= maxPadC; tryC += VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
            for (int64_t i = (tryC / 2 / VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) *
                             VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
                 i >= VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT; i -= VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
                if ((tryC % i == 0) && ((tryC / i) % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0) &&
                    (i < VPU::NCEInvariant::VPU_DIMENSION_LIMIT) &&
                    ((tryC / i) < VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                    divisor = i;
                    paddedC = tryC;
                    break;
                }
            }
            if (divisor != 1) {
                break;
            }
        }
    } else {
        // For non-aligned C: use original logic — find any valid divisor without alignment constraints.
        for (int64_t i = inShape[C] / 2; i > 2; i--) {
            if ((i < VPU::NCEInvariant::VPU_DIMENSION_LIMIT) && (inShape[C] % i == 0)) {
                divisor = i;
                break;
            }
        }
    }

    if (divisor == 1 || paddedC % divisor != 0) {
        return mlir::failure();
    }

    auto ctx = origOp.getContext();
    const SmallVector<unsigned> order = {0, 1, 3, 2};
    auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(order, ctx));

    // Transpose first (swaps H and W, does not touch C)
    auto transposeInResult = rewriter.createOrFold<IE::TransposeOp>(appendLoc(origOp->getLoc(), "transpose_in"),
                                                                    origOp.getInput(), nullptr, orderAttr);

    // If padding is needed, expand C with zeros after transpose to avoid transposing padded data
    mlir::Value reshapeInput = transposeInResult;
    if (paddedC != inShape[C]) {
        const auto padAmount = paddedC - inShape[C];
        SmallVector<int64_t> padsBegin(4, 0);
        SmallVector<int64_t> padsEnd(4, 0);
        padsEnd[C.ind()] = padAmount;
        reshapeInput = rewriter.create<IE::ExpandOp>(appendLoc(origOp->getLoc(), "pad_channels"), transposeInResult,
                                                     getIntArrayAttr(ctx, padsBegin), getIntArrayAttr(ctx, padsEnd));
    }

    const SmallVector<int64_t> newInputShape = {
            inShape[N],
            paddedC / divisor,
            inShape[W] * divisor,
            inShape[H],
    };
    const auto inputShapeAttr = getIntArrayAttr(ctx, newInputShape);
    auto reshapeInResult = rewriter.createOrFold<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "reshape_in"), reshapeInput,
                                                                inputShapeAttr);

    const auto newKernel = getIntArrayAttr(ctx, SmallVector<int64_t>{kernel[1], kernel[0]});
    const auto newStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{strides[1], strides[0]});
    auto maxpool = rewriter.create<IE::MaxPoolOp>(
            origOp.getLoc(), reshapeInResult, newKernel, newStrides, origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(),
            origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    // Reshape output back
    const SmallVector<int64_t> newOutputShape = {
            outShape[N],
            paddedC,
            outShape[W],
            outShape[H],
    };
    const auto outputShapeAttr = getIntArrayAttr(ctx, newOutputShape);
    auto reshapeOutResult = rewriter.createOrFold<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "reshape_out"),
                                                                 maxpool->getResult(0), outputShapeAttr);

    // If we padded, slice back to original C before transpose to avoid transposing padded data
    mlir::Value transposeInput = reshapeOutResult;
    if (paddedC != inShape[C]) {
        SmallVector<int64_t> offsets(4, 0);
        const SmallVector<int64_t> sizes = {
                outShape[N],
                outShape[C],
                outShape[W],
                outShape[H],
        };
        transposeInput = rewriter.create<IE::SliceOp>(appendLoc(origOp->getLoc(), "slice_channels"), reshapeOutResult,
                                                      getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));
    }

    auto transposeOutResult = rewriter.createOrFold<IE::TransposeOp>(appendLoc(origOp->getLoc(), "transpose_out"),
                                                                     transposeInput, nullptr, orderAttr);

    origOp.getOutput().replaceAllUsesWith(transposeOutResult);

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

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createReshapeMaxPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReshapeMaxPoolPass(Logger log) {
    return std::make_unique<ReshapeMaxPoolPass>(log);
}
