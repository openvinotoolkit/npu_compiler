//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <numeric>

namespace vpux::VPU {
#define GEN_PASS_DECL_FOLDAVGPOOLCHANNELEXPAND
#define GEN_PASS_DEF_FOLDAVGPOOLCHANNELEXPAND
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// FoldAvgPoolChannelExpand
//

// Folds a chunk of the width dimension into channels for an eltwise NCE.AveragePool fed by a channel-alignment
// Expand, trading a large channel Expand (e.g. 3->16 channels) for a much smaller width Expand. Requires
// static shapes.
class FoldAvgPoolChannelExpand final : public mlir::OpRewritePattern<VPU::NCEAveragePoolOp> {
public:
    FoldAvgPoolChannelExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEAveragePoolOp>(ctx), _log(log) {
        this->setDebugName("FoldAvgPoolChannelExpand");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FoldAvgPoolChannelExpand::matchAndRewrite(VPU::NCEAveragePoolOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (!VPU::isEltwiseNCEPooling(origOp)) {
        return mlir::failure();
    }

    if (origOp.getWeightTableScale() != nullptr || origOp.getWeightTableBias() != nullptr) {
        return matchFailed(_log, rewriter, origOp,
                           "NCE.AveragePool has a per-channel weight table, folding would corrupt semantics");
    }

    if (IE::isPerAxisQuant(origOp.getInput()) || IE::isPerAxisQuant(origOp.getOutput())) {
        return matchFailed(_log, rewriter, origOp, "NCE.AveragePool has per-axis quantization");
    }

    auto expandOp = origOp.getInput().getDefiningOp<VPU::ExpandOp>();
    if (expandOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "NCE.AveragePool is not fed by a VPU.Expand");
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBeginAttr());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEndAttr());
    if (padsBegin.size() != 4 || padsEnd.size() != 4) {
        return matchFailed(_log, rewriter, origOp, "Only 4D Expand is supported");
    }
    if (!llvm::all_of(padsBegin, [](int64_t val) {
            return val == 0;
        })) {
        return matchFailed(_log, rewriter, origOp, "Expand has non-zero pads_begin");
    }
    for (auto ind : irange(padsEnd.size())) {
        if (ind != checked_cast<size_t>(Dims4D::Act::C.ind()) && padsEnd[ind] != 0) {
            return matchFailed(_log, rewriter, origOp, "Expand pads a dimension other than channels");
        }
    }

    const auto realInput = expandOp.getInput();
    const auto realInputType = mlir::cast<NDTypeInterface>(realInput.getType());
    const auto poolOutputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    if (realInputType.getShape().isDynamic() || poolOutputType.getShape().isDynamic()) {
        return matchFailed(_log, rewriter, origOp, "Dynamic shapes are not supported");
    }

    // The operation to replace and the type of its result: the pool itself, or the Slice following it,
    // if the original input has the same number of channels as the Slice output has
    mlir::Operation* opToReplace = origOp.getOperation();
    auto origOutputType = poolOutputType;
    if (poolOutputType.getShape()[Dims4D::Act::C] ==
                mlir::cast<NDTypeInterface>(expandOp.getOutput().getType()).getShape()[Dims4D::Act::C] &&
        origOp->hasOneUse()) {
        if (auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(*origOp->user_begin())) {
            const auto sliceOutputType = mlir::cast<NDTypeInterface>(sliceOp.getResult().getType());
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
            const auto cutsOffChannels =
                    llvm::all_of(sliceOffsets,
                                 [](int64_t offset) {
                                     return offset == 0;
                                 }) &&
                    sliceOutputType.getShape()[Dims4D::Act::C] ==
                            mlir::cast<NDTypeInterface>(realInput.getType()).getShape()[Dims4D::Act::C] &&
                    sliceOutputType.getShape()[Dims4D::Act::N] == poolOutputType.getShape()[Dims4D::Act::N] &&
                    sliceOutputType.getShape()[Dims4D::Act::H] == poolOutputType.getShape()[Dims4D::Act::H] &&
                    sliceOutputType.getShape()[Dims4D::Act::W] == poolOutputType.getShape()[Dims4D::Act::W];
            if (cutsOffChannels) {
                opToReplace = sliceOp.getOperation();
                origOutputType = sliceOutputType;
            }
        }
    }

    const auto realInputShape = realInputType.getShape();
    if (realInputShape.size() != 4) {
        return matchFailed(_log, rewriter, origOp, "Only 4D shapes are supported");
    }

    const auto alignment = VPU::NCEInvariant::getAlignment(realInputType.getElementType());
    const auto numChannels = realInputShape[Dims4D::Act::C];
    if (numChannels % alignment == 0) {
        return matchFailed(_log, rewriter, origOp, "Channels are already aligned");
    }
    if (origOutputType.getShape()[Dims4D::Act::C] != numChannels) {
        return matchFailed(_log, rewriter, origOp,
                           "NCE.AveragePool's output channel count does not already match the real "
                           "(unaligned) input channel count and it is not followed by a Slice cutting the "
                           "channels off, folding would change its output shape");
    }

    const auto foldFactor = alignment / std::gcd(numChannels, alignment);
    const auto width = realInputShape[Dims4D::Act::W];
    const auto widthPad = alignValUp(width, foldFactor) - width;

    const auto channelPadOverhead = static_cast<double>(alignment - numChannels) / static_cast<double>(numChannels);
    const auto widthPadOverhead = static_cast<double>(widthPad) / static_cast<double>(width);
    if (widthPadOverhead >= channelPadOverhead) {
        return matchFailed(_log, rewriter, origOp, "Width-folding is not cheaper than the existing channel padding");
    }

    _log.trace("[{0}] Folding width into channels for '{1}' at '{2}'", getDebugName(), origOp->getName(),
               origOp->getLoc());

    auto ctx = rewriter.getContext();

    mlir::Value paddedInput = realInput;
    if (widthPad > 0) {
        SmallVector<int64_t> widthPadsBegin(4, 0);
        SmallVector<int64_t> widthPadsEnd(4, 0);
        widthPadsEnd[Dims4D::Act::W.ind()] = widthPad;
        paddedInput = VPU::ExpandOp::create(rewriter, takeOpLoc(origOp, "width_align"), realInput,
                                            getIntArrayAttr(ctx, widthPadsBegin), getIntArrayAttr(ctx, widthPadsEnd));
    }

    const auto paddedWidth = width + widthPad;
    Shape newShape{realInputShape[Dims4D::Act::N], numChannels * foldFactor, realInputShape[Dims4D::Act::H],
                   paddedWidth / foldFactor};

    auto inShapeCast =
            VPU::ShapeCastOp::create(rewriter, origOp->getLoc(), paddedInput, getIntArrayAttr(ctx, newShape));

    const auto newOutputType = origOutputType.changeShape(newShape);
    auto newPoolOp = VPU::NCEAveragePoolOp::create(
            rewriter, origOp->getLoc(), newOutputType, inShapeCast.getResult(), /*weightTableScale=*/nullptr,
            /*weightTableBias=*/nullptr, origOp.getKernelSizeAttr(), origOp.getStridesAttr(), origOp.getPadAttr(),
            origOp.getPpeAttr(), origOp.getMpeEngineAttr(), origOp.getMultiClusterStrategyAttr(),
            /*output_paddingAttr=*/nullptr, /*input_paddingAttr=*/nullptr);

    Shape paddedOutShape{realInputShape[Dims4D::Act::N], numChannels, realInputShape[Dims4D::Act::H], paddedWidth};
    auto outShapeCast = VPU::ShapeCastOp::create(rewriter, origOp->getLoc(), newPoolOp.getOutput(),
                                                 getIntArrayAttr(ctx, paddedOutShape));

    if (widthPad > 0) {
        const auto origOutputShape = origOutputType.getShape();
        SmallVector<int64_t> sliceOffsets(4, 0);
        rewriter.setInsertionPoint(opToReplace);
        rewriter.replaceOpWithNewOp<VPU::SliceOp>(opToReplace, outShapeCast.getResult(), sliceOffsets,
                                                  origOutputShape.raw());
    } else {
        rewriter.replaceOp(opToReplace, outShapeCast.getResult());
    }

    // The pool may still be used by the replaced Slice only, which is gone by now
    if (opToReplace != origOp.getOperation() && origOp->use_empty()) {
        rewriter.eraseOp(origOp);
    }

    return mlir::success();
}

//
// FoldAvgPoolChannelExpandPass
//

class FoldAvgPoolChannelExpandPass final :
        public VPU::impl::FoldAvgPoolChannelExpandBase<FoldAvgPoolChannelExpandPass> {
public:
    explicit FoldAvgPoolChannelExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FoldAvgPoolChannelExpandPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldAvgPoolChannelExpand>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFoldAvgPoolChannelExpandPass(Logger log) {
    return std::make_unique<FoldAvgPoolChannelExpandPass>(log);
}
