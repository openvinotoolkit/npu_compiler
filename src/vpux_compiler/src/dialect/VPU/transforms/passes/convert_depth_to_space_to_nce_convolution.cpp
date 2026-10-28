//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_REGISTRATION_CONVERTDEPTHTOSPACETONCECONVOLUTIONPASS
#define GEN_PASS_DEF_CONVERTDEPTHTOSPACETONCECONVOLUTIONPASS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class DepthToSpaceRewriter final : public mlir::OpRewritePattern<VPU::DepthToSpaceOp> {
public:
    DepthToSpaceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::DepthToSpaceOp>(ctx), _log(log) {
        this->setDebugName("DepthToSpaceRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::DepthToSpaceOp d2sOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DepthToSpaceRewriter::matchAndRewrite(VPU::DepthToSpaceOp d2sOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", d2sOp->getName(), d2sOp->getLoc());
    auto nestedLogger = _log.nest();

    if (d2sOp.getPaddedChannels().has_value()) {
        nestedLogger.trace("DepthToSpaceOp has padding, cannot be converted");
        return mlir::failure();
    }

    const auto blkSize = d2sOp.getBlockSize();
    if (blkSize != 2 && blkSize != 4) {
        nestedLogger.trace("DepthToSpaceOp has unsupported block size {0}", blkSize);
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getOutput().getType());

    if (inputType.getShape()[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }

    if (inputType.getDimsOrder() != outputType.getDimsOrder() || inputType.getDimsOrder() != DimsOrder::NHWC) {
        nestedLogger.trace(
                "DepthToSpaceOp has unsupported input and output layouts {0} and {1} for NCEConvolution conversion",
                inputType.getDimsOrder(), outputType.getDimsOrder());
        return mlir::failure();
    }

    // Const::buildWeightsConst does not support plain integer element types.
    const auto inElemType = inputType.getElementType();
    if (inElemType.isSignedInteger() || inElemType.isUnsignedInteger()) {
        nestedLogger.trace("DepthToSpaceOp has unsupported input element type {0} for NCEConvolution conversion",
                           inElemType);
        return mlir::failure();
    }

    const auto inChannels = inputType.getShape()[Dims4D::Act::C];
    const auto outputShape = outputType.getShape();
    const auto rank = static_cast<int64_t>(outputShape.size());

    auto* ctx = d2sOp.getContext();
    const auto mode = d2sOp.getMode();
    const auto variant = (mode == IE::DepthToSpaceMode::BLOCKS_FIRST) ? VPU::S2DD2SVariant::BLOCK_FIRST
                                                                      : VPU::S2DD2SVariant::DEPTH_FIRST;
    const auto s2dd2sConfig =
            VPU::S2DD2SConfigAttr::get(ctx, VPU::S2DD2SEnableAttr::get(ctx, VPU::S2DD2SEnable::D2S),
                                       VPU::S2DD2SVariantAttr::get(ctx, variant), vpux::getIntAttr(ctx, blkSize));

    const auto transformInfo = VPU::getODUS2DD2STransformInfo(s2dd2sConfig, rank);
    if (!transformInfo.has_value()) {
        nestedLogger.trace("DepthToSpaceOp does not provide active ODU D2S transform info");
        return mlir::failure();
    }

    const auto scales = VPU::getODUS2DD2SScaling(*transformInfo, rank);
    const auto preODUOutputShapeResult =
            VPU::invertODUScaling(scales, SmallVector<int64_t>(outputShape.raw()), d2sOp->getLoc());
    if (mlir::failed(preODUOutputShapeResult)) {
        nestedLogger.trace("DepthToSpaceOp failed to invert ODU scaling for output shape");
        return mlir::failure();
    }

    const auto preODUOutputChannels = (*preODUOutputShapeResult)[Dims4D::Act::C.ind()];
    if (preODUOutputChannels != inChannels) {
        nestedLogger.trace("DepthToSpaceOp pre-ODU output channels {0} do not match input channels {1}",
                           preODUOutputChannels, inChannels);
        return mlir::failure();
    }

    const Shape filterShape{preODUOutputChannels, inChannels, 1, 1};
    const auto filterType =
            mlir::RankedTensorType::get(filterShape.raw(), inputType.getElementType(),
                                        vpux::getTensorAttr(d2sOp.getContext(), DimsOrder::OYXI, nullptr));

    SmallVector<float> identityWeights(filterShape.totalSize(), 0.0f);
    for (auto channel : irange(inChannels)) {
        // Pre-ODU output channel index maps directly to the same input channel.
        identityWeights[channel * inChannels + channel] = 1.0f;
    }

    auto filterConst =
            Const::buildWeightsConst(rewriter, d2sOp->getLoc(), filterType, ArrayRef<float>(identityWeights));
    const SmallVector<int64_t> rawFilterShape = {preODUOutputChannels, inChannels, 1, 1};

    auto nceOp = VPU::NCEConvolutionOp::create(
            rewriter, d2sOp->getLoc(), d2sOp.getOutput().getType(),
            /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr, /*reduceGlobalMinMax=*/nullptr, d2sOp.getInput(),
            filterConst,
            /*weightsTable=*/nullptr,
            /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
            /*weight_zero_points=*/nullptr, vpux::getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1}),
            VPU::getPaddingAttr(ctx, 0, 0, 0, 0), VPU::getPpeConfig(ctx).retrievePPEAttribute(d2sOp),
            /*mpe_engine=*/nullptr, /*rawFilterShape=*/mlir::ValueRange{}, rawFilterShape,
            /*multiClusterStrategy=*/nullptr, /*output_padding=*/nullptr,
            /*input_padding=*/nullptr, s2dd2sConfig, /*axes_value=*/nullptr);

    rewriter.replaceOp(d2sOp, nceOp->getResult(0));
    return mlir::success();
}

class ConvertDepthToSpaceToNCEConvolutionPass final :
        public VPU::impl::ConvertDepthToSpaceToNCEConvolutionPassBase<ConvertDepthToSpaceToNCEConvolutionPass> {
public:
    explicit ConvertDepthToSpaceToNCEConvolutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDepthToSpaceToNCEConvolutionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DepthToSpaceRewriter>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertDepthToSpaceToNCEConvolutionPass(Logger log) {
    return std::make_unique<ConvertDepthToSpaceToNCEConvolutionPass>(log);
}
