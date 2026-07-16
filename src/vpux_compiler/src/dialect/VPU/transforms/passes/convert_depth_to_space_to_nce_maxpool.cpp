//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTDEPTHTOSPACETONCEMAXPOOLPASS
#define GEN_PASS_DEF_CONVERTDEPTHTOSPACETONCEMAXPOOLPASS
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

    // NCEMaxPool does not support plain integer element types (signed or unsigned).
    const auto inElemType = inputType.getElementType();
    if (inElemType.isSignedInteger() || inElemType.isUnsignedInteger()) {
        nestedLogger.trace("DepthToSpaceOp has unsupported input element type {0} for NCEMaxPool conversion",
                           inElemType);
        return mlir::failure();
    }

    if (inputType.getDimsOrder() != outputType.getDimsOrder() || inputType.getDimsOrder() != DimsOrder::NHWC) {
        nestedLogger.trace(
                "DepthToSpaceOp has unsupported input and output layouts {0} and {1} for NCEMaxPool conversion",
                inputType.getDimsOrder(), outputType.getDimsOrder());
        return mlir::failure();
    }

    const auto mode = d2sOp.getMode();

    auto* ctx = d2sOp.getContext();

    const auto variant = (mode == IE::DepthToSpaceMode::BLOCKS_FIRST) ? VPU::S2DD2SVariant::BLOCK_FIRST
                                                                      : VPU::S2DD2SVariant::DEPTH_FIRST;
    const auto s2dd2sConfig =
            VPU::S2DD2SConfigAttr::get(ctx, VPU::S2DD2SEnableAttr::get(ctx, VPU::S2DD2SEnable::D2S),
                                       VPU::S2DD2SVariantAttr::get(ctx, variant), vpux::getIntAttr(ctx, blkSize));

    const auto padAttr = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(d2sOp);
    const auto kernelSizeAttr = vpux::getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto stridesAttr = vpux::getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            d2sOp->getLoc(), d2sOp.getOutput().getType(),
            /*reduce_xy_max=*/mlir::Type{}, /*reduce_xy_min=*/mlir::Type{}, /*reduce_tensor_min_max=*/mlir::Type{},
            d2sOp.getInput(), /*weightsTable=*/nullptr, /*weight_table_scale=*/nullptr,
            /*weight_table_bias=*/nullptr, kernelSizeAttr, stridesAttr, padAttr, ppeAttr,
            /*mpe_engine=*/nullptr, /*multiClusterStrategy=*/nullptr, /*output_padding=*/nullptr,
            /*input_padding=*/nullptr, s2dd2sConfig, /*axes_value=*/nullptr);

    rewriter.replaceOp(d2sOp, nceOp->getResult(0));
    return mlir::success();
}

class ConvertDepthToSpaceToNCEMaxPoolPass final :
        public VPU::impl::ConvertDepthToSpaceToNCEMaxPoolPassBase<ConvertDepthToSpaceToNCEMaxPoolPass> {
public:
    explicit ConvertDepthToSpaceToNCEMaxPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDepthToSpaceToNCEMaxPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DepthToSpaceRewriter>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertDepthToSpaceToNCEMaxPoolPass(Logger log) {
    return std::make_unique<ConvertDepthToSpaceToNCEMaxPoolPass>(log);
}
