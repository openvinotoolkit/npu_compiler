//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_HANDLEEXCLUDEPADFORAVGPOOL
#define GEN_PASS_DEF_HANDLEEXCLUDEPADFORAVGPOOL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

mlir::Value createSubAvgPool(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter, ArrayRef<int64_t> begins,
                             ArrayRef<int64_t> ends, mlir::ArrayAttr avgPoolOpKernelAttr,
                             mlir::ArrayAttr avgPoolOpStridesAttr, StringRef locSuffix) {
    mlir::MLIRContext* ctx = origOp->getContext();
    const auto beginMask = getIntArrayAttr(ctx, ArrayRef({1, 1, 0, 0}));
    const auto endMask = getIntArrayAttr(ctx, ArrayRef({1, 1, 0, 0}));
    const auto newAxisMask = getIntArrayAttr(ctx, ArrayRef({0, 0, 0, 0}));
    const auto shrinkAxisMask = getIntArrayAttr(ctx, ArrayRef({0, 0, 0, 0}));
    const auto ellipsisMask = getIntArrayAttr(ctx, ArrayRef({0, 0, 0, 0}));
    const auto stridesAttr = getIntArrayAttr(ctx, ArrayRef({1, 1, 1, 1}));
    const auto beginsAttr = getIntArrayAttr(ctx, begins);
    const auto endsAttr = getIntArrayAttr(ctx, ends);

    auto stridedSliceOp = rewriter.createOrFold<IE::StridedSliceOp>(
            takeOpLoc(origOp, StringLiteral("slice_{0}"), locSuffix), origOp.getInput(), nullptr, nullptr, nullptr,
            beginsAttr, endsAttr, stridesAttr, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto zeroPadAttr = getIntArrayAttr(ctx, ArrayRef({0, 0}));

    auto avgPoolOp = rewriter.create<IE::AvgPoolOp>(takeOpLoc(origOp, StringLiteral("pool_{0}"), locSuffix),
                                                    stridedSliceOp == nullptr ? origOp.getInput() : stridedSliceOp,
                                                    avgPoolOpKernelAttr, avgPoolOpStridesAttr, zeroPadAttr, zeroPadAttr,
                                                    origOp.getRoundingTypeAttr(), nullptr, origOp.getPostOpAttr(),
                                                    origOp.getClampAttr(), origOp.getStaticScaleAttr(),
                                                    origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    return avgPoolOp.getOutput();
}

//
// AveragePoolRewriter
//

class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AveragePool layer at '{1}'", getDebugName(), origOp->getLoc());
    auto nestLog = _log.nest();
    auto* ctx = origOp->getContext();

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    const auto outputShape = getShape(origOp.getOutput());
    const auto outputWidth = outputShape[Dims4D::Act::W];
    const auto outputHeight = outputShape[Dims4D::Act::H];
    const auto inputShape = getShape(origOp.getInput());
    const auto inputWidth = inputShape[Dims4D::Act::W];
    const auto inputHeight = inputShape[Dims4D::Act::H];
    const auto origStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto kernelHeight = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto kernelWidth = kernelSize[Dims4D::Kernel::X.ind()];
    SmallVector<mlir::Value> inputs;
    SmallVector<SmallVector<int64_t>> staticOffsets;
    nestLog.trace("Create AvgPool center op without any padding");
    auto stridePadsbeginHDiff =
            padsBegin[Dims4D::PadsBegin::Top.ind()] == 0
                    ? 0
                    : origStrides[Dims4D::Strides::Y.ind()] - padsBegin[Dims4D::PadsBegin::Top.ind()];
    const auto beginH = stridePadsbeginHDiff > 0 ? stridePadsbeginHDiff : 0;
    auto stridePadsbeginWDiff =
            padsBegin[Dims4D::PadsBegin::Left.ind()] == 0
                    ? 0
                    : origStrides[Dims4D::Strides::X.ind()] - padsBegin[Dims4D::PadsBegin::Left.ind()];
    const auto beginW = stridePadsbeginWDiff > 0 ? stridePadsbeginWDiff : 0;

    auto avgCenterOp = createSubAvgPool(origOp, rewriter, /*begins=*/
                                        {0, 0, beginH, beginW},
                                        /*ends=*/{0, 0, inputHeight, inputWidth}, origOp.getKernelSize(),
                                        origOp.getStrides(), "center");

    // The padsEnd attribute maybe redundant. e.g.
    // input: 1x256x96x32xf16, output: 1x256x48x16xf16
    // kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    // For this specific case, only avgCenterOp should exist.
    const auto zeros = SmallVector<int64_t>{0, 0};
    if (padsBegin == zeros) {
        auto avgCenterOutputShape = getShape(avgCenterOp);
        if (avgCenterOutputShape == outputShape) {
            rewriter.replaceOp(origOp, avgCenterOp);
            return mlir::success();
        }
    }

    inputs.push_back(avgCenterOp);
    staticOffsets.push_back({0, 0, padsBegin[Dims4D::PadsBegin::Top.ind()] == 0 ? 0 : 1,
                             padsBegin[Dims4D::PadsBegin::Left.ind()] == 0 ? 0 : 1});

    nestLog.trace("Create SubAvgPool operation for top left corner");
    auto cornerKernelH = kernelHeight - padsBegin[Dims4D::PadsBegin::Top.ind()];
    auto cornerKernelW = kernelWidth - padsBegin[Dims4D::PadsBegin::Left.ind()];
    auto cornerKernelAttr = getIntArrayAttr(ctx, ArrayRef({cornerKernelH, cornerKernelW}));
    const auto cornerStridesAttr = getIntArrayAttr(ctx, ArrayRef({1, 1}));

    if (padsBegin[Dims4D::PadsBegin::Top.ind()] != 0 && padsBegin[Dims4D::PadsBegin::Left.ind()] != 0) {
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, 0},
                                          /*ends=*/{0, 0, cornerKernelH, cornerKernelW}, cornerKernelAttr,
                                          cornerStridesAttr, "top_left"));
        staticOffsets.push_back({0, 0, 0, 0});
    }

    nestLog.trace("Create SubAvgPool operation for top right corner");
    if (padsBegin[Dims4D::PadsBegin::Top.ind()] != 0 && padsEnd[Dims4D::PadsEnd::Right.ind()] != 0) {
        auto beginLocalW =
                (outputWidth - 1) * origStrides[Dims4D::Strides::X.ind()] - padsBegin[Dims4D::PadsBegin::Left.ind()];
        cornerKernelW = inputWidth - beginLocalW;
        cornerKernelAttr = getIntArrayAttr(ctx, ArrayRef({cornerKernelH, cornerKernelW}));
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, beginLocalW},
                                          /*ends=*/{0, 0, cornerKernelH, inputWidth}, cornerKernelAttr,
                                          cornerStridesAttr, "top_right"));
        staticOffsets.push_back({0, 0, 0, ((outputWidth - 1))});
    }

    nestLog.trace("Create SubAvgPool operation for bottom right corner");
    cornerKernelH = kernelHeight - padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    if (padsEnd[Dims4D::PadsEnd::Bottom.ind()] != 0 && padsEnd[Dims4D::PadsEnd::Right.ind()] != 0) {
        auto beginLocalW =
                (outputWidth - 1) * origStrides[Dims4D::Strides::X.ind()] - padsBegin[Dims4D::PadsBegin::Left.ind()];
        cornerKernelW = inputWidth - beginLocalW;
        cornerKernelAttr = getIntArrayAttr(ctx, ArrayRef({cornerKernelH, cornerKernelW}));
        inputs.push_back(createSubAvgPool(origOp, rewriter,
                                          /*begins=*/{0, 0, inputHeight - cornerKernelH, beginLocalW},
                                          /*ends=*/{0, 0, inputHeight, inputWidth}, cornerKernelAttr, cornerStridesAttr,
                                          "bottom_right"));
        staticOffsets.push_back({0, 0, (outputHeight - 1), (outputWidth - 1)});
    }

    nestLog.trace("Create SubAvgPool operation for bottom left corner");
    if (padsEnd[Dims4D::PadsEnd::Bottom.ind()] != 0 && padsBegin[Dims4D::PadsBegin::Left.ind()] != 0) {
        cornerKernelW = kernelWidth - padsBegin[Dims4D::PadsBegin::Left.ind()];
        cornerKernelAttr = getIntArrayAttr(ctx, ArrayRef({cornerKernelH, cornerKernelW}));
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, inputHeight - cornerKernelH, 0},
                                          /*ends=*/{0, 0, inputHeight, cornerKernelW}, cornerKernelAttr,
                                          cornerStridesAttr, "bottom_left"));
        staticOffsets.push_back({0, 0, (outputHeight - 1), 0});
    }
    nestLog.trace("Create SubAvgPool operation for left side");

    const auto verticalKernelH = kernelHeight;
    auto verticalKernelW = kernelWidth - padsBegin[Dims4D::PadsBegin::Left.ind()];
    SmallVector<int64_t> verticalStrides({origStrides[Dims4D::Strides::Y.ind()], 1});
    auto verticalStridesAttr = getIntArrayAttr(ctx, ArrayRef({verticalStrides[0], verticalStrides[1]}));
    auto verticalKernelAttr = getIntArrayAttr(ctx, ArrayRef({verticalKernelH, verticalKernelW}));
    if (padsBegin[Dims4D::PadsBegin::Left.ind()] != 0) {
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, beginH, 0},
                                          /*ends=*/{0, 0, inputHeight, verticalKernelW}, verticalKernelAttr,
                                          verticalStridesAttr, "left"));
        staticOffsets.push_back({0, 0, 1, 0});
    }

    nestLog.trace("Create SubAvgPool operation for right side");
    if (padsEnd[Dims4D::PadsEnd::Right.ind()] != 0) {
        auto beginLocalW =
                (outputWidth - 1) * origStrides[Dims4D::Strides::X.ind()] - padsBegin[Dims4D::PadsBegin::Left.ind()];
        verticalKernelW = inputWidth - beginLocalW;
        verticalKernelAttr = getIntArrayAttr(ctx, ArrayRef({verticalKernelH, verticalKernelW}));
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, beginH, beginLocalW},
                                          /*ends=*/{0, 0, inputHeight, inputWidth}, verticalKernelAttr,
                                          verticalStridesAttr, "right"));
        staticOffsets.push_back({0, 0, padsBegin[Dims4D::PadsBegin::Top.ind()] == 0 ? 0 : 1, (outputWidth - 1)});
    }

    nestLog.trace("Create SubAvgPool operation for top side");

    auto horizontalKernelH = kernelHeight - padsBegin[Dims4D::PadsBegin::Top.ind()];
    const auto horizontalKernelW = kernelWidth;
    SmallVector<int64_t> horizontalStrides({1, origStrides[Dims4D::Strides::X.ind()]});

    auto horizontalStridesAttr = getIntArrayAttr(ctx, ArrayRef({horizontalStrides[0], horizontalStrides[1]}));
    auto horizontalKernelAttr = getIntArrayAttr(ctx, ArrayRef({horizontalKernelH, horizontalKernelW}));
    if (padsBegin[Dims4D::PadsBegin::Top.ind()] != 0) {
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, 0, beginW},
                                          /*ends=*/{0, 0, horizontalKernelH, inputWidth}, horizontalKernelAttr,
                                          horizontalStridesAttr, "top"));
        staticOffsets.push_back({0, 0, 0, padsBegin[Dims4D::PadsBegin::Left.ind()] == 0 ? 0 : 1});
    }

    nestLog.trace("Create SubAvgPool operation for bottom side");
    horizontalKernelH = kernelHeight - padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    horizontalKernelAttr = getIntArrayAttr(ctx, ArrayRef({horizontalKernelH, horizontalKernelW}));
    if (padsEnd[Dims4D::PadsEnd::Bottom.ind()] != 0) {
        inputs.push_back(createSubAvgPool(origOp, rewriter, /*begins=*/{0, 0, inputHeight - horizontalKernelH, beginW},
                                          /*ends=*/{0, 0, inputHeight, inputWidth}, horizontalKernelAttr,
                                          horizontalStridesAttr, "bottom"));
        staticOffsets.push_back({0, 0, (outputHeight - 1), padsBegin[Dims4D::PadsBegin::Left.ind()] == 0 ? 0 : 1});
    }
    mlir::ArrayAttr staticOffsetsAttr = getIntArrayOfArray(ctx, staticOffsets);
    auto concatOp = rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp.getType(), inputs, staticOffsetsAttr);
    extendOpLoc(concatOp, "out_concat");
    return mlir::success();
}

//
// HandleExcludePadForAvgPoolPass
//

class HandleExcludePadForAvgPoolPass final :
        public IE::impl::HandleExcludePadForAvgPoolBase<HandleExcludePadForAvgPoolPass> {
public:
    explicit HandleExcludePadForAvgPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleExcludePadForAvgPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
        // Only suport input Rank == 4.
        if (inputType.getRank() != 4) {
            return true;
        }
        if (!op.getExcludePads() || op.getRoundingType() != vpux::IE::RoundingType::FLOOR) {
            return true;
        }
        const auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
        const auto zeros = SmallVector<int64_t>{0, 0};
        if (padsBegin == zeros && padsEnd == zeros) {
            return true;
        }
        return false;
    });

    target.addLegalOp<IE::StridedSliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AveragePoolRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// HandleExcludePadForAvgPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleExcludePadForAvgPoolPass(Logger log) {
    return std::make_unique<HandleExcludePadForAvgPoolPass>(log);
}
