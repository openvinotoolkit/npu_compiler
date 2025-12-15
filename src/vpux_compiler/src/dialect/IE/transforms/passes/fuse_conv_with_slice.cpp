//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSECONVWITHSLICE
#define GEN_PASS_DEF_FUSECONVWITHSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseConvWithSlice
//

//  Input(1x16x8x8)    Filter (64x16x1x1)         Input(1x16x8x8)      Filter(32x16x1x1)
//          \            /                                 \               /
//         Convolution(1x64x8x8)            To            Convolution(1x32x8x8)
//                  |
//            Slice(1x32x8x8)

// Or

//  Input(1x16x8x8)          Filter (64x16x1x1)
//          \                   /
//           Convolution(1x64x8x8)
//               |         |
//    Slice(1x32x8x8)     Slice(1x32x8x8)

//   To

//    Filter(32x16x1x1)            Input(1x16x8x8)         Filter(32x16x1x1)
//                 \                 /        \               /
//               Convolution(1x32x8x8)       Convolution(1x32x8x8)

class FuseConvWithSlice final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FuseConvWithSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("FuseConvWithSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool doesSliceMeetRequirement(IE::SliceOp sliceOp) {
    auto inShape = getShape(sliceOp.getSource());
    auto outShape = getShape(sliceOp.getResult());
    if (inShape.size() != 4 || outShape.size() != 4) {
        return false;
    }

    // only slice on channel
    if (inShape[Dims4D::Act::N] != outShape[Dims4D::Act::N] || inShape[Dims4D::Act::H] != outShape[Dims4D::Act::H] ||
        inShape[Dims4D::Act::W] != outShape[Dims4D::Act::W]) {
        return false;
    }

    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(sliceOp.getSource().getDefiningOp());
    if (iface == nullptr) {
        return false;
    }
    const auto alignment = iface.getOutputChannelAlignment();

    return inShape[Dims4D::Act::C] % alignment == 0 && outShape[Dims4D::Act::C] % alignment == 0;
}

bool slicesHaveOverlap(SmallVector<IE::SliceOp> slicesOp) {
    if (slicesOp.size() == 1) {
        return false;
    }

    auto vectorsHaveOverlap = [](std::pair<int64_t, int64_t> previous, std::pair<int64_t, int64_t> next) {
        return std::max(previous.first, next.first) <= std::min(previous.second, next.second);
    };

    SmallVector<std::pair<int64_t, int64_t>> startEnds;
    for (auto& slice : slicesOp) {
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(slice.getStaticOffsets());
        const auto sliceSizes = parseIntArrayAttr<int64_t>(slice.getStaticSizes());
        std::pair<int64_t, int64_t> startEnd{sliceOffsets[Dims4D::Act::C.ind()],
                                             sliceOffsets[Dims4D::Act::C.ind()] + sliceSizes[Dims4D::Act::C.ind()] - 1};
        for (auto& element : startEnds) {
            if (vectorsHaveOverlap(element, startEnd)) {
                return true;
            }
        }
        startEnds.push_back(startEnd);
    }

    return false;
}

bool checkConvParameters(IE::ConvolutionOp origOp) {
    auto inShape = getShape(origOp.getInput());
    if (inShape.size() != 4 || inShape[Dims4D::Act::N] != 1) {
        return false;
    }

    auto filterShape = getShape(origOp.getFilter());
    if (filterShape.size() != 4) {
        return false;
    }

    auto weightsCst = origOp.getFilter().getDefiningOp<Const::DeclareOp>();
    if (weightsCst == nullptr) {
        return false;
    }

    if (origOp.getBias()) {
        auto biasShape = getShape(origOp.getBias());
        if (biasShape.size() != 4) {
            return false;
        }
    }

    return true;
}

mlir::FailureOr<SmallVector<IE::SliceOp>> hasRequiredSliceUsers(IE::ConvolutionOp origOp) {
    SmallVector<IE::SliceOp> slices;
    for (auto user : origOp.getOutput().getUsers()) {
        if (auto slice = mlir::dyn_cast_or_null<IE::SliceOp>(user)) {
            if (doesSliceMeetRequirement(slice)) {
                slices.push_back(slice);
                continue;
            }
        }
        return mlir::failure();
    }

    if (slices.empty()) {
        return mlir::failure();
    }

    // if slices have overlap, here just skip since we may introduce more computation
    if (slicesHaveOverlap(slices)) {
        return mlir::failure();
    }

    return slices;
}

mlir::LogicalResult FuseConvWithSlice::matchAndRewrite(IE::ConvolutionOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got conv layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto nestedLogger = _log.nest();
    mlir::MLIRContext* ctx = origOp->getContext();

    if (!checkConvParameters(origOp)) {
        nestedLogger.trace("convolution parameters do not meet requirement");
        return mlir::failure();
    }

    // check if all users are slice and all slices' parameters meet requirement
    auto hasRequiredUsers = hasRequiredSliceUsers(origOp);
    if (mlir::failed(hasRequiredUsers)) {
        nestedLogger.trace("slice users do not meet requirement");
        return mlir::failure();
    }
    auto sliceUsers = hasRequiredUsers.value();

    auto filter = origOp.getFilter();
    auto filterShape = getShape(filter);
    auto bias = origOp.getBias();
    for (auto& slice : sliceUsers) {
        const auto sliceOffsets = Shape(parseIntArrayAttr<int64_t>(slice.getStaticOffsets()));
        const auto sliceSizes = Shape(parseIntArrayAttr<int64_t>(slice.getStaticSizes()));

        // slice filters
        Shape filterOffsets(sliceOffsets.size());
        filterOffsets[Dims4D::Act::N] = sliceOffsets[Dims4D::Act::C];
        SmallVector<int64_t> filterSizes{sliceSizes[Dims4D::Act::C], filterShape[Dims4D::Act::C],
                                         filterShape[Dims4D::Act::H], filterShape[Dims4D::Act::W]};
        auto filterslice =
                rewriter.create<IE::SliceOp>(origOp->getLoc(), filter, getIntArrayAttr(ctx, filterOffsets.raw()),
                                             getIntArrayAttr(ctx, filterSizes))
                        .getResult();

        // slice bias
        if (bias != nullptr) {
            auto biasShape = getShape(bias);
            SmallVector<int64_t> biasSizes{biasShape[Dims4D::Act::N], sliceSizes[Dims4D::Act::C],
                                           biasShape[Dims4D::Act::H], biasShape[Dims4D::Act::W]};
            bias = rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.getBias(),
                                                getIntArrayAttr(ctx, sliceOffsets.raw()),
                                                getIntArrayAttr(ctx, biasSizes))
                           .getResult();
        }
        rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
                slice, slice.getResult().getType(), origOp.getInput(), filterslice, bias, origOp.getStrides(),
                origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(), origOp.getPostOpAttr(),
                origOp.getClampAttr(), origOp.getStaticScaleAttr(), origOp.getOutputPaddingAttr(),
                origOp.getInputPaddingAttr());
    }
    nestedLogger.trace("fuse conv with slice success");
    return mlir::success();
}

//
// FuseConvWithSlicePass
//

class FuseConvWithSlicePass final : public IE::impl::FuseConvWithSliceBase<FuseConvWithSlicePass> {
public:
    explicit FuseConvWithSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseConvWithSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseConvWithSlice>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseConvWithSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseConvWithSlicePass(Logger log) {
    return std::make_unique<FuseConvWithSlicePass>(log);
}
