//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/locations_verifier.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MERGEWEIGHTSSHAREDCONV
#define GEN_PASS_DEF_MERGEWEIGHTSSHAREDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Convert:
//        Input1(1x128x2x2)       Filter(16x128x1x2)     Input2(1x128x3x2)
//                        \       /           \           /
//                IE.Conv(1x16x2x1)         IE.Conv(1x16x3x1)

// To:

//        Input1(1x128x2x2)       Input2(1x128x3x2)
//                        \       /
//                IE.Concat(1x128x5x2)       Filter(16x128x1x2)
//                                \            /
//                               IE.Conv(1x16x5x1)
//                                  |          |
//                      IE.Slice(1x16x2x1)    IE.Slice(1x16x3x1)

//
// MergeWeightsSharedConv
//

class MergeWeightsSharedConv final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    MergeWeightsSharedConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("MergeWeightsSharedConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isHConcatable(SmallVector<IE::ConvolutionOp>& convOps) {
    auto refConv = convOps.front();
    auto refInShape = getShape(refConv.getInput());
    for (auto conv : convOps) {
        auto inShape = getShape(conv.getInput());
        if (refInShape.size() != inShape.size()) {
            return false;
        }
        for (auto dim : irange(inShape.size())) {
            if (Dim(dim) == Dims4D::Act::H) {
                continue;
            }
            if (inShape[Dim(dim)] != refInShape[Dim(dim)]) {
                return false;
            }
        }
    }

    return true;
}

// the reason for have below limitation is: for below case, we need to concat Input1(%0)
// and Input2(%5), the concat operand at least should be 6%, but finally the concat is
// the parent or grand parent of Slice1(%4), which will trigger mlir error since parent
// is defined after child. See #E135334.
//        Input1(1x128x2x2)(%0)   Filter(16x128x1x2)(%1)   Input2(1x128x3x2) (%5)
//                        \       /           \             /
//                IE.Conv(1x16x2x1) (%3)        IE.Conv(1x16x3x1) (%6)
//                         |                          |
//                IE.Slice1(1x8x2x1) (%4)        IE.Slice2(1x8x3x1) (%7)
bool doesConvHaveSameUser(SmallVector<IE::ConvolutionOp>& convOps) {
    mlir::Operation* refUser = nullptr;
    for (auto conv : convOps) {
        if (!conv->hasOneUse()) {
            return false;
        }
        mlir::Operation* operation = conv;
        while (operation && !operation->getUsers().empty()) {
            if (!operation->hasOneUse()) {
                return false;
            }

            auto user = *operation->getUsers().begin();
            if (IE::isPureViewOp(user)) {
                operation = user;
            } else {
                if (refUser == nullptr) {
                    refUser = user;
                    break;
                }

                if (refUser == user) {
                    break;
                }
                return false;
            }
        }
    }

    return true;
}

// Convolutions need to have below requirement:
// 1. should no have padding on H dimension, since we will concat the input on H dimension
// 2. all the convs should have the same padding and stride info
// 3. all the convs shold not have post/clamp/scale/outchannel/bias info, it can be externed
// to support some of they, but currently haven't find such kind of conv need to be merge.
bool isSupportedConv(SmallVector<IE::ConvolutionOp>& convOps) {
    auto refConv = convOps.front();
    const auto refPadStart = Shape(parseIntArrayAttr<int64_t>(refConv.getPadsBegin()));
    const auto refPadEnd = Shape(parseIntArrayAttr<int64_t>(refConv.getPadsEnd()));
    const auto refStrides = Shape(parseIntArrayAttr<int64_t>(refConv.getStrides()));
    const auto refStaticScale = refConv.getStaticScaleAttr();

    // Since we concat on H dimension, so should not have pad in H dimension
    if (refPadStart[Dims4D::PadsBegin::Top] != 0 || refPadEnd[Dims4D::PadsEnd::Bottom] != 0) {
        return false;
    }

    for (auto conv : convOps) {
        const auto padStart = Shape(parseIntArrayAttr<int64_t>(conv.getPadsBegin()));
        const auto padEnd = Shape(parseIntArrayAttr<int64_t>(conv.getPadsEnd()));
        const auto strides = Shape(parseIntArrayAttr<int64_t>(conv.getStrides()));
        if (refPadStart != padStart || refPadEnd != padEnd || refStrides != strides) {
            return false;
        }

        if (conv.getPostOpAttr() != nullptr || conv.getClampAttr() != nullptr ||
            conv.getOutputChannelsAttr() != nullptr || conv.getBias() != nullptr ||
            conv.getStaticScaleAttr() != refStaticScale) {
            return false;
        }
    }

    return true;
}

// Merge weights shared conv for smaller size  will reduce the runtime idle time, and increase the HW efficient,
// like H from 1 to N. For bigger size, we could not see such kinds of benefit.
bool isBeneficialToMerge(SmallVector<IE::ConvolutionOp>& convOps) {
    for (auto conv : convOps) {
        auto inShape = getShape(conv.getInput());
        if (inShape[Dims4D::Act::N] != 1 || inShape[Dims4D::Act::H] != 1 || inShape[Dims4D::Act::W] != 1) {
            return false;
        }
    }

    return true;
}

SmallVector<IE::ConvolutionOp> sortConvsIfNecessary(SmallVector<IE::ConvolutionOp> convOps) {
    SmallVector<std::pair<IE::ConvolutionOp, IE::SliceOp>> convSliceOps;
    for (auto conv : convOps) {
        if (auto parent = conv.getInput().getDefiningOp()) {
            if (IE::isPureViewOp(parent)) {
                if (auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(parent->getOperand(0).getDefiningOp())) {
                    convSliceOps.push_back(std::make_pair(conv, sliceOp));
                }
            } else {
                if (auto sliceOp = mlir::dyn_cast<IE::SliceOp>(conv.getInput().getDefiningOp())) {
                    convSliceOps.push_back(std::make_pair(conv, sliceOp));
                }
            }
        }
    }
    if (convOps.size() != convSliceOps.size()) {
        return convOps;
    }

    auto refParent = convSliceOps.front().second.getSource();
    auto refSliceAxis = IE::getSingleDiffAxis(getShape(refParent), getShape(convSliceOps.front().second.getResult()));
    if (!refSliceAxis.has_value()) {
        return convOps;
    }

    for (auto convSlice : convSliceOps) {
        auto slice = convSlice.second;
        auto sliceAxis = IE::getSingleDiffAxis(getShape(slice.getSource()), getShape(slice.getResult()));
        if (!sliceAxis.has_value() || sliceAxis.value() != refSliceAxis.value() || slice.getSource() != refParent) {
            return convOps;
        }
    }

    std::sort(convSliceOps.begin(), convSliceOps.end(),
              [&](std::pair<IE::ConvolutionOp, IE::SliceOp> previous, std::pair<IE::ConvolutionOp, IE::SliceOp> next) {
                  const auto firstOffsets = parseIntArrayAttr<int64_t>(previous.second.getStaticOffsets());
                  const auto secondOffsets = parseIntArrayAttr<int64_t>(next.second.getStaticOffsets());
                  return firstOffsets[refSliceAxis.value().ind()] < secondOffsets[refSliceAxis.value().ind()];
              });

    SmallVector<IE::ConvolutionOp> sortConvOps;
    for (auto convSlice : convSliceOps) {
        sortConvOps.push_back(convSlice.first);
    }

    return sortConvOps;
}

mlir::LogicalResult MergeWeightsSharedConv::matchAndRewrite(IE::ConvolutionOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto filter = origOp.getFilter();
    SmallVector<IE::ConvolutionOp> convOps;

    bool sharedBeforeAffineReshape = false;
    auto refAffineReshapeOp = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(filter.getDefiningOp());
    if (refAffineReshapeOp) {
        sharedBeforeAffineReshape = true;
        for (auto user : refAffineReshapeOp.getInput().getUsers()) {
            auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(user);
            if (affineReshapeOp == nullptr) {
                continue;
            }
            if (affineReshapeOp.getType() != refAffineReshapeOp.getType() || !affineReshapeOp->hasOneUse()) {
                sharedBeforeAffineReshape = false;
                break;
            }
            auto conv = mlir::dyn_cast_or_null<IE::ConvolutionOp>(*affineReshapeOp.getOutput().getUsers().begin());
            if (conv == nullptr) {
                sharedBeforeAffineReshape = false;
                break;
            }
            // Sometimes, one conv's filter is another conv's input or bias, avoid the case
            if (conv.getFilter() == affineReshapeOp.getOutput() && conv.getInput() != affineReshapeOp.getOutput() &&
                conv.getBias() != affineReshapeOp.getOutput()) {
                convOps.push_back(conv);
                continue;
            }
            sharedBeforeAffineReshape = false;
            break;
        }
    }

    if (!sharedBeforeAffineReshape) {
        convOps.clear();
        for (auto user : filter.getUsers()) {
            auto conv = mlir::dyn_cast<IE::ConvolutionOp>(user);
            if (conv == nullptr) {
                return matchFailed(rewriter, origOp, "Weights are shared with non-Conv Op");
            }
            if (conv.getFilter() == filter && conv.getInput() != filter && conv.getBias() != filter) {
                convOps.push_back(conv);
            }
        }
    }

    if (convOps.size() < 2) {
        return matchFailed(rewriter, origOp, "Weights shared convolution less than 2");
    }

    if (!isBeneficialToMerge(convOps)) {
        return matchFailed(rewriter, origOp, "Not beneficial to merge");
    }

    if (!isSupportedConv(convOps)) {
        return matchFailed(rewriter, origOp, "Convolution don't have same parameter");
    }

    if (!isHConcatable(convOps)) {
        return matchFailed(rewriter, origOp, "Convolution is not H concatable");
    }

    if (!doesConvHaveSameUser(convOps)) {
        return matchFailed(rewriter, origOp, "Convolution don't have same user");
    }

    // If conv input is slicing from the same parent, then we can sort the conv to make them sliced
    // form continuous memory. like if conv1 is slice from offset[0, 0, 0, 2], size[1, 1, 1, 2],
    // conv2 is slice from offset[0, 0, 0, 0], size[1, 1, 1, 2], we can sort the convOps vector to
    // {conv2, conv1}, which maybe optimized in the coming slice->concat rewriter.
    convOps = sortConvsIfNecessary(convOps);
    SmallVector<mlir::Value> convInputs;
    for (auto conv : convOps) {
        convInputs.push_back(conv.getInput());
    }
    auto inputConcat = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_concat_input"),
                                                     mlir::ValueRange(convInputs), Dims4D::Act::H);

    for (auto input : convInputs) {
        if (!mlir::isa<mlir::BlockArgument>(input) && inputConcat->isBeforeInBlock(input.getDefiningOp())) {
            inputConcat->moveAfter(input.getDefiningOp());
        }
    }

    auto newConv = rewriter.create<IE::ConvolutionOp>(
            appendLoc(origOp->getLoc(), "_concat"), inputConcat.getOutput(), filter, origOp.getBias(),
            origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(), origOp.getOutputChannelsAttr(),
            origOp.getInputChannelsAttr());
    for (auto operand : newConv->getOperands()) {
        if (operand != nullptr && !mlir::isa<mlir::BlockArgument>(operand) &&
            newConv->isBeforeInBlock(operand.getDefiningOp())) {
            newConv->moveAfter(operand.getDefiningOp());
        }
    }

    int64_t offset = 0;
    SmallVector<IE::SliceOp> sliceOps;
    for (auto p : convOps | indexed) {
        auto conv = p.value();
        auto outShape = getShape(conv.getOutput());
        Shape offsets(outShape.size());
        offsets[Dims4D::Act::H] = offset;
        offset += outShape[Dims4D::Act::H];
        auto slice = rewriter.create<IE::SliceOp>(appendLoc(origOp->getLoc(), "_slice_{0}", p.index()),
                                                  newConv.getOutput(), getIntArrayAttr(rewriter.getContext(), offsets),
                                                  getIntArrayAttr(rewriter.getContext(), outShape.raw()));
        sliceOps.push_back(slice);
    }

    for (auto p : convOps | indexed) {
        auto conv = p.value();
        if (sliceOps[p.index()]->isBeforeInBlock(newConv)) {
            sliceOps[p.index()]->moveAfter(newConv);
        }
        rewriter.replaceOp(conv, sliceOps[p.index()].getResult());
    }

    _log.trace("Merge weights shared convolution successful");
    return mlir::success();
}

//
// MergeWeightsSharedConvPass
//

class MergeWeightsSharedConvPass final : public IE::impl::MergeWeightsSharedConvBase<MergeWeightsSharedConvPass> {
public:
    explicit MergeWeightsSharedConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MergeWeightsSharedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeWeightsSharedConv>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
//  createMergeWeightsSharedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMergeWeightsSharedConvPass(Logger log) {
    return std::make_unique<MergeWeightsSharedConvPass>(log);
}
