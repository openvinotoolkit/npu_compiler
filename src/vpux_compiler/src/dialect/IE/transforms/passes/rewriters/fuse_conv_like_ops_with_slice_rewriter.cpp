//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/rewriters/fuse_conv_like_ops_with_slice_rewriter.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized_fwd.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/token_length_regime_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {
// Avoid scenarios where either the original op is unaligned or the Slice results in an unaligned channel.
// Unaligned channels legalization (through adding Expand ops or other techniques) may end up negating the
// benefit from splitting, therefore we refrain from optimizing these cases.
bool checkAlignedOpConstraints(IE::SliceOp sliceOp, mlir::Operation* convLikeOp, Dim convChannelDim,
                               Dim sliceChannelDim, Logger log) {
    auto iface = mlir::dyn_cast_if_present<IE::AlignedChannelsOpInterface>(convLikeOp);
    if (iface == nullptr) {
        log.trace("Conv-like op '{0}' at '{1}' does not implement AlignedChannelsOpInterface", convLikeOp->getName(),
                  convLikeOp->getLoc());
        return false;
    }
    const auto alignment = iface.getOutputChannelAlignment();

    const auto sliceOutShape = getBoundedShape(sliceOp.getResult());
    if (sliceOutShape[sliceChannelDim] % alignment != 0) {
        log.trace("Slice at '{0}' fails channel alignment check: sliceOutC={1}, alignment={2}", sliceOp->getLoc(),
                  sliceOutShape[sliceChannelDim], alignment);
        return false;
    }

    const auto convLikeOutShape = getBoundedShape(convLikeOp->getResult(0));
    if (convLikeOutShape[convChannelDim] % alignment != 0) {
        log.trace("Conv-like op at '{0}' fails channel alignment check: convOutC={1}, alignment={2}",
                  convLikeOp->getLoc(), convLikeOutShape[convChannelDim], alignment);
        return false;
    }
    return true;
}

std::optional<Dim> getOutputChannelDimThroughViewOps(mlir::Operation* viewOp, const Dim& dim, const Logger& log) {
    auto inputType = mlir::cast<NDTypeInterface>(viewOp->getOperand(0).getType());
    auto outputType = mlir::cast<NDTypeInterface>(viewOp->getResult(0).getType());

    if (auto permCast = mlir::dyn_cast_if_present<IE::PermuteCastOp>(viewOp)) {
        auto result = vpux::inferDimAfterPermutation(dim, inputType.getDimsOrder(), outputType.getDimsOrder(),
                                                     permCast.getMemPerm());
        log.trace("PermuteCast at '{0}' maps dim {1} -> {2}", viewOp->getLoc(), dim, result);
        return result;
    }

    if (mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::ShapeCastOp>(viewOp)) {
        const auto inMemShape = inputType.getMemShape();
        const auto outMemShape = outputType.getMemShape();

        const auto memDim = inputType.getDimsOrder().toMemDim(dim);

        // product of 'memDim' together with all more-inner (faster changing) dims on the input side
        int64_t inProd = 1;
        for (int64_t idx = memDim.ind(); idx < static_cast<int64_t>(inMemShape.size()); ++idx) {
            inProd *= inMemShape[MemDim(idx)];
        }

        log.trace("Reshape-like op '{0}' at '{1}': dim {2} -> memDim {3}, inMemShape={4}, outMemShape={5}, "
                  "inProd={6}",
                  viewOp->getName(), viewOp->getLoc(), dim, memDim, inMemShape, outMemShape, inProd);

        // walk the output mem shape from the innermost dim outwards, looking for the dim whose
        // inner product matches 'inProd' - that dim is the equivalent of 'memDim' in the output
        int64_t outProd = 1;
        for (int64_t idx = static_cast<int64_t>(outMemShape.size()) - 1; idx >= 0; --idx) {
            const auto equivalentMemDim = MemDim(idx);
            outProd *= outMemShape[equivalentMemDim];
            if (outProd < inProd) {
                continue;
            }

            // the equivalent dim's own size must match as well, otherwise 'dim' got merged or
            // split together with a neighboring dim during the reshape
            if (outProd == inProd && outMemShape[equivalentMemDim] == inMemShape[memDim]) {
                const auto equivalentDim = outputType.getDimsOrder().toDim(equivalentMemDim);
                log.trace("Reshape-like op '{0}' at '{1}': found equivalent memDim {2} (dim {3}), outProd={4}",
                          viewOp->getName(), viewOp->getLoc(), equivalentMemDim, equivalentDim, outProd);
                return equivalentDim;
            }

            log.trace("Reshape-like op '{0}' at '{1}': memDim {2} was merged/split with a neighboring dim "
                      "(inProd={3}, outProd={4}, inSize={5}, outSize={6})",
                      viewOp->getName(), viewOp->getLoc(), memDim, inProd, outProd, inMemShape[memDim],
                      outMemShape[equivalentMemDim]);
            return std::nullopt;
        }

        return std::nullopt;
    }

    log.trace("Op '{0}' at '{1}' is not a supported view-like op", viewOp->getName(), viewOp->getLoc());
    return std::nullopt;
}

mlir::Value sliceOperand(mlir::Value operand, const Dim& channelDim, int64_t channelOffset, int64_t channelSize,
                         mlir::PatternRewriter& rewriter, mlir::Location loc) {
    VPUX_THROW_WHEN(operand == nullptr, "No input tensor passed.");
    auto ctx = operand.getContext();
    const auto actShape = getBoundedShape(operand);

    auto actOffsets = SmallVector<int64_t>(actShape.size(), 0);
    actOffsets[channelDim.ind()] = channelOffset;
    auto actSizes = SmallVector<int64_t>(actShape.raw());
    actSizes[channelDim.ind()] = channelSize;

    return rewriter.createOrFold<IE::SliceOp>(loc, operand, getIntArrayAttr(ctx, actOffsets),
                                              getIntArrayAttr(ctx, actSizes));
}

void adaptReshapeLikeOpShapeAttr(mlir::Operation* viewOp, ShapeRef newShape) {
    const auto newViewShapeAttr = getIntArrayAttr(viewOp->getContext(), newShape.raw());

    // reshape-like ops carry their output shape as an attribute alongside the result type;
    // that attribute must be kept in sync with the resized output shape
    if (auto reshapeOp = mlir::dyn_cast<IE::ReshapeOp>(viewOp)) {
        reshapeOp.setShapeValueAttr(newViewShapeAttr);
        return;
    }

    if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(viewOp)) {
        affineReshapeOp.setShapeValueAttr(newViewShapeAttr);
        return;
    }

    if (auto shapeCastOp = mlir::dyn_cast<IE::ShapeCastOp>(viewOp)) {
        shapeCastOp.setShapeAttr(newViewShapeAttr);
    }
}

bool hasBeneficialFusableActivationDownstream(IE::LayerWithPostOpInterface mainOp, const IE::ConsumerChain& consumers,
                                              Logger log) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.nest().trace("{0}", msg.str());
    };

    int64_t postOpFusableSize = 0;
    for (const auto& entry : consumers.slices) {
        auto slice = entry.slice;
        if (!slice->hasOneUse()) {
            return false;
        }

        auto consumer = *slice->getResult(0).getUsers().begin();
        if (mainOp.isSupportedPostOp(consumer, logCb)) {
            log.nest().trace("Found fusable postop of type: {0}", consumer->getName());
            const auto crtPostOpSize = getBoundedShape(consumer->getResult(0).getType()).totalSize();
            postOpFusableSize += crtPostOpSize;
        }
    }

    // Slicing the op does not come for free. Doing it at this early stage may prevent optimal pipelining tiling to be
    // found for the entire matmul. To ensure that the benefit from fusing the activation outweighs the cons, set a
    // threshold that prevents the fusing unless at least half of the sliced tensor's size can have a post op attached.
    // For odd total sizes, threshold is intentionally set to floor(totalSize / 2).
    const auto totalSize = getBoundedShape(mainOp->getResult(0).getType()).totalSize();
    return postOpFusableSize >= totalSize / 2;
}

bool hasOverlappingChannelRanges(ArrayRef<IE::ConsumerChain::SliceWithViewChain> sliceEntries,
                                 const Dim& opChannelDim) {
    if (sliceEntries.size() < 2) {
        return false;
    }

    auto vectorsHaveOverlap = [](std::pair<int64_t, int64_t> previous, std::pair<int64_t, int64_t> next) {
        return std::max(previous.first, next.first) <= std::min(previous.second, next.second);
    };

    // each Slice may have been reached through a different (or empty) view-like chain, so the channel dim
    // must be resolved per-Slice rather than assuming a single dim shared by all of them.
    SmallVector<std::pair<int64_t, int64_t>> startEnds;
    for (const auto& entry : sliceEntries) {
        const auto dim = entry.viewLikeChain.empty() ? opChannelDim : entry.viewLikeChain.back().second;
        auto slice = entry.slice;
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(slice.getStaticOffsets());
        const auto sliceSizes = parseIntArrayAttr<int64_t>(slice.getStaticSizes());

        const std::pair<int64_t, int64_t> startEnd{sliceOffsets[dim.ind()],
                                                   sliceOffsets[dim.ind()] + sliceSizes[dim.ind()] - 1};
        for (const auto& element : startEnds) {
            if (vectorsHaveOverlap(element, startEnd)) {
                return true;
            }
        }
        startEnds.push_back(startEnd);
    }

    return false;
}

std::optional<Dim> getFullyConnectedBiasChannelDim(mlir::Operation* op, const Dim outputChannel) {
    auto fc = mlir::dyn_cast_if_present<IE::FullyConnectedOp>(op);
    if (fc == nullptr) {
        return std::nullopt;
    }

    auto bias = fc.getBias();
    if (bias == nullptr) {
        return std::nullopt;
    }

    auto output = fc.getOutput();
    const auto outputShape = getShape(output);
    const auto outputChannelSize = outputShape[outputChannel];

    const auto biasShape = getShape(bias);
    std::optional<Dim> biasChannelDim = std::nullopt;
    for (size_t ind = 0; ind < biasShape.size(); ind++) {
        const auto dim = Dim(ind);
        // skip 1 dims
        if (biasShape[dim] == 1) {
            continue;
        }

        // check there isn't a dim != 1 or out channel size
        if (biasShape[dim] != outputChannelSize) {
            return std::nullopt;
        }

        // check there aren't 2 dims equal to out channel size
        if (biasChannelDim.has_value()) {
            return std::nullopt;
        }

        biasChannelDim = dim;
    }

    return biasChannelDim;
}

mlir::Operation* sliceOpAtOutputChannelForMatMulLikeops(mlir::Operation* op, const Dim& outputChannelDim,
                                                        const Dim& weightsChannelDim, int64_t channelOffset,
                                                        int64_t channelSize, int64_t sliceIdx,
                                                        mlir::PatternRewriter& rewriter) {
    mlir::IRMapping mapper;
    // slice filter
    auto filterSlice = sliceOperand(op->getOperand(1), weightsChannelDim, channelOffset, channelSize, rewriter,
                                    appendLoc(op->getLoc(), "slice_filter_{0}", sliceIdx));
    mapper.map(op->getOperand(1), filterSlice);

    if (const auto biasChannelDim = getFullyConnectedBiasChannelDim(op, outputChannelDim); biasChannelDim.has_value()) {
        auto biasSlice = sliceOperand(op->getOperand(2), biasChannelDim.value(), channelOffset, channelSize, rewriter,
                                      appendLoc(op->getLoc(), "slice_bias_{0}", sliceIdx));
        mapper.map(op->getOperand(2), biasSlice);
    }

    const auto newOpLoc = appendLoc(op->getLoc(), "slice_{0}", sliceIdx);
    auto clonedOp = rewriter.clone(*op, mapper);
    auto origOutType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto newOpShape = origOutType.getShape().toValues();
    newOpShape[outputChannelDim] = channelSize;
    clonedOp->getResult(0).setType(origOutType.changeShape(ShapeRef(newOpShape)));
    clonedOp->setLoc(newOpLoc);

    return clonedOp;
}
}  // namespace

template <class ConvLike>
bool IE::FuseWithSliceBase<ConvLike>::doesSliceMeetRequirement(IE::SliceOp sliceOp, mlir::Operation* convLikeOp,
                                                               Dim convChannelDim, const Dim& dimToSlice) const {
    auto inShape = getBoundedShape(sliceOp.getSource());
    auto outShape = getBoundedShape(sliceOp.getResult());

    for (auto dim : irange(inShape.size())) {
        if (dimToSlice.ind() == static_cast<int64_t>(dim)) {
            continue;
        }

        if (inShape[Dim(dim)] != outShape[Dim(dim)]) {
            _log.trace("Slice at '{0}' changes non-channel dim {1} ({2} -> {3})", sliceOp->getLoc(), dim,
                       inShape[Dim(dim)], outShape[Dim(dim)]);
            return false;
        }
    }

    return checkAlignedOpConstraints(sliceOp, convLikeOp, convChannelDim, dimToSlice, _log);
}

template <class ConvLike>
mlir::FailureOr<IE::ConsumerChain> IE::FuseWithSliceBase<ConvLike>::hasRequiredSliceUsers(
        mlir::Operation* origOp, const Dim& opChannelDim) const {
    SmallVector<IE::ConsumerChain::SliceWithViewChain> sliceEntries;

    auto nestedLog = _log.nest();
    for (auto user : origOp->getResult(0).getUsers()) {
        auto channelDim = opChannelDim;
        // the view-like chain (if any) leading from origOp to this particular top-level user's Slice(s).
        // Each top-level user is allowed to have its own chain (or no chain at all); chains are not required
        // to match across different top-level users.
        SmallVector<IE::ConsumerChain::OpAndDimPair> localViewChain;
        while (isSupportedViewOp(user)) {
            auto updatedDim = getOutputChannelDimThroughViewOps(user, channelDim, _log);
            if (!updatedDim.has_value()) {
                nestedLog.trace("View-like op '{0}' at '{1}' does not preserve the channel dim through the reshape",
                                user->getName(), user->getLoc());
                return mlir::failure();
            }

            channelDim = updatedDim.value();
            localViewChain.push_back({user, channelDim});

            if (!user->hasOneUse()) {
                nestedLog.trace("View-like op '{0}' at '{1}' has multiple uses; updated dim {2}", user->getName(),
                                user->getLoc(), updatedDim.value());
                break;
            }
            user = *user->getResult(0).getUsers().begin();
        }

        if (auto slice = mlir::dyn_cast_if_present<IE::SliceOp>(user)) {
            if (!doesSliceMeetRequirement(slice, origOp, opChannelDim, channelDim)) {
                nestedLog.trace("Slice user '{0}' does not meet requirement", slice->getLoc());
                return mlir::failure();
            }

            sliceEntries.push_back({slice, std::move(localViewChain)});
            continue;
        }

        if (!isSupportedViewOp(user)) {
            // user op is neither slice, nor a supported view-like op
            nestedLog.trace("User '{0}' at '{1}' is neither a Slice nor a supported view-like op", user->getName(),
                            user->getLoc());
            return mlir::failure();
        }

        if (user->getNumResults() != 1) {
            nestedLog.trace("View-like op '{0}' at '{1}' has multiple results", user->getName(), user->getLoc());
            return mlir::failure();
        }

        if (!user->hasOneUse()) {
            for (auto grandChild : user->getUsers()) {
                auto slice = mlir::dyn_cast_if_present<IE::SliceOp>(grandChild);
                // multi-use view op does not have only slice consumers
                if (slice == nullptr) {
                    nestedLog.trace("Multi-use view-like op '{0}' at '{1}' has a non-Slice consumer '{2}'",
                                    user->getName(), user->getLoc(), grandChild->getName());
                    return mlir::failure();
                }
                if (!doesSliceMeetRequirement(slice, origOp, opChannelDim, channelDim)) {
                    nestedLog.trace("Slice user '{0}' does not meet requirement", slice->getLoc());
                    return mlir::failure();
                }

                sliceEntries.push_back({slice, localViewChain});
            }
        }
    }

    if (sliceEntries.empty()) {
        nestedLog.trace("No Slice users found for op at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    // if slices have overlap, here just skip since we may introduce more computation. Each Slice's channel
    // range is resolved through its own view-like chain (which may differ, or be empty, per Slice).
    if (hasOverlappingChannelRanges(sliceEntries, opChannelDim)) {
        nestedLog.trace("Slice users of op at '{0}' overlap", origOp->getLoc());
        return mlir::failure();
    }

    ConsumerChain result;
    result.slices = std::move(sliceEntries);

    if (!isFusingBeneficial(origOp, result)) {
        nestedLog.trace("Fusing Slice is not beneficial.");
        return mlir::failure();
    }

    return result;
}

template <class ConvLike>
mlir::LogicalResult IE::FuseWithSliceBase<ConvLike>::matchAndRewrite(ConvLike origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto nestedLogger = _log.nest();

    if (!checkOpEligibility(origOp)) {
        return matchFailed(nestedLogger, rewriter, origOp, "Op parameters do not meet requirement");
    }

    // check if all users are slice and all slices' parameters meet requirement
    const auto outputChannelDim = getOutputChannelDim(origOp);
    auto hasRequiredUsers = hasRequiredSliceUsers(origOp, outputChannelDim);
    if (mlir::failed(hasRequiredUsers)) {
        return matchFailed(nestedLogger, rewriter, origOp, "Slice users do not meet requirement");
    }
    const auto& consumers = hasRequiredUsers.value();
    const auto& sliceEntries = consumers.slices;

    for (auto sliceIdx : irange(sliceEntries.size())) {
        const auto& entry = sliceEntries[sliceIdx];
        auto slice = entry.slice;
        const auto& viewChain = entry.viewLikeChain;

        // the slice's own offsets/sizes array is indexed using this Slice's own view chain output dim order,
        // which may differ from origOp's own outputChannelDim (e.g. after a PermuteCast), and may differ from
        // one Slice to another. Each view op in the chain is guaranteed (by getOutputChannelDimThroughViewOps)
        // to preserve the channel dim's size exactly, without merging or splitting it with a neighboring dim,
        // so the offset/size value found at that dim can be carried back as-is to represent the equivalent
        // offset/size along origOp's output channel dim.
        const auto channelDimAtSlice = viewChain.empty() ? outputChannelDim : viewChain.back().second;

        const auto sliceOffsets = parseIntArrayAttr<int64_t>(slice.getStaticOffsets());
        const auto sliceSizes = parseIntArrayAttr<int64_t>(slice.getStaticSizes());

        // back-infer the channel offset/size at the conv-like op's output through this Slice's own view chain
        const auto channelOffset = sliceOffsets[channelDimAtSlice.ind()];
        const auto channelSize = sliceSizes[channelDimAtSlice.ind()];

        auto slicedOp = sliceOpAtOutputChannel(origOp, channelOffset, channelSize, sliceIdx, rewriter);

        // reconstruct this Slice's own view op chain (if any) on top of the newly sliced conv-like op, so the
        // final replacement produces a value with the same type the original slice used to produce
        mlir::Value lastResult = slicedOp->getResult(0);
        for (auto& [viewOp, channelDimAfterOp] : viewChain) {
            mlir::IRMapping viewMapper;
            viewMapper.map(viewOp->getOperand(0), lastResult);
            auto clonedViewOp = rewriter.clone(*viewOp, viewMapper);

            auto origViewOutType = mlir::cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType());
            auto newViewShape = origViewOutType.getShape().toValues();
            newViewShape[channelDimAfterOp] = channelSize;

            adaptReshapeLikeOpShapeAttr(clonedViewOp, newViewShape);

            clonedViewOp->getResult(0).setType(origViewOutType.changeShape(ShapeRef(newViewShape)));
            clonedViewOp->setLoc(appendLoc(viewOp->getLoc(), "slice_{0}", sliceIdx));

            lastResult = clonedViewOp->getResult(0);
        }

        rewriter.replaceOp(slice, lastResult);
    }

    nestedLogger.trace("{0} has been successfully fused with Slice.", origOp->getName());
    return mlir::success();
}

bool IE::FuseConvWithSlice::checkOpEligibility(mlir::Operation* origOp) const {
    auto inShape = getBoundedShape(origOp->getOperand(0));
    if (inShape.size() != 4 || inShape[Dims4D::Act::N] != 1) {
        return false;
    }

    // Weights must be constant
    if (origOp->getOperand(1) == nullptr || origOp->getOperand(1).getDefiningOp<Const::DeclareOp>() == nullptr) {
        return false;
    }

    // Check weights, bias, scale, zp, etc. rank
    for (int64_t idx = 1; idx < origOp->getNumOperands(); idx++) {
        auto operand = origOp->getOperand(idx);
        if (operand == nullptr) {
            continue;
        }

        auto shape = getBoundedShape(operand);
        if (shape.size() != 4) {
            return false;
        }
    }

    return true;
}

mlir::Operation* IE::FuseConvWithSlice::sliceOpAtOutputChannel(IE::ConvolutionOp op, int64_t channelOffset,
                                                               int64_t channelSize, int64_t sliceIdx,
                                                               mlir::PatternRewriter& rewriter) const {
    mlir::IRMapping mapper;
    const auto weightsChannelDim = getWeightsChannelDim(op);
    const auto outputChannelDim = getOutputChannelDim(op);

    // slice filter
    auto filterSlice = sliceOperand(op.getFilter(), weightsChannelDim, channelOffset, channelSize, rewriter,
                                    appendLoc(op.getLoc(), "slice_filter_{0}", sliceIdx));
    mapper.map(op.getFilter(), filterSlice);

    if (op.getBias() != nullptr) {
        auto biasSlice = sliceOperand(op.getBias(), outputChannelDim, channelOffset, channelSize, rewriter,
                                      appendLoc(op.getLoc(), "slice_bias_{0}", sliceIdx));
        mapper.map(op.getBias(), biasSlice);
    }

    if (op.getScale() != nullptr) {
        auto scaleSlice = sliceOperand(op.getScale(), weightsChannelDim, channelOffset, channelSize, rewriter,
                                       appendLoc(op.getLoc(), "slice_scale_{0}", sliceIdx));
        mapper.map(op.getScale(), scaleSlice);
    }

    if (op.getZeroPoints() != nullptr) {
        auto zpSlice = sliceOperand(op.getZeroPoints(), weightsChannelDim, channelOffset, channelSize, rewriter,
                                    appendLoc(op.getLoc(), "slice_zp_{0}", sliceIdx));
        mapper.map(op.getZeroPoints(), zpSlice);
    }

    const auto newConvLoc = appendLoc(op.getLoc(), "slice_{0}", sliceIdx);
    auto clonedOp = rewriter.clone(*op.getOperation(), mapper);
    auto origOutType = mlir::cast<vpux::NDTypeInterface>(op.getResult().getType());
    auto newConvShape = origOutType.getShape().toValues();
    newConvShape[outputChannelDim] = channelSize;
    clonedOp->getResult(0).setType(origOutType.changeShape(ShapeRef(newConvShape)));
    clonedOp->setLoc(newConvLoc);

    return clonedOp;
}

bool IE::FuseMatMulWithSlice::checkOpEligibility(mlir::Operation* origOp) const {
    const auto filterRank = mlir::cast<NDTypeInterface>(origOp->getOperand(1).getType()).getRank();
    if (filterRank < 2) {
        return false;
    }

    const auto isDataBound = IE::isDataBoundTokenDim(origOp);
    // If op is data bound, further splitting the ops will be damaging to performance
    return isDataBound.has_value() && !isDataBound.value();
}

bool IE::FuseMatMulWithSlice::isFusingBeneficial(mlir::Operation* origOp, const ConsumerChain& consumers) const {
    if (consumers.slices.size() == 1) {
        // If there is only 1 Slice end consumer, we replace 1 larger op with 1 smaller op (with unnecessary computation
        // trimmed), therefore it should be beneficial
        return true;
    }

    // Otherwise, we must ensure there is a gain from splitting the operation.
    // One such gain may come from fusing an activation function found on one of the slice branches.
    auto layerWithPostOp = mlir::dyn_cast_if_present<IE::LayerWithPostOpInterface>(origOp);
    if (layerWithPostOp == nullptr) {
        return false;
    }
    return hasBeneficialFusableActivationDownstream(layerWithPostOp, consumers, _log.nest());
}

mlir::Operation* IE::FuseMatMulWithSlice::sliceOpAtOutputChannel(IE::MatMulOp op, int64_t channelOffset,
                                                                 int64_t channelSize, int64_t sliceIdx,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto weightsChannelDim = getWeightsChannelDim(op);
    const auto outputChannelDim = getOutputChannelDim(op);
    return sliceOpAtOutputChannelForMatMulLikeops(op, outputChannelDim, weightsChannelDim, channelOffset, channelSize,
                                                  sliceIdx, rewriter);
}

bool IE::FuseFullyConnectedWithSlice::checkOpEligibility(mlir::Operation* origOp) const {
    const auto isDataBound = IE::isDataBoundTokenDim(origOp);
    // Op is data bound, slicing it further is detrimental to performance
    if (!isDataBound.has_value() || isDataBound.value()) {
        return false;
    }

    auto fullyConnected = mlir::cast<IE::FullyConnectedOp>(origOp);
    auto bias = fullyConnected.getBias();
    if (bias == nullptr) {
        return true;
    }

    const auto outputChannelDim = getOutputChannelDim(fullyConnected);
    // check we can retrieve a valid channel dim for, if it exists
    return getFullyConnectedBiasChannelDim(fullyConnected, outputChannelDim).has_value();
}

bool IE::FuseFullyConnectedWithSlice::isFusingBeneficial(mlir::Operation* origOp,
                                                         const ConsumerChain& consumers) const {
    if (consumers.slices.size() == 1) {
        // if there is only 1 Slice end consumer, we replace 1 larger op with 1 smaller op (with unnecessary computation
        // trimmed), therefore it should be beneficial
        return true;
    }

    // Otherwise, we must ensure there is a gain from splitting the operation.
    // One such gain may come from fusing an activation function found on one of the slice branches.
    auto layerWithPostOp = mlir::dyn_cast_if_present<IE::LayerWithPostOpInterface>(origOp);
    if (layerWithPostOp == nullptr) {
        return false;
    }
    return hasBeneficialFusableActivationDownstream(layerWithPostOp, consumers, _log.nest());
}

mlir::Operation* IE::FuseFullyConnectedWithSlice::sliceOpAtOutputChannel(IE::FullyConnectedOp op, int64_t channelOffset,
                                                                         int64_t channelSize, int64_t sliceIdx,
                                                                         mlir::PatternRewriter& rewriter) const {
    const auto weightsChannelDim = getWeightsChannelDim(op);
    const auto outputChannelDim = getOutputChannelDim(op);
    return sliceOpAtOutputChannelForMatMulLikeops(op, outputChannelDim, weightsChannelDim, channelOffset, channelSize,
                                                  sliceIdx, rewriter);
}

// Explicit template instantiations
//
// FuseWithSliceBase<ConvLike>'s out-of-line members (matchAndRewrite, doesSliceMeetRequirement,
// hasRequiredSliceUsers) are defined only in this translation unit. The derived, concrete patterns
// (FuseConvWithSlice, FuseMatMulWithSlice, FuseFullyConnectedWithSlice) are instantiated in other .cpp
// files, which would otherwise be left with unresolved references to those base class members.
template class IE::FuseWithSliceBase<IE::ConvolutionOp>;
template class IE::FuseWithSliceBase<IE::MatMulOp>;
template class IE::FuseWithSliceBase<IE::FullyConnectedOp>;
