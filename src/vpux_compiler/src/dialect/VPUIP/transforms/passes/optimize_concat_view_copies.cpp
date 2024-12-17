//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

namespace {

//
// AvoidConcatExtraChannel
//

struct InputUpdateInfo {
    VPUIP::CopyOp distributedCopyOp;
    SmallVector<int64_t, 4> copyInOffsets;
    SmallVector<int64_t, 4> copyOutOffsets;
    SmallVector<int64_t, 4> copySizes;
};

class AvoidConcatExtraChannel : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    AvoidConcatExtraChannel(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult checkConcatUsers(mlir::Value concatOutput, std::optional<int64_t>& patternOutChannelSize,
                                         std::optional<int64_t>& patternOutChannelOffset) const;
    mlir::LogicalResult checkConcatInputs(mlir::ValueRange concatInputs, mlir::Value concatOutput,
                                          const int64_t patternOutChannelSize, const int64_t patternOutChannelOffset,
                                          SmallVector<InputUpdateInfo>& inTilingCopiesInfo) const;

    mlir::Operation* createOutputBuffer(mlir::PatternRewriter& rewriter, VPUIP::CopyOp copyOp, int64_t channels) const;

    Logger _log;
};

// Check if all Concat users are Subview with same channels slice
// less than Concat channels (m > n)
//
//                      Concat (m output channels)
//                          |       ...       |
//  Subview (n output channels)     ...      Subview (n output channels)
//
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatUsers(mlir::Value concatOutput,
                                                              std::optional<int64_t>& patternOutChannelSize,
                                                              std::optional<int64_t>& patternOutChannelOffset) const {
    auto subviews = concatOutput.getUsers();
    if (subviews.empty()) {
        return mlir::failure();
    }

    const auto concatOutChannelSize = getShape(concatOutput)[Dims4D::Act::C];
    for (const auto user : subviews) {
        auto subview = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(user);

        if (subview == nullptr) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.getStaticOffsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());

        if (subview.getStaticStrides().has_value()) {
            return mlir::failure();
        }

        if (patternOutChannelOffset.has_value() && patternOutChannelOffset.value() != offsets[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (patternOutChannelSize.has_value() && patternOutChannelSize.value() != sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (concatOutChannelSize <= sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        patternOutChannelSize = sizes[Dims4D::Act::C.ind()];
        patternOutChannelOffset = offsets[Dims4D::Act::C.ind()];
    }

    return mlir::success();
}

// Check if all Concat inputs copy NCE result with more channels
// than Subview after Concat
//
// Scenario 1: Concat joins its inputs not by channel dimension (m > n)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)     TilingCopy (m channels)
//            |               |          |                 |
//    Subview (m)          Concat (m channels)          Subview (m)
//                                 |
//                        Subview (n channels)
//
// Scenario 2: Concat joins its inputs by channel dimension (p > m && p > n && p > q)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)      TilingCopy (n channels)
//            |               |          |                  |
//    Subview (m)          Concat (p channels)           Subview (n)
//                                  |
//                         Subview (q channels)
//
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatInputs(mlir::ValueRange concatInputs, mlir::Value concatOutput,
                                                               const int64_t patternOutChannelSize,
                                                               const int64_t patternOutChannelOffset,
                                                               SmallVector<InputUpdateInfo>& inTilingCopiesInfo) const {
    if (concatInputs.empty() || concatOutput == nullptr) {
        return mlir::failure();
    }

    auto getConcatDims = [](ShapeRef inShape, ShapeRef outShape) {
        VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Got unexpect input and output shape");
        SmallVector<Dim> concatDims;
        auto ioShapes = zip(inShape, outShape);
        for (const auto& ioShape : ioShapes | indexed) {
            const auto inSize = std::get<0>(ioShape.value());
            const auto outSize = std::get<1>(ioShape.value());
            if (inSize != outSize) {
                concatDims.push_back(Dim(ioShape.index()));
            }
        }
        return concatDims;
    };

    const auto concatOutShape = getShape(concatOutput);
    const auto concatChannels = concatOutShape[Dims4D::Act::C];
    for (auto input : concatInputs) {
        auto tilingCopy = input.getDefiningOp<VPUIP::CopyOp>();

        if (!tilingCopy || !tilingCopy->getResult(0).hasOneUse()) {
            return mlir::failure();
        }

        // DistributionCopy is the output of the NCE task
        // If NCE task uses the SplitOverKernel strategy, it is illegal to optimize the channel
        if (auto distributedCopyType = tilingCopy.getInput().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            const auto distributionInfo = distributedCopyType.getDistribution();
            if (distributionInfo.getMode().getValue() == VPU::DistributionMode::SEGMENTED) {
                const auto numTiles = parseIntArrayAttr<int64_t>(distributionInfo.getNumTiles());
                if (numTiles[Dims4D::Act::C.ind()] != 1) {
                    return mlir::failure();
                }
            }
        }

        auto copyOpOutput = tilingCopy.getOutputs()[0];
        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();
        if (subview == nullptr) {
            return mlir::failure();
        }

        if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(subview.getSource()) == nullptr) {
            return mlir::failure();
        }

        if (subview.getStaticStrides().has_value()) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.getStaticOffsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());

        auto concatDims = getConcatDims(Shape(sizes), concatOutShape);
        if (concatDims.size() != 1) {
            return mlir::failure();
        }

        SmallVector<int64_t, 4> copyInOffsets(4, 0);
        SmallVector<int64_t, 4> copyOutOffsets(offsets.begin(), offsets.end());
        SmallVector<int64_t, 4> copySizes(sizes.begin(), sizes.end());
        const auto channelIdx = Dims4D::Act::C.ind();
        if (concatDims.front() == Dims4D::Act::C) {
            const auto currentInputChannelBegin = offsets[channelIdx];
            const auto currentInputChannelEnd = currentInputChannelBegin + sizes[channelIdx];
            const auto patternChannelBegin = patternOutChannelOffset;
            const auto patternChannelEnd = patternChannelBegin + patternOutChannelSize;

            bool isPatternBeginInside =
                    patternChannelBegin >= currentInputChannelBegin && patternChannelBegin <= currentInputChannelEnd;
            bool isPatternEndInside =
                    patternChannelEnd >= currentInputChannelBegin && patternChannelEnd <= currentInputChannelEnd;

            copyOutOffsets[channelIdx] = currentInputChannelBegin - patternChannelBegin;
            // ConcatView Across Channels:
            // |   Input_0   |   Input_1   |   Input_2   |   ...   |   Input_n   |
            // When the slice starts inside Input_0, it's legal scenario:
            //    | Input_0' |   Input_1   |   Input_2   |   ...   |   Input_n   |
            // When the slice starts inside an input other than Input_0, it is illegal scenario:
            //                   | Input_1'|   Input_2   |   ...   |   Input_n   |
            // When both the pattern start and end are inside an input, it is illegal scenario:
            //   |Input_0'|
            // For these two illegal cases, some inputs are unnecessary and can be removed
            // Although unlikely to appear in real models, it's better to have checks in place
            if (isPatternBeginInside) {
                if (currentInputChannelBegin != 0 || isPatternEndInside) {
                    return mlir::failure();
                }
                copyInOffsets[channelIdx] = patternChannelBegin;
                copyOutOffsets[channelIdx] = 0;
                copySizes[channelIdx] = sizes[channelIdx] - patternChannelBegin;
            }

            // Similar logic to isPatternBeginInside scenario
            // The slice must start in the first input and end in the last input
            if (isPatternEndInside) {
                if (currentInputChannelEnd != concatChannels || isPatternBeginInside) {
                    return mlir::failure();
                }
                copyInOffsets[channelIdx] = 0;
                copySizes[channelIdx] = patternChannelEnd - currentInputChannelBegin;
            }
        } else {
            copyInOffsets[channelIdx] = patternOutChannelOffset;
            copySizes[channelIdx] = patternOutChannelSize;
        }

        auto copyOpInput = tilingCopy.getInputs()[0];
        if (auto distributedType = copyOpInput.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
            if (tileIndex.has_value()) {
                auto tileIndexVal = tileIndex.value();
                if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithClusterCopy(to_small_vector(copyInOffsets),
                                                                                tileIndexVal, distributedType)) {
                    return mlir::failure();
                }
            }
        }

        inTilingCopiesInfo.push_back(
                InputUpdateInfo{tilingCopy, std::move(copyInOffsets), std::move(copyOutOffsets), std::move(copySizes)});
    }

    return mlir::success();
}

mlir::Operation* AvoidConcatExtraChannel::createOutputBuffer(mlir::PatternRewriter& rewriter, VPUIP::CopyOp copyOp,
                                                             int64_t channels) const {
    auto copyOpOutput = copyOp.getOutputs()[0];

    auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

    auto opOutputType = subview.getSource().getType().cast<vpux::NDTypeInterface>();
    auto sourceShape = opOutputType.getShape().toValues();
    sourceShape[Dims4D::Act::C] = channels;
    auto newOpOutputType = opOutputType.changeShape(ShapeRef(sourceShape));

    return allocateBuffersOfType(_log, copyOp->getLoc(), rewriter, newOpOutputType).front().getDefiningOp();
}

void recursivelyInferReturnTypes(mlir::Value value) {
    for (auto child : value.getUsers()) {
        if (mlir::isa_and_nonnull<VPUIP::SubViewOp, VPUIP::ShapeCastOp>(child)) {
            vpux::inferReturnTypes(child, vpux::InferShapedTypeMode::ALL);
            recursivelyInferReturnTypes(child->getResult(0));
        }
    }
}

// Scenario 1: Concat joins its inputs not by channel dimension (m > n)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)     TilingCopy (m channels)
//            |               |          |                 |
//    Subview (m)          Concat (m channels)          Subview (m)
//                                 |
//                        Subview (n channels)
//
// is converted to pattern
//
//        Input0 (m channels)             Input1 (m channels)
//                  |                            |
//        Subview (n channels)           Subview (n channels)
//                  |                            |
//        TilingCopy (n channels)     TilingCopy (n channels)
//           |               |            |            |
//    Subview (n)         Concat (n channels)        Subview (n)
//
// Scenario 2: Concat joins its inputs by channel dimension (p > m && p > n && p > q)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)      TilingCopy (n channels)
//            |               |          |                  |
//    Subview (m)          Concat (p channels)           Subview (n)
//                                  |
//                         Subview (q channels)
//
// is converted to pattern
//
//        Input0 (m channels)     Input1 (n channels)
//                |                         |
//                |               Subview (n - (p - q) channels)
//                |                         |
//    TilingCopy (m channels)     TilingCopy (n - (p - q) channels)
//           |           |           |           |
//    Subview (m)      Concat (q channels)    Subview (n - (p - q))
//
mlir::LogicalResult AvoidConcatExtraChannel::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got VPUIP.ConcatViewOp at '{0}'", concatOp->getLoc());
    auto nestedLogger = _log.nest();

    auto concatOutput = concatOp.getOutput();
    if (getShape(concatOutput).size() != 4) {
        nestedLogger.trace("Cannot optimize because of shape rank not being 4");
        return mlir::failure();
    }

    std::optional<int64_t> patternOutChannelSize = std::nullopt;
    std::optional<int64_t> patternOutChannelOffset = std::nullopt;
    if (checkConcatUsers(concatOutput, patternOutChannelSize, patternOutChannelOffset).failed()) {
        nestedLogger.trace("Cannot optimize because of users requirements");
        return mlir::failure();
    }

    auto concatInputs = concatOp.getInputs();
    SmallVector<InputUpdateInfo> inTilingCopiesInfo;
    inTilingCopiesInfo.reserve(concatInputs.size());
    if (checkConcatInputs(concatInputs, concatOutput, patternOutChannelSize.value(), patternOutChannelOffset.value(),
                          inTilingCopiesInfo)
                .failed()) {
        nestedLogger.trace("Cannot optimize because of input requirements");
        return mlir::failure();
    }

    auto* outputBuffer =
            createOutputBuffer(rewriter, inTilingCopiesInfo.front().distributedCopyOp, patternOutChannelSize.value());
    if (outputBuffer == nullptr) {
        nestedLogger.trace("Cannot allocate new output buffer");
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatInputs.size());
    for (auto inTilingCopyInfo : inTilingCopiesInfo) {
        auto copyOp = inTilingCopyInfo.distributedCopyOp;
        auto copyOpInput = copyOp.getInputs()[0];
        auto copyOpOutput = copyOp.getOutputs()[0];

        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

        auto newCopyInSubview = copyOpInput;
        if (Shape(inTilingCopyInfo.copySizes) != getShape(copyOpInput)) {
            newCopyInSubview = rewriter.create<VPUIP::SubViewOp>(
                    subview.getLoc(), copyOpInput,
                    getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copyInOffsets),
                    getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copySizes));
        }

        auto newCopyOutSubview = rewriter.create<VPUIP::SubViewOp>(
                subview.getLoc(), outputBuffer->getResult(0),
                getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copyOutOffsets),
                getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copySizes));

        auto newTilingCopy = rewriter.create<VPUIP::CopyOp>(copyOp.getLoc(), newCopyInSubview, newCopyOutSubview);

        newConcatInputs.push_back(newTilingCopy.getResult());
    }

    auto newOp =
            rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(concatOp, newConcatInputs, outputBuffer->getResult(0));

    for (auto user : newOp.getOutput().getUsers()) {
        if (auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
            auto newOffsets = parseIntArrayAttr<int64_t>(subviewOp.getStaticOffsetsAttr());
            newOffsets[Dims4D::Act::C.ind()] = 0;
            auto newOffsetsAttr = getIntArrayAttr(subviewOp.getContext(), ArrayRef(newOffsets));
            subviewOp->setAttr(subviewOp.getStaticOffsetsAttrName(), newOffsetsAttr);
            vpux::inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);

            recursivelyInferReturnTypes(subviewOp);
        }
    }

    for (auto inTilingCopyInfo : inTilingCopiesInfo) {
        rewriter.eraseOp(inTilingCopyInfo.distributedCopyOp);
    }

    nestedLogger.trace("Successfully Avoid Concat Extra Channel");
    return mlir::success();
}

//
// FuseConcatView
//

/*
    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp
               \                 /
                ConcatView (DDR)
                        |
                CopyOp(DDR2DDR)      TilingCopyOp/CopyOp
                        \              /
                        ConcatView (DDR)


    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp     TilingCopyOp/CopyOp
                     \                 |                  /
                                ConcatView (DDR)
*/

class FuseConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseConcatView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    bool isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasOneDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter,
                                                vpux::Logger log) const;

private:
    Logger _log;
};

bool FuseConcatView::hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    log.nest().trace("Checking hasCopyOpForAllInputs");

    auto isCopyOpWithSingleUser = [&log](mlir::Operation* op) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(op)) {
            if (!mlir::isa<VPUIP::SubViewOp>(copyOp.getOutputBuff().getDefiningOp())) {
                log.nest().nest().trace("Parent CopyOp's output buffer is not defined by a SubViewOp: '{0}'",
                                        copyOp->getLoc());
                return false;
            }

            return copyOp.getOutput().hasOneUse();
        }

        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(op)) {
            if (!vpux::VPUIP::hasDistributedOperand(copyOp)) {
                log.nest().nest().trace("ConcatView input is not a distributed Copy op: '{0}'", copyOp->getLoc());
                return false;
            }

            if (!mlir::isa<VPUIP::SubViewOp>(copyOp.getOutputBuff().getDefiningOp())) {
                log.nest().nest().trace("Parent distributed CopyOp output buffer is not defined by a SubViewOp: '{0}'",
                                        copyOp->getLoc());
                return false;
            }

            return copyOp->hasOneUse();
        }

        log.nest().nest().trace("ConcatView input is not Copy op: '{0}'", op->getLoc());
        return false;
    };

    return llvm::all_of(concatViewOp.getInputs(), [&](auto input) {
        return isCopyOpWithSingleUser(input.getDefiningOp());
    });
}

bool FuseConcatView::hasOneDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    log.nest().trace("Checking hasOneDDR2DDRCopyWithConcatViewConsumer");

    if (!concatViewOp.getOutput().hasOneUse()) {
        log.nest().nest().trace("ConcatView Op has more than one user");
        return false;
    }

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*concatViewOp.getOutput().getUsers().begin());
    if (!copyOp) {
        log.nest().nest().trace("Consumer of ConcatView Op is not Copy Op");
        return false;
    }

    if (!copyOp.getOutput().hasOneUse()) {
        log.nest().nest().trace("CopyOp Op no user or has more than one user");
        return false;
    }

    if (!mlir::isa<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin())) {
        log.nest().nest().trace("Consumer of Copy Op is not ConcatView Op");
        return false;
    }

    return VPUIP::isCopyFromDDR(copyOp) && VPUIP::isCopyToDDR(copyOp);
}

// Fuse ConcatView Ops to remove unnecessary copies, two conditions need to be satisfied:
// a) The Stride Level for each ConcatView input (after fusing) should be no more than 2;
//     It's a runtime and HW limitation in order to get the right NNDMA descriptor, we support a maximum of 3D DMA
//     transfers with 2 levels of striding.
// b) The number of inputs from the second ConcatView, which come from the output of the first should no more than 1;
//     For example, first ConcatView has M inputs, second ConcatView has N inputs, out of which P of them are the output
//     of the first ConcatView After fusing, the number of input copies is: M * P + (N - P)
//     Can't ensure we get benefit when P is of a large size. Limit optimization to P=1.
bool FuseConcatView::isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    if (concatViewOp.getOutput().use_empty()) {
        log.nest().trace("Cannot find user copy op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasCopyOpForAllInputs(concatViewOp, log)) {
        log.nest().trace("Not all inputs is CopyOp for first ConcatViewOp at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasOneDDR2DDRCopyWithConcatViewConsumer(concatViewOp, log)) {
        log.nest().trace("Not only one user is DDR2DDR copy with ConcatViewOp for op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    log.nest().trace("FuseConcatView: Found legal ConcatView pattern at op '{0}'", concatViewOp->getLoc());

    return true;
}

mlir::LogicalResult FuseConcatView::fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp,
                                                            mlir::PatternRewriter& rewriter, vpux::Logger log) const {
    // Get current concat's memref.alloc op, which will be removed
    auto firstConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.getOutputBuff());
    if (firstConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because current concat '{0}' output isn't master buffer",
                         concatViewOp->getLoc());
        return mlir::failure();
    }

    // Get Copy and next ConcatView Op
    auto outputCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*concatViewOp.getOutput().getUsers().begin());
    VPUX_THROW_UNLESS(outputCopyOp != nullptr, "Cannot get DDR to DDR Copy Op after '{0}'", concatViewOp->getLoc());
    VPUIP::SubViewOp outCopySubView = outputCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

    auto nextConcatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*outputCopyOp.getOutput().getUsers().begin());
    VPUX_THROW_UNLESS(nextConcatViewOp != nullptr, "Cannot get second ConcatView Op");

    auto nextConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(nextConcatViewOp.getOutputBuff());
    if (nextConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because next concat '{0}' output isn't master buffer",
                         nextConcatViewOp->getLoc());
        return mlir::failure();
    }

    // Create an array of the new input copy ops
    SmallVector<mlir::Value> newCopyInputs;
    SmallVector<mlir::Value> oldCopyInputs;
    SmallVector<VPUIP::SubViewOp> oldSubViewInputs;
    newCopyInputs.reserve(concatViewOp.getInputs().size() + nextConcatViewOp.getInputs().size() - 1);
    oldCopyInputs.reserve(concatViewOp.getInputs().size());
    oldSubViewInputs.reserve(concatViewOp.getInputs().size());

    auto isStrideConcat = [](VPUIP::SubViewOp subView) {
        if (subView.getStaticStridesAttr() == nullptr) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(subView.getStaticStridesAttr());
        return llvm::any_of(strides, [](auto stride) {
            return stride > 1;
        });
    };

    for (size_t nextInIdx = 0; nextInIdx < nextConcatViewOp.getInputs().size(); ++nextInIdx) {
        auto siblingCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextConcatViewOp.getInputs()[nextInIdx].getDefiningOp());
        if (!(siblingCopyOp && siblingCopyOp == outputCopyOp)) {
            newCopyInputs.push_back(nextConcatViewOp.getInputs()[nextInIdx]);
            continue;
        }

        SmallVector<int64_t> outCopyOffsets = parseIntArrayAttr<int64_t>(outCopySubView.getStaticOffsetsAttr());
        SmallVector<int64_t> outCopySizes = parseIntArrayAttr<int64_t>(outCopySubView.getStaticSizesAttr());
        if (isStrideConcat(outCopySubView)) {
            log.nest().trace("Fusing Concat Op with stride has no performance benefits");
            return mlir::failure();
        }

        for (size_t firstInIdx = 0; firstInIdx < concatViewOp.getInputs().size(); ++firstInIdx) {
            auto op = concatViewOp.getInputs()[firstInIdx].getDefiningOp();

            VPUIP::SubViewOp inCopySubView;
            auto inCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(op);
            if (inCopyOp) {
                inCopySubView = inCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
            }

            VPUX_THROW_WHEN(inCopySubView == nullptr, "Cannot get SubViewOp");
            oldCopyInputs.push_back(concatViewOp.getInputs()[firstInIdx]);
            oldSubViewInputs.push_back(inCopySubView);

            SmallVector<int64_t> inCopyOffsets = parseIntArrayAttr<int64_t>(inCopySubView.getStaticOffsetsAttr());
            SmallVector<int64_t> inCopySizes = parseIntArrayAttr<int64_t>(inCopySubView.getStaticSizesAttr());

            VPUX_THROW_WHEN(outCopyOffsets.size() != inCopyOffsets.size() || outCopySizes.size() != inCopySizes.size(),
                            "Input and output copy subviews have different-sized attributes");

            SmallVector<int64_t> newCopyOffsets(outCopyOffsets.size());
            SmallVector<int64_t> newCopySizes(outCopySizes.size());

            SmallVector<int64_t> newCopyStrides(inCopyOffsets.size(), 1);
            auto inCopyStrides = inCopySubView.getStaticStridesAttr();
            if (inCopyStrides != nullptr) {
                newCopyStrides = parseIntArrayAttr<int64_t>(inCopyStrides);
            }

            for (size_t idx = 0; idx < newCopyOffsets.size(); ++idx) {
                newCopySizes[idx] = inCopySizes[idx];
                newCopyOffsets[idx] = outCopyOffsets[idx] + inCopyOffsets[idx];
            }

            auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(outCopySubView->getLoc(), outCopySubView.getSource(),
                                                                  newCopyOffsets, newCopySizes, newCopyStrides);
            if (newSubViewOp->isBeforeInBlock(nextConcatMemAlloc)) {
                nextConcatMemAlloc->moveBefore(newSubViewOp);
            }
            auto newCopyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), op->getOperand(0), newSubViewOp.getResult());

            if (!VPUIP::hasLegalStridingLevel(newCopyOp)) {
                log.nest().trace("DMA Striding Level is illegal. Fusing Concat Op have no benefit");
                rewriter.eraseOp(newCopyOp);
                rewriter.eraseOp(newSubViewOp);
                return mlir::failure();
            }
            newCopyInputs.push_back(newCopyOp.getOutput());
        }
    }

    rewriter.setInsertionPoint(nextConcatViewOp);
    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(nextConcatViewOp, nextConcatViewOp.getOutput().getType(),
                                                     newCopyInputs, nextConcatViewOp.getOutputBuff());

    // Erase the old hanging structure
    rewriter.eraseOp(outputCopyOp);
    rewriter.eraseOp(outCopySubView);
    rewriter.eraseOp(concatViewOp);

    for (size_t inIdx = 0; inIdx < oldCopyInputs.size(); ++inIdx) {
        rewriter.eraseOp(oldCopyInputs[inIdx].getDefiningOp());
        rewriter.eraseOp(oldSubViewInputs[inIdx]);
    }

    rewriter.eraseOp(firstConcatMemAlloc);

    return mlir::success();
}

mlir::LogicalResult FuseConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!isLegalConcatViewPattern(concatViewOp, _log)) {
        _log.nest().trace("FuseConcatView: Cannot rewrite this concat Op");
        return mlir::failure();
    }

    return fuseTwoConcatViewInputs(concatViewOp, rewriter, _log);
}

// RemoveDDRToDDRCopyAfterConcatView
//

/*
            CopyOp     ...      CopyOp
               \                 /
                ConcatView (DDR)
                        |
                (Pure View Ops)
                        |
                CopyOp(DDR2DDR)

Optimized:
            CopyOp     ...      CopyOp
               \                 /
                ConcatView (DDR)
                        |
                (Pure View Ops)
*/

class RemoveDDRToDDRCopyAfterConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    RemoveDDRToDDRCopyAfterConcatView(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    mlir::Operation* getTargetCopyOp(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Operation* RemoveDDRToDDRCopyAfterConcatView::getTargetCopyOp(VPUIP::ConcatViewOp concatViewOp,
                                                                    vpux::Logger log) const {
    log.nest().trace("Checking ConcatView Copy pattern");

    auto childOp = *concatViewOp.getOutput().getUsers().begin();
    while (childOp != nullptr && mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp,
                                           VPUIP::ShapeCastOp, VPUIP::CopyOp>(childOp)) {
        _log.trace("childOp location: {0}", childOp->getLoc());
        if (!childOp->getResult(0).hasOneUse()) {
            log.nest().trace("child op user does not match");
            return nullptr;
        } else if (mlir::isa<VPUIP::CopyOp>(childOp) && !vpux::VPUIP::hasDistributedOperand(childOp)) {
            log.nest().trace("childOp is a CopyOp");
            return childOp;
        } else {
            childOp = *childOp->getResult(0).getUsers().begin();
            log.nest().trace("Returning childOp result user");
        }
    }
    log.nest().trace("Could not find ConcatView Copy pattern");
    return nullptr;
}

mlir::LogicalResult RemoveDDRToDDRCopyAfterConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("RemoveDDRToDDRCopyAfterConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!concatViewOp.getOutput().hasOneUse()) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Only support ConcatView has one user");
        return mlir::failure();
    }
    auto targetOp = getTargetCopyOp(concatViewOp, _log);
    if (targetOp == nullptr) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Cannot find the target Copy Op");
        return mlir::failure();
    }
    auto targetCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(targetOp);
    if (!VPUIP::isCopyToDDR(targetCopyOp) || !VPUIP::isCopyFromDDR(targetCopyOp)) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Target Copy Op is not from DDR to DDR");
        return mlir::failure();
    }

    // Check if the CopyOp copies to output
    if (targetCopyOp.getOutputBuff().isa<mlir::BlockArgument>()) {
        _log.trace("RemoveDDRToDDRCopyAfterConcatView: Cannot rewrite because it is last copy");
        return mlir::failure();
    }

    VPUIP::SubViewOp outCopySubView = targetCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
    if (outCopySubView != nullptr) {
        _log.nest().trace("Cannot remove copy op with subView");
        return mlir::failure();
    }

    targetCopyOp.getOutput().replaceAllUsesWith(targetCopyOp.getInput());
    rewriter.eraseOp(targetCopyOp);
    _log.trace("Successfully removed redundant copy Op after ConcatView");
    return mlir::success();
}

//
// OptimizeDDR2DDRCopyInputsOfConcatView
//

/*
    Move ConcatView from DDR to CMX when inputs and output DistributedCopy is Duplicated.
    TODO: Support more case when ConcatView has non-distributed CopyOp user, see E#102977

    Convert below pattern:

     DistributedCopy     ...     CopyOp
        (CMX -> DDR)          (DDR -> DDR)
               \                /
                ConcatView (DDR)
                        |
                (Pure View Ops)
                        |
                  DistributedCopy
                   (DDR -> CMX)
                        |

    to:

       DistributedCopy
        (CMX -> DDR)
             |
          AllocOp (DDR)
             |
       DistributedCopy  ...    DistributedCopy
        (DDR -> CMX)            (DDR -> CMX)
               \                /
                ConcatView (CMX)
                        |
                (Pure View Ops) (CMX)
                        |
                DistributedCast
                        |

    So that DDR2DDR copy inputs can be optimized.
*/

struct ConcatInputs {
    SmallVector<mlir::Value> inputCopies;
    SmallVector<mlir::Value> inputDistributedCopies;
};

struct ConcatOutputs {
    SmallVector<mlir::Operation*> viewLikeOps;
    VPUIP::CopyOp outputDistributedCopy;
};

struct ConcatOutputsOfSubViewCopyUsers {
    VPUIP::PermuteCastOp permuteCastOp;
    SmallVector<VPUIP::SubViewOp> outputSubViewOps;
    SmallVector<VPUIP::CopyOp> outputDistributedCopyOps;
};

class OptimizeDDR2DDRCopyInputsOfConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    OptimizeDDR2DDRCopyInputsOfConcatView(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
        setDebugName("OptimizeDDR2DDRCopyInputsOfConcatView");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    mlir::FailureOr<mlir::Operation*> searchCopyOpThroughViewLikeOps(VPUIP::ConcatViewOp concatViewOp,
                                                                     SmallVector<mlir::Operation*>& viewLikeOps) const;

    mlir::FailureOr<ConcatInputs> getValidConcatInputs(VPUIP::ConcatViewOp concatViewOp) const;
    mlir::FailureOr<ConcatOutputs> getValidConcatOutputs(VPUIP::ConcatViewOp concatViewOp) const;

    void convertCopyInputAndStore(ArrayRef<mlir::Value> inputCopies, mlir::Value outputBuffer,
                                  SmallVector<mlir::Value>& newConcatInputs, mlir::PatternRewriter& rewriter) const;
    void convertDistributedCopyInputAndStore(ArrayRef<mlir::Value> inputDistributedCopies, mlir::Value outputBuffer,
                                             SmallVector<mlir::Value>& newConcatInputs,
                                             mlir::PatternRewriter& rewriter) const;

    VPUIP::DistributedBufferType getDuplicatedDistributedType(NDTypeInterface ndType,
                                                              VPUIP::DistributedBufferType distributedType,
                                                              mlir::MLIRContext* ctx) const;
    mlir::Value rewriteViewLikeOps(mlir::Value input, ArrayRef<mlir::Operation*> viewLikeOps,
                                   VPUIP::DistributedBufferType origOutputBufferType,
                                   mlir::PatternRewriter& rewriter) const;

    mlir::FailureOr<ConcatOutputsOfSubViewCopyUsers> searchSubViewCopyUsersThroughPermuteCast(
            VPUIP::ConcatViewOp concatViewOp) const;
    mlir::FailureOr<ConcatOutputsOfSubViewCopyUsers> getValidConcatOutputsOfSubViewCopyUsers(
            VPUIP::ConcatViewOp concatViewOp) const;
    mlir::LogicalResult processConcatOutputsOfSubViewCopyUsers(VPUIP::ConcatViewOp concatViewOp,
                                                               const ConcatInputs& concatInputs,
                                                               const ConcatOutputsOfSubViewCopyUsers& concatOutputs,
                                                               mlir::PatternRewriter& rewriter) const;
};

VPUIP::DistributedBufferType OptimizeDDR2DDRCopyInputsOfConcatView::getDuplicatedDistributedType(
        NDTypeInterface ndType, VPUIP::DistributedBufferType distributedType, mlir::MLIRContext* ctx) const {
    const auto orderMap = mlir::AffineMapAttr::get(ndType.getDimsOrder().toAffineMap(ctx));
    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();

    auto distribution = distributedType.getDistribution();
    auto memSpace = distributedType.getMemSpace();

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                        "DistributedBufferType is not DUPLICATED, type = {0}", distributedType);

        auto newDistribution = VPU::getNonOverlappedDistributedAttr(shape, distribution.getMode(), nullptr,
                                                                    distribution.getNumClusters(), nullptr,
                                                                    distribution.getUniformDistributedSegments(), ctx);

        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, orderMap, memSpace, newDistribution);
    }

    auto newDistribution = VPU::DistributionInfoAttr::get(
            ctx, distribution.getMode(), distribution.getNumTiles(), nullptr, nullptr, nullptr,
            distribution.getNumClusters(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, orderMap, memSpace, newDistribution);
};

// Check inputs of ConcatView, below pattern is expected.
//     DistributedCopy   ...     CopyOp
//      (CMX -> DDR)          (DDR -> DDR)
//             \                /
//              ConcatView (DDR)
// Pattern matching requires below criteria:
// 1.If ConcatView has DistributedCopy inputs, they should be DUPLICATED.
// 2.ConcatView should have at least one DDR2DDR copy input.
// Return ConcatInputs struct if pattern can match, otherwise return mlir::failure().
mlir::FailureOr<ConcatInputs> OptimizeDDR2DDRCopyInputsOfConcatView::getValidConcatInputs(
        VPUIP::ConcatViewOp concatViewOp) const {
    const auto isDDR2DDRCopy = [](mlir::Value input) {
        auto op = input.getDefiningOp<VPUIP::CopyOp>();
        if (op == nullptr) {
            return false;
        }

        // check if output buff is a SubView for safety
        auto subViewOp = op.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        if (subViewOp == nullptr) {
            return false;
        }

        return VPUIP::isCopyToDDR(op) && VPUIP::isCopyFromDDR(op);
    };

    const auto isDuplicatedDistributedCopy = [](mlir::Value input) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        if (!copyOp || !vpux::VPUIP::hasDistributedOperand(copyOp)) {
            return false;
        }

        if (copyOp == nullptr || !VPUIP::isCopyToDDR(copyOp)) {
            return false;
        }

        // check if output buff is a SubView for safety
        auto subViewOp = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        if (subViewOp == nullptr) {
            return false;
        }

        auto tilingCopyInput = copyOp.getOperand(0);
        const auto inDistributedType = VPUIP::extractDataType(tilingCopyInput).dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_UNLESS(inDistributedType != nullptr, "Cannot get distributedType");

        auto distribution = inDistributedType.getDistribution();
        return VPU::isDuplicated(distribution);
    };

    struct ConcatInputs validInputs;

    for (const auto& input : concatViewOp.getInputs()) {
        if (isDDR2DDRCopy(input)) {
            validInputs.inputCopies.push_back(input);
        } else if (isDuplicatedDistributedCopy(input)) {
            validInputs.inputDistributedCopies.push_back(input);
        } else {
            _log.nest().trace("[{0}] Invalid input: not a valid Copy", getDebugName());
            return mlir::failure();
        }
    }

    if (validInputs.inputCopies.empty()) {
        _log.nest().trace("[{0}] Invalid input: not DDR2DDR Copy input", getDebugName());
        return mlir::failure();
    }

    return validInputs;
}

// Traverse output chain, store pure viewlike ops into viewLikeOps vector and return DistributedCopy.
// Return mlir::failure() if pattern does not match
mlir::FailureOr<mlir::Operation*> OptimizeDDR2DDRCopyInputsOfConcatView::searchCopyOpThroughViewLikeOps(
        VPUIP::ConcatViewOp concatViewOp, SmallVector<mlir::Operation*>& viewLikeOps) const {
    auto isSupportedViewlikeOp = [](mlir::Operation* user) {
        return mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::ShapeCastOp>(user);
    };

    mlir::Operation* operation = concatViewOp;
    while (operation && !operation->getUsers().empty()) {
        auto user = *(operation->getUsers().begin());

        if (mlir::isa_and_nonnull<VPUIP::CopyOp>(user) && vpux::VPUIP::hasDistributedOperand(user)) {
            return user;
        } else if (isSupportedViewlikeOp(user)) {
            if (!user->hasOneUse()) {
                return mlir::failure();
            }
            viewLikeOps.push_back(user);
            operation = user;
            continue;
        } else {
            break;
        }
    }
    return mlir::failure();
}

// Check ConcatView output chain.
// We expect ConcatView is followed by several viewlike ops(optional), and then a DUPLICATED DistributedCopy is
// connected. Like in below:
//      ConcatView
//          |
//    (Pure View Ops)
//          |
//    DistributedCopy
//          |
// Return ConcatOutputs struct if pattern can match, otherwise return mlir::failure().
mlir::FailureOr<ConcatOutputs> OptimizeDDR2DDRCopyInputsOfConcatView::getValidConcatOutputs(
        VPUIP::ConcatViewOp concatViewOp) const {
    if (!concatViewOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    struct ConcatOutputs validOutput;

    auto copyAfterViewLikeOps = searchCopyOpThroughViewLikeOps(concatViewOp, validOutput.viewLikeOps);
    if (mlir::failed(copyAfterViewLikeOps)) {
        _log.nest().trace("[{0}] Invalid output: no CopyOp after viewlike ops", getDebugName());
        return mlir::failure();
    }

    const auto isDuplicatedChildDistributedCopyOp = [](mlir::Operation* op) {
        auto copyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(op);
        if (copyOp == nullptr || !VPUIP::isCopyFromDDR(copyOp) || VPUIP::isCopyToDDR(copyOp)) {
            return false;
        }

        auto tilingCopyOutput = copyOp->getResult(0);
        const auto outputDistributedType =
                VPUIP::extractDataType(tilingCopyOutput).dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_UNLESS(outputDistributedType != nullptr, "Cannot get distributedType");

        auto distribution = outputDistributedType.getDistribution();
        return VPU::isDuplicated(distribution);
    };

    auto childOp = copyAfterViewLikeOps.value();
    if (!isDuplicatedChildDistributedCopyOp(childOp)) {
        _log.nest().trace("[{0}] Invalid output: no duplicated distributed CopyOp", getDebugName());
        return mlir::failure();
    }

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(childOp);
    auto outputBuffer = copyOp.getOutputBuff();
    auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outputBuffer);
    if (masterBuffer == nullptr) {
        _log.nest().trace("[{0}] Invalid output: buffer isn't master buffer", getDebugName());
        return mlir::failure();
    }

    validOutput.outputDistributedCopy = copyOp;

    return validOutput;
}

void OptimizeDDR2DDRCopyInputsOfConcatView::convertCopyInputAndStore(ArrayRef<mlir::Value> inputCopies,
                                                                     mlir::Value outputBuffer,
                                                                     SmallVector<mlir::Value>& newConcatInputs,
                                                                     mlir::PatternRewriter& rewriter) const {
    for (const auto& copyInput : inputCopies) {
        auto inputCopyOp = copyInput.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = inputCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subViewOp == nullptr, "Can't find SubViewOp");
        auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(subViewOp->getLoc(), "_subview_CMX"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

        auto newDistributedCopyOp =
                rewriter.create<VPUIP::CopyOp>(appendLoc(inputCopyOp->getLoc(), "_cvt_from_copy_input"),
                                               inputCopyOp.getInput(), newSubView.getResult());

        // remove old CopyOp
        rewriter.replaceOp(inputCopyOp, newDistributedCopyOp->getResult(0));

        newConcatInputs.push_back(newDistributedCopyOp.getResult());
    }
}

void OptimizeDDR2DDRCopyInputsOfConcatView::convertDistributedCopyInputAndStore(
        ArrayRef<mlir::Value> inputDistributedCopies, mlir::Value outputBuffer,
        SmallVector<mlir::Value>& newConcatInputs, mlir::PatternRewriter& rewriter) const {
    for (const auto& distributedCopyInput : inputDistributedCopies) {
        auto inputDistributedCopyOp = distributedCopyInput.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = inputDistributedCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subViewOp == nullptr, "Can't find SubViewOp");

        // Input data need copy to DDR then copy back to CMX since DistributedCopy from DistributedBufferType to
        // DistributedBufferType is not supported

        // CMX to DDR
        auto inputCopyType = inputDistributedCopyOp.getInput().getType();

        auto inputType = inputCopyType.isa<VPUIP::DistributedBufferType>()
                                 ? inputCopyType.cast<VPUIP::DistributedBufferType>()
                                           .getCompactType()
                                           .dyn_cast<vpux::NDTypeInterface>()
                                 : inputCopyType.dyn_cast<vpux::NDTypeInterface>();

        auto newDDRType = inputType.changeMemSpace(VPU::MemoryKind::DDR);
        auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(
                appendLoc(inputDistributedCopyOp->getLoc(), "_new_DDR_buffer"), newDDRType.cast<mlir::MemRefType>());

        auto cmxToDDRDistributedCopyOp = rewriter.create<VPUIP::CopyOp>(
                appendLoc(inputDistributedCopyOp->getLoc(), "_CMX_to_DDR_Copy"), inputDistributedCopyOp.getInput(),
                static_cast<mlir::Value>(newAllocDDROp));

        // DDR to CMX
        auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(subViewOp->getLoc(), "_subview_CMX"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        auto ddrToCMXDistributedCopyOp = rewriter.create<VPUIP::CopyOp>(
                appendLoc(inputDistributedCopyOp->getLoc(), "_DDR_to_CMX_Copy"),
                static_cast<mlir::Value>(cmxToDDRDistributedCopyOp.getResult()), newSubView.getResult());

        // remove old distributed CopyOp
        rewriter.replaceOp(inputDistributedCopyOp, ddrToCMXDistributedCopyOp->getResult(0));

        newConcatInputs.push_back(ddrToCMXDistributedCopyOp.getResult());
    }
}

mlir::Value OptimizeDDR2DDRCopyInputsOfConcatView::rewriteViewLikeOps(mlir::Value input,
                                                                      ArrayRef<mlir::Operation*> viewLikeOps,
                                                                      VPUIP::DistributedBufferType origOutputBufferType,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto output = input;
    for (const auto& viewlikeOp : viewLikeOps) {
        if (auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(viewlikeOp)) {
            auto origType = reshapeOp.getOutput().getType().cast<NDTypeInterface>();
            const auto newType = getDuplicatedDistributedType(origType, origOutputBufferType, ctx);
            auto newReshapeOp = rewriter.create<VPUIP::GenericReshapeOp>(reshapeOp->getLoc(), newType, output);
            output = newReshapeOp.getOutput();
        } else if (auto shapeCastOp = mlir::dyn_cast<VPUIP::ShapeCastOp>(viewlikeOp)) {
            auto newShapeCastOp =
                    rewriter.create<VPUIP::ShapeCastOp>(shapeCastOp->getLoc(), output, shapeCastOp.getShape());
            output = newShapeCastOp.getResult();
        } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(viewlikeOp)) {
            auto origType = permuteCastOp.getResult().getType().cast<NDTypeInterface>();
            const auto newType = getDuplicatedDistributedType(origType, origOutputBufferType, ctx);
            auto newPermuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(permuteCastOp->getLoc(), newType, output,
                                                                          permuteCastOp.getDstOrderAttr(),
                                                                          permuteCastOp.getMemPermAttr());
            output = newPermuteCastOp.getResult();
        } else {
            VPUX_THROW("Unsupported ViewLike Op");
        }
    }

    return output;
}

mlir::FailureOr<ConcatOutputsOfSubViewCopyUsers>
OptimizeDDR2DDRCopyInputsOfConcatView::searchSubViewCopyUsersThroughPermuteCast(
        VPUIP::ConcatViewOp concatViewOp) const {
    SmallVector<VPUIP::SubViewOp> subViewOps;
    SmallVector<VPUIP::CopyOp> copyOps;

    mlir::Operation* rootOp;
    auto permuteCastOp = mlir::dyn_cast_or_null<VPUIP::PermuteCastOp>(*concatViewOp->getUsers().begin());
    if (permuteCastOp != nullptr) {
        if (!concatViewOp->hasOneUse()) {
            return mlir::failure();
        }
        rootOp = permuteCastOp.getOperation();
    } else {
        rootOp = concatViewOp.getOperation();
    }

    for (auto* user : rootOp->getUsers()) {
        auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(user);
        if (subViewOp == nullptr || !subViewOp->hasOneUse()) {
            return mlir::failure();
        }

        auto copyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(*user->getUsers().begin());
        if (copyOp == nullptr || !copyOp->hasOneUse() || !vpux::VPUIP::hasDistributedOperand(copyOp)) {
            return mlir::failure();
        }

        subViewOps.push_back(subViewOp);
        copyOps.push_back(copyOp);
    }

    if (subViewOps.empty()) {
        return mlir::failure();
    }

    ConcatOutputsOfSubViewCopyUsers outputs;
    outputs.permuteCastOp = permuteCastOp;
    outputs.outputSubViewOps = std::move(subViewOps);
    outputs.outputDistributedCopyOps = std::move(copyOps);
    return outputs;
}

// Check ConcatView output chain.
// We expect ConcatView is followed by a PermuteCast, or multiple SubView and DUPLICATED DistributedCopy users.
// Like in below:
/*
                    ConcatView
                        |
                  [PermuteCastOp]
            /           |           \
    SubView         SubView         SubView
        |               |               |
DistributedCopy DistributedCopy DistributedCopy
        |               |               |
*/
// Return ConcatOutputsOfSubViewCopyUsers struct if pattern can match, otherwise return mlir::failure().
mlir::FailureOr<ConcatOutputsOfSubViewCopyUsers>
OptimizeDDR2DDRCopyInputsOfConcatView::getValidConcatOutputsOfSubViewCopyUsers(VPUIP::ConcatViewOp concatViewOp) const {
    auto outputs = searchSubViewCopyUsersThroughPermuteCast(concatViewOp);
    if (mlir::failed(outputs)) {
        _log.nest().trace("[{0}] Invalid output: can't find SubView and Copy users after PermuteCast", getDebugName());
        return mlir::failure();
    }

    const auto concatAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(concatViewOp.getOperands()[0]), getShape(concatViewOp.getResult()));
    if (concatAxes.empty() || concatAxes.size() != 1) {
        _log.nest().trace("[{0}] Only support concat on single dimension", getDebugName());
        return mlir::failure();
    }

    auto firstSubViewOp = outputs.value().outputSubViewOps.front();
    const auto firstSubViewAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(firstSubViewOp.getSource()), getShape(firstSubViewOp.getResult()));
    for (auto subViewOp : outputs.value().outputSubViewOps) {
        // SubView axis should be different with ConcatView axis
        // All SubView axes should be the same
        const auto subViewAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(subViewOp.getSource()), getShape(subViewOp.getResult()));
        if (subViewAxes.empty() || subViewAxes.size() != 1) {
            _log.nest().trace("[{0}] Only support SubView on single dimension", getDebugName());
            return mlir::failure();
        }

        if (subViewAxes.front() != firstSubViewAxes.front()) {
            _log.nest().trace("[{0}] All SubView axes should be the same", getDebugName());
            return mlir::failure();
        }

        if (subViewAxes.front() == concatAxes.front()) {
            _log.nest().trace("[{0}] SubView axis should be different with ConcatView axis", getDebugName());
            return mlir::failure();
        }
    }

    const auto isDuplicatedChildDistributedCopyOp = [](mlir::Operation* op) {
        auto copyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(op);
        if (copyOp == nullptr || !VPUIP::isCopyFromDDR(copyOp) || VPUIP::isCopyToDDR(copyOp)) {
            return false;
        }

        auto tilingCopyOutput = copyOp->getResult(0);
        const auto outputDistributedType =
                mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(tilingCopyOutput));
        if (outputDistributedType == nullptr) {
            return false;
        }

        auto distribution = outputDistributedType.getDistribution();
        return VPU::isDuplicated(distribution);
    };

    for (auto copyOp : outputs.value().outputDistributedCopyOps) {
        if (!isDuplicatedChildDistributedCopyOp(copyOp)) {
            _log.nest().trace("[{0}] Invalid output: no duplicated distributed CopyOp", getDebugName());
            return mlir::failure();
        }

        auto outputBuffer = copyOp.getOutputBuff();
        auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outputBuffer);
        if (masterBuffer == nullptr) {
            _log.nest().trace("[{0}] Invalid output: buffer isn't master buffer", getDebugName());
            return mlir::failure();
        }
    }

    auto permuteCastOp = outputs.value().permuteCastOp;
    // Currently only support PermuteCastOp that does not change the logical shape
    if (permuteCastOp != nullptr && getShape(permuteCastOp.getSource()) != getShape(permuteCastOp.getResult())) {
        return mlir::failure();
    }

    return outputs.value();
}

/*
    Eliminate DDR2DDR Copy operations of ConcatView inputs.

    Convert below pattern:
                                Copy(3584x1x1x1)      Copy(3584x15x1x1)
                                /           \               /           \
                            SubView     ConcatView(3584x16x1x1)         SubView
                                                    |
                                    PermuteCast(3584x16x1x1@NHWC)
                            /                       |                       \
    SubView(512x16x1x1@NHWC)            SubView(512x16x1x1@NHWC)    ...    SubView(512x16x1x1@NHWC)
                |                                   |                                   |
DistributedCopy(512x16x1x1@NHWC)    DistributedCopy(512x16x1x1@NHWC)    DistributedCopy(512x16x1x1@NHWC)
            |                                       |                                   |
        NCE Task0                               NCE Task1                            NCE TaskN

    to:

        SubView(512x1x1x1)             SubView(512x16x1x1)
                |                                   |
    DistributedCopy(512x1x1x1)      DistributedCopy(512x16x1x1)
            /           \                   /           \
        SubView         ConcatView(512x16x1x1)          SubView
                                |
                    PermuteCast(512x16x1x1@NHWC)
                                |
                            NCE Task0

                            ...

        SubView(512x1x1x1)             SubView(512x16x1x1)
                |                                   |
    DistributedCopy(512x1x1x1)      DistributedCopy(512x16x1x1)
            /           \                   /           \
        SubView         ConcatView(512x16x1x1)          SubView
                                |
                    PermuteCast(512x16x1x1@NHWC)
                                |
                            NCE TaskN

*/
mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::processConcatOutputsOfSubViewCopyUsers(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs,
        const ConcatOutputsOfSubViewCopyUsers& concatOutputs, mlir::PatternRewriter& rewriter) const {
    if (concatInputs.inputCopies.empty() || !concatInputs.inputDistributedCopies.empty()) {
        _log.nest().trace("[{0}] Only support DDR2DDR Copy inputs", getDebugName());
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();
    auto permuteCastOp = concatOutputs.permuteCastOp;
    const auto concatAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(concatViewOp.getOperands()[0]), getShape(concatViewOp.getResult()));
    const auto concatAxis = concatAxes.front();
    const auto origConcatShape = getShape(concatViewOp.getOutput());
    for (auto subViewOp : concatOutputs.outputSubViewOps) {
        auto childDistributedCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subViewOp->getUsers().begin());
        VPUX_THROW_WHEN(childDistributedCopyOp == nullptr, "Can't find CopyOp user");

        auto outputBuffer = childDistributedCopyOp.getOutputBuff();
        const auto outputBufferType = outputBuffer.getType().dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_WHEN(outputBufferType == nullptr, "Can't get DistributedBufferType");

        const auto subViewAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(subViewOp.getSource()), getShape(subViewOp.getResult()));
        const auto subViewAxis = subViewAxes.front();

        Shape newConcatShape = origConcatShape.raw();
        newConcatShape[subViewAxis] = getShape(subViewOp.getResult())[subViewAxis];
        auto newConcatType =
                mlir::cast<NDTypeInterface>(concatViewOp.getOutput().getType()).changeShape(newConcatShape);
        auto newConcatBufferType = getDuplicatedDistributedType(newConcatType, outputBufferType, ctx);

        // update buffer type so that new ConcatView can re-use this buffer on CMX
        outputBuffer.setType(newConcatBufferType);

        SmallVector<mlir::Value> newConcatInputs;
        rewriter.setInsertionPointAfter(outputBuffer.getDefiningOp());

        int64_t currentOutOffset = 0;
        for (auto input : concatInputs.inputCopies) {
            auto inputCopyOp = input.getDefiningOp<VPUIP::CopyOp>();

            auto srcSubViewOffsets = parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets());
            auto srcSubViewSizes = parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes());
            srcSubViewSizes[concatAxis.ind()] = getShape(inputCopyOp.getInput())[concatAxis];
            auto newSrcSubView = rewriter.create<VPUIP::SubViewOp>(
                    appendLoc(subViewOp->getLoc(), "_src_subview"), inputCopyOp.getInput(),
                    getIntArrayAttr(ctx, srcSubViewOffsets), getIntArrayAttr(ctx, srcSubViewSizes));

            auto dstSubViewOffsets = SmallVector<int64_t>(srcSubViewSizes.size(), 0);
            auto dstSubViewSizes = parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes());
            dstSubViewOffsets[concatAxis.ind()] = currentOutOffset;
            dstSubViewSizes[concatAxis.ind()] = getShape(inputCopyOp.getInput())[concatAxis];
            currentOutOffset += dstSubViewSizes[concatAxis.ind()];
            auto newDstSubView = rewriter.create<VPUIP::SubViewOp>(
                    appendLoc(subViewOp->getLoc(), "_dst_subview"), outputBuffer,
                    getIntArrayAttr(ctx, dstSubViewOffsets), getIntArrayAttr(ctx, dstSubViewSizes),
                    subViewOp.getStaticStridesAttr());

            auto newDistributedCopyOp =
                    rewriter.create<VPUIP::CopyOp>(appendLoc(inputCopyOp->getLoc(), "_copy_to_cmx"),
                                                   newSrcSubView.getResult(), newDstSubView.getResult());

            newConcatInputs.push_back(newDistributedCopyOp.getResult());
        }

        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(concatViewOp->getLoc(), newConcatInputs, outputBuffer);

        auto lastValue = newConcatOp.getOutput();
        if (permuteCastOp != nullptr) {
            auto origPermuteCastShape = getShape(permuteCastOp.getResult());
            Shape newPermuteCastShape = origPermuteCastShape.raw();
            newPermuteCastShape[subViewAxis] = getShape(subViewOp.getResult())[subViewAxis];
            auto newPermuteCastType =
                    permuteCastOp.getResult().getType().cast<NDTypeInterface>().changeShape(newPermuteCastShape);
            const auto newType = getDuplicatedDistributedType(newPermuteCastType, outputBufferType, ctx);
            auto newPermuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(
                    permuteCastOp->getLoc(), newType, newConcatOp.getOutput(), permuteCastOp.getDstOrderAttr(),
                    permuteCastOp.getMemPermAttr());
            lastValue = newPermuteCastOp.getResult();
        }

        auto distributedCastOp = rewriter.createOrFold<VPUIP::DistributedCastOp>(childDistributedCopyOp->getLoc(),
                                                                                 outputBufferType, lastValue);

        rewriter.replaceOp(childDistributedCopyOp, distributedCastOp);
    }

    // Remove old operations
    for (auto subViewOp : concatOutputs.outputSubViewOps) {
        rewriter.eraseOp(subViewOp);
    }
    if (permuteCastOp != nullptr) {
        rewriter.eraseOp(permuteCastOp);
    }
    rewriter.eraseOp(concatViewOp);
    for (auto input : concatInputs.inputCopies) {
        auto inputCopyOp = input.getDefiningOp<VPUIP::CopyOp>();
        rewriter.eraseOp(inputCopyOp);
    }

    return mlir::success();
}

mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto concatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.getOutputBuff());
    if (concatMemAlloc == nullptr) {
        _log.nest().trace("[{0}] Cannot rewrite because current concat '{1}' output isn't master buffer",
                          getDebugName(), concatViewOp->getLoc());
        return mlir::failure();
    }

    // Check inputs of ConcatView
    auto checkInputs = getValidConcatInputs(concatViewOp);
    if (mlir::failed(checkInputs)) {
        _log.nest().trace("[{0}] Invalid inputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    struct ConcatInputs concatInputs = checkInputs.value();

    // Check output of ConcatView
    auto checkOutputs = getValidConcatOutputs(concatViewOp);
    if (mlir::failed(checkOutputs)) {
        auto checkOutputsOfSubViewCopyUsers = getValidConcatOutputsOfSubViewCopyUsers(concatViewOp);
        if (mlir::succeeded(checkOutputsOfSubViewCopyUsers)) {
            _log.nest().trace("[{0}] Got '{1}' at '{2}' with SubView and Copy Users", getDebugName(),
                              concatViewOp->getName(), concatViewOp->getLoc());
            struct ConcatOutputsOfSubViewCopyUsers concatOutputs = checkOutputsOfSubViewCopyUsers.value();
            return processConcatOutputsOfSubViewCopyUsers(concatViewOp, concatInputs, concatOutputs, rewriter);
        }
        _log.nest().trace("[{0}] Invalid outputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    struct ConcatOutputs concatOutputs = checkOutputs.value();
    auto childDistributedCopyOp = concatOutputs.outputDistributedCopy;
    auto outputBuffer = childDistributedCopyOp.getOutputBuff();
    const auto outputBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputBuffer.getType());
    if (outputBufferType == nullptr) {
        _log.nest().trace("[{0}] ConcatView '{1}' at '{2}' user distributed copy buffer does not have distributedType",
                          getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());

    // Create new subgraph to move ConcatView and viewlike ops to CMX
    auto ctx = rewriter.getContext();

    auto origConcatNDType = concatViewOp.getOutput().getType().cast<NDTypeInterface>();
    auto newConcatBufferType = getDuplicatedDistributedType(origConcatNDType, outputBufferType, ctx);
    // update buffer type so that new ConcatView can re-use this buffer on CMX
    outputBuffer.setType(newConcatBufferType);

    SmallVector<mlir::Value> newConcatInputs;
    rewriter.setInsertionPointAfter(outputBuffer.getDefiningOp());

    convertCopyInputAndStore(concatInputs.inputCopies, outputBuffer, newConcatInputs, rewriter);
    convertDistributedCopyInputAndStore(concatInputs.inputDistributedCopies, outputBuffer, newConcatInputs, rewriter);
    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(concatViewOp->getLoc(), newConcatInputs, outputBuffer);

    auto subGraphOutput =
            rewriteViewLikeOps(newConcatOp.getOutput(), concatOutputs.viewLikeOps, outputBufferType, rewriter);

    // cast to original outputBufferType because alignment in distribution might be different
    auto distributedCastOp = rewriter.createOrFold<VPUIP::DistributedCastOp>(childDistributedCopyOp->getLoc(),
                                                                             outputBufferType, subGraphOutput);

    rewriter.replaceOp(childDistributedCopyOp, distributedCastOp);

    return mlir::success();
}

//
// OptimizeConcatSubviewPattern
//

class OptimizeConcatSubviewPattern : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    OptimizeConcatSubviewPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/*
Optimize subgraph like below, note that for copy0 and copy2, copy1 and copy3, they should have same size and offsets
  Input0      Input1
    |           |
   Copy0      Copy1
     \        /                 |         |
     ConcatView       =>      Input0    Input1
     /        \                 |         |
  Subview0   Subview1
    |           |
   Copy2      Copy3

*/
mlir::LogicalResult OptimizeConcatSubviewPattern::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto nestedLogger = _log.nest();

    auto concatOutput = concatOp.getOutput();
    if (getShape(concatOutput).size() != 4) {
        nestedLogger.trace("Cannot optimize because of shape rank not being 4");
        return mlir::failure();
    }

    SmallVector<VPUIP::SubViewOp> inputSubViews;
    SmallVector<VPUIP::SubViewOp> outputSubViews;
    SmallVector<VPUIP::CopyOp> inputTilingCopies;
    SmallVector<VPUIP::CopyOp> outputTilingCopies;

    // check input
    for (auto input : concatOp.getInputs()) {
        auto tilingCopy = input.getDefiningOp<VPUIP::CopyOp>();
        if (tilingCopy == nullptr || !tilingCopy->hasOneUse() || !vpux::VPUIP::hasDistributedOperand(tilingCopy)) {
            return mlir::failure();
        }

        auto copyOpOutput = tilingCopy.getOutputs()[0];
        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();
        if (subview == nullptr) {
            return mlir::failure();
        }

        if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(subview.getSource()) == nullptr) {
            return mlir::failure();
        }

        inputSubViews.push_back(subview);
        inputTilingCopies.push_back(tilingCopy);
    }

    // check output
    for (auto user : concatOp->getUsers()) {
        auto subview = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(user);
        if (subview == nullptr || !subview->hasOneUse()) {
            return mlir::failure();
        }

        auto tilingCopy = mlir::dyn_cast_or_null<VPUIP::CopyOp>(*(subview->getUsers().begin()));
        if (tilingCopy == nullptr || !vpux::VPUIP::hasDistributedOperand(tilingCopy)) {
            return mlir::failure();
        }

        outputSubViews.push_back(subview);
        outputTilingCopies.push_back(tilingCopy);
    }

    if (inputSubViews.empty() || (outputSubViews.empty())) {
        return mlir::failure();
    }

    auto isSameAttrSubview = [](VPUIP::SubViewOp inSubview, VPUIP::SubViewOp outSubview) {
        return inSubview.getStaticOffsetsAttr() == outSubview.getStaticOffsetsAttr() &&
               inSubview.getStaticSizesAttr() == outSubview.getStaticSizesAttr() &&
               inSubview.getStaticStridesAttr() == outSubview.getStaticStridesAttr();
    };
    auto isSameDistributedTypeCopy = [](VPUIP::CopyOp inCopy, VPUIP::CopyOp outCopy) {
        auto inputValue = inCopy.getInputs()[0];
        auto inputDistributedType = inputValue.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (inputDistributedType == nullptr) {
            return false;
        }

        auto outputValue = outCopy->getResult(0);
        auto outputDistributedType = outputValue.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (outputDistributedType == nullptr) {
            return false;
        }
        return mlir::succeeded(VPU::isDistributedCastCompatible(inputDistributedType, outputDistributedType));
    };

    mlir::DenseMap<int64_t, int64_t> outIndexToInIndex;
    auto inRange = irange(inputSubViews.size());
    for (auto outIdx : irange(outputSubViews.size())) {
        auto iter = llvm::find_if(inRange, [&](auto inIdx) {
            return isSameAttrSubview(inputSubViews[inIdx], outputSubViews[outIdx]) &&
                   isSameDistributedTypeCopy(inputTilingCopies[inIdx], outputTilingCopies[outIdx]);
        });
        if (iter == inRange.end()) {
            return mlir::failure();
        }
        auto inIdx = std::distance(inRange.begin(), iter);
        outIndexToInIndex[outIdx] = inIdx;
    }

    nestedLogger.trace("optimize concat->subview pattern at {0}", concatOp->getLoc());

    for (const auto& item : outIndexToInIndex) {
        const auto& outIdx = item.first;
        const auto& inIdx = item.second;
        auto& outputCopy = outputTilingCopies[outIdx];
        auto& inputCopy = inputTilingCopies[inIdx];
        auto newOutput = rewriter.createOrFold<VPUIP::DistributedCastOp>(
                outputCopy->getLoc(), outputCopy.getResult().getType(), inputCopy.getInput());

        rewriter.replaceAllUsesWith(outputCopy.getResult(), newOutput);
        rewriter.eraseOp(outputCopy);
        rewriter.eraseOp(outputSubViews[outIdx]);
    }

    rewriter.eraseOp(concatOp);

    for (auto ind : irange(inputTilingCopies.size())) {
        rewriter.eraseOp(inputTilingCopies[ind]);
        rewriter.eraseOp(inputSubViews[ind]);
    }

    return mlir::success();
}

template <class OpType>
OpType getSingleUserOfType(mlir::Operation* op) {
    return op->hasOneUse() ? mlir::dyn_cast<OpType>(*op->getUsers().begin()) : nullptr;
}

/*
    LeftBranch: [View]BlockArg(DDR) ------------\                                              /  SubView0 -> TiledCopy
                                                 \                                            /   SubView1 -> TiledCopy
                                                 | -> Concat -> GenericReshape -> PermCast -> |   SubView2 -> TiledCopy
                                                 /                                            \   SubView3 -> TiledCopy
    RightBranch:  DistrBuf(CMX) -> TiledCopy  --/                                              \  SubViewN -> TiledCopy

    To

    LeftBranch  -> GenericReshape -> PermCast \
    RightBranch -> GenericReshape -> PermCast |
                                              +-> Subview0Left  \
                                              |                 |-> Concat -> Distributed Op
                                              +-> Subview0Right /
                                              ...
                                              +-> SubviewNLeft  \
                                              |                 |-> Concat -> Distributed Op
                                              +-> SubviewNRight /

    In case of large DDR input, what is common for KV cache models, original pattern creates large DDR->DDR,
    which scheduled in the beginning of the inference and blocks prefetch/execution. New pattern is more friendly
    and can be distributed across schedule.
    This pattern has specific requirements for GenericReshape, see checkConcatReshapeCompatibility

    Concat(Left[1, 32, 128, 1023], Right[1, 32, 128, 1]) -> Reshape[32 * 128, 1024] -> PermCast -> Views
    to
    Reshape [32 * 128, 1]    -> PermCast -> View -\
                                                  |-> Concat
    Reshape [32 * 128, 1023] -> PermCast -> View -/
*/
class SplitUnbalancedDDRConcatBase : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
protected:
    // Describes parameters of original concat.
    struct PatternParamsInfo {
        int64_t leftConcatInputSize;  // Dim size of left buffer on concat axis.
        int64_t leftInputSize;   // If left buffer was viewed, DimSize of input, otherwise equals to leftConcatInputSize
        int64_t leftViewOffset;  // Offset from begging if was viewed
        int64_t rightInputSize;
        Dim origConcatDim;
        Dim newConcatDim;  // Concat Dim after GReshape
        VPUIP::PermuteCastOp castOp;
    };

public:
    SplitUnbalancedDDRConcatBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

private:
    // Suffix of child rewriter
    virtual StringRef getRewriterSuffix() const = 0;
    // Check compatibility of concat in CMX. Since consumers are segmented, we must ensure that we can do split.
    // If view/concat axis is the same with tiling axis, we can't do concatenation in same buffer.
    // To avoid it we must do concat in temporary buffer and then distribute
    virtual bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const = 0;

    // Input of the right side and copy operation
    virtual std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const = 0;

    virtual mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput,
                                           VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc) const = 0;

    virtual mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp origView,
                                              VPUIP::CopyOp distributedCopy, mlir::Location bufferLoc) const = 0;

    virtual void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp,
                                VPUIP::SubViewOp subViewOp, VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch,
                                mlir::Value newRightBranch, const PatternParamsInfo& params, size_t index) const = 0;

protected:
    // Propage reshape and cast through concat to split large DDR->DDR DMAs on sequence of small DDR->CMX, which can be
    // interleaved with NCE tasks
    mlir::Value propagateReshapeCast(mlir::PatternRewriter& rewriter, mlir::Value branchInput,
                                     VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc,
                                     StringRef locSuffix) const {
        auto inputType = branchInput.getType().cast<vpux::NDTypeInterface>();
        auto origShape = inputType.getShape();
        Shape newShape = origShape.toValues();
        newShape[Dim(0)] = origShape[Dim(1)] * origShape[Dim(2)];
        newShape[Dim(1)] = origShape[Dim(3)];
        newShape[Dim(2)] = 1;
        newShape[Dim(3)] = 1;

        vpux::NDTypeInterface afterReshapeType;
        if (auto distributedBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType)) {
            auto distribution = distributedBufferType.getDistribution();

            // Get new num_tiles attribute
            // shape:    [1, A, B, C] -> [A*B, C, 1, 1]
            // numTiles: [1, a, b, c] -> [a*b, c, 1, 1]
            auto origNumTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
            VPUX_THROW_UNLESS(origNumTiles.size() == 4, "Tile size should be the same as shape size, which is 4");
            SmallVector<int64_t> newNumTiles(origNumTiles.size(), 1);
            newNumTiles[0] = origNumTiles[1] * origNumTiles[2];
            newNumTiles[1] = origNumTiles[3];

            auto ctx = rewriter.getContext();

            auto mode = distribution.getMode();
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    newShape, mode, getIntArrayAttr(ctx, newNumTiles), distribution.getNumClusters(),
                    distribution.getAlignment(), distribution.getUniformDistributedSegments(), ctx);

            afterReshapeType = VPUIP::DistributedBufferType::get(
                    ctx, newShape.raw(), distributedBufferType.getElementType(), distributedBufferType.getLayout(),
                    distributedBufferType.getMemSpace(), newDistribution);
        } else {
            afterReshapeType = inputType.changeShape(newShape);
        }

        auto newReshapeOp = rewriter.createOrFold<VPUIP::GenericReshapeOp>(appendLoc(loc, "greshape_{0}", locSuffix),
                                                                           afterReshapeType, branchInput);

        auto permCastDimsOrder = permuteCastOp->getResultTypes()[0].cast<vpux::NDTypeInterface>().getDimsOrder();
        auto afterPermCastType = afterReshapeType.changeDimsOrder(permCastDimsOrder);
        return rewriter.create<VPUIP::PermuteCastOp>(appendLoc(loc, "permcast_{0}", locSuffix), afterPermCastType,
                                                     newReshapeOp, permuteCastOp.getDstOrderAttr(),
                                                     permuteCastOp.getMemPermAttr());
    }

    mlir::Value createNewCopyBranch(mlir::PatternRewriter& rewriter, mlir::Value src, mlir::Value dst,
                                    ShapeRef copyShape, ShapeRef srcOffset, ShapeRef dstOffset, mlir::Location baseLoc,
                                    StringRef locSuffix, size_t opId, bool needExtraCopy = true) const {
        if (!needExtraCopy) {
            return rewriter.createOrFold<VPUIP::SubViewOp>(appendLoc(baseLoc, "{0}_src_view_{1}", locSuffix, opId), src,
                                                           srcOffset, copyShape);
        }

        mlir::Value srcView = rewriter.createOrFold<VPUIP::SubViewOp>(
                appendLoc(baseLoc, "{0}_src_view_{1}", locSuffix, opId), src, srcOffset, copyShape);
        mlir::Value dstView = rewriter.createOrFold<VPUIP::SubViewOp>(
                appendLoc(baseLoc, "{0}_dst_view_{1}", locSuffix, opId), dst, dstOffset, copyShape);

        auto copyLoc = appendLoc(baseLoc, "{0}_copy_{1}", locSuffix, opId);
        return rewriter.create<VPUIP::CopyOp>(copyLoc, srcView, dstView);
    }

private:
    bool checkConcatReshapeCompatibility(VPUIP::ConcatViewOp concatOp, VPUIP::GenericReshapeOp genReshapeOp,
                                         vpux::Logger log) const {
        auto genReshapeType = vpux::getBufferType(genReshapeOp.getOutput());
        auto concatType = vpux::getBufferType(concatOp.getOutput());

        if (genReshapeType.getRank() != 4 || concatType.getRank() != 4) {
            log.trace("Only 4D tensors are supported");
            return false;
        }

        auto reshapeShape = genReshapeType.getShape();
        auto concatShape = concatType.getShape();
        if (concatShape[Dim(0)] != 1) {
            log.trace("Only Batch size 1 is supported");
            return false;
        }

        // [1, A, B, C] -> [A*B, C, 1, 1]
        bool compatible = reshapeShape[Dim(0)] == concatShape[Dim(1)] * concatShape[Dim(2)] &&
                          reshapeShape[Dim(1)] == concatShape[Dim(3)] &&
                          genReshapeType.getNumElements() == concatType.getNumElements();
        if (!compatible) {
            log.trace("Concat->Reshape shapes are not compatible: {0} vs {1}", concatShape, reshapeShape);
            return false;
        }
        return true;
    }

    // Input of the left side, must be block argument
    std::pair<mlir::Value, VPUIP::SubViewOp> getLeftBranchInput(VPUIP::ConcatViewOp concatOp) const {
        const size_t LEFT_INPUT_ID = 0;  // Left must be always first to preserve concat order
        auto inputCopy = concatOp.getInputs()[LEFT_INPUT_ID];
        if (auto copyOp = inputCopy.getDefiningOp<VPUIP::CopyOp>()) {
            auto input = copyOp.getInput();
            if (mlir::isa<mlir::BlockArgument>(input)) {
                return {input, nullptr};
            }
            if (auto viewOp = input.getDefiningOp<VPUIP::SubViewOp>()) {
                auto viewInput = viewOp.getSource();
                auto validInputView = viewOp->hasOneUse() && mlir::isa<mlir::BlockArgument>(viewInput);
                if (validInputView) {
                    return {viewInput, viewOp};
                }
            }
        }
        return {nullptr, nullptr};
    }

    mlir::Value prepareLeftBranch(mlir::PatternRewriter& rewriter, mlir::Value leftBranchInput,
                                  VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc) const {
        return propagateReshapeCast(rewriter, leftBranchInput, permuteCastOp, loc, "left");
    }

    std::optional<int64_t> getTilingAxis(mlir::Type type) const {
        auto outDistributedType = type.dyn_cast<VPUIP::DistributedBufferType>();
        if (outDistributedType == nullptr) {
            return std::nullopt;
        }

        const auto distAttr = outDistributedType.getDistribution();
        const auto distMode = distAttr.getMode().getValue();
        if (distMode != VPU::DistributionMode::SEGMENTED) {
            return std::nullopt;
        }
        const auto numTiles = distAttr.getNumTiles();
        const auto tilingScheme = parseIntArrayAttr<int64_t>(numTiles);
        return VPU::getDistributedTilingAxis(tilingScheme);
    };

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("SplitUnbalancedDDRConcat{0}: Got Concat at '{1}'", getRewriterSuffix(), concatOp.getLoc());
        auto nestedLog = _log.nest();

        if (concatOp.getInputs().size() != 2) {
            nestedLog.trace("Only 2 inputs supported");
            return mlir::failure();
        }

        // Output type is same as type of all inputs
        auto isDdrConcat = vpux::getBufferType(concatOp).getMemoryKind() == VPU::MemoryKind::DDR;
        if (!isDdrConcat) {
            nestedLog.trace("All inputs must be DDR copies");
            return mlir::failure();
        }

        auto genReshapeOp = getSingleUserOfType<VPUIP::GenericReshapeOp>(concatOp);
        if (genReshapeOp == nullptr) {
            nestedLog.trace("ConcatOp must followed by only one GenericReshape");
            return mlir::failure();
        }
        if (!checkConcatReshapeCompatibility(concatOp, genReshapeOp, nestedLog)) {
            return mlir::failure();
        }

        auto permuteCastOp = getSingleUserOfType<VPUIP::PermuteCastOp>(genReshapeOp);
        if (permuteCastOp == nullptr) {
            nestedLog.trace("GenericReshape must followed by only one PermuteCast");
            return mlir::failure();
        }

        const auto concatAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
        if (concatAxes.size() != 1) {
            nestedLog.trace("Cannot extract concat axis");
            return mlir::failure();
        }
        auto origConcatDim = concatAxes.front();
        Dim newConcatDim(0);
        // 0 - invalid, 1-2 -> 0, 3->1
        if (origConcatDim == Dim(0)) {
            nestedLog.trace("Unsupported orig concat dim {0}", origConcatDim);
            return mlir::failure();
        }
        if (origConcatDim == Dim(3)) {
            newConcatDim = Dim(1);
        }
        // GenericReshape collapses 1,2 axis into 0, and 3 to 1, see checkConcatReshapeCompatibility
        nestedLog.trace("Concat axis transformation: {0} -> {1}", origConcatDim, newConcatDim);

        auto [leftBranchInput, leftBranchViewOp] = getLeftBranchInput(concatOp);
        if (leftBranchInput == nullptr) {
            nestedLog.trace("Can't get left branch input");
            return mlir::failure();
        }
        SmallVector<int64_t> leftViewOffsets;
        int64_t leftSizeOnConcatDim = getShape(leftBranchInput)[origConcatDim];
        int64_t leftViewOffsetOnConcatDim = 0;
        if (leftBranchViewOp != nullptr) {
            leftViewOffsets = parseIntArrayAttr<int64_t>(leftBranchViewOp.getStaticOffsets());
            auto viewOffsetOnConcatDim = leftViewOffsets[origConcatDim.ind()];
            if (viewOffsetOnConcatDim != 1) {
                nestedLog.trace("Only off by one offset is supported");
                return mlir::failure();
            }
            leftSizeOnConcatDim = getShape(leftBranchViewOp->getResult(0))[origConcatDim];
            leftViewOffsetOnConcatDim = viewOffsetOnConcatDim;
        }

        auto [rightBranchInput, copiesToRemove] = getRightBranchInput(concatOp);
        if (rightBranchInput == nullptr || copiesToRemove.empty()) {
            nestedLog.trace("Can't get right branch input");
            return mlir::failure();
        }
        VPUX_THROW_WHEN(leftBranchInput == rightBranchInput, "Branches must have different inputs");

        SmallVector<VPUIP::SubViewOp> views;
        SmallVector<VPUIP::CopyOp> distributedCopies;
        for (auto user : permuteCastOp->getUsers()) {
            if (auto viewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
                if (!viewOp->hasOneUse()) {
                    nestedLog.trace("ViewOp at '{0}' must have only one user", viewOp->getLoc());
                    return mlir::failure();
                }
                views.push_back(viewOp);

                auto copyOp = getSingleUserOfType<VPUIP::CopyOp>(viewOp);
                if (copyOp == nullptr || !vpux::VPUIP::hasDistributedOperand(copyOp)) {
                    nestedLog.trace("View at '{0}' user is not a Distributed Copy", viewOp->getLoc());
                    return mlir::failure();
                }
                distributedCopies.push_back(copyOp);
            } else {
                nestedLog.trace("All users must be View operations");
                return mlir::failure();
            }
        }
        if (views.empty()) {
            nestedLog.trace("Cannot find any SubView-> Distributed Copy consumers");
            return mlir::failure();
        }

        if (distributedCopies.size() < 1) {
            nestedLog.trace("Expected at least 2 users after concat");
            return mlir::failure();
        }

        auto maybeTilingAxis = getTilingAxis(distributedCopies.front().getResult().getType());
        if (!maybeTilingAxis.has_value()) {
            nestedLog.trace("Only SEGMENTED distribution is supported for consumers");
            return mlir::failure();
        }

        auto tilingAxis = maybeTilingAxis.value();
        if (tilingAxis != 0) {
            nestedLog.trace("Only tiling on major dim is supported");
            return mlir::failure();
        }

        auto allTilingAxisAreSame = llvm::all_of(distributedCopies, [&](VPUIP::CopyOp tiledCopy) {
            auto currentTilingAxis = getTilingAxis(tiledCopy.getResult().getType());
            return currentTilingAxis.has_value() && currentTilingAxis.value() == tilingAxis;
        });
        if (!allTilingAxisAreSame) {
            nestedLog.trace("Concat users have different distribution axis");
            return mlir::failure();
        }

        nestedLog.trace("Found {0} copies to split. newConcatDim: {1}, segmentationDim: d{2}", views.size(),
                        newConcatDim, tilingAxis);
        if (!isSplitSupported(newConcatDim, tilingAxis)) {
            nestedLog.trace("Not supported combination of tiling/concat");
            return mlir::failure();
        }
        PatternParamsInfo params{leftSizeOnConcatDim,
                                 getShape(leftBranchInput)[origConcatDim],
                                 leftViewOffsetOnConcatDim,
                                 getShape(rightBranchInput)[origConcatDim],
                                 origConcatDim,
                                 newConcatDim,
                                 permuteCastOp};

        const auto isOverlappedOrNone = [](mlir::Value branchInput) {
            auto inputType = branchInput.getType().cast<vpux::NDTypeInterface>();
            auto distributedBufferType = inputType.dyn_cast<VPUIP::DistributedBufferType>();
            if (distributedBufferType == nullptr) {
                return false;
            }

            auto distribution = distributedBufferType.getDistribution();
            auto mode = distribution ? distribution.getMode() : nullptr;
            return mode == nullptr || mode.getValue() == VPU::DistributionMode::OVERLAPPED;
        };

        // We don't support OVERLAPPED mode
        if (isOverlappedOrNone(leftBranchInput) || isOverlappedOrNone(rightBranchInput)) {
            return mlir::failure();
        }

        mlir::Value newLeftBranch = prepareLeftBranch(rewriter, leftBranchInput, permuteCastOp, concatOp->getLoc());
        mlir::Value newRightBranch = prepareRightBranch(rewriter, rightBranchInput, permuteCastOp, concatOp->getLoc());

        for (size_t i = 0; i < distributedCopies.size(); ++i) {
            VPUIP::CopyOp distributedCopy = distributedCopies[i];
            VPUX_THROW_UNLESS(vpux::VPUIP::hasDistributedOperand(distributedCopy), "Expected a distributed Copy op");
            VPUIP::SubViewOp subViewOp = views[i];

            rewriter.setInsertionPoint(distributedCopy);
            rewriteSubview(rewriter, concatOp, subViewOp, distributedCopy, newLeftBranch, newRightBranch, params, i);
        }

        const size_t LEFT_CONCAT_INPUT_ID = 0;
        mlir::Value leftConcatInput = concatOp.getInputs()[LEFT_CONCAT_INPUT_ID];
        copiesToRemove.push_back(leftConcatInput.getDefiningOp());

        rewriter.eraseOp(permuteCastOp);
        rewriter.eraseOp(genReshapeOp);
        rewriter.eraseOp(concatOp);
        for (auto copy : copiesToRemove) {
            if (copy == nullptr) {
                continue;
            }
            if (copy->use_empty()) {
                rewriter.eraseOp(copy);
            }
        }

        nestedLog.trace("Done");
        _log.unnest();
        return mlir::success();
    }

private:
    Logger _log;
};

/*
    Concat(Left[1, 32, 1023, 128], Right[1, 32, 1, 128]) -> Reshape[32 * 1024, 128] -> PermCast -> Views
    to
    Reshape [32 * 1023, 128] -> PermCast -> View -\
                                                  |-> Concat -> TiledCopy to CMX
    FlatView -> Reshape [1, 128] -> PermCast     -/
*/
class SplitUnbalancedDDRConcatOnOtherAxis : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    StringRef getRewriterSuffix() const override {
        return "OnOtherAxis";
    }

    bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const override {
        return newConcatDim.ind() != tilingDim;
    }

    std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const override {
        const size_t RIGHT_INPUT_ID = 1;  // Right must be always second to preserve concat order
        auto inputCopy = concatOp.getInputs()[RIGHT_INPUT_ID];
        if (auto copyOp = inputCopy.getDefiningOp<VPUIP::CopyOp>()) {
            if (!mlir::isa<mlir::BlockArgument>(copyOp.getInput())) {
                return {copyOp.getInput(), {copyOp}};
            }
        }
        return {nullptr, {}};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput,
                                   VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc) const override {
        return propagateReshapeCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
    }

    // Buffer remains the same
    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp, VPUIP::CopyOp distributedCopy,
                                      mlir::Location bufferLoc) const override {
        auto dstBufferType = distributedCopy.getOutputBuff().getType();
        return rewriter.create<VPURT::AllocDistributed>(bufferLoc, dstBufferType, nullptr, nullptr);
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer = createNewConcatBuffer(rewriter, subViewOp, distributedCopy,
                                               takeOpLoc(origConcatOp, llvm::StringLiteral("buf_{0}"), index));

        // Concat on other axis, so original offset is used
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto createViewBranch = [&](mlir::Value src, int64_t origDimSize, int64_t dstOffsetVal, int64_t srcOffsetVal,
                                    StringRef locSuffix) -> mlir::Value {
            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[params.newConcatDim] = origDimSize;

            Shape newSrcOffset(srcOffset);
            newSrcOffset[params.newConcatDim] = srcOffsetVal;

            Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
            dstOffset[params.newConcatDim] = dstOffsetVal;

            // Rewrite the pattern as Input -> Subview -> DistributedCopy -> ...
            // If the Input is distributed, we don't need extra copy:
            // DistributedInput -> Subview -> ...
            bool needExtraCopy = !mlir::isa<VPUIP::DistributedBufferType>(src.getType());

            return createNewCopyBranch(rewriter, src, dstBuffer, copyShape, newSrcOffset, dstOffset,
                                       origConcatOp->getLoc(), locSuffix, index, needExtraCopy);
        };

        auto newLeftViewBranch = createViewBranch(newLeftBranch, params.leftConcatInputSize, /*dstOffsetVal=*/0,
                                                  params.leftViewOffset, "left");
        auto newRightViewBranch = createViewBranch(newRightBranch, params.rightInputSize, params.leftConcatInputSize,
                                                   /*srcOffsetVal=*/0, "right");

        SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(
                takeOpLoc(origConcatOp, llvm::StringLiteral("concat_{0}"), index), concatInputs, dstBuffer);
        rewriter.replaceAllUsesWith(distributedCopy->getResult(0), newConcatOp);
        rewriter.eraseOp(distributedCopy);
        rewriter.eraseOp(subViewOp);
    }
};

class SplitUnbalancedDDRConcatOnSameAxis : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    StringRef getRewriterSuffix() const override {
        return "OnSameAxis";
    }

    bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const override {
        return newConcatDim.ind() == tilingDim;
    }

    std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const override {
        const size_t RIGHT_INPUT_ID = 1;  // Right must be always second to preserve concat order
        mlir::Value patternInput = concatOp.getInputs()[RIGHT_INPUT_ID];

        // There is no sense to traverse more than 3 times, because eventually we must end with strided DDR->DDR started
        // from Distributed CMX. ViewLike ops aren't allowed, because they change layout/shape and we must consider them
        // in rewriter Longest possible chain is NCECompute -> f32->f16 Copy, CMX->DDR, DDR->Strided DDR
        int depth = 3;
        SmallVector<mlir::Operation*> copies;
        while (patternInput != nullptr && !patternInput.getType().isa<VPUIP::DistributedBufferType>() && depth > 0) {
            if (mlir::isa<mlir::BlockArgument>(patternInput) || patternInput.getDefiningOp() == nullptr) {
                patternInput = nullptr;
                break;
            }
            mlir::Value nextInput = nullptr;
            auto producerOp = patternInput.getDefiningOp();
            if (mlir::isa<VPUIP::CopyOp, VPUIP::ConvertDMAOp>(producerOp)) {
                nextInput = producerOp->getOperand(0);
            }
            patternInput = nextInput;
            copies.push_back(producerOp);
            --depth;
        }
        if (patternInput != nullptr && patternInput.getType().isa<VPUIP::DistributedBufferType>()) {
            return {patternInput, copies};
        }
        return {nullptr, {}};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter&, mlir::Value rightBranchInput, VPUIP::PermuteCastOp,
                                   mlir::Location) const override {
        return rightBranchInput;
    }

    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp, VPUIP::CopyOp distributedCopy,
                                      mlir::Location bufferLoc) const override {
        auto dstBufferType = distributedCopy.getOutputBuff().getType();
        return rewriter.create<VPURT::AllocDistributed>(bufferLoc, dstBufferType, nullptr, nullptr);
    }

    mlir::Value rewriterLeftSubViewBranch(mlir::PatternRewriter& rewriter, const PatternParamsInfo& params,
                                          size_t viewMultiplier, VPUIP::SubViewOp subViewOp, mlir::Value newLeftBranch,
                                          mlir::Value dstBuffer, mlir::Location baseLoc, size_t index) const {
        auto newConcatDim = params.newConcatDim;
        auto copyShape = getShape(subViewOp->getResult(0)).toValues();
        copyShape[newConcatDim] = params.leftConcatInputSize;

        Shape srcOffset(SmallVector<int64_t>(copyShape.size(), 0));
        srcOffset[newConcatDim] = params.leftInputSize * viewMultiplier + params.leftViewOffset;

        Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
        return createNewCopyBranch(rewriter, newLeftBranch, dstBuffer, copyShape, srcOffset, dstOffset, baseLoc, "left",
                                   index);
    }

    mlir::Value rewriterRightSubViewBranch(mlir::PatternRewriter& rewriter, const PatternParamsInfo& params,
                                           size_t viewMultiplier, VPUIP::SubViewOp subViewOp,
                                           mlir::Value newRightBranch, mlir::Value dstBuffer, mlir::Location baseLoc,
                                           size_t index) const {
        auto previewShape = getShape(newRightBranch).toValues();
        previewShape[Dim(1)] = params.rightInputSize;
        auto singleAxisView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                appendLoc(baseLoc, "pseudo_dst_view_{1}", index), newRightBranch, viewMultiplier);
        auto normalizedShape = propagateReshapeCast(rewriter, singleAxisView, params.castOp, baseLoc,
                                                    printToString("right_{0}", index));

        auto dstView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(appendLoc(baseLoc, "right_dst_view_{0}", index),
                                                                        dstBuffer, params.leftConcatInputSize);

        auto copyShape = getShape(subViewOp->getResult(0)).toValues();
        copyShape[params.newConcatDim] = params.rightInputSize;

        return rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "right_copy_{0}", index), normalizedShape, dstView);
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer = createNewConcatBuffer(rewriter, subViewOp, distributedCopy,
                                               takeOpLoc(origConcatOp, llvm::StringLiteral("buf_{0}"), index));

        auto baseLoc = origConcatOp->getLoc();
        // Concat on same axis, so must do manual strided access
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto origConcatDimSize = getShape(origConcatOp->getResult(0))[params.origConcatDim];
        auto viewMultiplier = srcOffset[params.newConcatDim] / origConcatDimSize;

        auto newLeftViewBranch = rewriterLeftSubViewBranch(rewriter, params, viewMultiplier, subViewOp, newLeftBranch,
                                                           dstBuffer, baseLoc, index);

        auto newRightViewBranch = rewriterRightSubViewBranch(rewriter, params, viewMultiplier, subViewOp,
                                                             newRightBranch, dstBuffer, baseLoc, index);

        SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(
                takeOpLoc(origConcatOp, StringLiteral("concat_{0}"), index), concatInputs, dstBuffer);
        rewriter.replaceAllUsesWith(distributedCopy->getResult(0), newConcatOp);
        rewriter.eraseOp(distributedCopy);
        rewriter.eraseOp(subViewOp);
    }
};

class SplitUnbalancedDDRConcatOnSameAxisDDR : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    StringRef getRewriterSuffix() const override {
        return "OnSameAxisDDR";
    }

    bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const override {
        return newConcatDim.ind() == tilingDim;
    }

    std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const override {
        const size_t RIGHT_INPUT_ID = 1;  // Right must be always second to preserve concat order
        auto inputCopy = concatOp.getInputs()[RIGHT_INPUT_ID];
        if (auto copyOp = inputCopy.getDefiningOp<VPUIP::CopyOp>()) {
            if (!mlir::isa<mlir::BlockArgument>(copyOp.getInput()) &&
                !mlir::isa<VPUIP::DistributedBufferType>(copyOp.getInput().getType())) {
                return {copyOp.getInput(), {copyOp}};
            }
        }
        return {nullptr, {}};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput,
                                   VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc) const override {
        return propagateReshapeCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
    }

    // For this pattern we need to create tmp buffer in DDR
    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp viewOp, VPUIP::CopyOp,
                                      mlir::Location bufferLoc) const override {
        auto srcBufferType = viewOp.getType();
        return rewriter.create<mlir::memref::AllocOp>(bufferLoc, srcBufferType.cast<mlir::MemRefType>());
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer = createNewConcatBuffer(rewriter, subViewOp, distributedCopy,
                                               takeOpLoc(origConcatOp, llvm::StringLiteral("buf_{0}"), index));

        auto newConcatDim = params.newConcatDim;
        // Concat on same axis, so must do manual strided access
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto origConcatDimSize = getShape(origConcatOp->getResult(0))[params.origConcatDim];
        auto viewMultiplier = srcOffset[newConcatDim] / origConcatDimSize;
        auto createViewBranch = [&](mlir::Value src, int64_t origDimSize, int64_t concatDimSize, int64_t dstOffsetVal,
                                    int64_t srcBaseOffset, StringRef locSuffix) -> mlir::Value {
            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[newConcatDim] = concatDimSize;

            Shape srcOffset(SmallVector<int64_t>(copyShape.size(), 0));
            srcOffset[newConcatDim] = origDimSize * viewMultiplier + srcBaseOffset;

            Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
            dstOffset[newConcatDim] = dstOffsetVal;

            return createNewCopyBranch(rewriter, src, dstBuffer, copyShape, srcOffset, dstOffset,
                                       origConcatOp->getLoc(), locSuffix, index);
        };

        auto newLeftViewBranch = createViewBranch(newLeftBranch, params.leftInputSize, params.leftConcatInputSize,
                                                  /*dstOffsetVal=*/0, params.leftViewOffset, "left");
        auto newRightViewBranch = createViewBranch(newRightBranch, params.rightInputSize, params.rightInputSize,
                                                   params.leftConcatInputSize, /*srcOffset=*/0, "right");

        SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(
                takeOpLoc(origConcatOp, llvm::StringLiteral("concat_{0}"), index), concatInputs, dstBuffer);
        rewriter.replaceAllUsesWith(subViewOp->getResult(0), newConcatOp);
        rewriter.eraseOp(subViewOp);
    }
};

//
// OptimizeConcatViewCopiesPass
//

class OptimizeConcatViewCopiesPass final : public VPUIP::OptimizeConcatViewCopiesBase<OptimizeConcatViewCopiesPass> {
public:
    explicit OptimizeConcatViewCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeConcatViewCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AvoidConcatExtraChannel>(&ctx, _log);
    patterns.add<FuseConcatView>(&ctx, _log);
    patterns.add<RemoveDDRToDDRCopyAfterConcatView>(&ctx, _log);
    patterns.add<OptimizeDDR2DDRCopyInputsOfConcatView>(&ctx, _log);
    patterns.add<OptimizeConcatSubviewPattern>(&ctx, _log);
    patterns.add<SplitUnbalancedDDRConcatOnOtherAxis>(&ctx, _log);
    patterns.add<SplitUnbalancedDDRConcatOnSameAxis>(&ctx, _log);
    patterns.add<SplitUnbalancedDDRConcatOnSameAxisDDR>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatViewCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeConcatViewCopiesPass(Logger log) {
    return std::make_unique<OptimizeConcatViewCopiesPass>(log);
}