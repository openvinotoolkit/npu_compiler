//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vpux/utils/core/range.hpp>
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZECONCATVIEWCOPIES
#define GEN_PASS_DEF_OPTIMIZECONCATVIEWCOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

// Checks whether the given PermuteCast operation has a trivial permutation and if the input and output memory shapes
// are compatible with the permutation attribute.
// The function returns true for the following case:
// 1. PermuteCast from memref<1x32x1024x96xf16> to memref<1x96x32x1024xf16, #NHWC> with perm #NCHW, where memory shapes
// and permutation attribute are compatible.
// The function returns false for the following cases:
// 1.PermuteCast from memref<1x32x1024x96xf16> to memref<1x32x1024x96xf16, #NHWC> with perm #NHWC, which is not a
// trivial permutation.
// 2.PermuteCast from memref<1x512x216x26xf16, #NHWC> to memref<1x512x216x26xf16> with perm #NCHW, which is a
// trivial permutation but the memory shapes are not compatible with the permutation attribute.
bool checkMemShapesCompatibilityWithPerm(VPUIP::PermuteCastOp permuteCastOp) {
    auto inMemShape = getMemShape(permuteCastOp.getSource());
    auto outMemShape = getMemShape(permuteCastOp.getResult());
    auto perm = permuteCastOp.getMemPerm();
    if (!isTrivialPermute(inMemShape, perm)) {
        return false;
    }

    return applyPerm(inMemShape, perm) == outMemShape;
}

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
    Dim inferDimAfterPermuteCast(Dim origDim, VPUIP::PermuteCastOp permuteCast) const;

    mlir::LogicalResult checkConcatUsers(VPUIP::ConcatViewOp concatOp, std::optional<int64_t>& patternOutChannelSize,
                                         std::optional<int64_t>& patternOutChannelOffset,
                                         VPUIP::PermuteCastOp maybePermuteCast) const;
    mlir::LogicalResult checkConcatInputs(mlir::ValueRange concatInputs, mlir::Value concatOutput,
                                          const int64_t patternOutChannelSize, const int64_t patternOutChannelOffset,
                                          SmallVector<InputUpdateInfo>& inTilingCopiesInfo) const;

    mlir::Operation* createOutputBuffer(mlir::PatternRewriter& rewriter, VPUIP::CopyOp copyOp, int64_t channels) const;

    Logger _log;
};

Dim AvoidConcatExtraChannel::inferDimAfterPermuteCast(Dim origDim, VPUIP::PermuteCastOp permuteCast) const {
    const auto inOrder = mlir::cast<NDTypeInterface>(permuteCast.getSource().getType()).getDimsOrder();
    const auto outOrder = mlir::cast<NDTypeInterface>(permuteCast.getResult().getType()).getDimsOrder();
    const auto perm = permuteCast.getMemPerm();

    return inferDimAfterPermutation(origDim, inOrder, outOrder, perm);
}

// Check if all Concat users are Subview with same channels slice
// less than Concat channels (m > n)
//
//                      Concat (m output channels)
//                          |       ...       |
//  Subview (n output channels)     ...      Subview (n output channels)
//
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatUsers(VPUIP::ConcatViewOp concatOp,
                                                              std::optional<int64_t>& patternOutChannelSize,
                                                              std::optional<int64_t>& patternOutChannelOffset,
                                                              VPUIP::PermuteCastOp maybePermuteCast) const {
    Dim targetSubViewDim = Dims4D::Act::C;
    auto subviews = concatOp.getOutput().getUsers();
    if (maybePermuteCast != nullptr) {
        if (!checkMemShapesCompatibilityWithPerm(maybePermuteCast)) {
            return mlir::failure();
        }
        targetSubViewDim = inferDimAfterPermuteCast(targetSubViewDim, maybePermuteCast);
        subviews = maybePermuteCast.getResult().getUsers();
    }
    if (subviews.empty()) {
        return mlir::failure();
    }

    const auto concatOutChannelSize = getShape(concatOp.getOutput())[Dims4D::Act::C];
    for (const auto user : subviews) {
        auto subview = mlir::dyn_cast_if_present<VPUIP::SubViewOp>(user);

        if (subview == nullptr) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.getStaticOffsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());

        if (subview.getStaticStrides().has_value()) {
            return mlir::failure();
        }

        if (patternOutChannelOffset.has_value() && patternOutChannelOffset.value() != offsets[targetSubViewDim.ind()]) {
            return mlir::failure();
        }

        if (patternOutChannelSize.has_value() && patternOutChannelSize.value() != sizes[targetSubViewDim.ind()]) {
            return mlir::failure();
        }

        if (concatOutChannelSize <= sizes[targetSubViewDim.ind()]) {
            return mlir::failure();
        }

        patternOutChannelSize = sizes[targetSubViewDim.ind()];
        patternOutChannelOffset = offsets[targetSubViewDim.ind()];
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
        if (auto distributedCopyType =
                    mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(tilingCopy.getInput().getType())) {
            const auto distributionInfo = distributedCopyType.getDistribution();
            // TODO: E191948-support new mode
            if (distributionInfo.getMode().getValue() ==
                (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED)) {
                return mlir::failure();
            }
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

        auto concatDims = getConcatDims(ShapeRef(sizes), concatOutShape);
        if (concatDims.size() != 1 && llvm::find(concatDims, Dims4D::Act::C) != concatDims.end()) {
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
        if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyOpInput.getType())) {
            const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
            if (tileIndex.has_value()) {
                auto tileIndexVal = tileIndex.value();
                if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithDistributedCopy(to_small_vector(copyInOffsets),
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

    auto opOutputType = mlir::cast<vpux::NDTypeInterface>(subview.getSource().getType());
    auto sourceShape = opOutputType.getShape().toValues();
    sourceShape[Dims4D::Act::C] = channels;
    auto newOpOutputType = opOutputType.changeShape(ShapeRef(sourceShape));

    return VPUIP::allocateBuffersOfType(_log, copyOp->getLoc(), rewriter, newOpOutputType).front().getDefiningOp();
}

void recursivelyInferReturnTypes(mlir::Value value) {
    for (auto child : value.getUsers()) {
        if (mlir::isa_and_nonnull<VPUIP::SubViewOp, VPUIP::ShapeCastOp>(child)) {
            vpux::inferReturnTypes(child, vpux::InferShapedTypeMode::ALL);
            recursivelyInferReturnTypes(child->getResult(0));
        } else if (mlir::isa_and_nonnull<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp>(child)) {
            const auto inType = mlir::cast<vpux::NDTypeInterface>(child->getOperand(0).getType());
            const auto outType = mlir::cast<vpux::NDTypeInterface>(child->getResult(0).getType());
            const auto strideUpdatedOutType = VPUIP::updateStridesForReshape(inType, outType);
            VPUX_THROW_WHEN(mlir::failed(strideUpdatedOutType),
                            "Failed to update strides for input '{0}' and output '{1}'", inType, outType);
            child->getResult(0).setType(strideUpdatedOutType.value());
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

    VPUIP::PermuteCastOp maybePermuteCast = nullptr;
    if (concatOutput.hasOneUse()) {
        maybePermuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*concatOutput.getUsers().begin());
    }

    std::optional<int64_t> patternOutChannelSize = std::nullopt;
    std::optional<int64_t> patternOutChannelOffset = std::nullopt;
    if (checkConcatUsers(concatOp, patternOutChannelSize, patternOutChannelOffset, maybePermuteCast).failed()) {
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

    auto targetSubViewDim = Dims4D::Act::C;
    mlir::Operation* newOp =
            rewriter.create<VPUIP::ConcatViewOp>(concatOp.getLoc(), newConcatInputs, outputBuffer->getResult(0));
    if (maybePermuteCast == nullptr) {
        rewriter.replaceAllUsesWith(concatOp, newOp->getResult(0));
    } else {
        targetSubViewDim = inferDimAfterPermuteCast(targetSubViewDim, maybePermuteCast);
        auto origShape = getShape(maybePermuteCast.getResult());
        auto newShape = origShape.toValues();
        newShape[targetSubViewDim] = patternOutChannelSize.value();
        auto newType = mlir::cast<NDTypeInterface>(maybePermuteCast.getResult().getType()).changeShape(newShape);

        newOp = rewriter.create<VPUIP::PermuteCastOp>(maybePermuteCast.getLoc(), newType, newOp->getResult(0),
                                                      maybePermuteCast.getDstOrderAttr(),
                                                      maybePermuteCast.getMemPermAttr());
        rewriter.replaceAllUsesWith(maybePermuteCast, newOp->getResult(0));
    }

    for (auto user : newOp->getResult(0).getUsers()) {
        if (auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
            auto newOffsets = parseIntArrayAttr<int64_t>(subviewOp.getStaticOffsetsAttr());
            newOffsets[targetSubViewDim.ind()] = 0;
            auto newOffsetsAttr = getIntArrayAttr(subviewOp.getContext(), ArrayRef(newOffsets));
            subviewOp->setAttr(subviewOp.getStaticOffsetsAttrName(), newOffsetsAttr);
            vpux::inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);

            recursivelyInferReturnTypes(subviewOp);
        }
    }

    nestedLogger.trace("Successfully Avoid Concat Extra Channel {0}", concatOp->getLoc());

    if (maybePermuteCast != nullptr) {
        rewriter.eraseOp(maybePermuteCast);
    }
    rewriter.eraseOp(concatOp);

    for (auto inTilingCopyInfo : inTilingCopiesInfo) {
        rewriter.eraseOp(inTilingCopyInfo.distributedCopyOp);
    }

    return mlir::success();
}

//
// Helpers for FuseConcatView through GenericReshape
//

// Check if a GenericReshape only changes the last two logical dims (H, W)
// while preserving H*W. Handles flatten/unflatten of spatial dims.
static bool isCompatibleSpatialReshapeForFusion(VPUIP::GenericReshapeOp reshapeOp) {
    if (!reshapeOp.getOutput().hasOneUse()) {
        return false;
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(reshapeOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(reshapeOp.getOutput().getType());

    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();

    if (inShape.size() != 4 || outShape.size() != 4) {
        return false;
    }

    if (inType.getDimsOrder() != outType.getDimsOrder()) {
        return false;
    }

    const auto dimOrder = inType.getDimsOrder();
    if (dimOrder.dimPos(Dims4D::Act::H) + 1 != dimOrder.dimPos(Dims4D::Act::W)) {
        return false;
    }

    // N and C must be unchanged
    if (inShape[Dims4D::Act::N] != outShape[Dims4D::Act::N] || inShape[Dims4D::Act::C] != outShape[Dims4D::Act::C]) {
        return false;
    }

    // H*W must be preserved
    if (inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W] != outShape[Dims4D::Act::H] * outShape[Dims4D::Act::W]) {
        return false;
    }

    return true;
}

// Transform SubView offsets and sizes through a spatial reshape of the last two dims.
// The reshape changes [N,C,H_old,W_old] -> [N,C,H_new,W_new] where H_old*W_old = H_new*W_new.
// Returns true if the transformation is valid for the given tile.
static bool transformSubViewThroughSpatialReshape(VPUIP::GenericReshapeOp reshapeOp, SmallVector<int64_t>& offsets,
                                                  SmallVector<int64_t>& sizes) {
    auto inType = mlir::cast<vpux::NDTypeInterface>(reshapeOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(reshapeOp.getOutput().getType());

    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();

    const int64_t oldW = inShape[Dims4D::Act::W];
    const int64_t newW = outShape[Dims4D::Act::W];

    // Tile must cover full W dimension of the old shape
    if (offsets[Dims4D::Act::W.ind()] != 0 || sizes[Dims4D::Act::W.ind()] != oldW) {
        return false;
    }

    // Compute linear position in H*W space
    const int64_t linearStart = offsets[Dims4D::Act::H.ind()] * oldW;
    const int64_t linearSize = sizes[Dims4D::Act::H.ind()] * oldW;

    // Map to new shape coordinates
    if (linearStart % newW != 0 || linearSize % newW != 0) {
        return false;
    }

    offsets[Dims4D::Act::H.ind()] = linearStart / newW;
    offsets[Dims4D::Act::W.ind()] = 0;
    sizes[Dims4D::Act::H.ind()] = linearSize / newW;
    sizes[Dims4D::Act::W.ind()] = newW;

    return true;
}

//
// FuseConcatView
//

/*
    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp
               \                 /
                ConcatView1 (DDR)
                        |
                CopyOp(DDR2DDR)      TilingCopyOp/CopyOp
                        \              /
                        ConcatView2 (DDR)


    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp     TilingCopyOp/CopyOp
                     \                 |                  /
                                ConcatView2 (DDR)

or if ConcatView1 has multi users, use CMX2DDR stride Copy to replace DDR2DDR stride Copy will be beneficial:

    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp
               \                 /
                ConcatView1 (DDR)
                /            \
        ShapeCastOp         CopyOp(DDR2DDR)      TilingCopyOp/CopyOp
                                \              /
                                ConcatView2 (DDR)


TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp  +  TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp     TilingCopyOp/CopyOp
          \                 /                                    \                 |                  /
            ConcatView1 (DDR)                                                ConcatView2 (DDR)
                |
            ShapeCastOp

or if ConcatView1 is followed by a GenericReshape (spatial flatten) before the DDR2DDR copy:

    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp
               \                 /
                ConcatView1 (DDR)
                        |
                GenericReshape (spatial flatten)
                        |
                CopyOp(DDR2DDR)      TilingCopyOp/CopyOp
                        \              /
                        ConcatView2 (DDR)

    Tile offsets/sizes are transformed through the reshape before composing with ConcatView2 SubView offsets.
*/

// Holds the ops matched by the FuseConcatView pattern
struct FuseConcatViewMatch {
    VPUIP::CopyOp userCopyOp{nullptr};
    VPUIP::GenericReshapeOp intermediateReshape{nullptr};
};

class FuseConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseConcatView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    bool isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log, FuseConcatViewMatch& match) const;
    bool hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log,
                                              FuseConcatViewMatch& match) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter,
                                                vpux::Logger log, FuseConcatViewMatch match) const;

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

bool FuseConcatView::hasDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log,
                                                          FuseConcatViewMatch& match) const {
    log.nest().trace("Checking hasDDR2DDRCopyWithConcatViewConsumer");

    auto isTargetCopy = [](VPUIP::CopyOp copyOp) {
        if (!copyOp.getOutput().hasOneUse()) {
            return false;
        }

        if (!mlir::isa<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin())) {
            return false;
        }
        return VPUIP::isCopyFromDDR(copyOp) && VPUIP::isCopyToDDR(copyOp);
    };

    VPUIP::CopyOp copyOp{nullptr};
    for (auto user : concatViewOp.getOutput().getUsers()) {
        copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*user);
        if (copyOp != nullptr && isTargetCopy(copyOp)) {
            match.userCopyOp = copyOp;
            match.intermediateReshape = nullptr;
            return true;
        }
    }

    // Multi-user case with intermediate reshape is not supported.
    // Return early to avoid creating ops before detecting the unsupported pattern.
    if (!concatViewOp.getOutput().hasOneUse()) {
        return false;
    }

    // Look through single-use GenericReshape (spatial flatten/unflatten)
    for (auto user : concatViewOp.getOutput().getUsers()) {
        auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(*user);
        if (reshapeOp == nullptr) {
            continue;
        }
        if (!isCompatibleSpatialReshapeForFusion(reshapeOp)) {
            continue;
        }
        for (auto reshapeUser : reshapeOp.getOutput().getUsers()) {
            copyOp = mlir::dyn_cast<VPUIP::CopyOp>(reshapeUser);
            if (copyOp != nullptr && isTargetCopy(copyOp)) {
                match.userCopyOp = copyOp;
                match.intermediateReshape = reshapeOp;
                log.nest().trace("Found DDR2DDR copy through GenericReshape");
                return true;
            }
        }
    }

    return false;
}

// Fuse ConcatView Ops to remove unnecessary copies, two conditions need to be satisfied:
// a) The Stride Level for each ConcatView input (after fusing) should be no more than 2;
//     It's a runtime and HW limitation in order to get the right NNDMA descriptor, we support a maximum of 3D DMA
//     transfers with 2 levels of striding.
// b) The number of inputs from the second ConcatView, which come from the output of the first should no more than 1;
//     For example, first ConcatView has M inputs, second ConcatView has N inputs, out of which P of them are the output
//     of the first ConcatView After fusing, the number of input copies is: M * P + (N - P)
//     Can't ensure we get benefit when P is of a large size. Limit optimization to P=1.
bool FuseConcatView::isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log,
                                              FuseConcatViewMatch& match) const {
    if (concatViewOp.getOutput().use_empty()) {
        log.nest().trace("Cannot find user copy op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasCopyOpForAllInputs(concatViewOp, log)) {
        log.nest().trace("Not all inputs is CopyOp for first ConcatViewOp at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasDDR2DDRCopyWithConcatViewConsumer(concatViewOp, log, match)) {
        log.nest().trace("Not only one user is DDR2DDR copy with ConcatViewOp for op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    log.nest().trace("FuseConcatView: Found legal ConcatView pattern at op '{0}'", concatViewOp->getLoc());

    return true;
}

mlir::LogicalResult FuseConcatView::fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp,
                                                            mlir::PatternRewriter& rewriter, vpux::Logger log,
                                                            FuseConcatViewMatch match) const {
    const bool hasMultiUsers = !concatViewOp.getOutput().hasOneUse();

    // Get current concat's memref.alloc op, which will be removed
    auto firstConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.getOutputBuff());
    if (firstConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because current concat '{0}' output isn't master buffer",
                         concatViewOp->getLoc());
        return mlir::failure();
    }

    if (!hasMultiUsers) {
        for (auto user : firstConcatMemAlloc->getResult(0).getUsers()) {
            // Allow the ConcatView itself to use the buffer
            if (user == concatViewOp.getOperation()) {
                continue;
            }

            auto subView = mlir::dyn_cast<VPUIP::SubViewOp>(user);
            if (subView == nullptr) {
                log.nest().trace("Alloc has non-SubView user, cannot fuse");
                return mlir::failure();
            }

            bool feedsIntoConcat = false;
            for (auto svUser : subView->getResult(0).getUsers()) {
                if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(svUser)) {
                    if (llvm::is_contained(concatViewOp.getInputs(), copyOp.getOutput())) {
                        feedsIntoConcat = true;
                        break;
                    }
                }
            }

            if (!feedsIntoConcat) {
                log.nest().trace("Alloc has SubView not feeding into concat, cannot fuse");
                return mlir::failure();
            }
        }
    }

    VPUX_THROW_UNLESS(match.userCopyOp != nullptr, "Cannot get DDR to DDR Copy Op after '{0}'", concatViewOp->getLoc());
    auto outCopySubView = match.userCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

    auto nextConcatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*match.userCopyOp.getOutput().getUsers().begin());
    if (nextConcatViewOp == nullptr) {
        log.nest().trace("Cannot get the next ConcatView op '{0}' for output Copy op", match.userCopyOp->getLoc());
        return mlir::failure();
    }
    const auto dimOrder = mlir::cast<NDTypeInterface>(nextConcatViewOp.getOutput().getType()).getDimsOrder();
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);

    auto concatDims = VPUIP::getConcatAxes(nextConcatViewOp);
    const auto outputCopyOutputShape = getShape(match.userCopyOp.getOutput());

    // When the ConcatView1 has multi users, only the concat is on the lowest dim and the concat stride is small, it's
    // greatly beneficial to use DDR to DDR copy replace CMX to DDR copy.
    if (hasMultiUsers &&
        (concatDims.size() != 1 || *concatDims.begin() != lowestDim || outputCopyOutputShape[lowestDim] != 1)) {
        return mlir::failure();
    }

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
        if (!(siblingCopyOp && siblingCopyOp == match.userCopyOp)) {
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

            // Transform tile offsets/sizes through the intermediate reshape if present
            if (match.intermediateReshape != nullptr) {
                if (!transformSubViewThroughSpatialReshape(match.intermediateReshape, inCopyOffsets, inCopySizes)) {
                    log.nest().trace("Cannot transform tile SubView through reshape");
                    return mlir::failure();
                }
            }

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

            // When there is an intermediate reshape, the SubView has the reshaped tile shape
            // (e.g. [1,512,32,1]), but the CMX source has the original tile shape (e.g. [1,512,8,4]).
            // Reshape the destination SubView to match the source shape instead of reshaping the source,
            // because the source may be a DistributedBuffer that cannot be trivially reshaped.
            auto copySource = op->getOperand(0);
            mlir::Value copyDest = newSubViewOp.getResult();
            if (match.intermediateReshape != nullptr) {
                const auto srcShape = getShape(copySource);
                auto subViewNDType = mlir::cast<vpux::NDTypeInterface>(copyDest.getType());

                // Only reshape if shapes actually differ (they always should when reshape is present)
                if (srcShape != subViewNDType.getShape()) {
                    // Compute strides for the unflattened shape.
                    // Determine the per-spatial-element stride in the linearized H*W space.
                    // When the SubView W = 1 (spatial dims flattened into H), each spatial
                    // element is separated by the H stride, not the W stride.
                    // When W > 1, each spatial step uses the W stride (contiguous H*W).
                    const auto subViewStrides = subViewNDType.getStrides();
                    const auto subViewShape = subViewNDType.getShape();
                    const auto wNew = srcShape[Dims4D::Act::W];
                    const auto spatialStride = (subViewShape[Dims4D::Act::W] == 1) ? subViewStrides[Dims4D::Act::H]
                                                                                   : subViewStrides[Dims4D::Act::W];

                    SmallVector<Bit> newDestStrides(4);
                    newDestStrides[Dims4D::Act::N.ind()] = subViewStrides[Dims4D::Act::N];
                    newDestStrides[Dims4D::Act::C.ind()] = subViewStrides[Dims4D::Act::C];
                    newDestStrides[Dims4D::Act::H.ind()] = Bit(spatialStride.count() * wNew);
                    newDestStrides[Dims4D::Act::W.ind()] = spatialStride;

                    auto reshapedDestType =
                            subViewNDType.changeShape(srcShape).changeStrides(StridesRef(newDestStrides));
                    copyDest = rewriter.create<VPUIP::GenericReshapeOp>(op->getLoc(), reshapedDestType,
                                                                        newSubViewOp.getResult());
                }
            }

            auto newCopyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), copySource, copyDest);

            if (!VPUIP::hasLegalStridingLevel(newCopyOp)) {
                log.nest().trace("DMA Striding Level is illegal. Fusing Concat Op have no benefit");
                rewriter.eraseOp(newCopyOp);
                if (copyDest != newSubViewOp.getResult()) {
                    rewriter.eraseOp(copyDest.getDefiningOp());
                }
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
    rewriter.eraseOp(match.userCopyOp);
    rewriter.eraseOp(outCopySubView);
    if (match.intermediateReshape != nullptr) {
        rewriter.eraseOp(match.intermediateReshape);
    }
    if (!hasMultiUsers) {
        rewriter.eraseOp(concatViewOp);

        for (size_t inIdx = 0; inIdx < oldCopyInputs.size(); ++inIdx) {
            rewriter.eraseOp(oldCopyInputs[inIdx].getDefiningOp());
            rewriter.eraseOp(oldSubViewInputs[inIdx]);
        }

        rewriter.eraseOp(firstConcatMemAlloc);
    }

    return mlir::success();
}

mlir::LogicalResult FuseConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    FuseConcatViewMatch match{};
    if (!isLegalConcatViewPattern(concatViewOp, _log, match)) {
        _log.nest().trace("FuseConcatView: Cannot rewrite this concat Op");
        return mlir::failure();
    }

    return fuseTwoConcatViewInputs(concatViewOp, rewriter, _log, match);
}

//
// ReuseConcatViewAsInput
//

/*
                       Input1 (CMX)              Input2 (CMX)
                      /            \               /    \
    TilingCopyOp(CMX2DDR)    AvgPoolOp(Identity)  /    TilingCopyOp(CMX2DDR)
                      \               \          /         /
                       \            ConcatView (CMX)      /
                        \                  |             /
                         \    NCEClusterTask/SwKernel   /
                          \                |           /
                           \   TilingCopyOp(CMX2DDR)  /
                            \              |         /
                                 ConcatView (DDR)

    ==>
                            Input1 (CMX)     Input2 (CMX)
                                  \          /
                     AvgPoolOp(Identity)    /
                                     \     /
                                  ConcatView (CMX)
                                  /          \
                                 |          NCEClusterTask/SwKernel
                                 |            |
                    TilingCopyOp(CMX2DDR)   TilingCopyOp(CMX2DDR)
                                  \          /
                                  ConcatView (DDR)
*/

class ReuseConcatViewAsInput final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    ReuseConcatViewAsInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    bool isIdentityPool(VPUIP::NCEClusterTaskOp avgPoolOp) const;
    bool isLegalConcatViewInputPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    VPUIP::CopyOp getResultCMXToDDRCopy(mlir::Operation* op) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult reuseConcatViewInputs(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter,
                                              vpux::Logger log) const;

private:
    Logger _log;
};

bool ReuseConcatViewAsInput::isIdentityPool(VPUIP::NCEClusterTaskOp avgPoolOp) const {
    const auto inputType = mlir::cast<NDTypeInterface>(avgPoolOp.getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(avgPoolOp.getOutput().getType());
    if (inputType.getShape() != outputType.getShape() || inputType.getElementType() != outputType.getElementType()) {
        return false;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(avgPoolOp.getKernelSizeAttr());
    const auto strides = parseIntArrayAttr<int64_t>(avgPoolOp.getKernelStridesAttr());
    const auto pads = avgPoolOp.getKernelPaddingAttr();
    const auto isOne = [](const int64_t val) -> bool {
        return val == 1;
    };

    if ((!llvm::all_of(kernelSize, isOne)) || (!llvm::all_of(strides, isOne))) {
        return false;
    }

    if (pads.getLeft().getInt() != 0 || pads.getRight().getInt() != 0 || pads.getTop().getInt() != 0 ||
        pads.getBottom().getInt() != 0) {
        return false;
    }

    const auto ppeOpaqueAttr = VPU::getPpeConfig(avgPoolOp.getContext()).retrievePPEAttribute(avgPoolOp);
    const auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeOpaqueAttr);
    if (intPpeAttr != nullptr && intPpeAttr.getMode().getValue() != VPU::PPEMode::NOOP) {
        return false;
    }

    return true;
}

VPUIP::CopyOp ReuseConcatViewAsInput::getResultCMXToDDRCopy(mlir::Operation* op) const {
    for (auto result : op->getResults()) {
        if (!result.hasOneUse()) {
            continue;
        }
        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*result.getUsers().begin());
        if (copyOp == nullptr || copyOp.use_empty()) {
            continue;
        }
        if (VPUIP::isCopyFromDDR(copyOp) || !VPUIP::isCopyToDDR(copyOp)) {
            continue;
        }
        return copyOp;
    }
    return nullptr;
}

bool ReuseConcatViewAsInput::isLegalConcatViewInputPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    // Check the pattern from the CMX ConcatViewOp, it has a user
    if (concatViewOp.getOutput().use_empty()) {
        log.nest().trace("Cannot find user op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasOneUniqueUser(concatViewOp.getOperation())) {
        log.nest().nest().trace("ConcatViewOp has more than one user");
        return false;
    }

    auto concatUserOp = *concatViewOp.getOutput().getUsers().begin();
    if (!mlir::isa<VPUIP::NCEClusterTaskOp, VPUIP::SwKernelOp>(concatUserOp)) {
        log.nest().nest().trace("ConcatViewOp has non NCEClusterTask or SwKernel user");
        return false;
    }

    // For multi-result ops (e.g. TopK), find the actual CMX-to-DDR CopyOp among all results.
    auto copyOp = getResultCMXToDDRCopy(concatUserOp);
    if (copyOp == nullptr) {
        log.nest().nest().trace("Consumer of concatViewOp has no valid result with a CMX-to-DDR copyOp user");
        return false;
    }

    // Check the copyOp has a DDR ConcatViewOp user
    auto copyUser = copyOp.getOutput().getUsers().begin();
    auto userConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyUser);
    if (userConcatOp == nullptr) {
        log.nest().nest().trace("Consumer of copyOp is not concatViewOp");
        return false;
    }

    // Check the user DDR ConcatViewOp contains all inputs of the CMX ConcatViewOp
    SmallVector<VPUIP::SubViewOp> preSubViews;
    SmallVector<VPUIP::SubViewOp> nextSubViews;
    SmallVector<mlir::Value> preParents;
    SmallVector<mlir::Value> nextParents;

    // Get the inputs and subviews of CMX ConcatViewOp
    for (auto input : concatViewOp.getInputs()) {
        auto nceClusterTaskOp = mlir::dyn_cast_if_present<VPUIP::NCEClusterTaskOp>(input.getDefiningOp());
        if (nceClusterTaskOp == nullptr) {
            return false;
        }

        auto curParent = input;
        if (nceClusterTaskOp.getTaskType() == VPUIP::NCETaskType::AVEPOOL && isIdentityPool(nceClusterTaskOp)) {
            curParent = nceClusterTaskOp.getInput();
        }

        auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(nceClusterTaskOp.getOutputs()[0].getDefiningOp());
        if (subviewOp == nullptr) {
            return false;
        }

        preSubViews.push_back(subviewOp);
        preParents.push_back(curParent);
    }

    // Get the inputs and subviews of user DDR ConcatViewOp
    for (auto input : userConcatOp.getInputs()) {
        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(input.getDefiningOp());
        if (copyOp == nullptr || !copyOp->hasOneUse()) {
            return false;
        }

        auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(copyOp.getOutputs()[0].getDefiningOp());
        if (subviewOp == nullptr) {
            return false;
        }

        nextSubViews.push_back(subviewOp);
        nextParents.push_back(copyOp.getInput());
    }

    auto isSameAttrSubview = [](VPUIP::SubViewOp inSubview, VPUIP::SubViewOp outSubview) {
        return inSubview.getStaticOffsetsAttr() == outSubview.getStaticOffsetsAttr() &&
               inSubview.getStaticSizesAttr() == outSubview.getStaticSizesAttr() &&
               inSubview.getStaticStridesAttr() == outSubview.getStaticStridesAttr();
    };

    // Check the same attributes and inputs between CMX ConcatViewOp and DDR ConcatViewOp
    for (auto inIdx : irange(preSubViews.size())) {
        if (!isSameAttrSubview(preSubViews[inIdx], nextSubViews[inIdx])) {
            return false;
        }

        if (preParents[inIdx] != nextParents[inIdx]) {
            return false;
        }
    }

    log.nest().trace("ReuseConcatViewAsInput: Found legal ConcatView pattern at op '{0}'", concatViewOp->getLoc());

    return true;
}

mlir::LogicalResult ReuseConcatViewAsInput::reuseConcatViewInputs(VPUIP::ConcatViewOp concatViewOp,
                                                                  mlir::PatternRewriter& rewriter,
                                                                  vpux::Logger log) const {
    auto concatUserOp = *concatViewOp.getOutput().getUsers().begin();
    auto copyOp = getResultCMXToDDRCopy(concatUserOp);
    VPUX_THROW_UNLESS(copyOp != nullptr, "ReuseConcatViewAsInput: no valid CMX-to-DDR CopyOp found at '{0}'",
                      concatUserOp->getLoc());
    auto userConcatOp = mlir::cast<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin());
    auto preSubviewsSize = concatViewOp.getInputs().size();
    SmallVector<VPUIP::SubViewOp> nextSubViews;
    SmallVector<VPUIP::CopyOp> nextCopys;

    // Get the copyOps and subviews of user DDR ConcatViewOp
    for (auto input : userConcatOp.getInputs()) {
        auto copyOp = mlir::cast<VPUIP::CopyOp>(input.getDefiningOp());
        auto subviewOp = mlir::cast<VPUIP::SubViewOp>(copyOp.getOutputs()[0].getDefiningOp());
        nextCopys.push_back(copyOp);
        nextSubViews.push_back(subviewOp);
    }

    auto concatOutType = mlir::cast<NDTypeInterface>(concatViewOp.getOutput().getType());
    auto concatOutShape = concatOutType.getShape().raw();
    SmallVector<int64_t> firstSubviewOffsets(concatOutShape.size(), 0);
    SmallVector<int64_t> firstSubviewSizes(concatOutShape.size());
    for (size_t idx = 0; idx < firstSubviewSizes.size(); ++idx) {
        firstSubviewSizes[idx] = concatOutShape[idx];
    }

    // Create new output buff
    auto origOutputBuff = userConcatOp.getOutputBuff();
    auto opOutputType = mlir::cast<vpux::NDTypeInterface>(origOutputBuff.getType());
    auto* outputBuffer =
            VPUIP::allocateBuffersOfType(log, copyOp->getLoc(), rewriter, opOutputType).front().getDefiningOp();

    // Create first subViewOp for copyOp
    rewriter.setInsertionPoint(userConcatOp);
    auto firstSubViewOp = rewriter.create<VPUIP::SubViewOp>(userConcatOp->getLoc(), outputBuffer->getResult(0),
                                                            firstSubviewOffsets, firstSubviewSizes);

    // Create first copyOp which copy from the CMX ConcatViewOp output
    auto firstCopyOp = rewriter.create<VPUIP::CopyOp>(userConcatOp->getLoc(), concatViewOp.getOutput(),
                                                      firstSubViewOp.getResult());

    // Update output buffer for subviewOp and copyOp, and create new inputs for user DDR ConcatViewOp
    SmallVector<mlir::Value> newConcatsInputs;
    newConcatsInputs.push_back(firstCopyOp.getOutput());

    for (size_t inIdx = preSubviewsSize; inIdx < nextSubViews.size(); inIdx++) {
        VPUIP::SubViewOp subviewOp = nextSubViews[inIdx];
        auto newSubViewOp = rewriter.replaceOpWithNewOp<VPUIP::SubViewOp>(subviewOp, outputBuffer->getResult(0),
                                                                          subviewOp.getStaticOffsetsAttr(),
                                                                          subviewOp.getStaticSizesAttr());

        VPUIP::CopyOp copyOp = nextCopys[inIdx];
        copyOp = rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, copyOp.getInput(), newSubViewOp.getResult());

        newConcatsInputs.push_back(copyOp.getOutput());
    }

    // Update user DDR ConcatViewOp
    rewriter.setInsertionPoint(userConcatOp);
    userConcatOp = rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(userConcatOp, userConcatOp.getOutput().getType(),
                                                                    newConcatsInputs, outputBuffer->getResult(0));

    for (size_t inIdx = 0; inIdx < preSubviewsSize; ++inIdx) {
        rewriter.eraseOp(nextCopys[inIdx]);
        rewriter.eraseOp(nextSubViews[inIdx]);
    }
    rewriter.eraseOp(origOutputBuff.getDefiningOp());

    log.nest().trace("ReuseConcatViewAsInput: Finish reuse ConcatView Inputs at op '{0}'", concatViewOp->getLoc());

    return mlir::success();
}

mlir::LogicalResult ReuseConcatViewAsInput::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("ReuseConcatViewAsInput: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!isLegalConcatViewInputPattern(concatViewOp, _log)) {
        _log.nest().trace("ReuseConcatViewAsInput: Cannot rewrite this concat Op");
        return mlir::failure();
    }

    return reuseConcatViewInputs(concatViewOp, rewriter, _log);
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
    if (mlir::isa<mlir::BlockArgument>(targetCopyOp.getOutputBuff())) {
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

     DistributedCopy  plain CMX→DDR  ...    CopyOp
        (CMX -> DDR)   (CMX -> DDR)       (DDR -> DDR)
               \             |               /
                         ConcatView (DDR)
                                |
                        (Pure View Ops)
                                |
                          DistributedCopy
                           (DDR -> CMX)
                                |

    to:

       DistributedCopy / plain CMX -> DDR
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

    DDR -> DDR inputs go directly to SubView(CMX output buffer).
    CMX inputs (distributed or plain) are staged through a small DDR temp buffer.
*/

struct ConcatInputs {
    SmallVector<mlir::Value> inputCopies;
    SmallVector<mlir::Value> inputDistributedCopies;
    SmallVector<mlir::Value> inputNonDistributedCopies;
};

struct ConcatOutputs {
    SmallVector<mlir::Operation*> viewLikeOps;
    VPUIP::CopyOp outputDistributedCopy;
};

struct ConcatOutputsOfSubViewCopyUsers {
    VPUIP::PermuteCastOp permuteCastOp;
    SmallVector<VPUIP::SubViewOp> outputSubViewOps;
    SmallVector<VPUIP::CopyOp> outputCopyOps;
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

    void convertCopyInputAndStore(ArrayRef<mlir::Value> inputCopies, mlir::Value outputBuffer,
                                  SmallVector<mlir::Value>& newConcatInputs, mlir::PatternRewriter& rewriter) const;
    void convertDistributedCopyInputAndStore(ArrayRef<mlir::Value> inputDistributedCopies, mlir::Value outputBuffer,
                                             SmallVector<mlir::Value>& newConcatInputs,
                                             mlir::PatternRewriter& rewriter) const;

    // Functions for output pattern: Copy user distribution is DUPLICATED
    mlir::FailureOr<ConcatOutputs> getValidConcatOutputsOfDuplicatedCopyUser(VPUIP::ConcatViewOp concatViewOp,
                                                                             const ConcatInputs& concatInputs) const;
    mlir::LogicalResult processConcatOutputsOfDuplicatedCopyUser(VPUIP::ConcatViewOp concatViewOp,
                                                                 const ConcatInputs& concatInputs,
                                                                 const ConcatOutputs& concatOutputs,
                                                                 mlir::PatternRewriter& rewriter) const;
    VPUIP::DistributedBufferType getDuplicatedDistributedType(NDTypeInterface ndType,
                                                              VPUIP::DistributedBufferType distributedType,
                                                              mlir::MLIRContext* ctx) const;
    mlir::Value rewriteViewLikeOpsDuplicated(mlir::Value input, ArrayRef<mlir::Operation*> viewLikeOps,
                                             VPUIP::DistributedBufferType origOutputBufferType,
                                             mlir::PatternRewriter& rewriter) const;

    // Functions for output pattern: Copy user distribution is SEGMENTED
    mlir::FailureOr<ConcatOutputs> getValidConcatOutputsOfSegmentedCopyUser(VPUIP::ConcatViewOp concatViewOp) const;
    mlir::FailureOr<ConcatOutputs> getValidUnbalancedConcat(VPUIP::ConcatViewOp concatViewOp,
                                                            const ConcatInputs& concatInputs) const;
    mlir::LogicalResult processConcatOutputsOfSegmentedCopyUser(VPUIP::ConcatViewOp concatViewOp,
                                                                const ConcatInputs& concatInputs,
                                                                const ConcatOutputs& concatOutputs,
                                                                mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult processUnbalancedConcat(VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs,
                                                const ConcatOutputs& concatOutputs,
                                                mlir::PatternRewriter& rewriter) const;

    VPUIP::DistributedBufferType getSegmentedDistributedType(mlir::MLIRContext* ctx, NDTypeInterface ndType,
                                                             int64_t tilingDim,
                                                             VPU::DistributionInfoAttr origDistribution) const;
    mlir::Value rewriteViewLikeOpsSegmented(mlir::Value input, ArrayRef<Dim> tilingDims,
                                            ArrayRef<mlir::Operation*> viewLikeOps,
                                            VPUIP::DistributedBufferType origOutputBufferType,
                                            mlir::PatternRewriter& rewriter) const;
    mlir::FailureOr<SmallVector<Dim>> backInferDimAfterChangedByViewLikeOperations(
            Dim origDim, ArrayRef<mlir::Operation*> viewLikeOps) const;

    bool checkConcatReshapeCompatibility(VPUIP::ConcatViewOp concatOp, VPUIP::GenericReshapeOp genReshapeOp,
                                         VPUIP::CopyOp copyOp) const;

    // Functions for output pattern: Users are SubView + DUPLICATED distributed Copy branches
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

    auto newDistribution =
            VPU::DistributionInfoAttr::get(ctx, distribution.getMode(), distribution.getNumTiles(), nullptr, nullptr,
                                           nullptr, distribution.getNumClusters(), nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, distribution.getMemoryNumTiles());

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, orderMap, memSpace, newDistribution);
};

VPUIP::DistributedBufferType OptimizeDDR2DDRCopyInputsOfConcatView::getSegmentedDistributedType(
        mlir::MLIRContext* ctx, NDTypeInterface ndType, int64_t tilingDim,
        VPU::DistributionInfoAttr origDistribution) const {
    auto getNumTiles = [](int64_t rank, mlir::IntegerAttr tileCount, int64_t tilingDim) {
        VPUX_THROW_WHEN(tilingDim >= rank, "Tiling dim {0} is out of rank {1} range", tilingDim, rank);
        SmallVector<int64_t> numTiles(rank, 1);
        numTiles[tilingDim] = tileCount.getInt();
        return numTiles;
    };
    const auto tileCount = origDistribution.getNumClusters();
    const auto distMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(ctx, getNumTiles(ndType.getRank(), tileCount, tilingDim));

    // Distributed type with alignment is not supported right now
    mlir::ArrayAttr alignmentAttr = nullptr;

    const auto uniformDistributedSegmentsAttr = origDistribution.getUniformDistributedSegments();
    auto distributionAttr = VPU::DistributionInfoAttr::get(ctx, distMode, numTiles, nullptr, nullptr, nullptr,
                                                           tileCount, alignmentAttr, uniformDistributedSegmentsAttr,
                                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto orderMap = mlir::AffineMapAttr::get(ndType.getDimsOrder().toAffineMap(ctx));
    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistribution)) {
        distributionAttr = VPU::getNonOverlappedDistributedAttr(shape, distMode, numTiles, tileCount, alignmentAttr,
                                                                uniformDistributedSegmentsAttr, ctx);
    }

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, orderMap, memSpace, distributionAttr);
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
        const auto inDistributedType =
                mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(tilingCopyInput));
        VPUX_THROW_UNLESS(inDistributedType != nullptr, "Cannot get distributedType");

        auto distribution = inDistributedType.getDistribution();
        return VPU::isDuplicated(distribution);
    };

    // Non-distributed CMX -> DDR
    const auto isCMXToDDRCopy = [](mlir::Value input) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        if (!copyOp || vpux::VPUIP::hasDistributedOperand(copyOp)) {
            return false;
        }
        if (!VPUIP::isCopyToDDR(copyOp) || VPUIP::isCopyFromDDR(copyOp)) {
            return false;
        }
        return copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>() != nullptr;
    };

    ConcatInputs validInputs;

    for (const auto& input : concatViewOp.getInputs()) {
        if (isDDR2DDRCopy(input)) {
            validInputs.inputCopies.push_back(input);
        } else if (isDuplicatedDistributedCopy(input)) {
            validInputs.inputDistributedCopies.push_back(input);
        } else if (isCMXToDDRCopy(input)) {
            validInputs.inputNonDistributedCopies.push_back(input);
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
mlir::FailureOr<ConcatOutputs> OptimizeDDR2DDRCopyInputsOfConcatView::getValidConcatOutputsOfDuplicatedCopyUser(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs) const {
    if (!concatViewOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    ConcatOutputs validOutput;

    auto copyAfterViewLikeOps = searchCopyOpThroughViewLikeOps(concatViewOp, validOutput.viewLikeOps);
    if (mlir::failed(copyAfterViewLikeOps)) {
        _log.nest().trace("[{0}] Invalid output: no CopyOp after viewlike ops", getDebugName());
        return mlir::failure();
    }

    // The case is handled by SplitUnbalancedDDRConcat rewriter.
    if (!validOutput.viewLikeOps.empty() && !concatInputs.inputNonDistributedCopies.empty()) {
        return mlir::failure();
    }

    const auto isDuplicatedChildDistributedCopyOp = [](mlir::Operation* op) {
        auto copyOp = mlir::dyn_cast_if_present<VPUIP::CopyOp>(op);
        if (copyOp == nullptr || !VPUIP::isCopyFromDDR(copyOp) || VPUIP::isCopyToDDR(copyOp)) {
            return false;
        }

        auto tilingCopyOutput = copyOp->getResult(0);
        const auto outputDistributedType =
                mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(tilingCopyOutput));
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

mlir::FailureOr<SmallVector<Dim>> OptimizeDDR2DDRCopyInputsOfConcatView::backInferDimAfterChangedByViewLikeOperations(
        Dim origDim, ArrayRef<mlir::Operation*> viewLikeOps) const {
    Dim currentDim = origDim;
    SmallVector<Dim> tilingDims = {currentDim};
    for (auto viewLikeOp : viewLikeOps | reversed) {
        auto inputType = mlir::cast<NDTypeInterface>(viewLikeOp->getOperand(0).getType());
        auto outputType = mlir::cast<NDTypeInterface>(viewLikeOp->getResult(0).getType());
        auto inOrder = inputType.getDimsOrder();
        auto outOrder = outputType.getDimsOrder();
        if (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::ShapeCastOp>(viewLikeOp)) {
            auto currentDimOpt = VPUIP::getDistributedOutTilingAxisAfterShapeChanged(
                    outputType.getShape(), std::move(outOrder), inputType.getShape(), std::move(inOrder),
                    currentDim.ind(), _log);
            if (mlir::failed(currentDimOpt)) {
                return mlir::failure();
            }

            currentDim = Dim(currentDimOpt.value());
            _log.trace("[DEBUG][backInferDimAfterChangedByViewLikeOperations]Original dim {0} -> current dim {1} after "
                       "shape changed",
                       origDim, currentDim);
        } else if (mlir::isa<VPUIP::PermuteCastOp>(viewLikeOp)) {
            auto permuteCastOp = mlir::cast<VPUIP::PermuteCastOp>(viewLikeOp);
            auto perm = permuteCastOp.getMemPerm();
            auto inVersedPerm = mlir::inversePermutation(perm);

            auto inferDim = inferDimAfterPermutation(currentDim, std::move(outOrder), std::move(inOrder), inVersedPerm);
            currentDim = inferDim;
            _log.debug("[DEBUG][backInferDimAfterChangedByViewLikeOperations]Original dim {0} -> current dim {1} after "
                       "PermuteCast operation",
                       origDim, currentDim);
        } else {
            _log.nest().trace("Unsupported view like operation");
            return mlir::failure();
        }

        tilingDims.insert(tilingDims.begin(), currentDim);
    }

    return tilingDims;
}

std::optional<int64_t> getMultiClusterTilingAxis(VPU::DistributionInfoAttr distribution, Logger log) {
    const auto mode = distribution.getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED) {
        return std::nullopt;
    }

    int64_t tileIndex = -1;
    const auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    for (size_t i = 0; i < numTiles.size(); ++i) {
        if (numTiles[i] > 1) {
            if (tileIndex != -1) {
                log.trace("distributed buffer only supports tiling on single dimension");
                return std::nullopt;
            }
            tileIndex = checked_cast<int64_t>(i);
        }
    }

    return tileIndex;
}

bool OptimizeDDR2DDRCopyInputsOfConcatView::checkConcatReshapeCompatibility(VPUIP::ConcatViewOp concatOp,
                                                                            VPUIP::GenericReshapeOp reshapeOp,
                                                                            VPUIP::CopyOp concatOutCopyOp) const {
    const auto reshapeType = vpux::getBufferType(reshapeOp.getOutput());
    const auto concatType = vpux::getBufferType(concatOp.getOutput());
    if (reshapeType.getRank() != 4 || concatType.getRank() != 4) {
        _log.trace("Only 4D tensors are supported");
        return false;
    }

    const auto concatAxes = VPUIP::getConcatAxes(concatOp);
    if (concatAxes.size() != 1) {
        return false;
    }

    // Concat axis is not changed after propagating GenericReshapeOp
    const auto concatAxis = *concatAxes.begin();
    if (concatAxis != Dims4D::Act::C) {
        return false;
    }

    // [1, A, B*C, 1] -> [1, A, B, C]
    const auto reshapeShape = reshapeType.getShape();
    const auto concatShape = concatType.getShape();
    if (reshapeType.getNumElements() != concatType.getNumElements()) {
        return false;
    }

    if (concatShape[Dims4D::Act::N] != 1 || concatShape[Dims4D::Act::W] != 1) {
        return false;
    }

    if (!(reshapeShape[Dims4D::Act::N] == concatShape[Dims4D::Act::N] &&
          reshapeShape[Dims4D::Act::C] == concatShape[Dims4D::Act::C] &&
          concatShape[Dims4D::Act::H] == reshapeShape[Dims4D::Act::H] * reshapeShape[Dims4D::Act::W])) {
        return false;
    }

    // Make sure H and W are adjacent in memory order
    const auto concatOrder = concatType.getDimsOrder();
    const auto memDimH = concatOrder.toMemDim(Dims4D::Act::H);
    const auto memDimW = concatOrder.toMemDim(Dims4D::Act::W);
    if ((memDimH.ind() != memDimW.ind() + 1) && (memDimW.ind() != memDimH.ind() + 1)) {
        _log.nest().trace("[{0}] ConcatView '{1}' at '{2}' H and W are not adjacent", getDebugName(),
                          concatOp->getName(), concatOp->getLoc());

        return false;
    }

    // Check if Cluster axis and Concat axis are compatible
    const auto outputBuffer = concatOutCopyOp.getOutputBuff();
    const auto outputBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputBuffer.getType());
    if (outputBufferType == nullptr) {
        _log.nest().trace("[{0}] ConcatView '{1}' at '{2}' user distributed copy buffer does not have distributedType",
                          getDebugName(), concatOp->getName(), concatOp->getLoc());
        return false;
    }

    auto tilesAxis = VPUIP::getSpecificAxisFromAttr(outputBufferType.getDistribution().getNumTiles());
    // Unable to obtain the correct buffer address when the clustering axis and the Concat axis are identical
    if (tilesAxis != -1 && concatAxis == Dim(tilesAxis)) {
        return false;
    }

    return true;
}

mlir::FailureOr<ConcatOutputs> OptimizeDDR2DDRCopyInputsOfConcatView::getValidConcatOutputsOfSegmentedCopyUser(
        VPUIP::ConcatViewOp concatViewOp) const {
    if (!concatViewOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    ConcatOutputs validOutput;

    auto copyAfterViewLikeOps = searchCopyOpThroughViewLikeOps(concatViewOp, validOutput.viewLikeOps);
    if (mlir::failed(copyAfterViewLikeOps)) {
        _log.nest().trace("[{0}] Invalid output: no CopyOp after viewlike ops", getDebugName());
        return mlir::failure();
    }

    auto concatAxes = VPUIP::getConcatAxes(concatViewOp);
    if (concatAxes.size() != 1) {
        return mlir::failure();
    }
    const auto concatAxis = *concatAxes.begin();

    const auto isValidSegmentedChildDistributedCopyOp = [&](mlir::Operation* op) {
        auto copyOp = mlir::dyn_cast_if_present<VPUIP::CopyOp>(op);
        if (copyOp == nullptr || !VPUIP::isCopyFromDDR(copyOp) || VPUIP::isCopyToDDR(copyOp)) {
            return false;
        }

        auto tilingCopyOutput = copyOp->getResult(0);
        const auto outputDistributedType =
                mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(tilingCopyOutput));
        VPUX_THROW_UNLESS(outputDistributedType != nullptr, "Cannot get distributedType");

        auto distribution = outputDistributedType.getDistribution();
        // Distributed type with alignment is not supported right now
        auto alignment = distribution.getAlignment();
        if (alignment != nullptr) {
            _log.trace("Not support distribution with alignment");
            return false;
        }

        auto getTilingAxis = getMultiClusterTilingAxis(distribution, _log);
        if (!getTilingAxis.has_value()) {
            _log.trace("Failed to get Multi-Cluster tiling axis");
            return false;
        }
        auto tileIndex = getTilingAxis.value();

        auto tilingDimsForConcat =
                backInferDimAfterChangedByViewLikeOperations(Dim(tileIndex), validOutput.viewLikeOps);
        if (mlir::failed(tilingDimsForConcat)) {
            _log.nest().trace("[{0}] Invalid output: Failed to back infer Multi-Cluster tiling dim for new Concat",
                              getDebugName());
            return false;
        }

        auto tileOverDimForConcat = tilingDimsForConcat.value().front();

        _log.debug(
                "[DEBUG]Original Concat axis is {0}, original multi-cluster segmented dimension is {1}, back infered "
                "multi-cluster segmented dimension for new Concat is {2}, ",
                concatAxis, Dim(tileIndex), tileOverDimForConcat);

        if (tileOverDimForConcat == concatAxis) {
            return false;
        }

        auto isMemoryContinuousOnClusters = [&]() {
            auto dimsOrder = mlir::cast<NDTypeInterface>(concatViewOp.getOutput().getType()).getDimsOrder();
            auto tileOverMemDimForConcat = dimsOrder.toMemDim(tileOverDimForConcat);
            auto concatMemDim = dimsOrder.toMemDim(concatAxis);

            if (concatMemDim.ind() < tileOverMemDimForConcat.ind()) {
                // Case 1: concat dimension is HIGHER than the segmented dimension
                // Ensure copy shape can be evenly split for clusters, otherwise there would be accuracy issues.
                // For example, concatenate [1x1x6x10], [1x1x6x10], and [1x1x6x10] on CMX on Dim C.
                // The output shape [1, 3, 6, 10] is segmented on Dim H with 4 clusters:
                //
                //  1x1x6x10     1x1x6x10    1x1x6x10
                //      \           |           /
                //            ConcatView(CMX)
                //                  |
                //              1x3x6x10
                //                  |
                //
                // Data on CMX would be like below:
                // C0: |----2x10----||----2x10----||----2x10----|
                // C1: |----2x10----||----2x10----||----2x10----|
                // C2: |-1x10-|      |-1x10-|      |-1x10-|
                // C3: |-1x10-|      |-1x10-|      |-1x10-|
                // Data on C2 & C3 are not stored continuously because dim size 6 can't be evenly split for 4 clusters

                // Case 2: concat dimension is LOWER than the segmented dimension
                // For example, concatenate [1x8x1023x64] and [1x8x1x64] on CMX on Dim H.
                // The output shape [1, 8, 1024, 64] is segmented on Dim C with 3 clusters:
                //
                //  1x8x1023x64         1x8x1x64
                //          \           /
                //          ConcatView(CMX)
                //                 |
                //            1x8x1024x64
                //                 |
                //
                // Data on CMX would be like below:
                // C0: |------1023x64------||-1x64-||------1023x64------||-1x64-||------1023x64------||-1x64-|
                // C1: |------1023x64------||-1x64-||------1023x64------||-1x64-||------1023x64------||-1x64-|
                // C2: |------1023x64------||-1x64-||------1023x64------||-1x64-|
                // Data on C2 is stored continuously even though data can't be distributed evenly for this case.

                const auto numClusters = distribution.getNumClusters().getInt();
                auto copyShape = outputDistributedType.getShape();
                if (copyShape[Dim(tileIndex)] % numClusters) {
                    _log.nest().trace("[{0}] Invalid output: Can't evenly split copy shape {1} for {2} clusters",
                                      getDebugName(), copyShape, numClusters);
                    return false;
                }
            }

            return true;
        };

        return isMemoryContinuousOnClusters();
    };

    auto childOp = copyAfterViewLikeOps.value();
    if (!isValidSegmentedChildDistributedCopyOp(childOp)) {
        _log.nest().trace("[{0}] Invalid output: no valid segmented distributed CopyOp", getDebugName());
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

//
// Move unbalanced ConcatView from DDR to CMX
//
// Convert below pattern:
//
//           input           BlockArgument/Constant
//             |                    |
//           CopyOp               CopyOp
//        (DDR -> DDR)          (DDR -> DDR)
//               \                /
//                ConcatView (DDR)
//                        |
//                 GenericReshapeOp
//                        |
//                  DistributedCopy
//                   (DDR -> CMX)
//                        |
//
// to:
//            input           BlockArgument/Constant
//              |                   |
//        GenericReshapeOp   GenericReshapeOp
//              |                   |
//       DistributedCopy      DistributedCopy
//        (DDR -> CMX)          (DDR -> CMX)
//               \                /
//                ConcatView (CMX)
//                     |
//
// So that DDR2DDR copy inputs can be optimized.
//
mlir::FailureOr<ConcatOutputs> OptimizeDDR2DDRCopyInputsOfConcatView::getValidUnbalancedConcat(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs) const {
    if (!concatInputs.inputDistributedCopies.empty() || !concatInputs.inputNonDistributedCopies.empty()) {
        return mlir::failure();
    }

    if (concatInputs.inputCopies.size() != 2) {
        return mlir::failure();
    }

    if (!concatViewOp->hasOneUse()) {
        return mlir::failure();
    }

    //
    // actInput             -> Copy ->
    //                                 ConcatView -> DistCopy
    // argumentOrConstInput -> Copy ->
    //
    mlir::Value argumentOrConstInput;
    mlir::Value actInput;
    for (auto& concatInput : concatInputs.inputCopies) {
        auto inputCopy = concatInput.getDefiningOp<VPUIP::CopyOp>();
        auto copyInput = inputCopy.getInput();

        if (mlir::isa_and_nonnull<VPUIP::SubViewOp>(copyInput.getDefiningOp())) {
            _log.nest().trace("[{0}] Got Subview in inputs", getDebugName());
            return mlir::failure();
        }

        if (mlir::isa<mlir::BlockArgument>(copyInput)) {
            _log.nest().trace("[{0}] Got BlockArgument in inputs", getDebugName());
            argumentOrConstInput = copyInput;
            continue;
        }

        if (mlir::isa_and_nonnull<Const::DeclareOp>(copyInput.getDefiningOp())) {
            _log.nest().trace("[{0}] Got const DeclareOp in inputs", getDebugName());
            argumentOrConstInput = copyInput;
            continue;
        }

        actInput = copyInput;
    }

    if (argumentOrConstInput == nullptr || actInput == nullptr) {
        _log.nest().trace("[{0}] No BlockArgument or constant input", getDebugName());
        return mlir::failure();
    }

    if (getShape(argumentOrConstInput).totalSize() < getShape(actInput).totalSize()) {
        _log.nest().trace("[{0}] It's not the unbalanced Concat", getDebugName());
        return mlir::failure();
    }

    // ConcatView -> GenericReshape -> DistCopy
    auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(*concatViewOp->getUsers().begin());
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
        _log.nest().trace("[{0}] Invalid output: no Reshape after Concat", getDebugName());
        return mlir::failure();
    }

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*reshapeOp->getUsers().begin());
    if (copyOp == nullptr || !VPUIP::isCopyFromDDR(copyOp) || VPUIP::isCopyToDDR(copyOp)) {
        _log.nest().trace("[{0}] Invalid output: no Copy after Reshape", getDebugName());
        return mlir::failure();
    }

    auto outputBuffer = copyOp.getOutputBuff();
    auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outputBuffer);
    if (masterBuffer == nullptr) {
        _log.nest().trace("[{0}] Invalid output: buffer isn't master buffer", getDebugName());
        return mlir::failure();
    }

    auto checkCompatibility = checkConcatReshapeCompatibility(concatViewOp, reshapeOp, copyOp);
    if (!checkCompatibility) {
        _log.nest().trace("[{0}] Shape/Axes compatibility check failed", getDebugName());

        return mlir::failure();
    }

    ConcatOutputs validOutput;
    validOutput.viewLikeOps.push_back(reshapeOp);
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
                appendLoc(subViewOp->getLoc(), "subview_CMX"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

        auto newDistributedCopyOp =
                rewriter.create<VPUIP::CopyOp>(appendLoc(inputCopyOp->getLoc(), "cvt_from_copy_input"),
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

        auto inputType =
                mlir::isa<vpux::VPUIP::DistributedBufferType>(inputCopyType)
                        ? mlir::dyn_cast<vpux::NDTypeInterface>(
                                  mlir::cast<vpux::VPUIP::DistributedBufferType>(inputCopyType).getCompactType())
                        : mlir::dyn_cast<vpux::NDTypeInterface>(inputCopyType);

        auto newDDRType = inputType.changeMemSpace(VPU::MemoryKind::DDR);
        auto newAllocDDROp =
                rewriter.create<mlir::memref::AllocOp>(appendLoc(inputDistributedCopyOp->getLoc(), "new_DDR_buffer"),
                                                       mlir::cast<mlir::MemRefType>(newDDRType));

        auto cmxToDDRDistributedCopyOp = rewriter.create<VPUIP::CopyOp>(
                appendLoc(inputDistributedCopyOp->getLoc(), "CMX_to_DDR_Copy"), inputDistributedCopyOp.getInput(),
                static_cast<mlir::Value>(newAllocDDROp));

        // DDR to CMX
        auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(subViewOp->getLoc(), "subview_CMX"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        auto ddrToCMXDistributedCopyOp = rewriter.create<VPUIP::CopyOp>(
                appendLoc(inputDistributedCopyOp->getLoc(), "DDR_to_CMX_Copy"),
                static_cast<mlir::Value>(cmxToDDRDistributedCopyOp.getResult()), newSubView.getResult());

        // remove old distributed CopyOp
        rewriter.replaceOp(inputDistributedCopyOp, ddrToCMXDistributedCopyOp->getResult(0));

        newConcatInputs.push_back(ddrToCMXDistributedCopyOp.getResult());
    }
}

mlir::Value OptimizeDDR2DDRCopyInputsOfConcatView::rewriteViewLikeOpsDuplicated(
        mlir::Value input, ArrayRef<mlir::Operation*> viewLikeOps, VPUIP::DistributedBufferType origOutputBufferType,
        mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto output = input;
    for (const auto& viewlikeOp : viewLikeOps) {
        if (auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(viewlikeOp)) {
            auto origType = mlir::cast<NDTypeInterface>(reshapeOp.getOutput().getType());
            const auto newType = getDuplicatedDistributedType(origType, origOutputBufferType, ctx);
            auto newReshapeOp = rewriter.create<VPUIP::GenericReshapeOp>(reshapeOp->getLoc(), newType, output);
            output = newReshapeOp.getOutput();
        } else if (auto shapeCastOp = mlir::dyn_cast<VPUIP::ShapeCastOp>(viewlikeOp)) {
            auto newShapeCastOp =
                    rewriter.create<VPUIP::ShapeCastOp>(shapeCastOp->getLoc(), output, shapeCastOp.getShape());
            output = newShapeCastOp.getResult();
        } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(viewlikeOp)) {
            auto origType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());
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

    mlir::Operation* rootOp = concatViewOp.getOperation();
    if (concatViewOp.use_empty()) {
        return mlir::failure();
    }

    auto permuteCastOp = mlir::dyn_cast_if_present<VPUIP::PermuteCastOp>(*concatViewOp->getUsers().begin());
    if (permuteCastOp != nullptr) {
        if (!concatViewOp->hasOneUse()) {
            return mlir::failure();
        }
        rootOp = permuteCastOp.getOperation();
    }

    for (auto* user : rootOp->getUsers()) {
        auto subViewOp = mlir::dyn_cast_if_present<VPUIP::SubViewOp>(user);
        if (subViewOp == nullptr || !subViewOp->hasOneUse()) {
            return mlir::failure();
        }

        auto copyOp = mlir::dyn_cast_if_present<VPUIP::CopyOp>(*user->getUsers().begin());
        auto postPermuteCastOp = mlir::dyn_cast_if_present<VPUIP::PermuteCastOp>(*user->getUsers().begin());
        if (postPermuteCastOp != nullptr) {
            if (!postPermuteCastOp->hasOneUse()) {
                return mlir::failure();
            }
            copyOp = mlir::dyn_cast_if_present<VPUIP::CopyOp>(*postPermuteCastOp->getUsers().begin());
        }

        if (copyOp == nullptr || !copyOp->hasOneUse()) {
            return mlir::failure();
        }

        const auto isDDR2CMX = VPUIP::isCopyFromDDR(copyOp) && !VPUIP::isCopyToDDR(copyOp);
        if (!isDDR2CMX) {
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
    outputs.outputCopyOps = std::move(copyOps);
    return outputs;
}

// Check ConcatView output chain.
// We expect ConcatView is followed by a PermuteCast, or multiple SubView users followed by valid DDR-to-CMX copy
// branches (DUPLICATED DistributedCopy or single-tile non-distributed copy).
// Like in below:
/*
                    ConcatView
                        |
                  [PermuteCastOp]
            /           |           \
    SubView         SubView         SubView
        |               |               |
 [PermuteCastOp]  [PermuteCastOp]  [PermuteCastOp]
        |               |               |
      Copy            Copy            Copy
        |               |               |
*/
// Return ConcatOutputsOfSubViewCopyUsers struct if pattern can match, otherwise return mlir::failure().
mlir::FailureOr<ConcatOutputsOfSubViewCopyUsers>
OptimizeDDR2DDRCopyInputsOfConcatView::getValidConcatOutputsOfSubViewCopyUsers(VPUIP::ConcatViewOp concatViewOp) const {
    auto nestedLog = _log.nest();
    auto outputs = searchSubViewCopyUsersThroughPermuteCast(concatViewOp);
    if (mlir::failed(outputs)) {
        nestedLog.trace("[{0}] Invalid output: can't find SubView and Copy users after PermuteCast", getDebugName());
        return mlir::failure();
    }

    const auto concatAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(concatViewOp.getOperands()[0]), getShape(concatViewOp.getResult()));
    if (concatAxes.empty() || concatAxes.size() != 1) {
        nestedLog.trace("[{0}] Only support concat on single dimension", getDebugName());
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
            nestedLog.trace("[{0}] Only support SubView on single dimension", getDebugName());
            return mlir::failure();
        }

        if (subViewAxes.front() != firstSubViewAxes.front()) {
            nestedLog.trace("[{0}] All SubView axes should be the same", getDebugName());
            return mlir::failure();
        }

        if (subViewAxes.front() == concatAxes.front()) {
            nestedLog.trace("[{0}] SubView axis should be different with ConcatView axis", getDebugName());
            return mlir::failure();
        }

        auto childPermuteCastOp = mlir::dyn_cast_if_present<VPUIP::PermuteCastOp>(*subViewOp->getUsers().begin());
        // Currently only support PermuteCastOp that does not change the logical shape
        if (childPermuteCastOp != nullptr &&
            getShape(childPermuteCastOp.getSource()) != getShape(childPermuteCastOp.getResult())) {
            return mlir::failure();
        }
    }

    // Check if CopyOp is a valid DDR to CMX copy:
    // 1. Distributed copy with DUPLICATED distribution, or
    // 2. Single-tile copy to CMX memref
    const auto isValidDDR2CMXCopyOp = [](mlir::Operation* op) {
        auto copyOp = mlir::dyn_cast_if_present<VPUIP::CopyOp>(op);
        if (copyOp == nullptr || !VPUIP::isCopyFromDDR(copyOp) || VPUIP::isCopyToDDR(copyOp)) {
            return false;
        }

        auto tilingCopyOutput = copyOp->getResult(0);
        const auto outputDistributedType =
                mlir::dyn_cast<VPUIP::DistributedBufferType>(VPUIP::extractDataType(tilingCopyOutput));

        if (outputDistributedType == nullptr) {
            auto outputType = mlir::cast<vpux::NDTypeInterface>(tilingCopyOutput.getType());
            return outputType.getMemoryKind() == vpux::VPU::MemoryKind::CMX_NN;
        }

        auto distribution = outputDistributedType.getDistribution();
        return VPU::isDuplicated(distribution);
    };

    for (auto copyOp : outputs.value().outputCopyOps) {
        if (!isValidDDR2CMXCopyOp(copyOp)) {
            nestedLog.trace("[{0}] Invalid output: no valid DDR to CMX CopyOp", getDebugName());
            return mlir::failure();
        }

        auto outputBuffer = copyOp.getOutputBuff();
        auto distributedMasterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outputBuffer);
        auto memrefMasterBuffer = VPUIP::getRootAlloc<mlir::memref::AllocOp>(outputBuffer);
        if (distributedMasterBuffer == nullptr && memrefMasterBuffer == nullptr) {
            nestedLog.trace("[{0}] Invalid output: buffer isn't master buffer", getDebugName());
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
        [PermuteCastOp]                     [PermuteCastOp]                     [PermuteCastOp]
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
        [PermuteCastOp (will be merged into the previous one)]
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
        [PermuteCastOp (will be merged into the previous one)]
                                |
                            NCE TaskN

*/
mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::processConcatOutputsOfSubViewCopyUsers(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs,
        const ConcatOutputsOfSubViewCopyUsers& concatOutputs, mlir::PatternRewriter& rewriter) const {
    if (concatInputs.inputCopies.empty() || !concatInputs.inputDistributedCopies.empty() ||
        !concatInputs.inputNonDistributedCopies.empty()) {
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
        auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subViewOp->getUsers().begin());
        auto childPermuteCastOp = mlir::dyn_cast_if_present<VPUIP::PermuteCastOp>(*subViewOp->getUsers().begin());
        if (childPermuteCastOp != nullptr) {
            childCopyOp = mlir::dyn_cast_if_present<VPUIP::CopyOp>(*childPermuteCastOp->getUsers().begin());
        }
        VPUX_THROW_WHEN(childCopyOp == nullptr, "Can't find CopyOp user");

        auto outputBuffer = childCopyOp.getOutputBuff();
        const auto outputDistributedBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputBuffer.getType());
        const auto outputNDType = mlir::cast<NDTypeInterface>(outputBuffer.getType());
        const bool isDistributed = (outputDistributedBufferType != nullptr);

        const auto subViewAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(subViewOp.getSource()), getShape(subViewOp.getResult()));
        const auto subViewAxis = subViewAxes.front();

        Shape newConcatShape = origConcatShape.raw();
        newConcatShape[subViewAxis] = getShape(subViewOp.getResult())[subViewAxis];
        auto newConcatType =
                mlir::cast<NDTypeInterface>(concatViewOp.getOutput().getType()).changeShape(newConcatShape);

        mlir::Type newConcatBufferType;
        if (isDistributed) {
            newConcatBufferType = getDuplicatedDistributedType(newConcatType, outputDistributedBufferType, ctx);
        } else {
            newConcatBufferType = newConcatType.changeMemSpace(outputNDType.getMemSpace());
        }

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
                    appendLoc(subViewOp->getLoc(), "src_subview"), inputCopyOp.getInput(),
                    getIntArrayAttr(ctx, srcSubViewOffsets), getIntArrayAttr(ctx, srcSubViewSizes));

            auto dstSubViewOffsets = SmallVector<int64_t>(srcSubViewSizes.size(), 0);
            auto dstSubViewSizes = parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes());
            dstSubViewOffsets[concatAxis.ind()] = currentOutOffset;
            dstSubViewSizes[concatAxis.ind()] = getShape(inputCopyOp.getInput())[concatAxis];
            currentOutOffset += dstSubViewSizes[concatAxis.ind()];
            auto newDstSubView = rewriter.create<VPUIP::SubViewOp>(
                    appendLoc(subViewOp->getLoc(), "dst_subview"), outputBuffer,
                    getIntArrayAttr(ctx, dstSubViewOffsets), getIntArrayAttr(ctx, dstSubViewSizes),
                    subViewOp.getStaticStridesAttr());

            auto newCopyOp = rewriter.create<VPUIP::CopyOp>(appendLoc(inputCopyOp->getLoc(), "copy_to_cmx"),
                                                            newSrcSubView.getResult(), newDstSubView.getResult());

            newConcatInputs.push_back(newCopyOp.getResult());
        }

        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(concatViewOp->getLoc(), newConcatInputs, outputBuffer);
        auto lastValue = newConcatOp.getOutput();

        // [PermuteCast0] -> Subview -> [PermuteCast1] -> Copy
        // becomes
        // Subview -> Copy -> ConcatView -> [PermuteCast0] -> [PermuteCast1]
        // which can be further simplified to
        // Subview -> Copy -> ConcatView -> [merged PermuteCast0 + PermuteCast1]
        // if both PermuteCast ops are present in the original pattern
        if (permuteCastOp != nullptr || childPermuteCastOp != nullptr) {
            auto lastPermCastType = childPermuteCastOp != nullptr
                                            ? mlir::cast<NDTypeInterface>(childPermuteCastOp.getResult().getType())
                                            : mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());
            auto newPermuteCastShape = Shape(lastPermCastType.getShape());
            newPermuteCastShape[subViewAxis] = getShape(subViewOp.getResult())[subViewAxis];

            mlir::Type newType;
            if (isDistributed) {
                newType = getDuplicatedDistributedType(lastPermCastType.changeShape(newPermuteCastShape),
                                                       outputDistributedBufferType, ctx);
            } else {
                newType = lastPermCastType.changeShape(newPermuteCastShape).changeMemSpace(outputNDType.getMemSpace());
            }
            const auto loc = permuteCastOp != nullptr ? permuteCastOp->getLoc() : childPermuteCastOp->getLoc();

            auto memPerm = childPermuteCastOp != nullptr ? childPermuteCastOp.getMemPermAttr().getAffineMap()
                                                         : permuteCastOp.getMemPermAttr().getAffineMap();
            if (permuteCastOp != nullptr && childPermuteCastOp != nullptr) {
                memPerm = memPerm.compose(permuteCastOp.getMemPermAttr().getAffineMap());
            }

            auto newPermuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(
                    loc, newType, newConcatOp.getOutput(),
                    mlir::cast<NDTypeInterface>(newType).getDimsOrder().toAffineMap(ctx), memPerm);
            lastValue = newPermuteCastOp.getResult();
        }

        if (isDistributed) {
            auto distributedCastOp = rewriter.createOrFold<VPUIP::DistributedCastOp>(
                    childCopyOp->getLoc(), outputDistributedBufferType, lastValue);
            rewriter.replaceOp(childCopyOp, distributedCastOp);
        } else {
            rewriter.replaceOp(childCopyOp, lastValue);
        }
    }

    // Remove old operations
    for (auto subViewOp : llvm::make_early_inc_range(concatOutputs.outputSubViewOps)) {
        if (!subViewOp->use_empty()) {
            if (auto childPermuteCastOp =
                        mlir::dyn_cast_if_present<VPUIP::PermuteCastOp>(*subViewOp->getUsers().begin())) {
                rewriter.eraseOp(childPermuteCastOp);
            }
        }

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

mlir::Value OptimizeDDR2DDRCopyInputsOfConcatView::rewriteViewLikeOpsSegmented(
        mlir::Value input, ArrayRef<Dim> tilingDims, ArrayRef<mlir::Operation*> viewLikeOps,
        VPUIP::DistributedBufferType origOutputBufferType, mlir::PatternRewriter& rewriter) const {
    if (viewLikeOps.empty()) {
        return input;
    }

    auto ctx = rewriter.getContext();
    auto origDistribution = origOutputBufferType.getDistribution();

    auto output = input;
    for (const auto& [viewlikeOp, tilingDim] : zip(viewLikeOps, tilingDims)) {
        if (auto shapeCastOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(viewlikeOp)) {
            auto origType = mlir::cast<NDTypeInterface>(viewlikeOp->getResult(0).getType());
            auto newType = getSegmentedDistributedType(ctx, origType, tilingDim.ind(), origDistribution);
            output = rewriter.create<VPUIP::GenericReshapeOp>(viewlikeOp->getLoc(), newType, output).getOutput();
        } else if (auto shapeCastOp = mlir::dyn_cast<VPUIP::ShapeCastOp>(viewlikeOp)) {
            output = rewriter.create<VPUIP::ShapeCastOp>(viewlikeOp->getLoc(), output, shapeCastOp.getShape())
                             .getResult();

        } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(viewlikeOp)) {
            auto origType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());
            auto newType = getSegmentedDistributedType(ctx, origType, tilingDim.ind(), origDistribution);
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

mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::processConcatOutputsOfDuplicatedCopyUser(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs, const ConcatOutputs& concatOutputs,
        mlir::PatternRewriter& rewriter) const {
    auto childDistributedCopyOp = concatOutputs.outputDistributedCopy;
    auto outputBuffer = childDistributedCopyOp.getOutputBuff();
    const auto outputBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputBuffer.getType());
    auto nestedLog = _log.nest();
    if (outputBufferType == nullptr) {
        nestedLog.trace("[{0}] ConcatView '{1}' at '{2}' user distributed copy buffer does not have distributedType",
                        getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());
        return mlir::failure();
    }

    nestedLog.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());

    // Create new subgraph to move ConcatView and viewlike ops to CMX
    auto ctx = rewriter.getContext();

    auto origConcatNDType = mlir::cast<NDTypeInterface>(concatViewOp.getOutput().getType());
    auto newConcatBufferType = getDuplicatedDistributedType(origConcatNDType, outputBufferType, ctx);
    // update buffer type so that new ConcatView can re-use this buffer on CMX
    outputBuffer.setType(newConcatBufferType);

    SmallVector<mlir::Value> newConcatInputs;
    rewriter.setInsertionPointAfter(outputBuffer.getDefiningOp());

    convertCopyInputAndStore(concatInputs.inputCopies, outputBuffer, newConcatInputs, rewriter);
    SmallVector<mlir::Value> inputCMXCopies;
    for (auto input : concatInputs.inputDistributedCopies) {
        inputCMXCopies.push_back(input);
    }
    for (auto input : concatInputs.inputNonDistributedCopies) {
        inputCMXCopies.push_back(input);
    }

    convertDistributedCopyInputAndStore(inputCMXCopies, outputBuffer, newConcatInputs, rewriter);
    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(concatViewOp->getLoc(), newConcatInputs, outputBuffer);

    auto subGraphOutput = rewriteViewLikeOpsDuplicated(newConcatOp.getOutput(), concatOutputs.viewLikeOps,
                                                       outputBufferType, rewriter);

    // cast to original outputBufferType because alignment in distribution might be different
    auto distributedCastOp = rewriter.createOrFold<VPUIP::DistributedCastOp>(childDistributedCopyOp->getLoc(),
                                                                             outputBufferType, subGraphOutput);

    rewriter.replaceOp(childDistributedCopyOp, distributedCastOp);

    return mlir::success();
}

mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::processConcatOutputsOfSegmentedCopyUser(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs, const ConcatOutputs& concatOutputs,
        mlir::PatternRewriter& rewriter) const {
    auto nestedLog = _log.nest();
    if (!concatInputs.inputNonDistributedCopies.empty()) {
        nestedLog.trace("[{0}] Only support DDR2DDR/DistributedCopy inputs", getDebugName());
        return mlir::failure();
    }

    auto concatOutElemType = mlir::cast<vpux::NDTypeInterface>(concatViewOp.getOutput().getType()).getElementType();
    if (vpux::isSubByteType(concatOutElemType)) {
        nestedLog.trace("[{0}] Sub-byte types are not supported", getDebugName());
        return mlir::failure();
    }

    auto childDistributedCopyOp = concatOutputs.outputDistributedCopy;
    auto outputBuffer = childDistributedCopyOp.getOutputBuff();
    const auto outputBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputBuffer.getType());
    if (outputBufferType == nullptr) {
        nestedLog.trace("[{0}] ConcatView '{1}' at '{2}' user distributed copy buffer does not have distributedType",
                        getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());
        return mlir::failure();
    }

    nestedLog.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());
    // Create new subgraph to move ConcatView and viewlike ops to CMX
    auto ctx = rewriter.getContext();

    auto origConcatNDType = mlir::cast<NDTypeInterface>(concatViewOp.getOutput().getType());
    auto origDistribution = outputBufferType.getDistribution();
    auto getTilingAxis = getMultiClusterTilingAxis(origDistribution, _log);
    if (!getTilingAxis.has_value()) {
        return mlir::failure();
    }
    auto tileIndex = getTilingAxis.value();

    auto tileOverDimForConcat = backInferDimAfterChangedByViewLikeOperations(Dim(tileIndex), concatOutputs.viewLikeOps);
    if (mlir::failed(tileOverDimForConcat)) {
        nestedLog.trace("[{0}] backInferDimAfterChangedByViewLikeOperations failed", getDebugName());
        return mlir::failure();
    }

    auto tilingDims = tileOverDimForConcat.value();
    if (tilingDims.size() != (concatOutputs.viewLikeOps.size() + 1)) {
        nestedLog.trace("[{0}] Tiling dimensions size mismatch", getDebugName());
        return mlir::failure();
    }

    auto newConcatTileIndex = tilingDims.front().ind();
    tilingDims.erase(tilingDims.begin());
    auto newConcatBufferType = getSegmentedDistributedType(ctx, origConcatNDType, newConcatTileIndex, origDistribution);

    // update buffer type so that new ConcatView can re-use this buffer on CMX
    outputBuffer.setType(newConcatBufferType);

    SmallVector<mlir::Value> newConcatInputs;
    rewriter.setInsertionPointAfter(outputBuffer.getDefiningOp());

    convertCopyInputAndStore(concatInputs.inputCopies, outputBuffer, newConcatInputs, rewriter);
    convertDistributedCopyInputAndStore(concatInputs.inputDistributedCopies, outputBuffer, newConcatInputs, rewriter);

    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(concatViewOp->getLoc(), newConcatInputs, outputBuffer);

    auto viewLikeSubGraph = rewriteViewLikeOpsSegmented(newConcatOp.getOutput(), std::move(tilingDims),
                                                        concatOutputs.viewLikeOps, outputBufferType, rewriter);

    rewriter.replaceOp(childDistributedCopyOp, viewLikeSubGraph);
    return mlir::success();
}

mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::processUnbalancedConcat(
        VPUIP::ConcatViewOp concatViewOp, const ConcatInputs& concatInputs, const ConcatOutputs& concatOutputs,
        mlir::PatternRewriter& rewriter) const {
    auto nestedLog = _log.nest();
    _log.trace("[{0}] Got '{1}' at '{2}' in processUnbalancedConcat", getDebugName(), concatViewOp->getName(),
               concatViewOp->getLoc());

    auto outDistributedCopyOp = concatOutputs.outputDistributedCopy;
    auto outputBuffer = outDistributedCopyOp.getOutputBuff();

    // Create new subgraph to move ConcatView to CMX
    auto reshapeOp = mlir::cast<VPUIP::GenericReshapeOp>(concatOutputs.viewLikeOps.front());
    auto reshapeOutShape = getShape(reshapeOp.getOutput());

    const auto rewriteReshapeOp = [&](mlir::Value input) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        auto newShape = inputType.getShape().toValues();
        newShape[Dims4D::Act::H] = reshapeOutShape[Dims4D::Act::H];
        newShape[Dims4D::Act::W] = reshapeOutShape[Dims4D::Act::W];
        auto newOutType = inputType.changeShape(newShape);
        return rewriter.create<VPUIP::GenericReshapeOp>(reshapeOp->getLoc(), newOutType, input).getOutput();
    };

    SmallVector<mlir::Value> newConcatInputs;
    rewriter.setInsertionPointAfter(outputBuffer.getDefiningOp());

    for (const auto& copyInput : concatInputs.inputCopies) {
        auto inputCopyOp = copyInput.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = inputCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subViewOp == nullptr, "Can't find SubViewOp");

        auto newReshapeOutput = rewriteReshapeOp(inputCopyOp.getInput());
        nestedLog.trace("view shape {0}", getShape(newReshapeOutput));

        auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(subViewOp->getLoc(), "subview_cmx"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                getIntArrayAttr(rewriter.getContext(), getShape(newReshapeOutput).toValues()),
                subViewOp.getStaticStridesAttr());

        auto newDistributedCopyOp = rewriter.create<VPUIP::CopyOp>(
                appendLoc(inputCopyOp->getLoc(), "cvt_from_copy_input"), newReshapeOutput, newSubView.getResult());

        rewriter.replaceOp(inputCopyOp, newDistributedCopyOp->getResult(0));
        newConcatInputs.push_back(newDistributedCopyOp.getResult());
    }

    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(appendLoc(concatViewOp->getLoc(), "cmx"), newConcatInputs,
                                                            outputBuffer);
    rewriter.replaceOp(outDistributedCopyOp, newConcatOp.getOutput());

    nestedLog.trace("processUnbalancedConcat Done");

    return mlir::success();
}

mlir::LogicalResult OptimizeDDR2DDRCopyInputsOfConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto nestedLog = _log.nest();
    nestedLog.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());
    auto concatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.getOutputBuff());
    if (concatMemAlloc == nullptr) {
        nestedLog.trace("[{0}] Cannot rewrite because current concat '{1}' output isn't master buffer", getDebugName(),
                        concatViewOp->getLoc());
        return mlir::failure();
    }

    // Check inputs of ConcatView
    auto checkInputs = getValidConcatInputs(concatViewOp);
    if (mlir::failed(checkInputs)) {
        nestedLog.trace("[{0}] Invalid inputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                        concatViewOp->getLoc());
        return mlir::failure();
    }

    const auto& concatInputs = checkInputs.value();

    // Check output of ConcatView and process corresponding pattern
    auto checkOutputsOfDuplicatedCopyUser = getValidConcatOutputsOfDuplicatedCopyUser(concatViewOp, concatInputs);
    if (mlir::succeeded(checkOutputsOfDuplicatedCopyUser)) {
        nestedLog.trace("[{0}] Got '{1}' at '{2}' with DUPLICATED Copy Users", getDebugName(), concatViewOp->getName(),
                        concatViewOp->getLoc());
        ConcatOutputs& concatOutputs = checkOutputsOfDuplicatedCopyUser.value();
        return processConcatOutputsOfDuplicatedCopyUser(concatViewOp, concatInputs, concatOutputs, rewriter);
    }

    auto checkOutputsOfSegmentedCopyUser = getValidConcatOutputsOfSegmentedCopyUser(concatViewOp);
    if (mlir::succeeded(checkOutputsOfSegmentedCopyUser)) {
        nestedLog.trace("[{0}] Got '{1}' at '{2}' with SEGMENTED Copy Users", getDebugName(), concatViewOp->getName(),
                        concatViewOp->getLoc());
        ConcatOutputs& concatOutputs = checkOutputsOfSegmentedCopyUser.value();
        return processConcatOutputsOfSegmentedCopyUser(concatViewOp, concatInputs, concatOutputs, rewriter);
    }

    auto checkOutputsOfSubViewCopyUsers = getValidConcatOutputsOfSubViewCopyUsers(concatViewOp);
    if (mlir::succeeded(checkOutputsOfSubViewCopyUsers)) {
        nestedLog.trace("[{0}] Got '{1}' at '{2}' with SubView and Copy Users", getDebugName(), concatViewOp->getName(),
                        concatViewOp->getLoc());
        ConcatOutputsOfSubViewCopyUsers& concatOutputs = checkOutputsOfSubViewCopyUsers.value();
        return processConcatOutputsOfSubViewCopyUsers(concatViewOp, concatInputs, concatOutputs, rewriter);
    }

    auto checkUnbalancedConcat = getValidUnbalancedConcat(concatViewOp, concatInputs);
    if (mlir::succeeded(checkUnbalancedConcat)) {
        nestedLog.trace("[{0}] Got '{1}' at '{2}' with Reshape Users", getDebugName(), concatViewOp->getName(),
                        concatViewOp->getLoc());
        ConcatOutputs& concatOutputs = checkUnbalancedConcat.value();
        return processUnbalancedConcat(concatViewOp, concatInputs, concatOutputs, rewriter);
    }

    return mlir::failure();
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

bool isSharedBufferOfInplaceEltwise(VPUIP::CopyOp copyOp) {
    for (auto user : copyOp->getResult(0).getUsers()) {
        auto clusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
        if (clusterTaskOp == nullptr) {
            continue;
        }
        if (clusterTaskOp.getTaskType() != VPUIP::NCETaskType::ELTWISE ||
            !clusterTaskOp.getIsInplace().value_or(false)) {
            continue;
        }

        // For in-place Eltwise op, check if the current copy op's buffer is shared by input and output
        auto outputRootBuf = VPUIP::getLayerOutputs(clusterTaskOp)[0];
        auto val = copyOp->getResult(0);
        vpux::ValueSourceInfo aliasInfo(val);
        auto inputRootBuf = *aliasInfo.getRoots(val).begin();
        if (outputRootBuf == inputRootBuf) {
            return true;
        }
    }
    return false;
}

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
    _log.trace("OptimizeConcatSubviewPattern: Got Concat at '{0}'", concatOp.getLoc());
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
        auto subview = mlir::dyn_cast_if_present<VPUIP::SubViewOp>(user);
        if (subview == nullptr || !subview->hasOneUse()) {
            return mlir::failure();
        }

        auto tilingCopy = mlir::dyn_cast_if_present<VPUIP::CopyOp>(*(subview->getUsers().begin()));
        if (tilingCopy == nullptr || !vpux::VPUIP::hasDistributedOperand(tilingCopy)) {
            return mlir::failure();
        }
        // Can't optimize in-place Eltwise's shared input copy
        // otherwise the Eltwise won't be in-place anymore, and the op will exceed CMX memory
        if (isSharedBufferOfInplaceEltwise(tilingCopy)) {
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
        auto inputDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputValue.getType());
        if (inputDistributedType == nullptr) {
            return false;
        }

        auto outputValue = outCopy->getResult(0);
        auto outputDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputValue.getType());
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
        int64_t leftViewOffset;  // Offset from beginning if was viewed
        int64_t rightInputSize;
        Dim origConcatDim;
        Dim newConcatDim;  // Concat Dim after GenericReshape + PermuteCast
        VPUIP::PermuteCastOp castOp;
    };

public:
    SplitUnbalancedDDRConcatBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

private:
    // Suffix of child rewriter
    virtual StringRef getRewriterSuffix() const = 0;

    // Input of the right side and copy operation
    virtual std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const = 0;

    virtual mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput,
                                           VPUIP::GenericReshapeOp genReshape, VPUIP::PermuteCastOp permuteCastOp,
                                           mlir::Location loc) const = 0;

    virtual mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp origView,
                                              VPUIP::CopyOp distributedCopy, mlir::Location bufferLoc) const = 0;

    virtual void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp,
                                VPUIP::SubViewOp subViewOp, VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch,
                                mlir::Value newRightBranch, const PatternParamsInfo& params, size_t index) const = 0;

    virtual VPUIP::DistributedBufferType updateDistributedType(mlir::Value dst, mlir::Value dstView,
                                                               ShapeRef copyShape) const = 0;

    // Check compatibility of concat in CMX. Since consumers are segmented, we must ensure that we can do split.
    // If view/concat axis is the same with tiling axis, we can't do concatenation in same buffer.
    // To avoid it we must do concat in temporary buffer and then distribute
    virtual bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const {
        return newConcatDim.ind() == tilingDim;
    }

    virtual bool isValidSegment(SmallVector<VPUIP::SubViewOp>& views, SmallVector<VPUIP::CopyOp>& distributedCopies,
                                Dim newConcatDim, int64_t leftConcatInputSize, int64_t rightConcatInputSize) const {
        VPUX_UNUSED(views);
        VPUX_UNUSED(distributedCopies);
        VPUX_UNUSED(newConcatDim);
        VPUX_UNUSED(leftConcatInputSize);
        VPUX_UNUSED(rightConcatInputSize);
        return true;
    }

    virtual bool isValidSubview(SmallVector<VPUIP::SubViewOp>& subviews, Dim newConcatDim, int64_t leftConcatInputSize,
                                int64_t rightConcatInputSize) const {
        return checkSubview(subviews, newConcatDim, leftConcatInputSize, rightConcatInputSize);
    }

    virtual bool isOutputDistributedCMX() const {
        return true;
    }

protected:
    bool checkSubview(SmallVector<VPUIP::SubViewOp>& subviews, Dim newConcatDim, int64_t leftConcatInputSize,
                      int64_t rightConcatInputSize) const {
        auto totalSize = leftConcatInputSize + rightConcatInputSize;
        for (auto subview : subviews) {
            auto outShape = getShape(subview.getResult());
            auto offset = Shape(parseIntArrayAttr<int64_t>(subview.getStaticOffsets()));
            // We could support the following case:
            //    0 -- 8319      1
            //          \      /
            //        Concat(0 -- 8320)
            //         |            |
            //     Subview         Subview
            //         |            |
            //     0 -- 4160     4160 - 8320

            // But if subview from 4160 to 8321, which actually need to concat
            // left(4160 -- 8319) + right(8320) + left(8320 -- 8321), the case is too complex,
            // we can not support it now
            auto block = offset[newConcatDim] / totalSize;
            if ((offset[newConcatDim] + outShape[newConcatDim] - 1) / totalSize != block) {
                return false;
            }

            if (offset[newConcatDim] % totalSize >= leftConcatInputSize) {
                // Only support 2 cases for now:
                // 1. subview on the left branch
                // 2. subview crosses the left and right branch
                return false;
            }
        }

        return true;
    }

    // Propagate reshape and cast through concat to split large DDR->DDR DMAs on sequence of small DDR->CMX, which can
    // be interleaved with NCE tasks
    mlir::Value propagateReshapeCast(mlir::PatternRewriter& rewriter, mlir::Value branchInput,
                                     VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc,
                                     StringRef locSuffix) const {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(branchInput.getType());
        auto origShape = inputType.getShape();
        auto permuteCastInputType = mlir::cast<vpux::NDTypeInterface>(permuteCastOp.getSource().getType());
        auto fourDim = permuteCastInputType.getRank() == 4;

        Shape newShape = fourDim ? origShape.toValues() : permuteCastInputType.getShape().toValues();
        if (fourDim) {
            newShape[Dim(0)] = origShape[Dim(1)] * origShape[Dim(2)];
            newShape[Dim(1)] = origShape[Dim(3)];
            newShape[Dim(2)] = 1;
            newShape[Dim(3)] = 1;
        } else {
            newShape = Shape{origShape[Dim(1)], origShape[Dim(2)], origShape[Dim(3)], 1, 1};
        }

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

            auto align = distribution.getAlignment();
            SmallVector<int64_t> newAlign(inputType.getShape().size(), 1);
            newAlign[0] = origShape[Dim(2)];
            if (align) {
                auto origAlign = parseIntArrayAttr<int64_t>(distribution.getAlignment());
                newAlign[0] = std::lcm(std::lcm(newAlign[0], origAlign[1]), origAlign[2]);
                newAlign[1] = origAlign[3];
            }

            auto ctx = rewriter.getContext();

            auto mode = distribution.getMode();
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    newShape, mode, getIntArrayAttr(ctx, newNumTiles), distribution.getNumClusters(),
                    getIntArrayAttr(ctx, newAlign), distribution.getUniformDistributedSegments(), ctx);

            afterReshapeType = VPUIP::DistributedBufferType::get(
                    ctx, newShape.raw(), distributedBufferType.getElementType(), distributedBufferType.getLayout(),
                    distributedBufferType.getMemSpace(), newDistribution);
        } else {
            afterReshapeType = inputType.changeShape(newShape);
        }

        // Check if branchInput has non-compact strides (e.g., from SubView)
        mlir::Value reshapeOutput = branchInput;
        const auto inReqs = StrideReqs::compact(inputType.getRank());
        const bool hasNonCompactStrides = !inReqs.checkStrides(inputType);
        if (hasNonCompactStrides) {
            const auto strideUpdatedOutType = VPUIP::updateStridesForReshape(inputType, afterReshapeType);
            VPUX_THROW_WHEN(mlir::failed(strideUpdatedOutType),
                            "Failed to update strides for input '{0}' and output '{1}'", inputType, afterReshapeType);
            afterReshapeType = strideUpdatedOutType.value();
        }

        reshapeOutput = rewriter.createOrFold<VPUIP::GenericReshapeOp>(appendLoc(loc, "reshape_{0}", locSuffix),
                                                                       afterReshapeType, branchInput);

        // Create PermuteCastOp on top of the reshape operation
        auto permCastDimsOrder = mlir::cast<vpux::NDTypeInterface>(permuteCastOp->getResultTypes()[0]).getDimsOrder();
        auto afterPermCastType = afterReshapeType.changeDimsOrder(std::move(permCastDimsOrder));
        return rewriter.create<VPUIP::PermuteCastOp>(appendLoc(loc, "permcast_{0}", locSuffix), afterPermCastType,
                                                     reshapeOutput, permuteCastOp.getDstOrderAttr(),
                                                     permuteCastOp.getMemPermAttr());
    }

    // Propagate PermuteCast through concat to split large DDR->DDR DMAs on sequence of small DDR->CMX, which can be
    // interleaved with NCE tasks
    mlir::Value propagatePermuteCast(mlir::PatternRewriter& rewriter, mlir::Value branchInput,
                                     VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc,
                                     StringRef locSuffix) const {
        auto branchInputType = mlir::cast<NDTypeInterface>(branchInput.getType());
        auto permCastDimsOrder =
                mlir::cast<vpux::NDTypeInterface>(permuteCastOp->getResult(0).getType()).getDimsOrder();

        NDTypeInterface outType = nullptr;
        if (auto distributedBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(branchInputType)) {
            const auto origMemShape = branchInputType.getMemShape();
            const auto permutedMemShape = applyPerm(origMemShape, permuteCastOp.getMemPerm());
            const auto permutedShape = permCastDimsOrder.toLogicalOrder(permutedMemShape);

            auto distribution = distributedBufferType.getDistribution();

            auto ctx = rewriter.getContext();

            const auto distrInfo = VPU::DistributionInfo::getClassFromAttr(distribution);
            auto newDistribution = VPU::applyPermutationOnDistributionInfo(
                    branchInputType, distrInfo, permuteCastOp.getMemPerm(), branchInputType.getDimsOrder(),
                    permCastDimsOrder, branchInputType.getShape(), permutedShape);

            VPUX_THROW_WHEN(mlir::failed(newDistribution), "Failed to get distribution for PermuteCast Op");

            outType = VPUIP::DistributedBufferType::get(
                    ctx, permutedShape.raw(), distributedBufferType.getElementType(),
                    mlir::AffineMapAttr::get(permCastDimsOrder.toAffineMap(ctx)), distributedBufferType.getMemSpace(),
                    VPU::DistributionInfo::getAttrFromClass(ctx, newDistribution.value()));
        } else {
            outType = inferNewTypeWithMemPerm(branchInputType, permuteCastOp.getMemPerm(),
                                              DimsOrder::fromAffineMap(permuteCastOp.getDstOrder()));
        }

        return rewriter.create<VPUIP::PermuteCastOp>(appendLoc(loc, "permcast_{0}", locSuffix), outType, branchInput,
                                                     permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());
    }

    mlir::Value createNewCopyBranch(mlir::PatternRewriter& rewriter, mlir::Value src, mlir::Value dst,
                                    ShapeRef copyShape, ShapeRef srcOffset, ShapeRef dstOffset, mlir::Location baseLoc,
                                    StringRef locSuffix, size_t opId, bool updateDistributionType = true) const {
        mlir::Value srcView = rewriter.createOrFold<VPUIP::SubViewOp>(
                appendLoc(baseLoc, "{0}_src_view_{1}", locSuffix, opId), src, srcOffset, copyShape);
        mlir::Value dstView = rewriter.createOrFold<VPUIP::SubViewOp>(
                appendLoc(baseLoc, "{0}_dst_view_{1}", locSuffix, opId), dst, dstOffset, copyShape);

        if (updateDistributionType) {
            auto newDstDistributedType = updateDistributedType(dst, dstView, copyShape);
            if (newDstDistributedType != nullptr) {
                dstView.setType(newDstDistributedType);
            }
        }

        auto rightSrcType = mlir::cast<vpux::NDTypeInterface>(srcView.getType());
        const auto rightSrcElementType = rightSrcType.getElementType();
        auto rightDstType = mlir::cast<vpux::NDTypeInterface>(dstView.getType());
        const auto rightDstElementType = rightDstType.getElementType();
        if (rightSrcElementType != rightDstElementType) {
            // Can't transfer data CMX2CMX directly
            if (mlir::isa<VPUIP::DistributedBufferType>(srcView.getType()) &&
                mlir::isa<VPUIP::DistributedBufferType>(dstView.getType())) {
                // Dst.size is usually less than src.size, due to the conversion fp32->fp16/bf16
                auto newDDRType =
                        mlir::dyn_cast<vpux::NDTypeInterface>(
                                mlir::cast<vpux::VPUIP::DistributedBufferType>(dstView.getType()).getCompactType())
                                .changeMemSpace(VPU::MemoryKind::DDR);
                auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(appendLoc(baseLoc, "new_DDR_buffer"),
                                                                            mlir::cast<mlir::MemRefType>(newDDRType));
                auto convertDMAOp = rewriter.create<VPUIP::ConvertDMAOp>(
                        appendLoc(baseLoc, "{0}_convert_dma_{1}", locSuffix, opId), srcView, newAllocDDROp);
                return rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "{0}_copy_{1}", locSuffix, opId),
                                                      convertDMAOp.getResult(), dstView);
            }
            return rewriter.create<VPUIP::ConvertDMAOp>(appendLoc(baseLoc, "{0}_convert_dma_{1}", locSuffix, opId),
                                                        srcView, dstView);

        } else {
            // Can't transfer data CMX2CMX directly
            if (mlir::isa<VPUIP::DistributedBufferType>(srcView.getType()) &&
                mlir::isa<VPUIP::DistributedBufferType>(dstView.getType())) {
                auto newDDRType =
                        mlir::dyn_cast<vpux::NDTypeInterface>(
                                mlir::cast<vpux::VPUIP::DistributedBufferType>(srcView.getType()).getCompactType())
                                .changeMemSpace(VPU::MemoryKind::DDR);
                auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(appendLoc(baseLoc, "new_DDR_buffer"),
                                                                            mlir::cast<mlir::MemRefType>(newDDRType));
                auto firstCopyOp = rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "{0}_copy_{1}", locSuffix, opId),
                                                                  srcView, newAllocDDROp);
                auto secondCopyOp = rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "{0}_copy_{1}", locSuffix, opId),
                                                                   firstCopyOp.getResult(), dstView);
                return secondCopyOp;
            }
            return rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "{0}_copy_{1}", locSuffix, opId), srcView,
                                                  dstView);
        }
    }

    SmallVector<Dim> getSubviewAxis(VPUIP::SubViewOp subview) const {
        auto inShape = getShape(subview.getSource());
        auto outShape = getShape(subview.getResult());
        SmallVector<Dim> subviewAxes = {};
        for (auto idx : irange(inShape.size())) {
            const auto dim = Dim(idx);
            if (inShape[dim] != outShape[dim]) {
                subviewAxes.push_back(dim);
            }
        }
        return subviewAxes;
    }

private:
    bool checkConcatReshapeCompatibility(VPUIP::ConcatViewOp concatOp, VPUIP::GenericReshapeOp genReshapeOp,
                                         vpux::Logger log) const {
        auto genReshapeType = vpux::getBufferType(genReshapeOp.getOutput());
        auto concatType = vpux::getBufferType(concatOp.getOutput());
        if ((genReshapeType.getRank() != 4 && genReshapeType.getRank() != 5) ||
            (concatType.getRank() != 4 && concatType.getRank() != 5)) {
            log.trace("Only 4/5D tensors are supported");
            return false;
        }

        auto reshapeShape = genReshapeType.getShape();
        auto concatShape = concatType.getShape();
        if (concatShape[Dim(0)] != 1) {
            log.trace("Only Batch size 1 is supported");
            return false;
        }

        auto compatibilityCheck = [&](auto reshapeShape, auto concatShape) {
            const bool numOfElementsCheck = genReshapeType.getNumElements() == concatType.getNumElements();
            if (reshapeShape.size() == 4) {
                // [1, A, B, C] -> [A*B, C, 1, 1]
                return reshapeShape[Dim(0)] == concatShape[Dim(1)] * concatShape[Dim(2)] &&
                       reshapeShape[Dim(1)] == concatShape[Dim(3)] && numOfElementsCheck;
            }
            // 1 , A, B, C -> A, B, C, 1 ,1 5D Grouped Matmul Case
            return reshapeShape[Dim(0)] == concatShape[Dim(1)] && reshapeShape[Dim(1)] == concatShape[Dim(2)] &&
                   reshapeShape[Dim(2)] == concatShape[Dim(3)] && numOfElementsCheck;
        };
        if (!compatibilityCheck(reshapeShape, concatShape)) {
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
            if (auto permCastOp = input.getDefiningOp<VPUIP::PermuteCastOp>()) {
                if (mlir::isa<mlir::BlockArgument>(permCastOp.getSource()) && permCastOp->hasOneUse()) {
                    return {input, nullptr};
                }
            }
        }
        return {nullptr, nullptr};
    }

    mlir::Value prepareLeftBranch(mlir::PatternRewriter& rewriter, mlir::Value leftBranchInput,
                                  VPUIP::GenericReshapeOp genReshape, VPUIP::PermuteCastOp permuteCastOp,
                                  mlir::Location loc) const {
        if (genReshape != nullptr) {
            return propagateReshapeCast(rewriter, leftBranchInput, permuteCastOp, loc, "left");
        }
        if (permuteCastOp == nullptr) {
            return leftBranchInput;
        }
        return propagatePermuteCast(rewriter, leftBranchInput, permuteCastOp, loc, "left");
    }

    std::optional<int64_t> getTilingAxis(mlir::Type type) const {
        auto outDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(type);
        if (outDistributedType == nullptr) {
            return std::nullopt;
        }

        const auto distribution = VPU::DistributionInfo::getClassFromAttr(outDistributedType.getDistribution());
        if (!VPU::isSegmentedLikeDistributionMode(mlir::cast<NDTypeInterface>(type), distribution)) {
            return std::nullopt;
        }
        const auto numTiles = distribution.getNumTiles();
        return VPU::getDistributedTilingAxis(numTiles);
    };

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("SplitUnbalancedDDRConcat{0}: Got Concat at '{1}'", getRewriterSuffix(), concatOp.getLoc());
        auto nestedLog = _log.nest();

        if (concatOp.getInputs().size() != 2) {
            nestedLog.trace("Only 2 inputs supported");
            return mlir::failure();
        }

        if (vpux::isSubByteType(mlir::cast<NDTypeInterface>(concatOp.getResult().getType()).getElementType())) {
            nestedLog.trace("Sub-byte types are not supported");
            return mlir::failure();
        }

        // Output type is same as type of all inputs
        auto isDdrConcat = vpux::getBufferType(concatOp).getMemoryKind() == VPU::MemoryKind::DDR;
        if (!isDdrConcat) {
            nestedLog.trace("All inputs must be DDR copies");
            return mlir::failure();
        }

        VPUIP::PermuteCastOp permuteCastOp = nullptr;
        auto genReshapeOp = getSingleUserOfType<VPUIP::GenericReshapeOp>(concatOp);
        if (genReshapeOp == nullptr) {
            permuteCastOp = getSingleUserOfType<VPUIP::PermuteCastOp>(concatOp);
            if (permuteCastOp == nullptr) {
                nestedLog.trace("No PermuteCast after concat, using already-propagated pattern");
            } else if (!checkMemShapesCompatibilityWithPerm(permuteCastOp)) {
                nestedLog.trace("PermuteCast compatibility check failed");
                return mlir::failure();
            }
        } else {
            nestedLog.trace("ConcatOp followed by GenericReshape");
            if (!checkConcatReshapeCompatibility(concatOp, genReshapeOp, nestedLog)) {
                return mlir::failure();
            }
            permuteCastOp = getSingleUserOfType<VPUIP::PermuteCastOp>(genReshapeOp);
            if (permuteCastOp == nullptr) {
                nestedLog.trace("GenericReshape must followed by only one PermuteCast");
                return mlir::failure();
            }
        }

        const auto concatAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
        if (concatAxes.size() != 1) {
            nestedLog.trace("Cannot extract concat axis");
            return mlir::failure();
        }
        auto origConcatDim = concatAxes.front();
        Dim newConcatDim = permuteCastOp == nullptr ? origConcatDim : Dim(0);

        if (genReshapeOp != nullptr) {
            // GenericReshape collapses 1,2 axis into 0, and 3 to 1, see checkConcatReshapeCompatibility
            // Therefore, new concat Dim is obtained as follows: 0 -> invalid, 1-2 -> 0, 3->1
            if (origConcatDim == Dim(0)) {
                nestedLog.trace("Unsupported orig concat dim {0}", origConcatDim);
                return mlir::failure();
            }
            if (auto is4D = vpux::getBufferType(genReshapeOp.getOutput()).getShape().size() == 4; is4D) {
                if (origConcatDim == Dim(3)) {
                    newConcatDim = Dim(1);
                }
            } else {  // 5D Grouped Matmul Case
                      // 1 , A, B, C -> A, B, C, 1 ,1
                newConcatDim = Dim(origConcatDim.ind() - 1);
            }

        } else if (permuteCastOp != nullptr) {
            // When there's only a PermuteCast on Concat output, just apply permute logic to get the new Concat dim
            auto permuteInOrder = mlir::cast<NDTypeInterface>(permuteCastOp.getSource().getType()).getDimsOrder();
            auto permuteOutOrder = mlir::cast<NDTypeInterface>(permuteCastOp->getResult(0).getType()).getDimsOrder();
            auto permuteMemPerm = DimsOrder::fromAffineMap(permuteCastOp.getMemPerm());
            auto concatAxisMemDim = permuteInOrder.toMemDim(origConcatDim);
            concatAxisMemDim = MemDim(permuteMemPerm.dimAt(concatAxisMemDim.ind()).ind());
            newConcatDim = permuteOutOrder.toDim(concatAxisMemDim);
        }

        nestedLog.trace("Concat axis transformation: {0} -> {1}", origConcatDim, newConcatDim);

        auto [leftBranchInput, leftBranchViewOp] = getLeftBranchInput(concatOp);
        if (leftBranchInput == nullptr) {
            nestedLog.trace("Can't get left branch input for concat at '{0}'", concatOp.getLoc());
            return mlir::failure();
        }
        // In the already-propagated pattern the PermuteCast was absorbed into the concat buffer layout,
        // but the PermuteCastOp still exists on each branch's input path. A raw BlockArgument (no
        // PermuteCastOp in the defining chain) means this is a different pattern, not already-propagated.
        if (permuteCastOp == nullptr && leftBranchInput.getDefiningOp<VPUIP::PermuteCastOp>() == nullptr) {
            nestedLog.trace("Already-propagated pattern requires PermuteCastOp on left branch input");
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
            nestedLog.trace("Can't get right branch input for concat at '{0}'", concatOp.getLoc());
            return mlir::failure();
        }

        if (leftBranchInput == rightBranchInput) {
            nestedLog.trace("Branches must have different inputs");
            return mlir::failure();
        }

        auto beforeSubviewOp = permuteCastOp != nullptr ? permuteCastOp : concatOp;
        auto beforeSubviewOpOutType = mlir::cast<NDTypeInterface>(beforeSubviewOp->getResult(0).getType());
        auto highestNonOneDim =
                getHighestNonTrivialDim(beforeSubviewOpOutType.getShape(), beforeSubviewOpOutType.getDimsOrder());

        SmallVector<VPUIP::SubViewOp> views;
        SmallVector<VPUIP::CopyOp> copyOps;

        const auto isNonDistributedCMXCopy = [](VPUIP::CopyOp copyOp) {
            auto bufferType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutputBuff().getType());
            return bufferType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                   !mlir::isa<vpux::VPUIP::DistributedBufferType>(bufferType);
        };
        for (auto user : beforeSubviewOp->getUsers()) {
            if (auto viewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
                if (!viewOp->hasOneUse()) {
                    nestedLog.trace("ViewOp at '{0}' must have only one user", viewOp->getLoc());
                    return mlir::failure();
                }
                views.push_back(viewOp);

                auto copyOp = getSingleUserOfType<VPUIP::CopyOp>(viewOp);
                if (copyOp == nullptr) {
                    nestedLog.trace("User is not a copy");
                    return mlir::failure();
                }
                if (isOutputDistributedCMX() && !vpux::VPUIP::hasDistributedOperand(copyOp)) {
                    nestedLog.trace("View at '{0}' user is not a Distributed Copy", viewOp->getLoc());
                    return mlir::failure();
                }
                if (!isOutputDistributedCMX() && !isNonDistributedCMXCopy(copyOp)) {
                    nestedLog.trace("View at '{0}' user is not a Non Distributed CMX Copy", viewOp->getLoc());
                    return mlir::failure();
                }
                copyOps.push_back(copyOp);
            } else {
                nestedLog.trace("All users must be View operations");
                return mlir::failure();
            }
        }
        if (views.empty()) {
            nestedLog.trace("Cannot find any SubView-> CMX Copy consumers");
            return mlir::failure();
        }

        if (copyOps.empty()) {
            nestedLog.trace("Expected at least 1 CMX copy user after concat");
            return mlir::failure();
        }

        auto rightBranchTilingAndSubviewOnSameAxis = false;
        PatternParamsInfo params{leftSizeOnConcatDim,
                                 getShape(leftBranchInput)[origConcatDim],
                                 leftViewOffsetOnConcatDim,
                                 getShape(rightBranchInput)[origConcatDim],
                                 origConcatDim,
                                 newConcatDim,
                                 permuteCastOp};
        // check the subview
        if (!isValidSubview(views, params.newConcatDim, params.leftConcatInputSize, params.rightInputSize)) {
            nestedLog.trace("SubView does not meet requirement");
            return mlir::failure();
        }
        if (!highestNonOneDim.has_value()) {
            nestedLog.trace("PermuteCast output shape is full on 1s");
            return mlir::failure();
        }

        if (isOutputDistributedCMX()) {
            auto maybeTilingAxis = getTilingAxis(copyOps.front().getResult().getType());
            if (!maybeTilingAxis.has_value()) {
                nestedLog.trace("Only SEGMENTED-like distribution is supported for consumers");
                return mlir::failure();
            }

            auto tilingAxis = maybeTilingAxis.value();
            auto rightBranchSubviewAxis = vpux::IE::getDiffInOutSizeDims(getShape(views.front().getOperand()),
                                                                         getShape(views.front().getResult()))
                                                  .front()
                                                  .ind();
            /*
            rightBranchInput (DistributedBuffer) -> Copy -> Concat (DDR) -> [Reshape/PermuteCast] -> Subview
            (rightBranchSubviewAxis) -> Copy -> DistributedBuffer (tilingAxis)

            To:

            rightBranchInput (DistributedBuffer) -> [Reshape/PermuteCast] -> Subview (rightBranchSubviewAxis) -> Copy ->
            Concat (CMX) -> DistributedBuffer (tilingAxis) if rightBranchSubviewAxis==tilingAxis

            If rightBranchSubviewAxis==tilingAxis, SubviewOp can't fetch data from single cluster as DistributedBuffer.
            Need to copy data to DDR. In SplitUnbalancedDDRConcatOnSameAxis, SubviewOp is replaced by
            ExtractFlatSliceOp. Therefore, don't need to apply the change to SplitUnbalancedDDRConcatOnSameAxis.
            */
            rightBranchTilingAndSubviewOnSameAxis =
                    mlir::isa<VPUIP::DistributedBufferType>(rightBranchInput.getType()) &&
                    (tilingAxis == rightBranchSubviewAxis) && (newConcatDim.ind() != tilingAxis);

            if (tilingAxis != 0 && tilingAxis != highestNonOneDim.value().ind()) {
                nestedLog.trace("Only tiling on major dim is supported");
                return mlir::failure();
            }

            auto allTilingAxisAreSame = llvm::all_of(copyOps, [&](VPUIP::CopyOp tiledCopy) {
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

            const auto isOverlappedOrNone = [](mlir::Value branchInput) {
                auto inputType = mlir::cast<vpux::NDTypeInterface>(branchInput.getType());
                auto distributedBufferType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType);
                if (distributedBufferType == nullptr) {
                    return false;
                }

                auto distribution = distributedBufferType.getDistribution();
                auto mode = distribution ? distribution.getMode() : nullptr;
                return mode == nullptr || mode.getValue() == VPU::DistributionMode::OVERLAPPED;
            };

            // We don't support OVERLAPPED mode
            if (isOverlappedOrNone(leftBranchInput) || isOverlappedOrNone(rightBranchInput)) {
                nestedLog.trace("Left or right input branches are OVERLAPPED or have NONE distribution mode.");
                return mlir::failure();
            }

            // check the segmented distribution alignment without explicit compute shapes and memory offsets
            if (!isValidSegment(views, copyOps, params.newConcatDim, params.leftConcatInputSize,
                                params.rightInputSize)) {
                nestedLog.trace("Segmented tiling is not aligned.");
                return mlir::failure();
            }
        }

        mlir::Value newLeftBranch =
                prepareLeftBranch(rewriter, leftBranchInput, genReshapeOp, permuteCastOp, concatOp->getLoc());

        mlir::Value newRightBranch;
        if (rightBranchTilingAndSubviewOnSameAxis) {
            auto newDDRType =
                    mlir::dyn_cast<vpux::NDTypeInterface>(
                            mlir::cast<vpux::VPUIP::DistributedBufferType>(rightBranchInput.getType()).getCompactType())
                            .changeMemSpace(VPU::MemoryKind::DDR);
            auto baseLoc = concatOp.getLoc();
            auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(appendLoc(baseLoc, "new_DDR_buffer"),
                                                                        mlir::cast<mlir::MemRefType>(newDDRType));
            auto firstCopyOp = rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "{0}_copy0_{1}"), rightBranchInput,
                                                              newAllocDDROp);
            newRightBranch = firstCopyOp;
            newRightBranch =
                    prepareRightBranch(rewriter, newRightBranch, genReshapeOp, permuteCastOp, concatOp->getLoc());
        } else {
            newRightBranch =
                    prepareRightBranch(rewriter, rightBranchInput, genReshapeOp, permuteCastOp, concatOp->getLoc());
        }

        for (size_t i = 0; i < copyOps.size(); ++i) {
            VPUIP::CopyOp copyOp = copyOps[i];
            VPUIP::SubViewOp subViewOp = views[i];

            rewriter.setInsertionPoint(copyOp);
            rewriteSubview(rewriter, concatOp, subViewOp, copyOp, newLeftBranch, newRightBranch, params, i);
        }
        const size_t LEFT_CONCAT_INPUT_ID = 0;
        mlir::Value leftConcatInput = concatOp.getInputs()[LEFT_CONCAT_INPUT_ID];
        copiesToRemove.push_back(leftConcatInput.getDefiningOp());
        if (permuteCastOp != nullptr) {
            rewriter.eraseOp(permuteCastOp);
        }

        if (genReshapeOp != nullptr) {
            rewriter.eraseOp(genReshapeOp);
        }

        nestedLog.trace("Successfully split unbalanced DDR Concat. {0}", concatOp->getLoc());

        rewriter.eraseOp(concatOp);
        for (auto copy : copiesToRemove) {
            if (copy == nullptr) {
                continue;
            }
            if (copy->use_empty()) {
                rewriter.eraseOp(copy);
            }
        }

        _log.unnest();
        return mlir::success();
    }

private:
    Logger _log;
};

/*
    Concat(Left[1, 32, 1023, 128], Right[1, 32, 1, 128]) -> (Reshape[32 * 1024, 128]) -> PermCast -> Views
    to
    (Reshape [32 * 1023, 128]) -> PermCast -> View -\
                                                     |-> Concat -> TiledCopy to CMX
    FlatView -> (Reshape [1, 128]) -> PermCast     -/
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

    VPUIP::DistributedBufferType updateDistributedType(mlir::Value, mlir::Value, ShapeRef) const override {
        return nullptr;
    }

    std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const override {
        const size_t RIGHT_INPUT_ID = 1;  // Right must be always second to preserve concat order
        mlir::Value patternInput = concatOp.getInputs()[RIGHT_INPUT_ID];
        if (mlir::isa<mlir::BlockArgument>(patternInput) || patternInput.getDefiningOp() == nullptr) {
            return {nullptr, {}};
        }

        SmallVector<mlir::Operation*> copies;
        auto copyOrConvertDMAOp = patternInput.getDefiningOp();
        // go through CopyOp/ConvertDMAOp with only one user.
        // all operations in this chain could be combined into one or a pair CopyOp/ConvertDMAOp
        while (mlir::isa_and_nonnull<VPUIP::CopyOp, VPUIP::ConvertDMAOp>(copyOrConvertDMAOp) &&
               (copyOrConvertDMAOp->hasOneUse())) {
            patternInput = copyOrConvertDMAOp->getOperand(0);
            if (patternInput == nullptr) {
                break;
            }
            copies.push_back(copyOrConvertDMAOp);
            copyOrConvertDMAOp = patternInput.getDefiningOp();
        }
        // Check if the input has stride. Because we will move permuteCast to the front. The pattern becomes
        // input(stride) -> permuteCast -> copy.
        // Since permuteCast can not support stride input, so we will lose the stride info for the
        // DMA after permuteCast, which finally cause accuracy issue.
        // TODO: #E180063 to support stride input for permuteCast.
        if (!copies.empty()) {
            const auto inType = mlir::cast<vpux::NDTypeInterface>(copies.back()->getOperand(0).getType());
            const auto dimsOrder = inType.getDimsOrder();
            const auto logicShape = inType.getShape();
            const auto memShape = dimsOrder.toMemoryOrder(logicShape);
            const auto compactMemStrides =
                    StrideReqs::compact(dimsOrder.numDims()).calcStrides(inType.getElemTypeSize(), memShape);
            const auto strides = inType.getStrides();
            const auto memStrides = dimsOrder.toMemoryOrder(strides);
            for (auto dim : irange(dimsOrder.numDims())) {
                // No need to check the highest dim if the shape is 1, such as below case, we think it is compact.
                // memref<1x32x1x96xf16, {order = #NCHW, strides = [9216, 96, 96, 1]}, @DDR>
                if (dim == 0 && memShape[MemDim(dim)] == 1) {
                    continue;
                }
                if (memStrides[MemDim(dim)] != compactMemStrides[MemDim(dim)]) {
                    return {nullptr, {}};
                }
            }
        }
        if (patternInput != nullptr) {
            return {patternInput, copies};
        }
        return {nullptr, {}};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput,
                                   VPUIP::GenericReshapeOp genReshape, VPUIP::PermuteCastOp permuteCastOp,
                                   mlir::Location loc) const override {
        if (genReshape != nullptr) {
            return propagateReshapeCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
        }
        if (permuteCastOp == nullptr) {
            return rightBranchInput;
        }
        return propagatePermuteCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
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
        auto dstBuffer =
                createNewConcatBuffer(rewriter, subViewOp, distributedCopy, takeOpLoc(origConcatOp, "buf_{0}", index));

        // Concat on other axis, so original offset is used
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto srcShape = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()));
        auto createViewBranch = [&](mlir::Value src, int64_t origDimSize, int64_t dstOffsetVal, int64_t srcOffsetVal,
                                    StringRef locSuffix, bool updateDistributionType = true) -> mlir::Value {
            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[params.newConcatDim] = origDimSize;

            Shape newSrcOffset(srcOffset);
            newSrcOffset[params.newConcatDim] = srcOffsetVal;

            Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
            dstOffset[params.newConcatDim] = dstOffsetVal;

            return createNewCopyBranch(rewriter, src, dstBuffer, copyShape, newSrcOffset, dstOffset,
                                       origConcatOp->getLoc(), locSuffix, index, updateDistributionType);
        };
        auto origConcatDimSize = getShape(origConcatOp->getResult(0))[params.origConcatDim];
        auto viewMultiplier = srcOffset[params.newConcatDim] / origConcatDimSize;
        auto leftBranchOffset = srcOffset[params.newConcatDim] - viewMultiplier * params.rightInputSize +
                                params.leftViewOffset + viewMultiplier * params.leftViewOffset;
        if (srcOffset[params.newConcatDim] % origConcatDimSize < params.leftConcatInputSize &&
            srcOffset[params.newConcatDim] % origConcatDimSize + srcShape[params.newConcatDim] >
                    params.leftConcatInputSize) {
            // SubView crosses left branch and right branch
            auto newLeftViewBranch =
                    createViewBranch(newLeftBranch, srcShape[params.newConcatDim] - params.rightInputSize,
                                     /*dstOffsetVal=*/0, leftBranchOffset, "left");
            auto newRightViewBranch =
                    createViewBranch(newRightBranch, params.rightInputSize,
                                     srcShape[params.newConcatDim] - params.rightInputSize, viewMultiplier, "right");
            SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
            auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(takeOpLoc(origConcatOp, "concat_{0}", index),
                                                                    concatInputs, dstBuffer);
            rewriter.replaceAllUsesWith(distributedCopy->getResult(0), newConcatOp);
        } else if (srcOffset[params.newConcatDim] % origConcatDimSize < params.leftConcatInputSize &&
                   srcOffset[params.newConcatDim] % origConcatDimSize + srcShape[params.newConcatDim] <=
                           params.leftConcatInputSize) {
            // SubView on the left branch
            auto newLeftViewBranch = createViewBranch(newLeftBranch, srcShape[params.newConcatDim], /*dstOffsetVal=*/0,
                                                      leftBranchOffset, "left", /*updateDistributionType*/ false);
            rewriter.replaceAllUsesWith(distributedCopy->getResult(0), newLeftViewBranch);
        } else {
            VPUX_THROW("Not supported case: SubView must be on the right branch");
        }

        rewriter.eraseOp(distributedCopy);
        rewriter.eraseOp(subViewOp);
    }
};

class SplitUnbalancedDDRConcatToNonDistributedCMX : public SplitUnbalancedDDRConcatOnOtherAxis {
public:
    using SplitUnbalancedDDRConcatOnOtherAxis::SplitUnbalancedDDRConcatOnOtherAxis;

private:
    StringRef getRewriterSuffix() const override {
        return "ToNonDistributedCMX";
    }

    bool isSplitSupported(Dim, int64_t) const override {
        return true;
    }

    bool isOutputDistributedCMX() const override {
        return false;
    }

    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp, VPUIP::CopyOp copy,
                                      mlir::Location bufferLoc) const override {
        auto bufferType = copy.getOutputBuff().getType();
        return rewriter.create<mlir::memref::AllocOp>(bufferLoc, mlir::cast<mlir::MemRefType>(bufferType));
    }
};

class SplitUnbalancedDDRConcatOnSameAxis : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    // Stores the slice dimension for right branch when it comes from a block argument
    // When set, indicates that the right branch should use SubView instead of ExtractFlatSlice
    mutable std::optional<Dim> _rightBranchBlockArgSliceDim = std::nullopt;

    StringRef getRewriterSuffix() const override {
        return "OnSameAxis";
    }

    bool isValidSegment(SmallVector<VPUIP::SubViewOp>& views, SmallVector<VPUIP::CopyOp>& distributedCopies,
                        Dim newConcatDim, int64_t leftConcatInputSize, int64_t rightConcatInputSize) const override {
        for (size_t i = 0; i < distributedCopies.size(); ++i) {
            VPUIP::CopyOp distributedCopy = distributedCopies[i];
            VPUX_THROW_UNLESS(vpux::VPUIP::hasDistributedOperand(distributedCopy), "Expected a distributed Copy op");
            auto resultType = distributedCopy->getResult(0).getType();

            if (auto dstDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(resultType)) {
                auto dstDistribution = dstDistributedType.getDistribution();
                auto dstDistributionInfo = VPU::DistributionInfo::getClassFromAttr(dstDistribution);

                if (isDistributionWithExplicitShapesAndOffsets(dstDistributionInfo)) {
                    const auto computeShapes = VPU::arrayAttrToVecOfShapes(dstDistribution.getComputeShapes());
                    // After optimization, if the tensor is not evenly split, the distributed copy for the left
                    // concatenation data will result in a smaller compute shape on the last cluster.
                    // Ensure that the left branch has sufficient data for the last cluster.
                    return computeShapes.back()[newConcatDim] > rightConcatInputSize;
                }

                const auto distributionMode = dstDistributionInfo.getDistributionMode();

                if (distributionMode != VPU::DistributionMode::SEGMENTED) {
                    return true;
                }

                VPUIP::SubViewOp subViewOp = views[i];
                auto copyShape = getShape(subViewOp->getResult(0)).toValues();
                copyShape[newConcatDim] = leftConcatInputSize;

                auto shape = to_small_vector(copyShape.raw());
                const auto numClusters = dstDistributionInfo.getNumClusters();

                const auto tilingScheme = dstDistributionInfo.getNumTiles();
                const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
                VPUX_THROW_UNLESS(axis < int64_t(tilingScheme.size()),
                                  "Segmented tiling scheme requires at least 1 dimension "
                                  "to be segmented but the tiling schema is [1, 1, 1, 1]");

                const auto segmentedShape =
                        VPU::splitSegmentedShape(shape, tilingScheme, numClusters, axis, std::nullopt,
                                                 dstDistributionInfo.hasUniformDistributedSegments());
                VPUX_THROW_UNLESS(segmentedShape.has_value(), "Improper split, '{0}' over '{1}' tiles", shape[axis],
                                  tilingScheme[axis]);
                const auto& segmentedShapeValue = segmentedShape.value();
                auto alignment = SmallVector<int64_t>(dstDistributionInfo.getAlignment());

                if (alignment.empty()) {
                    return true;
                }

                for (size_t i = 0; i + 1 < segmentedShapeValue.size(); ++i) {
                    for (size_t ind = 0; ind < copyShape.size(); ++ind) {
                        if (segmentedShapeValue[i][Dim(ind)] % alignment[ind] != 0) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    VPUIP::DistributedBufferType updateDistributedType(mlir::Value dst, mlir::Value dstView,
                                                       ShapeRef copyShape) const override {
        auto dstDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(dst.getType());
        if (!dstDistributedType) {
            return nullptr;
        }

        auto dstType = mlir::cast<vpux::NDTypeInterface>(dst.getType());
        auto dstShape = dstType.getShape();
        auto dstDistribution = dstDistributedType.getDistribution();

        auto dstDistributionInfo = VPU::DistributionInfo::getClassFromAttr(dstDistribution);
        if (!isDistributionWithExplicitShapesAndOffsets(dstDistributionInfo)) {
            return nullptr;
        }

        const auto dstComputeShapes = VPU::arrayAttrToVecOfShapes(dstDistribution.getComputeShapes());
        const auto dstMemoryOffsets = VPU::arrayAttrToVecOfShapes(dstDistribution.getMemoryOffsets());

        SmallVector<Shape> newComputeShapes;
        SmallVector<Shape> newMemoryOffsets;
        for (const auto& shape : dstComputeShapes) {
            newComputeShapes.push_back(shape);
        }
        for (const auto& offset : dstMemoryOffsets) {
            newMemoryOffsets.push_back(offset);
        }

        // We split the tensor unevenly to align the tensor, keeping the compute shapes except for the last
        // one which will be concatenated by the right branch. It should be smaller than the original
        // compute shape on the last cluster.
        // Example: left[1023, 128, 1, 1] right[1, 128, 1, 1] to be distributed on 3 clusters
        //    cluster0 [352, 128, 1, 1]
        //    cluster1 [336, 128, 1, 1]
        //    cluster2 [335, 128, 1, 1]
        for (size_t i = 0; i < dstShape.size(); ++i) {
            if (dstShape[Dim(i)] != copyShape[Dim(i)]) {
                newComputeShapes.back()[Dim(i)] -= dstShape[Dim(i)] - copyShape[Dim(i)];
                VPUX_THROW_WHEN(newComputeShapes.back()[Dim(i)] < 1, "Not supported subview shape");
            }
        }

        auto shapesAttr = vpux::getIntArrayOfArray(dstDistributedType.getContext(), newComputeShapes);
        auto offsetsAttr = vpux::getIntArrayOfArray(dstDistributedType.getContext(), newMemoryOffsets);

        auto dstViewDistribution = mlir::cast<VPUIP::DistributedBufferType>(dstView.getType()).getDistribution();
        auto newDistributionAttr = VPU::DistributionInfoAttr::get(
                dstDistributedType.getContext(),
                VPU::DistributionModeAttr::get(dstDistributedType.getContext(), VPU::DistributionMode::OVERLAPPED),
                dstViewDistribution.getNumTiles(), dstViewDistribution.getKernel(), dstViewDistribution.getPads(),
                dstViewDistribution.getStrides(), dstViewDistribution.getNumClusters(),
                dstViewDistribution.getAlignment(), dstViewDistribution.getUniformDistributedSegments(), shapesAttr,
                offsetsAttr, shapesAttr, offsetsAttr, dstViewDistribution.getEqualMemoryAndComputeView(),
                dstViewDistribution.getMemoryNumTiles());

        return VPUIP::DistributedBufferType::get(dstDistributedType.getContext(), copyShape,
                                                 dstDistributedType.getElementType(), dstDistributedType.getLayout(),
                                                 dstDistributedType.getMemSpace(), newDistributionAttr,
                                                 dstDistributedType.getSparsityCompression());
    };

    std::pair<mlir::Value, SmallVector<mlir::Operation*>> getRightBranchInput(
            VPUIP::ConcatViewOp concatOp) const override {
        const size_t RIGHT_INPUT_ID = 1;  // Right must be always second to preserve concat order
        mlir::Value patternInput = concatOp.getInputs()[RIGHT_INPUT_ID];

        // There is no sense to traverse more than 3 times, because eventually we must end with strided DDR->DDR started
        // from Distributed CMX. ViewLike ops aren't allowed, because they change layout/shape and we must consider them
        // in rewriter Longest possible chain is NCECompute -> f32->f16 Copy, CMX->DDR, DDR->Strided DDR
        int depth = 3;
        SmallVector<mlir::Operation*> copies;
        mlir::Value blockArgumentInput = nullptr;

        while (patternInput != nullptr && !mlir::isa<vpux::VPUIP::DistributedBufferType>(patternInput.getType()) &&
               depth > 0) {
            if (mlir::isa<mlir::BlockArgument>(patternInput)) {
                // Found BlockArgument input - store it and break
                blockArgumentInput = patternInput;
                break;
            }

            if (patternInput.getDefiningOp() == nullptr) {
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

        auto isSegmentedAndNotNull = [](mlir::Value input) {
            if (input == nullptr) {
                return false;
            }

            auto inputType = mlir::cast<NDTypeInterface>(input.getType());
            auto distributedBufferType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);
            if (distributedBufferType == nullptr) {
                return false;
            }

            auto distribution = distributedBufferType.getDistribution();
            auto mode = distribution ? distribution.getMode() : nullptr;

            return mode != nullptr && mode.getValue() == VPU::DistributionMode::SEGMENTED;
        };

        if (isSegmentedAndNotNull(patternInput)) {
            return {patternInput, copies};
        }

        // Support BlockArgument -> Copy pattern
        // In this case, we'll need to create a subview copy pair from DDR to CMX later
        if (blockArgumentInput != nullptr && !copies.empty()) {
            auto inputType = mlir::cast<NDTypeInterface>(blockArgumentInput.getType());
            if (inputType.getMemoryKind() == VPU::MemoryKind::DDR) {
                // Check concat dimension to determine the slice dimension
                const auto concatAxes = vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]),
                                                                       getShape(concatOp.getResult()));
                if (concatAxes.size() != 1) {
                    return {nullptr, {}};
                }
                auto origConcatDim = concatAxes.front();

                // Check that there's exactly one non-1 dimension above concat dim in memory layout
                // This ensures we can uniquely determine the slice dimension
                auto rightBranchShape = inputType.getShape();
                std::optional<Dim> sliceDim = std::nullopt;
                int64_t nonOneDimCount = 0;

                auto dimsOrder = inputType.getDimsOrder();
                auto concatMemDim = dimsOrder.toMemDim(origConcatDim);
                for (int64_t i = 0; i < concatMemDim.ind(); ++i) {
                    auto memDim = MemDim(i);
                    auto dim = dimsOrder.toDim(memDim);
                    if (rightBranchShape[dim] != 1) {
                        sliceDim = dim;
                        ++nonOneDimCount;
                    }
                }

                // Only support case where there's exactly one non-1 dimension above concat dim
                if (nonOneDimCount != 1 || !sliceDim.has_value()) {
                    return {nullptr, {}};
                }

                _rightBranchBlockArgSliceDim = sliceDim;
                return {blockArgumentInput, copies};
            }
        }

        return {nullptr, {}};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter&, mlir::Value rightBranchInput, VPUIP::GenericReshapeOp,
                                   VPUIP::PermuteCastOp, mlir::Location) const override {
        return rightBranchInput;
    }

    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp, VPUIP::CopyOp distributedCopy,
                                      mlir::Location bufferLoc) const override {
        auto dstBufferType = distributedCopy.getOutputBuff().getType();
        return rewriter.create<VPURT::AllocDistributed>(bufferLoc, dstBufferType, nullptr, nullptr);
    }

    mlir::Value rewriterLeftSubViewBranch(mlir::PatternRewriter& rewriter, const PatternParamsInfo& params,
                                          VPUIP::SubViewOp subViewOp, mlir::Value newLeftBranch, mlir::Value dstBuffer,
                                          mlir::Location baseLoc, size_t index, int64_t shapeSize, int64_t offset,
                                          bool updateDistributionType = true) const {
        auto newConcatDim = params.newConcatDim;
        auto copyShape = getShape(subViewOp->getResult(0)).toValues();
        copyShape[newConcatDim] = shapeSize;
        Shape srcOffset(SmallVector<int64_t>(copyShape.size(), 0));
        srcOffset[newConcatDim] = offset;

        Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
        return createNewCopyBranch(rewriter, newLeftBranch, dstBuffer, copyShape, srcOffset, dstOffset, baseLoc, "left",
                                   index, updateDistributionType);
    }

    mlir::Value rewriterRightSubViewBranch(mlir::PatternRewriter& rewriter, const PatternParamsInfo& params,
                                           size_t viewMultiplier, VPUIP::SubViewOp, mlir::Value newRightBranch,
                                           mlir::Value dstBuffer, mlir::Location baseLoc, size_t index,
                                           int64_t leftInputSize) const {
        // Extract the right branch slice based on input type
        mlir::Value rightBranchSlice;
        if (_rightBranchBlockArgSliceDim.has_value()) {
            // Optimized path: Create SubView on DDR BlockArgument directly
            auto rightBranchType = mlir::cast<NDTypeInterface>(newRightBranch.getType());
            auto rightBranchShape = rightBranchType.getShape();

            // Use the slice dimension determined during getRightBranchInput
            // This dimension is validated to be the only non-1 dimension above concat dim
            auto sliceDim = _rightBranchBlockArgSliceDim.value();

            // Create SubView on the DDR BlockArgument to extract the needed slice
            Shape srcOffset(SmallVector<int64_t>(rightBranchShape.size(), 0));
            srcOffset[sliceDim] = static_cast<int64_t>(viewMultiplier);

            Shape srcShape = rightBranchShape.toValues();
            srcShape[sliceDim] = 1;

            rightBranchSlice =
                    rewriter.createOrFold<VPUIP::SubViewOp>(appendLoc(baseLoc, "right_ddr_subview_{0}", index),
                                                            newRightBranch, srcOffset.raw(), srcShape.raw());
        } else {
            // Original path for distributed buffer input: use ExtractFlatSliceOp
            rightBranchSlice = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                    appendLoc(baseLoc, "pseudo_dst_view_{0}", index), newRightBranch, viewMultiplier);
        }

        // Apply reshape and permute cast to match the concat buffer layout
        auto normalizedShape = propagateReshapeCast(rewriter, rightBranchSlice, params.castOp, baseLoc,
                                                    printToString("right_{0}", index));

        // Create ExtractFlatSliceOp on the CMX concat buffer
        auto dstView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                appendLoc(baseLoc, "right_dst_view_{0}", index), dstBuffer, leftInputSize, params.rightInputSize);

        // Create DDR→CMX copy
        return rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "right_copy_{0}", index), normalizedShape, dstView);
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer =
                createNewConcatBuffer(rewriter, subViewOp, distributedCopy, takeOpLoc(origConcatOp, "buf_{0}", index));
        auto baseLoc = origConcatOp->getLoc();
        // Concat on same axis, so must do manual strided access
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto srcShape = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()));
        auto origConcatDimSize = getShape(origConcatOp->getResult(0))[params.origConcatDim];

        // Here we support subview size smaller than concat dim size, like:
        // left is 8319, right is 1, total 8320.
        // first subview is coming from 0 to 4160, which actually totally come from left branch.
        // second subview is coming from 4160 to 8320, which need to concat the rest left and right
        auto viewMultiplier = srcOffset[params.newConcatDim] / origConcatDimSize;
        auto leftBranchOffset = srcOffset[params.newConcatDim] - viewMultiplier * params.rightInputSize +
                                params.leftViewOffset + viewMultiplier * params.leftViewOffset;

        if (srcOffset[params.newConcatDim] % origConcatDimSize < params.leftConcatInputSize &&
            srcOffset[params.newConcatDim] % origConcatDimSize + srcShape[params.newConcatDim] >
                    params.leftConcatInputSize) {
            // SubView crosses left branch and right branch
            auto newLeftViewBranch =
                    rewriterLeftSubViewBranch(rewriter, params, subViewOp, newLeftBranch, dstBuffer, baseLoc, index,
                                              srcShape[params.newConcatDim] - params.rightInputSize, leftBranchOffset);
            auto newRightViewBranch =
                    rewriterRightSubViewBranch(rewriter, params, viewMultiplier, subViewOp, newRightBranch, dstBuffer,
                                               baseLoc, index, srcShape[params.newConcatDim] - params.rightInputSize);
            SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
            auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(takeOpLoc(origConcatOp, "concat_{0}", index),
                                                                    concatInputs, dstBuffer);
            rewriter.replaceAllUsesWith(distributedCopy->getResult(0), newConcatOp);

        } else if (srcOffset[params.newConcatDim] % origConcatDimSize < params.leftConcatInputSize &&
                   srcOffset[params.newConcatDim] % origConcatDimSize + srcShape[params.newConcatDim] <=
                           params.leftConcatInputSize) {
            // SubView on the left branch
            auto newLeftViewBranch = rewriterLeftSubViewBranch(rewriter, params, subViewOp, newLeftBranch, dstBuffer,
                                                               baseLoc, index, srcShape[params.newConcatDim],
                                                               leftBranchOffset, /*updateDistributionType*/ false);
            rewriter.replaceAllUsesWith(distributedCopy->getResult(0), newLeftViewBranch);
        } else {
            VPUX_THROW("Not supported case: SubView must be on the right branch");
        }
        rewriter.eraseOp(distributedCopy);
        rewriter.eraseOp(subViewOp);
    }
};

// Common base for multi-branch unbalanced DDR concat splitting patterns.
// Provides helpers shared by SplitMultiLeftUnbalancedDDRConcatOnSameAxis and
// SplitMultiLeftUnbalancedDDRConcatOnOtherAxis.
class SplitMultiUnbalancedDDRConcatBase : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    SplitMultiUnbalancedDDRConcatBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

protected:
    // Collects SubViewOp → CopyOp (to CMX_NN) pairs from all users of permuteCastOp.
    // Returns false on any unexpected user pattern or when no consumers are found.
    bool collectSubViewCopyConsumers(VPUIP::PermuteCastOp permuteCastOp, SmallVector<VPUIP::SubViewOp>& subViews,
                                     SmallVector<VPUIP::CopyOp>& cmxCopies, Logger log) const {
        for (auto* user : permuteCastOp->getUsers()) {
            auto viewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user);
            if (viewOp == nullptr || !viewOp->hasOneUse()) {
                log.trace("PermuteCast user is not a single-use SubViewOp");
                return false;
            }
            auto copyOp = getSingleUserOfType<VPUIP::CopyOp>(viewOp);
            if (copyOp == nullptr) {
                log.trace("SubViewOp user is not a CopyOp");
                return false;
            }
            auto dstType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutputBuff().getType());
            if (dstType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
                log.trace("CMX copy destination is not CMX_NN");
                return false;
            }
            subViews.push_back(viewOp);
            cmxCopies.push_back(copyOp);
        }
        if (subViews.empty()) {
            log.trace("No SubView consumers");
            return false;
        }
        return true;
    }

    // Applies GenericReshapeOp then PermuteCastOp to input, reusing the dimension order and
    // permutation attributes from permuteCastRef.
    mlir::Value applyReshapeAndPermuteCast(mlir::PatternRewriter& rewriter, mlir::Value input, ShapeRef newShape,
                                           VPUIP::PermuteCastOp permuteCastRef, mlir::Location reshapeLoc,
                                           mlir::Location permcastLoc) const {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        auto afterReshapeType = inputType.changeShape(newShape);
        auto reshapeOut = rewriter.createOrFold<VPUIP::GenericReshapeOp>(reshapeLoc, afterReshapeType, input);
        auto permCastDimsOrder =
                mlir::cast<vpux::NDTypeInterface>(permuteCastRef->getResult(0).getType()).getDimsOrder();
        auto afterPermCastType = afterReshapeType.changeDimsOrder(std::move(permCastDimsOrder));
        return rewriter.create<VPUIP::PermuteCastOp>(permcastLoc, afterPermCastType, reshapeOut,
                                                     permuteCastRef.getDstOrderAttr(), permuteCastRef.getMemPermAttr());
    }

    // Erases operations that have no remaining uses.
    void eraseDeadOps(mlir::PatternRewriter& rewriter, ArrayRef<mlir::Operation*> ops) const {
        for (auto* op : ops) {
            if (op != nullptr && op->use_empty()) {
                rewriter.eraseOp(op);
            }
        }
    }

    // Holds the matched GenericReshapeOp → PermuteCastOp structural result.
    struct ConcatPrefixResult {
        VPUIP::GenericReshapeOp genReshapeOp;
        VPUIP::PermuteCastOp permuteCastOp;
    };

    // Validates that concatOp has ≥ minInputs DDR inputs and is followed by a single
    // GenericReshapeOp → PermuteCastOp chain.  Returns the matched ops on success.
    mlir::FailureOr<ConcatPrefixResult> matchConcatReshapePermuteCastChain(VPUIP::ConcatViewOp concatOp,
                                                                           size_t minInputs, Logger log) const {
        if (concatOp.getInputs().size() < minInputs) {
            log.trace("Need at least {0} inputs, got {1}", minInputs, concatOp.getInputs().size());
            return mlir::failure();
        }
        if (vpux::getBufferType(concatOp).getMemoryKind() != VPU::MemoryKind::DDR) {
            log.trace("Concat output must be DDR");
            return mlir::failure();
        }
        auto genReshapeOp = getSingleUserOfType<VPUIP::GenericReshapeOp>(concatOp);
        if (genReshapeOp == nullptr) {
            log.trace("No single GenericReshape user");
            return mlir::failure();
        }
        auto permuteCastOp = getSingleUserOfType<VPUIP::PermuteCastOp>(genReshapeOp);
        if (permuteCastOp == nullptr) {
            log.trace("No single PermuteCast user after GenericReshape");
            return mlir::failure();
        }
        return ConcatPrefixResult{genReshapeOp, permuteCastOp};
    }

    Logger _log;
};

/*
    Multi-left unbalanced DDR Concat (N ≥ 2 left branches + 1 right CMX branch):

    Left0[1,C,H0,W]@DDR ──Copy──┐
    Left1[1,C,H1,W]@DDR ──Copy──┤
    ...                          ├─▶ ConcatView[1,C,H_total,W]@DDR
    LeftN[1,C,HN,W]@DDR ──Copy──┤      │
    Right[1,C,1,W]@CMX  ──Copy──┘   GenericReshape[C*H_total, W, 1, 1]
                                        │
                                   PermuteCast (NHWC)
                                     ┌──┤──┐
                              SubView0   ...  SubViewM
                                 │              │
                          CMX SEGMENTED Copy  (same)

    Transformed to – per consumer SubView [off, off+sz):

      For each left[i] that overlaps the SubView's H range:
        preparedLeft[i] (GenericReshape+PermuteCast of DDR block arg)
          → SubView of [sz_i, W, 1, 1] rows at srcRowBase
          → ExtractFlatSliceOp of CMX buffer at dstFlatBase → Copy directly into CMX
      For right branch (if SubView extends past totalLeft):
        ExtractFlatSlice(rightInput, viewMul)
          → GenericReshape+PermuteCast
          → ExtractFlatSliceOp of CMX buffer at dstFlatBase → Copy directly into CMX
      ConcatViewOp(allCopies, cmxBuf) replaces the original SubView→CMXCopy chain.

    Eliminates the large shared DDR concat buffer; all DMAs go directly DDR→CMX,
    enabling DMA scheduling to interleave with NCE tasks.
*/
class SplitMultiLeftUnbalancedDDRConcatOnSameAxis : public SplitMultiUnbalancedDDRConcatBase {
public:
    using SplitMultiUnbalancedDDRConcatBase::SplitMultiUnbalancedDDRConcatBase;

    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const override {
        _log.trace("SplitMultiLeftUnbalancedDDRConcatOnSameAxis: Got Concat at '{0}'", concatOp.getLoc());
        auto nestedLog = _log.nest();

        // Match: numInputs ≥ 3, DDR output, GenericReshape → PermuteCast chain.
        const size_t numInputs = concatOp.getInputs().size();
        auto prefixResult = matchConcatReshapePermuteCastChain(concatOp, /*minInputs=*/3, nestedLog);
        if (mlir::failed(prefixResult)) {
            return mlir::failure();
        }
        auto [genReshapeOp, permuteCastOp] = prefixResult.value();

        // Validate Concat→Reshape shapes: [1,C,H,W] → [C*H, W, 1, 1].
        if (!checkSameAxisReshapeCompatibility(concatOp, genReshapeOp, nestedLog)) {
            return mlir::failure();
        }

        // Only concat on dim2 (H) supported. After [1,C,H,W]→[C*H,W,1,1] it folds into dim0.
        const auto concatAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
        if (concatAxes.size() != 1 || concatAxes.front() != Dim(2)) {
            nestedLog.trace("Only concat on dim2 (H) is supported");
            return mlir::failure();
        }
        const Dim origConcatDim = Dim(2);
        const Dim newConcatDim = Dim(0);

        auto branchesResult = validateInputBranches(concatOp, numInputs, nestedLog);
        if (mlir::failed(branchesResult)) {
            return mlir::failure();
        }
        auto [ddrBranchSources, cmxRightBranch, rightCopiesToRemove] = branchesResult.value();

        // Collect PermuteCast users: SubViewOp → CopyOp pairs going to CMX.
        SmallVector<VPUIP::SubViewOp> subViews;
        SmallVector<VPUIP::CopyOp> cmxCopies;
        if (!collectSubViewCopyConsumers(permuteCastOp, subViews, cmxCopies, nestedLog)) {
            return mlir::failure();
        }

        // Cumulative row offsets across all DDR branches (left + optional DDR right).
        SmallVector<int64_t> ddrBranchCumSizes;
        ddrBranchCumSizes.reserve(ddrBranchSources.size() + 1);
        ddrBranchCumSizes.push_back(0);
        for (auto src : ddrBranchSources) {
            const auto branchShape = mlir::cast<vpux::NDTypeInterface>(src.getType()).getShape();
            ddrBranchCumSizes.push_back(ddrBranchCumSizes.back() + branchShape[origConcatDim]);
        }
        const int64_t ddrRowsTotal = ddrBranchCumSizes.back();
        const int64_t cmxRightSize =
                cmxRightBranch != nullptr
                        ? mlir::cast<vpux::NDTypeInterface>(cmxRightBranch.getType()).getShape()[origConcatDim]
                        : 0;
        const int64_t origConcatDimSize = getShape(concatOp.getResult())[origConcatDim];
        VPUX_THROW_UNLESS(ddrRowsTotal + cmxRightSize == origConcatDimSize, "Input sizes do not match concat dim size");

        if (mlir::failed(validateSubViewConsumers(subViews, cmxCopies, permuteCastOp, newConcatDim, origConcatDimSize,
                                                  ddrRowsTotal, nestedLog))) {
            return mlir::failure();
        }

        // Pre-validate cluster boundaries for every consumer before any IR modification.
        SmallVector<SmallVector<int64_t>> allClusterCumOffsets;
        allClusterCumOffsets.reserve(cmxCopies.size());
        for (size_t i = 0; i < cmxCopies.size(); ++i) {
            auto maybe = buildClusterCumOffsets(cmxCopies[i].getOutputBuff(), nestedLog);
            if (!maybe.has_value()) {
                nestedLog.trace("Cannot build cluster offsets for consumer {0} - skipping pattern", i);
                return mlir::failure();
            }
            allClusterCumOffsets.push_back(std::move(maybe.value()));
        }

        // Prepare every DDR branch: [1,C,Hi,W] → [C*Hi,W,1,1] → PermuteCast.
        // A CMX SEGMENTED right branch is prepared per-consumer inside `rewriteOneConsumer` because
        // ExtractFlatSliceOp on a distributed buffer must precede reshape/permcast.
        SmallVector<mlir::Value> preparedDDRBranches;
        preparedDDRBranches.reserve(ddrBranchSources.size());
        for (size_t i = 0; i < ddrBranchSources.size(); ++i) {
            const auto branchShape = mlir::cast<vpux::NDTypeInterface>(ddrBranchSources[i].getType()).getShape();
            Shape newShape{branchShape[Dim(1)] * branchShape[Dim(2)], branchShape[Dim(3)], 1, 1};
            preparedDDRBranches.push_back(
                    applyReshapeAndPermuteCast(rewriter, ddrBranchSources[i], newShape, permuteCastOp,
                                               appendLoc(concatOp->getLoc(), "multi_branch_reshape_{0}", i),
                                               appendLoc(concatOp->getLoc(), "multi_branch_permcast_{0}", i)));
        }

        const SameAxisRewriteContext ctx{newConcatDim,   origConcatDimSize, ddrBranchCumSizes, preparedDDRBranches,
                                         cmxRightBranch, cmxRightSize,      permuteCastOp};
        for (size_t consumerIdx = 0; consumerIdx < subViews.size(); ++consumerIdx) {
            rewriteOneConsumer(rewriter, concatOp->getLoc(), consumerIdx, subViews[consumerIdx], cmxCopies[consumerIdx],
                               allClusterCumOffsets[consumerIdx], ctx);
        }

        // Collect the concat input CopyOps for the DDR-only region (indices [0, numInputs-1))
        // before erasing concatOp to avoid use-after-erase. The last input's CopyOp chain is
        // handled separately by `rightCopiesToRemove` to avoid a double erase.
        SmallVector<mlir::Operation*> concatInputCopies;
        concatInputCopies.reserve(numInputs - 1);
        for (size_t i = 0; i + 1 < numInputs; ++i) {
            if (auto* defOp = concatOp.getInputs()[i].getDefiningOp()) {
                concatInputCopies.push_back(defOp);
            }
        }

        const auto concatLoc = concatOp->getLoc();
        rewriter.eraseOp(permuteCastOp);
        rewriter.eraseOp(genReshapeOp);
        rewriter.eraseOp(concatOp);
        eraseDeadOps(rewriter, rightCopiesToRemove);
        eraseDeadOps(rewriter, concatInputCopies);

        nestedLog.trace("Successfully rewrote multi-left DDR Concat at '{0}'", concatLoc);
        return mlir::success();
    }

private:
    // Result of input-branch matching.
    //   ddrBranchSources    : DDR sources in concat order (left DDR block args plus, when the last
    //                         input resolves to a DDR value, the DDR right branch appended at end).
    //   cmxRightBranch      : non-null if the last input resolves to a SEGMENTED DistributedBuffer.
    //   rightCopiesToRemove : intermediate CopyOps between the last concat input and its source.
    struct InputBranchesResult {
        SmallVector<mlir::Value> ddrBranchSources;
        mlir::Value cmxRightBranch;
        SmallVector<mlir::Operation*> rightCopiesToRemove;
    };

    // Matches every concat input. Left inputs [0..N-2] must each be a DDR CopyOp from a
    // BlockArgument; the last input resolves either to a SEGMENTED CMX buffer (preferred) or to a
    // DDR value reachable via a single CopyOp (fallback, enabling all-DDR N-branch patterns).
    mlir::FailureOr<InputBranchesResult> validateInputBranches(VPUIP::ConcatViewOp concatOp, size_t numInputs,
                                                               Logger log) const {
        InputBranchesResult result;
        result.ddrBranchSources.reserve(numInputs);
        for (size_t i = 0; i + 1 < numInputs; ++i) {
            auto blockArg = matchLeftDDRBlockArg(concatOp.getInputs()[i], i, log);
            if (mlir::failed(blockArg)) {
                return mlir::failure();
            }
            result.ddrBranchSources.push_back(blockArg.value());
        }
        auto lastInput = concatOp.getInputs()[numInputs - 1];
        if (auto cmx = tryMatchCmxSegmentedRight(lastInput, result.rightCopiesToRemove); cmx != nullptr) {
            result.cmxRightBranch = cmx;
            return result;
        }
        result.rightCopiesToRemove.clear();
        if (auto ddr = tryMatchDDRRight(lastInput, result.rightCopiesToRemove); ddr != nullptr) {
            result.ddrBranchSources.push_back(ddr);
            return result;
        }
        log.trace("Cannot find right branch input (neither SEGMENTED CMX nor DDR via CopyOp)");
        return mlir::failure();
    }

    // Left concat input must be a DDR CopyOp reading from a BlockArgument.
    mlir::FailureOr<mlir::Value> matchLeftDDRBlockArg(mlir::Value input, size_t idx, Logger log) const {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        if (copyOp == nullptr) {
            log.trace("Left input {0} is not a CopyOp result", idx);
            return mlir::failure();
        }
        auto blockArg = copyOp.getInput();
        if (!mlir::isa<mlir::BlockArgument>(blockArg)) {
            log.trace("Left input {0} source is not a block argument", idx);
            return mlir::failure();
        }
        if (mlir::cast<vpux::NDTypeInterface>(blockArg.getType()).getMemoryKind() != VPU::MemoryKind::DDR) {
            log.trace("Left input {0} block argument is not DDR", idx);
            return mlir::failure();
        }
        return blockArg;
    }

    // Walks the last input up to `kMaxRightCopyChainDepth` CopyOp/ConvertDMAOp defs, looking for a
    // SEGMENTED DistributedBuffer producer. Returns that buffer on success (intermediate ops
    // collected in `copiesToRemove`); returns nullptr otherwise.
    mlir::Value tryMatchCmxSegmentedRight(mlir::Value input, SmallVector<mlir::Operation*>& copiesToRemove) const {
        constexpr int kMaxRightCopyChainDepth = 3;
        mlir::Value cur = input;
        for (int depth = 0; depth < kMaxRightCopyChainDepth; ++depth) {
            if (cur == nullptr || mlir::isa<mlir::BlockArgument>(cur)) {
                return nullptr;
            }
            if (mlir::isa<VPUIP::DistributedBufferType>(cur.getType())) {
                break;
            }
            auto* defOp = cur.getDefiningOp();
            if (defOp == nullptr || !mlir::isa<VPUIP::CopyOp, VPUIP::ConvertDMAOp>(defOp)) {
                return nullptr;
            }
            copiesToRemove.push_back(defOp);
            cur = defOp->getOperand(0);
        }
        auto distType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(cur.getType());
        if (distType == nullptr) {
            return nullptr;
        }
        auto dist = distType.getDistribution();
        auto mode = dist ? dist.getMode() : nullptr;
        if (mode == nullptr || mode.getValue() != VPU::DistributionMode::SEGMENTED) {
            return nullptr;
        }
        return cur;
    }

    // Fallback right-branch matcher: accepts any DDR value reached through a single CopyOp.
    mlir::Value tryMatchDDRRight(mlir::Value input, SmallVector<mlir::Operation*>& copiesToRemove) const {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        if (copyOp == nullptr) {
            return nullptr;
        }
        auto ddrSrc = copyOp.getInput();
        if (mlir::cast<vpux::NDTypeInterface>(ddrSrc.getType()).getMemoryKind() != VPU::MemoryKind::DDR) {
            return nullptr;
        }
        copiesToRemove.push_back(copyOp);
        return ddrSrc;
    }

    // Verifies the SameAxis invariant:
    //   - every SubView slices only on newConcatDim and fits within one concat cycle,
    //   - every SubView starts within the DDR region (not the CMX-right region, when it exists),
    //   - every CMX copy destination is a SEGMENTED-like distributed buffer tiled on newConcatDim.
    mlir::LogicalResult validateSubViewConsumers(ArrayRef<VPUIP::SubViewOp> subViews, ArrayRef<VPUIP::CopyOp> cmxCopies,
                                                 VPUIP::PermuteCastOp permuteCastOp, Dim newConcatDim,
                                                 int64_t origConcatDimSize, int64_t ddrRowsTotal, Logger log) const {
        auto permOutShape = getShape(permuteCastOp->getResult(0));
        for (size_t i = 0; i < subViews.size(); ++i) {
            VPUIP::SubViewOp viewOp = subViews[i];
            auto viewShape = Shape(parseIntArrayAttr<int64_t>(viewOp.getStaticSizes()));
            for (size_t d = 0; d < permOutShape.size(); ++d) {
                if (Dim(d) != newConcatDim && viewShape[Dim(d)] != permOutShape[Dim(d)]) {
                    log.trace("SubView {0} slices on dim {1} != newConcatDim {2} — not supported", i, d,
                              newConcatDim.ind());
                    return mlir::failure();
                }
            }
            auto viewOffset = Shape(parseIntArrayAttr<int64_t>(viewOp.getStaticOffsets()));
            const int64_t off = viewOffset[newConcatDim];
            const int64_t sz = viewShape[newConcatDim];
            const int64_t withinGroup = off % origConcatDimSize;
            if ((off + sz - 1) / origConcatDimSize != off / origConcatDimSize) {
                log.trace("SubView {0} spans two concat cycles - not supported", i);
                return mlir::failure();
            }
            if (withinGroup >= ddrRowsTotal) {
                log.trace("SubView {0} starts in the CMX-right region - not supported", i);
                return mlir::failure();
            }
        }
        for (size_t i = 0; i < cmxCopies.size(); ++i) {
            VPUIP::CopyOp copyOp = cmxCopies[i];
            auto cmxBufType = mlir::dyn_cast<VPUIP::DistributedBufferType>(copyOp.getOutputBuff().getType());
            if (cmxBufType == nullptr) {
                log.trace("CMX copy {0} destination is not a DistributedBufferType", i);
                return mlir::failure();
            }
            auto distInfo = VPU::DistributionInfo::getClassFromAttr(cmxBufType.getDistribution());
            if (!VPU::isSegmentedLikeDistributionMode(mlir::cast<NDTypeInterface>(copyOp.getOutputBuff().getType()),
                                                      distInfo)) {
                log.trace("CMX copy {0} destination is not SEGMENTED-like", i);
                return mlir::failure();
            }
            const int64_t tilingAxis = VPU::getDistributedTilingAxis(distInfo.getNumTiles());
            if (tilingAxis != newConcatDim.ind()) {
                log.trace("CMX copy {0} tiling axis {1} != newConcatDim {2} — not supported for SameAxis pattern", i,
                          tilingAxis, newConcatDim.ind());
                return mlir::failure();
            }
        }
        return mlir::success();
    }

    // Per-consumer rewrite invariants.
    struct SameAxisRewriteContext {
        Dim newConcatDim;
        int64_t origConcatDimSize;
        ArrayRef<int64_t> ddrBranchCumSizes;  // size = preparedDDRBranches.size() + 1
        ArrayRef<mlir::Value> preparedDDRBranches;
        mlir::Value cmxRightBranch;  // nullable — SEGMENTED CMX right branch, if any
        int64_t cmxRightSize;        // valid only when cmxRightBranch is non-null
        VPUIP::PermuteCastOp permuteCastOp;
    };

    // Rewrites one SubView→CMXCopy consumer: emits per-cluster-boundary copy operations directly
    // into the CMX buffer via ExtractFlatSliceOp, then assembles them via ConcatViewOp.
    void rewriteOneConsumer(mlir::PatternRewriter& rewriter, mlir::Location baseLoc, size_t consumerIdx,
                            VPUIP::SubViewOp subViewOp, VPUIP::CopyOp cmxCopyOp, ArrayRef<int64_t> clusterCumOffsets,
                            const SameAxisRewriteContext& ctx) const {
        rewriter.setInsertionPoint(cmxCopyOp);
        const auto viewOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        const auto viewShape = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()));
        const int64_t off = viewOffset[ctx.newConcatDim];
        const int64_t sz = viewShape[ctx.newConcatDim];
        const int64_t viewMul = off / ctx.origConcatDimSize;
        const int64_t withinGroup = off % ctx.origConcatDimSize;
        auto cmxBuf = cmxCopyOp.getOutputBuff();
        SmallVector<mlir::Value> allCopies;
        allCopies.reserve(ctx.preparedDDRBranches.size() * clusterCumOffsets.size());

        // Every DDR branch (left + optional DDR right) uses the same source-row formula.
        for (size_t i = 0; i < ctx.preparedDDRBranches.size(); ++i) {
            emitDDRBranchCopies(rewriter, baseLoc, consumerIdx, i, ctx, viewMul, withinGroup, sz, viewShape,
                                clusterCumOffsets, cmxBuf, allCopies);
        }

        // Optional CMX SEGMENTED right branch — occupies the tail of each C-group cycle.
        if (ctx.cmxRightBranch != nullptr && withinGroup + sz > ctx.ddrBranchCumSizes.back()) {
            emitCMXRightBranchCopies(rewriter, baseLoc, consumerIdx, ctx, viewMul, withinGroup, sz, viewShape,
                                     clusterCumOffsets, cmxBuf, allCopies);
        }

        VPUX_THROW_UNLESS(!allCopies.empty(), "No copies generated for SubView consumer {0}", consumerIdx);
        auto newConcatView = rewriter.create<VPUIP::ConcatViewOp>(appendLoc(baseLoc, "cmx_concat_{0}", consumerIdx),
                                                                  allCopies, cmxBuf);
        rewriter.replaceAllUsesWith(cmxCopyOp.getResult(), newConcatView.getResult());
        rewriter.eraseOp(cmxCopyOp);
        rewriter.eraseOp(subViewOp);
    }

    // Emit per-cluster DDR→CMX copies for one DDR branch's overlap with the consumer's SubView.
    void emitDDRBranchCopies(mlir::PatternRewriter& rewriter, mlir::Location baseLoc, size_t consumerIdx,
                             size_t branchIdx, const SameAxisRewriteContext& ctx, int64_t viewMul, int64_t withinGroup,
                             int64_t sz, ShapeRef viewShape, ArrayRef<int64_t> clusterCumOffsets, mlir::Value cmxBuf,
                             SmallVector<mlir::Value>& outCopies) const {
        const int64_t branchStart = ctx.ddrBranchCumSizes[branchIdx];
        const int64_t branchEnd = ctx.ddrBranchCumSizes[branchIdx + 1];
        const int64_t branchH = branchEnd - branchStart;
        const int64_t overlapStart = std::max(withinGroup, branchStart);
        const int64_t overlapEnd = std::min(withinGroup + sz, branchEnd);
        if (overlapStart >= overlapEnd) {
            return;
        }
        const int64_t overlapSize = overlapEnd - overlapStart;
        // The viewMul-th C-group row-block starts at viewMul*branchH; shift by (overlapStart - branchStart).
        const int64_t srcRowBase = viewMul * branchH + (overlapStart - branchStart);
        const int64_t dstFlatBase = overlapStart - withinGroup;
        for (auto [subFlatOff, subLen] : splitByCluster(clusterCumOffsets, dstFlatBase, overlapSize)) {
            const int64_t subSrcOff = srcRowBase + (subFlatOff - dstFlatBase);
            Shape srcOff(SmallVector<int64_t>(viewShape.size(), 0));
            srcOff[ctx.newConcatDim] = subSrcOff;
            Shape srcSize(to_small_vector(viewShape.raw()));
            srcSize[ctx.newConcatDim] = subLen;
            auto srcView = rewriter.createOrFold<VPUIP::SubViewOp>(
                    appendLoc(baseLoc, "branch_{0}_src_{1}_{2}", branchIdx, consumerIdx, subFlatOff),
                    ctx.preparedDDRBranches[branchIdx], srcOff, srcSize);
            auto dstView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                    appendLoc(baseLoc, "branch_{0}_dst_{1}_{2}", branchIdx, consumerIdx, subFlatOff), cmxBuf,
                    subFlatOff, subLen);
            outCopies.push_back(rewriter.create<VPUIP::CopyOp>(
                    appendLoc(baseLoc, "branch_{0}_copy_{1}_{2}", branchIdx, consumerIdx, subFlatOff), srcView,
                    dstView));
        }
    }

    // Emit CMX-right-branch copies for one consumer: extract the viewMul-th cluster slice, apply
    // reshape+permcast to match the flat CMX layout, then write cluster-aligned sub-ranges.
    void emitCMXRightBranchCopies(mlir::PatternRewriter& rewriter, mlir::Location baseLoc, size_t consumerIdx,
                                  const SameAxisRewriteContext& ctx, int64_t viewMul, int64_t withinGroup, int64_t sz,
                                  ShapeRef viewShape, ArrayRef<int64_t> clusterCumOffsets, mlir::Value cmxBuf,
                                  SmallVector<mlir::Value>& outCopies) const {
        auto rightSlice = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                appendLoc(baseLoc, "right_src_slice_{0}", consumerIdx), ctx.cmxRightBranch, viewMul);
        auto rightSliceShape = mlir::cast<vpux::NDTypeInterface>(rightSlice.getType()).getShape();
        // [1, sliceC, 1, W] → [sliceC, W, 1, 1]
        Shape rightReshapeShape{rightSliceShape[Dim(1)] * rightSliceShape[Dim(2)], rightSliceShape[Dim(3)], 1, 1};
        auto rightPermCast = applyReshapeAndPermuteCast(rewriter, rightSlice, rightReshapeShape, ctx.permuteCastOp,
                                                        appendLoc(baseLoc, "right_reshape_{0}", consumerIdx),
                                                        appendLoc(baseLoc, "right_permcast_{0}", consumerIdx));
        const int64_t ddrRowsTotal = ctx.ddrBranchCumSizes.back();
        const int64_t dstFlatBase = ddrRowsTotal - withinGroup;
        // Consumer covers at most one C-cycle (validated), so the actual overlap with the CMX right
        // branch is `overlapSize` — not necessarily the whole `cmxRightSize`.
        const int64_t overlapSize = std::min(withinGroup + sz, ddrRowsTotal + ctx.cmxRightSize) - ddrRowsTotal;
        for (auto [subFlatOff, subLen] : splitByCluster(clusterCumOffsets, dstFlatBase, overlapSize)) {
            mlir::Value srcVal = rightPermCast;
            if (subLen != ctx.cmxRightSize) {
                Shape srcOff(SmallVector<int64_t>(viewShape.size(), 0));
                srcOff[ctx.newConcatDim] = subFlatOff - dstFlatBase;
                Shape srcSize(to_small_vector(viewShape.raw()));
                srcSize[ctx.newConcatDim] = subLen;
                srcVal = rewriter.createOrFold<VPUIP::SubViewOp>(
                        appendLoc(baseLoc, "right_src_sub_{0}_{1}", consumerIdx, subFlatOff), rightPermCast, srcOff,
                        srcSize);
            }
            auto dstView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                    appendLoc(baseLoc, "right_dst_{0}_{1}", consumerIdx, subFlatOff), cmxBuf, subFlatOff, subLen);
            outCopies.push_back(rewriter.create<VPUIP::CopyOp>(
                    appendLoc(baseLoc, "right_copy_{0}_{1}", consumerIdx, subFlatOff), srcVal, dstView));
        }
    }

    // Computes per-cluster cumulative offsets along the tiling axis for a SEGMENTED CMX buffer.
    // Returns the offsets vector, or std::nullopt if the distribution cannot be computed.
    std::optional<SmallVector<int64_t>> buildClusterCumOffsets(mlir::Value cmxBuf, Logger log) const {
        auto cmxBufType = mlir::dyn_cast<VPUIP::DistributedBufferType>(cmxBuf.getType());
        if (cmxBufType == nullptr) {
            log.trace("CMX consumer buffer is not a DistributedBufferType");
            return std::nullopt;
        }
        auto distInfo = VPU::DistributionInfo::getClassFromAttr(cmxBufType.getDistribution());
        const Dim axis(VPU::getDistributedTilingAxis(distInfo.getNumTiles()));

        const auto memOffsets = cmxBufType.getPerClusterMemoryShapeOffsets();
        const auto memShapes = cmxBufType.getPerClusterMemoryShapes();
        VPUX_THROW_UNLESS(memOffsets.size() == memShapes.size(),
                          "Per-cluster memory shapes and offsets size mismatch: {0} vs {1}", memShapes.size(),
                          memOffsets.size());

        SmallVector<int64_t> boundaries;
        boundaries.reserve(memShapes.size() + 1);
        boundaries.push_back(0);
        for (size_t c = 0; c < memShapes.size(); ++c) {
            const int64_t end = memOffsets[c][axis] + memShapes[c][axis];
            if (end <= boundaries.back()) {
                log.trace("Per-cluster memory boundaries are not strictly increasing on axis {0}", axis.ind());
                return std::nullopt;
            }
            boundaries.push_back(end);
        }
        return boundaries;
    }

    // Splits flat range [flatOff, flatOff+len) into sub-ranges each contained within a single cluster.
    SmallVector<std::pair<int64_t, int64_t>> splitByCluster(ArrayRef<int64_t> clusterCumOffsets, int64_t flatOff,
                                                            int64_t len) const {
        SmallVector<std::pair<int64_t, int64_t>> subRanges;
        int64_t pos = flatOff;
        while (pos < flatOff + len) {
            auto it = std::upper_bound(clusterCumOffsets.begin(), clusterCumOffsets.end(), pos);
            VPUX_THROW_UNLESS(it != clusterCumOffsets.end(), "Flat offset {0} is out of CMX buffer bounds (total {1})",
                              pos, clusterCumOffsets.back());
            const int64_t clusterEnd = *it;
            const int64_t subLen = std::min(flatOff + len - pos, clusterEnd - pos);
            subRanges.push_back({pos, subLen});
            pos += subLen;
        }
        return subRanges;
    }

    bool checkSameAxisReshapeCompatibility(VPUIP::ConcatViewOp concatOp, VPUIP::GenericReshapeOp genReshapeOp,
                                           vpux::Logger log) const {
        auto genReshapeType = vpux::getBufferType(genReshapeOp.getOutput());
        auto concatType = vpux::getBufferType(concatOp.getOutput());
        if (genReshapeType.getRank() != 4 || concatType.getRank() != 4) {
            log.trace("Only 4D tensors are supported");
            return false;
        }
        if (concatType.getShape()[Dim(0)] != 1) {
            log.trace("Only batch size 1 is supported");
            return false;
        }
        // [1, C, H, W] → [C*H, W, 1, 1]
        auto rs = genReshapeType.getShape();
        auto cs = concatType.getShape();
        if (rs[Dim(0)] != cs[Dim(1)] * cs[Dim(2)] || rs[Dim(1)] != cs[Dim(3)] ||
            genReshapeType.getNumElements() != concatType.getNumElements()) {
            log.trace("Concat→Reshape shapes are not compatible: {0} vs {1}", cs, rs);
            return false;
        }
        return true;
    }
};

/*
    Multi-left unbalanced DDR Concat on the W axis (dim3) or H axis (dim2) (N ≥ 3 DDR branches):

    W-axis example:
    Branch0[1,C,H,W0]@DDR ──Copy──┐
    Branch1[1,C,H,W1]@DDR ──Copy──┤
    ...                            ├─▶ ConcatView[1,C,H,W_total]@DDR
    BranchN[1,C,H,WN]@DDR ──Copy──┘         │
                                       GenericReshape (4D or 5D)
                                       [C*H, W_total, 1, 1]  — 4D case
                                       [C,   H, W_total, 1, 1] — 5D case
                                             │
                                        PermuteCast
                                          ┌──┤──┐
                                   SubView0   ...  SubViewM
                                      │               │
                               CMX SEGMENTED Copy   (same)

    H-axis example:
    Branch0[1,C,H0,W]@DDR ──Copy──┐
    Branch1[1,C,H1,W]@DDR ──Copy──┤
    ...                            ├─▶ ConcatView[1,C,H_total,W]@DDR
    BranchN[1,C,HN,W]@DDR ──Copy──┘         │
                                       GenericReshape (5D only)
                                       [C, H_total, W, 1, 1]
                                             │
                                        PermuteCast
                                          ┌──┤──┐
                                   SubView0   ...  SubViewM
                                      │               │
                               CMX SEGMENTED Copy (segmented on dim0)

    Supported reshape patterns (origConcatDim → newConcatDim, rowDim):
      W-axis, 4D: [1,C,H,W] → [C*H, W_total, 1, 1]    newConcatDim=Dim(1), rowDim=Dim(0)
      W-axis, 5D: [1,C,H,W] → [C,   H, W_total, 1, 1]  newConcatDim=Dim(2), rowDim=Dim(1)
      H-axis, 5D: [1,C,H,W] → [C,   H_total, W, 1, 1]  newConcatDim=Dim(1), rowDim=Dim(0)
    H-axis with 4D reshape folds H into dim0, making tiling and concat on the same axis;
    that case is handled by SplitMultiLeftUnbalancedDDRConcatOnSameAxis.
    In all supported cases rowDim ≠ newConcatDim, so tiling on rowDim is compatible with
    slicing the SEGMENTED CMX buffer on newConcatDim.

    Transformed to – per consumer SubView [row_off, col_off] × [row_sz, col_sz]:

      For each branch[i] whose concat-dim range overlaps the SubView's concat-dim range:
        preparedBranch[i] (GenericReshape+PermuteCast of the DDR source)
          → SubView selecting [row_sz, overlapLen] at [rowOff, srcColOff]
          → Copy directly into the SubView of the SEGMENTED CMX buffer on newConcatDim
            (legal because tiling is on rowDim ≠ newConcatDim)

      ConcatViewOp(allCopies, cmxBuf) replaces original SubView→CMXCopy chain.

    Eliminates the large DDR concat buffer; all DMAs go directly DDR→CMX, enabling
    interleaving with NCE tasks.
*/
class SplitMultiLeftUnbalancedDDRConcatOnOtherAxis : public SplitMultiUnbalancedDDRConcatBase {
public:
    using SplitMultiUnbalancedDDRConcatBase::SplitMultiUnbalancedDDRConcatBase;

    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const override {
        _log.trace("SplitMultiLeftUnbalancedDDRConcatOnOtherAxis: Got Concat at '{0}'", concatOp.getLoc());
        auto nestedLog = _log.nest();

        // Match: numInputs ≥ 3, DDR output, GenericReshape → PermuteCast chain.
        // The 2-input case is already handled by SplitUnbalancedDDRConcatOnOtherAxis /
        // SplitUnbalancedDDRConcatOnSameAxis.
        const size_t numInputs = concatOp.getInputs().size();
        auto prefixResult = matchConcatReshapePermuteCastChain(concatOp, /*minInputs=*/3, nestedLog);
        if (mlir::failed(prefixResult)) {
            return mlir::failure();
        }
        auto [genReshapeOp, permuteCastOp] = prefixResult.value();

        // Validate Concat→Reshape shapes: [1,C,H,W] → [C*H,W,1,1] (4D) or [C,H,W,1,1] (5D).
        if (!checkOtherAxisReshapeCompatibility(concatOp, genReshapeOp, nestedLog)) {
            return mlir::failure();
        }

        // Determine concat axis: dim2 (H) or dim3 (W).
        const auto concatAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
        if (concatAxes.size() != 1 || (concatAxes.front() != Dim(2) && concatAxes.front() != Dim(3))) {
            nestedLog.trace("Only concat on dim2 (H) or dim3 (W) is supported");
            return mlir::failure();
        }
        const Dim origConcatDim = concatAxes.front();
        const int64_t reshapeRank = vpux::getBufferType(genReshapeOp.getOutput()).getRank();
        // H-axis concat with 4D reshape folds H into dim0, yielding tiling and concat on the same axis.
        // That case is handled by SplitMultiLeftUnbalancedDDRConcatOnSameAxis.
        if (origConcatDim == Dim(2) && reshapeRank == 4) {
            nestedLog.trace("H-axis concat with 4D reshape is handled by SplitMultiLeftUnbalancedDDRConcatOnSameAxis");
            return mlir::failure();
        }
        // Derive newConcatDim (position of origConcatDim in the reshaped output) and
        // rowDim (the SubView row dimension, orthogonal to newConcatDim) from origConcatDim and reshapeRank:
        //   W-axis, 4D [1,C,H,W]→[C*H,W,1,1]:   W→dim1, rows on dim0
        //   W-axis, 5D [1,C,H,W]→[C,H,W,1,1]:   W→dim2, rows on dim1
        //   H-axis, 5D [1,C,H,W]→[C,H,W,1,1]:   H→dim1, rows on dim0
        Dim newConcatDim;
        Dim rowDim;
        if (origConcatDim == Dim(3)) {
            newConcatDim = (reshapeRank == 5) ? Dim(2) : Dim(1);
            rowDim = (reshapeRank == 5) ? Dim(1) : Dim(0);
        } else {
            // origConcatDim == Dim(2), reshapeRank == 5 guaranteed by the check above
            newConcatDim = Dim(1);
            rowDim = Dim(0);
        }

        // Validate all inputs and collect DDR sources.
        auto ddrSrcsResult = validateInputBranches(concatOp, numInputs, nestedLog);
        if (mlir::failed(ddrSrcsResult)) {
            return mlir::failure();
        }
        const auto& allDDRSrcs = ddrSrcsResult.value();

        // Collect PermuteCast users: SubViewOp → CopyOp pairs going to CMX.
        SmallVector<VPUIP::SubViewOp> subViews;
        SmallVector<VPUIP::CopyOp> cmxCopies;
        if (!collectSubViewCopyConsumers(permuteCastOp, subViews, cmxCopies, nestedLog)) {
            return mlir::failure();
        }

        // Compute cumulative sizes along origConcatDim for all branches.
        // colCumSizes[i] = sum of origConcatDim extents of allDDRSrcs[0..i-1].
        SmallVector<int64_t> colCumSizes;
        colCumSizes.push_back(0);
        for (auto ddrSrc : allDDRSrcs) {
            auto argShape = mlir::cast<vpux::NDTypeInterface>(ddrSrc.getType()).getShape();
            colCumSizes.push_back(colCumSizes.back() + argShape[origConcatDim]);
        }
        const int64_t totalW = getShape(concatOp.getResult())[origConcatDim];
        VPUX_THROW_UNLESS(colCumSizes.back() == totalW, "Total concat size {0} does not match sum of input sizes {1}",
                          totalW, colCumSizes.back());

        if (mlir::failed(validateConsumers(subViews, cmxCopies, newConcatDim, totalW, nestedLog))) {
            return mlir::failure();
        }

        // Prepare each branch: GenericReshape → PermuteCast applied to the DDR source.
        // W-axis: [1,C,H,Wi] → reshape [C*H,Wi,1,1] (4D) or [C,H,Wi,1,1] (5D) → permcast.
        // H-axis: [1,C,Hi,W] → reshape [C,Hi,W,1,1] (5D) → permcast.
        SmallVector<mlir::Value> preparedBranches;
        for (size_t i = 0; i < allDDRSrcs.size(); ++i) {
            auto branchShape = mlir::cast<vpux::NDTypeInterface>(allDDRSrcs[i].getType()).getShape();
            // 4D (W-axis): [1,C,H,Wi]→[C*H,Wi,1,1]; 5D (W-axis): [1,C,H,Wi]→[C,H,Wi,1,1]; 5D (H-axis):
            // [1,C,Hi,W]→[C,Hi,W,1,1]
            Shape newShape = (reshapeRank == 5)
                                     ? Shape{branchShape[Dim(1)], branchShape[Dim(2)], branchShape[Dim(3)], 1, 1}
                                     : Shape{branchShape[Dim(1)] * branchShape[Dim(2)], branchShape[Dim(3)], 1, 1};
            preparedBranches.push_back(
                    applyReshapeAndPermuteCast(rewriter, allDDRSrcs[i], newShape, permuteCastOp,
                                               appendLoc(concatOp->getLoc(), "multi_other_reshape_{0}", i),
                                               appendLoc(concatOp->getLoc(), "multi_other_permcast_{0}", i)));
        }

        // Rewrite each SubView→CMXCopy consumer. Tiling is on rowDim (≠ newConcatDim), so slicing
        // the SEGMENTED CMX buffer on newConcatDim is legal; the DMA distributes rows automatically.
        const OtherAxisRewriteContext rewriteCtx{newConcatDim, rowDim, totalW, colCumSizes, preparedBranches};
        for (size_t consumerIdx = 0; consumerIdx < subViews.size(); ++consumerIdx) {
            rewriteOneConsumer(rewriter, concatOp->getLoc(), consumerIdx, subViews[consumerIdx], cmxCopies[consumerIdx],
                               rewriteCtx);
        }

        // Collect input CopyOps before erasing concatOp to avoid use-after-erase.
        SmallVector<mlir::Operation*> inputCopiesToRemove;
        for (size_t i = 0; i < numInputs; ++i) {
            if (auto* defOp = concatOp.getInputs()[i].getDefiningOp()) {
                inputCopiesToRemove.push_back(defOp);
            }
        }

        // Erase the concat/reshape/permute chain, then any now-dead input CopyOps.
        rewriter.eraseOp(permuteCastOp);
        rewriter.eraseOp(genReshapeOp);
        rewriter.eraseOp(concatOp);
        eraseDeadOps(rewriter, inputCopiesToRemove);

        nestedLog.trace("Successfully rewrote multi-branch DDR Concat (origConcatDim={0}) at '{1}'",
                        origConcatDim.ind(), concatOp.getLoc());
        return mlir::success();
    }

private:
    // Validates OtherAxis inputs: every concat input must resolve to a DDR source, either via a
    // CopyOp (preferred) or directly as a DDR value (fallback for non-CopyOp DDR producers).
    // Returns the collected DDR source values on success.
    mlir::FailureOr<SmallVector<mlir::Value>> validateInputBranches(VPUIP::ConcatViewOp concatOp, size_t numInputs,
                                                                    Logger log) const {
        SmallVector<mlir::Value> allDDRSrcs;
        allDDRSrcs.reserve(numInputs);
        for (size_t i = 0; i < numInputs; ++i) {
            auto copyOp = concatOp.getInputs()[i].getDefiningOp<VPUIP::CopyOp>();
            if (copyOp != nullptr) {
                auto ddrSrc = copyOp.getInput();
                if (mlir::cast<vpux::NDTypeInterface>(ddrSrc.getType()).getMemoryKind() != VPU::MemoryKind::DDR) {
                    log.trace("Input {0} source is not in DDR", i);
                    return mlir::failure();
                }
                allDDRSrcs.push_back(ddrSrc);
                continue;
            }
            // Fallback: accept a direct DDR value without a CopyOp wrapper.
            auto directVal = concatOp.getInputs()[i];
            if (mlir::cast<vpux::NDTypeInterface>(directVal.getType()).getMemoryKind() == VPU::MemoryKind::DDR) {
                log.trace("Input {0} is a direct DDR value (no CopyOp wrapper)", i);
                allDDRSrcs.push_back(directVal);
                continue;
            }
            log.trace("Input {0} is neither a DDR CopyOp result nor a direct DDR value", i);
            return mlir::failure();
        }
        return allDDRSrcs;
    }

    // Verifies the OtherAxis invariant: every SubView's column slice fits within one totalW cycle,
    // and every CMX copy is tiled on an axis distinct from newConcatDim (and all on the same axis).
    // Unlike the SameAxis pattern, any SEGMENTED-like tiling mode is acceptable here because the
    // CMX SubView on newConcatDim is independent of the row-tiling axis; the DMA engine handles
    // row distribution automatically.
    mlir::LogicalResult validateConsumers(ArrayRef<VPUIP::SubViewOp> subViews, ArrayRef<VPUIP::CopyOp> cmxCopies,
                                          Dim newConcatDim, int64_t totalW, Logger log) const {
        for (VPUIP::SubViewOp viewOp : subViews) {
            auto viewOffset = Shape(parseIntArrayAttr<int64_t>(viewOp.getStaticOffsets()));
            auto viewShape = Shape(parseIntArrayAttr<int64_t>(viewOp.getStaticSizes()));
            const int64_t colOff = viewOffset[newConcatDim];
            const int64_t colSz = viewShape[newConcatDim];
            if ((colOff + colSz - 1) / totalW != colOff / totalW) {
                log.trace("SubView range on newConcatDim spans two concat cycles — not supported");
                return mlir::failure();
            }
        }
        int64_t commonTilingAxis = -1;
        for (size_t i = 0; i < cmxCopies.size(); ++i) {
            VPUIP::CopyOp copyOp = cmxCopies[i];
            auto cmxBufType = mlir::dyn_cast<VPUIP::DistributedBufferType>(copyOp.getOutputBuff().getType());
            if (cmxBufType == nullptr) {
                log.trace("Consumer {0}: destination is not a DistributedBufferType", i);
                return mlir::failure();
            }
            const int64_t tilingAxis = VPU::getDistributedTilingAxis(
                    VPU::DistributionInfo::getClassFromAttr(cmxBufType.getDistribution()).getNumTiles());
            if (tilingAxis == newConcatDim.ind()) {
                log.trace("Consumer {0}: tiling axis {1} matches newConcatDim {2} — not supported", i, tilingAxis,
                          newConcatDim.ind());
                return mlir::failure();
            }
            if (commonTilingAxis == -1) {
                commonTilingAxis = tilingAxis;
            } else if (tilingAxis != commonTilingAxis) {
                log.trace("Consumer {0}: tiling axis {1} differs from previous consumers' axis {2} — not supported", i,
                          tilingAxis, commonTilingAxis);
                return mlir::failure();
            }
        }
        return mlir::success();
    }

    // Holds the per-consumer invariants for the OtherAxis (column-SubView) rewrite.
    struct OtherAxisRewriteContext {
        Dim newConcatDim;
        Dim rowDim;
        int64_t totalW;
        ArrayRef<int64_t> colCumSizes;
        ArrayRef<mlir::Value> preparedBranches;
    };

    // Rewrites one SubView→CMXCopy consumer: for each overlapping DDR branch emits a DDR→CMX Copy
    // via column SubViews of the SEGMENTED CMX buffer, then assembles via ConcatViewOp.
    void rewriteOneConsumer(mlir::PatternRewriter& rewriter, mlir::Location baseLoc, size_t consumerIdx,
                            VPUIP::SubViewOp subViewOp, VPUIP::CopyOp cmxCopyOp,
                            const OtherAxisRewriteContext& ctx) const {
        rewriter.setInsertionPoint(cmxCopyOp);
        const auto viewOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        const auto viewShape = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()));
        const int64_t rowOff = viewOffset[ctx.rowDim];
        const int64_t rowSz = viewShape[ctx.rowDim];
        const int64_t colOff = viewOffset[ctx.newConcatDim];
        const int64_t colSz = viewShape[ctx.newConcatDim];
        const int64_t colWithinCycle = colOff % ctx.totalW;
        auto cmxBuf = cmxCopyOp.getOutputBuff();
        SmallVector<mlir::Value> allCopies;

        for (size_t i = 0; i < ctx.preparedBranches.size(); ++i) {
            const int64_t branchColStart = ctx.colCumSizes[i];
            const int64_t branchColEnd = ctx.colCumSizes[i + 1];

            const int64_t overlapColStart = std::max(colWithinCycle, branchColStart);
            const int64_t overlapColEnd = std::min(colWithinCycle + colSz, branchColEnd);
            if (overlapColStart >= overlapColEnd) {
                continue;
            }
            const int64_t overlapColLen = overlapColEnd - overlapColStart;
            const int64_t srcColOff = overlapColStart - branchColStart;
            const int64_t dstColOff = overlapColStart - colWithinCycle;

            Shape srcShape(to_small_vector(viewShape.raw()));
            srcShape[ctx.rowDim] = rowSz;
            srcShape[ctx.newConcatDim] = overlapColLen;
            Shape srcOff(to_small_vector(viewOffset.raw()));
            srcOff[ctx.rowDim] = rowOff;
            srcOff[ctx.newConcatDim] = srcColOff;
            auto srcView =
                    rewriter.createOrFold<VPUIP::SubViewOp>(appendLoc(baseLoc, "branch_{0}_src_{1}", i, consumerIdx),
                                                            ctx.preparedBranches[i], srcOff, srcShape);
            Shape dstOff(SmallVector<int64_t>(viewShape.size(), 0));
            dstOff[ctx.newConcatDim] = dstColOff;
            auto dstView = rewriter.createOrFold<VPUIP::SubViewOp>(
                    appendLoc(baseLoc, "branch_{0}_dst_{1}", i, consumerIdx), cmxBuf, dstOff, srcShape);
            allCopies.push_back(rewriter.create<VPUIP::CopyOp>(
                    appendLoc(baseLoc, "branch_{0}_copy_{1}", i, consumerIdx), srcView, dstView));
        }

        VPUX_THROW_UNLESS(!allCopies.empty(), "No copies generated for SubView consumer {0}", consumerIdx);
        auto newConcatView = rewriter.create<VPUIP::ConcatViewOp>(
                appendLoc(baseLoc, "cmx_other_concat_{0}", consumerIdx), allCopies, cmxBuf);
        rewriter.replaceAllUsesWith(cmxCopyOp.getResult(), newConcatView.getResult());
        rewriter.eraseOp(cmxCopyOp);
        rewriter.eraseOp(subViewOp);
    }

    bool checkOtherAxisReshapeCompatibility(VPUIP::ConcatViewOp concatOp, VPUIP::GenericReshapeOp genReshapeOp,
                                            vpux::Logger log) const {
        auto genReshapeType = vpux::getBufferType(genReshapeOp.getOutput());
        auto concatType = vpux::getBufferType(concatOp.getOutput());
        if (concatType.getRank() != 4) {
            log.trace("Concat output must be 4D, got rank {0}", concatType.getRank());
            return false;
        }
        if (concatType.getShape()[Dim(0)] != 1) {
            log.trace("Only batch size 1 is supported");
            return false;
        }
        auto rs = genReshapeType.getShape();
        auto cs = concatType.getShape();
        const int64_t reshapeRank = genReshapeType.getRank();
        if (reshapeRank == 4) {
            // Pattern: [1, C, H, W] → [C*H, W, 1, 1]
            if (rs[Dim(0)] != cs[Dim(1)] * cs[Dim(2)] || rs[Dim(1)] != cs[Dim(3)] || rs[Dim(2)] != 1 ||
                rs[Dim(3)] != 1 || genReshapeType.getNumElements() != concatType.getNumElements()) {
                log.trace("Concat→Reshape shapes are not compatible for 4D pattern: {0} vs {1}", cs, rs);
                return false;
            }
        } else if (reshapeRank == 5) {
            // Pattern: [1, C, H, W] → [C, H, W, 1, 1]
            if (rs[Dim(0)] != cs[Dim(1)] || rs[Dim(1)] != cs[Dim(2)] || rs[Dim(2)] != cs[Dim(3)] || rs[Dim(3)] != 1 ||
                rs[Dim(4)] != 1 || genReshapeType.getNumElements() != concatType.getNumElements()) {
                log.trace("Concat→Reshape shapes are not compatible for 5D pattern: {0} vs {1}", cs, rs);
                return false;
            }
        } else {
            log.trace("Unsupported reshape output rank {0}, expected 4 or 5", reshapeRank);
            return false;
        }
        return true;
    }
};

class SplitUnbalancedDDRConcatOnSameAxisDDR : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    StringRef getRewriterSuffix() const override {
        return "OnSameAxisDDR";
    }

    bool isValidSubview(SmallVector<VPUIP::SubViewOp>& subviews, Dim newConcatDim, int64_t leftConcatInputSize,
                        int64_t rightConcatInputSize) const override {
        for (auto subview : subviews) {
            auto axis = getSubviewAxis(subview);
            // SplitUnbalancedDDRConcatOnSameAxisDDR only support one axis now
            if (axis.size() != 1) {
                return false;
            }
        }

        return checkSubview(subviews, newConcatDim, leftConcatInputSize, rightConcatInputSize);
    }

    VPUIP::DistributedBufferType updateDistributedType(mlir::Value, mlir::Value, ShapeRef) const override {
        return nullptr;
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
                                   VPUIP::GenericReshapeOp genReshape, VPUIP::PermuteCastOp permuteCastOp,
                                   mlir::Location loc) const override {
        if (genReshape != nullptr) {
            return propagateReshapeCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
        }
        if (permuteCastOp == nullptr) {
            return rightBranchInput;
        }
        return propagatePermuteCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
    }

    // For this pattern we need to create tmp buffer in DDR
    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp viewOp, VPUIP::CopyOp,
                                      mlir::Location bufferLoc) const override {
        auto srcBufferType = viewOp.getType();
        return rewriter.create<mlir::memref::AllocOp>(bufferLoc, mlir::cast<mlir::MemRefType>(srcBufferType));
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::CopyOp distributedCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer =
                createNewConcatBuffer(rewriter, subViewOp, distributedCopy, takeOpLoc(origConcatOp, "buf_{0}", index));

        const auto newConcatDim = params.newConcatDim;
        const auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        const auto srcShape = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()));
        const auto origConcatDimSize = getShape(origConcatOp->getResult(0))[params.origConcatDim];

        auto createViewBranch = [&](mlir::Value src, int64_t origDimSize, int64_t dstOffsetVal, int64_t srcOffsetVal,
                                    StringRef locSuffix, bool hasCopy = true) -> mlir::Value {
            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[newConcatDim] = origDimSize;

            Shape newSrcOffset(srcOffset);
            newSrcOffset[newConcatDim] = srcOffsetVal;

            Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
            dstOffset[newConcatDim] = dstOffsetVal;

            if (hasCopy) {
                return createNewCopyBranch(rewriter, src, dstBuffer, copyShape, newSrcOffset, dstOffset,
                                           origConcatOp->getLoc(), locSuffix, index);
            } else {
                return rewriter.createOrFold<VPUIP::SubViewOp>(
                        appendLoc(origConcatOp->getLoc(), "{0}_src_view_{1}", locSuffix, index), src, newSrcOffset,
                        copyShape);
            }
        };

        const auto viewMultiplier = srcOffset[newConcatDim] / origConcatDimSize;
        const auto viewRemainder = srcOffset[newConcatDim] % origConcatDimSize;
        const auto leftBranchOffset = srcOffset[newConcatDim] - viewMultiplier * params.rightInputSize +
                                      params.leftViewOffset + viewMultiplier * params.leftViewOffset;
        const auto rightBranchOffset = viewMultiplier * params.rightInputSize;
        const auto newLeftSize = srcShape[newConcatDim] - params.rightInputSize;

        // see checkSubview
        VPUX_THROW_UNLESS(viewRemainder < params.leftConcatInputSize,
                          "Not supported case: Single SubView branch must be on the left branch");

        if (viewRemainder + srcShape[newConcatDim] > params.leftConcatInputSize) {
            // SubView crosses left branch and right branch
            auto newLeftViewBranch = createViewBranch(newLeftBranch, newLeftSize,
                                                      /*dstOffsetVal=*/0, leftBranchOffset, "left");
            auto newRightViewBranch =
                    createViewBranch(newRightBranch, params.rightInputSize, newLeftSize, rightBranchOffset, "right");

            SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
            auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(takeOpLoc(origConcatOp, "concat_{0}", index),
                                                                    concatInputs, dstBuffer);
            rewriter.replaceOp(subViewOp, newConcatOp);
        } else {
            // SubView on the left branch
            auto newLeftViewBranch = createViewBranch(newLeftBranch, srcShape[newConcatDim], /*dstOffsetVal=*/0,
                                                      leftBranchOffset, "left", false);
            rewriter.replaceOp(subViewOp, newLeftViewBranch);
        }
    }
};

//
// FuseGatherDMAWithSubViewCopy
//

/*
    Handles the case where ConvertParallelSlicesToGather has already converted parallel SliceOps
    into a GatherDMA.  The chain looks like this:

    Before:
        NCE_i  (CMX, DUPLICATED|SEGMENTED or SEGMENTED - on K only)
            |
        Copy  (CMX → SubView of alloc_concat, DDR)       ← one per NCE
              \   /
          ConcatViewOp  (DDR, shape 1xCxHxW, NHWC)
               |
          GenericReshape  (1xCx(H*W)x1)
               |
          PermuteCast  (mem_perm=[d1,d3,d0,d2])  → (H*W)xCx1x1
               |
          GenericReshape  (flat: numFlatRows x gatherRowSize)
               |
          GatherDMAOp  (gathers K rows → K x gatherRowSize, CMX)
               |
          GenericReshape  (K x gatherRowSize → 1 x K x gatherRowSize)
               |
          Copy  (CMX → SubView of downstream alloc, DDR strided)

    After — DUPLICATED inputs (K individual CMX→DDR copies, one per gathered row):
        For each row index r = indices[j]:
            NCE_inputIdx  (CMX, DUPLICATED or MULTICASTED)
                |
            SubView [0, channelOffset, h, w][1, gatherRowSize, 1, 1]
                |
            [GenericReshape + PermuteCast to match DDR dest dims]
                |
            Copy  (CMX → row-j slot of downstream alloc, DDR)

    After — SEGMENTED_ON_K inputs (each row may span multiple clusters):
        For each row index r = indices[j]:
            For each cluster piece p that covers channels [cursor, cursor+take):
                NCE_inputIdx  (CMX, SEGMENTED on C-axis)
                    |
                ExtractFlatSliceOp  offset=flatOffset, length=take
                    |           (flatOffset = localOffset + (inChOffsets[input] - inChOffsets[cluster0Start])
                    |            resolves the cluster-local channel offset within the distributed buffer)
                SubView [0, 0, h, w][1, take, 1, 1]
                    |
                [GenericReshape + PermuteCast to match DDR dest dims]
                    |
                Copy  (CMX cluster p → piece-p slot of row-j in DDR)
            ConcatViewOp  (assembles pieces into row-j slot of downstream alloc)
        Basically, SEGMENTED on K case is identical to the DUPLICATED one, except the fact that the slice on K
        is done through ExtractFlatSliceOp instead of Subview.

    Coordinate mapping:
        numFlatRows  = H * W * rowsPerSpatialPosition   (total rows in the flat 2-D view)
        rowsPerSpatialPosition = totalC / gatherRowSize  (rows per spatial position in the flat 2-D view)

        flat row r → spatialPosition = r / rowsPerSpatialPosition,
                 channelSliceIndex = r % rowsPerSpatialPosition
        absoluteChannelOffset      = channelSliceIndex * gatherRowSize   (absolute channel index in the concat)
        inputIdx/channelOffset are resolved from per-input C ranges of ConcatView inputs
        h = spatialPosition / W,  w = spatialPosition % W
*/

class FuseGatherDMAWithSubViewCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    FuseGatherDMAWithSubViewCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, /*benefit=*/2), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    struct CMXInputInfo {
        mlir::Value input;
        // cluster = -1 => mode is DUPLICATED, read from any cluster
        // cluster >= 0 => mode is SEGMENTED, read from the specified cluster
        int64_t cluster;
    };
    struct MatchedChain {
        // front: ops producing the GatherDMA data input, listed in source -> gather order.
        VPUIP::ConcatViewOp frontConcat;       // nullable: only the multi-copy concat variant
        SmallVector<mlir::Value> frontCopies;  // CMX->DDR copies to erase (empty for the CMX-direct variant)
        VPUIP::GenericReshapeOp frontReshapeBeforePermute;  // nullable
        VPUIP::PermuteCastOp frontPermuteCast;
        VPUIP::GenericReshapeOp frontReshapeFlat;
        VPUIP::CopyOp frontInputCopy;              // nullable: only the CMX-side copy-before-gather variant
        SmallVector<mlir::Value> frontCmxSources;  // raw CMX sources handed from matchFrontPattern to matchGather
        int64_t frontTotalChannels = 0;            // combined channel count handed from front to gather
        SmallVector<CMXInputInfo> frontCmxInputs;  // normalized per-row CMX sources (built in matchGather)
        SmallVector<int64_t> frontChannelOffsets;  // prefix-sum channel offsets across frontCmxInputs

        // gather: the pivot op and its indices operand.
        VPUIP::GatherDMAOp gatherDMA;
        VPUIP::CopyOp gatherIndicesCopy;
        SmallVector<int64_t> gatherRowIndices;

        // back: ops consuming the GatherDMA output.
        VPUIP::GenericReshapeOp backReshape;
        VPUIP::SubViewOp backDestSubView;

        // derived geometry.
        int64_t height = 0, width = 0, gatherRowSize = 0, numGatheredRows = 0, rowsPerSpatialPosition = 0;
        int64_t gatheredRowsDim = 0;
    };

    mlir::LogicalResult matchChain(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, MatchedChain& chain) const;
    // Matches the ops after the GatherDMA (gather consumers): GatherDMA -> backReshape -> Copy, plus the DDR dest.
    mlir::LogicalResult matchBackPattern(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                         MatchedChain& chain) const;
    // Matches the GatherDMA itself (addressing mode, beneficial size, indices and row indices) and derives the
    // gather geometry from the front CMX sources (per-row virtual inputs, channel offsets, row validation).
    mlir::LogicalResult matchGather(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, MatchedChain& chain) const;
    // Matches the structural ops before the GatherDMA (gather producers): the flat view chain and the raw CMX sources.
    mlir::LogicalResult matchFrontPattern(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                          MatchedChain& chain) const;
    // Erases the whole matched chain once it has been replaced, in back -> gather -> front order.
    void eraseMatchedChain(mlir::PatternRewriter& rewriter, MatchedChain& chain) const;
    std::optional<std::pair<int64_t, int64_t>> getInputByAbsoluteChannelOffset(ArrayRef<int64_t> inputChannelOffsets,
                                                                               int64_t absoluteChannelOffset) const;

    Logger _log;
};

std::optional<std::pair<int64_t, int64_t>> FuseGatherDMAWithSubViewCopy::getInputByAbsoluteChannelOffset(
        ArrayRef<int64_t> inputChannelOffsets, int64_t absoluteChannelOffset) const {
    if (inputChannelOffsets.size() < 2) {
        _log.nest().trace("getInputByAbsoluteChannelOffset: fewer than 2 offsets — no virtual inputs registered");
        return std::nullopt;
    }

    const auto numInputs = static_cast<int64_t>(inputChannelOffsets.size() - 1);
    for (int64_t i = 0; i < numInputs; ++i) {
        if (absoluteChannelOffset >= inputChannelOffsets[i] && absoluteChannelOffset < inputChannelOffsets[i + 1]) {
            const int64_t localOffset = absoluteChannelOffset - inputChannelOffsets[i];
            _log.nest().trace("getInputByAbsoluteChannelOffset: absoluteChannelOffset={0} -> "
                              "virtualInput={1} [range [{2}, {3})), localOffset={4}",
                              absoluteChannelOffset, i, inputChannelOffsets[i], inputChannelOffsets[i + 1],
                              localOffset);
            return std::make_pair(i, localOffset);
        }
    }

    _log.nest().trace("getInputByAbsoluteChannelOffset: absoluteChannelOffset={0} not covered by any "
                      "of {1} virtual inputs (total range [0, {2}))",
                      absoluteChannelOffset, numInputs, inputChannelOffsets.back());
    return std::nullopt;
}

mlir::LogicalResult FuseGatherDMAWithSubViewCopy::matchBackPattern(VPUIP::CopyOp copyOp,
                                                                   mlir::PatternRewriter& rewriter,
                                                                   MatchedChain& chain) const {
    //  Must be non-distributed CMX -> DDR.
    if (!VPUIP::isCopyToDDR(copyOp) || VPUIP::isCopyFromDDR(copyOp) ||
        mlir::isa<VPUIP::DistributedBufferType>(copyOp.getInput().getType())) {
        return matchFailed(_log.nest(), rewriter, copyOp, "CopyOp is not a non-distributed CMX->DDR copy");
    }

    // Back chain: GatherDMAOp -> GenericReshape -> Copy.
    auto backReshape = copyOp.getInput().getDefiningOp<VPUIP::GenericReshapeOp>();
    if (backReshape == nullptr || !backReshape.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "CopyOp input is not a single-use GenericReshapeOp (backReshape)");
    }
    auto gatherDMA = backReshape.getInput().getDefiningOp<VPUIP::GatherDMAOp>();
    if (gatherDMA == nullptr || !gatherDMA.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, copyOp, "backReshape input is not a single-use GatherDMAOp");
    }

    // Dest must be a SubView of an alloc (downstream ConcatView slot).
    auto backDestSubView = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
    if (!backDestSubView) {
        return matchFailed(_log.nest(), rewriter, copyOp, "CopyOp output buffer is not a SubViewOp");
    }
    if (backDestSubView.getSource().getDefiningOp<mlir::memref::AllocOp>() == nullptr) {
        return matchFailed(_log.nest(), rewriter, copyOp, "DDR destination SubView source is not a memref::AllocOp");
    }

    // The gathered-rows axis is the second-to-last dim of the DDR destination.
    auto ddrSubViewType = mlir::cast<vpux::NDTypeInterface>(backDestSubView.getResult().getType());
    const auto dstShape = ddrSubViewType.getShape();
    if (static_cast<int64_t>(dstShape.size()) < 2) {
        return matchFailed(_log.nest(), rewriter, copyOp, "DDR destination SubView has fewer than 2 dimensions");
    }

    chain.backReshape = backReshape;
    chain.gatherDMA = gatherDMA;
    chain.backDestSubView = backDestSubView;
    chain.gatheredRowsDim = static_cast<int64_t>(dstShape.size()) - 2;
    return mlir::success();
}

mlir::LogicalResult FuseGatherDMAWithSubViewCopy::matchGather(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                              MatchedChain& chain) const {
    auto gatherDMA = chain.gatherDMA;

    if (gatherDMA.getAddressingMode() != VPUIP::GatherAddressingMode::INDEXED) {
        return matchFailed(_log.nest(), rewriter, copyOp, "GatherDMAOp does not use INDEXED addressing mode");
    }

    // Skip if the gather output is larger than its input — the optimization would increase memory traffic.
    const auto gatherInputType = mlir::cast<vpux::NDTypeInterface>(gatherDMA.getInput().getType());
    const auto gatherOutputType = mlir::cast<vpux::NDTypeInterface>(gatherDMA.getOutput().getType());
    if (gatherOutputType.getShape().totalSize() >= gatherInputType.getShape().totalSize()) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "GatherDMA output is not smaller than input; optimization not beneficial");
    }

    // Gather indices: Const::DeclareOp -> Copy -> CMX alloc.
    auto gatherIndicesCopy = gatherDMA.getIndices().getDefiningOp<VPUIP::CopyOp>();
    if (!gatherIndicesCopy) {
        return matchFailed(_log.nest(), rewriter, copyOp, "GatherDMA indices are not sourced from a CopyOp");
    }
    auto declareOp = gatherIndicesCopy.getInput().getDefiningOp<Const::DeclareOp>();
    if (!declareOp) {
        return matchFailed(_log.nest(), rewriter, copyOp, "GatherDMA indices source is not a Const::DeclareOp");
    }

    // GatherDMA output: numGatheredRows x gatherRowSize. Both dims are owned by the gather here; the front and
    // back sides validate their own geometry against these values.
    const auto gatherOutShape = gatherOutputType.getShape();
    if (gatherOutShape.size() != 2) {
        return matchFailed(_log.nest(), rewriter, copyOp, "GatherDMA output is not 2D");
    }
    const int64_t numGatheredRows = gatherOutShape[Dim(0)];
    const int64_t gatherRowSize = gatherOutShape[Dim(1)];

    const auto dstShape = mlir::cast<vpux::NDTypeInterface>(chain.backDestSubView.getResult().getType()).getShape();
    if (dstShape[Dim(chain.gatheredRowsDim)] != numGatheredRows) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "DDR destination shape at gatheredRowsDim does not match numGatheredRows {0}",
                           numGatheredRows);
    }

    // Read gather indices.
    SmallVector<int64_t> rowIndices;
    const auto indicesContent = declareOp.getContent();
    for (auto idx : indicesContent.getValues<int64_t>()) {
        rowIndices.push_back(idx);
    }
    if (static_cast<int64_t>(rowIndices.size()) != numGatheredRows) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "Number of row indices {0} does not match numGatheredRows {1}", rowIndices.size(),
                           numGatheredRows);
    }

    // Resolve the front CMX sources (matched by matchFrontPattern) into per-row virtual inputs and validate that
    // every gathered row maps onto them. This step needs both the gather geometry (gatherRowSize, row indices) and
    // the front sources, so it lives here where all of that is available.
    const int64_t totalC = chain.frontTotalChannels;
    const auto& cmxSources = chain.frontCmxSources;

    const size_t numInputs = cmxSources.size();
    SmallVector<CMXInputInfo> cmxInputs;
    cmxInputs.reserve(numInputs);
    SmallVector<int64_t> inputChannelSizes;
    inputChannelSizes.reserve(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        const auto cmxVal = cmxSources[i];
        auto distType = mlir::dyn_cast<VPUIP::DistributedBufferType>(cmxVal.getType());
        if (!distType) {
            return matchFailed(_log.nest(), rewriter, copyOp, "CMX source {0} is not a DistributedBufferType", i);
        }
        const auto mode = distType.getDistribution().getMode().getValue();
        const auto isSegmentedOnK = VPU::isSegmentedOverC(distType.getDistribution());
        if (!isSegmentedOnK && !VPU::bitEnumContainsAll(mode, VPU::DistributionMode::DUPLICATED) &&
            !VPU::bitEnumContainsAll(mode, VPU::DistributionMode::MULTICASTED)) {
            return matchFailed(_log.nest(), rewriter, copyOp,
                               "CMX source {0} distribution mode is not DUPLICATED, MULTICASTED, or SEGMENTED on K", i);
        }

        if (!isSegmentedOnK) {
            const auto inputShape = getShape(cmxVal);
            const int64_t chSize = inputShape[Dims4D::Act::C];
            cmxInputs.push_back({cmxVal, -1});
            inputChannelSizes.push_back(chSize);
            _log.nest().trace("  cmxInputs[{0}]: DUPLICATED (cluster=-1), C={1}", cmxInputs.size() - 1, chSize);
            continue;
        }

        // ExtractFlatSliceOp requires compact SEGMENTED buffers with the tiling dim being the first dim > 1.
        const auto inputShape = distType.getShape();
        if (!StrideReqs::compact(inputShape.size()).checkStrides(distType)) {
            return matchFailed(
                    _log.nest(), rewriter, copyOp,
                    "CMX source {0} SEGMENTED_ON_K is not compact; cannot apply FuseGatherDMAWithSubViewCopy", i);
        }
        const auto tileIndex = VPUIP::getTilingDimIndex(distType);
        if (!tileIndex.has_value() || tileIndex.value() != Dims4D::Act::C.ind() || inputShape[Dims4D::Act::N] != 1) {
            return matchFailed(_log.nest(), rewriter, copyOp,
                               "CMX source {0} SEGMENTED_ON_K has unsupported tiling dim or N != 1; cannot use "
                               "ExtractFlatSliceOp",
                               i);
        }

        if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(distType.getElementType())) {
            return matchFailed(_log.nest(), rewriter, copyOp,
                               "CMX source {0} SEGMENTED_ON_K has per-axis quantization; cannot use "
                               "ExtractFlatSliceOp",
                               i);
        }

        const auto perClusterSizes = distType.getDistribution().getMemoryShapes();
        for (const auto& [cluster, sizeAttr] : enumerate(perClusterSizes)) {
            cmxInputs.push_back({cmxVal, static_cast<int64_t>(cluster)});

            auto sizeArrAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(sizeAttr);
            if (sizeArrAttr == nullptr) {
                return matchFailed(_log.nest(), rewriter, copyOp,
                                   "ConcatView input {0} per-cluster size is not an ArrayAttr", i);
            }
            const auto size = parseIntArrayAttr<int64_t>(sizeArrAttr);
            const int64_t chSize = size[Dims4D::Act::C.ind()];
            inputChannelSizes.push_back(chSize);
            _log.nest().trace("  cmxInputs[{0}]: SEGMENTED_ON_K (source {1}, cluster={2}), C={3}", cmxInputs.size() - 1,
                              i, cluster, chSize);
        }
    }

    SmallVector<int64_t> inputChannelOffsets;
    inputChannelOffsets.reserve(inputChannelSizes.size() + 1);
    inputChannelOffsets.push_back(0);
    for (const auto channels : inputChannelSizes) {
        inputChannelOffsets.push_back(inputChannelOffsets.back() + channels);
    }
    if (inputChannelOffsets.back() != totalC) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "Sum of input channel sizes {0} does not match total channels {1}",
                           inputChannelOffsets.back(), totalC);
    }
    _log.nest().trace("  inputChannelOffsets ({0} entries, {1} virtual inputs): {2}", inputChannelOffsets.size() - 1,
                      cmxInputs.size(), inputChannelOffsets);

    // Flat view shape: numFlatRows x gatherRowSize. Validate the front flat view row width matches gatherRowSize.
    auto flatType = mlir::cast<vpux::NDTypeInterface>(chain.frontReshapeFlat.getOutput().getType());
    const auto flatShape = flatType.getShape();
    if (flatShape.size() != 2) {
        return matchFailed(_log.nest(), rewriter, copyOp, "Flat view shape is not 2D");
    }
    const int64_t numFlatRows = flatShape[Dim(0)];
    if (flatShape[Dim(1)] != gatherRowSize) {
        return matchFailed(_log.nest(), rewriter, copyOp, "Flat view row width {0} does not match gatherRowSize {1}",
                           flatShape[Dim(1)], gatherRowSize);
    }
    if (totalC % gatherRowSize != 0) {
        return matchFailed(_log.nest(), rewriter, copyOp, "Total channels {0} is not divisible by gatherRowSize {1}",
                           totalC, gatherRowSize);
    }
    const int64_t rowsPerSpatialPosition = totalC / gatherRowSize;

    // Validate every gathered row maps onto the resolved CMX sources.
    for (int64_t r : rowIndices) {
        if (r < 0 || r >= numFlatRows) {
            return matchFailed(_log.nest(), rewriter, copyOp, "Row index {0} is out of range [0, {1})", r, numFlatRows);
        }

        const int64_t absoluteChannelOffset = (r % rowsPerSpatialPosition) * gatherRowSize;

        int64_t remaining = gatherRowSize;
        int64_t cursor = absoluteChannelOffset;
        while (remaining > 0) {
            const auto segment = getInputByAbsoluteChannelOffset(inputChannelOffsets, cursor);
            if (!segment.has_value()) {
                return matchFailed(_log.nest(), rewriter, copyOp, "Row {0} references out-of-range concat input", r);
            }

            const int64_t inputIdx = segment->first;
            const int64_t segmentEnd = inputChannelOffsets[inputIdx + 1];
            const int64_t take = std::min<int64_t>(remaining, segmentEnd - cursor);
            if (take <= 0) {
                return matchFailed(_log.nest(), rewriter, copyOp,
                                   "Non-positive segment length for row {0} at channel offset {1}", r, cursor);
            }

            remaining -= take;
            cursor += take;
        }
    }

    chain.gatherIndicesCopy = gatherIndicesCopy;
    chain.numGatheredRows = numGatheredRows;
    chain.gatherRowSize = gatherRowSize;
    chain.gatherRowIndices = std::move(rowIndices);
    chain.frontCmxInputs = std::move(cmxInputs);
    chain.frontChannelOffsets = std::move(inputChannelOffsets);
    chain.rowsPerSpatialPosition = rowsPerSpatialPosition;
    return mlir::success();
}

mlir::LogicalResult FuseGatherDMAWithSubViewCopy::matchFrontPattern(VPUIP::CopyOp copyOp,
                                                                    mlir::PatternRewriter& rewriter,
                                                                    MatchedChain& chain) const {
    auto gatherDMA = chain.gatherDMA;

    // GatherDMA reads a flat DDR buffer. Two topologies feed it:
    //  (A) DDR-side view chain: the flat GenericReshape is directly the GatherDMA input.
    //  (B) CMX-side view chain: the flat GenericReshape lives in CMX and a single CMX->DDR CopyOp sits between it
    //      and the GatherDMA (the reshape/permute/flatten chain was hoisted before that copy).
    auto frontInputCopy = gatherDMA.getInput().getDefiningOp<VPUIP::CopyOp>();
    if (frontInputCopy != nullptr && !frontInputCopy.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, copyOp, "GatherDMA input CopyOp has multiple uses");
    }
    const auto flatValue = (frontInputCopy != nullptr) ? frontInputCopy.getInput() : gatherDMA.getInput();
    auto frontReshapeFlat = flatValue.getDefiningOp<VPUIP::GenericReshapeOp>();
    if (!frontReshapeFlat || !frontReshapeFlat.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "GatherDMA input is not a single-use GenericReshapeOp (frontReshapeFlat)");
    }
    auto frontPermuteCast = frontReshapeFlat.getInput().getDefiningOp<VPUIP::PermuteCastOp>();
    if (!frontPermuteCast || !frontPermuteCast.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, copyOp, "frontReshapeFlat input is not a single-use PermuteCastOp");
    }

    const auto expectedPerm =
            mlir::AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 3, 0, 2}, frontPermuteCast.getContext());
    if (frontPermuteCast.getMemPerm() != expectedPerm) {
        return matchFailed(_log.nest(), rewriter, copyOp,
                           "PermuteCastOp does not have the expected [d1,d3,d0,d2] permutation");
    }

    // The value feeding PermuteCast (through an optional reshape). Depending on where the view chain lives it is:
    //  - a ConcatViewOp output (DDR side, several CMX->DDR copies), or
    //  - a single CMX->DDR CopyOp output (DDR side, one copy), or
    //  - a CMX DistributedBuffer read directly (CMX side, the copy-before-gather variant).
    mlir::Value upstreamValue;
    auto frontReshapeBeforePermute = frontPermuteCast.getSource().getDefiningOp<VPUIP::GenericReshapeOp>();
    if (frontReshapeBeforePermute == nullptr) {
        upstreamValue = frontPermuteCast.getSource();
    } else if (frontReshapeBeforePermute.getResult().hasOneUse()) {
        upstreamValue = frontReshapeBeforePermute.getInput();
    } else {
        return matchFailed(_log.nest(), rewriter, copyOp, "frontReshapeBeforePermute has multiple uses");
    }

    // Normalize every topology into a single list of CMX source buffers to slice per row. `frontCopies`
    // additionally records the CMX->DDR copies to clean up (empty for the CMX-side variant, which has none).
    auto frontConcat = upstreamValue.getDefiningOp<VPUIP::ConcatViewOp>();
    SmallVector<mlir::Value> cmxSources;
    SmallVector<mlir::Value> frontCopies;
    if (frontConcat != nullptr) {
        if (!frontConcat.getResult().hasOneUse()) {
            return matchFailed(_log.nest(), rewriter, copyOp, "Upstream ConcatViewOp has multiple uses");
        }
        for (const auto& [i, concatInput] : enumerate(frontConcat.getInputs())) {
            auto copy = concatInput.getDefiningOp<VPUIP::CopyOp>();
            if (!copy) {
                return matchFailed(_log.nest(), rewriter, copyOp, "ConcatView input {0} is not a CopyOp", i);
            }
            cmxSources.push_back(copy.getInput());
            frontCopies.push_back(concatInput);
        }
    } else if (auto singleCopy = upstreamValue.getDefiningOp<VPUIP::CopyOp>()) {
        cmxSources.push_back(singleCopy.getInput());
        frontCopies.push_back(upstreamValue);
    } else {
        // CMX-side view chain: the activation is a CMX DistributedBuffer sliced directly (no DDR copy to erase).
        cmxSources.push_back(upstreamValue);
    }

    // The combined value must be 4D NHWC; each CMX source is DUPLICATED/MULTICASTED/SEGMENTED-on-K. The per-source
    // distribution analysis and channel-offset resolution happen in matchGather, which runs after this.
    auto upstreamType = mlir::cast<vpux::NDTypeInterface>(upstreamValue.getType());
    if (upstreamType.getShape().size() != 4 || upstreamType.getDimsOrder() != DimsOrder::NHWC) {
        return matchFailed(_log.nest(), rewriter, copyOp, "Upstream value is not 4D NHWC");
    }
    const auto combinedShape = upstreamType.getShape();

    chain.frontConcat = frontConcat;
    chain.frontCopies = std::move(frontCopies);
    chain.frontReshapeBeforePermute = frontReshapeBeforePermute;
    chain.frontPermuteCast = frontPermuteCast;
    chain.frontReshapeFlat = frontReshapeFlat;
    chain.frontInputCopy = frontInputCopy;
    chain.frontCmxSources = std::move(cmxSources);
    chain.height = combinedShape[Dims4D::Act::H];
    chain.width = combinedShape[Dims4D::Act::W];
    chain.frontTotalChannels = combinedShape[Dims4D::Act::C];
    return mlir::success();
}

mlir::LogicalResult FuseGatherDMAWithSubViewCopy::matchChain(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                             MatchedChain& chain) const {
    // Match around the GatherDMA pivot: back (gather consumers) -> front (structural gather producers) ->
    // gather (op, indices and the derived geometry that needs both front and gather info).
    if (mlir::failed(matchBackPattern(copyOp, rewriter, chain))) {
        return mlir::failure();
    }
    if (mlir::failed(matchFrontPattern(copyOp, rewriter, chain))) {
        return mlir::failure();
    }
    if (mlir::failed(matchGather(copyOp, rewriter, chain))) {
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult FuseGatherDMAWithSubViewCopy::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseGatherDMAWithSubViewCopy: checking CopyOp at '{0}'", copyOp->getLoc());

    MatchedChain chain;
    if (mlir::failed(matchChain(copyOp, rewriter, chain))) {
        return mlir::failure();
    }

    _log.trace("FuseGatherDMAWithSubViewCopy: matched chain — "
               "{0} gathered rows, gatherRowSize={1}, H={2}, W={3}, "
               "{4} virtual cmxInputs, rowsPerSpatialPosition={5}",
               chain.numGatheredRows, chain.gatherRowSize, chain.height, chain.width, chain.frontCmxInputs.size(),
               chain.rowsPerSpatialPosition);

    auto gatherDMA = chain.gatherDMA;
    auto ddrDestSubView = chain.backDestSubView;
    const auto& cmxInputs = chain.frontCmxInputs;
    const auto& inputChannelOffsets = chain.frontChannelOffsets;
    const auto& rowIndices = chain.gatherRowIndices;
    const auto width = chain.width;
    const auto gatherRowSize = chain.gatherRowSize;
    const auto numGatheredRows = chain.numGatheredRows;
    const auto rowsPerSpatialPosition = chain.rowsPerSpatialPosition;
    const auto gatheredRowsDim = chain.gatheredRowsDim;

    auto* ctx = rewriter.getContext();
    const auto baseOffsets = parseIntArrayAttr<int64_t>(ddrDestSubView.getStaticOffsets());
    const auto baseSizes = parseIntArrayAttr<int64_t>(ddrDestSubView.getStaticSizes());

    SmallVector<mlir::Value> copyConcat;

    // channelDim is the per-row channel axis in the DDR destination (last dim of the subview).
    const auto channelDim = gatheredRowsDim + 1;

    auto createNewCopy = [&](mlir::Value cmxSrc, mlir::Value ddrDst, int64_t rowJ, int64_t piece) -> mlir::Value {
        auto cmxType = mlir::cast<vpux::NDTypeInterface>(cmxSrc.getType());
        auto ddrType = mlir::cast<vpux::NDTypeInterface>(ddrDst.getType());
        auto* ctx = rewriter.getContext();

        auto reshapedType = ddrType.changeShape(cmxType.getShape());
        auto ddrReshaped = rewriter.create<VPUIP::GenericReshapeOp>(
                appendLoc(copyOp->getLoc(), "_ddr_reshape_{0}_{1}", rowJ, piece), mlir::Type(reshapedType), ddrDst);

        const auto memPerm = vpux::getPermutationFromOrders(reshapedType.getDimsOrder(), cmxType.getDimsOrder(), ctx);
        auto permutedType = reshapedType.changeDimsOrder(cmxType.getDimsOrder());
        auto ddrPermuted = rewriter.create<VPUIP::PermuteCastOp>(
                appendLoc(copyOp->getLoc(), "_ddr_permute_{0}_{1}", rowJ, piece), mlir::Type(permutedType),
                ddrReshaped.getResult(), mlir::AffineMapAttr::get(permutedType.getDimsOrder().toAffineMap(ctx)),
                mlir::AffineMapAttr::get(memPerm));

        auto newCopy = rewriter.create<VPUIP::CopyOp>(
                appendLoc(copyOp->getLoc(), "_cmx_to_ddr_copy_{0}_{1}", rowJ, piece), cmxSrc, ddrPermuted.getResult());

        auto postReshape = rewriter.create<VPUIP::GenericReshapeOp>(
                                           appendLoc(copyOp->getLoc(), "_post_copy_reshape_{0}_{1}", rowJ, piece),
                                           mlir::Type(ddrType), newCopy.getOutput())
                                   .getOutput();
        return postReshape;
    };

    auto getCMXInputSlice = [&](ArrayRef<CMXInputInfo> inputInfo, ArrayRef<int64_t> inChOffsets, int64_t currInput,
                                int64_t currOffset, int64_t take, int64_t j, int64_t pieceIdx, int64_t h,
                                int64_t w) -> mlir::Value {
        const auto& input = inputInfo[currInput];
        if (input.cluster == -1) {
            _log.nest().trace("    row {0} piece {1}: SubViewOp from DUPLICATED cmxInputs[{2}] "
                              "at [0, {3}, {4}, {5}] size [1, {6}, 1, 1]",
                              j, pieceIdx, currInput, currOffset, h, w, take);
            return rewriter
                    .create<VPUIP::SubViewOp>(appendLoc(copyOp->getLoc(), "_cmx_sv_{0}_{1}", j, pieceIdx), input.input,
                                              getIntArrayAttr(ctx, SmallVector<int64_t>{0, currOffset, h, w}),
                                              getIntArrayAttr(ctx, SmallVector<int64_t>{1, take, 1, 1}))
                    .getResult();
        }

        // Make offset relative to the start of the cluster 0 of this distribution
        for (auto revIdx = currInput; revIdx >= 0; revIdx--) {
            if (inputInfo[revIdx].input == input.input && inputInfo[revIdx].cluster == 0) {
                currOffset += inChOffsets[currInput] - inChOffsets[revIdx];
                break;
            }
        }
        _log.nest().trace("    row {0} piece {1}: ExtractFlatSliceOp from SEGMENTED cmxInputs[{2}] "
                          "(cluster={3}), flatOffset={4}, take={5}",
                          j, pieceIdx, currInput, input.cluster, currOffset, take);
        auto extractFlatSlice =
                rewriter.create<VPUIP::ExtractFlatSliceOp>(appendLoc(copyOp->getLoc(), "_cmx_efs_{0}_{1}", j, pieceIdx),
                                                           input.input, currOffset, take)
                        .getResult();

        return rewriter
                .create<VPUIP::SubViewOp>(appendLoc(copyOp->getLoc(), "_cmx_sv_{0}_{1}", j, pieceIdx), extractFlatSlice,
                                          getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, h, w}),
                                          getIntArrayAttr(ctx, SmallVector<int64_t>{1, take, 1, 1}))
                .getResult();
    };

    for (int64_t j = 0; j < numGatheredRows; ++j) {
        // Convert gather indices to offset in original CMX buffer.
        const int64_t indices = rowIndices[j];
        const int64_t spatialPosition = indices / rowsPerSpatialPosition;
        const int64_t absoluteChannelOffset = (indices % rowsPerSpatialPosition) * gatherRowSize;
        const int64_t h = spatialPosition / width;
        const int64_t w = spatialPosition % width;
        _log.nest().trace("  gathered row {0}: flatIndex={1} -> spatialPos={2} (h={3}, w={4}), "
                          "absoluteChannelOffset={5}",
                          j, indices, spatialPosition, h, w, absoluteChannelOffset);

        SmallVector<int64_t> rowOffsets(baseOffsets.begin(), baseOffsets.end());
        SmallVector<int64_t> rowSizes(baseSizes.begin(), baseSizes.end());
        rowOffsets[gatheredRowsDim] += j;
        rowSizes[gatheredRowsDim] = 1;

        auto ddrRowSubview = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(copyOp->getLoc(), "_ddr_row_subview_{0}", j), ddrDestSubView.getSource(),
                getIntArrayAttr(ctx, rowOffsets), getIntArrayAttr(ctx, rowSizes));

        int64_t remaining = gatherRowSize;
        int64_t cursor = absoluteChannelOffset;
        SmallVector<mlir::Value> pieces;
        int64_t pieceIdx = 0;
        while (remaining > 0) {
            const auto segment = getInputByAbsoluteChannelOffset(inputChannelOffsets, cursor);
            VPUX_THROW_WHEN(!segment.has_value(), "Gather row {0} references out-of-range concat input", j);
            const int64_t curInput = segment->first;
            const int64_t curOffset = segment->second;
            const int64_t segmentEnd = inputChannelOffsets[curInput + 1];
            const int64_t take = std::min<int64_t>(remaining, segmentEnd - cursor);
            VPUX_THROW_WHEN(take <= 0, "Invalid channel segment for gather row {0}", j);

            _log.nest().trace("    piece {0}: cursor={1}, cmxInputs[{2}], localOffset={3}, take={4}", pieceIdx, cursor,
                              curInput, curOffset, take);
            auto cmxSubview =
                    getCMXInputSlice(cmxInputs, inputChannelOffsets, curInput, curOffset, take, j, pieceIdx, h, w);

            // Build corresponding DDR subview for this piece.
            SmallVector<int64_t> offset(rowOffsets.size(), 0);
            SmallVector<int64_t> size(rowSizes.begin(), rowSizes.end());
            offset[channelDim] = gatherRowSize - remaining;  // offset within the row's channel axis
            size[channelDim] = take;
            auto ddrSub = rewriter.create<VPUIP::SubViewOp>(appendLoc(copyOp->getLoc(), "_ddr_sv_{0}_{1}", j, pieceIdx),
                                                            ddrRowSubview.getResult(), getIntArrayAttr(ctx, offset),
                                                            getIntArrayAttr(ctx, size));

            auto newCopy = createNewCopy(cmxSubview, ddrSub.getResult(), j, pieceIdx);
            pieces.push_back(newCopy);

            remaining -= take;
            cursor += take;
            ++pieceIdx;
        }

        if (pieces.size() == 1) {
            copyConcat.push_back(pieces[0]);
        } else {
            auto combined =
                    rewriter.create<VPUIP::ConcatViewOp>(appendLoc(copyOp->getLoc(), "_row_pieces_concat_{0}", j),
                                                         ddrRowSubview.getType(), pieces, ddrRowSubview.getResult())
                            .getOutput();
            copyConcat.push_back(combined);
        }
    }

    mlir::Value rebuiltOutput;
    if (copyConcat.size() == 1) {
        rebuiltOutput = copyConcat[0];
    } else {
        rebuiltOutput =
                rewriter.create<VPUIP::ConcatViewOp>(appendLoc(copyOp->getLoc(), "_rebuilt_concat"),
                                                     ddrDestSubView.getType(), copyConcat, ddrDestSubView.getResult())
                        .getOutput();
    }
    rewriter.replaceOp(copyOp, rebuiltOutput);

    _log.trace("FuseGatherDMAWithSubViewCopy: replaced GatherDMA+Copy at '{0}' with {1} direct CMX→DDR copies",
               gatherDMA->getLoc(), numGatheredRows);

    eraseMatchedChain(rewriter, chain);

    return mlir::success();
}

void FuseGatherDMAWithSubViewCopy::eraseMatchedChain(mlir::PatternRewriter& rewriter, MatchedChain& chain) const {
    auto eraseEmptyOp = [&](mlir::Operation* op) {
        if (op && op->use_empty()) {
            rewriter.eraseOp(op);
        }
    };

    // Back: reshape after the gather, then the gather itself (with its indices) and its output buffer.
    eraseEmptyOp(chain.backReshape);
    auto* gatherOutputAlloc = chain.gatherDMA.getOutputBuff().getDefiningOp();
    if (chain.gatherDMA->use_empty()) {
        auto* idxAllocOp = chain.gatherIndicesCopy.getOutputBuff().getDefiningOp();
        rewriter.eraseOp(chain.gatherDMA);
        if (chain.gatherIndicesCopy->use_empty()) {
            rewriter.eraseOp(chain.gatherIndicesCopy);
            eraseEmptyOp(idxAllocOp);
        }
    }
    eraseEmptyOp(gatherOutputAlloc);

    // Front (CMX-side variant): erase the single CMX->DDR copy (and its DDR buffer) right before the GatherDMA.
    if (chain.frontInputCopy != nullptr) {
        auto* copyDdrBuf = chain.frontInputCopy.getOutputBuff().getDefiningOp();
        eraseEmptyOp(chain.frontInputCopy);
        eraseEmptyOp(copyDdrBuf);
    }

    // Front: the flat view chain, once no more GatherDMA users remain.
    eraseEmptyOp(chain.frontReshapeFlat);
    eraseEmptyOp(chain.frontPermuteCast);
    eraseEmptyOp(chain.frontReshapeBeforePermute);

    // Front: each upstream CMX->DDR copy (and its DDR output buffer) once it is dead.
    auto eraseCopyAndDDR = [&](mlir::Value copyVal) {
        auto cmxCopy = copyVal.getDefiningOp<VPUIP::CopyOp>();
        if (cmxCopy == nullptr || !cmxCopy->use_empty()) {
            return;
        }
        auto* ddrBuf = cmxCopy.getOutputBuff().getDefiningOp();
        rewriter.eraseOp(cmxCopy);
        eraseEmptyOp(ddrBuf);
    };

    if (chain.frontConcat != nullptr) {
        if (chain.frontConcat->use_empty()) {
            rewriter.eraseOp(chain.frontConcat);
            for (auto copyVal : chain.frontCopies) {
                eraseCopyAndDDR(copyVal);
            }
        }
    } else {
        // Single-input case: no concat to erase, just the sole CMX->DDR copy.
        for (auto copyVal : chain.frontCopies) {
            eraseCopyAndDDR(copyVal);
        }
    }
}

//
// OptimizeConcatViewCopiesPass
//

class OptimizeConcatViewCopiesPass final :
        public VPUIP::impl::OptimizeConcatViewCopiesBase<OptimizeConcatViewCopiesPass> {
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
    patterns.add<SplitMultiLeftUnbalancedDDRConcatOnSameAxis>(&ctx, _log);
    patterns.add<SplitMultiLeftUnbalancedDDRConcatOnOtherAxis>(&ctx, _log);
    // SplitUnbalancedDDRConcatToNonDistributedCMX inherited from SplitUnbalancedDDRConcatOnOtherAxis, any changes to
    // SplitUnbalancedDDRConcatOnOtherAxis rewriter also need to consider here.
    patterns.add<SplitUnbalancedDDRConcatToNonDistributedCMX>(&ctx, _log);
    patterns.add<ReuseConcatViewAsInput>(&ctx, _log);
    patterns.add<FuseGatherDMAWithSubViewCopy>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
