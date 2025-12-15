//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVEPUREVIEWOPBEFORECOPY
#define GEN_PASS_DEF_MOVEPUREVIEWOPBEFORECOPY
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// MoveViewOpToTheFrontOfCopy
//

class MoveViewOpToTheFrontOfCopy : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    MoveViewOpToTheFrontOfCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx), _log(log) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveViewOpToTheFrontOfCopy::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<VPUIP::LayerOpInterface>(*origOp)) {
        return mlir::failure();
    }

    if (!mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::ShapeCastOp>(*origOp)) {
        return mlir::failure();
    }

    _log.trace("Got pure view-like op: '{0}':'{1}'", origOp->getName(), origOp->getLoc());
    auto maybeCopy = origOp->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (maybeCopy == nullptr) {
        StringRef parentOpName = "None";
        if (auto parentOp = origOp->getOperand(0).getDefiningOp()) {
            parentOpName = parentOp->getName().getStringRef();
        }
        _log.trace("The operation defining the input is not Copy: '{0}'", parentOpName);
        return mlir::failure();
    }

    auto copyOpInput = maybeCopy.getInputs()[0];
    auto copyOpOutput = maybeCopy.getOutputs()[0];
    // When we have compress convolution we don't want to change
    // order between shapeCast and copy operation.
    // If shapeCast is moved before copy, instead of copying 4 channels,
    // copy operation will try to move 16 channels from memory.
    if (auto shapeCast = mlir::dyn_cast<VPUIP::ShapeCastOp>(*origOp)) {
        auto clusterTask = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(*shapeCast.getResult().getUsers().begin());
        if (clusterTask != nullptr && clusterTask.getInputChannelsCompression() == true) {
            return mlir::failure();
        }
    }

    if (!VPUIP::getRootAlloc<mlir::memref::AllocOp>(copyOpOutput)) {
        _log.trace("Skip complex case: the operation defining the output buffer is not Alloc");
        return mlir::failure();
    }

    auto copyOpInputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOpInput));
    auto copyOpOutputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOpOutput));

    auto viewOpInputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    auto viewOpOutputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    auto viewOpOutputShape = viewOpOutputType.getShape();
    auto viewOpOutputElemType = viewOpOutputType.getElementType();

    const auto inputShape = viewOpInputType.getShape();
    const auto outputShape = viewOpOutputType.getShape();
    const auto isRankChangedByViewOp = inputShape.size() != outputShape.size();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyOpInput.getType());
    mlir::FailureOr<std::pair<int64_t, int64_t>> getDistributedAxesMapping = mlir::failure();
    if (distributedType != nullptr && mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp)) {
        getDistributedAxesMapping = VPUIP::getDistributedAxesMappingAfterShapeChanged(
                viewOpInputType, viewOpOutputType, distributedType.getDistribution(), _log);
    }

    const auto isSupportedDuplicated = [&](const VPU::DistributionMode& mode) {
        if (isRankChangedByViewOp && mlir::failed(getDistributedAxesMapping)) {
            return false;
        }

        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
               VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
    };
    if (distributedType != nullptr) {
        const auto isSupportSegmented = [&](const VPUIP::DistributedBufferType distType) {
            // TODO: The num_tiles attribute also has to be adapted in case of different ranks
            if (isRankChangedByViewOp) {
                return false;
            }

            auto distribution = distType.getDistribution();
            const auto mode = distribution.getMode().getValue();

            if (mode != VPU::DistributionMode::SEGMENTED) {
                return false;
            }

            if (mlir::isa<VPUIP::QuantizeCastOp>(origOp)) {
                // Only support per-tensor uniform quantized type
                return (mlir::isa<mlir::quant::UniformQuantizedType>(distributedType.getElementType()) &&
                        mlir::isa<mlir::quant::UniformQuantizedType>(viewOpOutputElemType));
            }

            // If the distributed copy op has siblings, moving pureViewOp
            // in front of it may cause accuracy issues
            if (!copyOpInput.hasOneUse()) {
                return false;
            }

            if (auto permuteOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(origOp.getOperation())) {
                const auto inShape = getShape(permuteOp.getSource());
                const auto outShape = getShape(permuteOp.getResult());
                const auto inOrder = DimsOrder::fromValue(permuteOp.getSource());
                const auto dstOrder = DimsOrder::fromAffineMap(permuteOp.getDstOrder());
                if (inShape == outShape) {
                    // If op is non-trival reorder, do not move this op
                    return vpux::isTrivialReorder(inOrder, dstOrder, inShape);
                }

                const auto inMemShape = mlir::cast<NDTypeInterface>(permuteOp.getSource().getType()).getMemShape();
                const auto outMemShape = mlir::cast<NDTypeInterface>(permuteOp.getResult().getType()).getMemShape();
                auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
                const auto numTileDims = vpux::VPU::getNonOneDimInds(numTiles);
                // Only support permuteCast with single tiling dim. And memory shape shouldn't change else we may have
                // accuracy issue
                return (numTileDims.size() == 1) && (inMemShape == outMemShape);
            }

            if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp)) {
                const auto arch = config::getArch(origOp.getOperation());
                return VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(
                        distributedType, viewOpOutputShape, viewOpOutputType.getDimsOrder(), arch);
            }
            return false;
        };
        const auto isSupportedOverlapping = [&](const VPUIP::DistributedBufferType distType,
                                                const mlir::ViewLikeOpInterface viewOp, const mlir::Value copyInput) {
            // TODO: The num_tiles attribute also has to be adapted in case of different ranks
            if (isRankChangedByViewOp) {
                return false;
            }

            auto distribution = distType.getDistribution();
            const auto mode = distribution.getMode().getValue();
            if (mode != VPU::DistributionMode::OVERLAPPED) {
                return false;
            }
            // If the distributed copy op has siblings, moving pureViewOp
            // in front of it may cause accuracy issues
            if (!copyInput.hasOneUse()) {
                return false;
            }
            if (mlir::isa<VPUIP::QuantizeCastOp>(viewOp)) {
                const auto viewOpOutputType = mlir::cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType());
                const auto viewOpOutputElemType = viewOpOutputType.getElementType();
                // Only support per-tensor uniform quantized type or integer 8 bit types
                auto isI8OrPerTensorQuantized = [](const mlir::Type elemType) {
                    constexpr int8_t ELEMENT_TYPE_BIT_WIDTH = 8;
                    return mlir::isa<mlir::quant::UniformQuantizedType>(elemType) ||
                           elemType.isInteger(ELEMENT_TYPE_BIT_WIDTH);
                };

                if (isI8OrPerTensorQuantized(distType.getElementType()) &&
                    isI8OrPerTensorQuantized(viewOpOutputElemType)) {
                    return true;
                }
            }

            if (auto permuteOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(origOp.getOperation())) {
                const auto inShape = getShape(permuteOp.getSource());
                const auto outShape = getShape(permuteOp.getResult());
                const auto inOrder = DimsOrder::fromValue(permuteOp.getSource());
                const auto dstOrder = DimsOrder::fromAffineMap(permuteOp.getDstOrder());
                if (inShape == outShape) {
                    // If op is non-trival reorder, do not move this op
                    return vpux::isTrivialReorder(inOrder, dstOrder, inShape);
                }

                const auto inMemShape = mlir::cast<NDTypeInterface>(permuteOp.getSource().getType()).getMemShape();
                const auto outMemShape = mlir::cast<NDTypeInterface>(permuteOp.getResult().getType()).getMemShape();
                auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
                const auto numTileDims = vpux::VPU::getNonOneDimInds(numTiles);
                // Only support permuteCast with single tiling dim. And memory shape shouldn't change else we may have
                // accuracy issue
                return (numTileDims.size() == 1) && (inMemShape == outMemShape);
            }

            if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp)) {
                return VPUIP::isOverlappedDistributedCompatibleAfterShapeChangeForViewOps(
                        distributedType, viewOpOutputShape, viewOpOutputType.getDimsOrder());
            }

            return false;
        };
        const auto mode = distributedType.getDistribution().getMode().getValue();
        if (!isSupportedDuplicated(mode) && !isSupportSegmented(distributedType) &&
            !isSupportedOverlapping(distributedType, origOp, copyOpInput)) {
            _log.trace("Not supported distributed type");
            return mlir::failure();
        }
    }

    // TODO: #62719
    const auto inReqs = StrideReqs::compact(copyOpInputType.getRank());
    if (!inReqs.checkStrides(copyOpInputType)) {
        _log.trace("Skip complex case: input is strided");
        return mlir::failure();
    }

    vpux::NDTypeInterface newViewOpOutputType;

    auto getDistributionForViewOpOutput = [&]() -> VPU::DistributionInfoAttr {
        auto ctx = origOp->getContext();
        const auto arch = config::getArch(origOp.getOperation());
        const auto mode = distributedType.getDistribution().getMode().getValue();
        const auto origDistribution = distributedType.getDistribution();

        if (auto permuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*origOp)) {
            auto inPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCast->getOperand(0).getType());
            auto outPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCast->getResult(0).getType());

            return applyPermutationOnDistributionInfoAttr(distributedType, permuteCast.getMemPerm(),
                                                          inPermuteType.getDimsOrder(), outPermuteType.getDimsOrder(),
                                                          inPermuteType.getShape(), outPermuteType.getShape())
                    .value_or(nullptr);
        }

        const bool isShapeChangeOp = mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp);
        if (!isShapeChangeOp) {
            return origDistribution;
        }

        if (mode == VPU::DistributionMode::SEGMENTED) {
            return VPUIP::getSOHDistAttrWithNewShape(ctx, distributedType, viewOpOutputShape, arch);
        }

        if (mode == VPU::DistributionMode::OVERLAPPED) {
            return VPUIP::getOverlappedOverHDistAttrWithNewShape(ctx, distributedType, viewOpOutputShape);
        }

        if (!isSupportedDuplicated(mode)) {
            return origDistribution;
        }

        const auto duplicatedOutputMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistribution)) {
            if (isRankChangedByViewOp) {
                auto axesMapping = getDistributedAxesMapping.value();
                return VPUIP::changeDistributedAxisOnDistributionInfoAttr(
                        origDistribution, axesMapping.first, axesMapping.second, viewOpOutputType.getShape());
            }

            return VPU::DistributionInfoAttr::get(ctx, duplicatedOutputMode, nullptr, nullptr, nullptr, nullptr,
                                                  origDistribution.getNumClusters(), nullptr,
                                                  origDistribution.getUniformDistributedSegments(), nullptr, nullptr,
                                                  nullptr, nullptr, nullptr);
        }

        // GenericReshape and ShapeCast can change the output shape without needing to follow any rule.
        // Therefore, when having distributions such as SEGMENTED|DUPLICATED or SEGMENTED|MULTICASTED
        // we might end up with the "tiling dim" not having the same shape it had at input. It is also possible for
        // the new shape to not be tile-able over the number of clusters.
        // However, GenericReshape & ShapeCast are ops that work on the memory view and do not need compute view
        // at all, so to ensure we do not end up with an output with a clustering dim that cannot be tiled, we're
        // setting distribution as DUPLICATED for output.
        return VPU::getNonOverlappedDistributedAttr(viewOpOutputShape, duplicatedOutputMode, nullptr,
                                                    origDistribution.getNumClusters(), nullptr,
                                                    origDistribution.getUniformDistributedSegments(), ctx);
    };

    if (distributedType != nullptr) {
        auto ctx = origOp->getContext();
        const auto order = mlir::AffineMapAttr::get(viewOpOutputType.getDimsOrder().toAffineMap(ctx));
        auto viewOutDistributedAttr = getDistributionForViewOpOutput();

        if (viewOutDistributedAttr == nullptr) {
            return mlir::failure();
        }

        newViewOpOutputType =
                VPUIP::DistributedBufferType::get(ctx, viewOpOutputShape.raw(), viewOpOutputElemType, order,
                                                  distributedType.getMemSpace(), viewOutDistributedAttr);
    } else {
        newViewOpOutputType = viewOpOutputType.changeMemSpace(copyOpInputType.getMemSpace());
    }

    _log.trace("Set new input for '{0}': '{1}'", origOp->getName(), copyOpInput);
    origOp->setOperand(0, copyOpInput);

    _log.trace("Set new result type for '{0}': '{1}'", origOp->getName(), newViewOpOutputType);
    origOp->getResult(0).setType(newViewOpOutputType);

    rewriter.setInsertionPointAfter(origOp);

    auto newAllocType = viewOpOutputType.changeMemSpace(copyOpOutputType.getMemSpace());
    auto allocOp = allocateBuffersOfType(_log, maybeCopy->getLoc(), rewriter, newAllocType).front();
    auto newCopyOp = rewriter.create<VPUIP::CopyOp>(maybeCopy->getLoc(), origOp->getResult(0), allocOp);

    _log.trace("Replace all uses of pure view-like op with new Copy op: '{0}'", newCopyOp);
    rewriter.replaceAllUsesExcept(origOp->getResult(0), newCopyOp->getResults()[0], newCopyOp);

    auto sourceOp = copyOpOutput.getDefiningOp();

    if (sourceOp != nullptr && sourceOp->getResult(0).use_empty()) {
        rewriter.eraseOp(sourceOp);
    }

    if (maybeCopy->getResult(0).use_empty()) {
        rewriter.eraseOp(maybeCopy);
    }

    return mlir::success();
}

//
// MoveSubviewToTheFrontOfCopy
//

class MoveSubviewToTheFrontOfCopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    MoveSubviewToTheFrontOfCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveSubviewToTheFrontOfCopy::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Copy {0} at {1}", copyOp, copyOp.getLoc());
    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return mlir::failure();
    }
    auto subViewOp = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (subViewOp == nullptr) {
        return mlir::failure();
    }

    auto sourceOp = subViewOp.getSource().getDefiningOp();
    if (sourceOp == nullptr) {
        // Source is BlockArgument
        return mlir::failure();
    }

    auto parentCopyOp = subViewOp.getSource().getDefiningOp<VPUIP::CopyOp>();
    if (parentCopyOp == nullptr) {
        return mlir::failure();
    }

    // optimize happens only when the distributed op has one subview user
    if (!parentCopyOp->getResults()[0].hasOneUse()) {
        return mlir::failure();
    }

    auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.getOutputs()[0]);
    if (!mlir::isa_and_nonnull<mlir::memref::AllocOp>(allocOp)) {
        return mlir::failure();
    }

    // perform this optimization only when distributed buffer is compatible with subview
    // otherwise an accuracy degradation may occur
    auto originOperand = parentCopyOp->getOperand(0);
    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(originOperand.getType())) {
        if (!isSubViewCompatibleWithDistributedBuffer(subViewOp, distributedType)) {
            return mlir::failure();
        }
    }

    _log.trace("Move subview {0} in front of copy {1}", subViewOp->getLoc(), parentCopyOp->getLoc());

    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(originOperand)) {
        rewriter.setInsertionPointToStart(arg.getParentBlock());
    } else {
        rewriter.setInsertionPointAfter(originOperand.getDefiningOp());
    }

    // create and insert a new subview
    auto newSubViewOp =
            rewriter.create<VPUIP::SubViewOp>(subViewOp->getLoc(), originOperand, subViewOp.getStaticOffsetsAttr(),
                                              subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

    auto subViewOpShape = getShape(newSubViewOp);
    auto allocOpDtype = mlir::cast<vpux::NDTypeInterface>(allocOp->getResult(0).getType());
    // Per-axis quantization must be aligned with the shape.
    const auto targetElemType = mlir::cast<vpux::NDTypeInterface>(newSubViewOp.getResult().getType()).getElementType();
    allocOp->getResult(0).setType(allocOpDtype.changeShapeElemType(subViewOpShape, targetElemType));

    auto newParentOp =
            rewriter.create<VPUIP::CopyOp>(newSubViewOp->getLoc(), newSubViewOp->getResult(0), allocOp->getResult(0));
    if (newParentOp->isBeforeInBlock(allocOp)) {
        VPUIP::moveRootAllocBefore(allocOp, newParentOp);
    }

    rewriter.replaceAllUsesWith(parentCopyOp->getResults()[0], newParentOp->getResults()[0]);
    rewriter.eraseOp(parentCopyOp);

    // remove old subView
    rewriter.replaceAllUsesWith(subViewOp.getResult(), subViewOp.getSource());
    rewriter.eraseOp(subViewOp);
    return mlir::success();
}

//
// MovePureViewOpBeforeCopyPass
//

class MovePureViewOpBeforeCopyPass final :
        public VPUIP::impl::MovePureViewOpBeforeCopyBase<MovePureViewOpBeforeCopyPass> {
public:
    explicit MovePureViewOpBeforeCopyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MovePureViewOpBeforeCopyPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveViewOpToTheFrontOfCopy>(&ctx, _log);
    patterns.add<MoveSubviewToTheFrontOfCopy>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMovePureViewOpBeforeCopyPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMovePureViewOpBeforeCopyPass(Logger log) {
    return std::make_unique<MovePureViewOpBeforeCopyPass>(log);
}
