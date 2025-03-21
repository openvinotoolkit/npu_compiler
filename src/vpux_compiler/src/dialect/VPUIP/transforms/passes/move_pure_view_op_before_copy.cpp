//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVEPUREVIEWOPBEFORECOPY
#define GEN_PASS_DEF_MOVEPUREVIEWOPBEFORECOPY
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

using GetCopyFunctType = FuncRef<VPUIP::LayerOpInterface(mlir::Operation*)>;
using CreateCopyFunctType =
        FuncRef<VPUIP::LayerOpInterface(mlir::PatternRewriter&, mlir::Location, mlir::Value, mlir::Value)>;

VPUIP::LayerOpInterface getCopyOp(mlir::Operation* sourceOp) {
    return mlir::dyn_cast_or_null<VPUIP::CopyOp>(sourceOp);
}

VPUIP::LayerOpInterface createNewCopyOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                        mlir::Value outputBuff) {
    return rewriter.create<VPUIP::CopyOp>(loc, input, outputBuff);
}

VPUIP::LayerOpInterface getTillingCopyOp(mlir::Operation* sourceOp) {
    auto clusterTiling = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(sourceOp);
    if (clusterTiling == nullptr || clusterTiling.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return nullptr;
    }

    return clusterTiling;
}

VPUIP::LayerOpInterface createNewTillingCopyOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                               mlir::Value outputBuff) {
    const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };

    SmallVector<mlir::Value> inputsOutputOperands = {input, outputBuff};
    return rewriter.create<VPUIP::NCEClusterTilingOp>(loc, outputBuff.getType(), inputsOutputOperands,
                                                      copyOutBodyBuilder);
}

//
// LayerRewriter
//

class LayerRewriterBase : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    LayerRewriterBase(mlir::MLIRContext* ctx, GetCopyFunctType getCopyOp, CreateCopyFunctType createNewCopyOp,
                      Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx),
              _getCopyOp(getCopyOp),
              _createNewCopyOp(createNewCopyOp),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    GetCopyFunctType _getCopyOp;
    CreateCopyFunctType _createNewCopyOp;
    Logger _log;
};

mlir::LogicalResult LayerRewriterBase::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<VPUIP::LayerOpInterface>(*origOp)) {
        return mlir::failure();
    }

    if (!mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::ShapeCastOp>(*origOp)) {
        return mlir::failure();
    }

    _log.trace("Got pure view-like op: '{0}':'{1}'", origOp->getName(), origOp->getLoc());
    auto maybeCopy = _getCopyOp(origOp->getOperand(0).getDefiningOp());
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

    auto copyOpInputType = VPUIP::extractDataType(copyOpInput).cast<vpux::NDTypeInterface>();
    auto copyOpOutputType = VPUIP::extractDataType(copyOpOutput).cast<vpux::NDTypeInterface>();

    auto viewOpInputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto viewOpOutputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto viewOpOutputShape = viewOpOutputType.getShape();
    auto viewOpOutputElemType = viewOpOutputType.getElementType();

    const auto inputShape = viewOpInputType.getShape();
    const auto outputShape = viewOpOutputType.getShape();
    const auto isRankChangedByViewOp = inputShape.size() != outputShape.size();
    auto distributedType = copyOpInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
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
                return (distributedType.getElementType().isa<mlir::quant::UniformQuantizedType>() &&
                        viewOpOutputElemType.isa<mlir::quant::UniformQuantizedType>());
            }

            // If the cluster copy op has siblings, moving pureViewOp
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
                const auto arch = VPU::getArch(origOp.getOperation());
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
            // If the cluster copy op has siblings, moving pureViewOp
            // in front of it may cause accuracy issues
            if (!copyInput.hasOneUse()) {
                return false;
            }
            if (mlir::isa<VPUIP::QuantizeCastOp>(viewOp)) {
                const auto viewOpOutputType = viewOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
                const auto viewOpOutputElemType = viewOpOutputType.getElementType();
                // Only support per-tensor uniform quantized type
                if (distType.getElementType().isa<mlir::quant::UniformQuantizedType>() &&
                    viewOpOutputElemType.isa<mlir::quant::UniformQuantizedType>()) {
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
        const auto arch = VPU::getArch(origOp.getOperation());
        const auto mode = distributedType.getDistribution().getMode().getValue();
        const auto origDistribution = distributedType.getDistribution();

        if (auto permuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*origOp)) {
            auto inPermuteType = permuteCast->getOperand(0).getType().cast<vpux::NDTypeInterface>();
            auto outPermuteType = permuteCast->getResult(0).getType().cast<vpux::NDTypeInterface>();

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
    auto newCopyOp = _createNewCopyOp(rewriter, maybeCopy->getLoc(), origOp->getResult(0), allocOp);

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

class MoveViewOpToTheFrontOfCopy final : public LayerRewriterBase {
public:
    MoveViewOpToTheFrontOfCopy(mlir::MLIRContext* ctx, Logger log)
            : LayerRewriterBase(ctx, getCopyOp, createNewCopyOp, log) {
    }
};

//
// MoveViewOpToTheFrontOfTillingCopy
//

class MoveViewOpToTheFrontOfTillingCopy final : public LayerRewriterBase {
public:
    MoveViewOpToTheFrontOfTillingCopy(mlir::MLIRContext* ctx, Logger log)
            : LayerRewriterBase(ctx, getTillingCopyOp, createNewTillingCopyOp, log) {
    }
};

//
// MoveSubviewToTheFrontOfCopyBase
//
class MoveSubviewToTheFrontOfCopyBase : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    MoveSubviewToTheFrontOfCopyBase(mlir::MLIRContext* ctx, GetCopyFunctType getCopyOp,
                                    CreateCopyFunctType createNewCopyOp, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx),
              _getCopyOp(getCopyOp),
              _createNewCopyOp(createNewCopyOp),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    GetCopyFunctType _getCopyOp;
    CreateCopyFunctType _createNewCopyOp;
    Logger _log;
};

// SubView is not compatible with distributed buffer when:
// 1. Distributed buffer is segmented
// 2. SubView shrinks segmented axis
bool isSubViewCompatibleWithDistributedBuffer(VPUIP::SubViewOp subViewOp,
                                              VPUIP::DistributedBufferType distributedType) {
    const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
    if (!tileIndex.has_value()) {
        // DUPLICATED | MULTICASTED
        return true;
    }

    auto tileIndexVal = tileIndex.value();
    auto origShape = getShape(subViewOp.getSource());
    auto subShape = getShape(subViewOp.getResult());

    if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithClusterCopy(
                parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsetsAttr()), tileIndexVal, distributedType)) {
        return false;
    }

    // Be compatible if SubView does not shrink segmented axis
    return origShape[Dim(tileIndexVal)] == subShape[Dim(tileIndexVal)];
}

mlir::LogicalResult MoveSubviewToTheFrontOfCopyBase::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto subViewOp = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (subViewOp == nullptr) {
        return mlir::failure();
    }

    auto sourceOp = subViewOp.getSource().getDefiningOp();
    if (sourceOp == nullptr) {
        // Source is BlockArgument
        return mlir::failure();
    }

    auto parentCopyOp = _getCopyOp(subViewOp.getSource().getDefiningOp());
    if (parentCopyOp == nullptr) {
        return mlir::failure();
    }

    // optimize happens only when tillingOp has one subview user
    if (!parentCopyOp->getResults()[0].hasOneUse()) {
        return mlir::failure();
    }

    // perform this optimization only when distributed buffer is compatible with subview
    // otherwise an accuracy degradation may occur
    auto originOperand = parentCopyOp->getOperand(0);
    if (auto distributedType = originOperand.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        if (!isSubViewCompatibleWithDistributedBuffer(subViewOp, distributedType)) {
            return mlir::failure();
        }
    }

    _log.trace("Move subview {0} in front of copy {1}", subViewOp->getLoc(), parentCopyOp->getLoc());

    if (auto arg = originOperand.dyn_cast<mlir::BlockArgument>()) {
        rewriter.setInsertionPointToStart(arg.getParentBlock());
    } else {
        rewriter.setInsertionPointAfter(originOperand.getDefiningOp());
    }

    // create and insert a new subview
    auto newSubViewOp =
            rewriter.create<VPUIP::SubViewOp>(subViewOp->getLoc(), originOperand, subViewOp.getStaticOffsetsAttr(),
                                              subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

    auto subViewOpShape = getShape(newSubViewOp);
    auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.getOutputs()[0]);
    VPUX_THROW_UNLESS(mlir::isa_and_nonnull<mlir::memref::AllocOp>(allocOp),
                      "CopyOp output buffer should be AllocationOp");
    auto allocOpDtype = allocOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    // Per-axis quantization must be aligned with the shape.
    const auto targetElemType = newSubViewOp.getResult().getType().cast<vpux::NDTypeInterface>().getElementType();
    allocOp->getResult(0).setType(allocOpDtype.changeShapeElemType(subViewOpShape, targetElemType));

    auto newParentOp =
            _createNewCopyOp(rewriter, newSubViewOp->getLoc(), newSubViewOp.getResult(), allocOp->getResult(0));
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
// MoveSubviewToTheFrontOfCopy
//

/*
Move SubView to the front of Copy to make a chain of copies
     Copy(CMX2DDR)    =>          Subview
          |                          |
       Subview                  Copy(CMX2DDR)
          |                          |
        Copy                       Copy
*/

class MoveSubviewToTheFrontOfCopy final : public MoveSubviewToTheFrontOfCopyBase {
public:
    MoveSubviewToTheFrontOfCopy(mlir::MLIRContext* ctx, Logger log)
            : MoveSubviewToTheFrontOfCopyBase(ctx, getCopyOp, createNewCopyOp, log) {
    }
};

//
// MoveSubviewToTheFrontOfTillingCopy
//

/*
 Move SubView to the front of  TillingCopy, the assumption is copy src in CMX is faster than DDR
        NCEOp                      NCEOp
          |                          |
  TillingCopy(CMX2DDR)    =>      Subview
          |                          |
       Subview               TillingCopy(CMX2DDR)
          |                          |
        Copy                       Copy
*/

class MoveSubviewToTheFrontOfTillingCopy final : public MoveSubviewToTheFrontOfCopyBase {
public:
    MoveSubviewToTheFrontOfTillingCopy(mlir::MLIRContext* ctx, Logger log)
            : MoveSubviewToTheFrontOfCopyBase(ctx, getTillingCopyOp, createNewTillingCopyOp, log) {
    }
};

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
    patterns.add<MoveViewOpToTheFrontOfTillingCopy>(&ctx, _log);
    patterns.add<MoveSubviewToTheFrontOfCopy>(&ctx, _log);
    patterns.add<MoveSubviewToTheFrontOfTillingCopy>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
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
