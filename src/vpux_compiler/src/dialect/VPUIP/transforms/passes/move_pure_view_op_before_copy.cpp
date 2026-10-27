//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVEPUREVIEWOPBEFORECOPY
#define GEN_PASS_DEF_MOVEPUREVIEWOPBEFORECOPY
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool canViewOpBeMovedBeforeCopy(VPUIP::CopyOp copyOp, VPU::DistributionMode mode, int64_t inAxis, int64_t outAxis,
                                ShapeRef newShape, config::ArchKind arch) {
    auto copyOpInput = copyOp.getInput();
    auto copyOpOutput = copyOp.getOutput();

    if (mode == VPU::DistributionMode::SEGMENTED) {
        if (arch == config::ArchKind::NPU40XX) {
            if (!copyOp.getInput().hasOneUse()) {
                return false;
            }

            auto copyOpInputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOpInput));
            auto copyOpOutputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOpOutput));
            const auto isCmxToDdrCopy = copyOpInputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                                        copyOpOutputType.getMemoryKind() == VPU::MemoryKind::DDR;
            if (isCmxToDdrCopy && (inAxis == Dims4D::Act::C.ind() && outAxis == Dims4D::Act::H.ind())) {
                return false;
            }

            auto inDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyOpInput.getType());
            auto inDistAttr = inDistType.getDistribution();
            auto alignmentAttr = inDistAttr.getAlignment();
            if (alignmentAttr != nullptr) {
                auto alignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
                if (alignmentVec[inAxis] > newShape[Dim(outAxis)]) {
                    return false;
                }
            }
        }

        if (!copyOpOutput.hasOneUse()) {
            return false;
        }

        // TODO: E#217860 ConvertDMA -> ShapeCast -> Copy
        // ConvertDMAViewLikeCopy does not optimize it safely when distributed operands are present
        auto inputConvertDMAOp = copyOpInput.getDefiningOp<VPUIP::ConvertDMAOp>();
        if (inputConvertDMAOp != nullptr && VPUIP::hasDistributedOperand(inputConvertDMAOp.getOperation())) {
            return false;
        }
    }

    return true;
}

vpux::NDTypeInterface getEventualCopyDestinationType(mlir::Value copyResult) {
    mlir::Value current = copyResult;
    vpux::NDTypeInterface destType = nullptr;
    while (current.hasOneUse()) {
        auto* user = *current.getUsers().begin();
        if (auto nextCopy = mlir::dyn_cast<VPUIP::CopyOp>(user)) {
            const auto outType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(nextCopy.getOutputs()[0]));
            destType = outType;
            current = nextCopy.getOutput();
            continue;
        }
        if (mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::ShapeCastOp>(user)) {
            current = user->getResult(0);
            continue;
        }
        break;
    }
    return destType;
}

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
    auto copyOp = origOp->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (copyOp == nullptr) {
        StringRef parentOpName = "None";
        if (auto parentOp = origOp->getOperand(0).getDefiningOp()) {
            parentOpName = parentOp->getName().getStringRef();
        }
        _log.trace("The operation defining the input is not Copy: '{0}'", parentOpName);
        return mlir::failure();
    }

    auto copyOpInput = copyOp.getInputs()[0];
    auto copyOpOutput = copyOp.getOutputs()[0];
    // When we have compress convolution we don't want to change
    // order between shapeCast and copy operation.
    // If shapeCast is moved before copy, instead of copying 4 channels,
    // copy operation will try to move 16 channels from memory.
    if (auto shapeCast = mlir::dyn_cast<VPUIP::ShapeCastOp>(*origOp)) {
        auto clusterTask = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(*shapeCast.getResult().getUsers().begin());
        if (clusterTask != nullptr && clusterTask.getInputChannelsCompression() == true) {
            _log.trace("Skip the case that compress convolution with input channels compression");
            return mlir::failure();
        }
    }

    if (!VPUIP::getRootAlloc<mlir::memref::AllocOp>(copyOpOutput)) {
        _log.trace("Skip complex case: the operation defining the output buffer is not Alloc");
        return mlir::failure();
    }

    const auto arch = config::getArch(origOp.getOperation());
    const auto ctx = origOp->getContext();

    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(copyOpInput.getType());
    auto copyOpInputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOpInput));
    auto copyOpOutputType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(copyOpOutput));

    const auto inReqs = StrideReqs::compact(copyOpInputType.getRank());
    auto isInStridedCopy = !inReqs.checkStrides(copyOpInputType);
    const auto isOutStridedCopy = !inReqs.checkStrides(copyOpOutputType);
    const auto wasInStridedCopy = isInStridedCopy;
    const auto isEffectivelyInStridedCopy = VPUIP::isEffectivelyStrided(copyOpInputType);
    auto copyOpInputTypeForCheck = copyOpInputType;

    // When the input type has non-compact strides but is not effectively strided
    // (e.g., only size-1 boundary dims differ), use compact strides for legality checks.
    if (isInStridedCopy && !isEffectivelyInStridedCopy && distributedType == nullptr) {
        const auto elemSize = vpux::getElemTypeSize(copyOpInputType.getElementType());
        const auto dimsOrder = copyOpInputType.getDimsOrder();
        const auto memShape = dimsOrder.toMemoryOrder(Shape(copyOpInputType.getShape()));
        const auto compactMemStrides = StrideReqs::compact(dimsOrder.numDims()).calcStrides(elemSize, memShape);
        const auto compactStrides = dimsOrder.toLogicalOrder(compactMemStrides);
        copyOpInputTypeForCheck = copyOpInputTypeForCheck.changeStrides(StridesRef(compactStrides));
        isInStridedCopy = !inReqs.checkStrides(copyOpInputTypeForCheck);
        VPUX_THROW_UNLESS(isInStridedCopy == false, "Failed to convert to compact strides");
    }

    if (isInStridedCopy) {
        if (distributedType || isOutStridedCopy) {
            _log.trace("Skip complex case: input is strided CMX or both input and output are strided");
            return mlir::failure();
        }
        //                   DDR2DDR - inStride
        //             |                        |
        //          DDR2CMX1                DDR2CMX2
        //                           =>
        //    DDR2CMX3 - inStride      DDR2CMX4 - inStride
        //
        // The latency of (DDR2DDR + DDR2CMX1 + DDR2CMX2) may be less than (DDR2CMX3 + DDR2CMX4), if in-stride is not
        // efficient as non-stride.
        //
        // For single use, it should reduce the latency.
        //                   DDR2DDR - inStride
        //                           |
        //                        DDR2CMX
        //                           =>
        //                    DDR2CMX inStride
        auto copyIsEfficient = vpux::VPUIP::isDDRCopyEfficient(copyOpInputTypeForCheck, arch);
        if (!origOp->hasOneUse() && !copyIsEfficient) {
            _log.trace("Skip complex case: input DDR Copy is not efficient as contiguous one");
            return mlir::failure();
        }
    }

    auto viewOpInputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    auto viewOpOutputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    auto viewOpInputShape = viewOpInputType.getShape();
    auto viewOpOutputShape = viewOpOutputType.getShape();

    auto isDuplicatedMode = [](auto mode) {
        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
               VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
    };

    auto isMovableNonDistributedViewLikeOp = [](mlir::Operation* op) {
        return mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::ShapeCastOp>(op);
    };

    auto isMovableDistributedViewLikeOp = [&](mlir::Operation* op, vpux::VPUIP::DistributedBufferType distType) {
        if (distType == nullptr) {
            return false;
        }

        const auto mode = distType.getDistribution().getMode().getValue();
        if (isDuplicatedMode(mode)) {
            return true;
        }

        const auto inShape = viewOpInputType.getShape();
        const auto outShape = viewOpOutputType.getShape();

        if (auto permuteOp = mlir::dyn_cast_if_present<VPUIP::PermuteCastOp>(op)) {
            const auto dstOrder = DimsOrder::fromAffineMap(permuteOp.getDstOrder());
            return vpux::isPermuteCastCompatible(mlir::cast<NDTypeInterface>(permuteOp.getSource().getType()),
                                                 mlir::cast<NDTypeInterface>(permuteOp.getResult().getType()),
                                                 dstOrder);
        }

        if (mlir::isa<VPUIP::QuantizeCastOp>(op)) {
            // Per-axis quantized type is not allowed
            const auto inElemType = viewOpInputType.getElementType();
            const auto outElemType = viewOpOutputType.getElementType();
            return !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inElemType) &&
                   !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outElemType);
        }

        if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(op)) {
            const auto inOrder = viewOpInputType.getDimsOrder();
            const auto outOrder = viewOpOutputType.getDimsOrder();
            const auto isSameMemShapeWithOptionalLeadingG = [](ShapeRef lhsShape, const DimsOrder& lhsOrder,
                                                               ShapeRef rhsShape, const DimsOrder& rhsOrder) {
                if (lhsOrder == rhsOrder) {
                    return true;
                }

                auto hasLeadingG = [](const DimsOrder& order) {
                    const auto perm = order.toPermutation();
                    return order.numDims() == 5 && !perm.empty() && perm.front() == Dim(0);
                };
                auto getEffectiveMemShape = [](ShapeRef shape, const DimsOrder& order) {
                    auto memShape = to_small_vector(order.toMemoryOrder(shape).raw());
                    memShape.erase(std::remove(memShape.begin(), memShape.end(), 1), memShape.end());
                    return memShape;
                };
                if (hasLeadingG(rhsOrder) || hasLeadingG(lhsOrder)) {
                    return getEffectiveMemShape(lhsShape, lhsOrder) == getEffectiveMemShape(rhsShape, rhsOrder);
                }

                return false;
            };

            return isSameMemShapeWithOptionalLeadingG(inShape, inOrder, outShape, outOrder);
        }

        return false;
    };

    auto canBeMoved = distributedType == nullptr ? isMovableNonDistributedViewLikeOp(origOp)
                                                 : isMovableDistributedViewLikeOp(origOp, distributedType);
    if (!canBeMoved) {
        _log.trace("The view-like op is not movable for current distribution");
        return mlir::failure();
    }

    bool isOverlappedResizeHoist = false;
    vpux::NDTypeInterface newViewOpOutputType = viewOpOutputType.changeMemSpace(copyOpInputType.getMemSpace());
    if (distributedType != nullptr) {
        auto getNewDistributionInfoAttr = [&]() -> VPU::DistributionInfoAttr {
            const auto origDistribution = distributedType.getDistribution();

            // For trivial PermuteCast, we can get dist info from `applyPermutationOnDistributionInfoAttr` directly
            if (auto permuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*origOp)) {
                auto inPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCast->getOperand(0).getType());
                auto outPermuteType = mlir::cast<vpux::NDTypeInterface>(permuteCast->getResult(0).getType());

                return VPU::applyPermutationOnDistributionInfoAttr(
                               distributedType, permuteCast.getMemPerm(), inPermuteType.getDimsOrder(),
                               outPermuteType.getDimsOrder(), inPermuteType.getShape(), outPermuteType.getShape())
                        .value_or(nullptr);
            }

            // For QuantizeCast, we keep the same distribution
            if (mlir::isa<VPUIP::QuantizeCastOp>(*origOp)) {
                return origDistribution;
            }

            const auto newDistInfo = VPUIP::tryInferNewDistInfoAfterShapeChange<VPUIP::DistributedBufferType>(
                    distributedType, viewOpOutputShape, viewOpOutputType.getDimsOrder(), arch);
            if (mlir::failed(newDistInfo)) {
                _log.trace("The new shape is not compatible with the original distribution");
                return nullptr;
            }

            const auto mode = origDistribution.getMode().getValue();
            if (isDuplicatedMode(mode)) {
                return newDistInfo.value();
            }

            const auto axesMapping = VPUIP::getDistributedAxesMappingAfterShapeChanged(
                    distributedType, viewOpOutputShape, viewOpOutputType.getDimsOrder(), origDistribution, _log);
            VPUX_THROW_UNLESS(mlir::succeeded(axesMapping) && axesMapping->first != -1 && axesMapping->second != -1,
                              "Failed to get axes mapping after shape changed");
            auto [inAxis, outAxis] = axesMapping.value();

            if (!canViewOpBeMovedBeforeCopy(copyOp, mode, inAxis, outAxis, viewOpOutputShape, arch)) {
                _log.trace("The view-like op cannot be moved before Copy because of restrictions");
                return nullptr;
            }

            if (mode == VPU::DistributionMode::OVERLAPPED && VPU::isOverlappedOverH(origDistribution)) {
                if (viewOpInputShape[Dim(inAxis)] != viewOpOutputShape[Dim(outAxis)]) {
                    // Resizing the tiling axis is only safe for halo-free OVERLAPPED distributions.
                    // The unified inference helper has already validated this condition.
                    isOverlappedResizeHoist = true;
                }
            }

            return newDistInfo.value();
        };
        auto newDistributionInfoAttr = getNewDistributionInfoAttr();
        if (newDistributionInfoAttr == nullptr) {
            return mlir::failure();
        }

        const auto order = mlir::AffineMapAttr::get(viewOpOutputType.getDimsOrder().toAffineMap(ctx));
        newViewOpOutputType =
                VPUIP::DistributedBufferType::get(ctx, viewOpOutputShape, viewOpOutputType.getElementType(), order,
                                                  distributedType.getMemSpace(), newDistributionInfoAttr);
    }

    // Update strides when original input was strided DDR and output contiguous
    // (or input contiguous and output strided)
    if (wasInStridedCopy) {
        std::optional<vpux::NDTypeInterface> strideUpdatedOutType;
        if (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp>(origOp)) {
            strideUpdatedOutType = VPUIP::updateStridesForReshape(copyOpInputType, newViewOpOutputType);
        } else if (mlir::isa<VPUIP::ShapeCastOp>(origOp)) {
            auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(*origOp);
            VPUX_THROW_WHEN(iface == nullptr, "ShapeCastOp does not inherit InferTypeOpInterface");
            SmallVector<mlir::Type> newTypes;
            const auto isLegal = iface.inferReturnTypes(ctx, origOp->getLoc(), mlir::ValueRange{copyOpInput},
                                                        origOp->getAttrDictionary(), origOp->getPropertiesStorage(),
                                                        origOp->getRegions(), newTypes)
                                         .succeeded();
            if (isLegal) {
                strideUpdatedOutType = mlir::cast<vpux::NDTypeInterface>(newTypes[0]);
            }
        } else if (mlir::isa<VPUIP::QuantizeCastOp>(origOp)) {
            const auto inputStrides = copyOpInputType.getStrides();
            strideUpdatedOutType = newViewOpOutputType.changeStrides(inputStrides);
        }
        if (strideUpdatedOutType.has_value()) {
            newViewOpOutputType = strideUpdatedOutType.value();
            _log.trace("Updated type strides {0} for reshape op {1}", newViewOpOutputType, origOp->getLoc());
        } else {
            _log.trace("Failed to update strides for reshape op {0}", origOp->getLoc());
            return mlir::failure();
        }
    } else if (isOutStridedCopy) {
        // !isInStridedCopy && isOutStridedCopy
        const auto& inputStrides = copyOpInputType.getStrides();
        newViewOpOutputType = newViewOpOutputType.changeStrides(inputStrides);
    }

    const auto newAllocType = viewOpOutputType.changeMemSpace(copyOpOutputType.getMemSpace());
    if (isOverlappedResizeHoist) {
        const auto eventualDestType = getEventualCopyDestinationType(origOp->getResult(0));
        if (eventualDestType != nullptr && VPUIP::isEffectivelyStrided(eventualDestType)) {
            _log.trace("Skip: OVERLAPPED resize hoist would let the Copy be fused into an effectively "
                       "strided destination '{0}'",
                       eventualDestType);
            return mlir::failure();
        }
    }

    _log.trace("Set new input for '{0}': parent op '{1}'", origOp->getName(), copyOpInput.getLoc());
    origOp->setOperand(0, copyOpInput);

    _log.trace("Set new result type for '{0}': '{1}'", origOp->getName(), newViewOpOutputType);
    origOp->getResult(0).setType(newViewOpOutputType);

    rewriter.setInsertionPointAfter(origOp);

    auto allocOp = VPUIP::allocateBuffersOfType(_log, copyOp->getLoc(), rewriter, newAllocType).front();
    auto newCopyOp = VPUIP::CopyOp::create(rewriter, copyOp->getLoc(), origOp->getResult(0), allocOp);

    _log.trace("Replace all uses of pure view-like op with new Copy op.");
    rewriter.replaceAllUsesExcept(origOp->getResult(0), newCopyOp->getResults()[0], newCopyOp);

    auto sourceOp = copyOpOutput.getDefiningOp();

    if (sourceOp != nullptr && sourceOp->getResult(0).use_empty()) {
        rewriter.eraseOp(sourceOp);
    }

    if (copyOp->getResult(0).use_empty()) {
        rewriter.eraseOp(copyOp);
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
    _log.trace("Got CopyOp at {0}", copyOp.getLoc());
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
            VPUIP::SubViewOp::create(rewriter, subViewOp->getLoc(), originOperand, subViewOp.getStaticOffsetsAttr(),
                                     subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

    auto subViewOpShape = getShape(newSubViewOp);
    auto allocOpDtype = mlir::cast<vpux::NDTypeInterface>(allocOp->getResult(0).getType());
    // Per-axis quantization must be aligned with the shape.
    const auto targetElemType = mlir::cast<vpux::NDTypeInterface>(newSubViewOp.getResult().getType()).getElementType();
    allocOp->getResult(0).setType(allocOpDtype.changeShapeElemType(subViewOpShape, targetElemType));

    // Set insertion point after the newly created SubView to ensure proper parent-child relationship
    rewriter.setInsertionPointAfter(newSubViewOp);
    auto newParentOp =
            VPUIP::CopyOp::create(rewriter, newSubViewOp->getLoc(), newSubViewOp->getResult(0), allocOp->getResult(0));
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
