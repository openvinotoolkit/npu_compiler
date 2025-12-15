//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_permute_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_permute_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP::arch37xx {

SingleClusterPermuteDMARewriter::SingleClusterPermuteDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                 Logger log)
        : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
    setDebugName("SingleClusterPermuteDMARewriter");
}

mlir::LogicalResult SingleClusterPermuteDMARewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    // Skip PermuteDMA ops which have been unrolled by checking mem_perm attribute
    if (permuteDMAOp.getMemPermAttr() == nullptr) {
        return mlir::failure();
    }

    if (!isMultiClusterPermuteDMA(permuteDMAOp)) {
        _log.trace("Got PermuteDMAOp '{0}' at '{1}'", permuteDMAOp->getName(), permuteDMAOp->getLoc());
        return unroll(permuteDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterPermuteDMARewriter::unroll(VPUIP::PermuteDMAOp permuteDMAOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto vpurtTask = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto srcDeclBuff = permuteDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(srcDeclBuff != nullptr, "Can't get buffer for operand: {0}", permuteDMAOp.getInput());

    auto dstDeclBuff = permuteDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto inType = mlir::cast<vpux::NDTypeInterface>(permuteDMAOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(permuteDMAOp.getOutput().getType());
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcDeclBuff.getType());
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstDeclBuff.getType());
    auto srcOffset = srcDeclBuff.getByteOffset();
    auto dstOffset = dstDeclBuff.getByteOffset();

    // For unrolled DMA which is inside of cluster tiling, the dma descriptor is already calculated
    auto dmaDescriptorAttr = permuteDMAOp.getDmaDescriptorAttr();
    const auto memPerm = permuteDMAOp.getMemPerm().value();
    auto mergedMemPerm = VPUIP::getPermuteDMAMergedMemPerm(inType, memPerm);
    auto numPlaneDim = VPUIP::getPermuteDMANumPlaneDim(inType, memPerm);

    auto portIsAlreadyAssigned = true;
    if (dmaDescriptorAttr == nullptr) {
        auto ctx = permuteDMAOp->getContext();
        auto mergedInputShape = VPUIP::getPermuteDMAInputShape(inType, outType, memPerm, _log).value();
        auto mergedOutputShape = VPUIP::getPermuteDMAOutputShape(inType, outType, memPerm, _log).value();
        auto dmaDescriptorGenerator = VPUIP::PermuteDmaDescriptorGenerator(ctx, mergedMemPerm, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedInputShape, mergedOutputShape, elemTypeSize);
        portIsAlreadyAssigned = false;
    }

    auto subInput = VPUIP::getPermuteDMASubInputShapes(config::getArch(permuteDMAOp), inType, outType, memPerm,
                                                       _dmaPortCount, _log);
    VPUX_THROW_UNLESS(subInput.has_value(), "Cannot get unrolled subInputShapes for PermuteDMA op {0}", permuteDMAOp);
    auto subInputShapes = subInput.value();
    auto subOutputShapes = VPUIP::getPermuteDMASubOutputShapes(subInputShapes, inType, outType, memPerm);

    _log.trace("Unrolling PermuteDMAOp '{0}' at '{1}'", permuteDMAOp->getName(), permuteDMAOp->getLoc());

    int64_t dmaPort = 0;
    SmallVector<VPUIP::PermuteDMAOp> firstPermuteDMAsOnPorts;
    SmallVector<VPUIP::PermuteDMAOp> lastPermuteDMAsOnPorts;
    SmallVector<VPUIP::PermuteDMAOp> newPermuteDMAs;
    for (auto idx = 0; idx < checked_cast<int64_t>(subInputShapes.size()); idx++) {
        auto newDMADescriptorAttr = VPUIP::updateNumPlanes(dmaDescriptorAttr, subInputShapes[idx][numPlaneDim]);

        const auto dimOrder = (subInputShapes[0].size() == 2) ? DimsOrder::NC : DimsOrder::CHW;
        auto newSrcStrides =
                (subInputShapes[idx].size() == 2)
                        ? SmallVector<vpux::Bit>{Bit(subInputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())}
                        : SmallVector<vpux::Bit>{Bit(subInputShapes[idx][Dim(1)] * subInputShapes[idx][Dim(2)] *
                                                     Bit(elemTypeSize).count()),
                                                 Bit(subInputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};

        auto newSrcMemRef = vpux::getMemRefType(subInputShapes[idx], srcType.getElementType(), dimOrder,
                                                srcType.getMemSpace(), StridesRef(newSrcStrides));

        auto newSrcBuff =
                srcType.getMemSpace().getIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                  newSrcMemRef, srcDeclBuff.getSection(),
                                                                  srcType.getMemSpace().getIndex().value(), srcOffset)
                : srcDeclBuff.getSectionIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                  rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef, srcDeclBuff.getSection(),
                                  parseIntArrayAttr<int64_t>(srcDeclBuff.getSectionIndex().value()), srcOffset)
                        : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                  newSrcMemRef, srcDeclBuff.getSection(), srcOffset);

        auto newDstStrides =
                (subOutputShapes[idx].size() == 2)
                        ? SmallVector<vpux::Bit>{Bit(subOutputShapes[idx].back() * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())}
                        : SmallVector<vpux::Bit>{Bit(subOutputShapes[idx][Dim(1)] * subOutputShapes[idx][Dim(2)] *
                                                     Bit(elemTypeSize).count()),
                                                 Bit(subOutputShapes[idx][Dim(2)] * Bit(elemTypeSize).count()),
                                                 Bit(Bit(elemTypeSize).count())};
        mlir::Type newDstType;
        if (auto dstDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(dstType)) {
            auto ctx = permuteDMAOp->getContext();
            auto distributionAttr = dstDistributedType.getDistribution();
            VPUX_THROW_WHEN(
                    distributionAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "Issues with unrolling PermuteNNDMA; Buffer has distributed type != DUPLICATED after unroll");
            if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
                distributionAttr = VPU::getNonOverlappedDistributedAttr(
                        subOutputShapes[idx], distributionAttr.getMode(), nullptr, distributionAttr.getNumClusters(),
                        nullptr, distributionAttr.getUniformDistributedSegments(), dstDeclBuff.getContext());
            }

            const auto layout = mlir::AffineMapAttr::get(dimOrder.toAffineMap(ctx));

            // Although in the current implementation strides are compact and having the correct DMADescriptorAttr is
            // all that is needed for further lowering, apply the strides to the new type nonetheless as the
            // implementation may change in the future.
            newDstType = mlir::cast<NDTypeInterface>(VPUIP::DistributedBufferType::get(
                                                             ctx, subOutputShapes[idx].raw(), dstType.getElementType(),
                                                             layout, dstType.getMemSpace(), distributionAttr))
                                 .changeStrides(StridesRef(newDstStrides));
        } else {
            newDstType = vpux::getMemRefType(subOutputShapes[idx], dstType.getElementType(), dimOrder,
                                             dstType.getMemSpace(), StridesRef(newDstStrides));
        }

        VPUX_THROW_UNLESS(dstType.getMemSpace().getIndex().has_value() || dstDeclBuff.getSectionIndex().has_value(),
                          "No section index find at '{}'", dstDeclBuff.getLoc());
        auto newDstBuff =
                dstType.getMemSpace().getIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                                                  dstDeclBuff.getSection(),
                                                                  dstType.getMemSpace().getIndex().value(), dstOffset)
                : dstDeclBuff.getSectionIndex().has_value()
                        ? VPURT::createOp<VPURT::DeclareBufferOp>(
                                  rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType, dstDeclBuff.getSection(),
                                  parseIntArrayAttr<int64_t>(dstDeclBuff.getSectionIndex().value()), dstOffset)
                        : VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstType,
                                                                  dstDeclBuff.getSection(), dstOffset);

        _log.trace("Create unrolled PermuteDMA operation with input/output shape: {0}/{1}, SrcMemory at {2}, "
                   "DstMemory at {3}",
                   subInputShapes[idx], subOutputShapes[idx], newSrcBuff.getSection(), newDstBuff.getSection());

        const auto newLoc = appendLoc(vpurtTask->getLoc(), "_unrolled_permuteDMA");
        auto newDmaPort = portIsAlreadyAssigned ? permuteDMAOp.getPort().value() : dmaPort;
        auto newPermuteDMAOp = VPURT::wrapIntoTaskOp<VPUIP::PermuteDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, newSrcBuff, newDstBuff,
                vpux::getIntAttr(rewriter, newDmaPort), permuteDMAOp.getIsOutOfOrderAttr(),
                permuteDMAOp.getIsCriticalAttr(),
                /*mem_perm*/ nullptr, newDMADescriptorAttr, permuteDMAOp.getDmaHwpIdAttr(),
                permuteDMAOp.getProfilingMetadataAttr(), /*internalDataFlow=*/nullptr);

        newPermuteDMAs.push_back(newPermuteDMAOp);

        // find the first and last DMAs on different ports
        if (firstPermuteDMAsOnPorts.size() < static_cast<size_t>(_dmaPortCount)) {
            firstPermuteDMAsOnPorts.push_back(newPermuteDMAOp);
            lastPermuteDMAsOnPorts.push_back(newPermuteDMAOp);
        } else {
            lastPermuteDMAsOnPorts[newDmaPort] = newPermuteDMAOp;
        }

        dmaPort = (dmaPort + 1) % _dmaPortCount;

        auto numPlaneValue = newDMADescriptorAttr.getNumPlanes().getInt();
        auto srcPlaneStrideValue = newDMADescriptorAttr.getSrcPlaneStride().getInt();
        auto dstPlaneStrideValue = newDMADescriptorAttr.getDstPlaneStride().getInt();
        srcOffset += numPlaneValue * srcPlaneStrideValue;
        dstOffset += numPlaneValue * dstPlaneStrideValue;
    }

    for (auto& dmaOp : newPermuteDMAs) {
        auto vpurtTask = dmaOp->getParentOfType<VPURT::TaskOp>();

        // remove wait barrier dependency for these new permute DMA except first ones on each port
        if (std::find(firstPermuteDMAsOnPorts.begin(), firstPermuteDMAsOnPorts.end(), dmaOp) ==
            firstPermuteDMAsOnPorts.end()) {
            vpurtTask.getWaitBarriersMutable().clear();
        }

        // remove update barrier dependency for these new permute DMA except last ones on each port
        if (std::find(lastPermuteDMAsOnPorts.begin(), lastPermuteDMAsOnPorts.end(), dmaOp) ==
            lastPermuteDMAsOnPorts.end()) {
            vpurtTask.getUpdateBarriersMutable().clear();
        }
    }

    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

UnrollPermuteDMAStrategy::UnrollPermuteDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount)
        : _ctx(ctx), _dmaPortCount(dmaPortCount) {
}

void UnrollPermuteDMAStrategy::addPatterns(llvm::SmallVector<mlir::RewritePatternSet>& patterns, Logger& log) const {
    mlir::RewritePatternSet patternSet1(_ctx);
    patternSet1.add<vpux::VPUIP::MultiClusterPermuteDMARewriter>(_ctx, _dmaPortCount, log);
    mlir::RewritePatternSet patternSet2(_ctx);
    patternSet2.add<vpux::VPUIP::arch37xx::SingleClusterPermuteDMARewriter>(_ctx, _dmaPortCount, log);
    patterns.push_back(std::move(patternSet1));
    patterns.push_back(std::move(patternSet2));
}

}  // namespace vpux::VPUIP::arch37xx
