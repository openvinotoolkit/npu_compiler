//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_per_axis_tile_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_per_axis_tile_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch37xx {

using namespace vpux;

SingleClusterPerAxisTileDMARewriter::SingleClusterPerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                         Logger log)
        : mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
    setDebugName("SingleClusterPerAxisTileDMARewriter");
}

mlir::LogicalResult SingleClusterPerAxisTileDMARewriter::matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Process PerAxisTileDMAOp: {0}", perAxisTileDMAOp);

    if (perAxisTileDMAOp.getTilesAttr() == nullptr && perAxisTileDMAOp.getAxisAttr() == nullptr) {
        return mlir::failure();
    }

    if (!isMultiClusterPerAxisTileDMA(perAxisTileDMAOp)) {
        return unroll(perAxisTileDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterPerAxisTileDMARewriter::unroll(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto vpurtTask = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = mlir::cast<vpux::NDTypeInterface>(perAxisTileDMAOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(perAxisTileDMAOp.getOutput().getType());
    Byte elemTypeSize = inType.getElemTypeSize();

    auto srcDeclBuff = perAxisTileDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = perAxisTileDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcDeclBuff.getType());
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstDeclBuff.getType());

    auto createSubPerAxisTileDMAOp = [&](ShapeRef subInShape, ShapeRef subOutShape, int64_t srcOffset,
                                         int64_t dstOffset, VPUIP::DMADescriptorAttr dmaDescriptor, int64_t port) {
        const auto dimOrder = DimsOrder::CHW;
        auto getStrides = [](ShapeRef shape, Byte elemTypeSize) -> Strides {
            const auto strides = SmallVector<vpux::Bit>{Bit(shape[Dim(1)] * shape[Dim(2)] * Bit(elemTypeSize).count()),
                                                        Bit(shape.back() * Bit(elemTypeSize).count()),
                                                        Bit(Bit(elemTypeSize).count())};
            return Strides(strides);
        };

        const auto strides = getStrides(subInShape, elemTypeSize);
        auto newSrcMemRef =
                vpux::getMemRefType(subInShape, srcType.getElementType(), dimOrder, srcType.getMemSpace(), strides);

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

        mlir::Type newDstType;
        if (auto dstDistributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(dstType)) {
            auto distributionAttr = dstDistributedType.getDistribution();
            VPUX_THROW_WHEN(
                    distributionAttr.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "Issues with unrolling PerAxisTileDMA; Buffer has distributed type != DUPLICATED after unroll");
            if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributionAttr)) {
                distributionAttr = VPU::getNonOverlappedDistributedAttr(
                        subOutShape, distributionAttr.getMode(), nullptr, distributionAttr.getNumClusters(), nullptr,
                        distributionAttr.getUniformDistributedSegments(), dstDeclBuff.getContext());
            }

            const auto layout = mlir::AffineMapAttr::get(dimOrder.toAffineMap(ctx));
            newDstType = VPUIP::DistributedBufferType::get(ctx, subOutShape.raw(), dstType.getElementType(), layout,
                                                           dstType.getMemSpace(), distributionAttr);
        } else {
            const auto strides = getStrides(subOutShape, elemTypeSize);
            newDstType = vpux::getMemRefType(subOutShape, dstType.getElementType(), dimOrder, dstType.getMemSpace(),
                                             strides);
        }

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

        _log.trace("Creating Sub-PerAxisTileDMAOp with inShape: {0} outShape: {1}, SrcMemory at {2}, DstMemory at {3}",
                   subInShape, subOutShape, newSrcBuff.getSection(), newDstBuff.getSection());

        VPURT::wrapIntoTaskOp<VPUIP::PerAxisTileDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), newSrcBuff,
                newDstBuff, vpux::getIntAttr(rewriter, port), nullptr, nullptr, dmaDescriptor,
                perAxisTileDMAOp.getIsOutOfOrderAttr(), perAxisTileDMAOp.getIsCriticalAttr(),
                perAxisTileDMAOp.getDmaHwpIdAttr(), perAxisTileDMAOp.getProfilingMetadataAttr());
    };

    _log.trace("Unroll PerAxisTileDMAOp {0}", perAxisTileDMAOp->getLoc());

    auto axis = perAxisTileDMAOp.getAxis();
    auto tiles = perAxisTileDMAOp.getTiles();
    VPUX_THROW_UNLESS(axis.has_value() && tiles.has_value(), "Cannot get PerAxisTile attribution");
    auto mergedShapes = VPUIP::getPerAxisTileDMAMergedShape(inType, outType, axis.value(), tiles.value());
    auto dmaDescriptorAttr = perAxisTileDMAOp.getDmaDescriptorAttr();
    auto portIsAlreadyAssigned = true;
    if (dmaDescriptorAttr == nullptr) {
        auto dmaDescriptorGenerator = VPUIP::PerAxisTileDmaDescriptorGenerator(ctx, _log);
        dmaDescriptorAttr = dmaDescriptorGenerator.generate(mergedShapes.first, mergedShapes.second, tiles.value(),
                                                            elemTypeSize.count());
        portIsAlreadyAssigned = false;
    }

    const auto arch = config::getArch(perAxisTileDMAOp);
    auto subInputShapes = VPUIP::getPerAxisTileDMASubShapes(arch, mergedShapes.first);
    auto subOutputShapes = VPUIP::getPerAxisTileDMASubShapes(arch, mergedShapes.second);
    VPUX_THROW_UNLESS(subInputShapes.size() == subOutputShapes.size(),
                      "Unexpected PerAxisTileDMA subInput '{0}' and subOutput '{1}' number", subInputShapes.size(),
                      subOutputShapes.size());

    auto srcOffset = srcDeclBuff.getByteOffset();
    auto dstOffset = dstDeclBuff.getByteOffset();
    for (size_t idx = 0; idx < subInputShapes.size(); idx++) {
        auto dmaPort = idx % _dmaPortCount;

        auto newDmaPort = portIsAlreadyAssigned ? perAxisTileDMAOp.getPort().value() : dmaPort;
        auto newDMADescriptorAttr = VPUIP::updateNumPlanes(dmaDescriptorAttr, subInputShapes[idx][Dim(0)]);
        createSubPerAxisTileDMAOp(subInputShapes[idx], subOutputShapes[idx], srcOffset, dstOffset, newDMADescriptorAttr,
                                  newDmaPort);

        srcOffset += subInputShapes[idx].totalSize() * elemTypeSize.count();
        dstOffset += subOutputShapes[idx].totalSize() * elemTypeSize.count();
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

UnrollPerAxisTileDMAStrategy::UnrollPerAxisTileDMAStrategy(int64_t dmaPortCount): _dmaPortCount(dmaPortCount) {
}

void UnrollPerAxisTileDMAStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::VPUIP::arch37xx::SingleClusterPerAxisTileDMARewriter>(ctx, _dmaPortCount, log);
    patterns.add<vpux::VPUIP::MultiClusterPerAxisTileDMARewriter>(ctx, _dmaPortCount, true, log);
}

}  // namespace vpux::VPUIP::arch37xx
