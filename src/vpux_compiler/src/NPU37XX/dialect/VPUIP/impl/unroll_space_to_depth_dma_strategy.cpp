//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_space_to_depth_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_space_to_depth_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"

namespace vpux::VPUIP::arch37xx {

SingleClusterSpaceToDepthDMARewriter::SingleClusterSpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                           Logger log)
        : mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
    setDebugName("SingleClusterSpaceToDepthDMARewriter");
}

mlir::LogicalResult SingleClusterSpaceToDepthDMARewriter::matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (spaceToDepthDMAOp.getDmaDescriptor().has_value()) {
        _log.trace("This SpaceToDepthDMAOp has already been unrolled.");
        return mlir::failure();
    }

    if (!isMultiClusterSpaceToDepthDMAOp(spaceToDepthDMAOp)) {
        _log.trace("Got SpaceToDepthDMAOp '{0}' at '{1}'", spaceToDepthDMAOp->getName(), spaceToDepthDMAOp->getLoc());
        return unroll(spaceToDepthDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterSpaceToDepthDMARewriter::unroll(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto blockSize = spaceToDepthDMAOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    auto vpurtTask = spaceToDepthDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.setInsertionPointAfter(vpurtTask);

    const auto mode = spaceToDepthDMAOp.getMode();
    const auto inOrder = DimsOrder::fromValue(spaceToDepthDMAOp.getInput());
    const auto outOrder = DimsOrder::fromValue(spaceToDepthDMAOp.getOutput());

    _log.trace("Unroll SpaceToDepthDMAOp {0}", spaceToDepthDMAOp->getLoc());

    if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NCHW && mode == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        unrollBlocksFirstNCHW2NCHW(spaceToDepthDMAOp, vpurtTask, rewriter);
    } else if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NCHW && mode == IE::SpaceToDepthMode::DEPTH_FIRST) {
        unrollDepthFirstNCHW2NCHW(spaceToDepthDMAOp, vpurtTask, rewriter);
    } else if (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC &&
               mode == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        unrollBlocksFirstNHWC2NHWC(spaceToDepthDMAOp, vpurtTask, rewriter);
    } else if (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC && mode == IE::SpaceToDepthMode::DEPTH_FIRST) {
        unrollDepthFirstNHWC2NHWC(spaceToDepthDMAOp, vpurtTask, rewriter);
    } else if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC &&
               mode == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        unrollBlocksFirstNCHW2NHWC(spaceToDepthDMAOp, vpurtTask, rewriter);
    } else if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC && mode == IE::SpaceToDepthMode::DEPTH_FIRST) {
        unrollDepthFirstNCHW2NHWC(spaceToDepthDMAOp, vpurtTask, rewriter);
    } else {
        VPUX_THROW("SpaceToDepthDMA layout '{0}->{1}' mode {2} is not supported yet.", inOrder, outOrder, mode);
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

void SingleClusterSpaceToDepthDMARewriter::unrollBlocksFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp,
                                                                      vpux::VPURT::TaskOp vpurtTask,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto mode = origOp.getMode();
    const auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    const auto IC = inShape[Dims4D::Act::C];
    const auto IW = inShape[Dims4D::Act::W];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto spaceToDepthIndex = 0;
    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);
    auto subShape = Shape(SmallVector<int64_t>{inShape[Dims4D::Act::N], 1, blockSize, IW});
    for (int ic = 0; ic < IC; ic++) {
        for (int oh = 0; oh < OH; oh++) {
            auto dmaPort = spaceToDepthIndex % _dmaPortCount;
            createSpaceToDepthDMASubOp(origOp, vpurtTask, subShape, srcOffset, dstOffset, dmaDescriptor, dmaPort,
                                       rewriter);

            spaceToDepthIndex++;
            srcOffset += IW * blockSize * elemTypeSize.count();
            dstOffset += OW * elemTypeSize.count();
        }
    }
}

void SingleClusterSpaceToDepthDMARewriter::unrollBlocksFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp,
                                                                      vpux::VPURT::TaskOp vpurtTask,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto inShape = inType.getShape();
    const auto mode = origOp.getMode();
    const auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);

    const auto dmaPort = origOp.getPort();
    createSpaceToDepthDMASubOp(origOp, vpurtTask, inShape, srcOffset, dstOffset, dmaDescriptor, dmaPort.value_or(0),
                               rewriter);
}

void SingleClusterSpaceToDepthDMARewriter::unrollBlocksFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp,
                                                                      vpux::VPURT::TaskOp vpurtTask,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto mode = origOp.getMode();
    const auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    const auto IC = inShape[Dims4D::Act::C];
    const auto IW = inShape[Dims4D::Act::W];
    const auto OC = outShape[Dims4D::Act::C];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto spaceToDepthIndex = 0;
    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);
    auto subShape = Shape(SmallVector<int64_t>{inShape[Dims4D::Act::N], 1, blockSize, IW});
    for (int ic = 0; ic < IC; ic++) {
        auto startDstIdx = dstOffset;
        for (int oh = 0; oh < OH; oh++) {
            auto dmaPort = spaceToDepthIndex % _dmaPortCount;
            createSpaceToDepthDMASubOp(origOp, vpurtTask, subShape, srcOffset, dstOffset, dmaDescriptor, dmaPort,
                                       rewriter);

            spaceToDepthIndex++;
            srcOffset += IW * blockSize * elemTypeSize.count();
            dstOffset += OC * OW * elemTypeSize.count();
        }
        dstOffset = startDstIdx + elemTypeSize.count();
    }
}

void SingleClusterSpaceToDepthDMARewriter::unrollDepthFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp,
                                                                     vpux::VPURT::TaskOp vpurtTask,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto mode = origOp.getMode();
    const auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    const auto IC = inShape[Dims4D::Act::C];
    const auto IH = inShape[Dims4D::Act::H];
    const auto IW = inShape[Dims4D::Act::W];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto spaceToDepthIndex = 0;
    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);
    auto subShape = Shape(SmallVector<int64_t>{inShape[Dims4D::Act::N], 1, blockSize, IW});
    for (int ic = 0; ic < IC; ic++) {
        auto startDstIdx = dstOffset;
        for (int oh = 0; oh < OH; oh++) {
            auto dmaPort = spaceToDepthIndex % _dmaPortCount;
            createSpaceToDepthDMASubOp(origOp, vpurtTask, subShape, srcOffset, dstOffset, dmaDescriptor, dmaPort,
                                       rewriter);

            spaceToDepthIndex++;
            srcOffset += IW * blockSize * elemTypeSize.count();
            dstOffset += OW * elemTypeSize.count();
        }
        dstOffset = startDstIdx + IW * IH * elemTypeSize.count();
    }
}

void SingleClusterSpaceToDepthDMARewriter::unrollDepthFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp,
                                                                     vpux::VPURT::TaskOp vpurtTask,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto mode = origOp.getMode();
    const auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    const auto IC = inShape[Dims4D::Act::C];
    const auto IW = inShape[Dims4D::Act::W];
    const auto OC = outShape[Dims4D::Act::C];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto spaceToDepthIndex = 0;
    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);
    auto subShape = Shape(SmallVector<int64_t>{inShape[Dims4D::Act::N], IC, 1, IW});
    for (int oh = 0; oh < OH; oh++) {
        auto startDstIdx = dstOffset;
        for (int bs = 0; bs < blockSize; bs++) {
            auto dmaPort = spaceToDepthIndex % _dmaPortCount;
            createSpaceToDepthDMASubOp(origOp, vpurtTask, subShape, srcOffset, dstOffset, dmaDescriptor, dmaPort,
                                       rewriter);

            spaceToDepthIndex++;
            srcOffset += IW * IC * elemTypeSize.count();
            dstOffset += blockSize * elemTypeSize.count();
        }
        dstOffset = startDstIdx + OW * OC * elemTypeSize.count();
    }
}

void SingleClusterSpaceToDepthDMARewriter::unrollDepthFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp,
                                                                     vpux::VPURT::TaskOp vpurtTask,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto mode = origOp.getMode();
    const auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize <= 0,
                    "Unrolling failed; block size ({0}) of SpaceToDepthDMAOp is not a positive integer.", blockSize);

    const auto IC = inShape[Dims4D::Act::C];
    const auto IH = inShape[Dims4D::Act::H];
    const auto IW = inShape[Dims4D::Act::W];

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto spaceToDepthIndex = 0;
    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);
    auto subShape = Shape(SmallVector<int64_t>{inShape[Dims4D::Act::N], 1, IH, IW});
    for (int ic = 0; ic < IC; ic++) {
        auto dmaPort = spaceToDepthIndex % _dmaPortCount;
        createSpaceToDepthDMASubOp(origOp, vpurtTask, subShape, srcOffset, dstOffset, dmaDescriptor, dmaPort, rewriter);

        spaceToDepthIndex++;
        srcOffset += IW * IH * elemTypeSize.count();
        dstOffset += blockSize * blockSize * elemTypeSize.count();
    }
}

void SingleClusterSpaceToDepthDMARewriter::createSpaceToDepthDMASubOp(VPUIP::SpaceToDepthDMAOp origOp,
                                                                      vpux::VPURT::TaskOp vpurtTask, ShapeRef subShape,
                                                                      int64_t srcOffset, int64_t dstOffset,
                                                                      VPUIP::DMADescriptorAttr dmaDescriptor,
                                                                      int64_t port,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto srcDeclBuff = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcDeclBuff.getType());
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstDeclBuff.getType());

    auto newSrcMemRef = mlir::cast<mlir::MemRefType>(srcType.changeShape(subShape));
    auto newSrcBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                                              srcDeclBuff.getSection(), srcOffset);
    auto srcMemSpaceIndex = srcType.getMemSpace().getIndex();
    if (srcMemSpaceIndex.has_value()) {
        newSrcBuff =
                VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                                        srcDeclBuff.getSection(), srcMemSpaceIndex.value(), srcOffset);
    }

    auto newDstMemRef = mlir::cast<mlir::MemRefType>(dstType.changeShape(subShape));
    auto newDstBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstMemRef,
                                                              dstDeclBuff.getSection(), dstOffset);
    auto dstMemSpaceIndex = dstType.getMemSpace().getIndex();
    if (dstMemSpaceIndex.has_value()) {
        newDstBuff =
                VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(), newDstMemRef,
                                                        dstDeclBuff.getSection(), dstMemSpaceIndex.value(), dstOffset);
    }

    _log.trace("Create Sub-SpaceToDepthDMAOp with shape: {0}, SrcMemory at {1}, DstMemory at {2}, on port {3}",
               subShape, newSrcBuff.getSection(), newDstBuff.getSection(), port);

    VPURT::wrapIntoTaskOp<VPUIP::SpaceToDepthDMAOp>(
            rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), newSrcBuff,
            newDstBuff, vpux::getIntAttr(rewriter, port), origOp.getBlockSizeAttr(), origOp.getModeAttr(),
            dmaDescriptor, origOp.getIsOutOfOrderAttr(), origOp.getIsCriticalAttr(), origOp.getDmaHwpIdAttr(),
            origOp.getProfilingMetadataAttr(), /*internalDataFlow= */ nullptr);
}

UnrollSpaceToDepthDMAStrategy::UnrollSpaceToDepthDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount)
        : _ctx(ctx), _dmaPortCount(dmaPortCount) {
}

void UnrollSpaceToDepthDMAStrategy::addPatterns(llvm::SmallVector<mlir::RewritePatternSet>& patterns,
                                                Logger& log) const {
    mlir::RewritePatternSet patternSet1(_ctx);
    patternSet1.add<vpux::VPUIP::MultiClusterSpaceToDepthDMARewriter>(_ctx, _dmaPortCount, log);
    mlir::RewritePatternSet patternSet2(_ctx);
    patternSet2.add<vpux::VPUIP::arch37xx::SingleClusterSpaceToDepthDMARewriter>(_ctx, _dmaPortCount, log);
    patterns.push_back(std::move(patternSet1));
    patterns.push_back(std::move(patternSet2));
}

}  // namespace vpux::VPUIP::arch37xx
