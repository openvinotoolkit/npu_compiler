//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_depth_to_space_dma_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/unroll_depth_to_space_dma.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPUIP::arch37xx {

SingleClusterDepthToSpaceDMARewriter::SingleClusterDepthToSpaceDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount,
                                                                           Logger log)
        : mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(std::move(log)) {
    setDebugName("SingleClusterDepthToSpaceDMARewriter");
}

mlir::LogicalResult SingleClusterDepthToSpaceDMARewriter::matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (!isMultiClusterDepthToSpaceDMAOp(depthToSpaceDMAOp)) {
        _log.trace("Got DepthToSpaceDMAOp '{0}' at '{1}'", depthToSpaceDMAOp->getName(), depthToSpaceDMAOp->getLoc());
        return unroll(depthToSpaceDMAOp, rewriter);
    }

    return mlir::failure();
}

mlir::LogicalResult SingleClusterDepthToSpaceDMARewriter::unroll(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    const auto inOrder = DimsOrder::fromValue(depthToSpaceDMAOp.getInput());
    const auto outOrder = DimsOrder::fromValue(depthToSpaceDMAOp.getOutputBuff());

    auto vpurtTask = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = mlir::cast<vpux::NDTypeInterface>(depthToSpaceDMAOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(depthToSpaceDMAOp.getOutput().getType());
    Byte elemTypeSize = inType.getElemTypeSize();

    if (depthToSpaceDMAOp.getDmaDescriptor().has_value()) {
        _log.nest().trace("This DepthToSpaceDMAOp has already been unrolled.");
        return mlir::failure();
    }

    const auto inputShape = getShape(depthToSpaceDMAOp.getInput());
    const auto outputShape = getShape(depthToSpaceDMAOp.getOutputBuff());

    const auto inputC = inputShape[Dims4D::Act::C];
    const auto inputH = inputShape[Dims4D::Act::H];
    const auto inputW = inputShape[Dims4D::Act::W];
    const auto outputC = outputShape[Dims4D::Act::C];
    const auto outputW = outputShape[Dims4D::Act::W];
    auto blockSize = depthToSpaceDMAOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize == 0, "Invalid block size: {0}", blockSize);
    auto mode = depthToSpaceDMAOp.getMode();
    auto paddedIC =
            depthToSpaceDMAOp.getPaddedChannels() ? depthToSpaceDMAOp.getPaddedChannels().value().getInput() : nullptr;
    auto paddedOC =
            depthToSpaceDMAOp.getPaddedChannels() ? depthToSpaceDMAOp.getPaddedChannels().value().getOutput() : nullptr;

    auto srcDeclBuff = depthToSpaceDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = depthToSpaceDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto srcType = mlir::cast<vpux::NDTypeInterface>(srcDeclBuff.getType());
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstDeclBuff.getType());

    auto createSubDepthToSpaceDMAOp = [&](ShapeRef subShape, DimsOrder order, int64_t srcOffset, int64_t dstOffset,
                                          VPUIP::DMADescriptorAttr dmaDescriptor, int64_t port) {
        SmallVector<vpux::Bit> newStrides;
        const auto dataBitSize = Bit(elemTypeSize).count();
        if (order == DimsOrder::NHWC) {
            newStrides = SmallVector<vpux::Bit>{
                    Bit(subShape[Dims4D::Act::H] * subShape[Dims4D::Act::W] * subShape[Dims4D::Act::C] * dataBitSize),
                    Bit(dataBitSize), Bit(subShape[Dims4D::Act::W] * subShape[Dims4D::Act::C] * dataBitSize),
                    Bit(subShape[Dims4D::Act::C] * dataBitSize)};
        }

        auto newSrcMemRef = vpux::getMemRefType(subShape, srcType.getElementType(), inOrder, srcType.getMemSpace(),
                                                StridesRef(newStrides));

        auto newSrcBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(),
                                                                  newSrcMemRef, srcDeclBuff.getSection(),
                                                                  srcType.getMemSpace().getIndex().value(), srcOffset);

        auto newDstMemRef = vpux::getMemRefType(subShape, dstType.getElementType(), outOrder, dstType.getMemSpace(),
                                                StridesRef(newStrides));

        auto newDstBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, dstDeclBuff, vpurtTask.getLoc(),
                                                                  newDstMemRef, dstDeclBuff.getSection(),
                                                                  dstType.getMemSpace().getIndex().value(), dstOffset);

        _log.nest().trace("Create Sub-DepthToSpaceDMAOp with shape: {0}, SrcMemory at {1}, DstMemory at {2}", subShape,
                          newSrcBuff.getSection(), newDstBuff.getSection());

        VPURT::wrapIntoTaskOp<VPUIP::DepthToSpaceDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), vpurtTask.getLoc(), newSrcBuff,
                newDstBuff, vpux::getIntAttr(rewriter, port), depthToSpaceDMAOp.getBlockSizeAttr(),
                depthToSpaceDMAOp.getModeAttr(), dmaDescriptor, nullptr, depthToSpaceDMAOp.getIsOutOfOrderAttr(),
                depthToSpaceDMAOp.getIsCriticalAttr(), depthToSpaceDMAOp.getDmaHwpIdAttr(),
                /* profilingMetadata= */ nullptr, /*internalDataFlow= */ nullptr);
    };

    _log.nest().trace("Unroll DepthToSpaceDMAOp {0}", depthToSpaceDMAOp->getLoc());

    auto dmaDescriptorGenerator = VPUIP::DepthToSpaceDmaDescriptorGenerator(ctx, _log);

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(depthToSpaceDMAOp));
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes() - 1;

    // inputH is the planes number, need to split if it exceed the max number.
    auto numberOfTile = divUp(inputH, dmaMaxNumPlanes);
    auto tileHSize = inputH / numberOfTile;

    for (int idx = 0; idx < numberOfTile; idx++) {
        int64_t subInputH;
        if (idx == (numberOfTile - 1)) {
            subInputH = inputH - tileHSize * idx;
        } else {
            subInputH = tileHSize;
        }
        auto lineSrcOffset = idx * tileHSize * inputC * inputW * elemTypeSize.count() + srcDeclBuff.getByteOffset();
        auto lineDstOffset =
                idx * tileHSize * blockSize * outputC * outputW * elemTypeSize.count() + dstDeclBuff.getByteOffset();
        Shape inputSubShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputC, subInputH, inputW});
        Shape outputSubShape =
                Shape(SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputC, subInputH * blockSize, outputW});

        auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, inputSubShape, outputSubShape, mode,
                                                             blockSize, paddedIC, paddedOC);
        auto depthToSpaceIndex = 0;

        if (inOrder == DimsOrder::NHWC && mode == IE::DepthToSpaceMode::BLOCKS_FIRST) {
            auto blockShape =
                    Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputC / blockSize, subInputH, inputW});
            auto srcOffset = lineSrcOffset;
            auto dstOffset = lineDstOffset;
            for (int bs = 0; bs < blockSize; bs++) {
                auto dmaPort = depthToSpaceIndex % _dmaPortCount;
                createSubDepthToSpaceDMAOp(blockShape, inOrder, srcOffset, dstOffset, dmaDescriptor, dmaPort);

                depthToSpaceIndex++;

                auto srcOffsetSize =
                        paddedOC != nullptr ? blockSize * (outputC - paddedOC.getInt()) : inputC / blockSize;
                srcOffset += srcOffsetSize * elemTypeSize.count();
                dstOffset += outputC * outputW * elemTypeSize.count();
            }
        } else if (inOrder == DimsOrder::NHWC && mode == IE::DepthToSpaceMode::DEPTH_FIRST) {
            auto blockShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], blockSize, subInputH, inputW});
            auto dstOffset = lineDstOffset;
            auto srcOffset = lineSrcOffset;
            for (int idx = 0; idx < inputC / blockSize; idx++) {
                auto dmaPort = depthToSpaceIndex % _dmaPortCount;
                createSubDepthToSpaceDMAOp(blockShape, inOrder, srcOffset, dstOffset, dmaDescriptor, dmaPort);

                srcOffset += blockSize * elemTypeSize.count();

                depthToSpaceIndex++;
                auto idxC = depthToSpaceIndex / blockSize;
                auto idxH = depthToSpaceIndex % blockSize;
                dstOffset =
                        lineDstOffset + idxC * elemTypeSize.count() + outputC * outputW * idxH * elemTypeSize.count();
            }
        } else {
            VPUX_THROW("Unsupported parameter order = {0}, mode = {1}", inOrder, mode);
        }
    }

    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

UnrollDepthToSpaceDMAStrategy::UnrollDepthToSpaceDMAStrategy(int64_t dmaPortCount): _dmaPortCount(dmaPortCount) {
}

void UnrollDepthToSpaceDMAStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::VPUIP::arch37xx::SingleClusterDepthToSpaceDMARewriter>(ctx, _dmaPortCount, log);
    patterns.add<vpux::VPUIP::MultiClusterDepthToSpaceDMARewriter>(ctx, _dmaPortCount, log);
}

}  // namespace vpux::VPUIP::arch37xx
