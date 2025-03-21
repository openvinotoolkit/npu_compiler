//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch37xx {
#define GEN_PASS_DECL_UNROLLSPACETODEPTHDMA
#define GEN_PASS_DEF_UNROLLSPACETODEPTHDMA
#include "vpux/compiler/NPU37XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch37xx

using namespace vpux;

namespace {

//
// SpaceToDepthDMARewriter
//

class SpaceToDepthDMARewriter final : public mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp> {
public:
    SpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp>(ctx), _dmaPortCount(dmaPortCount), _log(log) {
        setDebugName("SpaceToDepthDMARewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult matchAndRewriteClusterDMA(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                  mlir::PatternRewriter& rewriter) const;

private:
    void unrollBlocksFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                    mlir::PatternRewriter& rewriter) const;
    void unrollBlocksFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                    mlir::PatternRewriter& rewriter) const;
    void unrollBlocksFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                    mlir::PatternRewriter& rewriter) const;
    void unrollDepthFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                   mlir::PatternRewriter& rewriter) const;
    void unrollDepthFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                   mlir::PatternRewriter& rewriter) const;
    void unrollDepthFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                   mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollSegmentedOrOverlapped(VPUIP::SpaceToDepthDMAOp spaceToDepthOp,
                                                    VPUIP::DistributedBufferType distributedType,
                                                    mlir::PatternRewriter& rewriter) const;

    void createSpaceToDepthDMASubOp(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask, ShapeRef subShape,
                                    int64_t srcOffset, int64_t dstOffset, VPUIP::DMADescriptorAttr dma_descriptor,
                                    int64_t port, mlir::PatternRewriter& rewriter) const;

private:
    int64_t _dmaPortCount;
    Logger _log;
};

void SpaceToDepthDMARewriter::unrollBlocksFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                         mlir::PatternRewriter& rewriter) const {
    auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

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

void SpaceToDepthDMARewriter::unrollBlocksFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                         mlir::PatternRewriter& rewriter) const {
    auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto inShape = inType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();
    const auto dmaPort = origOp.getPort();

    auto srcOffset = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();
    auto dstOffset = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset();

    auto dmaDescriptorGenerator = VPUIP::SpaceToDepthDmaDescriptorGenerator(getContext(), _log);
    auto dmaDescriptor = dmaDescriptorGenerator.generate(inType, outType, mode, blockSize);

    createSpaceToDepthDMASubOp(origOp, vpurtTask, inShape, srcOffset, dstOffset, dmaDescriptor, dmaPort.value_or(0),
                               rewriter);
}

void SpaceToDepthDMARewriter::unrollBlocksFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                         mlir::PatternRewriter& rewriter) const {
    auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

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

void SpaceToDepthDMARewriter::unrollDepthFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                        mlir::PatternRewriter& rewriter) const {
    auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

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

void SpaceToDepthDMARewriter::unrollDepthFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                        mlir::PatternRewriter& rewriter) const {
    auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

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

void SpaceToDepthDMARewriter::unrollDepthFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                        mlir::PatternRewriter& rewriter) const {
    auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const Byte elemTypeSize = inType.getElemTypeSize();
    const auto inShape = inType.getShape();
    const auto blockSize = origOp.getBlockSize();
    const auto mode = origOp.getMode();

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

void SpaceToDepthDMARewriter::createSpaceToDepthDMASubOp(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                                         ShapeRef subShape, int64_t srcOffset, int64_t dstOffset,
                                                         VPUIP::DMADescriptorAttr dma_descriptor, int64_t port,
                                                         mlir::PatternRewriter& rewriter) const {
    auto srcDeclBuff = origOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto dstDeclBuff = origOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

    auto srcType = srcDeclBuff.getType().cast<vpux::NDTypeInterface>();
    auto dstType = dstDeclBuff.getType().cast<vpux::NDTypeInterface>();

    auto newSrcMemRef = srcType.changeShape(subShape).cast<mlir::MemRefType>();
    auto newSrcBuff = VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                                              srcDeclBuff.getSection(), srcOffset);
    auto srcMemSpaceIndex = srcType.getMemSpace().getIndex();
    if (srcMemSpaceIndex.has_value()) {
        newSrcBuff =
                VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, srcDeclBuff, vpurtTask.getLoc(), newSrcMemRef,
                                                        srcDeclBuff.getSection(), srcMemSpaceIndex.value(), srcOffset);
    }

    auto newDstMemRef = dstType.changeShape(subShape).cast<mlir::MemRefType>();
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
            dma_descriptor, origOp.getIsOutOfOrderAttr(), origOp.getIsCriticalAttr(), origOp.getDmaHwpIdAttr(),
            origOp.getProfilingMetadataAttr());
}

mlir::LogicalResult SpaceToDepthDMARewriter::unrollSegmentedOrOverlapped(VPUIP::SpaceToDepthDMAOp spaceToDepthOp,
                                                                         VPUIP::DistributedBufferType distributedType,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto loc = spaceToDepthOp->getLoc();
    auto ctx = spaceToDepthOp->getContext();

    const auto input = spaceToDepthOp.getInput();
    const auto output = spaceToDepthOp.getOutputBuff();

    const auto inputType = spaceToDepthOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = distributedType.getCompactType().cast<vpux::NDTypeInterface>();

    const auto distributionAttr = distributedType.getDistribution();
    const auto distMode = distributionAttr.getMode().getValue();

    VPUX_THROW_UNLESS(distMode == VPU::DistributionMode::SEGMENTED || distMode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distribution mode: {0}", distMode);

    const auto blockSize = spaceToDepthOp.getBlockSize();

    const auto perClusterOutShapes = distributedType.getPerClusterMemoryShapes();
    const auto perClusterOutShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    const auto backInferInputShape = [&](ShapeRef outShape, int64_t blockSize) {
        auto inShape = Shape(outShape.raw());
        inShape[Dims4D::Act::H] *= blockSize;
        inShape[Dims4D::Act::W] *= blockSize;
        inShape[Dims4D::Act::C] /= (blockSize * blockSize);
        return inShape;
    };

    const auto origStrides = inputType.getStrides();
    const auto numClusters = perClusterOutShapes.size();

    SmallVector<NDTypeInterface> inTypes(numClusters);
    SmallVector<NDTypeInterface> outTypes(numClusters);
    for (auto clusterId : irange(numClusters)) {
        inTypes[clusterId] =
                inputType
                        .extractDenseTile(backInferInputShape(perClusterOutShapeOffsets[clusterId], blockSize),
                                          backInferInputShape(perClusterOutShapes[clusterId], blockSize))
                        .changeStrides(origStrides);
        outTypes[clusterId] =
                outputType.extractDenseTile(perClusterOutShapeOffsets[clusterId], perClusterOutShapes[clusterId]);
    }

    auto vpurtTask = spaceToDepthOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", spaceToDepthOp);

    rewriter.setInsertionPointAfter(vpurtTask);

    const auto getInputOperand = [&](mlir::Value operand, vpux::NDTypeInterface newType,
                                     mlir::Operation* insertionPoint, Byte offset) -> mlir::Value {
        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        Byte cmxOffset{declBuff.getByteOffset()};
        cmxOffset += offset;

        auto declBuffType = declBuff.getType().cast<vpux::NDTypeInterface>();
        VPUX_THROW_UNLESS(declBuffType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                          "Currently only support input in CMX");
        auto sectionIndex = declBuffType.getMemSpace().getIndex();
        VPUX_THROW_UNLESS(sectionIndex.has_value() && sectionIndex.value() == 0,
                          "Currently only support input in CMX0");

        auto section = declBuff.getSection();
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)), 0);
        newType = newType.changeMemSpace(symbolAttr);
        return VPURT::createOp<VPURT::DeclareBufferOp>(rewriter, insertionPoint, loc, newType, section,
                                                       cmxOffset.count());
    };

    const auto getOutputOperand = [&](int64_t clusterId, mlir::Value operand, vpux::NDTypeInterface newType,
                                      mlir::Operation* insertionPoint) -> mlir::Value {
        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
        auto newCMXType = newType.changeMemSpace(symbolAttr);

        return VPURT::createOp<VPURT::DeclareBufferOp>(
                rewriter, insertionPoint, spaceToDepthOp->getLoc(), newCMXType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(ctx, ArrayRef({clusterId})), declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
    };

    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tilingDim = Dim(VPU::getDistributedTilingAxis(tilingScheme));

    for (auto clusterId : irange(numClusters)) {
        const auto newInputType = inTypes[clusterId];
        const auto newOutType = outTypes[clusterId];

        const auto cmxOffset =
                perClusterOutShapeOffsets[clusterId][tilingDim] * static_cast<Byte>(newOutType.getStrides()[tilingDim]);
        const auto inputBuffer = getInputOperand(input, newInputType, inputInsertionPoint, cmxOffset);

        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getOutputOperand(clusterId, output, newOutType, outputInsertionPoint);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        const auto dmaPort = clusterId % _dmaPortCount;
        auto newSpaceToDepthDMAOp = VPURT::wrapIntoTaskOp<VPUIP::SpaceToDepthDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffer, outBuffer,
                vpux::getIntAttr(rewriter, dmaPort), spaceToDepthOp.getBlockSizeAttr(), spaceToDepthOp.getModeAttr(),
                /*dma_descriptor*/ nullptr, spaceToDepthOp.getIsOutOfOrderAttr(), spaceToDepthOp.getIsCriticalAttr(),
                spaceToDepthOp.getDmaHwpIdAttr(), spaceToDepthOp.getProfilingMetadataAttr());

        _log.trace("Insert new SpaceToDepthDMA: '{0}'", newSpaceToDepthDMAOp);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

mlir::LogicalResult SpaceToDepthDMARewriter::matchAndRewriteClusterDMA(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SpaceToDepthDMA with DistributedType: {0}", spaceToDepthDMAOp);

    const auto input = spaceToDepthDMAOp.getInput();
    const auto output = spaceToDepthDMAOp.getOutputBuff();

    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(inputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                              outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                      "Unexpected memory space: input {0}, output {1}", inputType.getMemoryKind(),
                      outputType.getMemoryKind());

    const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_WHEN(distributedType == nullptr, "Expect distributed type for SpaceToDepthDMA op output, but got: {0}",
                    outputType);

    const auto distributionAttr = distributedType.getDistribution();
    VPUX_THROW_WHEN(distributionAttr == nullptr, "Failed to extract distributon attribute from distributed type.");

    const auto modeAttr = distributionAttr.getMode();
    VPUX_THROW_WHEN(modeAttr == nullptr, "Failed to extract mode from distribution attribute.");
    const auto mode = modeAttr.getValue();

    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                      "Unsupported distribution mode: {0}", modeAttr);
    return unrollSegmentedOrOverlapped(spaceToDepthDMAOp, distributedType, rewriter);
}

mlir::LogicalResult SpaceToDepthDMARewriter::matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto outputType = spaceToDepthDMAOp.getOutputBuff().getType();
    if (auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>()) {
        return matchAndRewriteClusterDMA(spaceToDepthDMAOp, rewriter);
    }

    _log.trace("Get SpaceToDepthDMAOp : {0}", spaceToDepthDMAOp->getLoc());

    if (spaceToDepthDMAOp.getDmaDescriptor().has_value()) {
        _log.trace("This SpaceToDepthDMAOp has already been unrolled.");
        return mlir::failure();
    }

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

//
// UnrollSpaceToDepthDMAPass
//

class UnrollSpaceToDepthDMAPass final :
        public VPUIP::arch37xx::impl::UnrollSpaceToDepthDMABase<UnrollSpaceToDepthDMAPass> {
public:
    explicit UnrollSpaceToDepthDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSpaceToDepthDMAPass::safeRunOnFunc() {
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollSpaceToDepthDMAPass)) {
        return;
    }
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SpaceToDepthDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollSpaceToDepthDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch37xx::createUnrollSpaceToDepthDMAPass(Logger log) {
    return std::make_unique<UnrollSpaceToDepthDMAPass>(log);
}
