//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/dma_rewriter.hpp"
#include "vpux/compiler/conversion/passes/VPUIP2VPUMI40XX/buffer_conversion.hpp"

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"

using namespace vpux;

namespace {

bool enableMemorySideCache(NDTypeInterface inputType, NDTypeInterface outputType) {
    auto inputMemKind = inputType.getMemoryKind();
    auto outputMemKind = outputType.getMemoryKind();
    auto isDDR2CMX = inputMemKind == VPU::MemoryKind::CMX_NN && outputMemKind == VPU::MemoryKind::DDR;
    return isDDR2CMX;
}

uint32_t getListIndex(VPU::MemoryKind memoryKind) {
    static const llvm::SmallDenseMap<VPU::MemoryKind, uint32_t> memKind2Index = {{VPU::MemoryKind::DDR, 0},
                                                                                 {VPU::MemoryKind::CMX_NN, 1}};

    VPUX_THROW_UNLESS(memKind2Index.contains(memoryKind),
                      "Invalid DMA input/output memory kind, should be DDR or CMX_NN, but '{0}'", memoryKind);
    return memKind2Index.at(memoryKind);
}

}  // namespace

namespace vpux::vpuip2vpumi40xx {

mlir::LogicalResult NNDMARewriter::matchAndRewrite(VPUIP::NNDMAOp nnDMAOp, OpAdaptor adaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = nnDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto inputs = convertOrUnrollBuffer(rewriter, adaptor.getInput());
    assert(inputs.size() == 1);
    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());

    auto origOp = nnDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            nnDMAOp, indexType,
            nullptr,  // taskLocation
            inputs[0], dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            VPUMI40XX::NNDMATransactionAttr::get(ctx, inputType, outputType),
            nullptr,  // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            false,                   // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult PermuteDMARewriter::matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp, OpAdaptor adaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = permuteDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto internalDataFlow = adaptor.getInternalDataFlow();
    auto dmaTransaction = VPUMI40XX::PermuteDMATransactionAttr::get(
            ctx, internalDataFlow->getInputType(), internalDataFlow->getOutputType(),
            internalDataFlow->getMappingOrder(), internalDataFlow->getLoopOrder());

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = permuteDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            permuteDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,         // actCompressionSizeEntry
            nullptr,         // actCompressionSparsityMap
            dmaTransaction,  // dmaTransaction
            nullptr,         // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult ExpandDMARewriter::matchAndRewrite(VPUIP::ExpandDMAOp expandDMAOp, OpAdaptor adaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = expandDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    const auto dmaDescriptor = adaptor.getDmaDescriptor().value();
    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = expandDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            expandDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptor, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult ConvertDMARewriter::matchAndRewrite(VPUIP::ConvertDMAOp convertDMAOp, OpAdaptor adaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = convertDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = convertDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            convertDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            nullptr,  // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            false,                   // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult SpaceToDepthDMARewriter::matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                                             OpAdaptor adaptor,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = spaceToDepthDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());

    auto origOp = spaceToDepthDMAOp->getParentOfType<VPURT::TaskOp>();

    auto internalDataFlow = adaptor.getInternalDataFlow();
    auto dmaTransaction = VPUMI40XX::PermuteDMATransactionAttr::get(
            ctx, internalDataFlow->getInputType(), internalDataFlow->getOutputType(),
            internalDataFlow->getMappingOrder(), internalDataFlow->getLoopOrder());

    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            spaceToDepthDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,         // actCompressionSizeEntry
            nullptr,         // actCompressionSparsityMap
            dmaTransaction,  // dmaTransaction
            nullptr,         // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult DepthToSpaceDMARewriter::matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                                             OpAdaptor adaptor,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = depthToSpaceDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());

    auto origOp = depthToSpaceDMAOp->getParentOfType<VPURT::TaskOp>();

    auto internalDataFlow = adaptor.getInternalDataFlow();
    auto dmaTransaction = VPUMI40XX::PermuteDMATransactionAttr::get(
            ctx, internalDataFlow->getInputType(), internalDataFlow->getOutputType(),
            internalDataFlow->getMappingOrder(), internalDataFlow->getLoopOrder());

    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            depthToSpaceDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,         // actCompressionSizeEntry
            nullptr,         // actCompressionSparsityMap
            dmaTransaction,  // dmaTransaction
            nullptr,         // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult UpsamplingDMARewriter::matchAndRewrite(VPUIP::UpsamplingDMAOp upsamplingDMAOp, OpAdaptor adaptor,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = upsamplingDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    const auto dmaDescriptor = adaptor.getDmaDescriptor().value();
    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = upsamplingDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            upsamplingDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptor, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult PerAxisTileDMARewriter::matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp, OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = perAxisTileDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    const auto dmaDescriptor = adaptor.getDmaDescriptor().value();
    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = perAxisTileDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            perAxisTileDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptor, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult DecompressDMARewriter::matchAndRewrite(VPUIP::DecompressDMAOp decompressDMAOp, OpAdaptor adaptor,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = decompressDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = decompressDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            decompressDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DECOMPRESSION, adaptor.getActCompressionSizeEntry(),
            adaptor.getActCompressionSparsityMap(),
            nullptr,  // dmaTransaction
            nullptr,  // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult CompressDMARewriter::matchAndRewrite(VPUIP::CompressDMAOp compressDMAOp, OpAdaptor adaptor,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = compressDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = compressDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            compressDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::COMPRESSION, adaptor.getActCompressionSizeEntry(),
            adaptor.getActCompressionSparsityMap(),
            nullptr,  // dmaTransaction
            nullptr,  // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,                    // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult GatherDMARewriter::matchAndRewrite(VPUIP::GatherDMAOp gatherDMAOp, OpAdaptor adaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = gatherDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = gatherDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            gatherDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            nullptr,  // dmaDescriptor
            adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            true,  // allowDifferentInOutShapes
            adaptor.getIndices(),
            nullptr,                             // enqueueBarrier
            origOp.getWlmPageAttr(),             // wlmPageAttr
            nullptr,                             // physicalBarrierRangeAttr
            nullptr,                             // enqueueDMAAttr
            nullptr,                             // fetchDmaAttr
            gatherDMAOp.getAddressingModeAttr()  // addressingMode
    );
    return mlir::success();
}

mlir::LogicalResult SyncDMARewriter::matchAndRewrite(VPUIP::SyncDMAOp syncDMAOp, OpAdaptor adaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = syncDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto zeroAttr = getIntAttr(ctx, 0);
    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx,
                                                           zeroAttr,  // numPlane
                                                           zeroAttr,  // len
                                                           zeroAttr,  // srcWidth
                                                           zeroAttr,  // srcStride
                                                           zeroAttr,  // srcPlaneStride
                                                           zeroAttr,  // dstWidth
                                                           zeroAttr,  // dstStride
                                                           zeroAttr   // dstPlaneStride
    );

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = syncDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            syncDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptorAttr, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            false,                   // allowDifferentInOutShapes
            nullptr,                 // indices
            nullptr,                 // enqueueBarrier
            origOp.getWlmPageAttr()  // wlmPageAttr
    );

    return mlir::success();
}

mlir::LogicalResult ReadOnlyDMARewriter::matchAndRewrite(VPUIP::ReadOnlyDMAOp readOnlyDMAOp, OpAdaptor adaptor,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = readOnlyDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto inputs = convertOrUnrollBuffer(rewriter, adaptor.getInput());
    VPUX_THROW_WHEN(inputs.size() != 1, "Input size should be equal to 1");
    auto origOp = readOnlyDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(readOnlyDMAOp, indexType,
                                                    nullptr,             // taskLocation
                                                    inputs[0],           // input
                                                    mlir::ValueRange(),  // output_buffs
                                                    nullptr,             // previousTask
                                                    mlir::ValueRange(),  // waitBarriers
                                                    mlir::ValueRange(),  // updateBarriers
                                                    0,                   // startAfter
                                                    0,                   // cleanAfter
                                                    adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
                                                    false,  // enableMemorySideCache
                                                    tileIdx, VPUIP::DMAAccMode::DISABLE,
                                                    nullptr,  // actCompressionSizeEntry
                                                    nullptr,  // actCompressionSparsityMap
                                                    nullptr,  // dmaTransaction
                                                    nullptr,  // dmaDescriptor
                                                    adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
                                                    false,                   // allowDifferentInOutShapes
                                                    nullptr,                 // indices
                                                    nullptr,                 // enqueueBarrier
                                                    origOp.getWlmPageAttr()  // wlmPageAttr
    );
    return mlir::success();
}

mlir::LogicalResult BarrierProgDMARewriter::matchAndRewrite(VPUIP::BarProgDMAOp barProgDmaOp, OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = barProgDmaOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto zeroAttr = getIntAttr(ctx, 0);
    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx,
                                                           zeroAttr,  // numPlane
                                                           zeroAttr,  // len
                                                           zeroAttr,  // srcWidth
                                                           zeroAttr,  // srcStride
                                                           zeroAttr,  // srcPlaneStride
                                                           zeroAttr,  // dstWidth
                                                           zeroAttr,  // dstStride
                                                           zeroAttr   // dstPlaneStride
    );

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = barProgDmaOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            barProgDmaOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptorAttr, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            false,                                       // allowDifferentInOutShapes
            nullptr,                                     // indices
            nullptr,                                     // enqueueBarrier
            origOp.getWlmPageAttr(),                     // wlmPageAttr
            barProgDmaOp.getPhysicalBarrierRangeAttr(),  // physicalBarrierRangeAttr
            nullptr,                                     // enqueueDMAAttr
            nullptr,                                     // fetchDMAAttr
            nullptr                                      // addressingMode
    );

    return mlir::success();
}

mlir::LogicalResult FetchDMARewriter::matchAndRewrite(VPUIP::FetchDMAOp fetchDMAOp, OpAdaptor adaptor,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = fetchDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto zeroAttr = getIntAttr(ctx, 0);
    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx,
                                                           zeroAttr,  // numPlane
                                                           zeroAttr,  // len
                                                           zeroAttr,  // srcWidth
                                                           zeroAttr,  // srcStride
                                                           zeroAttr,  // srcPlaneStride
                                                           zeroAttr,  // dstWidth
                                                           zeroAttr,  // dstStride
                                                           zeroAttr   // dstPlaneStride
    );

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = fetchDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            fetchDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptorAttr, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            false,                         // allowDifferentInOutShapes
            nullptr,                       // indices
            nullptr,                       // enqueueBarrier
            origOp.getWlmPageAttr(),       // wlmPageAttr
            nullptr,                       // physicalBarrierRangeAttr
            nullptr,                       // enqueueDMAAttr
            fetchDMAOp.getFetchDmaAttr(),  // fetchDmaAttr
            nullptr                        // addressingMode
    );

    return mlir::success();
}

mlir::LogicalResult EnqueueDMARewriter::matchAndRewrite(VPUIP::EnqueueDMAOp enqueueDMAOp, OpAdaptor adaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = enqueueDMAOp.getContext();

    auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(adaptor.getOutputBuff().getType());

    const auto tileIdx = adaptor.getPort().value();
    auto indexType = VPURegMapped::IndexType::get(ctx, tileIdx, getListIndex(inputType.getMemoryKind()), 0);

    auto zeroAttr = getIntAttr(ctx, 0);
    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx,
                                                           zeroAttr,  // numPlane
                                                           zeroAttr,  // len
                                                           zeroAttr,  // srcWidth
                                                           zeroAttr,  // srcStride
                                                           zeroAttr,  // srcPlaneStride
                                                           zeroAttr,  // dstWidth
                                                           zeroAttr,  // dstStride
                                                           zeroAttr   // dstPlaneStride
    );

    auto dmaResults = convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff());
    auto origOp = enqueueDMAOp->getParentOfType<VPURT::TaskOp>();
    rewriter.replaceOpWithNewOp<VPUMI40XX::NNDMAOp>(
            enqueueDMAOp, indexType,
            nullptr,  // taskLocation
            adaptor.getInput(), dmaResults,
            nullptr,             // previousTask
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            0,                   // startAfter
            0,                   // cleanAfter
            adaptor.getIsOutOfOrder(), adaptor.getIsCritical(),
            _isMemorySideCacheEnabled && enableMemorySideCache(inputType, outputType), tileIdx,
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // actCompressionSizeEntry
            nullptr,  // actCompressionSparsityMap
            nullptr,  // dmaTransaction
            dmaDescriptorAttr, adaptor.getDmaHwpIdAttr(), adaptor.getProfilingMetadataAttr(),
            false,                             // allowDifferentInOutShapes
            nullptr,                           // indices
            nullptr,                           // enqueueBarrier
            origOp.getWlmPageAttr(),           // wlmPageAttr
            nullptr,                           // physicalBarrierRangeAttr
            enqueueDMAOp.getEnqueueDmaAttr(),  // enqueueDMAAttr
            nullptr,                           // fetchDmaAttr
            nullptr                            // addressingMode
    );

    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
