//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_SPLITDMATOBALANCELOAD
#define GEN_PASS_DEF_SPLITDMATOBALANCELOAD
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

using BuffersPair = std::pair<mlir::Value, mlir::Value>;

Shape getSplitShape(NDTypeInterface bufferType, vpux::Dim tileDim, int64_t newDimSize) {
    Shape subShape = bufferType.getShape().toValues();
    subShape[tileDim] = newDimSize;
    return subShape;
}

NDTypeInterface getNewBufferType(NDTypeInterface bufferType, vpux::Dim tileDim, int64_t dimOffset, int64_t newDimSize) {
    const auto newShape = getSplitShape(bufferType, tileDim, newDimSize);
    Shape newOffset(SmallVector<int64_t>(newShape.size(), 0));
    newOffset[tileDim] = dimOffset;

    NDTypeInterface newType;
    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(bufferType)) {
        const auto origDistAttr = distributedType.getDistribution();
        VPUX_THROW_UNLESS(VPU::isDuplicated(origDistAttr), "Only support DUPLICATED distributed buffer type");

        // When DistributionInfoAttr has explicit per cluster memory/compute shapes, recompute them for the new shape
        // Since changeShape is not appliable for explicit distribution
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistAttr)) {
            auto ctx = bufferType.getContext();
            auto duplicatedOutputMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    newShape, duplicatedOutputMode, nullptr, origDistAttr.getNumClusters(), nullptr,
                    origDistAttr.getUniformDistributedSegments(), ctx);

            auto newElemType = bufferType.getElementType();
            if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(bufferType.getElementType())) {
                newElemType = tileScalesAndZP(qType, newShape, newOffset);
            }

            auto order = mlir::AffineMapAttr::get(bufferType.getDimsOrder().toAffineMap(ctx));
            auto memSpace = mlir::cast<vpux::VPUIP::DistributedBufferType>(bufferType).getMemSpace();

            newType = VPUIP::DistributedBufferType::get(ctx, newShape.raw(), newElemType, order, memSpace,
                                                        newDistribution);

            return VPUIP::tileTypeSparsityCompression(newType, newOffset, newShape);
        }
    }

    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(bufferType.getElementType())) {
        auto newElemType = tileScalesAndZP(qType, newShape, newOffset);
        newType = bufferType.changeShapeElemType(newShape, newElemType);
    } else {
        newType = bufferType.changeShape(newShape);
    }
    return VPUIP::tileTypeSparsityCompression(newType, newOffset, newShape);
}

// Replace single allocation with 2 separate allocations. These allocations cover same memory range, but first points to
// the beginning of buffer, second points to the middle or place with offset
BuffersPair getReplacementBuffers(mlir::Value originalBuffer, vpux::Dim tileDim, mlir::OpBuilder builder) {
    const auto bufferType = mlir::cast<vpux::NDTypeInterface>(originalBuffer.getType());
    auto bufferOp = originalBuffer.getDefiningOp<VPURT::DeclareBufferOp>();

    auto origStrides = bufferType.getStrides();
    builder.setInsertionPoint(bufferOp);
    const auto getTiledBuf = [&](int64_t newOffset, int64_t newDimSize, int64_t extraOffset,
                                 StringRef locSuffix) -> mlir::Value {
        auto newType = getNewBufferType(bufferType, tileDim, newOffset, newDimSize);
        newType = newType.changeStrides(origStrides);

        auto newBufferOffset = bufferOp.getByteOffset() + extraOffset;
        const auto newLoc = takeOpLoc(bufferOp, locSuffix);

        return builder
                .create<VPURT::DeclareBufferOp>(newLoc, newType, bufferOp.getSectionAttr(),
                                                bufferOp.getSectionIndexAttr(), getIntAttr(builder, newBufferOffset),
                                                bufferOp.getSwizzlingKeyAttr())
                ->getResult(0);
    };

    const auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(bufferType, tileDim);
    const auto extraOffset = Byte(firstPartSize * origStrides[tileDim]).count();

    auto firstBuff = getTiledBuf(/*dimOffset=*/0, firstPartSize, /*byteOffset=*/0, "_first_part");
    auto secondBuff = getTiledBuf(firstPartSize, secondPartSize, extraOffset, "_second_part");

    return {firstBuff, secondBuff};
}

BuffersPair getConstantParts(mlir::Value originalConstant, vpux::Dim tileDim, mlir::OpBuilder builder) {
    auto cstOp = originalConstant.getDefiningOp<Const::DeclareOp>();

    const auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
    const auto origShape = cstType.getShape();
    builder.setInsertionPoint(cstOp);
    const auto createCstPart = [&](int64_t tileOffset, int64_t newDimSize, StringRef locSuffix) -> mlir::Value {
        Shape offset(SmallVector<int64_t>(origShape.size(), 0));
        offset[tileDim] = tileOffset;

        const auto newShape = getSplitShape(cstType, tileDim, newDimSize);
        const auto newLoc = takeOpLoc(cstOp, locSuffix);
        return builder.createOrFold<VPUIP::SubViewOp>(newLoc, cstOp, offset.raw(), newShape.raw());
    };

    const auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(cstType, tileDim);
    auto firstCst = createCstPart(0, firstPartSize, "_first_part");
    auto secondCst = createCstPart(firstPartSize, secondPartSize, "_second_part");

    return {firstCst, secondCst};
}

void replaceDmaWithTwoParts(VPURT::TaskOp taskOp, VPUIP::NNDMAOp dmaOp, BuffersPair inputs, BuffersPair outputs,
                            int64_t numDmaPorts, mlir::OpBuilder builder, vpux::Logger log) {
    builder.setInsertionPoint(taskOp);
    const auto insertNewDma = [&](mlir::Value input, mlir::Value output, int64_t newDmaPort, StringRef locSuffix) {
        const auto newLoc = takeOpLoc(taskOp, locSuffix);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                builder, taskOp.getWaitBarriers(), taskOp.getUpdateBarriers(), newLoc, input, output, newDmaPort,
                dmaOp.getIsOutOfOrder(), dmaOp.getIsCritical(), dmaOp.getSpillIdAttr(),
                /*compress_candidate=*/nullptr);  // split gives more improvement than compression
    };

    int64_t firstPartPort = dmaOp.getPort().value();
    int64_t secondPartPort = (firstPartPort + 1) % numDmaPorts;

    insertNewDma(inputs.first, outputs.first, firstPartPort, "_first_part");
    insertNewDma(inputs.second, outputs.second, secondPartPort, "_second_part");

    SmallVector<mlir::Value> oldArgs{dmaOp.getInput(), dmaOp.getOutputBuff()};
    taskOp->erase();
    for (mlir::Value prevArg : oldArgs) {
        mlir::Operation* bufferOp = prevArg.getDefiningOp();
        if (bufferOp->getUsers().empty()) {
            bufferOp->erase();
        }
    }
    log.trace("Replaced DMA with parts");
}

// Trivial constant is constant without LAST or PREFERRED_LAST transformation, so SubView transformation can be
// attached to the end of list
bool isTrivialConst(Const::DeclareOp cstOp) {
    const auto& contentAttr = cstOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();
    return transformations.empty() ||
           transformations.back().getPositionRequirement() == vpux::Const::details::PositionRequirement::NONE;
}

// Non trivial transforms requires folding and flattening to keep content correct
void splitFoldedConstToBufferDma(VPURT::TaskOp taskOp, VPUIP::NNDMAOp dmaOp, Const::DeclareOp cstOp, vpux::Dim tileDim,
                                 int64_t numDmaPorts, mlir::OpBuilder builder, vpux::Logger log) {
    const auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
    const auto strides = cstType.getStrides();
    // In case of subbyte type, which has non-byte stride along tiling dim - don't attempt to split this constant
    const auto tileDimStride = strides[tileDim];
    if (tileDimStride.count() % CHAR_BIT != 0) {
        log.trace("Can't split constant with non-byte stride");
        return;
    }
    auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(cstType, tileDim);
    if (firstPartSize != secondPartSize) {
        log.trace("Can't split constant with odd tiling dim");
        return;
    }

    const auto content = cstOp.getContent();
    const auto contentType = content.getType();
    const auto contentElemType = contentType.getElementType();
    const auto elemTypeBitSize = contentType.getElemTypeSize().count();
    const auto isUnsupportedSubByteStorageType = elemTypeBitSize < CHAR_BIT && elemTypeBitSize > 1;
    if (isUnsupportedSubByteStorageType) {
        log.trace("Can't split constant with unsupported element type");
        return;
    }
    log.trace("Splitting FoldedConst->Buffer DMA");
    const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(tempBuf.data(), bufSize));

    auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(contentType);
    if (auto qtype = mlir::dyn_cast<mlir::quant::QuantizedType>(contentElemType)) {
        rankedTensorType =
                mlir::cast<mlir::RankedTensorType>(contentType.changeElemType(normalizeQuantStorageType(qtype)));
    }

    const auto rankedElemType = rankedTensorType.getElementType();
    const auto fullShape = rankedTensorType.getShape();
    SmallVector<int64_t> newShapeVec(fullShape.begin(), fullShape.end());
    auto actualTileDims = getNonOneDim(ShapeRef(newShapeVec));
    if (actualTileDims.size() == 0) {
        log.trace("Can't split constant with all ones shape");
        return;
    }
    newShapeVec[actualTileDims[0].ind()] /= 2;

    builder.setInsertionPoint(cstOp);
    const auto createCstPart = [&](int64_t tileOffset, int64_t newDimSize, StringRef locSuffix) -> mlir::Value {
        const size_t offsetSize = Byte(tileOffset * tileDimStride).count();
        const size_t bufferSize = Byte(newDimSize * tileDimStride).count();
        char* baseContentPtr = tempBuf.data() + offsetSize;
        ArrayRef<char> partContent(baseContentPtr, baseContentPtr + bufferSize);
        const auto partRankedTensorType = rankedTensorType.clone(newShapeVec, rankedElemType);
        const auto denseAttr = Const::createConstContent(partRankedTensorType, partContent);
        const auto newLoc = takeOpLoc(cstOp, locSuffix);
        const auto newType = getNewBufferType(cstType, tileDim, tileOffset, newDimSize);

        return builder.create<Const::DeclareOp>(newLoc, newType, Const::ContentAttr::get(denseAttr));
    };

    auto firstCst = createCstPart(0, firstPartSize, "_first_part");
    auto secondCst = createCstPart(firstPartSize, secondPartSize, "_second_part");

    BuffersPair inputBuffers = {firstCst, secondCst};
    auto outputBuffers = getReplacementBuffers(dmaOp.getOutputBuff(), tileDim, builder);
    replaceDmaWithTwoParts(taskOp, dmaOp, inputBuffers, outputBuffers, numDmaPorts, builder, log);
}

void splitDmaIntoParts(VPURT::TaskOp taskOp, int64_t numDmaPorts, mlir::OpBuilder builder, vpux::Logger log) {
    auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
    const auto inputBuffType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
    const auto maybeTileDim = VPUIP::getCopyDMATilingDim(dmaOp);
    if (!maybeTileDim.has_value()) {
        log.trace("Can't find split dim for shape {0}, skip", inputBuffType.getShape());
        return;
    }
    const auto tileDim = maybeTileDim.value();

    BuffersPair inputBuffers;
    if (auto inputCst = dmaOp.getInput().getDefiningOp<Const::DeclareOp>()) {
        if (!isTrivialConst(inputCst)) {
            splitFoldedConstToBufferDma(taskOp, dmaOp, inputCst, tileDim, numDmaPorts, builder, log);
            return;
        }
        inputBuffers = getConstantParts(dmaOp.getInput(), tileDim, builder);
        log.trace("Splitting Const->Buffer DMA");
    } else {
        inputBuffers = getReplacementBuffers(dmaOp.getInput(), tileDim, builder);
        log.trace("Splitting Buffer->Buffer DMA");
    }

    auto outputBuffers = getReplacementBuffers(dmaOp.getOutputBuff(), tileDim, builder);
    replaceDmaWithTwoParts(taskOp, dmaOp, inputBuffers, outputBuffers, numDmaPorts, builder, log);
}

//
// SplitDMAToBalanceLoad
//

class SplitDMAToBalanceLoad final : public VPUIP::arch40xx::impl::SplitDMAToBalanceLoadBase<SplitDMAToBalanceLoad> {
public:
    explicit SplitDMAToBalanceLoad(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitDMAToBalanceLoad::safeRunOnFunc() {
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();
    if (dmaPortCount != 2) {
        return;
    }

    func->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return;
        }

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr) {
            return;
        }
        VPUX_THROW_UNLESS(dmaOp.getPort().has_value(), "DMA at '{0}' has no portId", dmaOp->getLoc());

        if (dmaOp.getSplitCandidateAttr() == nullptr) {
            return;
        }
        _log.trace("Found split candidate at '{0}'", dmaOp->getLoc());
        mlir::Operation* inputOp = dmaOp.getInput().getDefiningOp();
        if (!mlir::isa<VPURT::DeclareBufferOp, Const::DeclareOp>(inputOp)) {
            _log.warning("Can't split op because of unsupported source");
            return;
        }
        mlir::OpBuilder builder(taskOp.getOperation());
        splitDmaIntoParts(taskOp, dmaPortCount, builder, _log.nest());
    });
    _log.trace("Done");
}

}  // namespace

//
// createSplitDMAToBalanceLoadPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createSplitDMAToBalanceLoadPass(Logger log) {
    return std::make_unique<SplitDMAToBalanceLoad>(log);
}
