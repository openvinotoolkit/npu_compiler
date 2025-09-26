//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_NNDMATILING
#define GEN_PASS_DEF_NNDMATILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// SplitNNDMARewriter
//

class SplitNNDMARewriter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    SplitNNDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log, config::ArchKind arch)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _dmaPortCount(dmaPortCount), _arch(arch) {
        setDebugName("SplitNNDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void createTiles(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter, Logger log) const;

private:
    Logger _log;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
    config::ArchKind _arch;
};

Byte getDmaSize(VPUIP::NNDMAOp nndmaOp) {
    const auto inputShape = getShape(nndmaOp.getInput());
    const auto outputShape = getShape(nndmaOp.getOutput());
    VPUX_THROW_UNLESS(inputShape == outputShape,
                      "NNDMAOpTiling: NNDMAOp node's input and output have different shapes: {0} vs {1}", inputShape,
                      outputShape);

    // Sparse data is composed of multiple buffers which will later get ungrouped into individual Copy operations
    // Therefore, the maximum buffer size is selected for tiling
    if (auto sparseInput = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(nndmaOp.getInput().getType())) {
        auto dataSize = mlir::cast<vpux::NDTypeInterface>(sparseInput.getData()).getCompactAllocSize();
        auto sparsityMapSize =
                (sparseInput.getSparsityMap() != nullptr)
                        ? mlir::cast<vpux::NDTypeInterface>(sparseInput.getSparsityMap()).getCompactAllocSize()
                        : Byte(0);
        auto seTableSize =
                (sparseInput.getStorageElementTable() != nullptr)
                        ? mlir::cast<vpux::NDTypeInterface>(sparseInput.getStorageElementTable()).getCompactAllocSize()
                        : Byte(0);
        return std::max({dataSize, sparsityMapSize, seTableSize});
    }

    return static_cast<Byte>(getCompactSize(nndmaOp.getInput()));
}

void SplitNNDMARewriter::createTiles(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter, Logger log) const {
    // Currently, tiling is implemented only for 4D shapes.
    const auto origInputShape = getShape(nndmaOp.getInput());
    const auto origOutputShape = getShape(nndmaOp.getOutput());

    const auto fullCopySize = getDmaSize(nndmaOp);

    const auto maybeTileDim = VPUIP::getCopyDMATilingDim(nndmaOp);
    VPUX_THROW_UNLESS(maybeTileDim.has_value(), "Unable to find a dim to tile over it");
    auto tileDim = maybeTileDim.value();
    if (VPUIP::isSplitNeededForLargePlanesNum(nndmaOp)) {
        tileDim = VPUIP::getCopyDMATilingDimForLargePlaneNum(nndmaOp);
    }

    log.nest().trace("[{0}] tile on dim {1}", nndmaOp->getLoc(), tileDim);

    // We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;
    // The number of planes DMA could process within one tile. In case of small spatial dimensions of tensor (e.g.
    // 1x2048x8x8) it can exceed CMX_DMA_MAX_NUM_PLANES, so it's necessary to limit this value
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(_arch);
    const auto dmaMaxLength = dmaEngineLimits.getMaxLength();
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes() - 1;

    const auto desiredPlanesPerTileAmount = (dmaMaxLength / singlePlaneSize.count());
    VPUX_THROW_UNLESS(desiredPlanesPerTileAmount != 0,
                      "Couldn't split a NNDMAOp with single plane size greater than plane size limit");

    auto numPlanesPerTile = std::min(desiredPlanesPerTileAmount, dmaMaxNumPlanes);

    // Adjust numPlanesPerTile for even split, which provides better performance
    auto numTiles = numPlanesOfFullShape / numPlanesPerTile;
    if (numPlanesOfFullShape % numPlanesPerTile != 0) {
        numTiles++;
    }
    numPlanesPerTile = vpux::divUp(numPlanesOfFullShape, numTiles);

    auto inputDeclBuff = nndmaOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outputDeclBuff = nndmaOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(inputDeclBuff != nullptr && outputDeclBuff != nullptr,
                      "Can't get input or output buffer of NNDMAOp '{0}'", nndmaOp->getLoc());

    Byte inputOffset{inputDeclBuff.getByteOffset()};
    Byte outputOffset{outputDeclBuff.getByteOffset()};

    auto vpurtTask = nndmaOp->getParentOfType<VPURT::TaskOp>();
    rewriter.setInsertionPointAfter(vpurtTask);

    auto currentTileInShape = Shape(origInputShape.raw());
    auto currentTileOutShape = Shape(origOutputShape.raw());
    auto planesLeftToCopy = numPlanesOfFullShape;
    auto inputInsertionPoint = nndmaOp.getInput().getDefiningOp();
    auto outputInsertionPoint = nndmaOp.getOutputBuff().getDefiningOp();

    auto spillIdAttr = nndmaOp.getSpillIdAttr();

    const auto getTiledBuf = [](VPURT::DeclareBufferOp origBuf, vpux::ShapeRef subShape, vpux::Byte newOffset,
                                mlir::Operation* insertionPoint,
                                mlir::PatternRewriter& rewriter) -> VPURT::DeclareBufferOp {
        auto origType = mlir::cast<vpux::NDTypeInterface>(origBuf.getType());
        auto origStrides = origType.getStrides();
        auto newType = origType.changeShape(subShape);
        newType = newType.changeStrides(origStrides);

        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, origBuf, newType, newOffset.count());
    };

    for (int64_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(nndmaOp->getLoc(), "tile {0}", tileIdx);

        // The last tile may consume less number of planes for uneven split
        // e.g. Split 512 planes into 3 tiles
        // Tile 0: 171 planes
        // Tile 1: 171 planes
        // Tile 2: 170 planes
        currentTileInShape[tileDim] = std::min(planesLeftToCopy, numPlanesPerTile);
        currentTileOutShape[tileDim] = std::min(planesLeftToCopy, numPlanesPerTile);

        // Create new input buffer
        auto inputNewBuffer =
                getTiledBuf(inputDeclBuff, currentTileInShape, inputOffset, inputInsertionPoint, rewriter);
        inputInsertionPoint = inputNewBuffer.getResult().getDefiningOp();
        auto origInputStrides = mlir::cast<vpux::NDTypeInterface>(inputNewBuffer.getType()).getStrides();
        inputOffset += Byte(currentTileInShape[tileDim] * origInputStrides[tileDim]);

        // Create new output buffer
        auto outputNewBuffer =
                getTiledBuf(outputDeclBuff, currentTileOutShape, outputOffset, outputInsertionPoint, rewriter);
        outputInsertionPoint = outputNewBuffer.getResult().getDefiningOp();
        auto origOutputStrides = mlir::cast<vpux::NDTypeInterface>(outputDeclBuff.getType()).getStrides();
        outputOffset += Byte(currentTileOutShape[tileDim] * origOutputStrides[tileDim]);

        // Such distribution of tasks between ports may conflict with initial assumptions at memory scheduler level and
        // may potentially negatively impact previously prepared DMA tasks distribution and prefetching.
        auto newDMAPort = tileIdx % _dmaPortCount;
        const auto newNNDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), tileLoc, inputNewBuffer,
                outputNewBuffer, newDMAPort, false, false, spillIdAttr, nndmaOp.getCompressCandidateAttr());

        log.trace("New tile '{0}' NNDMA op: '{1}'", tileIdx, newNNDMA);

        planesLeftToCopy -= currentTileInShape[tileDim];
    }

    VPUX_THROW_UNLESS(planesLeftToCopy == 0, "SplitNNDMA: a part of the original shape was not covered by NNDMA tiles");

    rewriter.eraseOp(vpurtTask);
}

mlir::LogicalResult SplitNNDMARewriter::matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const {
    // Split NNDMAOp with large tensor size should no happen at the moment because NNDMAOp's data size can't be larger
    // than max plane size, see NNDMAOp::verify()
    // However, keep this part of code so that it looks similar to the processing logic in copy-op-tiling pass
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(_arch);
    const auto dmaMaxLength = dmaEngineLimits.getMaxLength();

    bool needSplitForLargeTensorSize = getDmaSize(nndmaOp) > Byte(dmaMaxLength);
    bool needSplitForLargePlanesNum = VPUIP::isSplitNeededForLargePlanesNum(nndmaOp);

    if (!needSplitForLargeTensorSize && !needSplitForLargePlanesNum) {
        return mlir::failure();
    }

    _log.trace("Process NNDMA op : {0}", nndmaOp);
    createTiles(nndmaOp, rewriter, _log);

    return mlir::success();
}

//
// NNDMATilingPass
//

class NNDMATilingPass final : public VPUIP::impl::NNDMATilingBase<NNDMATilingPass> {
public:
    explicit NNDMATilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void NNDMATilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);

    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitNNDMARewriter>(&ctx, dmaPortCount, _log, arch);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createNNDMATilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createNNDMATilingPass(Logger log) {
    return std::make_unique<NNDMATilingPass>(log);
}
