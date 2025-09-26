//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_COPYOPTILING
#define GEN_PASS_DEF_COPYOPTILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

Byte getDmaSize(VPUIP::CopyOp copyOp) {
    const auto inputShape = getShape(copyOp.getInput());
    const auto outputShape = getShape(copyOp.getOutput());
    VPUX_THROW_UNLESS(inputShape == outputShape,
                      "CopyOpTiling: Copy node's input and output have different shapes: {0} vs {1}", inputShape,
                      outputShape);
    return static_cast<Byte>(getCompactSize(copyOp.getInput()));
}

bool isLegalCopyOp(VPUIP::CopyOp copyOp) {
    // Distributed type is currently not needed as large DMAs to CMX are already handled by previous tiling pass and
    // size of CMX is nevertheless smaller then DMA limit
    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return true;
    }

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(copyOp));

    // If tensor size is greater than max plane size, its no longer legal operation
    if (getDmaSize(copyOp) > Byte(dmaEngineLimits.getMaxLength())) {
        return false;
    }

    return !VPUIP::isSplitNeededForLargePlanesNum(copyOp);
};

bool isLegalCopyOpWithConcatOp(VPUIP::CopyOp copyOp) {
    if (vpux::VPUIP::hasDistributedOperand(copyOp)) {
        return true;
    }

    // pattern copyOp + concatOp, and concat axis is the lowest Dim
    if (copyOp.getOutput().getUsers().empty()) {
        return true;
    }
    auto childConcatOp = mlir::dyn_cast_or_null<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin());
    if (childConcatOp == nullptr) {
        return true;
    }

    const auto concatInputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getOutput().getType());
    const auto concatOutputType = mlir::cast<vpux::NDTypeInterface>(childConcatOp.getOutput().getType());
    const auto concatInShape = concatInputType.getShape();
    const auto concatOutShape = concatOutputType.getShape();
    SmallVector<Dim> concatDims;
    for (auto idx : irange(concatInShape.size())) {
        if (concatInShape[Dim(idx)] != concatOutShape[Dim(idx)]) {
            concatDims.push_back(Dim(idx));
        }
    }
    if (concatDims.size() != 1 || concatDims[0] != concatInputType.getDimsOrder().dimAt(concatInShape.size() - 1)) {
        return true;
    }

    // For case tiling the copyOp with constOp input, trigger the canonicalizer issue on windows
    // Tracked by: #E120989
    if (copyOp->getOperand(0).getDefiningOp<Const::DeclareOp>() != nullptr) {
        return true;
    }

    // Input has level 1 stride, or Output has level 1 stride
    int64_t inputStridingLevel = VPUIP::getStridingLevel(copyOp->getOperand(0));
    int64_t outputStridingLevel = VPUIP::getStridingLevel(copyOp->getResult(0));
    if ((inputStridingLevel == 0 && outputStridingLevel == 0) || inputStridingLevel > 1 || outputStridingLevel > 1) {
        return true;
    }

    // Handle the single one copyOp case
    for (auto concatInput : childConcatOp.getInputs()) {
        auto op = mlir::dyn_cast_or_null<VPUIP::CopyOp>(concatInput.getDefiningOp());
        if (op != nullptr && !vpux::VPUIP::hasDistributedOperand(op) && op != copyOp) {
            return true;
        }
    }

    return false;
};

//
// CopyOpTilingPass
//

class CopyOpTilingPass final : public VPUIP::impl::CopyOpTilingBase<CopyOpTilingPass> {
public:
    explicit CopyOpTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// CopyOpTiling
//

// Splits large CopyOps into a bunch of smaller ones to fit DMA capabilities
class CopyOpTiling final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpTiling(mlir::MLIRContext* ctx, Logger log, config::ArchKind arch, int64_t dmaPortNum)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log), _arch(arch), _dmaPortNum(dmaPortNum) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> createTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
    config::ArchKind _arch;
    int64_t _dmaPortNum;
};

SmallVector<mlir::Value> CopyOpTiling::createTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto origInputShape = getShape(origOp.getInput());

    const auto fullCopySize = getDmaSize(origOp);

    const auto maybeTileDim = VPUIP::getCopyDMATilingDim(origOp);
    VPUX_THROW_UNLESS(maybeTileDim.has_value(), "Unable to find a dim to tile over it");
    auto tileDim = maybeTileDim.value();
    if (VPUIP::isSplitNeededForLargePlanesNum(origOp)) {
        tileDim = VPUIP::getCopyDMATilingDimForLargePlaneNum(origOp);
    }

    // We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    // Example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but it's obviously
    //          impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numPlanesOfFullShape = origInputShape[tileDim];
    const auto singlePlaneSize = fullCopySize / numPlanesOfFullShape;
    //  The number of planes DMA could process within one tile. In case of small spatial dimensions of tensor (e.g.
    // 1x2048x8x8) it can exceed CMX_DMA_MAX_NUM_PLANES, so it's necessary to limit this value
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(_arch);

    const auto dmaMaxLength = dmaEngineLimits.getMaxLength();
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes() - 1;

    const int64_t numPlanesPerTile = std::clamp(dmaMaxLength / singlePlaneSize.count(), int64_t(1), dmaMaxNumPlanes);

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>(origInputShape.size(), 0);
    auto currentTileShapeVector = to_small_vector(origInputShape);
    const auto planeDivideFactor = (numPlanesOfFullShape + numPlanesPerTile - 1) / numPlanesPerTile;

    auto needBalanceNum = planeDivideFactor % _dmaPortNum;
    if (needBalanceNum == 0) {
        needBalanceNum = _dmaPortNum;
    }
    // Here want to make the last dmaPortNum's DMA balance
    // e.g. When numPlanesPerTile = 256, dmaPortNum = 2
    //   If numPlanesOfFullShape = 512
    //      planeDivideFactor is: 2
    //      needBalanceNum is: 2
    //      The tile result will be 256, 256
    //   If numPlanesOfFullShape = 513
    //      planeDivideFactor is: 3
    //      needBalanceNum is: 1
    //    The tile result will be 256, 256, 1
    //   If numPlanesOfFullShape = 514
    //      planeDivideFactor is: 3
    //      needBalanceNum is: 1
    //      The tile result will be 256, 256, 1, 1
    //   If numPlanesOfFullShape = 515
    //      planeDivideFactor is: 3
    //      needBalanceNum is: 1
    //      The tile result will be 256, 256, 2, 1
    //   If numPlanesOfFullShape = 516
    //      planeDivideFactor is: 3
    //      needBalanceNum is: 1
    //      The tile result will be 256, 256, 2, 2
    auto tileDimVale = SmallVector<int64_t>(planeDivideFactor - needBalanceNum, numPlanesPerTile);
    auto reservedCopySize = numPlanesOfFullShape - ((planeDivideFactor - needBalanceNum) * numPlanesPerTile);
    auto reservedSingleCopySize = reservedCopySize / _dmaPortNum;
    auto reservedSingleCopyBiasNum = reservedCopySize % _dmaPortNum;
    tileDimVale.append(SmallVector<int64_t>(reservedSingleCopyBiasNum, reservedSingleCopySize + 1));
    if (reservedSingleCopySize) {
        tileDimVale.append(SmallVector<int64_t>(_dmaPortNum - reservedSingleCopyBiasNum, reservedSingleCopySize));
    }

    for (const auto tileIdx : irange(tileDimVale.size())) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), "tile {0}", tileIdx);
        currentTileShapeVector[tileDim.ind()] = tileDimVale[tileIdx];

        // Create the operations for it
        auto inputSubView =
                rewriter.create<VPUIP::SubViewOp>(tileLoc, origOp.getInput(), currentOffset, currentTileShapeVector);
        auto outputSubView = rewriter.create<VPUIP::SubViewOp>(tileLoc, origOp.getOutputBuff(), currentOffset,
                                                               currentTileShapeVector);
        auto copyTile = rewriter.create<VPUIP::CopyOp>(tileLoc, inputSubView.getResult(), outputSubView.getResult());

        concatInputs.push_back(copyTile.getOutput());
        _log.nest().trace("Created tile #{0} for {1} planes that requires {2}", tileIdx,
                          currentTileShapeVector[tileDim.ind()], getDmaSize(copyTile));

        // Take into account the part of the original tensor covered with the newly created tile
        currentOffset[tileDim.ind()] += currentTileShapeVector[tileDim.ind()];
    }

    VPUX_THROW_UNLESS(currentOffset[tileDim.ind()] == numPlanesOfFullShape,
                      "CopyOpTiling: a part of the original shape was not covered by Copy tiles {0} != {1}",
                      currentOffset[tileDim.ind()], numPlanesOfFullShape);

    return concatInputs;
}

mlir::LogicalResult CopyOpTiling::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Copy Operation '{0}'", origOp->getLoc());

    if (isLegalCopyOp(origOp)) {
        return mlir::failure();
    }

    const auto concatInputs = createTiles(origOp, rewriter);

    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, concatInputs, origOp.getOutputBuff());
    return mlir::success();
}

//
// SingleCopyOpWithConcatOpUserTiling
//

class SingleCopyOpWithConcatOpUserTiling final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    SingleCopyOpWithConcatOpUserTiling(mlir::MLIRContext* ctx, Logger log, int64_t dmaPortNum)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log), _dmaPortNum(dmaPortNum) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> createWithParallelTiles(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const;

    Logger _log;
    int64_t _dmaPortNum;
};

SmallVector<mlir::Value> SingleCopyOpWithConcatOpUserTiling::createWithParallelTiles(
        VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto origInputShape = getShape(origOp.getInput());
    const auto maybeTileDim = VPUIP::getCopyDMATilingDim(origOp);
    auto tileDim = maybeTileDim.value();
    const auto tileDimSize = origInputShape[tileDim];

    SmallVector<mlir::Value> concatInputs;
    auto currentOffset = SmallVector<int64_t>(origInputShape.size(), 0);
    auto currentTileShapeVector = to_small_vector(origInputShape);

    auto singleCopySize = tileDimSize / _dmaPortNum;
    auto singleCopyBiasNum = tileDimSize % _dmaPortNum;
    auto tileDimVale = SmallVector<int64_t>(singleCopyBiasNum, singleCopySize + 1);
    tileDimVale.append(SmallVector<int64_t>(_dmaPortNum - singleCopyBiasNum, singleCopySize));

    for (const auto tileIdx : irange(tileDimVale.size())) {
        // Get the proper shape and a new location for the tile
        const auto tileLoc = appendLoc(origOp->getLoc(), "tile {0}", tileIdx);
        currentTileShapeVector[tileDim.ind()] = tileDimVale[tileIdx];

        // Create the operations for it
        auto inputSubView =
                rewriter.create<VPUIP::SubViewOp>(tileLoc, origOp.getInput(), currentOffset, currentTileShapeVector);
        auto outputSubView = rewriter.create<VPUIP::SubViewOp>(tileLoc, origOp.getOutputBuff(), currentOffset,
                                                               currentTileShapeVector);
        auto copyTile = rewriter.create<VPUIP::CopyOp>(tileLoc, inputSubView.getResult(), outputSubView.getResult());

        concatInputs.push_back(copyTile.getOutput());

        currentOffset[tileDim.ind()] += currentTileShapeVector[tileDim.ind()];
    }

    VPUX_THROW_UNLESS(currentOffset[tileDim.ind()] == tileDimSize,
                      "CopyOpTiling: a part of the original shape was not covered by Copy tiles {0} != {1}",
                      currentOffset[tileDim.ind()], tileDimSize);

    return concatInputs;
}

mlir::LogicalResult SingleCopyOpWithConcatOpUserTiling::matchAndRewrite(VPUIP::CopyOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Found single copy operation '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto concatInputs = createWithParallelTiles(origOp, rewriter);

    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, concatInputs, origOp.getOutputBuff());

    return mlir::success();
}

//
// safeRunOnFunc
//

/*
For two strides DMA in VPU, it will be implemented through plane.
If a two strides DMA do this date movement:
123 456 789
  ||
  \/                 | plane |
 1XX2XX3XX XXXXXXXXX 4XX5XX6XX XXXXXXXXX 7XX8XX9XX XXXXXXXXX
 |  |                |                   |
 stride              |                   |
                     |<-  plane stride ->|
The higher dim stride is implemented through plane stride.

So if the higher dim with stride size large than CMX_DMA_MAX_NUM_PLANES, we need tile the copy on this dim
*/

void CopyOpTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    const auto dmaPortNum = dmaOp.getCount();

    // This rewriter will not handle Copy Op with a distributed type
    // For Copy Op that require tiling:
    // 1. Handle cases where the tensor size exceeds max plane size;
    // 2. Handle cases with stride level is MAX_STRIDING_LEVEL and planes exceeding MAX_NUM_PLANES;
    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<CopyOpTiling>(&ctx, _log, arch, dmaPortNum);

        if (mlir::failed(
                    mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
        }
    }

    // For case "CopyOp + TilingCopyOp -> ConcatOp" with strided copy, tiling the copyOp into 2 could benefit
    // performance as copy in parallel.
    // Generic solution tracked by: #E122243
    {
        mlir::ConversionTarget targetWithConcat(ctx);
        targetWithConcat.addDynamicallyLegalOp<VPUIP::CopyOp>(isLegalCopyOpWithConcatOp);

        mlir::RewritePatternSet patternsWithConcat(&ctx);
        patternsWithConcat.add<SingleCopyOpWithConcatOpUserTiling>(&ctx, _log, dmaPortNum);

        targetWithConcat.addLegalOp<VPUIP::SubViewOp>();
        targetWithConcat.addLegalOp<VPUIP::ConcatViewOp>();

        if (mlir::failed(
                    mlir::applyPartialConversion(getOperation(), targetWithConcat, std::move(patternsWithConcat)))) {
            signalPassFailure();
        }
    }
}

}  // namespace

//
// createCopyOpTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCopyOpTilingPass(Logger log) {
    return std::make_unique<CopyOpTilingPass>(log);
}
