//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_TILEGATHER
#define GEN_PASS_DEF_TILEGATHER
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// TileGatherElement
//

class TileGatherElement final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    TileGatherElement(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("TileGatherElement");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileGatherElement::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ true, /*isIndicesTile*/ false, _log)) {
        return mlir::failure();
    }

    size_t axis = origOp.getAxisValue().value();
    const auto inputShape = getShape(origOp.getInput());
    const auto outputShape = getShape(origOp.getOutput());
    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto arch = VPU::getArch(origOp);

    Shape nTilesOnDim(outputShape.size(), 1);
    DimArr tileDimOrder;
    // Tiling the dim after axis. Gather Output shape size is different from input size, but the dim after axis will
    // keep.
    auto shapeSizeDiff = outputShape.size() - inputShape.size();
    for (size_t idx = axis + 1; idx < inputShape.size(); ++idx) {
        tileDimOrder.push_back(vpux::Dim(idx + shapeSizeDiff));
    }

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(origOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        const size_t GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED = VPU::getGatherDMAMaxElementSize(arch);

        for (auto tile : tiles.value()) {
            size_t element_size = vpux::getElemTypeSize(outputType).to<Byte>().count();
            auto inputTiling = origOp.backInferTileInfo(tile, _log);
            auto& inTiles = inputTiling.tiles;
            for (size_t idx = axis + 1; idx < inputShape.size(); ++idx) {
                element_size *= inTiles.begin()->shape.raw()[idx];
            }
            if (element_size > GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED) {
                return false;
            }
        }
        return true;
    };

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;
    while (tileDimIter < tileDimOrder.end() && !isSupportedTileSize(nTilesOnDim)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            dimToTile = *(++tileDimIter);
        } else {
            ++nTilesOnDim[Dim(dimToTile)];
        }
    }

    const auto tilesNew = fillDividedTiles(origOp, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, _log.nest());
}

//
// TileGatherIndices
//

class TileGatherIndices final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    TileGatherIndices(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("TileGatherIndices");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileGatherIndices::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ false, /*isIndicesTile*/ true, _log)) {
        return mlir::failure();
    }

    const auto outputShape = getShape(origOp.getOutput());
    const auto indicesType = origOp.getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();
    const auto indicesRank = origOp.getIndicesRank().value_or(indicesShape.size());
    const auto arch = VPU::getArch(origOp);

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(origOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        const size_t DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED = VPU::getGatherDMAMaxIndicesListLength(arch);

        for (auto tile : tiles.value()) {
            const auto inputTiling = origOp.backInferTileInfo(tile, _log);
            const auto indicesTiling = inputTiling.tiles[1];
            const auto newIndicesType = indicesType.extractDenseTile(indicesTiling.offsets, indicesTiling.shape);
            const size_t numberOfIndices = newIndicesType.getNumElements();
            if (numberOfIndices <= DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED) {
                return true;
            }
        }
        return false;
    };

    int64_t axisValue = 0;

    if (origOp.getAxisValueAttr() != nullptr) {
        axisValue = origOp.getAxisValueAttr().cast<mlir::IntegerAttr>().getValue().getSExtValue();
    }
    if (origOp.getAxis() != nullptr) {
        auto axisConst = origOp.getAxis().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(axisConst != nullptr, "Only constant input is supported for axis");
        VPUX_THROW_UNLESS(axisConst.getContentAttr().isSplat(), "Axis value must be a scalar");
        const auto axisContent = axisConst.getContent();
        axisValue = axisContent.getSplatValue<int64_t>();
    }

    int64_t batchDims = 0;
    if (origOp.getBatchDimsAttr() != nullptr) {
        batchDims = origOp.getBatchDimsAttr().cast<mlir::IntegerAttr>().getValue().getSExtValue();
    }

    const auto dimToTile = axisValue + indicesRank - batchDims - 1;
    while (!isSupportedTileSize(nTilesOnDim)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            return mlir::failure();
        }
        ++nTilesOnDim[Dim(dimToTile)];
    }

    const auto tilesNew = fillDividedTiles(origOp, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, _log.nest());
}

//
// TileGatherPass
//

class TileGatherPass final : public VPU::impl::TileGatherBase<TileGatherPass> {
public:
    explicit TileGatherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void TileGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto function = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TileGatherElement>(&ctx, _log);
    patterns.add<TileGatherIndices>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(function, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createTileGatherPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createTileGatherPass(Logger log) {
    return std::make_unique<TileGatherPass>(log);
}
