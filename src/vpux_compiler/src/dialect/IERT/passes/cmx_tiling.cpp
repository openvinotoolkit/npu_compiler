//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>
#include <numeric>

using namespace vpux;

namespace {

using OutputTiling = SmallVector<Tile>;

//
// makeTile
//

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal, const Tile& tile,
                     StringRef valName) {
    const auto origType = origVal.getType().cast<mlir::MemRefType>();

    if (tile.shape == getShape(origType)) {
        return origVal;
    }

    const auto tileName = llvm::formatv("{0} tile {1}", valName, tile.offsets).str();
    const auto loc = appendLoc(baseLoc, tileName);

    const auto tileShape = tile.shape.raw();
    const auto tileOffsets = tile.offsets.raw();

    const auto attrOffsets = getIntArrayAttr(builder.getContext(), tileOffsets);
    const auto attrShape = getIntArrayAttr(builder.getContext(), tileShape);
    auto viewOp = builder.create<IERT::SubViewOp>(loc, origVal, attrOffsets, attrShape);

    const auto tileType = getDenseTileType(origType, tile.offsets, tile.shape);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, tileType);

    auto copyOp = builder.create<IERT::CopyOp>(loc, viewOp.result(), allocOp.memref());
    return copyOp.output();
}

//
// ConvolutionTiling
//

class ConvolutionTiling final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
    using TilerFunc = std::function<OutputTiling(IERT::ConvolutionOp)>;

public:
    ConvolutionTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("ConvolutionTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult ConvolutionTiling::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution at '{1}'", getDebugName(), origOp->getLoc());

    const auto tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const auto& outputTile : tilings) {
        const auto tileConf = backInferConvTile(origOp, outputTile);

        const auto& inputTile = tileConf.inputTile;
        const auto& filterTile = tileConf.filterTile;
        const auto& biasTile = tileConf.biasTile;

        SmallVector<int64_t> padsBegin = {tileConf.pads.padTop, tileConf.pads.padLeft};
        SmallVector<int64_t> padsEnd = {tileConf.pads.padBottom, tileConf.pads.padRight};

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");
        const auto filterInput = makeTile(rewriter, origOp->getLoc(), origOp.filter(), filterTile, "filter");
        const auto biasInput = origOp.bias() != nullptr
                                       ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile, "bias")
                                       : nullptr;

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = getDenseTileType(origOp.output().getType().cast<mlir::MemRefType>(),
                                                  outputTile.offsets, outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::ConvolutionOp>(loc, actInput, filterInput, biasInput, allocOutOp.memref(),
                                                            origOp.strides(), getIntArrayAttr(getContext(), padsBegin),
                                                            getIntArrayAttr(getContext(), padsEnd), origOp.dilations(),
                                                            origOp.post_opAttr());

        const auto attrOffsets = getIntArrayAttr(rewriter.getContext(), outputTile.offsets.raw());
        const auto attrShape = getIntArrayAttr(rewriter.getContext(), outputTile.shape.raw());
        auto subViewOut = rewriter.create<IERT::SubViewOp>(loc, origOp.output_buff(), attrOffsets, attrShape);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// EltwiseTiling
//

template <class ConcreteOp>
class EltwiseTiling final : public mlir::OpRewritePattern<ConcreteOp> {
    using TilerFunc = std::function<OutputTiling(ConcreteOp)>;

public:
    EltwiseTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        this->setDebugName("EltwiseTiling");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseTiling<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' Eltwise operation at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const OutputTiling tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const Tile& tile : tilings) {
        const mlir::Value actInput1 = makeTile(rewriter, origOp->getLoc(), origOp.input1(), tile, "input1");
        const mlir::Value actInput2 = makeTile(rewriter, origOp->getLoc(), origOp.input2(), tile, "input2");

        const std::string tileName = llvm::formatv("output tile {0}", tile.offsets).str();
        const mlir::Location loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut =
                getDenseTileType(origOp.output().getType().template cast<mlir::MemRefType>(), tile.offsets, tile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp =
                rewriter.create<IERT::AddOp>(loc, actInput1, actInput2, allocOutOp.memref(), origOp.post_opAttr());

        const auto attrOffsets = getIntArrayAttr(rewriter.getContext(), tile.offsets.raw());
        const auto attrShape = getIntArrayAttr(rewriter.getContext(), tile.shape.raw());
        auto subViewOut = rewriter.create<IERT::SubViewOp>(loc, origOp.output_buff(), attrOffsets, attrShape);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// MaxPoolTiling
//

class MaxPoolTiling final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
    using TilerFunc = std::function<OutputTiling(IERT::MaxPoolOp)>;

public:
    MaxPoolTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("MaxPoolTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult MaxPoolTiling::matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool at '{1}'", getDebugName(), origOp->getLoc());

    const auto tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const auto& outputTile : tilings) {
        const auto tileConf = backInferPoolTile(origOp, outputTile);

        const auto& inputTile = tileConf.inputTile;

        SmallVector<int64_t> padsBegin = {tileConf.pads.padTop, tileConf.pads.padLeft};
        SmallVector<int64_t> padsEnd = {tileConf.pads.padBottom, tileConf.pads.padRight};

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = getDenseTileType(origOp.output().getType().cast<mlir::MemRefType>(),
                                                  outputTile.offsets, outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::MaxPoolOp>(loc, actInput, allocOutOp.memref(), origOp.kernel_size(),
                                                        origOp.strides(), getIntArrayAttr(getContext(), padsBegin),
                                                        getIntArrayAttr(getContext(), padsEnd), origOp.post_opAttr());

        const auto attrOffsets = getIntArrayAttr(rewriter.getContext(), outputTile.offsets.raw());
        const auto attrShape = getIntArrayAttr(rewriter.getContext(), outputTile.shape.raw());
        auto subViewOut = rewriter.create<IERT::SubViewOp>(loc, origOp.output_buff(), attrOffsets, attrShape);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// GroupConvolutionTiling
//

class GroupConvolutionTiling final : public mlir::OpRewritePattern<IERT::GroupConvolutionOp> {
    using TilerFunc = std::function<OutputTiling(IERT::GroupConvolutionOp)>;

public:
    GroupConvolutionTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::GroupConvolutionOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("GroupConvolutionTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult GroupConvolutionTiling::matchAndRewrite(IERT::GroupConvolutionOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution at '{1}'", getDebugName(), origOp->getLoc());

    const auto tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const auto& outputTile : tilings) {
        const auto tileConf = backInferGroupConvTile(origOp, outputTile);

        const auto& inputTile = tileConf.inputTile;
        const auto& filterTile = tileConf.filterTile;
        const auto& biasTile = tileConf.biasTile;

        SmallVector<int64_t> padsBegin = {tileConf.pads.padTop, tileConf.pads.padLeft};
        SmallVector<int64_t> padsEnd = {tileConf.pads.padBottom, tileConf.pads.padRight};

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");
        const auto filterInput = makeTile(rewriter, origOp->getLoc(), origOp.filter(), filterTile, "filter");
        const auto biasInput = origOp.bias() != nullptr
                                       ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile, "bias")
                                       : nullptr;

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = getDenseTileType(origOp.output().getType().cast<mlir::MemRefType>(),
                                                  outputTile.offsets, outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        const auto groups = filterTile.shape[IE::Dims4D::Filter::OC];
        const auto groupsAttr = getIntAttr(getContext(), groups);

        auto tiledOp = rewriter.create<IERT::GroupConvolutionOp>(
                loc, actInput, filterInput, biasInput, allocOutOp.memref(), origOp.strides(),
                getIntArrayAttr(getContext(), padsBegin), getIntArrayAttr(getContext(), padsEnd), origOp.dilations(),
                groupsAttr, origOp.post_opAttr());

        const auto attrOffsets = getIntArrayAttr(rewriter.getContext(), outputTile.offsets.raw());
        const auto attrShape = getIntArrayAttr(rewriter.getContext(), outputTile.shape.raw());
        auto subViewOut = rewriter.create<IERT::SubViewOp>(loc, origOp.output_buff(), attrOffsets, attrShape);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// SimpleTiler
//

class SimpleTiler {
public:
    explicit SimpleTiler(Logger log): _log(log) {
    }

    void buildTilingPatterns(mlir::RewritePatternSet& patterns);

private:
    OutputTiling convolutionTiler(IERT::ConvolutionOp op) const;
    OutputTiling maxPoolTiler(IERT::MaxPoolOp op) const;
    OutputTiling groupConvolutionTiler(IERT::GroupConvolutionOp op) const;

    OutputTiling genericTiler(mlir::Operation* op, mlir::MemRefType outputType,
                              FuncRef<bool(ShapeRef)> isSupportedTileSize) const;
    OutputTiling groupConvTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                FuncRef<bool(ShapeRef)> isSupportedTileSize) const;

    template <class ConcreteOp>
    OutputTiling eltwiseTiler(ConcreteOp op) const;

private:
    Logger _log;
};

void SimpleTiler::buildTilingPatterns(mlir::RewritePatternSet& patterns) {
    const auto convTilerFunc = std::bind(&SimpleTiler::convolutionTiler, this, std::placeholders::_1);
    patterns.add<ConvolutionTiling>(patterns.getContext(), convTilerFunc, _log);

    const auto eltwiseAddTilerFunc = std::bind(&SimpleTiler::eltwiseTiler<IERT::AddOp>, this, std::placeholders::_1);
    patterns.add<EltwiseTiling<IERT::AddOp>>(patterns.getContext(), eltwiseAddTilerFunc, _log);

    const auto eltwiseMultTilerFunc =
            std::bind(&SimpleTiler::eltwiseTiler<IERT::MultiplyOp>, this, std::placeholders::_1);
    patterns.add<EltwiseTiling<IERT::MultiplyOp>>(patterns.getContext(), eltwiseMultTilerFunc, _log);

    const auto eltwiseSubTilerFunc =
            std::bind(&SimpleTiler::eltwiseTiler<IERT::SubtractOp>, this, std::placeholders::_1);
    patterns.add<EltwiseTiling<IERT::SubtractOp>>(patterns.getContext(), eltwiseSubTilerFunc, _log);

    const auto eltwiseAndTilerFunc = std::bind(&SimpleTiler::eltwiseTiler<IERT::AndOp>, this, std::placeholders::_1);
    patterns.add<EltwiseTiling<IERT::AndOp>>(patterns.getContext(), eltwiseAndTilerFunc, _log);

    const auto maxPoolTilerFunc = std::bind(&SimpleTiler::maxPoolTiler, this, std::placeholders::_1);
    patterns.add<MaxPoolTiling>(patterns.getContext(), maxPoolTilerFunc, _log);

    const auto groupConvTilerFunc = std::bind(&SimpleTiler::groupConvolutionTiler, this, std::placeholders::_1);
    patterns.add<GroupConvolutionTiling>(patterns.getContext(), groupConvTilerFunc, _log);
    // TODO Replace std::bind calls with corresponding anonymous functions.
}

OutputTiling SimpleTiler::genericTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                       FuncRef<bool(ShapeRef)> isSupportedTileSize) const {
    const auto outputShape = getShape(outputType);

    const auto minChannelSize = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());
    const auto maxChannelTiles = outputShape[IE::Dims4D::Act::C] / minChannelSize;

    Shape nTilesOnDim(outputShape.size(), 1);

    // Try to tile the largest dim (C or H) first, then proceed with other dims
    SmallVector<Dim> tileDimOrder = {IE::Dims4D::Act::C, IE::Dims4D::Act::H, IE::Dims4D::Act::W};
    if (outputShape[IE::Dims4D::Act::C] < outputShape[IE::Dims4D::Act::H])
        tileDimOrder = {IE::Dims4D::Act::H, IE::Dims4D::Act::C, IE::Dims4D::Act::W};

    auto tileDimIter = tileDimOrder.begin();
    Optional<Dim> dimToTile = *tileDimIter;

    const auto isSupportedChannelDivision = [&]() {
        if ((outputShape[IE::Dims4D::Act::C] % nTilesOnDim[IE::Dims4D::Act::C]) != 0) {
            return false;
        }

        const auto tileChannels = outputShape[IE::Dims4D::Act::C] / nTilesOnDim[IE::Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };

    const auto isDimLeftToTile = [&]() {
        if (dimToTile == IE::Dims4D::Act::C) {
            if (nTilesOnDim[IE::Dims4D::Act::C] < maxChannelTiles)
                return true;
        } else {  // Spatial dims
            const auto origSize = outputShape[dimToTile.getValue()];
            const auto prevDivisor = nTilesOnDim[dimToTile.getValue()];

            if (origSize / prevDivisor > 1)
                return true;
        }

        return false;
    };

    while (!isSupportedTileSize(nTilesOnDim)) {
        if (!isDimLeftToTile()) {
            dimToTile = *(++tileDimIter);
        }

        if (dimToTile == IE::Dims4D::Act::C) {
            do {
                ++nTilesOnDim[IE::Dims4D::Act::C];
            } while (!isSupportedChannelDivision());
        } else if (dimToTile == IE::Dims4D::Act::H || dimToTile == IE::Dims4D::Act::W) {
            nTilesOnDim[dimToTile.getValue()]++;
        } else {
            // Trying to tile in unsupported dimension, tiling in supported dimensions not sufficient
            VPUX_THROW("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        }
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

OutputTiling SimpleTiler::groupConvTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                         FuncRef<bool(ShapeRef)> isSupportedTileSize) const {
    const auto outputShape = getShape(outputType);

    Shape nTilesOnDim(outputShape.size(), 1);

    // FIXME tiling over channels has to leave 16 channels in each tile.
    // Otherwise, depthwise convolutions produce worse accuracy.
    const auto depthwiseOutChanCount = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());
    VPUX_THROW_UNLESS(outputShape[IE::Dims4D::Act::C] % depthwiseOutChanCount == 0,
                      "Depthwise convolution output channels must be a multiple of {0}, got {1}", depthwiseOutChanCount,
                      outputShape[IE::Dims4D::Act::C]);
    nTilesOnDim[IE::Dims4D::Act::C] = outputShape[IE::Dims4D::Act::C] / depthwiseOutChanCount;

    while (!isSupportedTileSize(nTilesOnDim)) {
        Optional<Dim> dimToTile;

        for (auto ind : irange(IE::Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = IE::Dims4D::Act::getSpatialDim(ind);

            const auto origSize = outputShape[spatialDim];
            const auto prevDivisor = nTilesOnDim[spatialDim];

            if (origSize / prevDivisor > 1) {
                dimToTile = spatialDim;
                break;
            }
        }

        VPUX_THROW_UNLESS(dimToTile.hasValue(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        nTilesOnDim[dimToTile.getValue()]++;
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

OutputTiling SimpleTiler::convolutionTiler(IERT::ConvolutionOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto filterType = op.filter().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const auto tileConf = backInferConvTile(op, outputTile);

            const auto inputTileType =
                    getDenseTileType(inputType, tileConf.inputTile.offsets, tileConf.inputTile.shape);
            const auto filterTileType =
                    getDenseTileType(filterType, tileConf.filterTile.offsets, tileConf.filterTile.shape);
            const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyConvCMX(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(),
                                                       inputTileType, filterTileType, outputTileType, _log));
        });
    };

    return genericTiler(op, outputType, isSupportedTileSize);
}

template <class ConcreteOp>
OutputTiling SimpleTiler::eltwiseTiler(ConcreteOp op) const {
    const auto input1Type = op.input1().getType().template cast<mlir::MemRefType>();
    const auto input2Type = op.input2().getType().template cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().template cast<mlir::MemRefType>();

    const ShapeRef outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& tile) {
            const auto input1TileType = getDenseTileType(input1Type, tile.offsets, tile.shape);
            const auto input2TileType = getDenseTileType(input2Type, tile.offsets, tile.shape);
            const auto outputTileType = getDenseTileType(outputType, tile.offsets, tile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyEltwiseCMX(op->getLoc(), op->template getParentOfType<mlir::ModuleOp>(),
                                                          input1TileType, input2TileType, outputTileType, _log));
        });
    };

    return genericTiler(op, outputType, isSupportedTileSize);
}

OutputTiling SimpleTiler::maxPoolTiler(IERT::MaxPoolOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const auto tileConf = backInferPoolTile(op, outputTile);

            const auto inputTileType =
                    getDenseTileType(inputType, tileConf.inputTile.offsets, tileConf.inputTile.shape);
            const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

            return mlir::succeeded(VPUIP::NCEInvariant::verifyPoolCMX(
                    op->getLoc(), op->getParentOfType<mlir::ModuleOp>(), inputTileType, outputTileType,
                    op.kernel_size(), op.strides(), _log));
        });
    };

    return genericTiler(op, outputType, isSupportedTileSize);
}

OutputTiling SimpleTiler::groupConvolutionTiler(IERT::GroupConvolutionOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto filterType = op.filter().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const auto tileConf = backInferGroupConvTile(op, outputTile);

            const auto inputTileType =
                    getDenseTileType(inputType, tileConf.inputTile.offsets, tileConf.inputTile.shape);
            const auto filterTileType =
                    getDenseTileType(filterType, tileConf.filterTile.offsets, tileConf.filterTile.shape);
            const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyConvCMX(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(),
                                                       inputTileType, filterTileType, outputTileType, _log));
        });
    };

    return groupConvTiler(op, outputType, isSupportedTileSize);
}

//
// CMXTilingPass
//

class CMXTilingPass final : public IERT::CMXTilingBase<CMXTilingPass> {
public:
    explicit CMXTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <class ConcreteOp>
bool isSupportedByNCE(ConcreteOp op, Logger log) {
    return VPUIP::NCEInvariant::verifyKernel(op, log).succeeded() &&
           VPUIP::NCEInvariant::verifyChannels(op, log).succeeded();
}

template <class ConcreteOp>
bool isLegal(ConcreteOp origOp, Logger log) {
    if (!isSupportedByNCE(origOp, log.nest())) {
        // It will be computed on SHAVEs
        return true;
    }

    return VPUIP::NCEInvariant::verifyCMX(origOp, log.nest()).succeeded();
}

void CMXTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addDynamicallyLegalOp<IERT::ConvolutionOp>([&](IERT::ConvolutionOp op) {
        return isLegal(op, _log);
    });
    target.addDynamicallyLegalOp<IERT::AddOp>([&](IERT::AddOp op) {
        return isLegal(op, _log);
    });
    target.addDynamicallyLegalOp<IERT::MultiplyOp>([&](IERT::MultiplyOp op) {
        return isLegal(op, _log);
    });
    target.addDynamicallyLegalOp<IERT::SubtractOp>([&](IERT::SubtractOp op) {
        return isLegal(op, _log);
    });
    target.addDynamicallyLegalOp<IERT::AndOp>([&](IERT::AndOp op) {
        return isLegal(op, _log);
    });
    target.addDynamicallyLegalOp<IERT::MaxPoolOp>([&](IERT::MaxPoolOp op) {
        return isLegal(op, _log);
    });
    target.addDynamicallyLegalOp<IERT::GroupConvolutionOp>([&](IERT::GroupConvolutionOp op) {
        return isLegal(op, _log);
    });

    mlir::RewritePatternSet patterns(&ctx);

    SimpleTiler simpleTiler(_log);
    simpleTiler.buildTilingPatterns(patterns);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCMXTilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCMXTilingPass(Logger log) {
    return std::make_unique<CMXTilingPass>(log);
}