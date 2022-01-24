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

#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

OutputTiling generatePrefetchTiles(mlir::Operation* op, Logger log) {
    log.trace("Generating prefetch tiles for op {0} at {1}", op->getName(), op->getLoc());

    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    const auto outputShape = getShape(op->getResult(0).getType().cast<mlir::ShapedType>());
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());
    auto getDimsToTile = [](const Shape& nTilesOnDim) -> SmallVector<Dim> {
        SmallVector<Dim> res;
        for (unsigned i = 0; i < nTilesOnDim.size(); i++) {
            if (nTilesOnDim[Dim(i)] > 1)
                res.emplace_back(Dim(i));
        }
        return res;
    };

    // step 1: compute a general tiling strategy to fit into the CMX
    Shape nTilesOnDim = IE::computeGeneralTileStrategy(op, log);
    auto dimsToTile = getDimsToTile(nTilesOnDim);
    VPUX_THROW_WHEN(dimsToTile.size() == 0, "Must tile at least on one dimension");
    if (dimsToTile.size() > 1) {
        // return general tiling when getting nested tiles.
        return fillDividedTiles(nTilesOnDim, outputShape);
    }
    //    if (nTilesOnDim[Dims4D::Act::C] <= 1) {
    //        // Only enable weights prefetch for now.
    //        return fillDividedTiles(nTilesOnDim, outputShape);
    //    }

    // step 2: increase the general tile strategy to satisfy prefetching
    const auto targetDim = dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    while (prefetchableTilesOnDim[targetDim] < 3 * nTilesOnDim[targetDim] &&
           !tilingInfo.isSupportedPrefetchTiling(prefetchableTilesOnDim, log)) {
        // The "3" here is an experimental number from MCM activation prefetch pass.
        // The purpose is to avoid excessive tiling.
        prefetchableTilesOnDim[targetDim]++;
    }

    if (tilingInfo.isSupportedPrefetchTiling(prefetchableTilesOnDim, log) && prefetchableTilesOnDim != nTilesOnDim)
        std::cout << llvm::formatv("{1}~~~{2}~~~prefetch~~~{0}~~~tiles:", prefetchableTilesOnDim, op->getLoc(),
                                   op->getName())
                             .str()
                  << std::endl;
    else
        std::cout
                << llvm::formatv("{1}~~~{2}~~~original~~~{0}~~~tiles:", nTilesOnDim, op->getLoc(), op->getName()).str()
                << std::endl;

    return tilingInfo.isSupportedPrefetchTiling(prefetchableTilesOnDim, log)
                   ? fillDividedTiles(prefetchableTilesOnDim, outputShape)
                   : fillDividedTiles(nTilesOnDim, outputShape);
}

SmallVector<Shape> generatePrefetchPatternTiles(mlir::Operation* op, mlir::Operation* parentOp, Logger log) {
    // TODO: merge with IE::computeGeneralTileStrategy
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<IE::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
    const auto parentOutputType = parentOp->getResult(0).getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);
    const auto parentOutputShape = getShape(parentOutputType);

    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getOutputChannelAlignment();
    }

    Shape nTilesOnDim(outputShape.size(), 1);
    Shape nTilesOnDimParent(parentOutputShape.size(), 1);

    // Try to tile the largest dim (C or H)
    Dim dimToTile = Dims4D::Act::C;
    if (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) {
        dimToTile = Dims4D::Act::H;
    }

    const auto isSupportedTilesPattern = [&]() -> bool {
        return tilingInfo.isSupportedPrefetchPattern(nTilesOnDim, parentOp, nTilesOnDimParent, log);
    };
    const auto isSupportedChannelDivision = [&]() -> bool {
        if ((outputShape[Dims4D::Act::C] % nTilesOnDim[Dims4D::Act::C]) != 0) {
            return false;
        }
        const auto tileChannels = outputShape[Dims4D::Act::C] / nTilesOnDim[Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };
    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&]() -> bool {
        return nTilesOnDim[dimToTile] < maxNumTiles[dimToTile.ind()];
    };
    while (!isSupportedTilesPattern()) {
        if (!isDimLeftToTile()) {
            VPUX_THROW("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        }
        // increase current op tiles
        if (dimToTile == Dims4D::Act::C) {
            do {
                ++nTilesOnDim[Dims4D::Act::C];
            } while (!isSupportedChannelDivision());
        } else {
            nTilesOnDim[dimToTile]++;
        }
        // increase parent op tiles
        if (!isSupportedTilesPattern()) {
            if (dimToTile == Dims4D::Act::C) {
                do {
                    ++nTilesOnDimParent[Dims4D::Act::C];
                } while (!isSupportedChannelDivision());
            } else {
                nTilesOnDimParent[dimToTile]++;
            }
        }
    }
    return {nTilesOnDim, nTilesOnDimParent};
}

bool needTilingToMultiOpsPrefetch(mlir::Operation* op, Logger log) {
    auto parentOp = op->getOperand(0).getDefiningOp();
    std::cout << llvm::formatv("needTilingToMultiOpsPrefetch: check op {0} {1}; parent {2}, {3}", op->getLoc(),
                               op->getName(), parentOp->getLoc(), parentOp->getName())
                         .str()
              << std::endl;
    auto opTilingInter = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    auto parentTilingInter = mlir::dyn_cast<IE::TilingInfoOpInterface>(parentOp);
    if (!opTilingInter || !parentTilingInter) {
        return false;
    }
    // For parallel sub-graphs, the order is undecided yet
    // Abandon prefetching these cases
    if (!parentOp->getResult(0).hasOneUse()) {
        return false;
    }

    const auto resShape = getShape(op->getResult(0));
    const Shape neutralTile(resShape.size(), 1);
    return !opTilingInter.isSupportedPrefetchPattern(neutralTile, parentOp, neutralTile, log);
}
//
// PrefetchTiling
//

class PrefetchTiling final : public mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface> {
public:
    PrefetchTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("PrefetchTiling");
    }
    mlir::LogicalResult matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PrefetchTiling::matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    const auto resShape = getShape(op->getResult(0));
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    if (tilingInfo.isSupportedTiling({TileInfo(resShape)}, _log.nest())) {
        // If the current op fits CMX but still run into here
        // The op needs tiling to be prefetched by its parent
        auto parentOp = op->getOperand(0).getDefiningOp();
        const auto parentResShape = getShape(parentOp->getResult(0));
        auto tiles = generatePrefetchPatternTiles(op, parentOp, _log.nest());

        auto getDimsToTile = [](ShapeRef nTilesOnDim) -> SmallVector<Dim> {
            SmallVector<Dim> res;
            for (unsigned i = 0; i < nTilesOnDim.size(); i++) {
                if (nTilesOnDim[Dim(i)] > 1)
                    res.emplace_back(Dim(i));
            }
            return res;
        };
        auto curTiles = fillDividedTiles(tiles[0], resShape);
        auto parentTiles = fillDividedTiles(tiles[1], parentResShape);
        std::cout << llvm::formatv("cur tile: {0}", tiles[0]).str() << std::endl;
        std::cout << llvm::formatv("parent tile: {0}", tiles[1]).str() << std::endl;
        if (getDimsToTile(tiles[1]).size() == 0) {
            return applyTileStrategy(origOp, curTiles, rewriter, _log);
        } else {
            if (mlir::succeeded(applyTileStrategy(mlir::dyn_cast<IE::TilingBuilderOpInterface>(parentOp), parentTiles,
                                                  rewriter, _log)) &&
                mlir::succeeded(applyTileStrategy(origOp, curTiles, rewriter, _log))) {
                return mlir::success();
            }
        }
        return mlir::success();
    } else {
        const auto tiles = generatePrefetchTiles(origOp.getOperation(), _log.nest());
        _log.nest(1).trace("Create {0} tiles:", tiles.size());
        return applyTileStrategy(origOp, tiles, rewriter, _log);
    }
    return mlir::success();
}

//
// PrefetchTilingPass
//
class PrefetchTilingPass final : public IE::PrefetchTilingBase<PrefetchTilingPass> {
public:
    explicit PrefetchTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void PrefetchTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp, IE::ConcatOp>();
    target.markUnknownOpDynamicallyLegal([this](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
            const auto resShape = getShape(op->getResult(0));
            if (!iface.isSupportedTiling({TileInfo(resShape)}, _log.nest())) {
                return false;
            }
            if (needTilingToMultiOpsPrefetch(op, _log)) {
                std::cout << "needTilingToMultiOpsPrefetch: YES!!" << std::endl;
                return false;
            }
        }

        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PrefetchTiling>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPrefetchTilingPass(Logger log) {
    return std::make_unique<PrefetchTilingPass>(log);
}
