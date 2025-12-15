//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_APPLYTILINGMVN1SUM
#define GEN_PASS_DEF_APPLYTILINGMVN1SUM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

uint32_t getMVN1SumOutputHeight(VPU::MVN1SumOp op) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto inH = inType.getShape()[Dims4D::Act::H];

    auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto numCluster = config::getTileExecutor(module).getCount();
    VPUX_THROW_WHEN(numCluster <= 0, "Number of clusters should be a positive integer, while it is {0}", numCluster);
    const auto numActShave = config::getTotalNumOfEngines(op, VPU::ExecutorKind::SHAVE_ACT);
    const auto numActShavePerCluster = static_cast<int64_t>(numActShave / numCluster);

    uint32_t outputHeight = 1;
    auto highestDim = vpux::getHighestNonTrivialDim(inType.getShape(), inType.getDimsOrder()).value_or(Dim(0));
    if (op.getMultiClusterStrategy().has_value()) {
        const auto strategy = op.getMultiClusterStrategy().value();
        // Correct output height for Multi Cluster feature
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
            outputHeight = numCluster;
        }

        if (highestDim != Dims4D::Act::H) {
            return outputHeight;
        }

        // Correct output height for Multi Shave feature
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && inH >= numActShave) {
            outputHeight = numActShave;
        } else if (strategy == VPU::MultiClusterStrategy::Clustering && inH >= numActShavePerCluster) {
            outputHeight = numActShavePerCluster;
        }
    } else {
        outputHeight =
                (highestDim == Dims4D::Act::H && inH >= numActShavePerCluster) ? numActShavePerCluster : outputHeight;
    }

    return outputHeight;
}

mlir::FailureOr<OutputTiling> findNumOfTiles(VPU::MVN1SumOp op, bool enablePrefetchTiling, Logger log) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(op.getSum().getType());
    auto inShape = inType.getShape();

    auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numClusters = config::getTileExecutor(module).getCount();
    if (numClusters <= 0) {
        return mlir::failure();
    }

    auto newInShape = Shape(inShape);

    // Restrict max-search to {W, H}, since with 'internal_reshape' feature,
    // 1x32x1048576x1 turns into 1x512x256x256 with C being maxDim
    const auto maxDim = std::distance(inShape.begin(), std::max_element(inShape.begin() + 2, inShape.end()));
    const auto tileDim = Dim(maxDim);

    // MVN1SumOp only supports Clustering, SplitOverHeight, and SplitOverKernel strategies
    auto tileClusters = 1;
    if (op.getMultiClusterStrategy().has_value()) {
        auto strategy = op.getMultiClusterStrategy().value();
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            newInShape[Dims4D::Act::C] = divUp(newInShape[Dims4D::Act::C], numClusters);
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
            if (tileDim != Dims4D::Act::H) {
                newInShape[Dims4D::Act::H] = divUp(newInShape[Dims4D::Act::H], numClusters);
            } else {
                tileClusters = numClusters;
            }
        }
    }

    // Step1. get an feasible isolated tiling strategy
    int64_t tilesNum = 1;
    auto maxNumTiles = newInShape;
    maxNumTiles[Dims4D::Act::H] = inShape[Dims4D::Act::H] / tileClusters;
    while (!op.fitIntoCMX(SmallVector<vpux::NDTypeInterface>{inType.changeShape(newInShape), outType})) {
        tilesNum++;
        if (tilesNum > maxNumTiles[tileDim]) {
            return errorAt(op.getLoc(), "Can't tile MVN1SumOp over one dimension.");
        }

        newInShape[tileDim] = divUp(inShape[tileDim], tilesNum * tileClusters);
    }
    log.trace("MVN1Sum isolated tiling strategy: {0} @ {1} for {2}", tilesNum, tileDim, inShape);

    // Step2. For pipelining, continue to increase on the dimension of isolated tiling
    if (enablePrefetchTiling) {
        auto availableCMX = vpux::VPU::getTotalCMXSize(op.getOperation());
        auto pipeliningTiles = tilesNum;
        auto maxNumPipeliningTiles = std::min(maxNumTiles[tileDim], MAX_PREFETCH_TILING_TIME * tilesNum);
        while (pipeliningTiles <= maxNumPipeliningTiles) {
            if (pipeliningTiles * tileClusters - 1 <= 0) {
                pipeliningTiles++;
                continue;
            }

            newInShape[tileDim] = divUp(inShape[tileDim], pipeliningTiles * tileClusters);
            auto inType0 = inType.changeShape(newInShape);
            if (pipeliningTiles * tileClusters > 1) {
                newInShape[tileDim] = divUp(inShape[tileDim] - newInShape[tileDim], pipeliningTiles * tileClusters - 1);
            }
            auto inType1 = inType.changeShape(newInShape);
            auto requiredCMX = VPU::getRequiredCMXSize(inType0) + VPU::getRequiredCMXSize(inType1) +
                               VPU::getRequiredCMXSize(outType) * 2;

            if (requiredCMX <= availableCMX) {
                tilesNum = pipeliningTiles;
                log.trace("MVN1Sum pipelining tiling strategy: {0} @ {1} for {2}", tilesNum, tileDim, inShape);
                break;
            }

            pipeliningTiles++;
        }
    }

    Shape divisors(newInShape.size(), 1);
    divisors[tileDim] = tilesNum;
    auto resultTiles = fillDividedTiles(op, divisors, inShape);
    if (mlir::failed(resultTiles)) {
        return mlir::failure();
    }

    return resultTiles;
}

mlir::Value reifyTileMVN1Sum(VPU::MVN1SumOp MVN1SumOp, const TileInfo& inputTile, mlir::OpBuilder& builder,
                             Logger log) {
    log.trace("{0}", inputTile);

    auto numClusters = getMVN1SumOutputHeight(MVN1SumOp);

    const auto valInputName = printToString("input");

    const auto tiledSliceInput =
            vpux::VPU::makeTile(builder, MVN1SumOp.getLoc(), MVN1SumOp.getInput(), inputTile, valInputName);

    auto tileMVN1SumOp = builder.create<VPU::MVN1SumOp>(MVN1SumOp.getLoc(), tiledSliceInput,
                                                        MVN1SumOp.getAcrossChannels(), MVN1SumOp.getNormalizeVariance(),
                                                        numClusters, MVN1SumOp.getMultiClusterStrategyAttr());

    return tileMVN1SumOp.getResult();
}

//
// ApplyTilingMVN1Sum
//

class ApplyTilingMVN1Sum final : public VPU::impl::ApplyTilingMVN1SumBase<ApplyTilingMVN1Sum> {
public:
    explicit ApplyTilingMVN1Sum(bool enablePrefetchTiling, Logger log): _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class MVN1SumTiling;
    class MVN1SumCorrectHeight;

private:
    void safeRunOnFunc() final;

    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult ApplyTilingMVN1Sum::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_enablePrefetchTiling' field to the value {1} of the "
                   "pass option 'tilingMode' generated by MLIR",
                   _enablePrefetchTiling, tilingMode.getValue());
        _enablePrefetchTiling = tilingMode.getValue() != "ISOLATED";
    }

    return mlir::success();
}

//
// MVN1SumTiling
//

class ApplyTilingMVN1Sum::MVN1SumTiling final : public mlir::OpRewritePattern<VPU::MVN1SumOp> {
public:
    MVN1SumTiling(mlir::MLIRContext* ctx, bool enablePrefetchTiling, Logger log)
            : mlir::OpRewritePattern<VPU::MVN1SumOp>(ctx), _enablePrefetchTiling(enablePrefetchTiling), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVN1SumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enablePrefetchTiling = true;
    Logger _log;
};

mlir::LogicalResult ApplyTilingMVN1Sum::MVN1SumTiling::matchAndRewrite(VPU::MVN1SumOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (mlir::isa_and_nonnull<VPU::SliceOp>(origOp.getInput().getDefiningOp())) {
        return matchFailed(rewriter, origOp, "Op already tiled.");
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getSum().getType());
    if (origOp.fitIntoCMX(SmallVector<vpux::NDTypeInterface>{inputType, outputType})) {
        return matchFailed(rewriter, origOp, "Op fits into CMX");
    }

    const auto tiles = findNumOfTiles(origOp, _enablePrefetchTiling, _log);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }
    // apply the generated fake tiling strategy and convert MVN to Slice.
    SmallVector<mlir::Value> resultTileVals;

    for (const auto& inputTile : tiles.value()) {
        const auto tiledRes = reifyTileMVN1Sum(origOp, inputTile, rewriter, _log);
        resultTileVals.emplace_back(tiledRes);
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, mlir::ValueRange(resultTileVals), 3);

    return mlir::success();
}

//
// MVN1SumCorrectHeight
//

class ApplyTilingMVN1Sum::MVN1SumCorrectHeight final : public mlir::OpRewritePattern<VPU::MVN1SumOp> {
public:
    MVN1SumCorrectHeight(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::MVN1SumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVN1SumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ApplyTilingMVN1Sum::MVN1SumCorrectHeight::matchAndRewrite(VPU::MVN1SumOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (mlir::isa_and_nonnull<VPU::SliceOp>(origOp.getInput().getDefiningOp())) {
        return matchFailed(rewriter, origOp, "Op height was already corrected.");
    }

    auto correctHeightValue = getMVN1SumOutputHeight(origOp);

    const auto output = mlir::cast<vpux::NDTypeInterface>(origOp.getSum().getType());

    const auto newOutputShape = to_small_vector(output.getShape());
    if (newOutputShape[Dims4D::Act::H.ind()] == correctHeightValue) {
        return matchFailed(rewriter, origOp, "Op height is already correct.");
    }

    rewriter.replaceOpWithNewOp<VPU::MVN1SumOp>(origOp, origOp.getInput(), origOp.getAcrossChannels(),
                                                origOp.getNormalizeVariance(), correctHeightValue,
                                                origOp.getMultiClusterStrategyAttr());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ApplyTilingMVN1Sum::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<MVN1SumCorrectHeight>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<MVN1SumTiling>(&ctx, _enablePrefetchTiling, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
}

}  // namespace

//
// createApplyTilingMVN1SumPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createApplyTilingMVN1SumPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<ApplyTilingMVN1Sum>(enablePrefetchTiling, log);
}
