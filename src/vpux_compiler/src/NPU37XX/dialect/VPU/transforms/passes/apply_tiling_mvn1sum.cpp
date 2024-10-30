//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

uint32_t getMVN1SumOutputHeight(VPU::MVN1SumOp op) {
    const auto inType = op.getInput().getType().cast<NDTypeInterface>();
    const auto inH = inType.getShape()[Dims4D::Act::H];

    auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto numCluster = IE::getTileExecutor(module).getCount();
    const auto numActShave = IE::getTotalNumOfEngines(op, VPU::ExecutorKind::SHAVE_ACT);
    const auto numActShavePerCluster = static_cast<int64_t>(numActShave / numCluster);

    uint32_t outputHeight = 1;
    if (op.getMultiClusterStrategy().has_value()) {
        const auto strategy = op.getMultiClusterStrategy().value();
        // Correct output height for Multi Cluster feature
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
            outputHeight = numCluster;
        }

        // Correct output height for Multi Shave feature
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && inH >= numActShave &&
            VPU::getHighestDimFromType(inType) == Dims4D::Act::H) {
            outputHeight = numActShave;
        } else if (strategy == VPU::MultiClusterStrategy::Clustering && inH >= numActShavePerCluster &&
                   VPU::getHighestDimFromType(inType) == Dims4D::Act::H) {
            outputHeight = numActShavePerCluster;
        }
    }

    return outputHeight;
}

mlir::FailureOr<OutputTiling> findNumOfTiles(VPU::MVN1SumOp op) {
    const auto inType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.getSum().getType().cast<vpux::NDTypeInterface>();
    auto inShape = inType.getShape();

    auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numClusters = IE::getTileExecutor(module).getCount();
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

    Shape divisors(newInShape.size(), 1);
    divisors[tileDim] = tilesNum;

    return fillDividedTiles(op, divisors, inShape);
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

class ApplyTilingMVN1Sum final : public VPU::arch37xx::ApplyTilingMVN1SumBase<ApplyTilingMVN1Sum> {
public:
    explicit ApplyTilingMVN1Sum(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class MVN1SumTiling;
    class MVN1SumCorrectHeight;

private:
    void safeRunOnFunc() final;
};

//
// MVN1SumTiling
//

class ApplyTilingMVN1Sum::MVN1SumTiling final : public mlir::OpRewritePattern<VPU::MVN1SumOp> {
public:
    MVN1SumTiling(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPU::MVN1SumOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVN1SumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ApplyTilingMVN1Sum::MVN1SumTiling::matchAndRewrite(VPU::MVN1SumOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (mlir::isa_and_nonnull<VPU::SliceOp>(origOp.getInput().getDefiningOp())) {
        return matchFailed(rewriter, origOp, "Op already tiled.");
    }

    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.getSum().getType().cast<vpux::NDTypeInterface>();
    if (origOp.fitIntoCMX(SmallVector<vpux::NDTypeInterface>{inputType, outputType})) {
        return matchFailed(rewriter, origOp, "Op fits into CMX");
    }

    const auto tiles = findNumOfTiles(origOp);
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
    MVN1SumCorrectHeight(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPU::MVN1SumOp>(ctx, benefit), _log(log) {
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

    const auto output = origOp.getSum().getType().cast<vpux::NDTypeInterface>();

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
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MVN1SumCorrectHeight>(&ctx, benefitLevels[0], _log);
    patterns.add<MVN1SumTiling>(&ctx, benefitLevels[1], _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createApplyTilingMVN1SumPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createApplyTilingMVN1SumPass(Logger log) {
    return std::make_unique<ApplyTilingMVN1Sum>(log);
}
