//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/IR/Iterators.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_LINALGTILEANDFUSESWLAYERS
#define GEN_PASS_DEF_LINALGTILEANDFUSESWLAYERS
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {
class LinalgTileAndFuseSwLayersPass final :
        public ShaveCodeGen::impl::LinalgTileAndFuseSwLayersBase<LinalgTileAndFuseSwLayersPass> {
public:
    explicit LinalgTileAndFuseSwLayersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;
    mlir::LogicalResult tileAndFuseOp(mlir::TilingInterface tileOp, bool isReduce);
    mlir::LogicalResult tileAndFuseOps(mlir::func::FuncOp func, mlir::MLIRContext& ctx, bool isReduce);
};

std::optional<SmallVector<mlir::OpFoldResult>> getTileSizes(mlir::TilingInterface tileOp, bool isReduce,
                                                            mlir::IRRewriter& rewriter) {
    // Use a tile size of one on all applicable dimensions for now.
    // This works in all cases and avoids the issue of picking out a sensible
    // tile size (at least until we want to start using the vector dialect).
    // Note that we'll still get vector code out at the end since
    // llvm will/should be able to vectorize.
    auto zero = rewriter.getIndexAttr(0);
    auto one = rewriter.getIndexAttr(1);
    SmallVector<mlir::OpFoldResult> tileSizes;
    bool hasValidDim = false;
    for (auto itTy : tileOp.getLoopIteratorTypes()) {
        // If the iterator type matches the type we're looking for (reduction for reduce or parallel otherwise)
        // then choose a tile size of one. If not, choose a tile size of zero for the dimension (meaning it
        // doesn't get tiled).
        if ((itTy == mlir::utils::IteratorType::reduction) == isReduce) {
            tileSizes.push_back(one);
            hasValidDim = true;
            continue;
        }
        tileSizes.push_back(zero);
    }
    if (!hasValidDim) {
        return std::nullopt;
    }
    return std::make_optional(tileSizes);
}

mlir::LogicalResult LinalgTileAndFuseSwLayersPass::tileAndFuseOp(mlir::TilingInterface tileOp, bool isReduce) {
    mlir::IRRewriter rewriter(tileOp);
    if (tileOp->use_empty()) {
        _log.trace("Operation {0} is dead, nothing to do", tileOp->getName());
        rewriter.eraseOp(tileOp);
        return mlir::success();
    }

    auto tileSizes = getTileSizes(tileOp, isReduce, rewriter);
    if (!tileSizes) {
        _log.trace("Operation {0} is already tiled, nothing to do", tileOp->getName());
        return mlir::success();
    }

    mlir::scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(*tileSizes);
    tilingOptions.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForOp);

    if (isReduce) {
        // Don't fuse reduce ops for now. For fusion we would have to check
        // the legality. Additionally this seems to work just as well for all
        // cases so far.
        auto tiledResults = mlir::scf::tileUsingSCF(rewriter, tileOp, tilingOptions);
        if (failed(tiledResults)) {
            _log.trace("Failed to tile reduction dimensions for {0}", tileOp->getName());
            return mlir::failure();
        }
        _log.trace("Success tiling reduction dimensions for {0}", tileOp->getName());
        rewriter.replaceOp(tileOp, tiledResults->replacements);
        return mlir::success();
    }

    mlir::scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.tilingOptions = std::move(tilingOptions);
    auto tiledResults = mlir::scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tileOp, tileAndFuseOptions);

    if (failed(tiledResults)) {
        _log.trace("Failed to tile and fuse parallel dimensions for {0}", tileOp->getName());
        return mlir::failure();
    }
    _log.trace("Success for tile and fuse on parallel dimensions for {0}", tileOp->getName());

    for (mlir::OpResult res : tileOp->getResults()) {
        if (auto replacement = tiledResults->replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, replacement);
        }
    }

    if (tileOp->use_empty()) {
        rewriter.eraseOp(tileOp);
    }

    return mlir::success();
}

mlir::LogicalResult LinalgTileAndFuseSwLayersPass::tileAndFuseOps(mlir::func::FuncOp func, mlir::MLIRContext& ctx,
                                                                  bool isReduce) {
    mlir::LogicalResult allMatched = mlir::success();
    SmallVector<mlir::TilingInterface> candidates;
    func.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>([&](mlir::Operation* op) {
        auto tileOp = mlir::dyn_cast<mlir::TilingInterface>(op);
        if (!tileOp) {
            return;
        }
        candidates.push_back(tileOp);
    });
    for (auto op : candidates) {
        if (failed(tileAndFuseOp(op, isReduce))) {
            allMatched = mlir::failure();
        }
    }

    if (failed(allMatched)) {
        return mlir::failure();
    }

    // Cleanup the IR post-tiling phase.
    mlir::RewritePatternSet patterns(&ctx);
    mlir::linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    mlir::tensor::populateFoldTensorEmptyPatterns(patterns);
    if (failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return mlir::failure();
    }
    return mlir::success();
}

void LinalgTileAndFuseSwLayersPass::safeRunOnModule() {
    // Algorithm inspired from the IREE LLVMCPUTileAndFuse pass.
    // We greedily tile in two phases, first over all parallel dimensions,
    // then handle the reductions in a second phase. Currently
    // we don't fuse producers for reduction tiling to avoid legality
    // issues.
    auto& ctx = getContext();
    auto moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

    auto funcOps = swModule.getOps<mlir::func::FuncOp>();
    for (auto func : funcOps) {
        if (failed(tileAndFuseOps(func, ctx, /*isReduce=*/false))) {
            signalPassFailure();
            return;
        }
        // Finally tile on the reduce dimensions.
        if (failed(tileAndFuseOps(func, ctx, /*isReduce=*/true))) {
            signalPassFailure();
            return;
        }
    }

    return;
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createLinalgTileAndFuseSwLayersPass(Logger log) {
    return std::make_unique<LinalgTileAndFuseSwLayersPass>(log);
}
