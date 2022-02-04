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

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvToMultiCluster
//

class ConvToMultiCluster final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    ConvToMultiCluster(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvToMultiCluster");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvToMultiCluster::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Got operation {0}", origOp);
    if (origOp->getParentOfType<VPU::NCEClusterTilingOp>()) {
        _log.trace("The operation is already wrapped");
        return mlir::failure();
    }
    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };
    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    rewriter.replaceOpWithNewOp<VPU::NCEClusterTilingOp>(origOp, origOp->getResultTypes(), origOp->getOperands(),
                                                         bodyBuilder);
    return mlir::success();
}

//
// MultiClusterStrategyAssignmentPass
//

class MultiClusterStrategyAssignmentPass final :
        public VPU::MultiClusterStrategyAssignmentBase<MultiClusterStrategyAssignmentPass> {
public:
    explicit MultiClusterStrategyAssignmentPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MultiClusterStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getFunction();
    StrategyManager strategyManager(func, _log);
    strategyManager.computeOptimalMultiClusterStrategy();

    // Created distributed tensors
    strategyManager.insertCopyOpForDistributedTensor();

    auto& ctx = getContext();
    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<ConvToMultiCluster>(&ctx, _log);

    // if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig())))
    // {
    //     signalPassFailure();
    // }
}

}  // namespace

//
// createMultiClusterStrategyAssignmentPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createMultiClusterStrategyAssignmentPass(Logger log) {
    return std::make_unique<MultiClusterStrategyAssignmentPass>(log);
}