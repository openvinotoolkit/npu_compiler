//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SPLITNCEOPSONTOWORKLOADS
#define GEN_PASS_DEF_SPLITNCEOPSONTOWORKLOADS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// NCEWorkloadSplitRewrite
//

class NCEWorkloadSplitRewrite final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    NCEWorkloadSplitRewrite(mlir::MLIRContext* ctx, int64_t numDPU, config::ArchKind arch,
                            std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx),
              _numDPU(numDPU),
              _arch(arch),
              _costModel(std::move(costModel)),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numDPU;
    config::ArchKind _arch;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    Logger _log;
};

mlir::LogicalResult NCEWorkloadSplitRewrite::matchAndRewrite(VPU::NCEOpInterface nceOp,
                                                             mlir::PatternRewriter& rewriter) const {
    return genericNCEWorkloadSplit(nceOp, rewriter, _arch, _numDPU, _costModel, _log);
}

//
// NCEWorkloadSplitPreSplitRewrite
//

class NCEWorkloadSplitPreSplitRewrite final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    NCEWorkloadSplitPreSplitRewrite(mlir::MLIRContext* ctx, int64_t numDPU, int64_t numTiles, config::ArchKind arch,
                                    std::shared_ptr<VPUNN::VPULayerCostModel> layerCostModel,
                                    std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx),
              _numDPU(numDPU),
              _numTiles(numTiles),
              _arch(arch),
              _layerCostModel(std::move(layerCostModel)),
              _costModel(std::move(costModel)),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPUNN::LayerSplitInfo retrieveLayerSplitInfoFromVPUNN(VPU::NCEOpInterface nceOp, uint32_t& cost, Logger log) const;

private:
    int64_t _numDPU;
    int64_t _numTiles;

    config::ArchKind _arch;
    std::shared_ptr<VPUNN::VPULayerCostModel> _layerCostModel;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;

    Logger _log;
};

VPUNN::LayerSplitInfo NCEWorkloadSplitPreSplitRewrite::retrieveLayerSplitInfoFromVPUNN(VPU::NCEOpInterface nceOp,
                                                                                       uint32_t& cost,
                                                                                       Logger log) const {
    const auto costParams = getWorkloadCostParam(nceOp, _arch, _numDPU, _numTiles);

    auto perClusterVPUNNLayers = VPU::getPerClusterDPULayers(nceOp, costParams, log);
    VPUNN::LayerSplitInfo layerSplitInfo;
    cost = checkAndReturnCost(
            _layerCostModel->LayersPreSplit(perClusterVPUNNLayers, _numDPU, /*input_in_ddr=*/false,
                                            /*output_in_ddr=*/false, /*prefetching=*/true, layerSplitInfo),
            log);
    VPUX_THROW_WHEN(layerSplitInfo.empty(), "layerSplitInfo is empty with cost '{0}'", cost);
    if (cost >= VPU::INVALID_COST_BASE) {
        log.warning("Get invalid cost for op {0}", nceOp);
        printVPUNNLayers(perClusterVPUNNLayers, log);
        VPUX_THROW("Cost is Invalid");
    }
    return layerSplitInfo;
}

mlir::LogicalResult NCEWorkloadSplitPreSplitRewrite::matchAndRewrite(VPU::NCEOpInterface nceOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());
    if (!isSupportedPreSplitNCEOp(nceOp)) {
        _log.trace("Use heuristic mode for op {0}'", nceOp->getName());
        return genericNCEWorkloadSplit(nceOp, rewriter, _arch, _numDPU, _costModel, _log);
    }
    uint32_t cost = 0;
    auto layerSplitInfo = retrieveLayerSplitInfoFromVPUNN(nceOp, cost, _log);

    // Track E#159358
    // VPUNN will update the workload split logic to meet max workload requirement
    // This check can be removed when VPUNN submodule is updated
    const auto maxAvailableSlots = VPUIP::getBarrierMaxVariantCount(nceOp);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(nceOp);
    const auto availableSlot = std::min(maxAvailableSlots, maxSlotsSum) / 2;
    for (auto item : layerSplitInfo) {
        // item OneTileLayerInfo
        auto workloadCost = item.best_intra_tile_split;
        if (workloadCost.second.size() > availableSlot) {
            _log.trace("There are too many workloads, Use heuristic mode for op {0}'", nceOp->getName());
            return genericNCEWorkloadSplit(nceOp, rewriter, _arch, _numDPU, _costModel, _log);
        }
    }

    // For debug purpose, check VPUNN split by `VPU::printLayerSplitInfo(layerSplitInfo, _log)`
    // Apply workload split
    _log.trace("Applying workload split");
    rewriter.modifyOpInPlace(nceOp, [&]() {
        splitWorkloadsWithInfo(nceOp, rewriter, layerSplitInfo, _log);
    });

    nceOp->setAttr(DPUCost, getIntAttr(nceOp->getContext(), cost));

    return mlir::success();
}

//
// SplitNCEOpsOntoWorkloadsPass
//

class SplitNCEOpsOntoWorkloadsPass final :
        public VPU::impl::SplitNCEOpsOntoWorkloadsBase<SplitNCEOpsOntoWorkloadsPass> {
public:
    explicit SplitNCEOpsOntoWorkloadsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitNCEOpsOntoWorkloadsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = config::getArch(module);

    auto nceCluster = config::getTileExecutor(module);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(VPU::ExecutorKind::DPU);
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    const auto numDPUs = dpuExec.getCount();

    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            return !nceOp.getWorkloads().empty();
        }
        return true;
    });
    target.addLegalOp<VPU::DPUWorkloadOp>();

    mlir::RewritePatternSet patterns(&ctx);
    const auto enableVPUNNPreSplit = config::hasVPUNNPreSplit(module);
    if (enableVPUNNPreSplit) {
        const auto numTiles = nceCluster.getCount();
        auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
        auto layerCostModel =
                VPU::LayerCostModelAnalysis::getOrCreateLayerCostModel(maybeLayerCostModelAnalysis, arch, _log);
        patterns.add<NCEWorkloadSplitPreSplitRewrite>(&ctx, numDPUs, numTiles, arch, layerCostModel, costModel, _log);
    } else {
        patterns.add<NCEWorkloadSplitRewrite>(&ctx, numDPUs, arch, costModel, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitNCEOpsOntoWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitNCEOpsOntoWorkloadsPass(Logger log) {
    return std::make_unique<SplitNCEOpsOntoWorkloadsPass>(log);
}
