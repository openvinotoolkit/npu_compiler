//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ASSIGNTILESFORSWIODMAMODULES
#define GEN_PASS_DEF_ASSIGNTILESFORSWIODMAMODULES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Returns the SW IO DMA ops of the kernel function if they are its only compute ops, an empty list otherwise.
// Only constants, aux buffer allocations and function boundary casts are allowed next to them, so that no
// other compute op is moved to a multi-tile module.
SmallVector<VPU::SwIoDmaOpInterface> getSwIoDmaOnlyOps(mlir::func::FuncOp funcOp) {
    SmallVector<VPU::SwIoDmaOpInterface> swIoDmaOps;

    const auto walkResult = funcOp.walk([&](mlir::Operation* op) {
        if (op == funcOp.getOperation() ||
            mlir::isa<mlir::func::ReturnOp, VPU::EmptyOp, Core::ReinterpretCastOp, Const::DeclareOp>(op)) {
            return mlir::WalkResult::advance();
        }

        auto swIoDmaOp = mlir::dyn_cast<VPU::SwIoDmaOpInterface>(op);
        if (swIoDmaOp == nullptr) {
            return mlir::WalkResult::interrupt();
        }

        swIoDmaOps.push_back(swIoDmaOp);
        return mlir::WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
        return {};
    }

    return swIoDmaOps;
}

//
// AssignTilesForSwIoDmaModulesPass
//

class AssignTilesForSwIoDmaModulesPass final :
        public VPU::impl::AssignTilesForSwIoDmaModulesBase<AssignTilesForSwIoDmaModulesPass> {
public:
    explicit AssignTilesForSwIoDmaModulesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void AssignTilesForSwIoDmaModulesPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    // Only platforms that define MinTiles above one tile get multi-tile kernel modules
    const auto minTiles = config::getMinTiles(moduleOp);
    if (!minTiles.has_value() || minTiles.value() <= 1) {
        return;
    }

    VPUX_THROW_UNLESS(config::hasTileExecutor(moduleOp), "Expected module '{0}' to have 'config::ResourcesOp'",
                      moduleOp.getName());
    auto tileOp = config::getTileExecutor(moduleOp);

    // Only single-tile compilations are changed; a module compiled for more tiles keeps them
    if (tileOp.getCount() != 1) {
        _log.trace("Module '{0}' is compiled for {1} tiles, skip", moduleOp.getName(), tileOp.getCount());
        return;
    }

    // PackNestedModules gives its own PipelineOptions and Resources only to modules with a single function
    auto funcOps = to_small_vector(moduleOp.getOps<mlir::func::FuncOp>());
    if (funcOps.size() != 1) {
        _log.warning("Expected kernel module '{0}' to have a single function, got {1}", moduleOp.getName(),
                     funcOps.size());
        signalPassFailure();
        return;
    }

    const auto swIoDmaOps = getSwIoDmaOnlyOps(funcOps.front());
    if (swIoDmaOps.empty()) {
        return;
    }

    const auto numTiles = minTiles.value();
    const auto maxTiles = checked_cast<int64_t>(VPU::getMaxDPUClusterNum(config::getPlatform(moduleOp)));
    if (numTiles > maxTiles) {
        moduleOp.emitError() << "'" << config::MIN_TILES << "' (" << numTiles
                             << ") exceeds the tile count of the platform (" << maxTiles << ")";
        signalPassFailure();
        return;
    }

    // Multi-cluster strategies were assigned for the network tile count, before the kernel modules were created, so
    // ops that accept Clustering on the new tile count get it here, as MultiClusterStrategyAssignment would. The other
    // ops stay single-tile, as in a multi-tile compilation. Every SW IO DMA op is a ClusteredOpInterface
    // (VPU_SwIoDmaOp). Clustering keeps whole tensors as input and output: every tile runs the kernel with its own copy
    // of the CMX aux buffer and the kernel splits the work at runtime through tile_id / shave_id.
    SmallVector<VPU::ClusteredOpInterface> clusteringOps;
    for (auto swIoDmaOp : swIoDmaOps) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(swIoDmaOp.getOperation());
        if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering,
                                                   checked_cast<size_t>(numTiles))) {
            clusteringOps.push_back(clusteredOp);
        } else {
            _log.trace("'{0}' at '{1}' does not support Clustering, it stays on a single tile", swIoDmaOp->getName(),
                       swIoDmaOp->getLoc());
        }
    }

    if (clusteringOps.empty()) {
        _log.trace("No op of module '{0}' can use more tiles, skip", moduleOp.getName());
        return;
    }

    _log.info("Set {0} tiles for module '{1}', {2} of {3} SW IO DMA op(s) use Clustering", numTiles, moduleOp.getName(),
              clusteringOps.size(), swIoDmaOps.size());

    tileOp.setCount(numTiles);
    for (auto clusteredOp : clusteringOps) {
        clusteredOp.setMultiClusterStrategy(VPU::MultiClusterStrategy::Clustering);
    }
}

}  // namespace

//
// createAssignTilesForSwIoDmaModulesPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAssignTilesForSwIoDmaModulesPass(Logger log) {
    return std::make_unique<AssignTilesForSwIoDmaModulesPass>(log);
}
