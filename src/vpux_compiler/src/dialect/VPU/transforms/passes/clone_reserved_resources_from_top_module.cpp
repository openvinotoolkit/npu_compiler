//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CLONERESERVEDRESOURCESFROMTOPMODULE
#define GEN_PASS_DEF_CLONERESERVEDRESOURCESFROMTOPMODULE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class CloneReservedResourcesFromTopModulePass final :
        public VPU::impl::CloneReservedResourcesFromTopModuleBase<CloneReservedResourcesFromTopModulePass> {
public:
    explicit CloneReservedResourcesFromTopModulePass(Logger log): CloneReservedResourcesFromTopModuleBase() {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    bool isModuleExecutable(mlir::ModuleOp moduleOp);
    void safeRunOnModule() final;
};

bool CloneReservedResourcesFromTopModulePass::isModuleExecutable(mlir::ModuleOp moduleOp) {
    bool isExecutableNestedModule = false;
    moduleOp->walk([&](mlir::func::FuncOp) {
        isExecutableNestedModule = true;
    });

    return isExecutableNestedModule;
}

void CloneReservedResourcesFromTopModulePass::safeRunOnModule() {
    auto nestedModuleOp = getOperation();
    if (!isModuleExecutable(nestedModuleOp)) {
        return;
    }

    auto topModuleOp = nestedModuleOp->getParentOfType<mlir::ModuleOp>();
    if (!topModuleOp) {
        // If we are already at the top-level module, just return. No error emission is needed.
        // This makes the pass safe to run in initializePipeline: it does nothing on the top module,
        // but copies ReservedResource when initializePipeline runs on nested modules.
        return;
    }

    auto nestedTileExecutor = config::getTileExecutor(nestedModuleOp);
    auto topTileExecutor = config::getTileExecutor(topModuleOp);
    if (!nestedTileExecutor || !topTileExecutor) {
        return signalPassFailure();
    }

    auto reservedResources = topTileExecutor.lookupSymbol<mlir::ModuleOp>(config::resMemModuleName);
    if (!reservedResources) {
        return signalPassFailure();
    }

    mlir::OpBuilder nestedBuilder(nestedTileExecutor->getRegion(0));
    nestedBuilder.clone(*reservedResources);
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createCloneReservedResourcesFromTopModulePass(Logger log) {
    return std::make_unique<CloneReservedResourcesFromTopModulePass>(log);
}
