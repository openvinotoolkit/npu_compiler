//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/utils/core/error.hpp"

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
    auto topModuleOp = getOperation();

    auto topPipelineOptions = topModuleOp.getOps<config::PipelineOptionsOp>();
    VPUX_THROW_WHEN(
            topPipelineOptions.empty() || std::distance(topPipelineOptions.begin(), topPipelineOptions.end()) > 1,
            "No valid count of config.PipelineOptionsOp found {0}",
            std::distance(topPipelineOptions.begin(), topPipelineOptions.end()));

    auto topPipelineOptionsOp = *topPipelineOptions.begin();
    auto nestedModules = topModuleOp.getOps<mlir::ModuleOp>();

    if (nestedModules.empty()) {
        return;
    }

    for (auto nestedModuleOp : nestedModules) {
        if (nestedModuleOp == topModuleOp) {
            continue;
        }
        if (!isModuleExecutable(nestedModuleOp)) {
            continue;
        }

        for (auto attr : topModuleOp->getAttrs()) {
            if (!nestedModuleOp->hasAttr(attr.getName())) {
                nestedModuleOp->setAttr(attr.getName(), attr.getValue());
            }
        }

        mlir::OpBuilder nestedBuilder(nestedModuleOp.getRegion());
        nestedBuilder.clone(*topPipelineOptionsOp);

        for (auto reservedResource : topModuleOp.getOps<config::ResourcesOp>()) {
            nestedBuilder.clone(*reservedResource);
        }
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createCloneReservedResourcesFromTopModulePass(Logger log) {
    return std::make_unique<CloneReservedResourcesFromTopModulePass>(log);
}
