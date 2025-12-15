//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/wrap_vf_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/wrap_vf_rewriter.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_WRAPVERTICALFUSIONREGION
#define GEN_PASS_DEF_WRAPVERTICALFUSIONREGION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// WrapVerticalFusionRegionPass
//

class WrapVerticalFusionRegionPass final :
        public VPU::impl::WrapVerticalFusionRegionBase<WrapVerticalFusionRegionPass> {
public:
    explicit WrapVerticalFusionRegionPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
};

mlir::LogicalResult WrapVerticalFusionRegionPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt.getValue();
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void WrapVerticalFusionRegionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);

    if (_workloadManagementMode <= WorkloadManagementMode::PWLM_V0_LCA) {
        patterns.add<VPU::VF::v1::WrapVFRewriter>(&ctx, _log);
    } else {
        patterns.add<VPU::VF::v2::WrapVFRewriter>(&ctx, _log);
    }

    if (mlir::failed(
                mlir::applyPatternsGreedily(getOperation(), std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createWrapVerticalFusionRegion
//

std::unique_ptr<mlir::Pass> VPU::createWrapVerticalFusionRegionPass(const WorkloadManagementMode workloadManagementMode,
                                                                    Logger log) {
    return std::make_unique<WrapVerticalFusionRegionPass>(workloadManagementMode, log);
}
