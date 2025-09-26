//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/move_view_ops_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/move_view_ops_rewriter.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_MOVEVIEWOPSTOVF
#define GEN_PASS_DEF_MOVEVIEWOPSTOVF
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// MoveViewOpsToVFPass
//

class MoveViewOpsToVFPass final : public VPU::impl::MoveViewOpsToVFBase<MoveViewOpsToVFPass> {
public:
    explicit MoveViewOpsToVFPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
};

mlir::LogicalResult MoveViewOpsToVFPass::initialize(mlir::MLIRContext* ctx) {
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

void MoveViewOpsToVFPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);

    if (_workloadManagementMode <= WorkloadManagementMode::PWLM_V0_LCA) {
        patterns.add<VPU::VF::v1::MoveViewOpsRewriter>(&ctx, _log);
    } else {
        patterns.add<VPU::VF::v2::MoveViewOpsRewriter>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveViewOpsToVerticalFusionPass
//

std::unique_ptr<mlir::Pass> VPU::createMoveViewOpsToVerticalFusionPass(
        const WorkloadManagementMode workloadManagementMode, Logger log) {
    return std::make_unique<MoveViewOpsToVFPass>(workloadManagementMode, log);
}
