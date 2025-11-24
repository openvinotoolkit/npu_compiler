//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vf_tiling_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_tiling_rewriter.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/dialect.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_VFTILING
#define GEN_PASS_DEF_VFTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// VfTilingPass
//

class VfTilingPass final : public VPU::impl::VfTilingBase<VfTilingPass> {
public:
    explicit VfTilingPass(bool enableVerticalFusionPipelining, const WorkloadManagementMode workloadManagementMode,
                          Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA;
};

mlir::LogicalResult VfTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading VfTilingPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt.getValue();
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void VfTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
    auto layerCostModel =
            VPU::LayerCostModelAnalysis::getOrCreateLayerCostModel(maybeLayerCostModelAnalysis, arch, _log);

    const auto costFunction = std::make_unique<VPU::LayerVPUNNCost>(func, layerCostModel, _log);

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::VerticalFusionOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);

    if (_workloadManagementMode <= WorkloadManagementMode::PWLM_V0_LCA) {
        patterns.add<VPU::VF::v1::VerticalFusionTilingRewriter>(&ctx, _enableVerticalFusionPipelining, costFunction,
                                                                _log);
    } else {
        patterns.add<VPU::VF::v2::VerticalFusionTilingRewriter>(&ctx, _enableVerticalFusionPipelining, costFunction,
                                                                _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createVfTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createVfTilingPass(bool enableVerticalFusionPipelining,
                                                    const WorkloadManagementMode workloadManagementMode, Logger log) {
    return std::make_unique<VfTilingPass>(enableVerticalFusionPipelining, workloadManagementMode, log);
}
