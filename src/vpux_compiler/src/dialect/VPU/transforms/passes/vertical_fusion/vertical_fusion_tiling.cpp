//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vf_tiling_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_tiling_rewriter.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
    explicit VfTilingPass(bool enableVerticalFusionPipelining, bool enableVFScheduleTrace,
                          const WorkloadManagementMode workloadManagementMode, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _enableVerticalFusionScheduleTrace(enableVFScheduleTrace),
              _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
    bool _enableVerticalFusionScheduleTrace = false;
    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_1_PAGES;
};

mlir::LogicalResult VfTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading VfTilingPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    if (enableVFScheduleTrace.hasValue()) {
        _log.trace("Overloading VfTilingPass argument by MLIR variable");
        _enableVerticalFusionScheduleTrace = enableVFScheduleTrace;
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
    auto maybeLayerCostModelAnalysis = getCachedParentAnalysis<VPU::LayerCostModelAnalysis>(module);
    auto layerCostModel =
            VPU::LayerCostModelAnalysis::getOrCreateLayerCostModel(maybeLayerCostModelAnalysis, &ctx, _log);

    const auto costFunction = std::make_unique<VPU::LayerVPUNNCost>(func, layerCostModel, _log);

    func->walk([vfIndex = 0ll](VPU::VerticalFusionOp vfOp) mutable {
        vfOp->setAttr(VF_LOOP_INDEX_ATTR_NAME, VFLoopIndexAttr::get(vfOp->getContext(), vfIndex));
        ++vfIndex;
    });

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::VerticalFusionOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();

    if (config::isPureHostCompileFunc(func)) {
        // host pipeline related
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();
        target.addLegalDialect<mlir::tensor::TensorDialect>();
    }

    mlir::RewritePatternSet patterns(&ctx);

    if (_workloadManagementMode <= WorkloadManagementMode::PWLM_V0_1_PAGES) {
        patterns.add<VPU::VF::v1::VerticalFusionTilingRewriter>(&ctx, _enableVerticalFusionPipelining, costFunction,
                                                                _log);
    } else {
        if (_enableVerticalFusionScheduleTrace) {
            _log.trace("Vertical Fusion Schedule Tracing is enabled.");
            VPU::VF::v2::printVFSchedulingTrace(func, costFunction, _log);
        }
        auto& vfCache = getAnalysis<VPU::VF::v2::VFCacheAnalysis>();
        patterns.add<VPU::VF::v2::VerticalFusionTilingRewriter>(&ctx, _enableVerticalFusionPipelining, costFunction,
                                                                _log, vfCache);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createVfTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createVfTilingPass(bool enableVerticalFusionPipelining, bool enableVFScheduleTrace,
                                                    const WorkloadManagementMode workloadManagementMode, Logger log) {
    return std::make_unique<VfTilingPass>(enableVerticalFusionPipelining, enableVFScheduleTrace, workloadManagementMode,
                                          log);
}
