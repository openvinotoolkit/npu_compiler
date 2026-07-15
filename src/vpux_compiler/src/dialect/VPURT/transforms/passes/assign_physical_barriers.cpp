//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <llvm/ADT/SetOperations.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPURT {
#define GEN_PASS_DECL_ASSIGNPHYSICALBARRIERS
#define GEN_PASS_DEF_ASSIGNPHYSICALBARRIERS
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

//
// VirtualBarrierRewrite
//

class VirtualBarrierRewrite final : public mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp> {
public:
    VirtualBarrierRewrite(mlir::MLIRContext* ctx, const VPURT::BarrierSimulator& barrierSim, Logger log)
            : mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp>(ctx), _barrierSim(barrierSim), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const VPURT::BarrierSimulator& _barrierSim;
    Logger _log;
};

mlir::LogicalResult VirtualBarrierRewrite::matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DeclareVirtualBarrierOp Operation '{0}'", origOp->getLoc());

    const auto& conf = _barrierSim.getConfig(origOp.getBarrier());
    _log.nest().trace("Use physical barrier ID '{0}'", conf.realId);

    rewriter.replaceOpWithNewOp<VPURT::ConfigureBarrierOp>(origOp, conf.realId, origOp.getIsFinalBarrier(),
                                                           origOp.getIsStartBarrier(), origOp.getWlmPageAttr());
    return mlir::success();
}

//
// AssignPhysicalBarriersPass
//

class AssignPhysicalBarriersPass final : public VPURT::impl::AssignPhysicalBarriersBase<AssignPhysicalBarriersPass> {
public:
    explicit AssignPhysicalBarriersPass(std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::optional<WorkloadManagementMode> _workloadManagementMode = std::nullopt;
};

void AssignPhysicalBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    VPURT::BarrierSimulator barrierSim(func);

    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt;
    }

    if (!barrierSim.isDynamicBarriers()) {
        return;
    }

    auto wlmFlag = !config::isArchVPUX3XXX(config::getArch(module));
    if (!wlmFlag) {
        if (mlir::failed(barrierSim.checkProducerCount(_log.nest()))) {
            signalPassFailure();
            return;
        }
        if (mlir::failed(barrierSim.checkProducerAndConsumerCount(_log.nest()))) {
            signalPassFailure();
            return;
        }
    }

    mlir::RewritePatternSet patterns(&ctx);
    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPURT::DeclareVirtualBarrierOp>();
    target.addLegalOp<VPURT::ConfigureBarrierOp>();

    // Use old round-robin method of assigning physical barriers
    if (wlmFlag) {
        _log.trace("Assign barriers using WLM page approach");
        auto partialWlmEnabled = (_workloadManagementMode.has_value() &&
                                  _workloadManagementMode.value() != WorkloadManagementMode::FWLM_V1_PAGES);
        if (mlir::failed(barrierSim.simulateBarriersForWlmPageApproach(_log.nest(), numBarriers, partialWlmEnabled))) {
            _log.error("Barrier simulation (with WLM page) failed with {0} barriers", numBarriers);
            signalPassFailure();
            return;
        }
    } else {
        if (mlir::failed(barrierSim.simulateBarriers(_log.nest(), numBarriers))) {
            _log.error("Barrier simulation failed with {0} barriers", numBarriers);
            signalPassFailure();
            return;
        }
    }
    patterns.add<VirtualBarrierRewrite>(&ctx, barrierSim, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAssignPhysicalBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignPhysicalBarriersPass(
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<AssignPhysicalBarriersPass>(workloadManagementMode, log);
}
