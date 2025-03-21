//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/utils/color_bin_barrier_assignment.hpp"

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
                                                           origOp.getIsStartBarrier());
    return mlir::success();
}

//
// BarrierColorBinVirtualBarrierRewrite
//

class BarrierColorBinVirtualBarrierRewrite final : public mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp> {
public:
    BarrierColorBinVirtualBarrierRewrite(mlir::MLIRContext* ctx, BarrierGraphInfo& BarrierGraphInfo,
                                         VPURT::BarrierColorBin& BarrierColorBinAssignment, Logger log)
            : mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp>(ctx),
              _BarrierGraphInfo(BarrierGraphInfo),
              _BarrierColorBin(BarrierColorBinAssignment),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    BarrierGraphInfo& _BarrierGraphInfo;
    VPURT::BarrierColorBin& _BarrierColorBin;
    Logger _log;
};

mlir::LogicalResult BarrierColorBinVirtualBarrierRewrite::matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DeclareVirtualBarrierOp Operation '{0}'", origOp->getLoc());

    auto& barrierInfo = _BarrierGraphInfo.getBarrierInfo();
    auto barrierid = barrierInfo.getIndex(origOp);
    auto physicalId = _BarrierColorBin.getPhysicalBarrier(barrierid);

    rewriter.replaceOpWithNewOp<VPURT::ConfigureBarrierOp>(origOp, physicalId, origOp.getIsFinalBarrier());
    return mlir::success();
}

//
// AssignPhysicalBarriersPass
//

class AssignPhysicalBarriersPass final : public VPURT::impl::AssignPhysicalBarriersBase<AssignPhysicalBarriersPass> {
public:
    explicit AssignPhysicalBarriersPass(const bool barrierColorBinFlag,
                                        std::optional<int> virtualBarrierThresholdForWlm, Logger log)
            : _barrierColorBinFlag(barrierColorBinFlag), _virtualBarrierThresholdForWlm(virtualBarrierThresholdForWlm) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool _barrierColorBinFlag;
    std::optional<int> _virtualBarrierThresholdForWlm = std::nullopt;
};

void AssignPhysicalBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto wlmFlag = vpux::VPUIP::getWlmStatus(module) == vpux::VPUIP::WlmStatus::ENABLED;

    const auto barrierColorBinFlag =
            colorBinEnableOpt.hasValue() ? static_cast<bool>(colorBinEnableOpt.getValue()) : _barrierColorBinFlag;

    VPURT::BarrierSimulator barrierSim(func);

    auto barriers = func.getOps<VPURT::DeclareVirtualBarrierOp>();
    auto numVirtualBarriers = static_cast<int64_t>(std::distance(barriers.begin(), barriers.end()));

    auto virtualBarrierThresholdForWlm = virtualBarrierThresholdForWlmOpt.hasValue() ? virtualBarrierThresholdForWlmOpt
                                                                                     : _virtualBarrierThresholdForWlm;

    if (wlmFlag && virtualBarrierThresholdForWlm.has_value() &&
        numVirtualBarriers > virtualBarrierThresholdForWlm.value()) {
        _log.trace("WLM flag turned off because number of barrier is above threshold {0} > {1}", numVirtualBarriers,
                   virtualBarrierThresholdForWlm.value());
        wlmFlag = false;
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
    }

    if (!barrierSim.isDynamicBarriers()) {
        return;
    }
    if (mlir::failed(barrierSim.checkProducerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    if (mlir::failed(barrierSim.checkProducerAndConsumerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }

    if (barrierColorBinFlag && numVirtualBarriers <= numBarriers) {
        _log.info("BarrierColorBin optimization requested but will not be used because physical barrier amount {0} is "
                  "enough to cover all virtual barriers {1}",
                  numBarriers, numVirtualBarriers);
    }

    mlir::RewritePatternSet patterns(&ctx);
    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPURT::DeclareVirtualBarrierOp>();
    target.addLegalOp<VPURT::ConfigureBarrierOp>();

    if (barrierColorBinFlag && numVirtualBarriers > numBarriers) {
        auto& barrierGraphInfo = getAnalysis<BarrierGraphInfo>();
        auto arch = VPU::getArch(func);
        VPURT::BarrierColorBin BarrierColorBinAssignment(numBarriers, arch, _log);

        // Apply color binning algorithm for physical barrier assignment
        VPUX_THROW_UNLESS(BarrierColorBinAssignment.calculateBinSize(barrierGraphInfo),
                          "BarrierColorBin failed during bin size calulation");
        VPUX_THROW_UNLESS(BarrierColorBinAssignment.assignPhysicalBarrier(barrierGraphInfo, barrierSim),
                          "BarrierColorBin failed during barrier assignment");

        BarrierColorBinAssignment.reorderBarriers(barrierGraphInfo, func);
        patterns.add<BarrierColorBinVirtualBarrierRewrite>(&ctx, barrierGraphInfo, BarrierColorBinAssignment, _log);

        if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
            signalPassFailure();
        }
        barrierGraphInfo.clearAttributes();

    } else {
        // Use old round-robin method of assigining physical barriers
        if (wlmFlag) {
            auto barrierInfo = vpux::BarrierInfo{func};
            barrierSim.configureForWlm(barrierInfo);
            if (mlir::failed(barrierSim.simulateBarriers(_log.nest(), numBarriers))) {
                _log.error("Barrier simulation (with WLM restrictions) failed with {0} barriers", numBarriers);
                signalPassFailure();
                return;
            }
            barrierInfo.clearAttributes();
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
}
}  // namespace

//
// createAssignPhysicalBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignPhysicalBarriersPass(
        const bool barrierColorBinFlag, std::optional<int> virtualBarrierThresholdForWlm, Logger log) {
    return std::make_unique<AssignPhysicalBarriersPass>(barrierColorBinFlag, virtualBarrierThresholdForWlm, log);
}
