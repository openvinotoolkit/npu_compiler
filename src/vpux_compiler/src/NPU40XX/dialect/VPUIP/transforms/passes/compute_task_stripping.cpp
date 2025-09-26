//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_COMPUTETASKSTRIPPING
#define GEN_PASS_DEF_COMPUTETASKSTRIPPING
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;
using namespace VPUIP;

namespace {

//
// Compute Task Stripping
//

class ComputeTaskStrippingPass final :
        public VPUIP::arch40xx::impl::ComputeTaskStrippingBase<ComputeTaskStrippingPass> {
public:
    explicit ComputeTaskStrippingPass(Logger log, VPU::DPUDryRunMode dpuDryRun, bool shaveDryRun)
            : _log(log), _dpuDryRun(dpuDryRun), _shaveDryRun(shaveDryRun) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

private:
    Logger _log;
    VPU::DPUDryRunMode _dpuDryRun;
    bool _shaveDryRun;
    bool isLegalExecutorType(VPURT::TaskOp taskOp);
    bool allBarrierProducersLegal(VPURT::BarrierOpInterface barrierOp, BarrierInfo& barrierInfo);
    bool allBarrierConsumersLegal(VPURT::BarrierOpInterface barrierOp, BarrierInfo& barrierInfo);
    void findTopLegalProducers(BarrierInfo::TaskSet& notLegalProducers, BarrierInfo::TaskSet& topLegalProducers,
                               BarrierInfo::TaskSet& tasksToRemove, BarrierInfo& barrierInfo);
    BarrierInfo::TaskSet legalizeBarrierProducers(VPURT::BarrierOpInterface barrierOp, BarrierInfo& barrierInfo);
    void findBottomLegalConsumers(BarrierInfo::TaskSet& notLegalConsumers, BarrierInfo::TaskSet& bottomLegalConsumers,
                                  BarrierInfo::TaskSet& tasksToRemove, BarrierInfo& barrierInfo);
    BarrierInfo::TaskSet legalizeBarrierConsumers(VPURT::BarrierOpInterface barrierOp, BarrierInfo& barrierInfo);
};

mlir::LogicalResult ComputeTaskStrippingPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    if (dpuDryRun.hasValue()) {
        _dpuDryRun = vpux::VPU::getDPUDryRunMode(dpuDryRun.getValue());
    }

    if (shaveDryRun.hasValue()) {
        _shaveDryRun = shaveDryRun.getValue();
    }

    return mlir::success();
}

bool ComputeTaskStrippingPass::isLegalExecutorType(VPURT::TaskOp taskOp) {
    auto opExecutorKind = taskOp.getExecutorKind();
    if (opExecutorKind == VPU::ExecutorKind::DPU && _dpuDryRun == VPU::DPUDryRunMode::STRIP) {
        return false;
    }
    if (opExecutorKind == VPU::ExecutorKind::SHAVE_ACT && _shaveDryRun == true) {
        return false;
    }
    return true;
}

bool ComputeTaskStrippingPass::allBarrierProducersLegal(VPURT::BarrierOpInterface barrierOp, BarrierInfo& barrierInfo) {
    for (const auto& producer : barrierInfo.getBarrierProducers(barrierOp)) {
        const auto producerOp = barrierInfo.getTaskOpAtIndex(producer);
        if (!ComputeTaskStrippingPass::isLegalExecutorType(producerOp)) {
            return false;
        }
    }
    return true;
}

bool ComputeTaskStrippingPass::allBarrierConsumersLegal(VPURT::BarrierOpInterface barrierOp, BarrierInfo& barrierInfo) {
    for (const auto& consumer : barrierInfo.getBarrierConsumers(barrierOp)) {
        const auto consumerOp = barrierInfo.getTaskOpAtIndex(consumer);
        if (!ComputeTaskStrippingPass::isLegalExecutorType(consumerOp)) {
            return false;
        }
    }
    return true;
}

void ComputeTaskStrippingPass::findTopLegalProducers(BarrierInfo::TaskSet& notLegalProducers,
                                                     BarrierInfo::TaskSet& topLegalProducers,
                                                     BarrierInfo::TaskSet& tasksToRemove, BarrierInfo& barrierInfo) {
    BarrierInfo::TaskSet notLegalTopProducers;
    for (auto producerInd : notLegalProducers) {
        for (auto waitBarier : barrierInfo.getWaitBarriers(producerInd)) {
            for (auto topProducerInd : barrierInfo.getBarrierProducers(waitBarier)) {
                if (ComputeTaskStrippingPass::isLegalExecutorType(barrierInfo.getTaskOpAtIndex(topProducerInd))) {
                    topLegalProducers.insert(topProducerInd);
                } else {
                    notLegalTopProducers.insert(topProducerInd);

                    tasksToRemove.insert(topProducerInd);
                }
            }
        }
    }

    if (!notLegalTopProducers.empty()) {
        ComputeTaskStrippingPass::findTopLegalProducers(notLegalTopProducers, topLegalProducers, tasksToRemove,
                                                        barrierInfo);
    }
}

BarrierInfo::TaskSet ComputeTaskStrippingPass::legalizeBarrierProducers(VPURT::BarrierOpInterface barrierOp,
                                                                        BarrierInfo& barrierInfo) {
    // Store tasks to remove
    BarrierInfo::TaskSet tasksToRemove;
    // Store legal producers
    BarrierInfo::TaskSet topLegalProducers;

    // Find not legal producers
    BarrierInfo::TaskSet notLegalProducers;
    for (auto producerInd : barrierInfo.getBarrierProducers(barrierOp)) {
        if (ComputeTaskStrippingPass::isLegalExecutorType(barrierInfo.getTaskOpAtIndex(producerInd))) {
            topLegalProducers.insert(producerInd);
        } else {
            notLegalProducers.insert(producerInd);
            tasksToRemove.insert(producerInd);
        }
    }

    // Find top legal producers
    ComputeTaskStrippingPass::findTopLegalProducers(notLegalProducers, topLegalProducers, tasksToRemove, barrierInfo);

    // If no legal producer everything above can be removed
    auto allBarrierProducers = barrierInfo.getBarrierProducers(barrierOp);
    if (topLegalProducers.empty()) {
        barrierInfo.removeProducers(barrierOp, allBarrierProducers);
        return tasksToRemove;
    }

    // Replace barrier producers with topLegalProducers
    barrierInfo.removeProducers(barrierOp, allBarrierProducers);
    barrierInfo.addProducers(barrierInfo.getIndex(barrierOp), topLegalProducers);

    return tasksToRemove;
}

void ComputeTaskStrippingPass::findBottomLegalConsumers(BarrierInfo::TaskSet& notLegalConsumers,
                                                        BarrierInfo::TaskSet& bottomLegalConsumers,
                                                        BarrierInfo::TaskSet& tasksToRemove, BarrierInfo& barrierInfo) {
    BarrierInfo::TaskSet notLegalBottomConsumers;
    for (auto consumerInd : notLegalConsumers) {
        for (auto updateBarier : barrierInfo.getUpdateBarriers(consumerInd)) {
            for (auto bottomConsumerInd : barrierInfo.getBarrierConsumers(updateBarier)) {
                if (ComputeTaskStrippingPass::isLegalExecutorType(barrierInfo.getTaskOpAtIndex(bottomConsumerInd))) {
                    bottomLegalConsumers.insert(bottomConsumerInd);
                } else {
                    notLegalBottomConsumers.insert(bottomConsumerInd);
                    tasksToRemove.insert(bottomConsumerInd);
                }
            }
        }
    }

    if (!notLegalBottomConsumers.empty()) {
        ComputeTaskStrippingPass::findBottomLegalConsumers(notLegalBottomConsumers, bottomLegalConsumers, tasksToRemove,
                                                           barrierInfo);
    }
}

BarrierInfo::TaskSet ComputeTaskStrippingPass::legalizeBarrierConsumers(VPURT::BarrierOpInterface barrierOp,
                                                                        BarrierInfo& barrierInfo) {
    // Store tasks to remove
    BarrierInfo::TaskSet tasksToRemove;
    // Store legal consumers
    BarrierInfo::TaskSet bottomLegalConsumers;

    // Find not legal consumers
    BarrierInfo::TaskSet notLegalConsumers;
    for (auto consumerInd : barrierInfo.getBarrierConsumers(barrierOp)) {
        if (ComputeTaskStrippingPass::isLegalExecutorType(barrierInfo.getTaskOpAtIndex(consumerInd))) {
            bottomLegalConsumers.insert(consumerInd);
        } else {
            notLegalConsumers.insert(consumerInd);
            tasksToRemove.insert(consumerInd);
        }
    }

    // Find bottom legal consumers
    ComputeTaskStrippingPass::findBottomLegalConsumers(notLegalConsumers, bottomLegalConsumers, tasksToRemove,
                                                       barrierInfo);

    // If no legal consumer everything bellow can be removed
    auto allBarrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
    if (bottomLegalConsumers.empty()) {
        barrierInfo.removeConsumers(barrierOp, allBarrierConsumers);
        return tasksToRemove;
    }

    // Replace barrier consumers with bottomLegalConsumers
    barrierInfo.removeConsumers(barrierOp, allBarrierConsumers);
    barrierInfo.addConsumers(barrierInfo.getIndex(barrierOp), bottomLegalConsumers);

    return tasksToRemove;
}

void ComputeTaskStrippingPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    const auto numOfBarriers = barrierInfo.getNumOfBarrierOps();
    if (numOfBarriers == 0) {
        return;
    }

    BarrierInfo::TaskSet tasksToErase;

    // Handle producers
    for (size_t barInd = 0; barInd < numOfBarriers; barInd++) {
        if (!allBarrierProducersLegal(barrierInfo.getBarrierOpAtIndex(barInd), barrierInfo)) {
            auto taskListToErase = legalizeBarrierProducers(barrierInfo.getBarrierOpAtIndex(barInd), barrierInfo);
            tasksToErase.insert(taskListToErase.begin(), taskListToErase.end());
        }
    }
    // Handle consumers
    for (size_t barInd = (numOfBarriers - 1); barInd > 0; barInd--) {
        if (!allBarrierConsumersLegal(barrierInfo.getBarrierOpAtIndex(barInd), barrierInfo)) {
            auto taskListToErase = legalizeBarrierConsumers(barrierInfo.getBarrierOpAtIndex(barInd), barrierInfo);
            tasksToErase.insert(taskListToErase.begin(), taskListToErase.end());
        }
    }

    // Erase Tasks
    llvm::SmallVector<VPURT::TaskOp> tasksOps;
    for (auto task : tasksToErase) {
        tasksOps.push_back(barrierInfo.getTaskOpAtIndex(task));
    }
    barrierInfo.updateIR();
    barrierInfo.clearAttributes();
    for (auto& task : tasksOps) {
        task->erase();
    }

    // Erase Unused Barriers
    barrierInfo = BarrierInfo{func};
    func.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        if (barrierInfo.getBarrierProducers(barrierOp).empty() || barrierInfo.getBarrierConsumers(barrierOp).empty()) {
            barrierInfo.resetBarrier(barrierOp);
        }
    });

    barrierInfo.updateIR();
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createComputeTaskStrippingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createComputeTaskStrippingPass(Logger log,
                                                                                  VPU::DPUDryRunMode dpuDryRun,
                                                                                  const bool shaveDryRun) {
    return std::make_unique<ComputeTaskStrippingPass>(log, dpuDryRun, shaveDryRun);
}
