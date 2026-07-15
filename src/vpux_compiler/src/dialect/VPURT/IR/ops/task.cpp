//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/types.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/shave.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

mlir::Operation* vpux::VPURT::TaskOp::getInnerTaskOp() {
    return &getBody().front().front();
}

void vpux::VPURT::TaskOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers) {
    build(odsBuilder, odsState, /*profiling_data*/ nullptr, waitBarriers, updateBarriers,
          /*enqueueBarrier*/ nullptr, /*isTrailingSWLayer*/ false, /*wlmPage*/ nullptr,
          /*taskIndex*/ std::nullopt, /*start_after*/ nullptr, /*clean_after*/ nullptr);
}

void vpux::VPURT::TaskOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                                mlir::Value enqueueBarrier) {
    build(odsBuilder, odsState, /*profiling_data*/ nullptr, waitBarriers, updateBarriers, enqueueBarrier,
          /*isTrailingSWLayer*/ false, /*wlmPage*/ nullptr,
          /*taskIndex*/ std::nullopt, /*start_after*/ nullptr, /*clean_after*/ nullptr);
}

config::ExecutorKind vpux::VPURT::TaskOp::getExecutorKind() {
    auto innerTaskOp = getInnerTaskOp();
    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(innerTaskOp);
    VPUX_THROW_UNLESS(task != nullptr, "Inner task at {0} does not implement TaskOpInterface", innerTaskOp->getLoc());

    return task.getExecutorKind();
}

mlir::LogicalResult vpux::VPURT::TaskOp::verify() {
    const auto task = getOperation();
    if (getBody().getBlocks().size() != 1) {
        return errorAt(task, "The task body should contain exactly one block");
    }

    auto numOps = getBody().front().getOperations().size();
    if (numOps != 1) {
        return errorAt(task, "The task body should contain exactly one operation. Got: {0}", numOps);
    }

    auto& innerOp = getBody().front().front();
    if (!mlir::isa<mlir::MemoryEffectOpInterface>(innerOp)) {
        return errorAt(task, "The task body should contain operation with memory effects");
    }

    return mlir::success();
}

void vpux::VPURT::TaskOp::getEffects(SmallVectorImpl<MemoryEffect>& effects) {
    for (const auto waitBarrier : getWaitBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), waitBarrier, VPURT::BarrierResource::get());
    }

    for (const auto updateBarrier : getUpdateBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), updateBarrier, VPURT::BarrierResource::get());
    }

    auto bodyEffects = mlir::cast<mlir::MemoryEffectOpInterface>(getInnerTaskOp());
    bodyEffects.getEffects(effects);
}

VPURT::TaskQueueType vpux::VPURT::getTaskQueueType(TaskOp taskOp, bool ignoreIndexForNce) {
    TaskQueueType queueType(config::ExecutorKind::UNKNOWN, 0);
    // Keep queue type resolution local to this helper. The wrapped inner op is inspected
    // directly below to derive both executor kind and queue id for the handled task kinds.
    // TaskOp::getExecutorKind() is only used in the fallback path.
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();
    if (auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp)) {
        queueType.type = config::ExecutorKind::DPU;
        if (!ignoreIndexForNce) {
            auto& variantsRegion = nceTask.getVariants();
            VPUX_THROW_WHEN(variantsRegion.empty(), "Could not get DPU task");

            VPUIP::DPUTaskOp dpuTask = nullptr;

            // Fast path: expect the first operation in the first block to be the DPU task.
            auto& variantsRegionFront = variantsRegion.front();
            if (!variantsRegionFront.empty()) {
                dpuTask = mlir::dyn_cast<VPUIP::DPUTaskOp>(variantsRegionFront.front());
            }

            // Fallback: search for the first DPUTaskOp in the entire variants region.
            if (dpuTask == nullptr) {
                auto dpuVariantOps = variantsRegion.getOps<VPUIP::DPUTaskOp>();
                if (dpuVariantOps.begin() != dpuVariantOps.end()) {
                    dpuTask = *dpuVariantOps.begin();
                }
            }
            VPUX_THROW_WHEN(dpuTask == nullptr,
                            "No VPUIP::DPUTaskOp found in NCE variants region while resolving VPURT task queue type");
            queueType.id = dpuTask.getClusterId().value_or(0);
        }
    } else if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(wrappedTaskOp)) {
        queueType.type = config::ExecutorKind::SHAVE_ACT;
        if (!ignoreIndexForNce) {
            auto numTiles = config::getNumOfTiles(swKernelOp);
            auto tileIndex = swKernelOp.getTileIndex().value_or(0);
            auto listIndex = swKernelOp.getListIndex().value_or(0);
            queueType.id = getShaveQueueIdEncoding(numTiles, tileIndex, listIndex);
        }
    } else if (auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(wrappedTaskOp)) {
        queueType.type = config::ExecutorKind::DMA_NN;
        queueType.id = getDMAQueueIdEncoding(VPUIP::getDMAPortValue(wrappedTaskOp), dmaTask.getChannelType());
    } else {
        // Fallback for task kinds that are not handled above and do not carry extra queue-id info.
        // This is not a hot path.
        queueType.type = taskOp.getExecutorKind();
        queueType.id = 0;
    }
    return queueType;
}

std::map<VPURT::TaskQueueType, std::pair<VPURT::TaskOp, VPURT::TaskOp>> vpux::VPURT::getTaskQueuesFirstAndLastOp(
        mlir::func::FuncOp funcOp) {
    std::map<VPURT::TaskQueueType, std::pair<VPURT::TaskOp, VPURT::TaskOp>> taskQueuesFirstAndLastOp;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);

        if (taskQueuesFirstAndLastOp.find(taskQueueType) == taskQueuesFirstAndLastOp.end()) {
            // First occurrence of task on this queue
            taskQueuesFirstAndLastOp[taskQueueType] = std::make_pair(taskOp, taskOp);
        } else {
            // In case new task spotted, update last task info
            taskQueuesFirstAndLastOp[taskQueueType].second = taskOp;
        }
    });

    return taskQueuesFirstAndLastOp;
}

std::pair<size_t, size_t> vpux::VPURT::getTileAndListIndex(VPURT::TaskQueueType queueType, int64_t numTiles,
                                                           config::ArchKind arch) {
    if (queueType.type == config::ExecutorKind::SHAVE_ACT) {
        auto tileIndex = getShaveTileIndexFromEncodedId(queueType.id, numTiles);
        auto listIndex = getShaveListIndexFromEncodedId(queueType.id, numTiles);
        return std::make_pair(tileIndex, listIndex);
    } else if (queueType.type == config::ExecutorKind::DPU) {
        return std::make_pair(queueType.id, 0);
    } else if (queueType.type == config::ExecutorKind::DMA_NN) {
        auto tileIndex = getDMAPortFromEncodedId(queueType.id);
        auto channelType = getDMAChannelTypeFromEncodedId(queueType.id, arch);
        int64_t listIndex = (channelType == VPUIP::DmaChannelType::CMX) ? 1 : 0;
        return std::make_pair(tileIndex, listIndex);
    }

    VPUX_THROW("Unsupported queue type {0} for getting tile and list index", stringifyEnum(queueType.type));
}

VPURT::TaskQueueType vpux::VPURT::getQueueTypeFromTileAndListIndex(config::ExecutorKind executorKind, size_t tileIndex,
                                                                   size_t listIndex, int64_t numTiles) {
    if (executorKind == config::ExecutorKind::SHAVE_ACT) {
        return {executorKind, getShaveQueueIdEncoding(numTiles, tileIndex, listIndex)};
    } else if (executorKind == config::ExecutorKind::DPU) {
        VPUX_THROW_WHEN(listIndex != 0, "DPU queue does not support list index different than 0, got {0}", listIndex);
        return {executorKind, static_cast<int64_t>(tileIndex)};
    } else if (executorKind == config::ExecutorKind::DMA_NN) {
        VPUX_THROW_WHEN(listIndex > 1, "DMA queue supports only list index 0 and 1, got {0}", listIndex);
        auto channelType = (listIndex == 1) ? VPUIP::DmaChannelType::CMX : VPUIP::DmaChannelType::DDR;
        return {executorKind, getDMAQueueIdEncoding(tileIndex, channelType)};
    }

    VPUX_THROW("Unsupported executor kind {0}", stringifyEnum(executorKind));
}

size_t vpux::VPURT::TaskOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto innerOp = getInnerTaskOp();

    auto cycleCostInterface = mlir::dyn_cast<VPUIP::CycleCostInterface>(innerOp);
    if (cycleCostInterface == nullptr) {
        return VPU::NO_COST;
    }

    return cycleCostInterface.getOperationCycleCost(costModel);
}

// Function to get tile index for DPU/SHV Op
size_t vpux::VPURT::getTileIndexForDpuOrShv(VPURT::TaskOp taskOp, VPURT::TaskQueueType queueType) {
    if (auto dmaOp = taskOp.getInnerTaskOpOfType<VPUIP::NNDMAOp>()) {
        VPUX_THROW("getTileIndexForDpuOrShv called for DMAOp {0}", taskOp);
    }

    if (auto swOp = taskOp.getInnerTaskOpOfType<VPUIP::SwKernelOp>()) {
        return swOp.getTileIndex().value_or(0);
    }

    return queueType.id;
}

// Function to get list index for DPU/SHV Op
size_t vpux::VPURT::getListIndexForDpuOrShv(VPURT::TaskOp taskOp) {
    if (auto dmaOp = taskOp.getInnerTaskOpOfType<VPUIP::NNDMAOp>()) {
        VPUX_THROW("getListIndexForDpuOrShv called for DMAOp {0}", taskOp);
    }

    if (auto swOp = taskOp.getInnerTaskOpOfType<VPUIP::SwKernelOp>()) {
        return swOp.getListIndex().value_or(0);
    }

    // All DPU tasks are expected to be on list 0
    return 0;
}

// Get number of workloads under single processed task
// In case of DPU it is number of DPU variants
// In case of SHV it is number of SHV kernel run ops
// For other return 1
size_t vpux::VPURT::getNumberOfWorkloads(VPURT::TaskOp taskOp) {
    const auto executorKind = taskOp.getExecutorKind();

    switch (executorKind) {
    case config::ExecutorKind::DPU: {
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task for DPU executor");
        return nceOp.getNumVariants();
    }
    case config::ExecutorKind::SHAVE_ACT: {
        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(swKernelOp != nullptr, "Could not cast to SW kernel task for SHAVE_ACT executor");
        auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
        return std::distance(swKernelRun.begin(), swKernelRun.end());
    }
    case config::ExecutorKind::DMA_NN:
        return 1;

    default:
        VPUX_THROW("Unsupported executor: {0}", executorKind);
    }
}
