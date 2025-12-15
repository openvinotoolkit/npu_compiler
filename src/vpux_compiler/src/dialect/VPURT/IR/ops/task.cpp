//
// Copyright (C) 2022-2025 Intel Corporation.
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

VPU::ExecutorKind vpux::VPURT::TaskOp::getExecutorKind() {
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

void VPURT::TaskOp::printProperties(mlir::MLIRContext* /*ctx*/, mlir::OpAsmPrinter& printer,
                                    const Properties& properties, mlir::ArrayRef<llvm::StringRef> /*elidedProps*/) {
    // Property is default-valued, so skip printing it if its value is the default one
    const auto hasIsTrailingSWLayer =
            properties.isTrailingSWLayer != nullptr && properties.isTrailingSWLayer.getValue();
    const auto hasTaskIndex = properties.taskIndex.has_value();
    if (!hasIsTrailingSWLayer && !hasTaskIndex) {
        return;
    }
    bool shouldPrintComma = false;
    printer << "<{";
    if (hasIsTrailingSWLayer) {
        printer << "isTrailingSWLayer = " << properties.isTrailingSWLayer.getValue();
        shouldPrintComma = true;
    }
    if (hasTaskIndex) {
        if (shouldPrintComma) {
            printer << ", ";
        }
        printer << "taskIndex = " << properties.taskIndex;
    }
    printer << "}>";
}

mlir::ParseResult VPURT::TaskOp::parseProperties(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    const auto parseEnd = [&]() {
        if (mlir::failed(parser.parseRBrace())) {
            return mlir::failure();
        }
        if (mlir::failed(parser.parseGreater())) {
            return mlir::failure();
        }
        return mlir::success();
    };

    auto& prop = result.getOrAddProperties<Properties>();
    if (mlir::failed(parser.parseOptionalLess())) {
        return mlir::success();
    }
    if (mlir::failed(parser.parseLBrace())) {
        return mlir::failure();
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("isTrailingSWLayer"))) {
        if (mlir::failed(parser.parseEqual())) {
            return mlir::failure();
        }
        mlir::BoolAttr boolAttr;
        if (mlir::failed(parser.parseAttribute(boolAttr))) {
            return mlir::failure();
        }
        prop.setIsTrailingSWLayer(boolAttr);
        if (mlir::failed(parser.parseOptionalComma())) {
            return parseEnd();
        }
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("taskIndex"))) {
        if (mlir::failed(parser.parseEqual())) {
            return mlir::failure();
        }
        int64_t value = 0;
        if (mlir::failed(parser.parseInteger(value))) {
            return mlir::failure();
        }
        prop.setTaskIndex(value);
    }
    return parseEnd();
}

VPURT::TaskQueueType vpux::VPURT::getTaskQueueType(TaskOp taskOp, bool ignoreIndexForNce) {
    TaskQueueType queueType;
    queueType.type = taskOp.getExecutorKind();
    if (queueType.type == VPU::ExecutorKind::DPU && !ignoreIndexForNce) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp();
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
        VPUX_THROW_WHEN(nceTask == nullptr || nceTask.getVariants().getOps<VPUIP::DPUTaskOp>().empty(),
                        "Could not get DPU task");
        auto dpuTask = *(nceTask.getVariants().getOps<VPUIP::DPUTaskOp>().begin());
        queueType.id = dpuTask.getClusterId().value_or(0);
    } else if (queueType.type == VPU::ExecutorKind::SHAVE_ACT && !ignoreIndexForNce) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp();
        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(wrappedTaskOp);
        VPUX_THROW_WHEN(swKernelOp == nullptr, "Could not get SW kernel task");
        auto numTiles = VPU::getNumTiles(swKernelOp);
        auto tileIndex = swKernelOp.getTileIndex().value_or(0);
        auto listIndex = swKernelOp.getListIndex().value_or(0);
        queueType.id = getShaveQueueIdEncoding(numTiles, tileIndex, listIndex);
    } else if (queueType.type == VPU::ExecutorKind::DMA_NN) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp();

        auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(wrappedTaskOp);
        VPUX_THROW_WHEN(dmaTask == nullptr, "Not a DMA task");
        queueType.id = getDMAQueueIdEncoding(vpux::getDMAPortValue(wrappedTaskOp), dmaTask.getChannelType());
    } else {
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
    if (queueType.type == VPU::ExecutorKind::SHAVE_ACT) {
        auto tileIndex = getShaveTileIndexFromEncodedId(queueType.id, numTiles);
        auto listIndex = getShaveListIndexFromEncodedId(queueType.id, numTiles);
        return std::make_pair(tileIndex, listIndex);
    } else if (queueType.type == VPU::ExecutorKind::DPU) {
        return std::make_pair(queueType.id, 0);
    } else if (queueType.type == VPU::ExecutorKind::DMA_NN) {
        auto tileIndex = getDMAPortFromEncodedId(queueType.id);
        auto channelType = getDMAChannelTypeFromEncodedId(queueType.id, arch);
        int64_t listIndex = (channelType == VPUIP::DmaChannelType::CMX) ? 1 : 0;
        return std::make_pair(tileIndex, listIndex);
    }

    VPUX_THROW("Unsupported queue type {0} for getting tile and list index", stringifyEnum(queueType.type));
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
