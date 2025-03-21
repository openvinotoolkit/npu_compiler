//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

constexpr uint32_t NPU_DEFAULT_INVARIANT_COUNT = 64;
constexpr uint32_t NPU_DEFAULT_VARIANT_COUNT = 128;
constexpr uint32_t NPU_DEFAULT_KERNEL_RANGE_COUNT = 64;
constexpr uint32_t NPU_DEFAULT_KERNEL_INVO_COUNT = 64;
constexpr uint32_t NPU_DEFAULT_MEDIA_COUNT = 4;
constexpr uint32_t NPU_DEFAULT_DMA_TASK_COUNT = 80;

constexpr uint32_t NPU37XX_DMA_TASK_COUNT = 256;
constexpr uint32_t NPU37XX_INVARIANT_COUNT = 32;
constexpr uint32_t NPU37XX_VARIANT_COUNT = 256;
constexpr uint32_t NPU37XX_KERNEL_RANGE_COUNT = 32;
constexpr uint32_t NPU37XX_KERNEL_INVO_COUNT = 64;
struct TaskListKey {
    VPU::ArchKind archKind;
    VPU::TaskType taskType;
    bool operator==(const TaskListKey& other) const {
        return (archKind == other.archKind && taskType == other.taskType);
    }
};
struct TaskListKeyHash {
    std::size_t operator()(const TaskListKey& key) const noexcept {
        auto hashTask = std::hash<VPU::TaskType>{}(key.taskType);
        auto hashArch = std::hash<VPU::ArchKind>{}(key.archKind);
        // make sure the hash function is good enough for minimizing collision occurrence (same output
        // for different key values)
        return hashTask ^ (hashArch << 3);
    }
};

const std::unordered_map<TaskListKey, uint32_t, TaskListKeyHash> taskListsDefaultCapacityMap = {
        {{VPU::ArchKind::NPU37XX, VPU::TaskType::DPUInvariant}, NPU37XX_INVARIANT_COUNT},
        {{VPU::ArchKind::NPU37XX, VPU::TaskType::DPUVariant}, NPU37XX_VARIANT_COUNT},
        {{VPU::ArchKind::NPU37XX, VPU::TaskType::ActKernelInvocation}, NPU37XX_KERNEL_INVO_COUNT},
        {{VPU::ArchKind::NPU37XX, VPU::TaskType::ActKernelRange}, NPU37XX_KERNEL_RANGE_COUNT},
        {{VPU::ArchKind::NPU37XX, VPU::TaskType::DMA}, NPU37XX_DMA_TASK_COUNT},
        {{VPU::ArchKind::NPU40XX, VPU::TaskType::DPUInvariant}, NPU_DEFAULT_INVARIANT_COUNT},
        {{VPU::ArchKind::NPU40XX, VPU::TaskType::DPUVariant}, NPU_DEFAULT_VARIANT_COUNT},
        {{VPU::ArchKind::NPU40XX, VPU::TaskType::ActKernelInvocation}, NPU_DEFAULT_KERNEL_INVO_COUNT},
        {{VPU::ArchKind::NPU40XX, VPU::TaskType::ActKernelRange}, NPU_DEFAULT_KERNEL_RANGE_COUNT},
        {{VPU::ArchKind::NPU40XX, VPU::TaskType::M2I}, NPU_DEFAULT_MEDIA_COUNT},
        {{VPU::ArchKind::NPU40XX, VPU::TaskType::DMA}, NPU_DEFAULT_DMA_TASK_COUNT},
};

size_t VPU::getConstraint(mlir::Operation* op, StringRef attrName) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch constraint");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(attrName);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find IE.OptionOp attribute", attrName);
    return static_cast<size_t>(attrValue.getOptionValue());
}

uint32_t VPU::getDefaultTaskListCount(VPU::TaskType taskType, VPU::ArchKind archKind) {
    auto taskListCapacityIter = taskListsDefaultCapacityMap.find({archKind, taskType});
    VPUX_THROW_WHEN(taskListCapacityIter == taskListsDefaultCapacityMap.end(),
                    "getDefaultTaskListCount: Unknown task type {0} for arch {1}", taskType, archKind);

    return taskListCapacityIter->second;
}
