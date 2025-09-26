//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"
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
    config::ArchKind archKind;
    VPU::TaskType taskType;
    bool operator==(const TaskListKey& other) const {
        return (archKind == other.archKind && taskType == other.taskType);
    }
};
struct TaskListKeyHash {
    std::size_t operator()(const TaskListKey& key) const noexcept {
        auto hashTask = std::hash<VPU::TaskType>{}(key.taskType);
        auto hashArch = std::hash<config::ArchKind>{}(key.archKind);
        // make sure the hash function is good enough for minimizing collision occurrence (same output
        // for different key values)
        return hashTask ^ (hashArch << 3);
    }
};

const std::unordered_map<TaskListKey, uint32_t, TaskListKeyHash> taskListsDefaultCapacityMap = {
        {{config::ArchKind::NPU37XX, VPU::TaskType::DPUInvariant}, NPU37XX_INVARIANT_COUNT},
        {{config::ArchKind::NPU37XX, VPU::TaskType::DPUVariant}, NPU37XX_VARIANT_COUNT},
        {{config::ArchKind::NPU37XX, VPU::TaskType::ActKernelInvocation}, NPU37XX_KERNEL_INVO_COUNT},
        {{config::ArchKind::NPU37XX, VPU::TaskType::ActKernelRange}, NPU37XX_KERNEL_RANGE_COUNT},
        {{config::ArchKind::NPU37XX, VPU::TaskType::DMA}, NPU37XX_DMA_TASK_COUNT},
        {{config::ArchKind::NPU40XX, VPU::TaskType::DPUInvariant}, NPU_DEFAULT_INVARIANT_COUNT},
        {{config::ArchKind::NPU40XX, VPU::TaskType::DPUVariant}, NPU_DEFAULT_VARIANT_COUNT},
        {{config::ArchKind::NPU40XX, VPU::TaskType::ActKernelInvocation}, NPU_DEFAULT_KERNEL_INVO_COUNT},
        {{config::ArchKind::NPU40XX, VPU::TaskType::ActKernelRange}, NPU_DEFAULT_KERNEL_RANGE_COUNT},
        {{config::ArchKind::NPU40XX, VPU::TaskType::M2I}, NPU_DEFAULT_MEDIA_COUNT},
        {{config::ArchKind::NPU40XX, VPU::TaskType::DMA}, NPU_DEFAULT_DMA_TASK_COUNT},
};

uint32_t VPU::getDefaultTaskListCount(VPU::TaskType taskType, config::ArchKind archKind) {
    auto taskListCapacityIter = taskListsDefaultCapacityMap.find({archKind, taskType});
    VPUX_THROW_WHEN(taskListCapacityIter == taskListsDefaultCapacityMap.end(),
                    "getDefaultTaskListCount: Unknown task type {0} for arch {1}", taskType, archKind);

    return taskListCapacityIter->second;
}
